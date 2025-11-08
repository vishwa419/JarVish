# backend/langgraph_agents/subagents.py
"""
Subagent factory for creating specialized LangGraph agents with error handling.
"""
from typing import List, Dict, Any
from datetime import datetime
from langgraph.graph import StateGraph, MessagesState, START, END 
from langgraph.prebuilt import ToolNode 
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool 
from langchain_openai import ChatOpenAI
from backend.models import MCPServerConfig
from backend.multi_mcp_client import get_langchain_tools
from backend.langgraph_agents.tools import (
    timeout,
    format_intermediate_steps,
    calculate_agent_statistics,
    create_error_result,
    create_timeout_result,
    sanitize_tool_output
)


class SubagentState(MessagesState):
    """Extended state to track intermediate steps and execution metadata."""
    intermediate_steps: list[Dict[str, Any]] = []
    iteration_count: int = 0
    failed_tools: list[str] = []
    error_message: str = ""


def create_subagent(
    name: str,
    system_prompt: str,
    tools: List[BaseTool],
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_iterations: int = 10,
    tool_timeout: int = 30
):
    """
    Generic subagent factory with full error handling and timeouts.

    Args:
        name: Agent name (e.g., "gmail", "calendar", "slack")
        system_prompt: Instructions for this agent's specialty
        tools: List of LangChain BaseTool objects
        model_name: OpenAI model to use
        temperature: Model temperature
        max_iterations: Maximum reasoning iterations before forcing stop
        tool_timeout: Timeout in seconds for each tool execution

    Returns:
        Compiled LangGraph agent workflow with step tracking
    """
    if not tools:
        raise ValueError(f"Subagent '{name}' requires at least one tool")

    # Create the LLM with tools bound
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        verbose=True,
        streaming=True,
        timeout=30
    )
    
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: SubagentState):
        """Main agent reasoning node with step tracking and error handling."""
        start_time = datetime.now()
        iteration_count = state.get("iteration_count", 0) + 1
        
        print(f"  🤖 {name.upper()} agent executing with {len(tools)} tools...")
        print(f"  🔄 Reasoning iteration #{iteration_count}/{max_iterations}")
        
        # Check max iterations
        if iteration_count > max_iterations:
            print(f"  ⚠️  MAX ITERATIONS ({max_iterations}) REACHED - Forcing completion")
            error_msg = f"Agent reached maximum {max_iterations} iterations without completion."
            return {
                "messages": [AIMessage(content=f"I've reached my iteration limit. Here's my progress so far based on the available information.")],
                "intermediate_steps": state.get("intermediate_steps", []),
                "iteration_count": iteration_count,
                "error_message": error_msg
            }
        
        messages = state["messages"]
        intermediate_steps = state.get("intermediate_steps", [])
        failed_tools = state.get("failed_tools", [])
        
        # Add system prompt at the beginning if not already there
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        
        # Add context about failed tools to help agent adapt
        if failed_tools:
            error_context = f"\n\nNOTE: These tools failed previously: {', '.join(failed_tools)}. Try a different approach or provide a response based on available information."
            messages = messages[:-1] + [
                SystemMessage(content=messages[-1].content + error_context)
            ] if messages else messages
        
        try:
            # Call the LLM with tools (with timeout)
            @timeout(seconds=30)
            def call_llm():
                return llm_with_tools.invoke(messages)
            
            response = call_llm()
            
        except TimeoutError as e:
            print(f"  ❌ LLM call timed out: {e}")
            return {
                "messages": [AIMessage(content="I encountered a timeout while processing. Please try again.")],
                "intermediate_steps": intermediate_steps,
                "iteration_count": iteration_count,
                "error_message": "LLM reasoning timed out after 30s"
            }
        except Exception as e:
            print(f"  ❌ LLM call failed: {e}")
            return {
                "messages": [AIMessage(content=create_error_result(name, e))],
                "intermediate_steps": intermediate_steps,
                "iteration_count": iteration_count,
                "error_message": f"LLM error: {str(e)}"
            }
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Track this reasoning step
        step_data = {
            "type": "reasoning",
            "iteration": iteration_count,
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "model_response": {
                "content": response.content,
                "has_tool_calls": bool(hasattr(response, 'tool_calls') and response.tool_calls)
            }
        }
        
        # If there are tool calls, capture them
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls_info = []
            for tc in response.tool_calls:
                tool_info = {
                    "tool_name": tc.get('name', 'unknown'),
                    "tool_args": tc.get('args', {}),
                    "tool_id": tc.get('id', 'unknown')
                }
                tool_calls_info.append(tool_info)
                print(f"  🔧 Tool call: {tool_info['tool_name']}")
                print(f"     📥 Args: {tool_info['tool_args']}")
            
            step_data["tool_calls"] = tool_calls_info
        else:
            print(f"  💬 {name} final response: {response.content[:150]}...")
            step_data["is_final"] = True
        
        intermediate_steps.append(step_data)
        
        return {
            "messages": [response],
            "intermediate_steps": intermediate_steps,
            "iteration_count": iteration_count
        }

    def tracked_tool_node(state: SubagentState):
        """Tool execution with result tracking, timeout, and error handling."""
        print(f"  ⚙️  Executing tools (timeout: {tool_timeout}s per tool)...")
        
        start_time = datetime.now()
        intermediate_steps = state.get("intermediate_steps", [])
        failed_tools = state.get("failed_tools", []).copy()
        
        # Get the last message to find tool calls
        messages = state["messages"]
        last_message = messages[-1] if messages else None
        
        if not last_message or not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            print("  ⚠️  No tool calls found in last message")
            return {
                "messages": [ToolMessage(content="No tool calls to execute", tool_call_id="none")],
                "intermediate_steps": intermediate_steps
            }
        
        # Execute each tool call individually with timeout and error handling
        tool_messages = []
        tool_results = []
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get('name', 'unknown')
            tool_id = tool_call.get('id', 'unknown')
            tool_args = tool_call.get('args', {})
            
            print(f"  🔨 Executing: {tool_name}")
            
            try:
                # Find the tool
                tool_obj = next((t for t in tools if t.name == tool_name), None)
                if not tool_obj:
                    raise ValueError(f"Tool '{tool_name}' not found")
                
                # Execute with timeout
                @timeout(seconds=tool_timeout)
                def execute_tool():
                    return tool_obj.invoke(tool_args)
                
                tool_output = execute_tool()
                
                # Sanitize output
                sanitized_output = sanitize_tool_output(tool_output)
                
                # Create success message
                tool_msg = ToolMessage(
                    content=sanitized_output,
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(tool_msg)
                
                tool_result = {
                    "tool_name": tool_name,
                    "tool_output": sanitized_output,
                    "tool_id": tool_id,
                    "status": "success"
                }
                tool_results.append(tool_result)
                
                print(f"  ✅ {tool_name} succeeded: {sanitized_output[:150]}...")
                
            except TimeoutError:
                print(f"  ⏱️  {tool_name} TIMED OUT after {tool_timeout}s")
                error_content = f"Tool '{tool_name}' timed out after {tool_timeout}s. The operation may be too slow or the service is unresponsive."
                
                tool_msg = ToolMessage(
                    content=error_content,
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(tool_msg)
                
                tool_result = {
                    "tool_name": tool_name,
                    "tool_output": error_content,
                    "tool_id": tool_id,
                    "status": "timeout"
                }
                tool_results.append(tool_result)
                failed_tools.append(tool_name)
                
            except Exception as e:
                print(f"  ❌ {tool_name} FAILED: {str(e)}")
                error_content = f"Tool '{tool_name}' failed: {str(e)}"
                
                tool_msg = ToolMessage(
                    content=error_content,
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(tool_msg)
                
                tool_result = {
                    "tool_name": tool_name,
                    "tool_output": error_content,
                    "tool_id": tool_id,
                    "status": "error",
                    "error": str(e)
                }
                tool_results.append(tool_result)
                failed_tools.append(tool_name)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Track this tool execution step
        step_data = {
            "type": "tool_execution",
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "tools_executed": len(tool_results),
            "results": tool_results,
            "successful": len([r for r in tool_results if r["status"] == "success"]),
            "failed": len([r for r in tool_results if r["status"] != "success"])
        }
        
        intermediate_steps.append(step_data)
        
        print(f"  📊 Tool execution complete: {step_data['successful']}/{step_data['tools_executed']} succeeded")
        
        return {
            "messages": tool_messages,
            "intermediate_steps": intermediate_steps,
            "failed_tools": failed_tools
        }

    def should_continue(state: SubagentState) -> str:
        """Route to tools or end based on last message."""
        messages = state["messages"]
        last_message = messages[-1] if messages else None
        iteration_count = state.get("iteration_count", 0)
        
        # Force end if max iterations reached
        if iteration_count >= max_iterations:
            print(f"  🛑 Max iterations reached, forcing END")
            return "end"
        
        # If there's an error message, end
        if state.get("error_message"):
            print(f"  ⚠️  Error detected, ending: {state['error_message']}")
            return "end"
        
        # If there are tool calls, continue to tools
        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print(f"  ➡️  Routing to tools...")
            return "tools"
        
        # Otherwise we're done
        stats = calculate_agent_statistics(state.get("intermediate_steps", []))
        
        print(f"  ✅ {name} agent complete")
        print(f"  📊 Stats: {stats['total_steps']} total steps, {stats['reasoning_steps']} reasoning, {stats['tool_executions']} tool executions")
        
        return "end"

    # Build LangGraph workflow
    workflow = StateGraph(SubagentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tracked_tool_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")
    
    print(f"  ✓ Compiled {name} subagent workflow (max_iter={max_iterations}, timeout={tool_timeout}s)")
    return workflow.compile()


def create_subagents_from_servers(
    server_configs: List[MCPServerConfig],
    max_iterations: int = 10,
    tool_timeout: int = 30
):
    """
    Create LangGraph subagents for each MCP server configuration.
    
    Args:
        server_configs: List of MCP server configurations
        max_iterations: Max iterations per subagent
        tool_timeout: Timeout for tool execution
        
    Returns:
        Dictionary mapping server names to compiled subagent graphs
    """
    subagents = {}
    for server in server_configs:
        tools = get_langchain_tools([server])
        if not tools:
            print(f"⚠️  No tools for {server.name}, skipping...")
            continue
            
        compiled_graph = create_subagent(
            name=server.name,
            system_prompt=f"You are the {server.name} agent. Use your tools to handle {server.description} requests.",
            tools=tools,
            max_iterations=max_iterations,
            tool_timeout=tool_timeout
        )
        subagents[server.name] = compiled_graph
    
    return subagents
