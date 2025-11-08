# backend/langgraph_agents/orchestrator.py
"""
Main orchestrator agent that routes to specialized subagents with sequential execution.
"""
from typing import Dict, Any, List
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from datetime import datetime
from backend.models import MCPServerConfig
from backend.langgraph_agents.subagents import create_subagent
from backend.multi_mcp_client import get_langchain_tools
from backend.utils import fetch_mcp_prompt
from backend.langgraph_agents.tools import (
    timeout_wrapper,
    format_intermediate_steps,
    format_execution_summary,
    build_agent_context,
    build_synthesis_context,
    log_agent_start,
    log_agent_completion,
    log_orchestrator_decision,
    log_context_passing,
    create_error_result,
    create_timeout_result,
    validate_agent_result
)


class OrchestratorState(MessagesState):
    """State for the orchestrator agent with full tracking."""
    next_agents: list[str] = []
    agent_results: dict[str, str] = {}
    agent_intermediate_steps: dict[str, list[Dict[str, Any]]] = {}
    execution_timeline: list[Dict[str, Any]] = []
    final_answer: str = ""
    failed_agents: list[str] = []
    error_messages: dict[str, str] = {}
    agents_completed: list[str] = []


def create_orchestrator(
    mcp_servers: list[MCPServerConfig],
    max_subagent_iterations: int = 10,
    tool_timeout: int = 30,
    subagent_timeout: int = 120
):
    """
    Creates the main orchestrator agent with sequential subagent execution.
    
    Architecture:
    User -> Orchestrator -> [Sequential Subagents with Context] -> Synthesizer -> User
    
    Each subagent receives results from previous agents for sequential task execution.
    
    Args:
        mcp_servers: List of MCP server configurations
        max_subagent_iterations: Max iterations per subagent before forcing stop
        tool_timeout: Timeout in seconds for individual tool execution
        subagent_timeout: Timeout in seconds for entire subagent execution
        
    Returns:
        Compiled orchestrator graph
    """
    
    # Get all tools
    all_tools = get_langchain_tools(mcp_servers)
    
    # Dynamically create subagents from MCP servers
    subagents = {}
    agent_descriptions = []
    
    for server in mcp_servers:
        # Get tools for this server
        server_tools = [t for t in all_tools if t.metadata.get("server") == server.name]
        
        if not server_tools:
            print(f"⚠️  Warning: No tools found for {server.name}, skipping...")
            continue
        
        # Get MCP prompt for context
        mcp_prompt = fetch_mcp_prompt(server_tools[0].name, [server])
        
        # Build system prompt for this subagent
        system_prompt = _build_subagent_system_prompt(
            server, mcp_prompt, server_tools, max_subagent_iterations
        )
        
        # Create the subagent with timeout settings
        subagents[server.name] = create_subagent(
            name=server.name,
            system_prompt=system_prompt,
            tools=server_tools,
            max_iterations=max_subagent_iterations,
            tool_timeout=tool_timeout
        )
        
        agent_descriptions.append(f"- {server.name}: {server.description}")
        print(f"✓ Created subagent: {server.name} ({len(server_tools)} tools)")
    
    if not subagents:
        raise ValueError("No subagents could be created. Check your MCP server configuration.")
    
    # Orchestrator model
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, verbose=True, streaming=True, timeout=30)
    
    # Define node functions
    def orchestrator_node(state: OrchestratorState):
        """Main orchestrator that decides which subagent(s) to call next."""
        start_time = datetime.now()
        print("\n" + "="*80)
        print("🎯 ORCHESTRATOR: Analyzing request...")
        print("="*80)
        
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        execution_timeline = state.get("execution_timeline", [])
        failed_agents = state.get("failed_agents", [])
        agents_completed = state.get("agents_completed", [])
        agent_results = state.get("agent_results", {})
        
        # Build orchestrator prompt
        system_prompt = _build_orchestrator_prompt(
            agent_descriptions, agents_completed, failed_agents, 
            agent_results, last_message
        )
        
        try:
            response = model.invoke([SystemMessage(content=system_prompt)])
            decision = response.content.strip().lower()
            agents_to_call = [a.strip() for a in decision.split(",")]
            
            # Filter out already completed agents
            agents_to_call = [a for a in agents_to_call if a not in agents_completed and a != "synthesize"]
            
            # If no new agents to call, go to synthesize
            if not agents_to_call:
                agents_to_call = ["synthesize"]
                
        except Exception as e:
            print(f"❌ Orchestrator LLM call failed: {e}")
            agents_to_call = ["synthesize"]
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Track orchestrator decision
        execution_timeline.append({
            "stage": "orchestrator",
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "decision": agents_to_call,
            "reasoning": response.content if 'response' in locals() else "fallback",
            "agents_completed": agents_completed.copy()
        })
        
        log_orchestrator_decision(agents_to_call, duration)
        
        return {
            "next_agents": agents_to_call,
            "execution_timeline": execution_timeline
        }
    
    def route_decision(state: OrchestratorState) -> str:
        """Router function based on orchestrator's decision."""
        next_agents = state.get("next_agents", [])
        
        if not next_agents:
            return "synthesize"
        
        first_agent = next_agents[0]
        
        if first_agent == "synthesize":
            return "synthesize"
        
        if first_agent in subagents:
            print(f"\n🔀 Routing to: {first_agent.upper()}")
            return first_agent
        
        print(f"⚠️  Unknown agent '{first_agent}', going to synthesize")
        return "synthesize"
    
    def create_subagent_node(agent_name: str):
        """Factory to create a node function for a specific subagent."""
        def subagent_node(state: OrchestratorState):
            start_time = datetime.now()
            log_agent_start(agent_name, subagent_timeout, len([t for t in all_tools if t.metadata.get("server") == agent_name]))
            
            messages = state["messages"]
            execution_timeline = state.get("execution_timeline", [])
            failed_agents = state.get("failed_agents", []).copy()
            error_messages = state.get("error_messages", {}).copy()
            agents_completed = state.get("agents_completed", []).copy()
            agent_results_dict = state.get("agent_results", {}).copy()
            
            # Build context from previous agents
            messages_with_context = _build_messages_with_context(
                messages, agent_results_dict
            )
            
            try:
                # Invoke subagent with timeout
                @timeout_wrapper(seconds=subagent_timeout)
                def call_subagent():
                    return subagents[agent_name].invoke({"messages": messages_with_context})
                
                result = call_subagent()
                
                # Validate and extract result
                is_valid, error = validate_agent_result(result)
                if not is_valid:
                    raise ValueError(f"Invalid agent result: {error}")
                
                # Extract final message
                last_msg = result["messages"][-1]
                agent_result = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                
                # Get intermediate steps
                intermediate_steps = result.get("intermediate_steps", [])
                
                # Check for errors
                if result.get("error_message"):
                    print(f"⚠️  {agent_name} reported error: {result['error_message']}")
                    error_messages[agent_name] = result["error_message"]
                    failed_agents.append(agent_name)
                else:
                    agents_completed.append(agent_name)
                
                success = agent_name not in failed_agents
                
            except TimeoutError as e:
                print(f"⏱️  {agent_name.upper()} TIMED OUT after {subagent_timeout}s")
                agent_result = create_timeout_result(agent_name, subagent_timeout)
                intermediate_steps = []
                error_messages[agent_name] = str(e)
                failed_agents.append(agent_name)
                success = False
                
            except Exception as e:
                print(f"❌ {agent_name.upper()} FAILED: {e}")
                agent_result = create_error_result(agent_name, e)
                intermediate_steps = []
                error_messages[agent_name] = str(e)
                failed_agents.append(agent_name)
                success = False
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Store results
            agent_results_dict[agent_name] = agent_result
            
            agent_intermediate_steps = state.get("agent_intermediate_steps", {}).copy()
            agent_intermediate_steps[agent_name] = intermediate_steps
            
            # Track in timeline
            execution_timeline.append({
                "stage": f"agent_{agent_name}",
                "timestamp": start_time.isoformat(),
                "duration_seconds": duration,
                "total_steps": len(intermediate_steps),
                "tool_executions": len([s for s in intermediate_steps if s["type"] == "tool_execution"]),
                "reasoning_iterations": len([s for s in intermediate_steps if s["type"] == "reasoning"]),
                "final_result_preview": agent_result[:100],
                "success": success
            })
            
            # Remove current agent from queue
            next_agents = state.get("next_agents", [])[1:]
            
            # Display results
            if intermediate_steps:
                print(f"\n📋 {agent_name.upper()} INTERMEDIATE STEPS:")
                print(format_intermediate_steps(intermediate_steps))
            
            log_agent_completion(agent_name, duration, agent_result, success)
            
            return {
                "agent_results": agent_results_dict,
                "agent_intermediate_steps": agent_intermediate_steps,
                "next_agents": next_agents,
                "execution_timeline": execution_timeline,
                "failed_agents": failed_agents,
                "error_messages": error_messages,
                "agents_completed": agents_completed
            }
        
        return subagent_node
    
    def after_subagent_router(state: OrchestratorState) -> str:
        """Decide next action after subagent completes."""
        next_agents = state.get("next_agents", [])
        
        if not next_agents:
            print("\n🔄 No more agents in queue, returning to orchestrator...")
            return "orchestrator"
        
        next_agent = next_agents[0]
        if next_agent == "synthesize":
            return "synthesize"
        
        if next_agent in subagents:
            return next_agent
        
        return "orchestrator"
    
    def synthesizer_node(state: OrchestratorState):
        """Synthesize results from all subagents."""
        start_time = datetime.now()
        print("\n" + "="*80)
        print("🎨 SYNTHESIZER: Combining results...")
        print("="*80)
        
        agent_results = state.get("agent_results", {})
        agent_intermediate_steps = state.get("agent_intermediate_steps", {})
        execution_timeline = state.get("execution_timeline", [])
        failed_agents = state.get("failed_agents", [])
        agents_completed = state.get("agents_completed", [])
        error_messages = state.get("error_messages", {})
        original_request = state["messages"][0].content if state["messages"] else ""
        
        # Build synthesis context
        results_text = build_synthesis_context(
            agent_results, agent_intermediate_steps, failed_agents,
            agents_completed, error_messages
        )
        
        synthesis_prompt = f"""You are a synthesis agent. Combine the following results into a clear, concise final answer.

Original user request: {original_request}

Agent Results (in execution order):
{results_text}

{"Some agents failed or timed out. " if failed_agents else ""}Provide a clear, natural summary of what was accomplished across ALL agents. Show the sequential flow of work. Be specific and helpful. If some tasks failed, explain what was attempted and what succeeded."""
        
        try:
            response = model.invoke([SystemMessage(content=synthesis_prompt)])
            final_answer = response.content
        except Exception as e:
            print(f"❌ Synthesizer LLM call failed: {e}")
            final_answer = f"I encountered an error while synthesizing results. Here's what happened:\n\n{results_text}"
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Track synthesis
        execution_timeline.append({
            "stage": "synthesizer",
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "agents_synthesized": list(agent_results.keys()),
            "final_answer_length": len(final_answer),
            "failed_agents": failed_agents,
            "completed_agents": agents_completed
        })
        
        print(f"\n✨ Final answer: {final_answer[:300]}...")
        print(format_execution_summary(execution_timeline))
        
        return {
            "messages": [AIMessage(content=final_answer)],
            "final_answer": final_answer,
            "execution_timeline": execution_timeline
        }
    
    # Build the graph
    workflow = StateGraph(OrchestratorState)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("synthesizer", synthesizer_node)
    
    for agent_name in subagents.keys():
        workflow.add_node(agent_name, create_subagent_node(agent_name))
    
    # Add edges
    workflow.add_edge(START, "orchestrator")
    
    routing_map = {agent: agent for agent in subagents.keys()}
    routing_map["synthesize"] = "synthesizer"
    routing_map["orchestrator"] = "orchestrator"
    
    workflow.add_conditional_edges("orchestrator", route_decision, routing_map)
    
    for agent_name in subagents.keys():
        workflow.add_conditional_edges(agent_name, after_subagent_router, routing_map)
    
    workflow.add_edge("synthesizer", END)
    
    print("\n" + "="*80)
    print("🎉 Orchestrator graph compiled with SEQUENTIAL EXECUTION!")
    print(f"   Max iterations per agent: {max_subagent_iterations}")
    print(f"   Tool timeout: {tool_timeout}s")
    print(f"   Subagent timeout: {subagent_timeout}s")
    print(f"   Context passing: ENABLED ✓")
    print("="*80 + "\n")
    return workflow.compile()


# Private helper functions

def _build_subagent_system_prompt(
    server: MCPServerConfig,
    mcp_prompt: str,
    tools: List,
    max_iterations: int
) -> str:
    """Build system prompt for a subagent."""
    return f"""You are a {server.name} specialist agent working as part of a multi-agent system.

Domain: {server.description}
Context: {mcp_prompt}

Available tools: {', '.join(t.name for t in tools)}

IMPORTANT TIPS FOR USING TOOLS:
- For qdrant_search: Use natural language queries, always set top_k=5 or higher
- When searching for contact info (emails, phones), search for: "person_name email" or "person_name contact"
- Review ALL results carefully - the information might be in any of the top results
- Extract specific information (like email addresses) from the returned text passages
- If a tool fails or times out, try a simpler query or provide a response based on what you know
- You have a maximum of {max_iterations} iterations - use them wisely

SEQUENTIAL EXECUTION:
- You may receive results from previous agents in the conversation history
- Use information from previous agents to complete your task
- Build upon their work rather than duplicating it
- Be specific about what YOU accomplished vs. what previous agents did

Your job is to execute {server.name}-related tasks using the available tools.
Be thorough and use the tools to complete the user's request. If tools fail, do your best with available information."""


def _build_orchestrator_prompt(
    agent_descriptions: List[str],
    agents_completed: List[str],
    failed_agents: List[str],
    agent_results: Dict[str, str],
    last_message: str
) -> str:
    """Build orchestrator decision prompt."""
    agent_list = "\n".join(agent_descriptions)
    
    context_notes = []
    if agents_completed:
        context_notes.append(f"Completed agents: {', '.join(agents_completed)}")
    if failed_agents:
        context_notes.append(f"Failed agents: {', '.join(failed_agents)}")
    if agent_results:
        results_summary = "\n".join([f"  - {agent}: {result[:100]}..." 
                                     for agent, result in agent_results.items()])
        context_notes.append(f"Previous results:\n{results_summary}")
    
    context_text = "\n".join(context_notes) if context_notes else ""
    
    return f"""You are Jarvis, an orchestrator agent. Analyze the user's request and decide which subagent(s) to call NEXT.

Available subagents:
{agent_list}

{context_text}

Rules:
1. Respond with a comma-separated list of SUBAGENT NAMES to call NEXT (e.g., "gmail_agent,calendar_agent")
2. DO NOT include agents that have already completed successfully
3. Consider the SEQUENCE - which agent should go next based on what's been done?
4. Use "synthesize" when you have all results and need to combine them
5. Think carefully about dependencies between tasks
6. The subagent names are listed above - use EXACTLY those names
7. If agents have failed, try alternative approaches or go directly to synthesize

User request: {last_message}

Which subagents should handle this NEXT? (respond with subagent names only, comma-separated):"""


def _build_messages_with_context(messages, agent_results_dict):
    """Build message list with context from previous agents."""
    if not agent_results_dict:
        print(f"📭 No previous context (first agent)")
        return messages
    
    print(f"📥 Passing context from {len(agent_results_dict)} previous agent(s)")
    
    context_message_content = build_agent_context(agent_results_dict)
    context_preview = context_message_content[:200]
    
    log_context_passing(len(agent_results_dict), context_preview)
    
    return [
        messages[0],
        SystemMessage(content=context_message_content),
        *messages[1:]
    ]
