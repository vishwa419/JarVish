"""
LangGraph-based multi-agent chain for Jarvis
"""
import json
import logging
from typing import List, Literal
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, END

from backend.models import JarvisState, SubTask, MCPServerConfig
from backend.utils import resolve_placeholders, fetch_mcp_prompt, get_tool_instance, format_chat_history
from backend.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# NODE DEFINITIONS
# ============================================================================

def planner_node(state: JarvisState) -> JarvisState:
    """
    Lightweight planner that decomposes user request into subtasks.
    
    Responsibilities:
    - Parse user intent
    - Map to available tools
    - Extract minimal parameters (names, dates, keywords)
    - Define dependencies between subtasks
    
    Does NOT generate content (email bodies, search queries, etc.)
    """
    logger.info("📋 PLANNER: Decomposing user request into subtasks...")
    
    # Build list of available tools
    tool_names = ", ".join(state["available_tools"])
    
    # Planner system prompt
    planner_prompt = f"""You are a task planning agent. Your job is to decompose user requests into a sequence of subtasks.

Available tools: {tool_names}

RULES:
1. Each subtask uses ONE tool
2. Extract only MINIMAL parameters (names, dates, keywords) - NO content generation
3. Use placeholders for dependencies: <result_from_step_N.field>
4. Return ONLY a JSON array, no other text

TOOL USAGE GUIDELINES:
- qdrant_search: Use for finding information in documents (emails, contacts, etc.)
- qdrant_store: Use when user uploads/stores new documents
- list_documents: Use to see available documents
- gmail_send: Use for sending emails (do NOT write email body here)
- gmail_search: Use for searching emails
- calendar_create_event: Use for scheduling events (do NOT write description here)
- calendar_list_events: Use for viewing calendar

OUTPUT FORMAT (JSON only):
[
  {{
    "step": 0,
    "tool": "tool_name",
    "params": {{"key": "value"}},
    "depends_on": null
  }},
  {{
    "step": 1,
    "tool": "another_tool",
    "params": {{"recipient": "<result_from_step_0.email>"}},
    "depends_on": [0]
  }}
]

EXAMPLE:
User: "Find John's email and send him a meeting invite"
Output:
[
  {{"step": 0, "tool": "qdrant_search", "params": {{"query": "John"}}, "depends_on": null}},
  {{"step": 1, "tool": "gmail_send", "params": {{"to": "<result_from_step_0.email>", "context": "meeting invite"}}, "depends_on": [0]}}
]

Chat History:
{format_chat_history(state['chat_history'])}

User Request: {state['user_message']}

Return ONLY the JSON array:"""

    # Call LLM
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        openai_api_key=settings.openai_api_key
    )
    
    try:
        response = llm.invoke([
            {"role": "user", "content": planner_prompt}
        ])
        
        # Parse JSON response
        content = response.content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        subtasks = json.loads(content)
        
        # Add status tracking
        for task in subtasks:
            task["status"] = "pending"
            task["result"] = None
            task["error"] = None
        
        logger.info(f"✅ PLANNER: Created {len(subtasks)} subtasks")
        for task in subtasks:
            logger.info(f"   • Step {task['step']}: {task['tool']}")
        
        return {
            **state,
            "subtasks": subtasks,
            "step_results": {}
        }
        
    except Exception as e:
        logger.error(f"❌ PLANNER ERROR: {e}")
        return {
            **state,
            "subtasks": [],
            "error": f"Planning failed: {str(e)}"
        }


def dependency_resolver_node(state: JarvisState) -> JarvisState:
    """
    Resolve dependencies and create sequential execution order.
    For now, we execute strictly sequentially (no parallelization).
    """
    logger.info("🔗 DEPENDENCY RESOLVER: Creating execution order...")
    
    subtasks = state["subtasks"]
    
    # Simple sequential execution based on step order
    execution_order = [task["step"] for task in sorted(subtasks, key=lambda x: x["step"])]
    
    logger.info(f"✅ DEPENDENCY RESOLVER: Execution order: {execution_order}")
    
    return {
        **state,
        "execution_order": execution_order,
        "current_step_index": 0
    }


def executor_node(state: JarvisState) -> JarvisState:
    """
    Execute ONE subtask using a tool-specific ReAct agent.
    This node will be called multiple times (loops on itself).
    """
    current_idx = state["current_step_index"]
    execution_order = state["execution_order"]
    
    if current_idx >= len(execution_order):
        logger.info("✅ EXECUTOR: All subtasks completed")
        return state
    
    step_id = execution_order[current_idx]
    subtasks = state["subtasks"]
    task = subtasks[step_id]
    
    logger.info(f"⚙️ EXECUTOR: Executing Step {step_id} - {task['tool']}")
    
    # Mark as executing
    task["status"] = "executing"
    
    try:
        # Resolve placeholders
        resolved_params = resolve_placeholders(task["params"], state["step_results"])
        logger.info(f"   Parameters: {resolved_params}")
        
        # Execute tool-specific agent
        result = execute_tool_agent(
            tool_name=task["tool"],
            params=resolved_params,
            user_context=state["user_message"],
            chat_history=state["chat_history"],
            mcp_servers=state["mcp_servers"],
            all_tools=state.get("_all_tools", [])  # Internal: passed from main
        )
        
        if result.get("success"):
            task["status"] = "completed"
            task["result"] = result["data"]
            state["step_results"][step_id] = result["data"]
            logger.info(f"✅ EXECUTOR: Step {step_id} completed successfully")
        else:
            task["status"] = "failed"
            task["error"] = result.get("error", "Unknown error")
            logger.error(f"❌ EXECUTOR: Step {step_id} failed - {task['error']}")
    
    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)
        logger.error(f"❌ EXECUTOR: Step {step_id} exception - {e}")
    
    # Move to next step
    return {
        **state,
        "current_step_index": current_idx + 1
    }


def execute_tool_agent(
    tool_name: str,
    params: dict,
    user_context: str,
    chat_history: List[tuple],
    mcp_servers: List[MCPServerConfig],
    all_tools: List[BaseTool]
) -> dict:
    """
    Spawn a tool-specific ReAct agent.
    
    Steps:
    1. Fetch MCP prompt for this tool
    2. Create ReAct agent with tool + MCP prompt
    3. Agent generates content and executes tool
    4. Return observation
    """
    logger.info(f"   🤖 Spawning ReAct agent for {tool_name}...")
    
    # Fetch MCP prompt
    mcp_prompt = fetch_mcp_prompt(tool_name, mcp_servers)
    
    # Get tool instance
    tool_instance = get_tool_instance(tool_name, all_tools)
    if not tool_instance:
        return {"success": False, "error": f"Tool {tool_name} not found"}
    
    # Build tool-specific ReAct prompt
    tool_react_prompt = f"""You are a specialized agent for the {tool_name} tool.

MCP TOOL GUIDANCE:
{mcp_prompt}

USER'S ORIGINAL REQUEST: {user_context}

YOUR SUBTASK PARAMETERS: {json.dumps(params)}

CHAT HISTORY:
{format_chat_history(chat_history)}

YOUR JOB:
1. Use the MCP guidance above to understand how to use this tool effectively
2. Generate any necessary content (email body, search query, etc.) based on the context
3. Execute the {tool_name} tool with complete parameters
4. Return the result

Use this format:
Thought: [reasoning about what to do]
Action: {tool_name}
Action Input: {{"param": "value", ...}}
Observation: [tool output]
... (repeat Thought/Action/Observation as needed)
Thought: I have the final result
Final Answer: [result in JSON format if structured, or plain text]

Begin!

{{agent_scratchpad}}"""

    # Create mini ReAct agent
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.7,
        openai_api_key=settings.openai_api_key
    )
    
    prompt = PromptTemplate.from_template(tool_react_prompt)
    
    agent = create_react_agent(
        llm=llm,
        tools=[tool_instance],
        prompt=prompt
    )
    
    executor = AgentExecutor(
        agent=agent,
        tools=[tool_instance],
        max_iterations=3,
        handle_parsing_errors=True,
        verbose=True
    )
    
    # Execute
    try:
        result = executor.invoke({"input": json.dumps(params), "agent_scratchpad": ""})
        output = result.get("output", "")
        
        # Try to parse as JSON, fallback to string
        try:
            structured_output = json.loads(output)
        except:
            structured_output = {"output": output}
        
        return {
            "success": True,
            "data": structured_output
        }
    
    except Exception as e:
        logger.error(f"   ❌ Tool agent execution failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def aggregator_node(state: JarvisState) -> JarvisState:
    """
    Synthesize all subtask results into a coherent final response.
    """
    logger.info("📊 AGGREGATOR: Synthesizing final response...")
    
    # Collect results
    summaries = []
    completed_count = 0
    failed_count = 0
    
    for task in state["subtasks"]:
        if task["status"] == "completed":
            completed_count += 1
            result_str = json.dumps(task["result"]) if isinstance(task["result"], dict) else str(task["result"])
            summaries.append(f"✓ {task['tool']}: {result_str}")
        elif task["status"] == "failed":
            failed_count += 1
            summaries.append(f"✗ {task['tool']}: Failed - {task['error']}")
    
    # Synthesis prompt
    synthesis_prompt = f"""You are Jarvis, an AI assistant. The user asked: "{state['user_message']}"

You executed the following tasks:
{chr(10).join(summaries)}

Provide a natural, helpful response summarizing what was accomplished. Be concise and friendly.
If any tasks failed, acknowledge them but focus on what succeeded.
"""

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.7,
        openai_api_key=settings.openai_api_key
    )
    
    try:
        response = llm.invoke([
            {"role": "system", "content": "You are Jarvis, a helpful AI assistant."},
            {"role": "user", "content": synthesis_prompt}
        ])
        
        final_response = response.content
        logger.info(f"✅ AGGREGATOR: Response generated ({completed_count} succeeded, {failed_count} failed)")
        
    except Exception as e:
        logger.error(f"❌ AGGREGATOR ERROR: {e}")
        final_response = f"I completed {completed_count} tasks, but encountered {failed_count} failures."
    
    return {
        **state,
        "final_response": final_response
    }


# ============================================================================
# CONDITIONAL EDGE ROUTER
# ============================================================================

def should_continue_execution(state: JarvisState) -> Literal["executor", "aggregator"]:
    """
    Router function: Continue executing subtasks or move to aggregation?
    """
    current_idx = state["current_step_index"]
    total_steps = len(state["execution_order"])
    
    if current_idx < total_steps:
        logger.info(f"🔄 ROUTER: Continue execution ({current_idx + 1}/{total_steps})")
        return "executor"
    else:
        logger.info("🔄 ROUTER: All subtasks done, moving to aggregation")
        return "aggregator"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_jarvis_graph(
    mcp_servers: List[MCPServerConfig],
    tools: List[BaseTool]
) -> StateGraph:
    """
    Build and compile the LangGraph for Jarvis.
    
    Args:
        mcp_servers: List of MCP server configurations
        tools: List of LangChain tools discovered from MCP servers
        
    Returns:
        Compiled StateGraph
    """
    logger.info("🔧 Building Jarvis LangGraph...")
    logger.info(f"tools: {len(tools)}")
    # Initialize graph
    graph = StateGraph(JarvisState)
    
    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("dependency_resolver", dependency_resolver_node)
    graph.add_node("executor", executor_node)
    graph.add_node("aggregator", aggregator_node)
    
    # Define edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "dependency_resolver")
    graph.add_edge("dependency_resolver", "executor")
    
    # Conditional edge: loop executor or move to aggregator
    graph.add_conditional_edges(
        "executor",
        should_continue_execution,
        {
            "executor": "executor",      # Loop back for next subtask
            "aggregator": "aggregator"   # All done, synthesize
        }
    )
    
    graph.add_edge("aggregator", END)
    
    logger.info("✅ LangGraph constructed successfully")
    
    return graph.compile()
