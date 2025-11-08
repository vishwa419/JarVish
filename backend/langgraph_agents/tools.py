# backend/langgraph_agents/tools.py
"""
Utility functions for agent system including timeout handling,
formatting, and helper functions.
"""
from typing import List, Dict, Any, Callable
import concurrent.futures
from functools import wraps
from datetime import datetime


# =====================================================================
# TIMEOUT DECORATORS
# =====================================================================

def timeout_wrapper(seconds: int = 60):
    """
    Timeout wrapper for any function execution.
    
    Args:
        seconds: Timeout duration in seconds
        
    Returns:
        Decorated function that raises TimeoutError if exceeded
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"{func.__name__} exceeded {seconds}s timeout")
        return wrapper
    return decorator


def timeout(seconds: int = 30):
    """
    Simplified timeout decorator for tool execution.
    
    Args:
        seconds: Timeout duration in seconds
        
    Returns:
        Decorated function that raises TimeoutError if exceeded
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"{func.__name__} exceeded {seconds}s timeout")
        return wrapper
    return decorator


# =====================================================================
# FORMATTING UTILITIES
# =====================================================================

def format_intermediate_steps(steps: List[Dict[str, Any]]) -> str:
    """
    Format intermediate steps into readable text for logging/display.
    
    Args:
        steps: List of step dictionaries with type, duration, and results
        
    Returns:
        Formatted string representation of all steps
    """
    if not steps:
        return "No intermediate steps recorded."
    
    output = []
    
    for i, step in enumerate(steps, 1):
        if step["type"] == "reasoning":
            output.append(f"\n{'='*60}")
            output.append(f"Step {i}: REASONING (Iteration #{step['iteration']})")
            output.append(f"Duration: {step['duration_seconds']:.2f}s")
            
            if step["model_response"]["content"]:
                content = step["model_response"]["content"][:200]
                output.append(f"Thought: {content}...")
            
            if "tool_calls" in step:
                output.append(f"🔧 Decided to call {len(step['tool_calls'])} tool(s):")
                for tc in step["tool_calls"]:
                    output.append(f"  - {tc['tool_name']}: {tc['tool_args']}")
        
        elif step["type"] == "tool_execution":
            output.append(f"\n{'='*60}")
            output.append(f"Step {i}: TOOL EXECUTION")
            output.append(f"Duration: {step['duration_seconds']:.2f}s")
            output.append(f"Success: {step.get('successful', 0)}/{step['tools_executed']}")
            
            for result in step["results"]:
                status_emoji = "✅" if result["status"] == "success" else "❌"
                output.append(f"\n  {status_emoji} {result['tool_name']} ({result['status']}):")
                output_preview = str(result['tool_output'])[:300]
                output.append(f"     {output_preview}...")
    
    output.append(f"\n{'='*60}\n")
    return "\n".join(output)


def format_execution_summary(execution_timeline: List[Dict[str, Any]]) -> str:
    """
    Format execution timeline into a summary report.
    
    Args:
        execution_timeline: List of execution stage dictionaries
        
    Returns:
        Formatted summary string
    """
    if not execution_timeline:
        return "No execution data available."
    
    total_duration = sum(step.get("duration_seconds", 0) for step in execution_timeline)
    
    output = ["\n" + "="*80]
    output.append("📊 EXECUTION SUMMARY")
    output.append("="*80)
    output.append(f"Total Duration: {total_duration:.2f}s")
    output.append(f"Total Stages: {len(execution_timeline)}")
    
    for i, stage in enumerate(execution_timeline, 1):
        stage_name = stage.get("stage", "unknown")
        duration = stage.get("duration_seconds", 0)
        
        if stage_name == "orchestrator":
            decision = stage.get("decision", [])
            output.append(f"\n{i}. Orchestrator ({duration:.2f}s)")
            output.append(f"   Decision: {decision}")
            
        elif stage_name.startswith("agent_"):
            agent_name = stage_name.replace("agent_", "")
            success = stage.get("success", False)
            status = "✅" if success else "❌"
            output.append(f"\n{i}. {agent_name.upper()} Agent ({duration:.2f}s) {status}")
            output.append(f"   Steps: {stage.get('total_steps', 0)}")
            output.append(f"   Tool Executions: {stage.get('tool_executions', 0)}")
            
        elif stage_name == "synthesizer":
            output.append(f"\n{i}. Synthesizer ({duration:.2f}s)")
            output.append(f"   Agents Synthesized: {len(stage.get('agents_synthesized', []))}")
    
    output.append("="*80 + "\n")
    return "\n".join(output)


# =====================================================================
# CONTEXT BUILDING UTILITIES
# =====================================================================

def build_agent_context(agent_results: Dict[str, str]) -> str:
    """
    Build a context message from previous agent results.
    
    Args:
        agent_results: Dictionary mapping agent names to their results
        
    Returns:
        Formatted context string to pass to next agent
    """
    if not agent_results:
        return ""
    
    context_parts = ["=== CONTEXT FROM PREVIOUS AGENTS ===\n"]
    
    for agent_name, result in agent_results.items():
        context_parts.append(f"\n[{agent_name.upper()} completed]")
        context_parts.append(f"{result}")
        context_parts.append("")
    
    context_parts.append("=== YOUR TASK ===")
    context_parts.append("Use the information above to complete your part of the task.")
    context_parts.append("Do NOT repeat work already done by previous agents.")
    context_parts.append("Build upon their results to accomplish the next step.")
    
    return "\n".join(context_parts)


def build_synthesis_context(
    agent_results: Dict[str, str],
    agent_intermediate_steps: Dict[str, List[Dict[str, Any]]],
    failed_agents: List[str],
    agents_completed: List[str],
    error_messages: Dict[str, str]
) -> str:
    """
    Build comprehensive context for synthesis from all agent results.
    
    Args:
        agent_results: Results from each agent
        agent_intermediate_steps: Intermediate steps from each agent
        failed_agents: List of agent names that failed
        agents_completed: List of agent names that completed successfully
        error_messages: Error messages from failed agents
        
    Returns:
        Formatted context string for synthesis
    """
    results_with_context = []
    
    for agent, result in agent_results.items():
        steps = agent_intermediate_steps.get(agent, [])
        tool_calls = []
        
        for step in steps:
            if step.get("type") == "tool_execution":
                for tool_result in step.get("results", []):
                    tool_calls.append(f"  - Used {tool_result['tool_name']}")
        
        # Determine status
        if agent in agents_completed:
            status = "✅ COMPLETED"
        elif agent in failed_agents:
            status = "❌ FAILED"
        else:
            status = "⚠️  UNKNOWN"
        
        context = f"{agent.upper()} {status}:\n{result}\n"
        
        if tool_calls:
            context += f"Tools used:\n" + "\n".join(tool_calls) + "\n"
        
        if agent in error_messages:
            context += f"Error: {error_messages[agent]}\n"
        
        results_with_context.append(context)
    
    return "\n\n".join(results_with_context) if results_with_context else "No subagent results available."


# =====================================================================
# LOGGING UTILITIES
# =====================================================================

def log_agent_start(agent_name: str, timeout: int, num_tools: int):
    """Log agent execution start."""
    print("\n" + "="*80)
    print(f"🚀 {agent_name.upper()} AGENT: Processing task...")
    print(f"⏱️  Timeout: {timeout}s")
    print(f"🔧 Tools available: {num_tools}")
    print("="*80)


def log_agent_completion(agent_name: str, duration: float, result_preview: str, success: bool = True):
    """Log agent execution completion."""
    status_emoji = "✅" if success else "❌"
    print(f"{status_emoji} {agent_name} completed in {duration:.2f}s")
    print(f"📝 Final result: {result_preview[:200]}...")


def log_orchestrator_decision(decision: List[str], duration: float):
    """Log orchestrator routing decision."""
    print(f"📊 Decision: {decision}")
    print(f"⏱️  Decision took: {duration:.2f}s")


def log_context_passing(num_previous_agents: int, context_preview: str):
    """Log context being passed between agents."""
    print(f"📥 Passing context from {num_previous_agents} previous agent(s)")
    print(f"📋 Context preview: {context_preview[:200]}...")


def log_tool_execution(tool_name: str, status: str, duration: float, output_preview: str):
    """Log individual tool execution."""
    status_emoji = "✅" if status == "success" else "❌"
    print(f"  {status_emoji} {tool_name} ({status}) - {duration:.2f}s")
    print(f"     Output: {output_preview[:150]}...")


# =====================================================================
# VALIDATION UTILITIES
# =====================================================================

def validate_agent_result(result: Any) -> tuple[bool, str]:
    """
    Validate agent execution result structure.
    
    Args:
        result: Agent execution result
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not result:
        return False, "Empty result"
    
    if not isinstance(result, dict):
        return False, "Result is not a dictionary"
    
    if "messages" not in result:
        return False, "Result missing 'messages' key"
    
    if not result["messages"]:
        return False, "Messages list is empty"
    
    return True, ""


def sanitize_tool_output(output: Any, max_length: int = 5000) -> str:
    """
    Sanitize and truncate tool output for safe logging/storage.
    
    Args:
        output: Raw tool output
        max_length: Maximum length for output string
        
    Returns:
        Sanitized string output
    """
    output_str = str(output)
    
    if len(output_str) > max_length:
        output_str = output_str[:max_length] + f"... (truncated, original length: {len(output_str)})"
    
    return output_str


# =====================================================================
# ERROR HANDLING UTILITIES
# =====================================================================

def create_error_result(agent_name: str, error: Exception) -> str:
    """
    Create a standardized error result message.
    
    Args:
        agent_name: Name of the agent that failed
        error: Exception that occurred
        
    Returns:
        Formatted error message string
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    return f"Agent '{agent_name}' encountered a {error_type}: {error_msg}"


def create_timeout_result(agent_name: str, timeout_seconds: int) -> str:
    """
    Create a standardized timeout result message.
    
    Args:
        agent_name: Name of the agent that timed out
        timeout_seconds: Timeout duration that was exceeded
        
    Returns:
        Formatted timeout message string
    """
    return f"Agent '{agent_name}' timed out after {timeout_seconds}s. The task may be too complex or the service is slow."


# =====================================================================
# STATISTICS UTILITIES
# =====================================================================

def calculate_agent_statistics(intermediate_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics from agent intermediate steps.
    
    Args:
        intermediate_steps: List of intermediate step dictionaries
        
    Returns:
        Dictionary with calculated statistics
    """
    total_steps = len(intermediate_steps)
    tool_executions = len([s for s in intermediate_steps if s.get("type") == "tool_execution"])
    reasoning_steps = len([s for s in intermediate_steps if s.get("type") == "reasoning"])
    
    total_duration = sum(s.get("duration_seconds", 0) for s in intermediate_steps)
    
    successful_tools = 0
    failed_tools = 0
    
    for step in intermediate_steps:
        if step.get("type") == "tool_execution":
            successful_tools += step.get("successful", 0)
            failed_tools += step.get("failed", 0)
    
    return {
        "total_steps": total_steps,
        "tool_executions": tool_executions,
        "reasoning_steps": reasoning_steps,
        "total_duration": total_duration,
        "successful_tools": successful_tools,
        "failed_tools": failed_tools,
        "avg_step_duration": total_duration / total_steps if total_steps > 0 else 0
    }
