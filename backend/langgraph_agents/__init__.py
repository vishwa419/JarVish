# backend/langgraph_agents/__init__.py
"""
LangGraph multi-agent orchestration system.

This module provides a hierarchical agent system with:
- Orchestrator: Routes tasks to specialized subagents
- Subagents: Execute domain-specific tasks with tools
- Tools: Utility functions for timeout, formatting, and error handling
"""

from backend.langgraph_agents.orchestrator import (
    create_orchestrator,
    OrchestratorState
)

from backend.langgraph_agents.subagents import (
    create_subagent,
    create_subagents_from_servers,
    SubagentState
)

from backend.langgraph_agents.tools import (
    # Timeout utilities
    timeout,
    timeout_wrapper,
    
    # Formatting utilities
    format_intermediate_steps,
    format_execution_summary,
    
    # Context building
    build_agent_context,
    build_synthesis_context,
    
    # Logging utilities
    log_agent_start,
    log_agent_completion,
    log_orchestrator_decision,
    log_context_passing,
    log_tool_execution,
    
    # Error handling
    create_error_result,
    create_timeout_result,
    
    # Validation
    validate_agent_result,
    sanitize_tool_output,
    
    # Statistics
    calculate_agent_statistics
)

__all__ = [
    # Main factory functions
    "create_orchestrator",
    "create_subagent",
    "create_subagents_from_servers",
    
    # State classes
    "OrchestratorState",
    "SubagentState",
    
    # Timeout utilities
    "timeout",
    "timeout_wrapper",
    
    # Formatting
    "format_intermediate_steps",
    "format_execution_summary",
    
    # Context building
    "build_agent_context",
    "build_synthesis_context",
    
    # Logging
    "log_agent_start",
    "log_agent_completion",
    "log_orchestrator_decision",
    "log_context_passing",
    "log_tool_execution",
    
    # Error handling
    "create_error_result",
    "create_timeout_result",
    
    # Validation
    "validate_agent_result",
    "sanitize_tool_output",
    
    # Statistics
    "calculate_agent_statistics"
]

__version__ = "1.0.0"
