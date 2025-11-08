"""
Utility functions for Jarvis multi-agent system
"""
import re
import json
import requests
import logging
from typing import Dict, Any, List, Optional
from langchain.tools import BaseTool

from backend.models import MCPServerConfig

logger = logging.getLogger(__name__)


def resolve_placeholders(params: Dict[str, Any], step_results: Dict[int, Any]) -> Dict[str, Any]:
    """
    Replace placeholders like <result_from_step_0.email> with actual values.
    
    Args:
        params: Dictionary containing placeholder strings
        step_results: Mapping of step_id → result data
        
    Returns:
        Dictionary with placeholders resolved
        
    Example:
        params = {"to": "<result_from_step_0.email>"}
        step_results = {0: {"email": "sathvika@example.com"}}
        → {"to": "sathvika@example.com"}
    """
    resolved = {}
    
    for key, value in params.items():
        if isinstance(value, str) and value.startswith("<result_from_step_"):
            # Pattern: <result_from_step_N.field>
            match = re.match(r"<result_from_step_(\d+)\.(\w+)>", value)
            if match:
                step_id = int(match.group(1))
                field = match.group(2)
                
                if step_id in step_results:
                    result_data = step_results[step_id]
                    
                    # Handle different result structures
                    if isinstance(result_data, dict) and field in result_data:
                        resolved[key] = result_data[field]
                    elif isinstance(result_data, str):
                        # If result is just a string, use it directly
                        resolved[key] = result_data
                    else:
                        logger.warning(f"Field '{field}' not found in step {step_id} result")
                        resolved[key] = value  # Keep placeholder
                else:
                    logger.warning(f"Step {step_id} result not available yet")
                    resolved[key] = value  # Keep placeholder
            else:
                resolved[key] = value
        else:
            resolved[key] = value
    
    return resolved


def fetch_mcp_prompt(tool_name: str, mcp_servers: List[MCPServerConfig]) -> Optional[str]:
    """
    Fetch tool-specific prompt from MCP server.
    
    Args:
        tool_name: Name of the tool
        mcp_servers: List of MCP server configurations
        
    Returns:
        Tool-specific prompt string, or None if not found
    """
    # Try each MCP server to find the tool's prompt
    for server in mcp_servers:
        try:
            # Attempt to fetch prompt from MCP server
            # Note: This assumes MCP servers expose a /tools/{name}/prompt endpoint
            # If your MCP servers don't have this, we'll use a fallback generic prompt
            
            url = f"{server.url}/tools/{tool_name}/prompt"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                prompt_data = response.json()
                if "prompt" in prompt_data:
                    logger.info(f"✓ Fetched MCP prompt for {tool_name} from {server.name}")
                    return prompt_data["prompt"]
        except Exception as e:
            logger.debug(f"Could not fetch prompt from {server.name}: {e}")
            continue
    
    # Fallback: Return generic prompt if MCP doesn't provide one
    logger.info(f"Using fallback generic prompt for {tool_name}")
    return f"""You are a specialized agent for the {tool_name} tool.
Your job is to use this tool effectively based on the parameters provided.
Be thorough and precise in your execution."""


def get_tool_instance(tool_name: str, all_tools: List[BaseTool]) -> Optional[BaseTool]:
    """
    Retrieve a LangChain tool instance by name.
    
    Args:
        tool_name: Name of the tool to retrieve
        all_tools: List of all available LangChain tools
        
    Returns:
        Tool instance or None if not found
    """
    logger.info(f"🔍 Searching for tool '{tool_name}' in {len(all_tools)} available tools")
    logger.info(f"   Available tool names: {[tool.name for tool in all_tools[:5]]}...")
    
    for tool in all_tools:
        if tool.name == tool_name:
            logger.info(f"   ✅ Found tool: {tool_name}")
            return tool
    
    logger.error(f"❌ Tool '{tool_name}' not found in available tools")
    logger.error(f"   All tool names: {[tool.name for tool in all_tools]}")
    return None


def format_chat_history(history: List[tuple], max_exchanges: int = 3) -> str:
    """
    Format conversation history for prompts.
    
    Args:
        history: List of (human, assistant) tuples
        max_exchanges: Maximum number of recent exchanges to include
        
    Returns:
        Formatted string of conversation history
    """
    if not history:
        return "No previous conversation."
    
    formatted = []
    for human, assistant in history[-max_exchanges:]:
        formatted.append(f"Human: {human}")
        formatted.append(f"Assistant: {assistant}")
    
    return "\n".join(formatted)


def parse_tool_result(result: Any) -> Dict[str, Any]:
    """
    Parse and structure tool execution results.
    
    Args:
        result: Raw result from tool execution
        
    Returns:
        Structured dictionary with result data
    """
    if isinstance(result, dict):
        return result
    elif isinstance(result, str):
        # Try to parse as JSON
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"output": result}
    else:
        return {"output": str(result)}
