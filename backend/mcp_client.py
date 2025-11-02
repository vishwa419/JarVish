"""
MCP HTTP Client - Connects to FastMCP server over HTTP
Pattern based on repoScoutWorkflow.py - handles async properly.
"""
import logging
import asyncio
from typing import List, Dict, Any
from langchain.tools import Tool
from fastmcp import Client

logger = logging.getLogger(__name__)


class MCPClientWrapper:
    """Wrapper for FastMCP Client that handles async properly."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize MCP client wrapper.
        
        Args:
            base_url: Base URL of the MCP server
        """
        self.base_url = base_url.rstrip('/')
        # Store client but don't keep it in context
        self.mcp_client = Client(f"{self.base_url}/mcp")
        self._tools_cache: List[Dict[str, Any]] = []
        
    async def _async_list_tools(self) -> List[Dict[str, Any]]:
        """List tools using fresh context."""
        async with self.mcp_client:
            result = await self.mcp_client.list_tools()
            
            # Handle different return types from FastMCP
            if hasattr(result, 'tools'):
                tools_list = result.tools
            else:
                tools_list = result
            
            # Convert Pydantic Tool objects to dicts
            tools_as_dicts = []
            for tool in tools_list:
                if hasattr(tool, 'model_dump'):
                    # Pydantic v2
                    tools_as_dicts.append(tool.model_dump())
                elif hasattr(tool, 'dict'):
                    # Pydantic v1
                    tools_as_dicts.append(tool.dict())
                elif isinstance(tool, dict):
                    tools_as_dicts.append(tool)
                else:
                    # Convert to dict manually
                    tools_as_dicts.append({
                        'name': getattr(tool, 'name', str(tool)),
                        'description': getattr(tool, 'description', ''),
                        'inputSchema': getattr(tool, 'inputSchema', {})
                    })
            
            return tools_as_dicts
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """Synchronous wrapper for listing tools."""
        return asyncio.run(self._async_list_tools())
    
    async def _async_call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call tool using fresh context."""
        async with self.mcp_client:
            result = await self.mcp_client.call_tool(tool_name, arguments)
            
            # Extract text from result
            if hasattr(result, 'content'):
                content = result.content
                if isinstance(content, list) and len(content) > 0:
                    # Handle list of content items
                    if hasattr(content[0], 'text'):
                        return content[0].text
                    return str(content[0])
                return str(content)
            
            return str(result)
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Synchronous wrapper for calling tools.
        Detects if event loop is running and handles appropriately.
        """
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - can't use asyncio.run()
            # Need to use run_coroutine_threadsafe or similar
            import concurrent.futures
            
            # Create a new thread to run the async call
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._async_call_tool(tool_name, arguments)
                )
                return future.result()
                
        except RuntimeError:
            # No event loop running - safe to use asyncio.run()
            return asyncio.run(self._async_call_tool(tool_name, arguments))


def format_tools_for_langchain(mcp_tools: List[Dict[str, Any]], mcp_wrapper: MCPClientWrapper) -> List[Tool]:
    """
    Convert MCP tools to LangChain Tool format.
    
    Args:
        mcp_tools: List of MCP tool definitions (as dictionaries)
        mcp_wrapper: MCP client wrapper instance
        
    Returns:
        List of LangChain Tool objects
    """
    langchain_tools = []
    
    for tool_info in mcp_tools:
        # Handle both dict and object access
        if isinstance(tool_info, dict):
            tool_name = tool_info.get('name')
            tool_description = tool_info.get('description', f"Tool: {tool_name}")
            input_schema = tool_info.get('inputSchema', {})
        else:
            # Handle Pydantic objects
            tool_name = getattr(tool_info, 'name', None)
            tool_description = getattr(tool_info, 'description', f"Tool: {tool_name}")
            input_schema = getattr(tool_info, 'inputSchema', {})
        
        if not tool_name:
            logger.warning(f"⚠️ Skipping tool with no name: {tool_info}")
            continue
        
        # Create wrapper function for this specific tool
        def create_tool_func(name: str, schema: Dict[str, Any]):
            """Create a sync tool function that calls MCP via asyncio.run()."""
            
            def tool_func(tool_input: Any) -> str:
                """Synchronous tool function that wraps async MCP call."""
                try:
                    # Parse input into arguments
                    if isinstance(tool_input, dict):
                        arguments = tool_input
                    elif isinstance(tool_input, str):
                        # Handle string inputs
                        properties = schema.get('properties', {})
                        param_names = list(properties.keys())
                        
                        if '|' in tool_input:
                            # Pipe-separated format: "arg1|arg2|arg3"
                            parts = [p.strip() for p in tool_input.split('|')]
                            arguments = {}
                            for i, param_name in enumerate(param_names):
                                if i < len(parts) and parts[i]:
                                    arguments[param_name] = parts[i]
                        else:
                            # Single parameter
                            if param_names:
                                arguments = {param_names[0]: tool_input}
                            else:
                                arguments = {"input": tool_input}
                    else:
                        arguments = {"input": str(tool_input)}
                    
                    # Call tool synchronously (it uses asyncio.run internally)
                    result = mcp_wrapper.call_tool(name, arguments)
                    return result
                    
                except Exception as e:
                    error_msg = f"Error calling {name}: {str(e)}"
                    logger.error(error_msg)
                    return f"❌ {error_msg}"
            
            return tool_func
        
        # Create LangChain tool
        langchain_tool = Tool(
            name=tool_name,
            func=create_tool_func(tool_name, input_schema),
            description=tool_description
        )
        
        langchain_tools.append(langchain_tool)
        logger.info(f"  ✓ Registered: {tool_name}")
    
    return langchain_tools


def get_langchain_tools(base_url: str = "http://localhost:8000") -> List[Tool]:
    """
    Get LangChain tools from MCP server.
    This is called synchronously from __init__ with asyncio.run().
    
    Args:
        base_url: MCP server base URL
        
    Returns:
        List of LangChain Tool objects
    """
    logger.info(f"🔌 Connecting to MCP server at {base_url}")
    
    # Create wrapper
    mcp_wrapper = MCPClientWrapper(base_url)
    
    # List tools (uses asyncio.run internally)
    mcp_tools = mcp_wrapper.list_tools()
    
    if not mcp_tools:
        logger.warning("⚠️ No tools discovered from MCP server")
        return []
    
    logger.info(f"📋 Discovered {len(mcp_tools)} tools from MCP server")
    
    # Convert to LangChain format
    langchain_tools = format_tools_for_langchain(mcp_tools, mcp_wrapper)
    
    logger.info(f"✅ Converted {len(langchain_tools)} MCP tools to LangChain format")
    
    return langchain_tools


# Initialize on import
logger.info("🔧 MCP HTTP Client initialized - ready to connect to server")
