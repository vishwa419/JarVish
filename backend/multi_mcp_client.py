"""
Multi-MCP Client - Connects to multiple FastMCP servers over HTTP
Aggregates tools from all servers and handles routing.
"""
import logging
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain.tools import Tool
from fastmcp import Client

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    name: str
    url: str
    description: str = ""


class MCPClientWrapper:
    """Wrapper for a single FastMCP Client that handles async properly."""
    
    def __init__(self, server_name: str, base_url: str):
        """
        Initialize MCP client wrapper for a single server.
        
        Args:
            server_name: Name identifier for this server
            base_url: Base URL of the MCP server
        """
        self.server_name = server_name
        self.base_url = base_url.rstrip('/')
        self.mcp_client = Client(f"{self.base_url}/mcp")
        
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
            logger.info(f"🔍 DEBUG - tool_name: {tool_name}")
            logger.info(f"🔍 DEBUG - arguments type: {type(arguments)}")
            logger.info(f"🔍 DEBUG - arguments value: {arguments}")

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


class MultiMCPClientWrapper:
    """Manages connections to multiple MCP servers and aggregates tools."""
    
    def __init__(self, server_configs: List[MCPServerConfig]):
        """
        Initialize multi-MCP client wrapper.
        
        Args:
            server_configs: List of MCP server configurations
        """
        self.server_configs = server_configs
        self.clients: Dict[str, MCPClientWrapper] = {}
        self.tool_to_server: Dict[str, str] = {}  # Maps tool_name -> server_name
        
        # Create client wrappers for each server
        for config in server_configs:
            self.clients[config.name] = MCPClientWrapper(
                server_name=config.name,
                base_url=config.url
            )
            logger.info(f"📡 Registered MCP server: {config.name} at {config.url}")
    
    async def _async_discover_all_tools(self) -> List[Dict[str, Any]]:
        """Discover tools from all MCP servers asynchronously."""
        all_tools = []
        
        for server_name, client in self.clients.items():
            try:
                logger.info(f"🔍 Discovering tools from {server_name}...")
                tools = await client._async_list_tools()
                
                # Track which server each tool belongs to
                for tool in tools:
                    tool_name = tool.get('name') or tool.get('name')
                    
                    # Check for duplicate tool names
                    if tool_name in self.tool_to_server:
                        logger.warning(
                            f"⚠️ Duplicate tool name '{tool_name}' found! "
                            f"Already exists in '{self.tool_to_server[tool_name]}', "
                            f"now also in '{server_name}'. Using the first one."
                        )
                        continue
                    
                    # Register tool -> server mapping
                    self.tool_to_server[tool_name] = server_name
                    
                    # Add server context to tool metadata
                    tool['_mcp_server'] = server_name
                    all_tools.append(tool)
                
                logger.info(f"✅ Found {len(tools)} tools from {server_name}")
                
            except Exception as e:
                logger.error(f"❌ Error discovering tools from {server_name}: {e}")
                # Continue with other servers
        
        return all_tools
    
    def discover_all_tools(self) -> List[Dict[str, Any]]:
        """Synchronous wrapper for discovering all tools."""
        return asyncio.run(self._async_discover_all_tools())
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Route tool call to the appropriate MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        # Find which server handles this tool
        server_name = self.tool_to_server.get(tool_name)
        
        if not server_name:
            error_msg = f"Tool '{tool_name}' not found in any MCP server"
            logger.error(error_msg)
            return f"❌ {error_msg}"
        
        # Get the client for this server
        client = self.clients.get(server_name)
        
        if not client:
            error_msg = f"MCP server '{server_name}' not available"
            logger.error(error_msg)
            return f"❌ {error_msg}"
        
        # Call the tool on the appropriate server
        logger.debug(f"📞 Routing {tool_name} to {server_name}")
        return client.call_tool(tool_name, arguments)
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get information about all connected servers and their tools."""
        info = {
            "total_servers": len(self.clients),
            "total_tools": len(self.tool_to_server),
            "servers": {}
        }
        
        for server_name, client in self.clients.items():
            server_tools = [
                tool_name for tool_name, srv in self.tool_to_server.items()
                if srv == server_name
            ]
            info["servers"][server_name] = {
                "url": client.base_url,
                "tools": server_tools,
                "tool_count": len(server_tools)
            }
        
        return info


def format_tools_for_langchain(
    mcp_tools: List[Dict[str, Any]], 
    multi_client: MultiMCPClientWrapper
) -> List[Tool]:
    """
    Convert MCP tools to LangChain Tool format.
    
    Args:
        mcp_tools: List of MCP tool definitions (as dictionaries)
        multi_client: Multi-MCP client wrapper instance
        
    Returns:
        List of LangChain Tool objects
    """
    langchain_tools = []
    
    for tool_info in mcp_tools:
        # Extract tool information
        if isinstance(tool_info, dict):
            tool_name = tool_info.get('name')
            tool_description = tool_info.get('description', f"Tool: {tool_name}")
            input_schema = tool_info.get('inputSchema', {})
            server_name = tool_info.get('_mcp_server', 'unknown')
        else:
            tool_name = getattr(tool_info, 'name', None)
            tool_description = getattr(tool_info, 'description', f"Tool: {tool_name}")
            input_schema = getattr(tool_info, 'inputSchema', {})
            server_name = getattr(tool_info, '_mcp_server', 'unknown')
        
        if not tool_name:
            logger.warning(f"⚠️ Skipping tool with no name: {tool_info}")
            continue
        
        # Create wrapper function for this specific tool
        def create_tool_func(name: str, schema: Dict[str, Any], srv_name: str):
            """Create a sync tool function that calls MCP via multi-client."""
            
            def tool_func(tool_input: Any) -> str:
                """Synchronous tool function that wraps async MCP call."""
                try:
                    # Parse input into arguments
                    logger.info(f"🔍 DEBUG - tool_input type: {type(tool_input)}")
                    logger.info(f"🔍 DEBUG - tool_input value: {tool_input}")
                    if isinstance(tool_input, dict):
                        arguments = tool_input
                    elif isinstance(tool_input, str):
                        try:
                            arguments = json.loads(tool_input)
                        except json.JSONDecodeError:
                            # Not valid JSON, fall back to other parsing methods
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
                    
                    # Call tool via multi-client (routes to correct server)
                    result = multi_client.call_tool(name, arguments)
                    return result
                    
                except Exception as e:
                    error_msg = f"Error calling {name} on {srv_name}: {str(e)}"
                    logger.error(error_msg)
                    return f"❌ {error_msg}"
            
            return tool_func
        
        # Create LangChain tool
        langchain_tool = Tool(
            name=tool_name,
            func=create_tool_func(tool_name, input_schema, server_name),
            description=f"{tool_description} [Server: {server_name}]"
        )
        
        langchain_tools.append(langchain_tool)
        logger.info(f"  ✓ Registered: {tool_name} (from {server_name})")
    
    return langchain_tools


def get_langchain_tools(server_configs: List[MCPServerConfig]) -> List[Tool]:
    """
    Get LangChain tools from multiple MCP servers.
    
    Args:
        server_configs: List of MCP server configurations
        
    Returns:
        List of LangChain Tool objects
    """
    logger.info(f"🔌 Connecting to {len(server_configs)} MCP servers...")
    
    # Create multi-client wrapper
    multi_client = MultiMCPClientWrapper(server_configs)
    
    # Discover all tools
    mcp_tools = multi_client.discover_all_tools()
    
    if not mcp_tools:
        logger.warning("⚠️ No tools discovered from any MCP server")
        return []
    
    logger.info(f"📋 Discovered {len(mcp_tools)} total tools from all servers")
    
    # Log server info
    server_info = multi_client.get_server_info()
    logger.info(f"📊 Server summary:")
    for srv_name, srv_data in server_info["servers"].items():
        logger.info(f"   • {srv_name}: {srv_data['tool_count']} tools")
    
    # Convert to LangChain format
    langchain_tools = format_tools_for_langchain(mcp_tools, multi_client)
    
    logger.info(f"✅ Converted {len(langchain_tools)} MCP tools to LangChain format")
    
    return langchain_tools


# Initialize on import
logger.info("🔧 Multi-MCP HTTP Client initialized - ready to connect to multiple servers")
