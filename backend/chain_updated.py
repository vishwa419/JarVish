"""
ReAct Agent with Multi-MCP Support
Discovers and uses tools from multiple MCP servers automatically.
"""
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from typing import List
import logging

from backend.config import settings
from backend.multi_mcp_client import get_langchain_tools, MCPServerConfig

logger = logging.getLogger(__name__)


# ReAct Prompt Template
REACT_PROMPT = """You are Jarvis, an intelligent AI assistant with access to multiple tools from different services.

TOOLS:
------
You have access to the following tools:

{tools}

RESPONSE FORMAT:
---------------
To use a tool, use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: {{"param1": "value1", "param2": "value2"}}  <-- MUST BE VALID JSON
Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

GUIDELINES:
----------
- Use qdrant_search when users ask about information in their documents
- Use qdrant_store when users upload new documents (filename will be provided)
- Use list_documents to see what documents are available
- Use calendar tools for scheduling and viewing events
- Use Gmail tools for searching emails
- You can use multiple tools in sequence to answer complex questions
- Always explain what you found when using search tools
- Be proactive and helpful

Begin!

Previous conversation:
{chat_history}

New question: {input}
{agent_scratchpad}"""

class JarvisChain:
    """ReAct agent with multi-MCP support"""

    def __init__(self, mcp_server_configs: List[MCPServerConfig] = None):
        """
        Initialize ReAct agent configuration.
        
        Args:
            mcp_server_configs: List of MCP server configurations
                               If None, uses default RAG server only
        """
        if mcp_server_configs is None:
            mcp_server_configs = [
                MCPServerConfig(
                    name = "rag",
                    url = "http://localhost:8001",
                    description="RAG server with Qdrant vector search"
                    ),
                MCPServerConfig(
                    name = "gcp_tools",
                    url = "http://localhost:8002",
                    description="GCP tools MCP access"
                    )
            ]

        self.mcp_server_configs = mcp_server_configs
        self.chat_history: List[tuple] = []

        self.llm = ChatOpenAI(
                model = settings.openai_model,
                temperature = 0.7,
                openai_api_key=settings.openai_api_key
        )

        self.tools = []
        self.agent = None
        self.agent_executor = None

    @classmethod
    async def create(cls, mcp_server_configs: List[MCPServerConfig] = None) -> "JarvisChain":
        """
        Async factory method to create and initialize JarvisChain.
        
        Args:
            mcp_server_configs: List of MCP server configurations
            
        Returns:
            Initialized JarvisChain instance
        """
        instance = cls(mcp_server_configs)
        await instance._initialize_tools()
        return instance
    
    async def _initialize_tools(self):
        """Initialize tools from all configured MCP servers."""
        logger.info(f"🔍 Auto-discovering tools from {len(self.mcp_server_configs)} MCP servers...")
        
        # Log configured servers
        for config in self.mcp_server_configs:
            logger.info(f"   📡 {config.name}: {config.url}")
        
        # Run get_langchain_tools in thread pool to avoid blocking
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()

        self.tools = await loop.run_in_executor(
                executor,
                get_langchain_tools,
                self.mcp_server_configs
        )

        if not self.tools:
            logger.error("❌ No tools discovered from any MCP server!")
            raise RuntimeError("Failed to discover tools from MCP servers")
        
        logger.info(f"✅ Discovered {len(self.tools)} tools:")
        for tool in self.tools:
            logger.info(f"   • {tool.name}")
        
        # Create ReAct prompt
        prompt = PromptTemplate.from_template(REACT_PROMPT)
        
        # Create ReAct agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,  # Shows thinking process in terminal
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=False
        )
        
        logger.info("✅ ReAct Agent initialized successfully")
    def _format_chat_history(self) -> str:
        """Format chat history for prompt."""
        if not self.chat_history:
            return "No previous conversation."
        
        formatted = []
        for human, ai in self.chat_history[-3:]:  # Last 3 exchanges
            formatted.append(f"Human: {human}")
            formatted.append(f"Assistant: {ai}")
        
        return "\n".join(formatted)
    
    async def process_message(self, user_message: str) -> str:
        """
        Process user message using ReAct agent.
        
        Args:
            user_message: User's input
            
        Returns:
            Agent's response
        """
        try:
            logger.info(f"💬 Processing: {user_message[:50]}...")
            
            # Prepare input
            agent_input = {
                "input": user_message,
                "chat_history": self._format_chat_history()
            }
            
            # Run agent (synchronous, but tools use asyncio.run internally)
            result = self.agent_executor.invoke(agent_input)
            
            # Extract response
            response = result.get("output", "I apologize, but I couldn't process that request.")
            
            # Add to history
            self.chat_history.append((user_message, response))
            
            logger.info("✅ Agent completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in agent execution: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.chat_history.clear()
        logger.info("🔄 Conversation history cleared")
    
    def get_history(self) -> List[tuple]:
        """Get conversation history."""
        return self.chat_history
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.tools]
    
    def get_server_info(self) -> dict:
        """Get information about connected MCP servers."""
        return {
            "servers": [
                {
                    "name": config.name,
                    "url": config.url,
                    "description": config.description
                }
                for config in self.mcp_server_configs
            ],
            "total_tools": len(self.tools),
            "tools": self.get_available_tools()
        }
    
    async def reload_tools_async(self) -> int:
        """
        Async version of reload_tools for use in async contexts.
        
        Returns:
            Number of tools loaded
        """
        logger.info("🔄 Reloading tools from all MCP servers (async)...")
        
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        
        # Run in thread pool to avoid blocking
        self.tools = await loop.run_in_executor(
            executor,
            get_langchain_tools,
            self.mcp_server_configs
        )
        
        # Recreate agent with new tools
        prompt = PromptTemplate.from_template(REACT_PROMPT)
        self.agent = create_react_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=False
        )
        
        logger.info(f"✅ Reloaded {len(self.tools)} tools")
        return len(self.tools)
