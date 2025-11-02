"""
ReAct Agent with HTTP-based MCP Tools
Tools are automatically discovered from FastMCP HTTP server.
Pattern based on repoScoutWorkflow.py - synchronous with asyncio.run() in __init__ only.
"""
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from typing import List
import logging

from backend.config import settings
from backend.mcp_client import get_langchain_tools

logger = logging.getLogger(__name__)


# ReAct Prompt Template
REACT_PROMPT = """You are Jarvis, an intelligent AI assistant with access to tools.

TOOLS:
------
You have access to the following tools:

{tools}

RESPONSE FORMAT:
---------------
To use a tool, use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

GUIDELINES:
----------
- Use rag_search when users ask about information that might be in their documents
- Use add_event when users want to schedule something (format: "title|date|time|description")
- Use get_events when users ask about their schedule (format: "YYYY-MM-DD" or "" for all)
- You can use multiple tools in sequence to answer a question
- Always explain what you found when using rag_search
- Be proactive and helpful

Begin!

Previous conversation:
{chat_history}

New question: {input}
{agent_scratchpad}"""


class JarvisChain:
    """ReAct Agent with HTTP-based MCP tools."""
    
    def __init__(self, mcp_server_url: str = "http://localhost:8000"):
        """
        Initialize ReAct agent configuration (without connecting to MCP yet).
        Actual MCP connection happens in async create() method.
        
        Args:
            mcp_server_url: URL of the MCP server
        """
        self.mcp_server_url = mcp_server_url
        self.chat_history: List[tuple] = []
        
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            openai_api_key=settings.openai_api_key
        )
        
        # These will be set by _initialize_tools()
        self.tools = []
        self.agent = None
        self.agent_executor = None
    
    @classmethod
    async def create(cls, mcp_server_url: str = "http://localhost:8000") -> "JarvisChain":
        """
        Async factory method to create and initialize JarvisChain.
        Use this instead of __init__ when event loop is already running.
        
        Args:
            mcp_server_url: URL of the MCP server
            
        Returns:
            Initialized JarvisChain instance
        """
        instance = cls(mcp_server_url)
        await instance._initialize_tools()
        return instance
    
    async def _initialize_tools(self):
        """Initialize tools from MCP server (async operation)."""
        logger.info(f"🔍 Auto-discovering tools from MCP HTTP server at {self.mcp_server_url}...")
        
        # Run get_langchain_tools in thread pool to avoid blocking
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        
        # Run the blocking asyncio.run() call in a separate thread
        self.tools = await loop.run_in_executor(
            executor,
            get_langchain_tools,
            self.mcp_server_url
        )
        
        if not self.tools:
            logger.error("❌ No tools discovered from MCP server!")
            logger.error(f"Make sure MCP server is running at {self.mcp_server_url}")
            raise RuntimeError("Failed to discover tools from MCP HTTP server")
        
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
        
        The agent will:
        1. Think about what tools it needs
        2. Execute tools as needed (each tool call uses asyncio.run internally)
        3. Observe results
        4. Repeat until it has enough information
        5. Generate final answer
        
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
            
            # Run agent (this handles the full ReAct loop)
            # Agent executor is synchronous, tools use asyncio.run() internally
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
    
    def reload_tools(self) -> int:
        """
        Reload tools from MCP server (useful if tools change at runtime).
        Note: This is synchronous and will block. Consider using async version.
        
        Returns:
            Number of tools loaded
        """
        logger.info("🔄 Reloading tools from MCP server...")
        
        # This will use asyncio.run() which may cause issues in async context
        # Use with caution or call from sync context only
        self.tools = get_langchain_tools(self.mcp_server_url)
        
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
    
    async def reload_tools_async(self) -> int:
        """
        Async version of reload_tools for use in async contexts.
        
        Returns:
            Number of tools loaded
        """
        logger.info("🔄 Reloading tools from MCP server (async)...")
        
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        
        # Run in thread pool to avoid blocking
        self.tools = await loop.run_in_executor(
            executor,
            get_langchain_tools,
            self.mcp_server_url
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
