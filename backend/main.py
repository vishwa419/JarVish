"""
FastAPI application for Jarvis - Multi-Agent LangGraph Edition
"""
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import logging
from typing import List

from backend.config import settings
from backend.models import MCPServerConfig
from backend.chain import create_jarvis_graph
from backend.multi_mcp_client import get_langchain_tools
from langgraph.graph import StateGraph

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Jarvis AI Assistant - Multi-Agent",
    description="LangGraph-based multi-agent RAG assistant",
    version="0.3.0"
)

# Setup templates
templates = Jinja2Templates(directory="frontend/templates")

# Global state
jarvis_graph: StateGraph = None
all_tools = None  # Will be populated at startup
mcp_servers = None
conversation_history: List[tuple] = []

# Ensure directories exist
settings.ensure_directories()

# MCP Server configurations
MCP_SERVERS = [
    MCPServerConfig(
        name="rag",
        url="http://localhost:8001",
        description="RAG server with Qdrant vector search and document management"
    ),
    MCPServerConfig(
        name="gcp_tools",
        url="http://localhost:8002",
        description="GCP tools with Gmail, Calendar, and Google Workspace integration"
    ),
]


@app.on_event("startup")
async def startup_event():
    """Initialize Jarvis multi-agent system on startup."""
    global jarvis_graph
    global  all_tools, mcp_servers
    
    logger.info("=" * 70)
    logger.info("🚀 Jarvis AI Assistant - Multi-Agent LangGraph Edition")
    logger.info("=" * 70)
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"OpenAI Model: {settings.openai_model}")
    logger.info(f"Server: http://{settings.backend_host}:{settings.backend_port}")
    logger.info(f"MCP Servers: {len(MCP_SERVERS)} configured")
    for srv in MCP_SERVERS:
        logger.info(f"   • {srv.name}: {srv.url}")
    logger.info("=" * 70)
    
    try:
        # Discover tools from MCP servers
        logger.info("🔍 Discovering tools from MCP servers...")
        
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        
        discovered_tools = await loop.run_in_executor(
            executor,
            get_langchain_tools,
            MCP_SERVERS
        )
        all_tools = discovered_tools
        if not discovered_tools:
            raise RuntimeError("No tools discovered from MCP servers")
        
        logger.info(f"✅ Discovered {len(all_tools)} tools:")
        for tool in all_tools:
            logger.info(f"   • {tool.name}")
        
        # Build LangGraph
        logger.info("🔧 Building multi-agent LangGraph...")
        mcp_servers = MCP_SERVERS
        jarvis_graph = create_jarvis_graph(mcp_servers, all_tools)
        
        logger.info("=" * 70)
        logger.info("✅ Jarvis multi-agent system initialized successfully!")
        logger.info(f"📊 Total tools: {len(all_tools)}")
        logger.info(f"🤖 Architecture: Planner → Executor(s) → Aggregator")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error("=" * 70)
        logger.error(f"❌ Failed to initialize Jarvis: {e}")
        logger.error("=" * 70)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("👋 Jarvis is shutting down...")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the chat interface."""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, message: str = Form(...)):
    """
    Handle chat messages using LangGraph multi-agent system.
    
    Args:
        message: User's message from form
        
    Returns:
        HTML fragment with assistant's response
    """
    global conversation_history
    
    try:
        # Validate
        if jarvis_graph is None:
            raise HTTPException(status_code=503, detail="Jarvis is not initialized yet")
        
        if not message or len(message.strip()) == 0:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if len(message) > 2000:
            raise HTTPException(status_code=400, detail="Message too long (max 2000 characters)")
        
        logger.info(f"💬 User: {message[:100]}...")
        
        # Debug: Check global tools
        logger.info(f"🔍 DEBUG: Global all_tools has {len(all_tools)} tools")
        if all_tools:
            logger.info(f"   First tool: {all_tools[0].name}")
        
        # Prepare initial state
        initial_state = {
            "user_message": message.strip(),
            "chat_history": conversation_history,
            "available_tools": [tool.name for tool in all_tools],
            "mcp_servers": mcp_servers,
            "subtasks": [],
            "execution_order": [],
            "current_step_index": 0,
            "step_results": {},
            "final_response": None,
            "error": None,
            "_all_tools": all_tools  # Pass reference to global tools list
        }
        
        # Debug: Verify tools in state
        logger.info(f"🔍 DEBUG: State _all_tools has {len(initial_state['_all_tools'])} tools")
        
        # Invoke graph
        logger.info("🚀 Invoking LangGraph...")
        final_state = jarvis_graph.invoke(initial_state)
        
        # Extract response
        response = final_state.get("final_response", "I apologize, but I couldn't process that request.")
        
        if final_state.get("error"):
            response = f"⚠️ {final_state['error']}\n\n{response}"
        
        # Update conversation history
        conversation_history.append((message, response))
        
        # Keep only last 5 exchanges
        if len(conversation_history) > 5:
            conversation_history = conversation_history[-5:]
        
        logger.info("✅ Request completed successfully")
        
        # Return HTML fragments for HTMX
        return f"""
        <div class="message user-message">
            <div class="message-content">{message}</div>
        </div>
        <div class="message assistant-message">
            <div class="message-content">{response}</div>
        </div>
        """
        
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {str(e)}")
        return f"""
        <div class="message error-message">
            <div class="message-content">❌ Error: {str(e)}</div>
        </div>
        """


@app.post("/clear")
async def clear_history():
    """Clear conversation history."""
    global conversation_history
    
    if jarvis_graph is None:
        raise HTTPException(status_code=503, detail="Jarvis is not initialized yet")
    
    conversation_history.clear()
    logger.info("🔄 Conversation history cleared")
    
    return {"status": "success", "message": "History cleared"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if jarvis_graph is not None else "initializing",
        "jarvis_ready": jarvis_graph is not None,
        "environment": settings.environment,
        "model": settings.openai_model,
        "architecture": "multi-agent-langgraph",
        "mcp_servers": len(MCP_SERVERS),
        "total_tools": len(all_tools),
        "conversation_exchanges": len(conversation_history)
    }


@app.get("/tools")
async def list_tools():
    """List all available tools from all MCP servers."""
    if jarvis_graph is None:
        raise HTTPException(status_code=503, detail="Jarvis is not initialized yet")
    
    return {
        "total_tools": len(all_tools),
        "tools": [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in all_tools
        ],
        "mcp_servers": [
            {
                "name": srv.name,
                "url": srv.url,
                "description": srv.description
            }
            for srv in MCP_SERVERS
        ]
    }


@app.get("/architecture")
async def get_architecture():
    """Return information about the multi-agent architecture."""
    return {
        "architecture": "multi-agent-langgraph",
        "nodes": [
            {
                "name": "planner",
                "role": "Decompose user request into subtasks",
                "output": "List of subtasks with dependencies"
            },
            {
                "name": "dependency_resolver",
                "role": "Order subtasks for sequential execution",
                "output": "Execution order"
            },
            {
                "name": "executor",
                "role": "Execute one subtask at a time with tool-specific ReAct agent",
                "output": "Task result (loops until all done)"
            },
            {
                "name": "aggregator",
                "role": "Synthesize all results into final response",
                "output": "Final user-facing response"
            }
        ],
        "edges": [
            "START → planner",
            "planner → dependency_resolver",
            "dependency_resolver → executor",
            "executor → executor (loop if more tasks)",
            "executor → aggregator (when done)",
            "aggregator → END"
        ],
        "key_features": [
            "Tool-specific ReAct agents per subtask",
            "MCP prompts fetched at runtime",
            "Dependency resolution with placeholders",
            "Sequential execution (parallelization not yet implemented)"
        ]
    }
