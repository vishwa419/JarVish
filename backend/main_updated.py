"""
FastAPI application entry point - Updated for Multi-MCP support
Main web server for Jarvis assistant.
"""
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import logging
from pathlib import Path

from backend.config import settings
from backend.chain_updated import JarvisChain
from backend.multi_mcp_client import MCPServerConfig
from backend.models import ChatRequest, ChatResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Jarvis AI Assistant",
    description="Multi-MCP RAG-powered AI assistant",
    version="0.2.0"
)

# Setup templates
templates = Jinja2Templates(directory="frontend/templates")

# Declare Jarvis chain (will be initialized in startup event)
jarvis = None

# Ensure directories exist
settings.ensure_directories()

# MCP Server configurations
MCP_SERVERS = [
    MCPServerConfig(
        name="rag",
        url="http://localhost:8001",
        description="RAG server with Qdrant vector search and document management"
    ),
    # Future servers will be added here:
    MCPServerConfig(
        name="utility",
        url="http://localhost:8002",
        description="Utility server with Gmail, Calendar google workspace tools"
    ),
]


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    global jarvis
    
    logger.info("=" * 70)
    logger.info("🚀 Jarvis AI Assistant - Multi-MCP Edition")
    logger.info("=" * 70)
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"OpenAI Model: {settings.openai_model}")
    logger.info(f"Server: http://{settings.backend_host}:{settings.backend_port}")
    logger.info(f"MCP Servers: {len(MCP_SERVERS)} configured")
    for srv in MCP_SERVERS:
        logger.info(f"   • {srv.name}: {srv.url}")
    logger.info("=" * 70)
    
    # Initialize Jarvis chain with multi-MCP support
    logger.info("🔧 Initializing Jarvis agent with multi-MCP support...")
    try:
        jarvis = await JarvisChain.create(mcp_server_configs=MCP_SERVERS)
        logger.info("=" * 70)
        logger.info("✅ Jarvis initialized successfully!")
        logger.info(f"📊 Total tools available: {len(jarvis.get_available_tools())}")
        logger.info("=" * 70)
    except Exception as e:
        logger.error("=" * 70)
        logger.error(f"❌ Failed to initialize Jarvis: {e}")
        logger.error("=" * 70)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("👋 Jarvis is shutting down...")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the chat interface."""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, message: str = Form(...)):
    """
    Handle chat messages from HTMX.
    
    Args:
        message: User's message from form
        
    Returns:
        HTML fragment with assistant's response
    """
    try:
        # Check if Jarvis is initialized
        if jarvis is None:
            raise HTTPException(status_code=503, detail="Jarvis is not initialized yet")
        
        # Validate message
        if not message or len(message.strip()) == 0:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if len(message) > 2000:
            raise HTTPException(status_code=400, detail="Message too long (max 2000 characters)")
        
        # Process message through Jarvis
        response = await jarvis.process_message(message.strip())
        
        # Return HTML fragments for HTMX to inject
        return f"""
        <div class="message user-message">
            <div class="message-content">{message}</div>
        </div>
        <div class="message assistant-message">
            <div class="message-content">{response}</div>
        </div>
        """
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return f"""
        <div class="message error-message">
            <div class="message-content">❌ Error: {str(e)}</div>
        </div>
        """


@app.post("/clear")
async def clear_history():
    """Clear conversation history."""
    if jarvis is None:
        raise HTTPException(status_code=503, detail="Jarvis is not initialized yet")
    
    jarvis.clear_history()
    logger.info("Conversation history cleared")
    return {"status": "success", "message": "History cleared"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if jarvis is not None else "initializing",
        "jarvis_ready": jarvis is not None,
        "environment": settings.environment,
        "model": settings.openai_model,
        "mcp_servers": len(MCP_SERVERS),
        "total_tools": len(jarvis.get_available_tools()) if jarvis else 0
    }


@app.get("/tools")
async def list_tools():
    """List all available tools from all MCP servers."""
    if jarvis is None:
        raise HTTPException(status_code=503, detail="Jarvis is not initialized yet")
    
    return {
        "tools": jarvis.get_available_tools(),
        "server_info": jarvis.get_server_info()
    }


@app.post("/reload-tools")
async def reload_tools():
    """Reload tools from all MCP servers."""
    if jarvis is None:
        raise HTTPException(status_code=503, detail="Jarvis is not initialized yet")
    
    try:
        tool_count = await jarvis.reload_tools_async()
        return {
            "status": "success",
            "message": f"Reloaded {tool_count} tools",
            "tools": jarvis.get_available_tools()
        }
    except Exception as e:
        logger.error(f"Error reloading tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))
