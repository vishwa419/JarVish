"""
FastAPI application entry point.
Main web server for Jarvis assistant.
"""
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import logging
from pathlib import Path

from backend.config import settings
from backend.chain import JarvisChain
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
    description="RAG-powered AI assistant with tool integration",
    version="0.1.0"
)

# Setup templates
templates = Jinja2Templates(directory="frontend/templates")

# Declare Jarvis chain (will be initialized in startup event)
jarvis = None

# Ensure directories exist
settings.ensure_directories()


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    global jarvis
    
    logger.info("🚀 Jarvis is starting up...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"OpenAI Model: {settings.openai_model}")
    logger.info(f"Server: http://{settings.backend_host}:{settings.backend_port}")
    
    # Initialize Jarvis chain here (after event loop is running)
    logger.info("🔧 Initializing Jarvis agent...")
    try:
        jarvis = await JarvisChain.create()
        logger.info("✅ Jarvis agent initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Jarvis: {e}")
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
        "model": settings.openai_model
    }
