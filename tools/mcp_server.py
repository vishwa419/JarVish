"""
MCP (Model Context Protocol) Server with FastMCP
Single source of truth for all tool definitions.
"""
import logging
from typing import Optional

from backend.config import settings
from tools.rag_tool import rag_tool as rag_instance
from tools.calendar_tool import calendar_tool as calendar_instance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import FastMCP after basic setup
from fastmcp import FastMCP

# Initialize FastMCP server - this is our SINGLE SOURCE OF TRUTH
mcp = FastMCP("JarvisMCP")


@mcp.tool()
def rag_search(query: str, top_k: int = 3) -> str:
    """
    Search through local documents for relevant information using semantic similarity.
    
    Use this tool when the user asks about:
    - Finding information in their documents
    - Looking up specific topics, people, or concepts
    - Retrieving details from their knowledge base
    - Questions about project documentation
    - Any query that needs grounding in their personal documents
    
    Args:
        query: The search query or topic to find in documents
        top_k: Number of relevant document chunks to return (default: 3)
        
    Returns:
        Formatted text containing the most relevant document passages
    """
    try:
        results = rag_instance.search(query, top_k=top_k)
        
        if not results:
            return f"No documents found matching '{query}'. The knowledge base may be empty or the query is too specific."
        
        formatted = rag_instance.format_results_for_llm(results)
        return f"Found {len(results)} relevant document(s) for '{query}':\n\n{formatted}"
        
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return f"Error searching documents: {str(e)}"


@mcp.tool()
def add_event(title: str, date: str, time: str, description: str = "") -> str:
    """
    Schedule a new calendar event.
    
    Use this tool when the user wants to:
    - Schedule a meeting or appointment
    - Create a reminder
    - Add an event to their calendar
    - Set up a call or gathering
    
    Args:
        title: Name/title of the event (e.g., "Team Meeting", "Lunch with Alice")
        date: Date in YYYY-MM-DD format (e.g., "2025-11-02")
        time: Time in HH:MM 24-hour format (e.g., "15:00" for 3pm, "09:30" for 9:30am)
        description: Optional additional details about the event
        
    Returns:
        Confirmation message with event details
    """
    try:
        event = calendar_instance.add_event(title, date, time, description)
        
        return f"""✅ Event scheduled successfully!

Title: {title}
Date: {date}
Time: {time}
{f'Description: {description}' if description else ''}

Event ID: {event['id']}"""
        
    except Exception as e:
        logger.error(f"Error in add_event: {e}")
        return f"❌ Failed to schedule event: {str(e)}"


@mcp.tool()
def get_events(date: Optional[str] = None) -> str:
    """
    Retrieve calendar events from the schedule.
    
    Use this when the user wants to:
    - See their schedule
    - Check what events are coming up
    - View events for a specific date
    - Ask "what's on my calendar"
    
    Args:
        date: Optional date filter in YYYY-MM-DD format. 
              If not provided, returns all events.
              Example: "2025-11-02" for November 2nd, 2025
        
    Returns:
        Formatted list of scheduled events
    """
    try:
        events = calendar_instance.get_events(date)
        
        if not events:
            if date:
                return f"No events scheduled for {date}."
            return "No events in calendar."
        
        # Format events nicely
        formatted = []
        for event in sorted(events, key=lambda x: (x.get('date', ''), x.get('time', ''))):
            event_str = f"• {event['title']} - {event['date']} at {event['time']}"
            if event.get('description'):
                event_str += f"\n  Description: {event['description']}"
            formatted.append(event_str)
        
        header = f"📅 Events for {date}:" if date else "📅 All scheduled events:"
        return f"{header}\n\n" + "\n\n".join(formatted)
        
    except Exception as e:
        logger.error(f"Error in get_events: {e}")
        return f"❌ Error retrieving events: {str(e)}"


@mcp.tool()
def reload_documents() -> str:
    """
    Reload all documents from the docs folder into the vector database.
    
    Use this when:
    - User mentions adding new documents
    - User says documents have been updated
    - The search results seem outdated
    - User explicitly asks to refresh/reload documents
    
    Returns:
        Status message with number of chunks loaded
    """
    try:
        count = rag_instance.reload_documents()
        return f"✅ Successfully reloaded {count} document chunks from {settings.docs_path}"
    except Exception as e:
        logger.error(f"Reload error: {e}")
        return f"❌ Error reloading documents: {str(e)}"


# Log registered tools on import
logger.info("🔧 FastMCP Server initialized with tools:")
logger.info("  ✓ rag_search - Search documents semantically")
logger.info("  ✓ add_event - Schedule calendar events") 
logger.info("  ✓ get_events - View scheduled events")
logger.info("  ✓ reload_documents - Refresh document database")


# Only run server if executed directly (not when imported)
if __name__ == "__main__":
    logger.info(f"🚀 Starting MCP Server as standalone...")
    mcp.run(transport="http")
    
