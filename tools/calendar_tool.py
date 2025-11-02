"""
Calendar Tool - Schedule and manage events
"""
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging

from backend.config import settings

logger = logging.getLogger(__name__)


class CalendarTool:
    """
    Manages calendar events stored in JSON format.
    """
    
    def __init__(self):
        """Initialize calendar tool."""
        self.calendar_path = Path(settings.calendar_db_path)
        self._ensure_calendar_file()
    
    def _ensure_calendar_file(self) -> None:
        """Create calendar file if it doesn't exist."""
        if not self.calendar_path.exists():
            self.calendar_path.parent.mkdir(parents=True, exist_ok=True)
            self.calendar_path.write_text("[]")
            logger.info(f"Created calendar file at {self.calendar_path}")
    
    def _load_events(self) -> List[Dict]:
        """Load all events from calendar file."""
        try:
            with open(self.calendar_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Calendar file corrupted, resetting")
            return []
        except Exception as e:
            logger.error(f"Error loading events: {e}")
            return []
    
    def _save_events(self, events: List[Dict]) -> None:
        """Save events to calendar file."""
        try:
            with open(self.calendar_path, 'w') as f:
                json.dump(events, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving events: {e}")
            raise
    
    def add_event(
        self, 
        title: str, 
        date: str, 
        time: str,
        description: str = ""
    ) -> Dict:
        """
        Add a new event to the calendar.
        
        Args:
            title: Event title
            date: Date in YYYY-MM-DD format
            time: Time in HH:MM format (24-hour)
            description: Optional event description
            
        Returns:
            The created event dict
        """
        try:
            # Create event object
            event = {
                "id": datetime.now().isoformat(),
                "title": title,
                "date": date,
                "time": time,
                "description": description,
                "created_at": datetime.now().isoformat()
            }
            
            # Load existing events
            events = self._load_events()
            
            # Add new event
            events.append(event)
            
            # Save
            self._save_events(events)
            
            logger.info(f"✅ Added event: {title} on {date} at {time}")
            return event
            
        except Exception as e:
            logger.error(f"Error adding event: {e}")
            raise
    
    def get_events(self, date: Optional[str] = None) -> List[Dict]:
        """
        Get all events, optionally filtered by date.
        
        Args:
            date: Optional date filter in YYYY-MM-DD format
            
        Returns:
            List of events
        """
        events = self._load_events()
        
        if date:
            events = [e for e in events if e.get("date") == date]
        
        return events
    
    def delete_event(self, event_id: str) -> bool:
        """
        Delete an event by ID.
        
        Args:
            event_id: The event ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        events = self._load_events()
        initial_count = len(events)
        
        events = [e for e in events if e.get("id") != event_id]
        
        if len(events) < initial_count:
            self._save_events(events)
            logger.info(f"✅ Deleted event: {event_id}")
            return True
        
        return False
    
    def clear_all(self) -> int:
        """
        Clear all events.
        
        Returns:
            Number of events cleared
        """
        events = self._load_events()
        count = len(events)
        self._save_events([])
        logger.info(f"🗑️ Cleared {count} events")
        return count


# Global calendar tool instance
calendar_tool = CalendarTool()


# MCP Tool Functions
def add_event(title: str, date: str, time: str, description: str = "") -> str:
    """
    Schedule a new calendar event.
    
    Use this tool when the user wants to:
    - Schedule a meeting or appointment
    - Create a reminder
    - Add an event to their calendar
    
    Args:
        title: Name/title of the event (e.g., "Team Meeting")
        date: Date in YYYY-MM-DD format (e.g., "2025-11-02")
        time: Time in HH:MM 24-hour format (e.g., "15:00" for 3pm)
        description: Optional additional details about the event
        
    Returns:
        Confirmation message with event details
        
    Example:
        add_event("Lunch with Alice", "2025-11-02", "12:00", "At downtown cafe")
    """
    try:
        event = calendar_tool.add_event(title, date, time, description)
        
        return f"""✅ Event scheduled successfully!

Title: {title}
Date: {date}
Time: {time}
{f'Description: {description}' if description else ''}

Event ID: {event['id']}"""
        
    except Exception as e:
        logger.error(f"Error in add_event tool: {e}")
        return f"❌ Failed to schedule event: {str(e)}"


def get_events(date: Optional[str] = None) -> str:
    """
    Retrieve calendar events.
    
    Use this when the user wants to:
    - See their schedule
    - Check what events are coming up
    - View events for a specific date
    
    Args:
        date: Optional date filter in YYYY-MM-DD format. If not provided, returns all events.
        
    Returns:
        Formatted list of events
        
    Example:
        get_events("2025-11-02") → Shows events for Nov 2, 2025
        get_events() → Shows all events
    """
    try:
        events = calendar_tool.get_events(date)
        
        if not events:
            if date:
                return f"No events scheduled for {date}."
            return "No events in calendar."
        
        # Format events
        formatted = []
        for event in sorted(events, key=lambda x: (x.get('date', ''), x.get('time', ''))):
            formatted.append(
                f"• {event['title']} - {event['date']} at {event['time']}"
                + (f"\n  {event['description']}" if event.get('description') else "")
            )
        
        header = f"Events for {date}:" if date else "All events:"
        return f"{header}\n\n" + "\n".join(formatted)
        
    except Exception as e:
        logger.error(f"Error in get_events tool: {e}")
        return f"❌ Error retrieving events: {str(e)}"


def delete_event(event_id: str) -> str:
    """
    Delete a calendar event.
    
    Args:
        event_id: The ID of the event to delete
        
    Returns:
        Confirmation message
    """
    try:
        success = calendar_tool.delete_event(event_id)
        
        if success:
            return f"✅ Event {event_id} deleted successfully"
        else:
            return f"❌ Event {event_id} not found"
            
    except Exception as e:
        return f"❌ Error deleting event: {str(e)}"
