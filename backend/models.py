"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Literal
from datetime import datetime


class Message(BaseModel):
    """Single chat message."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ConversationHistory(BaseModel):
    """In-memory conversation storage."""
    messages: List[Message] = Field(default_factory=list)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to history."""
        self.messages.append(Message(role=role, content=content))
    
    def get_openai_format(self) -> List[dict]:
        """Convert to OpenAI API format."""
        return [
            {"role": msg.role, "content": msg.content} 
            for msg in self.messages
        ]
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
