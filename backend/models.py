"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Literal, Any, TypedDict, Optional, Dict
from datetime import datetime
from dataclasses import dataclass


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


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""
    name: str
    url: str
    description: str


class SubTask(TypedDict):
    """Single subtask in the execution plan"""
    step: int
    tool: str
    params: Dict[str, Any]  # Minimal params + placeholders
    depends_on: Optional[List[int]]  # Which steps must complete first
    status: str  # "pending" | "executing" | "completed" | "failed"
    result: Optional[Any]  # Tool execution result
    error: Optional[str]  # Error message if failed


class JarvisState(TypedDict):
    """State flowing through the LangGraph"""
    # Input
    user_message: str
    chat_history: List[tuple]  # Previous conversation context
    
    # MCP and tool context
    available_tools: List[str]  # Tool names available
    mcp_servers: List[MCPServerConfig]  # MCP server configs
    
    # Planner outputs
    subtasks: List[SubTask]  # Decomposed plan
    execution_order: List[int]  # Sequential order of step indices
    
    # Executor tracking
    current_step_index: int  # Which step in execution_order we're on
    step_results: Dict[int, Any]  # step_id → result mapping
    
    # Final output
    final_response: Optional[str]
    error: Optional[str]
