"""Data models for the voice AI agent system."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class ToDoItem:
    """Represents a to-do item with description, status, and timestamps.
    
    Attributes:
        id: Unique identifier (UUID)
        description: Task description
        status: One of "pending", "completed", "cancelled"
        created_at: Creation timestamp
        updated_at: Last modification timestamp
    """
    id: str
    description: str
    status: str
    created_at: datetime
    updated_at: datetime
    
    VALID_STATUSES = {"pending", "completed", "cancelled"}
    
    def __post_init__(self):
        """Validate status after initialization."""
        if self.status not in self.VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{self.status}'. "
                f"Must be one of: {', '.join(sorted(self.VALID_STATUSES))}"
            )
    
    def to_dict(self) -> Dict:
        """Serializes to dictionary for storage.
        
        Returns:
            Dictionary representation with ISO format timestamps
        """
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ToDoItem':
        """Deserializes from dictionary.
        
        Args:
            data: Dictionary with to-do item data
            
        Returns:
            ToDoItem instance
            
        Raises:
            ValueError: If status is invalid or required fields are missing
        """
        return cls(
            id=data["id"],
            description=data["description"],
            status=data["status"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )


@dataclass
class Memory:
    """Represents a memory with content, tags, context, and embedding.
    
    Attributes:
        id: Unique identifier (UUID)
        content: Memory content
        tags: Associated tags for categorization
        context: Additional context metadata
        timestamp: When memory was created
        embedding: Vector embedding for semantic search (optional)
    """
    id: str
    content: str
    tags: List[str]
    context: Dict
    timestamp: datetime
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate content after initialization."""
        if not self.content or not self.content.strip():
            raise ValueError("Memory content cannot be empty")
    
    def to_dict(self) -> Dict:
        """Serializes to dictionary for storage.
        
        Returns:
            Dictionary representation with ISO format timestamp
        """
        return {
            "id": self.id,
            "content": self.content,
            "tags": self.tags,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Memory':
        """Deserializes from dictionary.
        
        Args:
            data: Dictionary with memory data
            
        Returns:
            Memory instance
            
        Raises:
            ValueError: If content is empty or required fields are missing
        """
        return cls(
            id=data["id"],
            content=data["content"],
            tags=data["tags"],
            context=data["context"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            embedding=data.get("embedding")
        )


@dataclass
class ToolCall:
    """Represents a tool invocation with name and parameters.
    
    Attributes:
        tool_name: Name of the tool to invoke
        parameters: Dictionary of parameters for the tool
    """
    tool_name: str
    parameters: Dict


@dataclass
class ToolResult:
    """Represents the result of a tool execution.
    
    Attributes:
        tool_name: Name of the tool that was executed
        success: Whether the tool execution succeeded
        result: The result data from the tool (if successful)
        error: Error message (if failed)
    """
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None


@dataclass
class AgentResponse:
    """Represents the agent's response to user input.
    
    Attributes:
        text: Response text for TTS
        tool_calls: List of tools that were invoked
        success: Whether the request was successful
        error: Error message if the request failed
    """
    text: str
    tool_calls: List[ToolCall]
    success: bool
    error: Optional[str] = None
