"""Unit tests for data models."""

import pytest
from datetime import datetime
from src.models import ToDoItem, Memory, ToolCall, ToolResult, AgentResponse


class TestToDoItem:
    """Unit tests for ToDoItem data model."""
    
    def test_create_valid_todo_item(self):
        """Test creating a valid to-do item with pending status."""
        now = datetime.now()
        item = ToDoItem(
            id="test-id-123",
            description="Buy groceries",
            status="pending",
            created_at=now,
            updated_at=now
        )
        
        assert item.id == "test-id-123"
        assert item.description == "Buy groceries"
        assert item.status == "pending"
        assert item.created_at == now
        assert item.updated_at == now
    
    def test_create_todo_with_completed_status(self):
        """Test creating a to-do item with completed status."""
        now = datetime.now()
        item = ToDoItem(
            id="test-id-456",
            description="Finish report",
            status="completed",
            created_at=now,
            updated_at=now
        )
        
        assert item.status == "completed"
    
    def test_create_todo_with_cancelled_status(self):
        """Test creating a to-do item with cancelled status."""
        now = datetime.now()
        item = ToDoItem(
            id="test-id-789",
            description="Old task",
            status="cancelled",
            created_at=now,
            updated_at=now
        )
        
        assert item.status == "cancelled"
    
    def test_invalid_status_raises_error(self):
        """Test that invalid status raises ValueError."""
        now = datetime.now()
        
        with pytest.raises(ValueError) as exc_info:
            ToDoItem(
                id="test-id",
                description="Test task",
                status="invalid_status",
                created_at=now,
                updated_at=now
            )
        
        assert "Invalid status" in str(exc_info.value)
        assert "invalid_status" in str(exc_info.value)
    
    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        created = datetime(2024, 1, 15, 10, 30, 0)
        updated = datetime(2024, 1, 15, 11, 45, 0)
        
        item = ToDoItem(
            id="uuid-123",
            description="Test task",
            status="pending",
            created_at=created,
            updated_at=updated
        )
        
        result = item.to_dict()
        
        assert result == {
            "id": "uuid-123",
            "description": "Test task",
            "status": "pending",
            "created_at": "2024-01-15T10:30:00",
            "updated_at": "2024-01-15T11:45:00"
        }
    
    def test_from_dict_deserialization(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "uuid-456",
            "description": "Another task",
            "status": "completed",
            "created_at": "2024-01-20T14:00:00",
            "updated_at": "2024-01-20T15:30:00"
        }
        
        item = ToDoItem.from_dict(data)
        
        assert item.id == "uuid-456"
        assert item.description == "Another task"
        assert item.status == "completed"
        assert item.created_at == datetime(2024, 1, 20, 14, 0, 0)
        assert item.updated_at == datetime(2024, 1, 20, 15, 30, 0)
    
    def test_round_trip_serialization(self):
        """Test that to_dict and from_dict are inverse operations."""
        now = datetime.now()
        original = ToDoItem(
            id="round-trip-id",
            description="Round trip test",
            status="pending",
            created_at=now,
            updated_at=now
        )
        
        # Serialize and deserialize
        data = original.to_dict()
        restored = ToDoItem.from_dict(data)
        
        assert restored.id == original.id
        assert restored.description == original.description
        assert restored.status == original.status
        # Compare timestamps (may have microsecond precision differences)
        assert restored.created_at.replace(microsecond=0) == original.created_at.replace(microsecond=0)
        assert restored.updated_at.replace(microsecond=0) == original.updated_at.replace(microsecond=0)
    
    def test_from_dict_with_invalid_status(self):
        """Test that from_dict validates status."""
        data = {
            "id": "test-id",
            "description": "Test",
            "status": "wrong_status",
            "created_at": "2024-01-15T10:00:00",
            "updated_at": "2024-01-15T10:00:00"
        }
        
        with pytest.raises(ValueError) as exc_info:
            ToDoItem.from_dict(data)
        
        assert "Invalid status" in str(exc_info.value)



class TestMemory:
    """Unit tests for Memory data model."""
    
    def test_create_valid_memory(self):
        """Test creating a valid memory with all fields."""
        now = datetime.now()
        memory = Memory(
            id="mem-123",
            content="User prefers morning meetings",
            tags=["preference", "schedule"],
            context={"topic": "meetings"},
            timestamp=now,
            embedding=[0.1, 0.2, 0.3]
        )
        
        assert memory.id == "mem-123"
        assert memory.content == "User prefers morning meetings"
        assert memory.tags == ["preference", "schedule"]
        assert memory.context == {"topic": "meetings"}
        assert memory.timestamp == now
        assert memory.embedding == [0.1, 0.2, 0.3]
    
    def test_create_memory_without_embedding(self):
        """Test creating a memory without embedding (optional field)."""
        now = datetime.now()
        memory = Memory(
            id="mem-456",
            content="User lives in Seattle",
            tags=["location"],
            context={},
            timestamp=now
        )
        
        assert memory.embedding is None
    
    def test_empty_content_raises_error(self):
        """Test that empty content raises ValueError."""
        now = datetime.now()
        
        with pytest.raises(ValueError) as exc_info:
            Memory(
                id="mem-789",
                content="",
                tags=[],
                context={},
                timestamp=now
            )
        
        assert "Memory content cannot be empty" in str(exc_info.value)
    
    def test_whitespace_only_content_raises_error(self):
        """Test that whitespace-only content raises ValueError."""
        now = datetime.now()
        
        with pytest.raises(ValueError) as exc_info:
            Memory(
                id="mem-999",
                content="   ",
                tags=[],
                context={},
                timestamp=now
            )
        
        assert "Memory content cannot be empty" in str(exc_info.value)
    
    def test_to_dict_serialization_with_embedding(self):
        """Test serialization to dictionary with embedding."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        
        memory = Memory(
            id="mem-abc",
            content="Important information",
            tags=["important", "work"],
            context={"project": "alpha"},
            timestamp=timestamp,
            embedding=[0.5, 0.6, 0.7]
        )
        
        result = memory.to_dict()
        
        assert result == {
            "id": "mem-abc",
            "content": "Important information",
            "tags": ["important", "work"],
            "context": {"project": "alpha"},
            "timestamp": "2024-01-15T10:30:00",
            "embedding": [0.5, 0.6, 0.7]
        }
    
    def test_to_dict_serialization_without_embedding(self):
        """Test serialization to dictionary without embedding."""
        timestamp = datetime(2024, 1, 20, 14, 0, 0)
        
        memory = Memory(
            id="mem-def",
            content="Another memory",
            tags=["personal"],
            context={},
            timestamp=timestamp
        )
        
        result = memory.to_dict()
        
        assert result == {
            "id": "mem-def",
            "content": "Another memory",
            "tags": ["personal"],
            "context": {},
            "timestamp": "2024-01-20T14:00:00",
            "embedding": None
        }
    
    def test_from_dict_deserialization_with_embedding(self):
        """Test deserialization from dictionary with embedding."""
        data = {
            "id": "mem-ghi",
            "content": "Test memory",
            "tags": ["test", "example"],
            "context": {"key": "value"},
            "timestamp": "2024-01-25T09:15:00",
            "embedding": [0.1, 0.2, 0.3, 0.4]
        }
        
        memory = Memory.from_dict(data)
        
        assert memory.id == "mem-ghi"
        assert memory.content == "Test memory"
        assert memory.tags == ["test", "example"]
        assert memory.context == {"key": "value"}
        assert memory.timestamp == datetime(2024, 1, 25, 9, 15, 0)
        assert memory.embedding == [0.1, 0.2, 0.3, 0.4]
    
    def test_from_dict_deserialization_without_embedding(self):
        """Test deserialization from dictionary without embedding."""
        data = {
            "id": "mem-jkl",
            "content": "Memory without embedding",
            "tags": [],
            "context": {},
            "timestamp": "2024-02-01T12:00:00"
        }
        
        memory = Memory.from_dict(data)
        
        assert memory.id == "mem-jkl"
        assert memory.content == "Memory without embedding"
        assert memory.embedding is None
    
    def test_round_trip_serialization_with_embedding(self):
        """Test that to_dict and from_dict are inverse operations with embedding."""
        now = datetime.now()
        original = Memory(
            id="round-trip-mem",
            content="Round trip test",
            tags=["test"],
            context={"test": True},
            timestamp=now,
            embedding=[1.0, 2.0, 3.0]
        )
        
        # Serialize and deserialize
        data = original.to_dict()
        restored = Memory.from_dict(data)
        
        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.tags == original.tags
        assert restored.context == original.context
        assert restored.timestamp.replace(microsecond=0) == original.timestamp.replace(microsecond=0)
        assert restored.embedding == original.embedding
    
    def test_round_trip_serialization_without_embedding(self):
        """Test that to_dict and from_dict are inverse operations without embedding."""
        now = datetime.now()
        original = Memory(
            id="round-trip-mem-2",
            content="Another round trip",
            tags=["tag1", "tag2"],
            context={"data": "value"},
            timestamp=now
        )
        
        # Serialize and deserialize
        data = original.to_dict()
        restored = Memory.from_dict(data)
        
        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.tags == original.tags
        assert restored.context == original.context
        assert restored.timestamp.replace(microsecond=0) == original.timestamp.replace(microsecond=0)
        assert restored.embedding is None
    
    def test_from_dict_with_empty_content(self):
        """Test that from_dict validates content."""
        data = {
            "id": "mem-invalid",
            "content": "",
            "tags": [],
            "context": {},
            "timestamp": "2024-01-15T10:00:00"
        }
        
        with pytest.raises(ValueError) as exc_info:
            Memory.from_dict(data)
        
        assert "Memory content cannot be empty" in str(exc_info.value)



class TestToolCall:
    """Unit tests for ToolCall data model."""
    
    def test_create_tool_call(self):
        """Test creating a tool call with name and parameters."""
        tool_call = ToolCall(
            tool_name="create_todo",
            parameters={"description": "Buy milk"}
        )
        
        assert tool_call.tool_name == "create_todo"
        assert tool_call.parameters == {"description": "Buy milk"}
    
    def test_create_tool_call_with_multiple_parameters(self):
        """Test creating a tool call with multiple parameters."""
        tool_call = ToolCall(
            tool_name="update_todo",
            parameters={
                "id": "123",
                "description": "Updated task",
                "status": "completed"
            }
        )
        
        assert tool_call.tool_name == "update_todo"
        assert tool_call.parameters["id"] == "123"
        assert tool_call.parameters["description"] == "Updated task"
        assert tool_call.parameters["status"] == "completed"
    
    def test_create_tool_call_with_empty_parameters(self):
        """Test creating a tool call with empty parameters."""
        tool_call = ToolCall(
            tool_name="list_todos",
            parameters={}
        )
        
        assert tool_call.tool_name == "list_todos"
        assert tool_call.parameters == {}


class TestToolResult:
    """Unit tests for ToolResult data model."""
    
    def test_create_successful_tool_result(self):
        """Test creating a successful tool result."""
        result = ToolResult(
            tool_name="create_todo",
            success=True,
            result={"id": "123", "description": "Buy milk", "status": "pending"}
        )
        
        assert result.tool_name == "create_todo"
        assert result.success is True
        assert result.result == {"id": "123", "description": "Buy milk", "status": "pending"}
        assert result.error is None
    
    def test_create_failed_tool_result(self):
        """Test creating a failed tool result with error message."""
        result = ToolResult(
            tool_name="get_todo",
            success=False,
            result=None,
            error="To-do item not found"
        )
        
        assert result.tool_name == "get_todo"
        assert result.success is False
        assert result.result is None
        assert result.error == "To-do item not found"
    
    def test_create_tool_result_with_list_result(self):
        """Test creating a tool result with list data."""
        result = ToolResult(
            tool_name="list_todos",
            success=True,
            result=[
                {"id": "1", "description": "Task 1"},
                {"id": "2", "description": "Task 2"}
            ]
        )
        
        assert result.tool_name == "list_todos"
        assert result.success is True
        assert len(result.result) == 2
        assert result.error is None


class TestAgentResponse:
    """Unit tests for AgentResponse data model."""
    
    def test_create_successful_agent_response_without_tools(self):
        """Test creating a successful agent response without tool calls."""
        response = AgentResponse(
            text="Hello! How can I help you today?",
            tool_calls=[],
            success=True
        )
        
        assert response.text == "Hello! How can I help you today?"
        assert response.tool_calls == []
        assert response.success is True
        assert response.error is None
    
    def test_create_successful_agent_response_with_tools(self):
        """Test creating a successful agent response with tool calls."""
        tool_call = ToolCall(
            tool_name="create_todo",
            parameters={"description": "Buy groceries"}
        )
        
        response = AgentResponse(
            text="I've added 'Buy groceries' to your to-do list.",
            tool_calls=[tool_call],
            success=True
        )
        
        assert response.text == "I've added 'Buy groceries' to your to-do list."
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool_name == "create_todo"
        assert response.success is True
        assert response.error is None
    
    def test_create_failed_agent_response(self):
        """Test creating a failed agent response with error message."""
        response = AgentResponse(
            text="I'm having trouble processing your request right now.",
            tool_calls=[],
            success=False,
            error="LLM API connection failed"
        )
        
        assert response.text == "I'm having trouble processing your request right now."
        assert response.tool_calls == []
        assert response.success is False
        assert response.error == "LLM API connection failed"
    
    def test_create_agent_response_with_multiple_tool_calls(self):
        """Test creating an agent response with multiple tool calls."""
        tool_call_1 = ToolCall(
            tool_name="store_memory",
            parameters={"content": "User prefers morning meetings", "tags": ["preference"]}
        )
        tool_call_2 = ToolCall(
            tool_name="create_todo",
            parameters={"description": "Schedule morning meeting"}
        )
        
        response = AgentResponse(
            text="I've noted your preference and added a reminder.",
            tool_calls=[tool_call_1, tool_call_2],
            success=True
        )
        
        assert response.text == "I've noted your preference and added a reminder."
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].tool_name == "store_memory"
        assert response.tool_calls[1].tool_name == "create_todo"
        assert response.success is True
