"""Unit tests for AgentCore component."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.agent_core import AgentCore
from src.todo_manager import ToDoManager
from src.memory_system import MemorySystem
from src.models import ToDoItem, Memory, AgentResponse, ToolCall, ToolResult
from datetime import datetime


class TestAgentCore:
    """Unit tests for AgentCore orchestration and tool calling."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        return Mock()
    
    @pytest.fixture
    def mock_todo_manager(self):
        """Create a mock ToDoManager."""
        return Mock(spec=ToDoManager)
    
    @pytest.fixture
    def mock_memory_system(self):
        """Create a mock MemorySystem."""
        return Mock(spec=MemorySystem)
    
    @pytest.fixture
    def agent(self, mock_openai_client, mock_todo_manager, mock_memory_system):
        """Create an AgentCore instance with mocked dependencies."""
        return AgentCore(
            openai_client=mock_openai_client,
            todo_manager=mock_todo_manager,
            memory_system=mock_memory_system
        )
    
    def test_initialization_with_default_prompt(self, mock_openai_client, mock_todo_manager, mock_memory_system):
        """Test that AgentCore initializes with default system prompt."""
        agent = AgentCore(
            openai_client=mock_openai_client,
            todo_manager=mock_todo_manager,
            memory_system=mock_memory_system
        )
        
        assert agent._system_prompt is not None
        assert len(agent._system_prompt) > 0
        assert "to-do" in agent._system_prompt.lower() or "todo" in agent._system_prompt.lower()
        assert "memory" in agent._system_prompt.lower() or "memories" in agent._system_prompt.lower()
    
    def test_initialization_with_custom_prompt(self, mock_openai_client, mock_todo_manager, mock_memory_system):
        """Test that AgentCore accepts custom system prompt."""
        custom_prompt = "Custom agent prompt"
        agent = AgentCore(
            openai_client=mock_openai_client,
            todo_manager=mock_todo_manager,
            memory_system=mock_memory_system,
            system_prompt=custom_prompt
        )
        
        assert agent._system_prompt == custom_prompt
    
    def test_tool_registry_contains_all_tools(self, agent):
        """Test that tool registry includes all required tools."""
        tool_names = [tool["function"]["name"] for tool in agent._tools]
        
        assert "create_todo" in tool_names
        assert "list_todos" in tool_names
        assert "update_todo" in tool_names
        assert "delete_todo" in tool_names
        assert "store_memory" in tool_names
        assert "recall_memories" in tool_names
    
    def test_execute_tool_create_todo_success(self, agent, mock_todo_manager):
        """Test successful execution of create_todo tool."""
        # Setup mock
        mock_item = ToDoItem(
            id="test-id",
            description="Test task",
            status="pending",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        mock_todo_manager.create_todo.return_value = mock_item
        
        # Execute tool
        result = agent.execute_tool("create_todo", {"description": "Test task"})
        
        assert result.success is True
        assert result.tool_name == "create_todo"
        assert result.result["description"] == "Test task"
        assert result.result["status"] == "pending"
        assert result.error is None
        mock_todo_manager.create_todo.assert_called_once_with("Test task")
    
    def test_execute_tool_create_todo_failure(self, agent, mock_todo_manager):
        """Test create_todo tool with invalid input."""
        # Setup mock to raise ValueError
        mock_todo_manager.create_todo.side_effect = ValueError("Description cannot be empty")
        
        # Execute tool
        result = agent.execute_tool("create_todo", {"description": ""})
        
        assert result.success is False
        assert result.tool_name == "create_todo"
        assert "Invalid input" in result.error
        assert result.result is None
    
    def test_execute_tool_list_todos_success(self, agent, mock_todo_manager):
        """Test successful execution of list_todos tool."""
        # Setup mock
        mock_items = [
            ToDoItem(
                id="id1",
                description="Task 1",
                status="pending",
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            ToDoItem(
                id="id2",
                description="Task 2",
                status="completed",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
        mock_todo_manager.list_todos.return_value = mock_items
        
        # Execute tool
        result = agent.execute_tool("list_todos", {})
        
        assert result.success is True
        assert result.tool_name == "list_todos"
        assert len(result.result) == 2
        assert result.result[0]["description"] == "Task 1"
        assert result.result[1]["description"] == "Task 2"
        assert result.error is None
    
    def test_execute_tool_list_todos_empty(self, agent, mock_todo_manager):
        """Test list_todos tool with no items."""
        # Setup mock
        mock_todo_manager.list_todos.return_value = []
        
        # Execute tool
        result = agent.execute_tool("list_todos", {})
        
        assert result.success is True
        assert result.result == []
    
    def test_execute_tool_update_todo_success(self, agent, mock_todo_manager):
        """Test successful execution of update_todo tool."""
        # Setup mock
        mock_item = ToDoItem(
            id="test-id",
            description="Updated task",
            status="completed",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        mock_todo_manager.update_todo.return_value = mock_item
        
        # Execute tool
        result = agent.execute_tool("update_todo", {
            "todo_id": "test-id",
            "description": "Updated task",
            "status": "completed"
        })
        
        assert result.success is True
        assert result.tool_name == "update_todo"
        assert result.result["description"] == "Updated task"
        assert result.result["status"] == "completed"
        assert result.error is None
    
    def test_execute_tool_update_todo_not_found(self, agent, mock_todo_manager):
        """Test update_todo tool with non-existent item."""
        # Setup mock to raise KeyError
        mock_todo_manager.update_todo.side_effect = KeyError("To-do item not found")
        
        # Execute tool
        result = agent.execute_tool("update_todo", {
            "todo_id": "nonexistent",
            "status": "completed"
        })
        
        assert result.success is False
        assert "Item not found" in result.error
    
    def test_execute_tool_delete_todo_success(self, agent, mock_todo_manager):
        """Test successful execution of delete_todo tool."""
        # Setup mock
        mock_todo_manager.delete_todo.return_value = True
        
        # Execute tool
        result = agent.execute_tool("delete_todo", {"todo_id": "test-id"})
        
        assert result.success is True
        assert result.tool_name == "delete_todo"
        assert result.result["deleted"] is True
        assert result.error is None
        mock_todo_manager.delete_todo.assert_called_once_with("test-id")
    
    def test_execute_tool_delete_todo_not_found(self, agent, mock_todo_manager):
        """Test delete_todo tool with non-existent item."""
        # Setup mock to raise KeyError
        mock_todo_manager.delete_todo.side_effect = KeyError("To-do item not found")
        
        # Execute tool
        result = agent.execute_tool("delete_todo", {"todo_id": "nonexistent"})
        
        assert result.success is False
        assert "Item not found" in result.error
    
    def test_execute_tool_store_memory_success(self, agent, mock_memory_system):
        """Test successful execution of store_memory tool."""
        # Setup mock
        mock_memory_system.store_memory.return_value = "memory-id-123"
        
        # Execute tool
        result = agent.execute_tool("store_memory", {
            "content": "User likes coffee",
            "tags": ["preference"]
        })
        
        assert result.success is True
        assert result.tool_name == "store_memory"
        assert result.result["memory_id"] == "memory-id-123"
        assert result.error is None
        mock_memory_system.store_memory.assert_called_once_with(
            content="User likes coffee",
            tags=["preference"],
            context={}
        )
    
    def test_execute_tool_recall_memories_success(self, agent, mock_memory_system):
        """Test successful execution of recall_memories tool."""
        # Setup mock
        mock_memories = [
            Memory(
                id="mem1",
                content="User likes coffee",
                tags=["preference"],
                context={},
                timestamp=datetime.now()
            ),
            Memory(
                id="mem2",
                content="User works in tech",
                tags=["personal"],
                context={},
                timestamp=datetime.now()
            )
        ]
        mock_memory_system.retrieve_memories.return_value = mock_memories
        
        # Execute tool
        result = agent.execute_tool("recall_memories", {
            "query": "user preferences",
            "limit": 5
        })
        
        assert result.success is True
        assert result.tool_name == "recall_memories"
        assert len(result.result) == 2
        assert result.result[0]["content"] == "User likes coffee"
        assert result.result[1]["content"] == "User works in tech"
        assert result.error is None
    
    def test_execute_tool_unknown_tool(self, agent):
        """Test execution of unknown tool."""
        result = agent.execute_tool("unknown_tool", {})
        
        assert result.success is False
        assert "Unknown tool" in result.error
    
    def test_format_response_fallback_create_todo(self, agent):
        """Test fallback response formatting for create_todo."""
        tool_results = [
            ToolResult(
                tool_name="create_todo",
                success=True,
                result={"description": "Buy milk", "status": "pending"},
                error=None
            )
        ]
        
        response = agent._format_response_fallback(tool_results)
        
        assert "Buy milk" in response
        assert "Created" in response or "created" in response
    
    def test_format_response_fallback_list_todos(self, agent):
        """Test fallback response formatting for list_todos."""
        tool_results = [
            ToolResult(
                tool_name="list_todos",
                success=True,
                result=[
                    {"description": "Task 1", "status": "pending"},
                    {"description": "Task 2", "status": "completed"}
                ],
                error=None
            )
        ]
        
        response = agent._format_response_fallback(tool_results)
        
        assert "Task 1" in response
        assert "Task 2" in response
    
    def test_format_response_fallback_empty_list(self, agent):
        """Test fallback response formatting for empty todo list."""
        tool_results = [
            ToolResult(
                tool_name="list_todos",
                success=True,
                result=[],
                error=None
            )
        ]
        
        response = agent._format_response_fallback(tool_results)
        
        assert "no tasks" in response.lower()
    
    def test_format_response_fallback_error(self, agent):
        """Test fallback response formatting for tool error."""
        tool_results = [
            ToolResult(
                tool_name="delete_todo",
                success=False,
                result=None,
                error="Item not found"
            )
        ]
        
        response = agent._format_response_fallback(tool_results)
        
        assert "Error" in response
        assert "Item not found" in response
    
    def test_process_input_conversational_no_tools(self, agent, mock_openai_client, mock_memory_system):
        """Test processing conversational input without tool calls."""
        # Setup mocks
        mock_memory_system.retrieve_memories.return_value = []
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Hello! How can I help you?"
        mock_response.choices[0].message.tool_calls = None
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Process input
        result = agent.process_input("Hello", session_id="test-session")
        
        assert result.success is True
        assert result.text == "Hello! How can I help you?"
        assert len(result.tool_calls) == 0
        assert result.error is None
    
    def test_process_input_with_memory_retrieval(self, agent, mock_openai_client, mock_memory_system):
        """Test that process_input automatically retrieves relevant memories."""
        # Setup mocks
        mock_memories = [
            Memory(
                id="mem1",
                content="User likes coffee",
                tags=["preference"],
                context={},
                timestamp=datetime.now()
            )
        ]
        mock_memory_system.retrieve_memories.return_value = mock_memories
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "I remember you like coffee!"
        mock_response.choices[0].message.tool_calls = None
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Process input
        result = agent.process_input("What do I like?", session_id="test-session")
        
        # Verify memory retrieval was called
        mock_memory_system.retrieve_memories.assert_called_once()
        assert result.success is True
    
    def test_process_input_maintains_conversation_context(self, agent, mock_openai_client, mock_memory_system):
        """Test that conversation context is maintained across turns."""
        # Setup mocks
        mock_memory_system.retrieve_memories.return_value = []
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = None
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # First turn
        agent.process_input("First message", session_id="test-session")
        
        # Second turn
        agent.process_input("Second message", session_id="test-session")
        
        # Verify context was maintained
        assert "test-session" in agent._conversation_contexts
        context = agent._conversation_contexts["test-session"]
        assert len(context) == 4  # 2 user messages + 2 assistant messages
        assert context[0]["content"] == "First message"
        assert context[2]["content"] == "Second message"
    
    def test_process_input_handles_llm_error(self, agent, mock_openai_client, mock_memory_system):
        """Test error handling when LLM API fails."""
        # Setup mocks
        mock_memory_system.retrieve_memories.return_value = []
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        # Process input
        result = agent.process_input("Hello", session_id="test-session")
        
        assert result.success is False
        assert result.error is not None
        assert "error" in result.text.lower()
    
    def test_build_messages_includes_system_prompt(self, agent):
        """Test that _build_messages includes system prompt."""
        messages = agent._build_messages(
            user_input="Test",
            session_id="test",
            memories=[]
        )
        
        assert messages[0]["role"] == "system"
        assert len(messages[0]["content"]) > 0
    
    def test_build_messages_includes_memories(self, agent):
        """Test that _build_messages includes memory context."""
        mock_memories = [
            Memory(
                id="mem1",
                content="User likes coffee",
                tags=["preference"],
                context={},
                timestamp=datetime.now()
            )
        ]
        
        messages = agent._build_messages(
            user_input="Test",
            session_id="test",
            memories=mock_memories
        )
        
        # Should have system prompt, memory context, and user message
        assert len(messages) >= 3
        # Check if memory content is in messages
        memory_found = any("coffee" in str(msg.get("content", "")) for msg in messages)
        assert memory_found
    
    def test_build_messages_includes_conversation_history(self, agent):
        """Test that _build_messages includes conversation history."""
        # Add some context
        agent._conversation_contexts["test"] = [
            {"role": "user", "content": "Previous message"},
            {"role": "assistant", "content": "Previous response"}
        ]
        
        messages = agent._build_messages(
            user_input="Current message",
            session_id="test",
            memories=[]
        )
        
        # Should include system prompt, history, and current message
        assert len(messages) >= 4
        assert any(msg.get("content") == "Previous message" for msg in messages)
        assert any(msg.get("content") == "Current message" for msg in messages)
