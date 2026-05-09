"""Unit tests for ToDoManager component."""

import pytest
import tempfile
import os
from datetime import datetime
from src.todo_manager import ToDoManager
from src.models import ToDoItem


class TestToDoManager:
    """Unit tests for ToDoManager CRUD operations."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create a temporary file for test storage."""
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.remove(path)
    
    def test_create_todo_with_valid_description(self, temp_storage):
        """Test creating a to-do item with valid description."""
        manager = ToDoManager(storage_path=temp_storage)
        
        item = manager.create_todo("Buy groceries")
        
        assert item.description == "Buy groceries"
        assert item.status == "pending"
        assert item.id is not None
        assert len(item.id) > 0
        assert item.created_at is not None
        assert item.updated_at is not None
        assert item.created_at <= item.updated_at
    
    def test_create_todo_generates_unique_ids(self, temp_storage):
        """Test that multiple to-do items get unique IDs."""
        manager = ToDoManager(storage_path=temp_storage)
        
        item1 = manager.create_todo("Task 1")
        item2 = manager.create_todo("Task 2")
        item3 = manager.create_todo("Task 3")
        
        assert item1.id != item2.id
        assert item1.id != item3.id
        assert item2.id != item3.id
    
    def test_create_todo_with_empty_description_raises_error(self, temp_storage):
        """Test that empty description raises ValueError."""
        manager = ToDoManager(storage_path=temp_storage)
        
        with pytest.raises(ValueError) as exc_info:
            manager.create_todo("")
        
        assert "Description cannot be empty" in str(exc_info.value)
    
    def test_create_todo_with_whitespace_only_description_raises_error(self, temp_storage):
        """Test that whitespace-only description raises ValueError."""
        manager = ToDoManager(storage_path=temp_storage)
        
        with pytest.raises(ValueError) as exc_info:
            manager.create_todo("   ")
        
        assert "Description cannot be empty" in str(exc_info.value)
    
    def test_get_todo_returns_created_item(self, temp_storage):
        """Test retrieving a to-do item by ID."""
        manager = ToDoManager(storage_path=temp_storage)
        
        created_item = manager.create_todo("Test task")
        retrieved_item = manager.get_todo(created_item.id)
        
        assert retrieved_item.id == created_item.id
        assert retrieved_item.description == created_item.description
        assert retrieved_item.status == created_item.status
    
    def test_get_todo_with_nonexistent_id_raises_error(self, temp_storage):
        """Test that retrieving non-existent item raises KeyError."""
        manager = ToDoManager(storage_path=temp_storage)
        
        with pytest.raises(KeyError) as exc_info:
            manager.get_todo("nonexistent-id")
        
        assert "not found" in str(exc_info.value)
        assert "nonexistent-id" in str(exc_info.value)
    
    def test_list_todos_returns_empty_list_initially(self, temp_storage):
        """Test that list_todos returns empty list when no items exist."""
        manager = ToDoManager(storage_path=temp_storage)
        
        todos = manager.list_todos()
        
        assert todos == []
        assert len(todos) == 0
    
    def test_list_todos_returns_all_created_items(self, temp_storage):
        """Test that list_todos returns all created to-do items."""
        manager = ToDoManager(storage_path=temp_storage)
        
        item1 = manager.create_todo("Task 1")
        item2 = manager.create_todo("Task 2")
        item3 = manager.create_todo("Task 3")
        
        todos = manager.list_todos()
        
        assert len(todos) == 3
        assert item1 in todos
        assert item2 in todos
        assert item3 in todos
    
    def test_update_todo_description(self, temp_storage):
        """Test updating only the description of a to-do item."""
        manager = ToDoManager(storage_path=temp_storage)
        
        item = manager.create_todo("Original description")
        original_status = item.status
        original_created_at = item.created_at
        
        updated_item = manager.update_todo(item.id, description="Updated description")
        
        assert updated_item.id == item.id
        assert updated_item.description == "Updated description"
        assert updated_item.status == original_status  # Preserved
        assert updated_item.created_at == original_created_at  # Preserved
        assert updated_item.updated_at > original_created_at  # Changed
    
    def test_update_todo_status(self, temp_storage):
        """Test updating only the status of a to-do item."""
        manager = ToDoManager(storage_path=temp_storage)
        
        item = manager.create_todo("Test task")
        original_description = item.description
        
        updated_item = manager.update_todo(item.id, status="completed")
        
        assert updated_item.id == item.id
        assert updated_item.description == original_description  # Preserved
        assert updated_item.status == "completed"
    
    def test_update_todo_both_fields(self, temp_storage):
        """Test updating both description and status."""
        manager = ToDoManager(storage_path=temp_storage)
        
        item = manager.create_todo("Original task")
        
        updated_item = manager.update_todo(
            item.id, 
            description="Updated task", 
            status="completed"
        )
        
        assert updated_item.description == "Updated task"
        assert updated_item.status == "completed"
    
    def test_update_todo_with_nonexistent_id_raises_error(self, temp_storage):
        """Test that updating non-existent item raises KeyError."""
        manager = ToDoManager(storage_path=temp_storage)
        
        with pytest.raises(KeyError) as exc_info:
            manager.update_todo("nonexistent-id", description="New description")
        
        assert "not found" in str(exc_info.value)
    
    def test_update_todo_with_invalid_status_raises_error(self, temp_storage):
        """Test that updating with invalid status raises ValueError."""
        manager = ToDoManager(storage_path=temp_storage)
        
        item = manager.create_todo("Test task")
        
        with pytest.raises(ValueError) as exc_info:
            manager.update_todo(item.id, status="invalid_status")
        
        assert "Invalid status" in str(exc_info.value)
    
    def test_update_todo_with_empty_description_raises_error(self, temp_storage):
        """Test that updating with empty description raises ValueError."""
        manager = ToDoManager(storage_path=temp_storage)
        
        item = manager.create_todo("Test task")
        
        with pytest.raises(ValueError) as exc_info:
            manager.update_todo(item.id, description="")
        
        assert "Description cannot be empty" in str(exc_info.value)
    
    def test_update_todo_preserves_unspecified_fields(self, temp_storage):
        """Test that update preserves fields not specified in the update."""
        manager = ToDoManager(storage_path=temp_storage)
        
        item = manager.create_todo("Original task")
        original_id = item.id
        original_description = item.description
        original_status = item.status
        original_created_at = item.created_at
        
        # Update with no parameters (only timestamp should change)
        updated_item = manager.update_todo(item.id)
        
        assert updated_item.id == original_id
        assert updated_item.description == original_description
        assert updated_item.status == original_status
        assert updated_item.created_at == original_created_at
        assert updated_item.updated_at > original_created_at
    
    def test_delete_todo_removes_item(self, temp_storage):
        """Test that delete removes the to-do item."""
        manager = ToDoManager(storage_path=temp_storage)
        
        item = manager.create_todo("Task to delete")
        
        result = manager.delete_todo(item.id)
        
        assert result is True
        
        # Verify item is no longer in list
        todos = manager.list_todos()
        assert item not in todos
        assert len(todos) == 0
    
    def test_delete_todo_with_nonexistent_id_raises_error(self, temp_storage):
        """Test that deleting non-existent item raises KeyError."""
        manager = ToDoManager(storage_path=temp_storage)
        
        with pytest.raises(KeyError) as exc_info:
            manager.delete_todo("nonexistent-id")
        
        assert "not found" in str(exc_info.value)
    
    def test_delete_todo_prevents_retrieval(self, temp_storage):
        """Test that deleted item cannot be retrieved."""
        manager = ToDoManager(storage_path=temp_storage)
        
        item = manager.create_todo("Task to delete")
        item_id = item.id
        
        manager.delete_todo(item_id)
        
        with pytest.raises(KeyError):
            manager.get_todo(item_id)
    
    def test_multiple_operations_sequence(self, temp_storage):
        """Test a sequence of CRUD operations."""
        manager = ToDoManager(storage_path=temp_storage)
        
        # Create multiple items
        item1 = manager.create_todo("Task 1")
        item2 = manager.create_todo("Task 2")
        item3 = manager.create_todo("Task 3")
        
        # List should have 3 items
        assert len(manager.list_todos()) == 3
        
        # Update one item
        manager.update_todo(item2.id, status="completed")
        updated = manager.get_todo(item2.id)
        assert updated.status == "completed"
        
        # Delete one item
        manager.delete_todo(item1.id)
        assert len(manager.list_todos()) == 2
        
        # Verify correct items remain
        remaining = manager.list_todos()
        remaining_ids = [item.id for item in remaining]
        assert item1.id not in remaining_ids
        assert item2.id in remaining_ids
        assert item3.id in remaining_ids
    
    def test_persistence_saves_to_file(self, temp_storage):
        """Test that creating a to-do item saves to file."""
        manager = ToDoManager(storage_path=temp_storage)
        
        item = manager.create_todo("Persistent task")
        
        # Verify file exists and contains data
        assert os.path.exists(temp_storage)
        with open(temp_storage, 'r') as f:
            import json
            data = json.load(f)
        
        assert len(data) == 1
        assert data[0]['description'] == "Persistent task"
        assert data[0]['id'] == item.id
    
    def test_persistence_loads_from_file(self, temp_storage):
        """Test that to-do items are loaded from file on initialization."""
        # Create first manager and add items
        manager1 = ToDoManager(storage_path=temp_storage)
        item1 = manager1.create_todo("Task 1")
        item2 = manager1.create_todo("Task 2")
        
        # Create second manager - should load from file
        manager2 = ToDoManager(storage_path=temp_storage)
        
        todos = manager2.list_todos()
        assert len(todos) == 2
        
        # Verify items match
        ids = [item.id for item in todos]
        assert item1.id in ids
        assert item2.id in ids
    
    def test_persistence_updates_file_on_update(self, temp_storage):
        """Test that updating a to-do item updates the file."""
        manager = ToDoManager(storage_path=temp_storage)
        item = manager.create_todo("Original")
        
        manager.update_todo(item.id, description="Updated")
        
        # Load from file and verify
        manager2 = ToDoManager(storage_path=temp_storage)
        loaded_item = manager2.get_todo(item.id)
        assert loaded_item.description == "Updated"
    
    def test_persistence_updates_file_on_delete(self, temp_storage):
        """Test that deleting a to-do item updates the file."""
        manager = ToDoManager(storage_path=temp_storage)
        item1 = manager.create_todo("Task 1")
        item2 = manager.create_todo("Task 2")
        
        manager.delete_todo(item1.id)
        
        # Load from file and verify
        manager2 = ToDoManager(storage_path=temp_storage)
        todos = manager2.list_todos()
        assert len(todos) == 1
        assert todos[0].id == item2.id
    
    def test_persistence_handles_nonexistent_file(self, temp_storage):
        """Test that manager handles nonexistent file gracefully."""
        # Remove the temp file
        if os.path.exists(temp_storage):
            os.remove(temp_storage)
        
        # Should create new manager with empty storage
        manager = ToDoManager(storage_path=temp_storage)
        assert len(manager.list_todos()) == 0
    
    def test_persistence_handles_corrupted_file(self, temp_storage):
        """Test that manager handles corrupted JSON file."""
        # Write invalid JSON to file
        with open(temp_storage, 'w') as f:
            f.write("{ invalid json }")
        
        # Should raise IOError
        with pytest.raises(IOError) as exc_info:
            ToDoManager(storage_path=temp_storage)
        
        assert "Failed to load todos" in str(exc_info.value)
