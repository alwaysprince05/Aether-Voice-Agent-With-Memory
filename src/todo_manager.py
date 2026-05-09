"""To-Do Manager component for CRUD operations on to-do items."""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from src.models import ToDoItem


class ToDoManager:
    """Manages CRUD operations for to-do items with persistent storage.
    
    Provides methods to create, retrieve, update, and delete to-do items.
    All items are stored in memory and automatically persisted to JSON file.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize the ToDoManager with persistent storage."""
        self._todos: Dict[str, ToDoItem] = {}
        
        # Cloud Persistence Check
        self._mongo_uri = os.environ.get("MONGO_URI")
        self._db = None
        self._collection = None
        
        if self._mongo_uri:
            try:
                from pymongo import MongoClient
                client = MongoClient(self._mongo_uri)
                self._db = client.get_database("aether")
                self._collection = self._db.get_collection("todos")
                print("[AETHER] Connected to Cloud Persistence (MongoDB)")
                self._load_from_mongo()
                return
            except Exception as e:
                print(f"[AETHER] Failed to connect to MongoDB: {e}. Falling back to local storage.")

        # Set storage path
        if storage_path is None:
            storage_path = os.path.expanduser("~/.voice-agent/todos.json")
        self._storage_path = storage_path
        
        # Load existing todos from file
        self._load_from_file()

    def _load_from_mongo(self) -> None:
        """Load todos from MongoDB."""
        self._todos = {}
        for doc in self._collection.find():
            # Use doc['id'] instead of _id
            item = ToDoItem.from_dict(doc)
            self._todos[item.id] = item
    
    def create_todo(self, description: str) -> ToDoItem:
        """Creates a new to-do item with pending status.
        
        Args:
            description: Task description (must be non-empty)
            
        Returns:
            ToDoItem: The created to-do item with unique ID and timestamps
            
        Raises:
            ValueError: If description is empty or whitespace-only
        """
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")
        
        now = datetime.now()
        todo_id = str(uuid.uuid4())
        
        item = ToDoItem(
            id=todo_id,
            description=description,
            status="pending",
            created_at=now,
            updated_at=now
        )
        
        self._todos[todo_id] = item
        self._save_to_file()
        return item
    
    def get_todo(self, todo_id: str) -> ToDoItem:
        """Retrieves a specific to-do item by ID.
        
        Args:
            todo_id: Unique identifier of the to-do item
            
        Returns:
            ToDoItem: The requested to-do item
            
        Raises:
            KeyError: If the to-do item does not exist
        """
        if todo_id not in self._todos:
            raise KeyError(f"To-do item {todo_id} not found")
        
        return self._todos[todo_id]
    
    def list_todos(self) -> List[ToDoItem]:
        """Returns all to-do items.
        
        Returns:
            List[ToDoItem]: List of all to-do items (empty if none exist)
        """
        return list(self._todos.values())
    
    def update_todo(self, todo_id: str, description: Optional[str] = None, 
                   status: Optional[str] = None) -> ToDoItem:
        """Updates specified fields of a to-do item.
        
        Performs partial updates - only specified fields are modified.
        Unspecified fields are preserved. Updates the updated_at timestamp.
        
        Args:
            todo_id: Unique identifier of the to-do item
            description: New description (optional)
            status: New status (optional, must be valid if provided)
            
        Returns:
            ToDoItem: The updated to-do item
            
        Raises:
            KeyError: If the to-do item does not exist
            ValueError: If status is invalid or description is empty
        """
        if todo_id not in self._todos:
            raise KeyError(f"To-do item {todo_id} not found")
        
        item = self._todos[todo_id]
        
        # Validate and update description if provided
        if description is not None:
            if not description or not description.strip():
                raise ValueError("Description cannot be empty")
            item.description = description
        
        # Validate and update status if provided
        if status is not None:
            if status not in ToDoItem.VALID_STATUSES:
                raise ValueError(
                    f"Invalid status '{status}'. "
                    f"Must be one of: {', '.join(sorted(ToDoItem.VALID_STATUSES))}"
                )
            item.status = status
        
        # Update timestamp
        item.updated_at = datetime.now()
        
        self._save_to_file()
        return item
    
    def delete_todo(self, todo_id: str) -> bool:
        """Deletes a to-do item by ID.
        
        Args:
            todo_id: Unique identifier of the to-do item
            
        Returns:
            bool: True if deletion was successful
            
        Raises:
            KeyError: If the to-do item does not exist
        """
        if todo_id not in self._todos:
            raise KeyError(f"To-do item {todo_id} not found")
        
        del self._todos[todo_id]
        self._save_to_file()
        return True
    
    def _load_from_file(self) -> None:
        """Load to-do items from JSON file.
        
        Creates the storage directory and file if they don't exist.
        Handles file I/O errors gracefully by starting with empty storage.
        """
        try:
            # Check if file exists
            if not os.path.exists(self._storage_path):
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
                # Start with empty storage
                self._todos = {}
                return
            
            # Read and parse JSON file
            with open(self._storage_path, 'r') as f:
                content = f.read().strip()
                # Handle empty file
                if not content:
                    self._todos = {}
                    return
                data = json.loads(content)
            
            # Deserialize to-do items
            self._todos = {}
            for todo_dict in data:
                item = ToDoItem.from_dict(todo_dict)
                self._todos[item.id] = item
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Handle corrupted or invalid JSON data
            raise IOError(f"Failed to load todos from {self._storage_path}: {e}")
        except OSError as e:
            # Handle file system errors (permissions, etc.)
            raise IOError(f"Failed to read todos file {self._storage_path}: {e}")
    
    def _sync(self, item: Optional[ToDoItem] = None, delete_id: Optional[str] = None) -> None:
        """Synchronize changes to the storage backend (File or MongoDB)."""
        if self._collection:
            try:
                if delete_id:
                    self._collection.delete_one({"id": delete_id})
                elif item:
                    self._collection.replace_one({"id": item.id}, item.to_dict(), upsert=True)
                return
            except Exception as e:
                print(f"[AETHER] MongoDB Sync Error: {e}")

        # Fallback to local file
        self._save_to_file()

    def create_todo(self, description: str) -> ToDoItem:
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")
        now = datetime.now()
        todo_id = str(uuid.uuid4())
        item = ToDoItem(id=todo_id, description=description, status="pending", created_at=now, updated_at=now)
        self._todos[todo_id] = item
        self._sync(item=item)
        return item
    
    def update_todo(self, todo_id: str, description: Optional[str] = None, status: Optional[str] = None) -> ToDoItem:
        if todo_id not in self._todos:
            raise KeyError(f"To-do item {todo_id} not found")
        item = self._todos[todo_id]
        if description is not None:
            if not description or not description.strip(): raise ValueError("Description cannot be empty")
            item.description = description
        if status is not None:
            if status not in ToDoItem.VALID_STATUSES:
                raise ValueError(f"Invalid status '{status}'. Must be one of: {', '.join(sorted(ToDoItem.VALID_STATUSES))}")
            item.status = status
        item.updated_at = datetime.now()
        self._sync(item=item)
        return item
    
    def delete_todo(self, todo_id: str) -> bool:
        if todo_id not in self._todos:
            raise KeyError(f"To-do item {todo_id} not found")
        del self._todos[todo_id]
        self._sync(delete_id=todo_id)
        return True

    def _save_to_file(self) -> None:
        """Save to-do items to JSON file."""
        try:
            os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
            data = [item.to_dict() for item in self._todos.values()]
            temp_path = self._storage_path + '.tmp'
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(temp_path, self._storage_path)
        except OSError as e:
            raise IOError(f"Failed to save todos to {self._storage_path}: {e}")
