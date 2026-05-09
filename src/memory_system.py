"""Memory System component for storing and retrieving conversation memories."""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import math
from src.models import Memory


class MemorySystem:
    """Manages storage and retrieval of conversation memories.
    
    Provides methods to store memories with tags and context, retrieve memories
    ordered by recency, and search memories by tag filtering.
    All memories are stored in memory and automatically persisted to JSON file.
    Supports semantic search using OpenAI embeddings and cosine similarity.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize the MemorySystem with persistent storage."""
        self._memories: Dict[str, Memory] = {}
        
        # Cloud Persistence Check
        self._mongo_uri = os.environ.get("MONGO_URI")
        self._db = None
        self._collection = None
        
        if self._mongo_uri:
            try:
                from pymongo import MongoClient
                client = MongoClient(self._mongo_uri)
                self._db = client.get_database("aether")
                self._collection = self._db.get_collection("memories")
                print("[AETHER] Connected to Cloud Persistence (MongoDB)")
                self._load_from_mongo()
                return
            except Exception as e:
                print(f"[AETHER] Failed to connect to MongoDB: {e}. Falling back to local storage.")

        # Set storage path
        if storage_path is None:
            storage_path = os.path.expanduser("~/.voice-agent/memories.json")
        self._storage_path = storage_path
        
        # Load existing memories from file
        self._load_from_file()

    def _load_from_mongo(self) -> None:
        """Load memories from MongoDB."""
        self._memories = {}
        for doc in self._collection.find():
            memory = Memory.from_dict(doc)
            self._memories[memory.id] = memory
    
    def store_memory(self, content: str, tags: Optional[List[str]] = None, 
                    context: Optional[Dict] = None) -> str:
        """Stores a memory with metadata and returns the memory ID.
        
        Args:
            content: Memory content (must be non-empty)
            tags: Associated tags for categorization (optional, defaults to empty list)
            context: Additional context metadata (optional, defaults to empty dict)
            
        Returns:
            str: Unique identifier of the stored memory
            
        Raises:
            ValueError: If content is empty or whitespace-only
        """
        if not content or not content.strip():
            raise ValueError("Memory content cannot be empty")
        
        # Set defaults for optional parameters
        if tags is None:
            tags = []
        if context is None:
            context = {}
        
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Generate embedding if enabled
        embedding = None
        # Always try to generate embedding since it's local now
        embedding = self._generate_embedding(content)
        
        memory = Memory(
            id=memory_id,
            content=content,
            tags=tags,
            context=context,
            timestamp=timestamp,
            embedding=embedding
        )
        
        self._memories[memory_id] = memory
        self._save_to_file()
        return memory_id
    
    def retrieve_memories(self, query: Optional[str] = None, limit: Optional[int] = None) -> List[Memory]:
        """Retrieves memories ordered by relevance (if query provided) or recency.
        
        Args:
            query: Optional search query for semantic search. If provided and embeddings are available,
                   memories are ranked by cosine similarity. Otherwise, falls back to recency ordering.
            limit: Maximum number of memories to return (optional)
            
        Returns:
            List[Memory]: List of memories ordered by relevance (if query provided) or timestamp (newest first)
        """
        # If there's a query, use semantic search
        if query:
            return self._semantic_search(query, limit)
        
        # Otherwise, fall back to recency-based retrieval
        # Sort memories by timestamp in descending order (most recent first)
        sorted_memories = sorted(
            self._memories.values(),
            key=lambda m: m.timestamp,
            reverse=True
        )
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            return sorted_memories[:limit]
        
        return sorted_memories
    
    def search_memories(self, tags: Optional[List[str]] = None, 
                       content_query: Optional[str] = None) -> List[Memory]:
        """Searches memories with tag filtering and optional content matching.
        
        Args:
            tags: List of tags to filter by (returns memories with ANY of these tags)
            content_query: Optional string to search for in memory content (case-insensitive)
            
        Returns:
            List[Memory]: List of matching memories ordered by recency (newest first)
        """
        results = []
        
        for memory in self._memories.values():
            # Check tag filter
            if tags is not None and len(tags) > 0:
                # Memory must have at least one of the specified tags
                if not any(tag in memory.tags for tag in tags):
                    continue
            
            # Check content query
            if content_query is not None and content_query.strip():
                # Case-insensitive content search
                if content_query.lower() not in memory.content.lower():
                    continue
            
            results.append(memory)
        
        # Sort by recency (most recent first)
        results.sort(key=lambda m: m.timestamp, reverse=True)
        return results
    
    def clear_all_memories(self) -> bool:
        """Clears all stored memories from memory and file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._memories = {}
            self._save_to_file()
            return True
        except Exception as e:
            print(f"Error clearing memories: {e}")
            return False
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Skip embedding generation for cloud performance."""
        return None

    def _semantic_search(self, query: str, limit: Optional[int] = None) -> List[Memory]:
        """Lightweight keyword-based search for free cloud tier."""
        if not query:
            return self.retrieve_memories(limit=limit)
        
        query_words = set(query.lower().split())
        scored_memories = []
        
        for memory in self._memories.values():
            content_lower = memory.content.lower()
            # Score based on how many query words appear in the memory
            score = sum(1 for word in query_words if word in content_lower)
            if score > 0:
                scored_memories.append((memory, score))
        
        # Sort by score (highest match first), then by recency
        scored_memories.sort(key=lambda x: (x[1], x[0].timestamp), reverse=True)
        
        results = [m for m, s in scored_memories]
        if limit: return results[:limit]
        return results
    
    def _load_from_file(self) -> None:
        """Load memories from JSON file.
        
        Creates the storage directory and file if they don't exist.
        Handles file I/O errors gracefully by starting with empty storage.
        """
        try:
            # Check if file exists
            if not os.path.exists(self._storage_path):
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
                # Start with empty storage
                self._memories = {}
                return
            
            # Read and parse JSON file
            with open(self._storage_path, 'r') as f:
                content = f.read().strip()
                # Handle empty file
                if not content:
                    self._memories = {}
                    return
                data = json.loads(content)
            
            # Deserialize memories
            self._memories = {}
            for memory_dict in data:
                memory = Memory.from_dict(memory_dict)
                self._memories[memory.id] = memory
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Handle corrupted or invalid JSON data
            raise IOError(f"Failed to load memories from {self._storage_path}: {e}")
        except OSError as e:
            # Handle file system errors (permissions, etc.)
            raise IOError(f"Failed to read memories file {self._storage_path}: {e}")
    
    def _sync(self, item: Optional[Memory] = None, clear_all: bool = False) -> None:
        """Synchronize changes to the storage backend (File or MongoDB)."""
        if self._collection:
            try:
                if clear_all:
                    self._collection.delete_many({})
                elif item:
                    self._collection.replace_one({"id": item.id}, item.to_dict(), upsert=True)
                return
            except Exception as e:
                print(f"[AETHER] MongoDB Sync Error: {e}")

        # Fallback to local file
        self._save_to_file()

    def store_memory(self, content: str, tags: Optional[List[str]] = None, context: Optional[Dict] = None) -> str:
        if not content or not content.strip(): raise ValueError("Memory content cannot be empty")
        if tags is None: tags = []
        if context is None: context = {}
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now()
        embedding = self._generate_embedding(content)
        memory = Memory(id=memory_id, content=content, tags=tags, context=context, timestamp=timestamp, embedding=embedding)
        self._memories[memory_id] = memory
        self._sync(item=memory)
        return memory_id

    def clear_all_memories(self) -> bool:
        try:
            self._memories = {}
            self._sync(clear_all=True)
            return True
        except Exception as e:
            print(f"Error clearing memories: {e}")
            return False

    def _save_to_file(self) -> None:
        """Save memories to JSON file."""
        try:
            os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
            data = [memory.to_dict() for memory in self._memories.values()]
            temp_path = self._storage_path + '.tmp'
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(temp_path, self._storage_path)
        except OSError as e:
            raise IOError(f"Failed to save memories to {self._storage_path}: {e}")
