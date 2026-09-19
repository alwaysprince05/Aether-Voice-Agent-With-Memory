"""Memory System component for storing and retrieving conversation memories."""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import math
from src.models import Memory


class MemorySystem:
    """Manages storage and retrieval of conversation memories.

    Provides methods to store memories with tags and context, retrieve memories
    ordered by relevance (if query provided) or recency, and search memories by
    tag filtering. All memories are stored in memory and automatically persisted
    to JSON file.

    If the ``sentence_transformers`` package is available, embeddings are
    generated locally and used for semantic search (cosine similarity). When
    unavailable, the system transparently falls back to lightweight keyword
    scoring so retrieval always works.
    """

    def __init__(self, storage_path: Optional[str] = None, openai_client=None):
        """Initialize the MemorySystem with persistent storage.

        Args:
            storage_path: Path to the JSON storage file (default: ~/.voice-agent/memories.json)
            openai_client: Deprecated/unused. Kept for backward compatibility;
                embeddings are now generated locally via sentence-transformers.
        """
        self._memories: Dict[str, Memory] = {}

        # Lazy sentence-transformers handle — must be initialized in ALL modes
        # (including MongoDB mode) so _generate_embedding never hits AttributeError.
        self._st_model = None

        # Cloud Persistence Check
        self._mongo_uri = os.environ.get("MONGO_URI")
        self._db = None
        self._collection = None

        if self._mongo_uri:
            try:
                from pymongo import MongoClient
                client = MongoClient(self._mongo_uri, serverSelectionTimeoutMS=8000)
                self._db = client.get_database("aether")
                self._collection = self._db.get_collection("memories")
                # find() triggers the real connection — a failure here raises and
                # sends us to the local-file fallback below.
                self._load_from_mongo()
                print("[AETHER] Connected to Cloud Persistence (MongoDB)")
                return
            except Exception as e:
                print(f"[AETHER] Failed to connect to MongoDB: {e}. Falling back to local storage.")
                self._collection = None
                self._db = None

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

    def _get_st_model(self):
        """Return a shared SentenceTransformer instance, or None if unavailable."""
        if self._st_model is not None:
            return self._st_model
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            return self._st_model
        except Exception:
            self._st_model = False  # mark as unavailable; don't retry every call
            return None

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

        if tags is None:
            tags = []
        if context is None:
            context = {}

        memory_id = str(uuid.uuid4())
        timestamp = datetime.now()
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
        self._sync(item=memory)
        return memory_id

    def retrieve_memories(self, query: Optional[str] = None, limit: Optional[int] = None) -> List[Memory]:
        """Retrieves memories ordered by relevance (if query provided) or recency.

        Args:
            query: Optional search query for semantic search. If provided, memories are
                   ranked by semantic similarity (or keyword score as fallback).
                   Otherwise, falls back to recency ordering.
            limit: Maximum number of memories to return (optional)

        Returns:
            List[Memory]: List of memories ordered by relevance (if query provided) or timestamp (newest first)
        """
        if query:
            return self._semantic_search(query, limit)

        # Sort memories by timestamp in descending order (most recent first)
        sorted_memories = sorted(
            self._memories.values(),
            key=lambda m: m.timestamp,
            reverse=True
        )

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
                if not any(tag in memory.tags for tag in tags):
                    continue

            # Check content query
            if content_query is not None and content_query.strip():
                if content_query.lower() not in memory.content.lower():
                    continue

            results.append(memory)

        # Sort by recency (most recent first)
        results.sort(key=lambda m: m.timestamp, reverse=True)
        return results

    def clear_all_memories(self) -> bool:
        """Clears all stored memories from memory and storage backend."""
        try:
            self._memories = {}
            self._sync(clear_all=True)
            return True
        except Exception as e:
            print(f"Error clearing memories: {e}")
            return False

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate a local embedding for the text.

        Returns None when sentence-transformers is unavailable or fails, so
        storage never breaks because of the embedding pipeline.
        """
        try:
            model = self._get_st_model()
            if model is None:
                return None
            vector = model.encode(text)
            return [float(x) for x in vector]
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Returns 0.0 for empty vectors, mismatched lengths, or zero vectors.
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _semantic_search(self, query: str, limit: Optional[int] = None) -> List[Memory]:
        """Rank memories by semantic similarity to the query.

        Uses stored embeddings when available; otherwise falls back to a
        lightweight keyword-overlap score so search still returns useful
        results (e.g. on free cloud tiers without the ML dependency).
        """
        if not query:
            return self.retrieve_memories(limit=limit)

        query_embedding = self._generate_embedding(query)

        if query_embedding:
            scored = [
                (memory, self._cosine_similarity(memory.embedding or [], query_embedding))
                for memory in self._memories.values()
            ]
            scored = [(m, s) for m, s in scored if s > 0]
        else:
            # Keyword fallback
            query_words = set(query.lower().split())
            scored = []
            for memory in self._memories.values():
                content_lower = memory.content.lower()
                score = sum(1 for word in query_words if word in content_lower)
                if score > 0:
                    scored.append((memory, score))

        # Sort by score (highest first), then by recency
        scored.sort(key=lambda x: (x[1], x[0].timestamp), reverse=True)

        results = [m for m, _ in scored]
        if limit is not None and limit > 0:
            return results[:limit]
        return results

    def _sync(self, item: Optional[Memory] = None, clear_all: bool = False) -> None:
        """Synchronize changes to the storage backend (File or MongoDB)."""
        # NOTE: pymongo Collection objects raise NotImplementedError when evaluated
        # for truthiness — always compare with None instead of `if collection:`.
        if self._collection is not None:
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

    def _load_from_file(self) -> None:
        """Load memories from JSON file.

        Creates the storage directory and file if they don't exist.
        Handles file I/O errors gracefully by starting with empty storage.
        """
        try:
            if not os.path.exists(self._storage_path):
                os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
                self._memories = {}
                return

            with open(self._storage_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    self._memories = {}
                    return
                data = json.loads(content)

            self._memories = {}
            for memory_dict in data:
                memory = Memory.from_dict(memory_dict)
                self._memories[memory.id] = memory

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise IOError(f"Failed to load memories from {self._storage_path}: {e}")
        except OSError as e:
            raise IOError(f"Failed to read memories file {self._storage_path}: {e}")

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
