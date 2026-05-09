"""Unit tests for MemorySystem component."""

import os
import tempfile
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from src.memory_system import MemorySystem
from src.models import Memory


class TestMemorySystem:
    """Test suite for MemorySystem CRUD operations."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage file for testing."""
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.remove(path)
    
    @pytest.fixture
    def memory_system(self, temp_storage):
        """Create a MemorySystem instance with temporary storage."""
        return MemorySystem(storage_path=temp_storage)
    
    def test_store_memory_basic(self, memory_system):
        """Test storing a basic memory returns a valid ID."""
        memory_id = memory_system.store_memory("User likes pizza")
        
        assert memory_id is not None
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0
    
    def test_store_memory_with_tags(self, memory_system):
        """Test storing a memory with tags."""
        memory_id = memory_system.store_memory(
            "User prefers morning meetings",
            tags=["preference", "schedule"]
        )
        
        assert memory_id is not None
        memories = memory_system.retrieve_memories()
        assert len(memories) == 1
        assert memories[0].tags == ["preference", "schedule"]
    
    def test_store_memory_with_context(self, memory_system):
        """Test storing a memory with context metadata."""
        context = {"session_id": "abc123", "user_id": "user456"}
        memory_id = memory_system.store_memory(
            "User mentioned vacation plans",
            context=context
        )
        
        assert memory_id is not None
        memories = memory_system.retrieve_memories()
        assert len(memories) == 1
        assert memories[0].context == context
    
    def test_store_memory_empty_content_raises_error(self, memory_system):
        """Test that storing empty content raises ValueError."""
        with pytest.raises(ValueError, match="Memory content cannot be empty"):
            memory_system.store_memory("")
        
        with pytest.raises(ValueError, match="Memory content cannot be empty"):
            memory_system.store_memory("   ")
    
    def test_retrieve_memories_empty(self, memory_system):
        """Test retrieving memories when none exist."""
        memories = memory_system.retrieve_memories()
        assert memories == []
    
    def test_retrieve_memories_ordered_by_recency(self, memory_system):
        """Test that memories are retrieved in recency order (newest first)."""
        id1 = memory_system.store_memory("First memory")
        id2 = memory_system.store_memory("Second memory")
        id3 = memory_system.store_memory("Third memory")
        
        memories = memory_system.retrieve_memories()
        assert len(memories) == 3
        assert memories[0].content == "Third memory"
        assert memories[1].content == "Second memory"
        assert memories[2].content == "First memory"
    
    def test_retrieve_memories_with_limit(self, memory_system):
        """Test retrieving a limited number of memories."""
        memory_system.store_memory("Memory 1")
        memory_system.store_memory("Memory 2")
        memory_system.store_memory("Memory 3")
        memory_system.store_memory("Memory 4")
        
        memories = memory_system.retrieve_memories(limit=2)
        assert len(memories) == 2
        assert memories[0].content == "Memory 4"
        assert memories[1].content == "Memory 3"
    
    def test_search_memories_by_tags(self, memory_system):
        """Test searching memories by tags."""
        memory_system.store_memory("User likes coffee", tags=["preference", "food"])
        memory_system.store_memory("Meeting at 9am", tags=["schedule"])
        memory_system.store_memory("User dislikes tea", tags=["preference", "food"])
        
        # Search for preference tag
        results = memory_system.search_memories(tags=["preference"])
        assert len(results) == 2
        assert all("preference" in m.tags for m in results)
        
        # Search for schedule tag
        results = memory_system.search_memories(tags=["schedule"])
        assert len(results) == 1
        assert results[0].content == "Meeting at 9am"
    
    def test_search_memories_by_multiple_tags(self, memory_system):
        """Test searching memories with multiple tags (OR logic)."""
        memory_system.store_memory("Coffee preference", tags=["preference"])
        memory_system.store_memory("Morning meeting", tags=["schedule"])
        memory_system.store_memory("Lunch plans", tags=["food"])
        
        # Search for memories with either preference OR schedule tag
        results = memory_system.search_memories(tags=["preference", "schedule"])
        assert len(results) == 2
    
    def test_search_memories_by_content(self, memory_system):
        """Test searching memories by content query."""
        memory_system.store_memory("User likes pizza")
        memory_system.store_memory("User prefers morning meetings")
        memory_system.store_memory("Pizza is favorite food")
        
        results = memory_system.search_memories(content_query="pizza")
        assert len(results) == 2
        assert all("pizza" in m.content.lower() for m in results)
    
    def test_search_memories_by_tags_and_content(self, memory_system):
        """Test searching memories with both tag and content filters."""
        memory_system.store_memory("User likes coffee", tags=["preference"])
        memory_system.store_memory("Coffee meeting at 9am", tags=["schedule"])
        memory_system.store_memory("User dislikes tea", tags=["preference"])
        
        results = memory_system.search_memories(
            tags=["preference"],
            content_query="coffee"
        )
        assert len(results) == 1
        assert results[0].content == "User likes coffee"
    
    def test_search_memories_no_filters_returns_all(self, memory_system):
        """Test that search with no filters returns all memories."""
        memory_system.store_memory("Memory 1")
        memory_system.store_memory("Memory 2")
        memory_system.store_memory("Memory 3")
        
        results = memory_system.search_memories()
        assert len(results) == 3
    
    def test_persistence_across_instances(self, temp_storage):
        """Test that memories persist across MemorySystem instances."""
        # Create first instance and store memories
        ms1 = MemorySystem(storage_path=temp_storage)
        id1 = ms1.store_memory("Persistent memory", tags=["test"])
        
        # Create second instance and verify memory is loaded
        ms2 = MemorySystem(storage_path=temp_storage)
        memories = ms2.retrieve_memories()
        
        assert len(memories) == 1
        assert memories[0].content == "Persistent memory"
        assert memories[0].tags == ["test"]
        assert memories[0].id == id1


class TestMemorySystemSemanticSearch:
    """Test suite for semantic search functionality."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage file for testing."""
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.remove(path)
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mock SentenceTransformer."""
        mock_st = MagicMock()
        mock_st.encode.return_value = np.array([0.1, 0.2, 0.3])
        return mock_st
    
    def test_store_memory_generates_embedding(self, temp_storage, mock_sentence_transformer):
        """Test that storing a memory generates an embedding."""
        with patch.dict('sys.modules', {'sentence_transformers': MagicMock(SentenceTransformer=MagicMock(return_value=mock_sentence_transformer))}):
            memory_system = MemorySystem(storage_path=temp_storage)
            memory_id = memory_system.store_memory("Test content")
            
            # Verify embedding was generated
            mock_sentence_transformer.encode.assert_called_once_with("Test content")
            
            # Verify memory has embedding
            memories = memory_system.retrieve_memories()
            assert len(memories) == 1
            assert memories[0].embedding == [0.1, 0.2, 0.3]
    
    def test_retrieve_memories_with_query_uses_semantic_search(self, temp_storage, mock_sentence_transformer):
        """Test that retrieve_memories with query uses semantic search."""
        # Setup mock responses for storing memories and query
        mock_sentence_transformer.encode.side_effect = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.15, 0.25, 0.35])
        ]
        
        with patch.dict('sys.modules', {'sentence_transformers': MagicMock(SentenceTransformer=MagicMock(return_value=mock_sentence_transformer))}):
            memory_system = MemorySystem(storage_path=temp_storage)
            memory_system.store_memory("Pizza is delicious")
            
            # Retrieve with query
            results = memory_system.retrieve_memories(query="food preferences")
            
            # Verify encode was called twice (once for store, once for query)
            assert mock_sentence_transformer.encode.call_count == 2
            assert len(results) == 1
    
    def test_retrieve_memories_without_query_uses_recency(self, temp_storage, mock_sentence_transformer):
        """Test that retrieve_memories without query uses recency ordering."""
        with patch.dict('sys.modules', {'sentence_transformers': MagicMock(SentenceTransformer=MagicMock(return_value=mock_sentence_transformer))}):
            memory_system = MemorySystem(storage_path=temp_storage)
            memory_system.store_memory("First memory")
            memory_system.store_memory("Second memory")
            
            # Retrieve without query
            results = memory_system.retrieve_memories()
            
            # Should be ordered by recency (newest first)
            assert len(results) == 2
            assert results[0].content == "Second memory"
            assert results[1].content == "First memory"
    
    def test_cosine_similarity_calculation(self, temp_storage):
        """Test cosine similarity calculation."""
        memory_system = MemorySystem(storage_path=temp_storage)
        
        # Test identical vectors (similarity = 1.0)
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = memory_system._cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001
        
        # Test orthogonal vectors (similarity = 0.0)
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = memory_system._cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 0.001
        
        # Test opposite vectors (similarity = -1.0)
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = memory_system._cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 0.001
    
    def test_cosine_similarity_edge_cases(self, temp_storage):
        """Test cosine similarity with edge cases."""
        memory_system = MemorySystem(storage_path=temp_storage)
        
        # Test empty vectors
        assert memory_system._cosine_similarity([], []) == 0.0
        
        # Test mismatched lengths
        assert memory_system._cosine_similarity([1.0, 2.0], [1.0]) == 0.0
        
        # Test zero vectors
        assert memory_system._cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0
    
    def test_semantic_search_orders_by_relevance(self, temp_storage, mock_sentence_transformer):
        """Test that semantic search orders results by relevance."""
        # Setup mock responses
        embeddings = [
            np.array([1.0, 0.0, 0.0]),  # "I love pizza"
            np.array([0.0, 1.0, 0.0]),  # "Meeting at 9am"
            np.array([0.9, 0.1, 0.0]),  # "Pizza is great"
        ]
        
        # Mock response for query (similar to first and third memories)
        query_embedding = np.array([0.95, 0.05, 0.0])
        
        mock_sentence_transformer.encode.side_effect = embeddings + [query_embedding]
        
        with patch.dict('sys.modules', {'sentence_transformers': MagicMock(SentenceTransformer=MagicMock(return_value=mock_sentence_transformer))}):
            memory_system = MemorySystem(storage_path=temp_storage)
            memory_system.store_memory("I love pizza")
            memory_system.store_memory("Meeting at 9am")
            memory_system.store_memory("Pizza is great")
            
            # Search with query
            results = memory_system.retrieve_memories(query="pizza preferences")
            
            # Results should be ordered by similarity to query
            assert len(results) == 3
            # First result should be most similar (pizza-related)
            assert "pizza" in results[0].content.lower()
            assert "pizza" in results[1].content.lower()
            # Last result should be least similar (meeting)
            assert "meeting" in results[2].content.lower()
    
    def test_semantic_search_with_limit(self, temp_storage, mock_sentence_transformer):
        """Test that semantic search respects limit parameter."""
        # Setup mock responses
        embeddings = [np.array([1.0, 0.0]), np.array([0.9, 0.1]), np.array([0.8, 0.2])]
        query_embedding = np.array([1.0, 0.0])
        
        mock_sentence_transformer.encode.side_effect = embeddings + [query_embedding]
        
        with patch.dict('sys.modules', {'sentence_transformers': MagicMock(SentenceTransformer=MagicMock(return_value=mock_sentence_transformer))}):
            memory_system = MemorySystem(storage_path=temp_storage)
            memory_system.store_memory("Memory 1")
            memory_system.store_memory("Memory 2")
            memory_system.store_memory("Memory 3")
            
            # Search with limit
            results = memory_system.retrieve_memories(query="test", limit=2)
            
            assert len(results) == 2
    
    def test_embedding_generation_failure_graceful_handling(self, temp_storage, mock_sentence_transformer):
        """Test that embedding generation failure doesn't crash the system."""
        # Mock embedding generation to fail
        mock_sentence_transformer.encode.side_effect = Exception("API Error")
        
        with patch.dict('sys.modules', {'sentence_transformers': MagicMock(SentenceTransformer=MagicMock(return_value=mock_sentence_transformer))}):
            memory_system = MemorySystem(storage_path=temp_storage)
            
            # Should not raise exception, just store without embedding
            memory_id = memory_system.store_memory("Test content")
            
            memories = memory_system.retrieve_memories()
            assert len(memories) == 1
            assert memories[0].embedding is None
    
    def test_semantic_search_fallback_on_query_embedding_failure(self, temp_storage, mock_sentence_transformer):
        """Test that semantic search falls back to recency when query embedding fails."""
        # Query embedding fails
        mock_sentence_transformer.encode.side_effect = [
            np.array([0.1, 0.2, 0.3]),
            Exception("API Error")
        ]
        
        with patch.dict('sys.modules', {'sentence_transformers': MagicMock(SentenceTransformer=MagicMock(return_value=mock_sentence_transformer))}):
            memory_system = MemorySystem(storage_path=temp_storage)
            memory_system.store_memory("Test memory")
            
            # Should fall back to recency ordering
            results = memory_system.retrieve_memories(query="test query")
            
            assert len(results) == 1
            assert results[0].content == "Test memory"
    
    def test_persistence_with_embeddings(self, temp_storage, mock_sentence_transformer):
        """Test that embeddings are persisted and loaded correctly."""
        with patch.dict('sys.modules', {'sentence_transformers': MagicMock(SentenceTransformer=MagicMock(return_value=mock_sentence_transformer))}):
            # Create first instance and store memory with embedding
            ms1 = MemorySystem(storage_path=temp_storage)
            memory_id = ms1.store_memory("Test with embedding")
            
            # Create second instance and verify embedding is loaded
            ms2 = MemorySystem(storage_path=temp_storage)
            memories = ms2.retrieve_memories()
            
            assert len(memories) == 1
            assert memories[0].embedding == [0.1, 0.2, 0.3]
