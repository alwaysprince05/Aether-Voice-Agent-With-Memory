# Semantic Search in Memory System

## Overview

The Memory System now supports semantic search using OpenAI embeddings and cosine similarity. This enhancement allows the system to retrieve memories based on meaning and context rather than just keyword matching or recency.

## Features

### 1. Embedding Generation
- Automatically generates embeddings for memory content when stored
- Uses OpenAI's `text-embedding-3-small` model
- Embeddings are persisted along with memory data

### 2. Cosine Similarity Scoring
- Calculates similarity between query and stored memories
- Returns results ordered by relevance (highest similarity first)
- Handles edge cases (empty vectors, mismatched dimensions, zero vectors)

### 3. Semantic Search
- Query-based retrieval using natural language
- Falls back to recency-based ordering if embeddings unavailable
- Supports limiting number of results

### 4. Graceful Degradation
- Works without OpenAI client (falls back to recency ordering)
- Handles API failures gracefully
- Maintains backward compatibility with existing code

## Usage

### Basic Setup

```python
from openai import OpenAI
from src.memory_system import MemorySystem

# Initialize with OpenAI client for semantic search
openai_client = OpenAI(api_key="your-api-key")
memory_system = MemorySystem(
    storage_path="./data/memories.json",
    openai_client=openai_client
)

# Or without semantic search (recency-based only)
memory_system = MemorySystem(storage_path="./data/memories.json")
```

### Storing Memories

```python
# Store a memory - embedding is generated automatically
memory_id = memory_system.store_memory(
    content="User prefers morning meetings",
    tags=["preference", "schedule"]
)
```

### Semantic Search

```python
# Retrieve memories using semantic search
results = memory_system.retrieve_memories(
    query="When should I schedule meetings?",
    limit=5
)

for memory in results:
    print(f"Content: {memory.content}")
    print(f"Tags: {memory.tags}")
```

### Recency-Based Retrieval

```python
# Retrieve by recency (no query)
recent_memories = memory_system.retrieve_memories(limit=10)
```

## API Changes

### MemorySystem.__init__()

**New Parameter:**
- `openai_client` (optional): OpenAI client instance for embedding generation

```python
def __init__(self, storage_path: Optional[str] = None, openai_client=None)
```

### retrieve_memories()

**Updated Signature:**
- `query` (optional): Search query for semantic search

```python
def retrieve_memories(
    self, 
    query: Optional[str] = None, 
    limit: Optional[int] = None
) -> List[Memory]
```

**Behavior:**
- If `query` is provided and OpenAI client is available: semantic search
- Otherwise: recency-based ordering (newest first)

## Implementation Details

### Embedding Model
- Model: `text-embedding-3-small`
- Dimension: 1536 (default for this model)
- Cost: ~$0.02 per 1M tokens

### Cosine Similarity
- Range: -1.0 (opposite) to 1.0 (identical)
- Formula: `dot(A, B) / (||A|| * ||B||)`
- Handles edge cases: empty vectors, zero vectors, mismatched dimensions

### Performance
- Embedding generation: ~100-200ms per request
- Similarity calculation: O(n*d) where n=memories, d=dimension
- Target retrieval time: <500ms for typical use cases

## Error Handling

### Embedding Generation Failures
- Logs warning but doesn't fail the operation
- Memory is stored without embedding
- Falls back to recency-based retrieval

### API Errors
- Network timeouts: Graceful degradation
- Rate limits: Logged but operation continues
- Invalid API key: Falls back to non-semantic mode

## Testing

### Unit Tests
- Embedding generation with/without client
- Cosine similarity calculations
- Semantic search ordering
- Graceful degradation
- Persistence with embeddings

### Run Tests
```bash
pytest tests/unit/test_memory_system.py::TestMemorySystemSemanticSearch -v
```

## Example

See `examples/semantic_search_demo.py` for a complete demonstration:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Run the demo
python examples/semantic_search_demo.py
```

## Requirements Validation

This implementation satisfies:
- **Requirement 8.1**: Memory retrieval functionality
- **Requirement 8.2**: Ordering by relevance (semantic similarity)
- **Requirement 8.4**: Performance target (<500ms retrieval)

## Future Enhancements

Potential improvements:
1. Support for other embedding models (Cohere, Hugging Face)
2. Hybrid search (combining semantic + keyword + recency)
3. Caching of embeddings to reduce API calls
4. Batch embedding generation for multiple memories
5. Vector database integration (Pinecone, Weaviate, ChromaDB)
