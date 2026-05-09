"""
Demonstration of semantic search functionality in the Memory System.

This example shows how to use the MemorySystem with OpenAI embeddings
for semantic search capabilities.

Requirements:
- Set OPENAI_API_KEY environment variable
- Install required packages: pip install -r requirements.txt
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from src.memory_system import MemorySystem

# Load environment variables
load_dotenv()

def main():
    """Demonstrate semantic search with the Memory System."""
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set it in your .env file or environment")
        return
    
    openai_client = OpenAI(api_key=api_key)
    
    # Create memory system with semantic search enabled
    print("Initializing Memory System with semantic search...")
    memory_system = MemorySystem(
        storage_path="./data/demo_memories.json",
        openai_client=openai_client
    )
    
    # Store some example memories
    print("\nStoring memories...")
    memories_to_store = [
        ("User loves Italian food, especially pizza and pasta", ["preference", "food"]),
        ("User prefers morning meetings between 9-11am", ["preference", "schedule"]),
        ("User is learning Python and interested in AI/ML", ["interest", "technology"]),
        ("User has a dog named Max who loves playing fetch", ["personal", "pets"]),
        ("User is planning a vacation to Japan in spring", ["personal", "travel"]),
    ]
    
    for content, tags in memories_to_store:
        memory_id = memory_system.store_memory(content, tags=tags)
        print(f"  Stored: {content[:50]}... (ID: {memory_id[:8]}...)")
    
    # Demonstrate semantic search with different queries
    print("\n" + "="*70)
    print("SEMANTIC SEARCH DEMONSTRATIONS")
    print("="*70)
    
    queries = [
        "What kind of food does the user like?",
        "When should I schedule a meeting?",
        "What are the user's hobbies and interests?",
        "Tell me about the user's pets",
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 70)
        
        # Retrieve relevant memories using semantic search
        results = memory_system.retrieve_memories(query=query, limit=2)
        
        if results:
            for i, memory in enumerate(results, 1):
                print(f"  {i}. {memory.content}")
                print(f"     Tags: {', '.join(memory.tags)}")
        else:
            print("  No relevant memories found")
    
    # Compare with recency-based retrieval (no query)
    print("\n" + "="*70)
    print("RECENCY-BASED RETRIEVAL (No Query)")
    print("="*70)
    
    recent_memories = memory_system.retrieve_memories(limit=3)
    for i, memory in enumerate(recent_memories, 1):
        print(f"  {i}. {memory.content}")
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)

if __name__ == "__main__":
    main()
