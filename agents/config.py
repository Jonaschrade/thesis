"""
Global configuration
"""
import os

# Models
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
LLM_MODEL = "qwen2.5:14b" # swap, depending on RAM
EMBED_MODEL="nomic-embed-text" # model for text embeddings into vectors

# Memory
REFLECT_EVERY = 2 # trigger reflection after ever N rounds
MAX_MEMORIES_SEED = 15 # how many recent memories to feed into reflection
MAX_MEMORIES_RETRIEVE = 5 # how many memories surface per response

MEMORY_PERSIST=False # Whether memory storage should saved to disc or not
MEMORY_DIR="./memory.db" # persistent memory ChromaDB

# Retrieval scoring weights
WEIGHT_RECENCY = 0.3
WEIGHT_IMPORTANCE = 0.3
WEIGHT_RELEVANCE = 0.4

# Conversation
DEFAULT_MAX_ROUNDS = 12 # number of interaction rounds before conversation ends

# Social Feedback
EVAL_EVERY=4 # trigger social feedback every N rounds