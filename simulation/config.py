"""
Global configuration — single source of truth for all tunable parameters.

All constants used across the simulation live here.  No magic numbers should
appear in source files; import from this module instead.

Terminology
-----------
turn       — one respond() call; one agent speaks once
exchange   — one full back-and-forth; both agents speak once (2 turns)
discussion — DISCUSSION_TURNS exchanges (DISCUSSION_TURNS × 2 turns total)
round      — one complete discussion between a pair; identical meaning in both modes
"""

import os

# ── Models ───────────────────────────────────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
LLM_MODEL   = "qwen2.5:14b"       # swap depending on available RAM
EMBED_MODEL = "nomic-embed-text"   # embedding model for vector memory

# ── Memory ───────────────────────────────────────────────────────────────────
REFLECT_EVERY         = 2    # trigger reflection every N simulation rounds (both modes)
MAX_MEMORIES_SEED     = 15   # recent memories fed into the reflection prompt
MAX_MEMORIES_RETRIEVE = 5    # memories surfaced per agent response

MEMORY_PERSIST = False         # persist ChromaDB to disk (True) or keep in-memory (False)
MEMORY_DIR     = "./memory.db" # path used when MEMORY_PERSIST is True

# ── Retrieval scoring weights ─────────────────────────────────────────────────
# Must sum to 1.0.  Composite score = recency·w + importance·w + relevance·w
WEIGHT_RECENCY    = 0.3
WEIGHT_IMPORTANCE = 0.3
WEIGHT_RELEVANCE  = 0.4

# ── Pairwise simulation ───────────────────────────────────────────────────────
NUM_AGENTS = 2   # agents to sample and instantiate

# ── Network simulation ────────────────────────────────────────────────────────
NUM_AGENTS_NETWORK = 20     # total agents in the network graph
NETWORK_MAX_ROUNDS = 30     # total simulation rounds (used by both modes)
DISCUSSION_TURNS   = 3      # exchanges per discussion (total turns = DISCUSSION_TURNS × 2)
INITIAL_GRAPH_K    = 4      # Watts-Strogatz k parameter (initial avg degree ≈ k)
INITIAL_GRAPH_P    = 0.3    # Watts-Strogatz p parameter (random rewiring probability)
STRENGTH_CAP       = 3.0    # ceiling on edge strength
STRENGTH_FLOOR     = 0.0    # edge removed when strength falls to or below this value
STRENGTH_DELTA     = 0.3    # strength change per round = combined_score × STRENGTH_DELTA

# Discussion topics distributed across rounds.
# Rounds are divided into equal blocks — one block per topic in insertion order.
# NETWORK_MAX_ROUNDS should be divisible by len(TOPICS); if not, the last topic
# absorbs the remainder rounds.  A single entry reproduces single-topic behaviour.
TOPICS = {
    "Migrationspolitik": (
        "Deutschland nimmt jedes Jahr Hunderttausende Migranten auf – "
        "doch Integration scheitert immer wieder an Sprache, Arbeit und Kultur. "
        "Sollte Deutschland die Grenzen für Nicht-EU-Ausländer dauerhaft schließen?"
    ),
}
