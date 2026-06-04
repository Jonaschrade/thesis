"""
Global configuration — single source of truth for all tunable parameters.

All constants used across the simulation live here.  No magic numbers should
appear in source files; import from this module instead.

Terminology
-----------
turn         — one respond() call; one agent speaks once
exchange     — one full back-and-forth; both agents speak once (2 turns)
interaction  — one asymmetric event: one expresser, one responder, one exchange,
               one Q-update for the expresser only
round        — INTERACTIONS_PER_ROUND interactions; used as the snapshotting unit
"""

import os

# ── Models ───────────────────────────────────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
LLM_MODEL   = "qwen2.5:32b"       # swap depending on available RAM
EMBED_MODEL = "nomic-embed-text"   # embedding model for vector memory

# ── Memory ───────────────────────────────────────────────────────────────────
REFLECT_EVERY         = 2    # trigger reflection every N simulation rounds (both modes)
MAX_MEMORIES_SEED     = 15   # recent memories fed into the reflection prompt
MAX_MEMORIES_RETRIEVE = 5    # relevant memories retrieved per agent

MEMORY_PERSIST = False         # persist ChromaDB to disk (True) or keep in-memory (False)
MEMORY_DIR     = "./memory.db" # path used when MEMORY_PERSIST is True

# ── Retrieval scoring weights ─────────────────────────────────────────────────
# Must sum to 1.0.  Composite score = recency·w + importance·w + relevance·w
WEIGHT_RECENCY    = 0.3
WEIGHT_IMPORTANCE = 0.3
WEIGHT_RELEVANCE  = 0.4

# ── Social Feedback Theory (SFT) — Q-value learning ──────────────────────────
# Implements the temporal-difference update from Banisch & Olbrich (2019):
#   Q(o_i) ← (1 − α) · Q(o_i) + α · r
LEARNING_RATE = 0.1  # α: step size for the Q-value TD update

# Softmax inverse temperature β for opinion expression.
# p(+1) = exp(β·q_pos) / (exp(β·q_pos) + exp(β·q_neg))
# β = 0  → 50/50 random (fixes the all-+1 initialisation artifact)
# β > 0  → preference for the higher Q-value
# β → ∞  → deterministic argmax
OPINION_BETA = 5.0

# Homophily partner-selection bias h (Jacob & Banisch 2023).
# Weights the responder draw by exp(−h · |Δq_expresser − Δq_neighbour|).
# h = 0  → uniform draw over neighbours (replicates Banisch & Olbrich 2019)
# h > 0  → neighbours with similar conviction are preferred
HOMOPHILY_H = 0.0

# ── Network simulation ────────────────────────────────────────────────────────
NUM_AGENTS         = 4    # total agents in the network graph (pairwise mode always uses 2)
NETWORK_MAX_ROUNDS = 3    # total snapshot rounds

# Interactions per snapshot round — applies to both network and pairwise modes.
# One interaction = one asymmetric event (expresser drawn uniformly, responder
# drawn with homophily bias h in network mode or as the sole other agent in
# pairwise mode, expresser's Q updated).
# Default = NUM_AGENTS gives each agent one expected interaction per round on
# average in network mode (random sequential update convention).
INTERACTIONS_PER_ROUND = NUM_AGENTS

# Graph topology — Stochastic Block Model (SBM).
# The inter-community coupling SBM_P_INTER is the primary experimental variable:
# low values produce stable polarization; high values drive consensus.
# This reproduces the phase transition established in Banisch & Olbrich (2019).
SBM_NUM_COMMUNITIES = 2    # number of opinion communities (blocks)
SBM_P_INTRA         = 0.7  # within-community edge probability
SBM_P_INTER         = 0.1  # between-community edge probability (sweep this for phase transition)

# Graph dynamics — set False to hold the graph fixed (main SFT experiments).
# Set True to enable endogenous tie rewiring via the edge-valuation mechanism
# (extension chapter: homophilic tie formation).
GRAPH_DYNAMIC = False

# Edge dynamics — only active when GRAPH_DYNAMIC = True.
STRENGTH_CAP    = 3.0  # ceiling on each agent's internal edge valuation
STRENGTH_FLOOR  = 0.0  # edge removed when either agent's valuation falls to or below this
STRENGTH_DELTA  = 0.3  # valuation change per interaction = mean_reward × STRENGTH_DELTA
REWARD_WINDOW_M = 5    # rolling window length (interactions) for reward-history edge evaluation

# ── Discussion topic ─────────────────────────────────────────────────────────
# The simulation runs on a single fixed topic for all rounds.  A single topic
# is required for Q-value coherence: the +1/−1 stance dimension must refer to
# the same opinion object throughout, or accumulated conviction has no stable
# semantic interpretation.
TOPIC_LABEL = "Migrationspolitik"
TOPIC_TEXT  = (
    "Deutschland sollte die Zuwanderung von Nicht-EU-Ausländern dauerhaft und stark begrenzen."
)
