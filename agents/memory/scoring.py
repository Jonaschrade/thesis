"""
Memory retrieval scoring.

Each candidate memory is ranked by a weighted combination of three signals:
    Recency    — exponential decay from the time the memory was stored
    Importance — LLM-rated significance (1–10), normalised to [0, 1]
    Relevance  — inverted cosine distance to the query embedding

Weights are configured in config.py (WEIGHT_RECENCY, WEIGHT_IMPORTANCE,
WEIGHT_RELEVANCE).
"""

import math
import time

from config import WEIGHT_RECENCY, WEIGHT_IMPORTANCE, WEIGHT_RELEVANCE


def score_memory(meta: dict, distance: float) -> float:
    """Compute a composite retrieval score for a single memory candidate.

    Args:
        meta:     ChromaDB metadata dict containing:
                    'ts'         (float) — Unix timestamp of when the memory was stored
                    'importance' (float) — LLM-rated significance on a 1–10 scale
        distance: ChromaDB cosine distance — lower means more similar to the query.

    Returns:
        A float roughly in [0, 1]; higher values indicate the memory is more
        worth surfacing to the agent.
    """
    age_hours  = (time.time() - meta["ts"]) / 3600
    recency    = math.exp(-0.5 * age_hours)   # faster decay in the first hours
    importance = meta["importance"] / 10       # normalise to [0, 1]
    relevance  = 1 / (1 + distance)           # invert distance → similarity

    return (
        WEIGHT_RECENCY    * recency    +
        WEIGHT_IMPORTANCE * importance +
        WEIGHT_RELEVANCE  * relevance
    )
