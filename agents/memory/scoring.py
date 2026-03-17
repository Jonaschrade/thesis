"""
Memory retrieval scoring 

Each candidate memory is ranked by weighted combination of: 
    - Recency: exponential decay from time memory was stored
    - Importance: LLM-rated significance (1-10), normalized to [0,1]
    - Relevance: cosine-distance proximity to query embedding 
"""
import math
import time
from config import WEIGHT_RECENCY, WEIGHT_IMPORTANCE, WEIGHT_RELEVANCE

def score_memory(meta: dict, distance: float) -> float: 
    """
    Compute composite retrieval score for single memory.

    Args: 
        meta: ChromaDB metadata dict with key "ts" (float) and "importance" (float 1-10)
        distance: ChromaDB L2 / cosine similarity - lower means more similar
    
    Returns: 
        float score roughly [0,1]: higher = more worth surfacing
    """

    age_hours = (time.time() - meta["ts"])/3600 # gives the age of a memory in hours
    recency = math.exp(-0.5*age_hours) # faster decay within first hours
    importance = meta["importance"] / 10 # normalize importance to [0,1]
    relevance = 1/(1+distance) # invert distance to get similarity

    return (WEIGHT_RECENCY * recency + WEIGHT_IMPORTANCE * importance + WEIGHT_RELEVANCE * relevance) # return the composite retrieval score