"""
MemoryStore - per-agent ChromaDB-backed memory
Responsibilities: 
    - Write memories with embeddings, type, importance, timestamp
    - Retrieve memories scored and weighted by recency + importance + relevance
    - Dump recent raw memories for reflection seed
"""

import time
import chromadb
import uuid

from config import MAX_MEMORIES_RETRIEVE, MEMORY_DIR, MEMORY_PERSIST
from memory.scoring import score_memory

class MemoryStore: 
    def __init__(self, agent_name: str, persist: bool=MEMORY_PERSIST):
        """
        Creates or gets in-memory/persistent ChromaDB database
        """
        if persist:
            client=chromadb.PersistentClient(path=MEMORY_DIR)
        else:
            client=chromadb.EphemeralClient()
        self.col = client.get_or_create_collection(f"mem_{agent_name}") # creates named "collection" of memories for each agent, isolated so memories do not mix
    
    # write ----------------------------------------
    def add(self, content: str, mem_type: str, importance: float, embedding: list,) -> None:
        """
            - Increment memory counter
            - Add memory with document, embedding, meta-data, and id
        """
        self.col.add(documents = [content], 
                     embeddings = [embedding],
                     metadatas = [{
                         "type": mem_type,
                         "importance": importance,
                         "ts": time.time()
                     }],
                     ids = [str(uuid.uuid4())],
                     )
        

    # read ------------------------------------------

    def retrieve(self, query_embedding: list, n: int=MAX_MEMORIES_RETRIEVE,) -> list[dict]:
        """
        Return top-n memories ranked by retrieval score.
        """
        if self.col.count() == 0:
            return[]
        
        results = self.col.query(
            # Queries memory collection using the query embedding, retrieving a max of n memories most similar to query embedding
            query_embeddings = [query_embedding],
            n_results = min(n, self.col.count()), 
            include = ["documents", "metadatas", "distances"]
        )

        scored = [
            # Stores content, type (interaction/reflection), and retrieval score of queried memories into clean list
            {"content": doc,
             "type": meta["type"],
             "score": score_memory(meta, dist),
             } for doc, meta, dist in zip(
                 results["documents"][0],
                 results["metadatas"][0],
                 results["distances"][0],
                 )
        ]
        
        return sorted(scored, key=lambda x: x["score"], reverse=True) # Return queried memories sorted by retrieval score
    

    def all_recent(self, limit: int) -> list[str]:
        """
        Raw dump of most recent "limit" memory strings (for reflection)
        """
        if self.col.count()==0:
            return []
        return self.col.get(limit=limit, include=["documents"])["documents"] # Retrieves limit-most recent memories (similarity does not matter)

        


        
