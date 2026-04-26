"""
MemoryStore: per-agent ChromaDB-backed memory.

Each agent gets its own isolated collection so memories never cross between
agents.  Memories are stored with an embedding, importance score, type tag,
and Unix timestamp, which the scoring function uses to rank retrieval results.
"""

import time
import uuid

import chromadb

from config import MAX_MEMORIES_RETRIEVE, MEMORY_DIR, MEMORY_PERSIST
from memory.scoring import score_memory


class MemoryStore:
    def __init__(self, agent_name: str, persist: bool = MEMORY_PERSIST):
        """Create or reopen a ChromaDB collection for agent_name.

        Args:
            agent_name: Used as the collection name prefix to keep agent
                        memories isolated from each other.
            persist:    If True, data is written to MEMORY_DIR on disk;
                        otherwise an in-memory ephemeral client is used.
        """
        if persist:
            client = chromadb.PersistentClient(path=MEMORY_DIR)
        else:
            client = chromadb.EphemeralClient()
        self.col = client.get_or_create_collection(f"mem_{agent_name}")

    # write ------------------------------------------------------------------

    def add(self, content: str, mem_type: str, importance: float, embedding: list) -> None:
        """Persist a single memory with its metadata.

        Args:
            content:    The memory text (interaction summary or reflection insight).
            mem_type:   'interaction' or 'reflection'.
            importance: LLM-rated significance on a 1–10 scale.
            embedding:  Pre-computed vector embedding of content.
        """
        self.col.add(
            documents  = [content],
            embeddings = [embedding],
            metadatas  = [{"type": mem_type, "importance": importance, "ts": time.time()}],
            ids        = [str(uuid.uuid4())],
        )

    # read -------------------------------------------------------------------

    def retrieve(self, query_embedding: list, n: int = MAX_MEMORIES_RETRIEVE) -> list[dict]:
        """Return the top-n memories most relevant to query_embedding.

        Candidates are fetched by vector similarity, then re-ranked by a
        composite score that also accounts for recency and importance.

        Args:
            query_embedding: Embedding of the current message or question.
            n:               Maximum number of memories to return.

        Returns:
            List of {"content": str, "type": str, "score": float} dicts,
            sorted by descending composite score.
        """
        if self.col.count() == 0:
            return []

        results = self.col.query(
            query_embeddings = [query_embedding],
            n_results        = min(n, self.col.count()),
            include          = ["documents", "metadatas", "distances"],
        )

        scored = [
            {"content": doc, "type": meta["type"], "score": score_memory(meta, dist)}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

        return sorted(scored, key=lambda x: x["score"], reverse=True)

    def all_recent(self, limit: int) -> list[str]:
        """Return the limit most recently added memory strings.

        Used to seed the reflection step, where recency matters more than
        relevance to a specific query.
        """
        if self.col.count() == 0:
            return []
        return self.col.get(limit=limit, include=["documents"])["documents"]
