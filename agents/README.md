# Multi-Agent Conversation Framework

A thesis research project implementing a multi-agent conversational system with persistent memory, reflection, and social evaluation — built on [LangGraph](https://github.com/langchain-ai/langgraph) and [Ollama](https://ollama.ai).

## Overview

Multiple AI agents engage in structured conversations, each maintaining individual memory stores. The design is inspired by Stanford's Generative Agents research (Park et al., 2023): agents reflect on recent interactions to form insights and vote on whether to continue or end a conversation.

## Architecture

```
agents/      Agent class, persona sampler, and persona pool (200 k entries)
graph/       LangGraph graph construction, routing logic, and shared state schema
memory/      ChromaDB-backed memory storage and composite scoring algorithm
config.py    Central configuration (models, weights, thresholds)
main.py      Entry point
```

### Conversation flow

1. Agents take turns responding in a fixed cyclic order.
2. Every `REFLECT_EVERY` rounds all agents simultaneously reflect, synthesising insights from recent memories.
3. Every `EVAL_EVERY` rounds all agents vote `continue` or `move`; any `move` vote ends the conversation.
4. The conversation also ends after `DEFAULT_MAX_ROUNDS` rounds.

### Memory system

Each agent has a per-agent [ChromaDB](https://www.trychroma.com) collection. Memories are ranked by a composite score:

```
score = 0.3 × recency + 0.3 × importance + 0.4 × relevance
```

- **Recency** — exponential decay over time
- **Importance** — LLM-rated significance (1–10), normalised
- **Relevance** — cosine similarity to the current query embedding

## Requirements

- Python 3.10+
- A running [Ollama](https://ollama.ai) server at `127.0.0.1:11434`
- Models pulled: `qwen2.5:14b` (LLM) and `nomic-embed-text` (embeddings)

```bash
pip install -r requirements.txt
```

## Usage

```bash
source ~/myvenv/bin/activate   # or your own venv
python main.py
```

On startup, `NUM_AGENTS` personas are sampled at random from the pool of ~200 000 role descriptions in `agents/personas.json`. The LLM then expands each one-liner into a full name and 2-3 sentence persona description before the conversation begins.

## Key configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `qwen2.5:14b` | Ollama model used for responses and scoring |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `NUM_AGENTS` | `2` | Number of personas to sample per run |
| `REFLECT_EVERY` | `2` | Reflection frequency (rounds) |
| `EVAL_EVERY` | `4` | Evaluation frequency (rounds) |
| `DEFAULT_MAX_ROUNDS` | `12` | Hard conversation limit |
| `MAX_MEMORIES_SEED` | `15` | Recent memories fed to reflection |
| `MAX_MEMORIES_RETRIEVE` | `5` | Memories surfaced per agent response |
| `MEMORY_PERSIST` | `False` | Persist ChromaDB to disk |

## References

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *UIST 2023*.
