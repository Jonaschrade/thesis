# Multi-Agent Deliberation Simulation

A thesis research project implementing a multi-agent deliberation system with persistent memory, reflection, and social evaluation — powered by local LLMs via [Ollama](https://ollama.ai).

The project provides two simulation modes:

| Mode | Entry point | Description |
|---|---|---|
| **Pairwise** | `main.py` | Two agents converse in a sequential round-based loop |
| **Network** | `main_network.py` | N agents form a graph; N/2 pairs discuss each round; edges evolve |

Both modes share the same agent primitives (`respond`, `reflect`, `evaluate`) and memory system. The network module is the canonical simulation driver; the pairwise mode is a lightweight special case of N=2.

---

## Architecture

```
agents/           Agent class (respond, reflect, evaluate) and persona sampler
memory/           ChromaDB-backed memory storage and composite scoring algorithm
network/          Network simulation module (see below)
data/             Raw survey data (ZA9089_JSON.xlsx) and persona pool (german_personas.json)
notebooks/        Exploratory notebooks (data export, persona sampling, distribution plots)
logs/             Simulation output — created at runtime by main_network.py
img/              Generated plots
config.py         Central configuration (models, weights, thresholds)
main.py           Entry point — pairwise mode
main_network.py   Entry point — network mode
```

### `network/` module

```
network/
├── state.py       NetworkState and EdgeData dataclasses
├── matching.py    Agent pairing (max-weight matching) and reconnection logic
├── discussion.py  Pairwise discussion runner (turn loop + evaluate())
├── edges.py       Edge lifecycle: strengthen on mutual "continue", drop on "move"
└── logger.py      Structured JSONL event log and per-round network snapshots
```

---

## Pairwise mode

### Conversation flow

1. Agents take turns responding in a fixed cyclic order; each round both agents speak once.
2. Every `REFLECT_EVERY` rounds all agents simultaneously reflect, synthesising insights from recent memories.
3. Every `EVAL_EVERY` rounds all agents vote `weiter` or `wechseln`; any `wechseln` vote ends the conversation.
4. The conversation ends after `DEFAULT_MAX_ROUNDS` rounds at the latest.

### Memory system

Each agent has a per-agent [ChromaDB](https://www.trychroma.com) collection. Memories are ranked by a composite score:

```
score = 0.3 × recency + 0.3 × importance + 0.4 × relevance
```

- **Recency** — exponential decay over time
- **Importance** — LLM-rated significance (1–10), normalised
- **Relevance** — cosine similarity to the current query embedding

---

## Network mode

### Overview

`main_network.py` runs a network simulation where agents are nodes in a graph and edges represent active two-way communication channels. Each simulation round, agents are matched into pairs and hold a multi-turn discussion. After each discussion, both agents cast a social-reward vote using the existing `evaluate()` prompt. Edges are maintained or severed based on those votes.

This design is grounded in the social-feedback model of Banisch & Olbrich (2019): agents who consistently experience agreement with a partner strengthen that relationship, while agents who find little resonance sever it and seek new partners, producing emergent network dynamics (clustering, echo chambers, fragmentation).

### Network round structure

```
For each round:
  1. Compute pairings   max-weight matching over existing edges (strength-weighted)
                        + random fallback for unmatched agents
  2. For each pair      DISCUSSION_TURNS alternating LLM calls via agent.respond()
                        → both agents call agent.evaluate() → "weiter" / "wechseln"
  3. Edge update        both "weiter" → strengthen (+0.2, cap 3.0)
                        either "wechseln" → edge removed
  4. Reconnect          agents with degree 0 are reconnected (friends-of-friends, then random)
  5. Reflection         if round % REFLECT_EVERY == 0: all agents reflect
  6. Snapshot           network state written to logs/run_<timestamp>/network_rounds/
```

### Initial topology

The network is initialised as a **Watts-Strogatz small-world graph** with parameters `INITIAL_GRAPH_K` (average degree ≈ k) and `INITIAL_GRAPH_P` (rewiring probability). This topology produces the community clustering that Banisch & Olbrich identify as the structural precondition for polarisation, while also providing long-range ties that allow views to cross community boundaries.

### Output

Each run produces a timestamped directory under `logs/`:

```
logs/run_<timestamp>/
├── events.jsonl                  # one JSON record per event (discussion, edge, reflection)
└── network_rounds/
    ├── round_0000.json           # baseline snapshot before round 1
    ├── round_0001.json
    └── ...
```

**`events.jsonl`** event types:

| `type` | Fields |
|---|---|
| `discussion` | `round`, `agent_a`, `agent_b`, `turns`, `vote_a`, `reason_a`, `vote_b`, `reason_b` |
| `edge_maintained` | `round`, `agent_a`, `agent_b` |
| `edge_dropped` | `round`, `agent_a`, `agent_b` |
| `edge_added` | `round`, `agent_a`, `agent_b` (reconnection) |
| `reflection` | `round`, `agent` |

**`round_NNNN.json`** snapshot fields: `round`, `idle_agent`, `nodes`, `edges` (with `strength` and `rounds_active`), `metrics` (`density`, `n_components`, `avg_degree`, `n_edges`).

---

## Requirements

- Python 3.10+
- A running [Ollama](https://ollama.ai) server at `127.0.0.1:11434`
- Models pulled: `qwen2.5:14b` (LLM) and `nomic-embed-text` (embeddings)

```bash
pip install -r requirements.txt
```

---

## Usage

### Pairwise mode

```bash
python main.py
```

### Network mode

```bash
python main_network.py
```

For a quick smoke test, set in `config.py`:

```python
NUM_AGENTS_NETWORK = 4
NETWORK_MAX_ROUNDS = 3
DISCUSSION_TURNS   = 2
```

On startup, personas are sampled at random from `data/german_personas.json` (5 246 German citizen survey records). The LLM derives a realistic German first name and a 2–3 sentence German persona description (`Du bist …`) from each record's demographic and attitudinal attributes. All agent prompts, responses, reflections, and evaluations run in German.

---

## Key configuration (`config.py`)

### Shared

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `qwen2.5:14b` | Ollama model used for all LLM calls |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `REFLECT_EVERY` | `2` | Reflection frequency (rounds) |
| `MAX_MEMORIES_SEED` | `15` | Recent memories fed to reflection |
| `MAX_MEMORIES_RETRIEVE` | `5` | Memories surfaced per agent response |
| `MEMORY_PERSIST` | `False` | Persist ChromaDB to disk |

### Pairwise mode only

| Setting | Default | Description |
|---|---|---|
| `NUM_AGENTS` | `2` | Agents per run |
| `DEFAULT_MAX_ROUNDS` | `12` | Hard conversation limit |
| `EVAL_EVERY` | `4` | Evaluation frequency (rounds) |

### Network mode only

| Setting | Default | Description |
|---|---|---|
| `NUM_AGENTS_NETWORK` | `20` | Agents in the network |
| `NETWORK_MAX_ROUNDS` | `30` | Total simulation rounds |
| `DISCUSSION_TURNS` | `6` | LLM turns per pairwise discussion |
| `INITIAL_GRAPH_K` | `4` | Watts-Strogatz k (initial avg degree) |
| `INITIAL_GRAPH_P` | `0.3` | Watts-Strogatz p (rewiring probability) |
| `STRENGTH_CAP` | `3.0` | Maximum edge strength |

---

## Prompt design and Banisch & Olbrich alignment

Both prompts in `Agent` are designed to support the social-feedback mechanism of Banisch & Olbrich (2019).

**Response prompt (`respond`)** instructs the agent to *take a clear position* on the discussed topic and express their opinion directly — not merely to respond in character.  This is necessary because `evaluate()` judges concordance based on the expressed positions: if agents hedge or respond conversationally without committing to a stance, the opinion signal is unreliable and the concordance detection breaks down.

**Evaluation prompt (`evaluate`)** asks the agent to judge the conversation *exclusively on opinion concordance*: did the partner share the same position or contradict it?  Agreement is framed as a positive experience that reinforces conviction (→ continue); disagreement as a negative experience that undermines conviction (→ switch partner).  General social factors — politeness, conversational style, feeling heard — are explicitly excluded.  This mirrors Banisch's reward signal exactly: r = +1 for agreement, r = −1 for disagreement, with no role for indifference.

---

## Extension: Banisch & Olbrich opinion tracking

The network module is structured so that adding Banisch & Olbrich (2019) opinion state requires minimal changes. The required steps are:

1. Create `network/opinion.py` with `AgentOpinionState`, `init_opinion_states`, `update_opinion_states`, and `compute_metrics`.
2. In `main_network.py`, uncomment the three EXTENSION POINT blocks (≈ 5 lines total).
3. In `network/discussion.py`, uncomment the `reward_a` / `reward_b` fields (2 lines).
4. In `network/state.py`, uncomment the `opinion_states` field (1 line).

No other file needs modification. See `PLAN_network_simulation.md` for the full extension design.

---

## References

Banisch, S., & Olbrich, E. (2019). Opinion polarization by learning from social feedback. *The Journal of Mathematical Sociology, 43*(2), 76–103.

Jacob, D., & Banisch, S. (2023). Polarization in social media: A virtual worlds-based approach. *Journal of Artificial Societies and Social Simulation, 26*(3), 11.

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *UIST 2023*.
