# Multi-Agent Deliberation Simulation

A thesis research project implementing a multi-agent deliberation system with persistent memory, reflection, and social evaluation — powered by local LLMs via [Ollama](https://ollama.ai).

The research question is how social reinforcement produces polarisation structures in a network — grounded in the social-feedback model of **Banisch & Olbrich (2019)**.

The project provides two simulation modes:

| Mode | Entry point | Description |
|---|---|---|
| **Pairwise** | `main.py` | Two agents converse in a sequential round-based loop |
| **Network** | `main_network.py` | N agents form a graph; N/2 pairs discuss each round; edges evolve |

Both modes share the same agent primitives (`respond`, `reflect`, `evaluate`), memory system, and round structure. The network module is the canonical simulation driver; the pairwise mode is a lightweight special case with two agents and no graph.

No LangGraph. No OpenAI. Local Ollama only (`qwen2.5:14b` + `nomic-embed-text`).

---

## Terminology

| Term | Definition |
|---|---|
| **Turn** | One `respond()` call — one agent speaks once |
| **Exchange** | One full back-and-forth — both agents speak once (2 turns) |
| **Discussion** | `DISCUSSION_TURNS` exchanges (`DISCUSSION_TURNS × 2` turns total) |
| **Simulation round** | One complete discussion between a pair — identical meaning in both modes |

These terms are used consistently throughout the codebase. "Round" always means simulation round; "exchange" always means one back-and-forth; "turn" always means a single agent utterance.

---

## Architecture

```
config.py              single source of truth for all constants
agents/agent.py        Agent class — respond(), reflect(), evaluate()
agents/personas.py     samples survey records, expands to name+persona via LLM
memory/store.py        per-agent ChromaDB collection
memory/scoring.py      composite score: 0.3·recency + 0.3·importance + 0.4·relevance
network/state.py       NetworkState, EdgeData dataclasses
network/matching.py    compute_pairings(), reconnect_isolated(), ensure_connectivity()
network/discussion.py  run_discussion() — turn loop + evaluate()
network/edges.py       update_edge() — adjust per-agent valuations by concordance score, drop when either ≤ STRENGTH_FLOOR
network/logger.py      SimulationLogger — events.jsonl + round_NNNN.json snapshots
main.py                pairwise entry point
main_network.py        network entry point
data/                  ZA9089_JSON.xlsx (raw survey), german_personas.json (persona pool)
logs/                  simulation output — created at runtime by main_network.py
```

---

## Agent primitives

All three methods are called directly — no graph framework wraps them.

### `respond(message, speaker) -> str`
- Retrieves up to `MAX_MEMORIES_RETRIEVE` relevant memories (embedding similarity + composite score)
- Prompt instructs the agent to **take a clear position** on the topic, not merely stay in character
- Stores the full interaction as a new memory afterward
- **Do not soften the position-taking instruction** — it is required for `evaluate()` to produce a reliable concordance signal

### `reflect()`
- Two-step: generate 2 reflection questions from the most recent `MAX_MEMORIES_SEED` memories → synthesise 1 insight per question
- Stores insights as high-importance `"reflection"` memories, influencing future `respond()` calls
- Triggered every `REFLECT_EVERY` rounds in both modes

### `evaluate(messages) -> {"agent", "score", "reason"}`
- Rates the conversation **exclusively on opinion concordance** (Banisch reward signal)
- Returns a continuous score in [−1.0, 1.0]: positive = agreement, negative = disagreement
- Output format: `BEWERTUNG: <float>` + `MEINUNGSABGLEICH: <concordance sentence>`
- Parser keys on `"BEWERTUNG:"` and `"MEINUNGSABGLEICH:"` prefixes
- **Do not add social/emotional criteria** (politeness, feeling heard, etc.) — excluded by design

---

## Memory system

Each agent has a per-agent [ChromaDB](https://www.trychroma.com) collection. Memories are ranked by a composite score:

```
score = 0.3 × recency + 0.3 × importance + 0.4 × relevance
```

- **Recency** — exponential decay over time
- **Importance** — LLM-rated significance (1–10), normalised to [0, 1]
- **Relevance** — cosine similarity to the current query embedding

---

## Pairwise mode

### Round structure

Pairwise mode uses the same round structure as network mode — each round is one complete discussion of `DISCUSSION_TURNS` exchanges via `run_discussion()`.

1. Determine the active topic from the `TOPICS` schedule.
2. Run one discussion (`DISCUSSION_TURNS` exchanges); both agents evaluate and return a concordance score.
3. Each agent's internal edge valuation is adjusted by their own score × `STRENGTH_DELTA`. The conversation ends early if either agent's valuation falls to or below `STRENGTH_FLOOR`.
4. Every `REFLECT_EVERY` rounds all agents reflect on their recent memories.
5. The conversation ends after `NETWORK_MAX_ROUNDS` rounds at the latest.

---

## Network mode

### Overview

`main_network.py` runs a network simulation where agents are nodes in a graph and edges represent active two-way communication channels. Each simulation round, agents are matched into pairs and hold a multi-turn discussion. After each discussion, each agent's concordance score independently adjusts their internal valuation of the edge; the edge is severed as soon as either agent's valuation falls to or below `STRENGTH_FLOOR`.

Rounds are grouped into topic blocks: the `TOPICS` dict in `config.py` defines an ordered set of discussion questions, and `NETWORK_MAX_ROUNDS` is divided evenly across them so that agents deliberate on each topic for an equal number of rounds before moving to the next.

This design is grounded in the social-feedback model of Banisch & Olbrich (2019): agents who consistently agree with a partner strengthen that relationship, while persistent disagreement erodes it until the edge severs naturally and the agent seeks new partners — producing emergent network dynamics (clustering, echo chambers, fragmentation).

### Network round structure

```
For each round:
  1. Determine topic    active topic = TOPICS block for this round
  2. Compute pairings   max-weight matching over existing edges (strength-weighted);
                        unmatched agents pause the round (strengths + memories unchanged)
  3. For each pair      DISCUSSION_TURNS × 2 alternating LLM calls via agent.respond()
                        → both agents call agent.evaluate() → score ∈ [−1.0, 1.0]
  4. Edge update        edge.strengths[agent_a] += score_a × STRENGTH_DELTA
                        edge.strengths[agent_b] += score_b × STRENGTH_DELTA
                        either value ≤ STRENGTH_FLOOR → edge removed
  5. Reconnect          agents with degree 0 are reconnected to a uniformly random partner
  6. Reflection         if round % REFLECT_EVERY == 0: all agents reflect
  7. Snapshot           network state written to logs/run_<timestamp>/network_rounds/
```

### Edge lifecycle

Each edge stores a per-agent internal valuation (`EdgeData.strengths`, keyed by agent name), both starting at 1.0.  After each discussion each agent's value is updated independently:

```
edge.strengths[agent_a] += score_a × STRENGTH_DELTA
edge.strengths[agent_b] += score_b × STRENGTH_DELTA
each value = clamp(value, 0, STRENGTH_CAP)
if either value ≤ STRENGTH_FLOOR: remove edge
```

The edge is severed as soon as *either* agent's valuation falls to or below `STRENGTH_FLOOR` — one agent's dissatisfaction is sufficient.  The matching weight is the sum of both values, so mutually valued relationships are preferred in pairing.

`EdgeData.rounds_active` is incremented whenever the edge survives, for post-hoc analysis.

### Initial topology

The network is initialised as a **Watts-Strogatz small-world graph** with parameters `INITIAL_GRAPH_K` (average degree ≈ k) and `INITIAL_GRAPH_P` (rewiring probability). This topology produces the community clustering that Banisch & Olbrich identify as the structural precondition for polarisation, while also providing long-range ties that allow views to cross community boundaries.

### Output

Each run produces a timestamped directory under `logs/`:

```
logs/run_<timestamp>/
├── personas.json                 # agent names and persona descriptions (written at startup)
├── events.jsonl                  # one JSON record per event (discussion, edge, reflection)
└── network_rounds/
    ├── round_0000.json           # baseline snapshot before round 1
    ├── round_0001.json
    └── ...
```

**`events.jsonl`** event types:

| `type` | Fields |
|---|---|
| `discussion` | `round`, `agent_a`, `agent_b`, `topic_label`, `turns`, `score_a`, `reason_a`, `score_b`, `reason_b` |
| `edge_maintained` | `round`, `agent_a`, `agent_b` |
| `edge_dropped` | `round`, `agent_a`, `agent_b` |
| `edge_added` | `round`, `agent_a`, `agent_b` (reconnection) |
| `reflection` | `round`, `agent` |

**`round_NNNN.json`** snapshot fields: `round`, `idle_agent`, `nodes`, `edges` (with `strengths` dict and `rounds_active`), `metrics` (`density`, `n_components`, `avg_degree`, `n_edges`).

---

## Theoretical grounding (Banisch & Olbrich 2019)

| Banisch concept | This simulation |
|---|---|
| Public opinion o_i ∈ {−1,+1} | Agent's expressed position in `respond()` |
| Social reward r = o_i·o_j ∈ {−1,+1} | `evaluate()` score ∈ [−1.0, 1.0] |
| Q-value update | Not yet implemented — see extension point below |
| Network co-evolution | Per-agent edge valuations adjusted via `update_edge()`; severed when either ≤ `STRENGTH_FLOOR` |
| Structural polarisation (n_d) | Not yet logged — see extension point below |

The continuous score extends Banisch's binary ±1 reward to a gradient, allowing partial agreement to be captured. In Banisch's formulation r = o_i · o_j is binary because public opinions are discrete; the gradient here is a deliberate extension.

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
NUM_AGENTS         = 4
NETWORK_MAX_ROUNDS = 3   # must be divisible by len(TOPICS) for equal blocks
DISCUSSION_TURNS   = 2   # 2 exchanges per discussion = 4 turns total
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
| `MAX_MEMORIES_SEED` | `15` | Recent memories fed to prompt reflection question |
| `MAX_MEMORIES_RETRIEVE` | `5` | Memories surfaced per agent response/reflection |
| `MEMORY_PERSIST` | `False` | Persist ChromaDB to disk |

### Network mode only

| Setting | Default | Description |
|---|---|---|
| `NUM_AGENTS` | `20` | Agents in the network (pairwise mode always uses 2) |
| `NETWORK_MAX_ROUNDS` | `30` | Total simulation rounds (used by both modes) |
| `DISCUSSION_TURNS` | `3` | Exchanges per discussion (total turns = value × 2) |
| `INITIAL_GRAPH_K` | `4` | Watts-Strogatz k (initial avg degree) |
| `INITIAL_GRAPH_P` | `0.3` | Watts-Strogatz p (rewiring probability) |
| `STRENGTH_CAP` | `3.0` | Ceiling on each agent's internal edge valuation |
| `STRENGTH_FLOOR` | `0.0` | Edge removed when either agent's valuation falls to or below this |
| `STRENGTH_DELTA` | `0.3` | Valuation change per round = agent_score × STRENGTH_DELTA |
| `TOPICS` | *(dict)* | Ordered label → question mapping; rounds divided evenly across entries |

---

## Prompt design and Banisch & Olbrich alignment

**Response prompt (`respond`)** instructs the agent to *take a clear position* on the discussed topic and express their opinion directly — not merely to respond in character. This is necessary because `evaluate()` judges concordance based on the expressed positions: if agents hedge or respond conversationally without committing to a stance, the opinion signal is unreliable and concordance detection breaks down.

**Evaluation prompt (`evaluate`)** asks the agent to rate the conversation *exclusively on opinion concordance* on a continuous scale from −1.0 (complete disagreement) to +1.0 (complete agreement). General social factors — politeness, conversational style, feeling heard — are explicitly excluded. The score is used directly to adjust the calling agent's internal edge valuation; the edge is severed when either agent's valuation reaches `STRENGTH_FLOOR`. Output format: `BEWERTUNG: <float>` + `MEINUNGSABGLEICH: <sentence>`.

Changing either prompt changes the simulation's theoretical alignment. Document any prompt change in this section.

---

## Conventions

**Language** — all LLM prompts and agent outputs are in German. Code, docstrings, comments, and logs are in English.

**Config** — all tunables live in `config.py`. No magic numbers in source files.

**`DISCUSSION_TURNS`** — counts exchanges per discussion, not individual turns. Total turns = `DISCUSSION_TURNS × 2`. Do not reinterpret as turns.

**`TOPICS`** — insertion order determines round scheduling. `NETWORK_MAX_ROUNDS` should be divisible by `len(TOPICS)`; if not, the last topic absorbs the remainder. A single entry reproduces single-topic behaviour. The active label appears in each round's console header and in `events.jsonl` as `topic_label`.

**Edge drop rule** — threshold-based: edge is severed when *either* agent's internal valuation (`EdgeData.strengths[name]`) falls to or below `STRENGTH_FLOOR`. One dissatisfied agent is sufficient; no bilateral consensus required.

**Matching** — uses `nx.max_weight_matching` with the sum of both agents' internal valuations as the edge weight. Agents left unmatched pause that round; their edge strengths and memories are unchanged. No introductory edges are created for unmatched agents.

**Logging** — `SimulationLogger` writes to `logs/run_<timestamp>/`. The `extra_metrics` parameter on `snapshot_network()` is the designated slot for Banisch polarisation metrics; do not add separate log files.

**No opinion state yet** — do not add Q-values, opinion-stance extraction, or polarisation metrics until the extension is explicitly requested.

### What not to touch without good reason

- `agents/agent.py` — core primitives; prompt wording has theoretical justification
- `memory/scoring.py` — scoring weights (0.3/0.3/0.4) are tuned
- `network/matching.py` — blossom algorithm + random reconnection logic
- The `BEWERTUNG` / `MEINUNGSABGLEICH` output labels in `evaluate()` — parsed by prefix match
- `STRENGTH_DELTA`, `STRENGTH_FLOOR`, `STRENGTH_CAP` — tuned for edge lifecycle dynamics

---

## Extension: Banisch & Olbrich opinion tracking

The network module is structured so that adding Banisch & Olbrich (2019) opinion state requires minimal changes:

1. Create `network/opinion.py` with `AgentOpinionState`, `init_opinion_states`, `update_opinion_states`, and `compute_metrics`.
2. In `main_network.py`, uncomment the three `# EXTENSION POINT` blocks (≈ 5 lines total).
3. In `network/state.py`, uncomment the `opinion_states` field (1 line).

`score_a` and `score_b` from `run_discussion()` are already the continuous reward signal in [−1.0, 1.0] and can be passed directly to `update_opinion_states` — no changes to `network/discussion.py` are needed.

---

## References

Banisch, S., & Olbrich, E. (2019). Opinion polarization by learning from social feedback. *The Journal of Mathematical Sociology, 43*(2), 76–103.

Jacob, D., & Banisch, S. (2023). Polarization in social media: A virtual worlds-based approach. *Journal of Artificial Societies and Social Simulation, 26*(3), 11.

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *UIST 2023*.
