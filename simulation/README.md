# Multi-Agent Deliberation Simulation

A thesis research project implementing a hybrid LLM + Social Feedback Theory (SFT) simulation of opinion dynamics in structured social networks — powered by local LLMs via [Ollama](https://ollama.ai).

**Research question:** Can Social Feedback Theory serve as an explicit, parameterised reinforcement-learning governor for LLM agents embedded in a structured graph, producing opinion dynamics that are both more mechanistically interpretable than free-form LLM-ABMs and more behaviourally realistic than classical SFT?

The project provides two simulation modes:

| Mode | Entry point | Description |
|---|---|---|
| **Network** | `main_network.py` | N agents in an SBM graph; asymmetric interactions; Q-values govern opinion expression |
| **Pairwise** | `main_pairwise.py` | Two agents; same SFT mechanisms as network mode (Q-values, softmax β, asymmetric interactions); no graph or community structure |

Network mode is the canonical simulation driver for multi-agent experiments. Pairwise mode runs the identical SFT mechanisms for exactly two agents with no graph.

No LangGraph. No OpenAI. Local Ollama only (`qwen2.5:14b` + `nomic-embed-text`).

---

## Terminology

| Term | Definition |
|---|---|
| **Turn** | One `respond()` call — one agent speaks once |
| **Exchange** | One full back-and-forth — both agents speak once (2 turns) |
| **Interaction** | One asymmetric event: one expresser drawn, one responder drawn, one exchange, one Q-update for the expresser only |
| **Round** | `INTERACTIONS_PER_ROUND` interactions; the snapshotting unit |
| **Discussion** | One asymmetric interaction in both modes: 1 exchange (expresser speaks, responder reacts) |

---

## Architecture

```
config.py               single source of truth for all constants
agents/agent.py         Agent class — respond(), classify_reward(), reflect(); evaluate() [legacy]
agents/personas.py      samples survey records, expands to name+persona via LLM
memory/store.py         per-agent ChromaDB collection
memory/scoring.py       composite score: 0.3·recency + 0.3·importance + 0.4·relevance
network/opinion.py      SFT Q-value state — AgentOpinionState, softmax_opinion(), update_q_value(), metrics
network/state.py        NetworkState, EdgeData dataclasses
network/matching.py     select_responder() — homophily-weighted partner draw; also compute_pairings(), ensure_connectivity()
network/discussion.py   run_discussion() — exchange loop + classify_reward()
network/edges.py        update_edge() — active only when GRAPH_DYNAMIC = True
network/logger.py       SimulationLogger — events.jsonl + round_NNNN.json snapshots
main_network.py         network entry point
main_pairwise.py        pairwise entry point
data/                   ZA9089_JSON.xlsx (raw survey), german_personas.json (persona pool)
logs/                   simulation output — created at runtime
```

---

## SFT architecture

The Q-update rule lives entirely in code. The LLM handles three functions per interaction: generating the expresser's utterance (`respond()`), generating the responder's reaction (`respond()`), and classifying that reaction as a scalar reward (`classify_reward()`). Every opinion shift traces to an auditable Q-trajectory, not to an opaque LLM forward pass.

### Agent internal state

Each agent holds:

- **`AgentOpinionState`** (`network/opinion.py`) — Q-values `q_pos` (Q(+1)) and `q_neg` (Q(−1)), representing accumulated social value of each stance. The **preferred opinion** (`preferred_opinion` property) is `argmax(q_pos, q_neg)` — the stance the agent *would prefer* based on accumulated Q-values — and is used for metrics and logging. The **actual expressed stance** in each interaction is drawn stochastically by `softmax_opinion(β)` — at low conviction (q_gap near 0) the two can differ. Both are recorded in the per-round snapshot under `"preferred"` and `"expressed"` keys respectively.
- **Memory** — a ChromaDB collection of past interactions and reflections, used to condition `respond()`.
- **Persona** — a German-language profile from survey data, injected into every LLM prompt.

The persona and memory condition LLM text generation only. The Q-update rule sees only the scalar classifier reward.

### Interaction loop (one asymmetric interaction)

```
1. Draw expresser     uniform random from agents with ≥1 neighbour
2. Draw responder     select_responder() from expresser's adjacency with bias h
                      h=0 → uniform; h>0 → similar-conviction neighbours preferred
3. Express            softmax(β) draw over Q-values → expressed stance (+1 or −1)
                      β=0 → 50/50; β→∞ → deterministic argmax
4. Exchange           1 exchange: expresser speaks (anchored to expressed stance),
                                   responder reacts (anchored to their own stance).
                                   On the first meeting the expresser responds to the
                                   moderator's topic; on repeat meetings it responds to
                                   the partner's last message, continuing the dialogue.
                                   The topic is always in the stance hint.
5. Reward             classify_reward(responder's last message) → r ∈ [−1, +1]
                      minimal prompt, no persona/transcript context
6. Q-update           Q(expressed) ← (1−α)·Q(expressed) + α·r  [expresser only]
7. Edge update        update_edge(reward_a)  [GRAPH_DYNAMIC only]
                      appends r to edge's rolling history (window REWARD_WINDOW_M);
                      signal = mean(history) drives strength update
```

### Two reward signals

| Signal | Source | Used for |
|---|---|---|
| `reward_a` | `classify_reward()` on responder's last message | Q-value TD update (expresser only); edge-history evaluation in network mode (`GRAPH_DYNAMIC`); strength update in pairwise mode |
| `reward_b` | `classify_reward()` on expresser's last message | Available for symmetric edge/strength extension; not used in any active update path |

`classify_reward()` uses a minimal prompt — no persona, no history — so that expression and reward remain as causally independent as possible. `run_discussion()` computes both signals; `reward_b` is available for a future symmetric extension.

---

## Agent primitives

### `respond(message, speaker, expressed_opinion=None, topic=None) -> str`

Retrieves up to `MAX_MEMORIES_RETRIEVE` relevant memories, optionally anchors to an SFT stance (`expressed_opinion`: +1 or −1), and generates a position-taking reply. When `topic` is provided it is embedded in the stance hint so the agent remains oriented to the discussion question regardless of whether the immediately preceding message was the moderator's opening or a partner's continuation. The full interaction is stored as a new memory.

**Do not soften the position-taking instruction.** A legible stance is required for `classify_reward()` to detect agreement/disagreement reliably.

### `classify_reward(reaction_text) -> float`

Rates a single reaction text in [−1.0, 1.0] using a minimal prompt with no persona or transcript context. Returns the scalar `r` for the Q-value TD update. Ambivalent reactions map to values near 0, preserving the graded signal that binary SFT cannot represent.

### `reflect()`

Two-step: generate 2 reflection questions from the most recent `MAX_MEMORIES_SEED` memories → synthesise 1 insight per question. Stores insights as high-importance `"reflection"` memories. Triggered every `REFLECT_EVERY` rounds.

### `evaluate(messages) -> {"agent", "score", "reason"}` [legacy]

Rates the full transcript exclusively on opinion concordance, returning a score in [−1.0, 1.0] with a one-sentence explanation. Previously drove edge valuation in `GRAPH_DYNAMIC` mode; superseded by the reward-history mechanism in `network/edges.py`, which derives the edge signal from the rolling mean of `classify_reward()` outputs.

Not called by any active simulation path. Retained in the `Agent` class in case the concordance-based evaluation path is revisited.

Output format: `BEWERTUNG: <float>` + `MEINUNGSABGLEICH: <sentence>`.

---

## Memory system

Each agent has an isolated [ChromaDB](https://www.trychroma.com) collection. Retrieval ranks by composite score:

```
score = 0.3 × recency + 0.3 × importance + 0.4 × relevance
```

- **Recency** — exponential decay over time
- **Importance** — LLM-rated significance (1–10), normalised to [0, 1]
- **Relevance** — cosine similarity to the current query embedding

---

## Network mode

### Round structure

```
For each round (= INTERACTIONS_PER_ROUND events):
  For each interaction:
    1. Draw expresser   uniform random from agents with ≥1 neighbour
    2. Draw responder   select_responder(h) from expresser's neighbours
    3. Softmax draw     expressed_a = softmax(β) over expresser's Q-values
    4. Exchange         expresser speaks → responder reacts  (1 exchange each)
                        First meeting: expresser responds to moderator's topic.
                        Repeat meeting: expresser responds to partner's last message
                        (continuation); topic always present in stance hint.
    5. Reward           classify_reward(responder's message) → reward_a
    6. Q-update         Q(expressed_a) ← (1−α)·Q(expressed_a) + α·reward_a
    7. Edge update      update_edge(reward_a)  [GRAPH_DYNAMIC only]
                        appends reward_a to edge's rolling history window;
                        signal = mean(history[-REWARD_WINDOW_M:])
  After all interactions:
    8. Reconnect        [GRAPH_DYNAMIC only] degree-0 agents reconnected
    9. Reflection       if round % REFLECT_EVERY == 0: all agents reflect
   10. Snapshot         Q-trajectories + polarization metrics → logs/
```

### Initial topology — Stochastic Block Model

The network is initialised as a **Stochastic Block Model** with `SBM_NUM_COMMUNITIES` communities. `SBM_P_INTRA` sets within-community density; `SBM_P_INTER` sets between-community density.

`SBM_P_INTER` is the primary experimental variable: sweeping it from low to high reproduces SFT's polarization-to-consensus phase transition. At low coupling, agents mostly interact within their community, opinion clusters align structurally, and polarization is stable. At high coupling, cross-community feedback erodes conviction and drives consensus.

Each node carries a `"community"` attribute for post-hoc structural analysis.

### Graph dynamics

| `GRAPH_DYNAMIC` | Behaviour |
|---|---|
| `False` (default) | Graph structure fixed for the full run; steps 7–8 are skipped. Use for main SFT experiments to avoid confounding opinion dynamics with structural dynamics. |
| `True` | Edge valuations evolve via `update_edge()`; edges severed at `STRENGTH_FLOOR`; isolated agents reconnected. Use for the homophilic tie-formation extension. |

### Output

Each run produces a timestamped directory under `logs/`:

```
logs/run_<timestamp>/
├── personas.json              # agent names and persona descriptions
├── events.jsonl               # one JSON record per event
└── network_rounds/
    ├── round_0000.json        # baseline snapshot before round 1
    ├── round_0001.json
    └── ...
```

**`events.jsonl`** event types:

| `type` | Key fields |
|---|---|
| `discussion` | `round`, `agent_a` (expresser), `agent_b` (responder), `topic_label`, `turns`, `reward_a`, `reward_b`, `expressed_a` (softmax draw for agent_a), `expressed_b` (softmax draw for agent_b), `preferred_a` (argmax for agent_a pre-update), `preferred_b` (argmax for agent_b pre-update) |
| `edge_maintained` | `round`, `agent_a`, `agent_b` |
| `edge_dropped` | `round`, `agent_a`, `agent_b` |
| `edge_added` | `round`, `agent_a`, `agent_b` |
| `reflection` | `round`, `agent` |

**`round_NNNN.json`** snapshot:

```json
{
  "round": 5,
  "idle_agent": null,
  "nodes": ["Anna", "Ben", "Clara", "David"],
  "edges": [{"a": "Anna", "b": "Ben", "strengths": {"Anna": 1.2, "Ben": 0.9}, "rounds_active": 3}],
  "metrics": {
    "density": 0.33,
    "n_components": 1,
    "avg_degree": 2.0,
    "n_edges": 4,
    "n_pos": 2,
    "n_neg": 2,
    "dispersion": 1.0,
    "mean_q_gap": 0.12
  },
  "opinion_states": {
    "Anna": {"q_pos": 0.18, "q_neg": 0.04, "preferred": 1, "expressed": 1, "q_gap": 0.14}
  }
}
```

`opinion_states` records the Q-trajectory for every agent after every round, enabling the interpretability check: testing whether Q-gap trajectories predict observed opinion switches.  `"preferred"` is the deterministic argmax; `"expressed"` is the actual softmax-drawn stance from the last interaction in which the agent was expresser (absent if the agent was never expresser in that round).

---

## Pairwise mode

Runs the identical SFT mechanisms as network mode for exactly two agents with no graph. Each round consists of `INTERACTIONS_PER_ROUND` asymmetric interactions, with the expresser drawn uniformly at random each time (with two agents, the responder is always the other agent).

**What is absent:** SBM graph, community structure. Homophily parameter `h` has no effect (only one possible responder).

### Round structure

```
For each round (INTERACTIONS_PER_ROUND asymmetric interactions; expresser drawn uniformly):
  For each interaction:
    1. Softmax draw    expressed = softmax(β) over expresser's Q-values
    2. Exchange        expresser speaks → responder reacts  (1 exchange)
                       First meeting: expresser responds to moderator's topic.
                       Repeat meeting: expresser responds to partner's last message
                       (continuation); topic always present in stance hint.
    3. Reward          classify_reward(responder's message) → reward_a
    4. Q-update        Q(expressed) ← (1−α)·Q(expressed) + α·reward_a  [expresser only]
    5. Strength        [GRAPH_DYNAMIC only] reward_a → strength[expresser] update;
                       exit if either strength ≤ STRENGTH_FLOOR
  After all interactions:
    6. Reflection      if round % REFLECT_EVERY == 0
```

---

## Theoretical alignment

| Banisch & Olbrich (2019) concept | This simulation |
|---|---|
| Public opinion o_i ∈ {−1, +1} | Drawn by `softmax_opinion(β)`; deterministic preferred indicator is `preferred_opinion` = `argmax(q_pos, q_neg)` |
| Social reward r = o_i · o_j ∈ {−1, +1} | `classify_reward()` output ∈ [−1.0, 1.0] — continuous, preserves ambivalence |
| Q-value update Q(o_i) ← (1−α)·Q(o_i) + α·r | `update_q_value(expressed, reward, α)` — expresser only, per interaction |
| Asymmetric one-directional update | One expresser drawn per event; only their Q updates (Jacob & Banisch 2023) |
| Homophily partner selection | `select_responder(h)` — h=0 uniform (2019), h>0 conviction-similarity weighted (2023) |
| Network co-evolution (endogenous) | `GRAPH_DYNAMIC = True` — extension mode |
| Phase transition via modularity | `SBM_P_INTER` sweep — primary experimental variable |
| Structural polarization | `dispersion`, `n_pos/n_neg` in round snapshots |

The continuous reward on [−1, +1] is a deliberate extension of Banisch's binary ±1, preserving ambivalent feedback that Sárközi et al. (2022) found most influential in empirical tests.

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

```bash
python main_network.py   # full SFT network simulation
python main_pairwise.py  # two-agent mode (no SFT Q-layer)
```

For a quick smoke test, set in `config.py`:

```python
NUM_AGENTS             = 4
NETWORK_MAX_ROUNDS     = 3
INTERACTIONS_PER_ROUND = 4  # = NUM_AGENTS
```

Both entry points use these values, including `INTERACTIONS_PER_ROUND`.

On startup, personas are sampled from `data/german_personas.json` (5 246 German citizen survey records). The LLM generates a realistic German first name and 2–3 sentence persona description from each record's demographic and attitudinal attributes. All agent prompts and responses run in German.

---

## Configuration (`config.py`)

### Shared (both modes)

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `qwen2.5:14b` | Ollama model for all LLM calls |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `NETWORK_MAX_ROUNDS` | `3` | Total rounds; snapshot unit in network mode, loop bound in pairwise mode |
| `REFLECT_EVERY` | `2` | Reflection frequency (rounds) |
| `MAX_MEMORIES_SEED` | `15` | Recent memories fed to reflection prompt |
| `MAX_MEMORIES_RETRIEVE` | `5` | Memories retrieved per `respond()` call |
| `MEMORY_PERSIST` | `False` | Persist ChromaDB to disk |
| `STRENGTH_CAP` | `3.0` | Ceiling on per-agent strength/edge valuation |
| `STRENGTH_FLOOR` | `0.0` | Termination threshold: conversation/edge ends when either agent's value ≤ this |
| `STRENGTH_DELTA` | `0.3` | Strength change per interaction = mean_reward × STRENGTH_DELTA |
| `INTERACTIONS_PER_ROUND` | `= NUM_AGENTS` | Asymmetric interactions per snapshot round — applies to both modes; expresser drawn uniformly each time |

### SFT Q-learning (network mode only)

| Setting | Default | Description |
|---|---|---|
| `LEARNING_RATE` | `0.1` | α in Q(o) ← (1−α)·Q(o) + α·r |
| `OPINION_BETA` | `5.0` | β: softmax inverse temperature; β=0 → 50/50, β→∞ → argmax |
| `HOMOPHILY_H` | `0.0` | h: responder-selection bias; 0 = uniform (2019 baseline), >0 = conviction-similarity weighted |

### Network topology

| Setting | Default | Description |
|---|---|---|
| `NUM_AGENTS` | `4` | Agents in the network graph |
| `SBM_NUM_COMMUNITIES` | `2` | Number of blocks in the SBM graph |
| `SBM_P_INTRA` | `0.7` | Within-community edge probability |
| `SBM_P_INTER` | `0.1` | Between-community edge probability — **sweep this for the phase transition** |

### Graph dynamics (network mode, `GRAPH_DYNAMIC = True` only)

| Setting | Default | Description |
|---|---|---|
| `GRAPH_DYNAMIC` | `False` | Enable endogenous tie rewiring via edge valuations |
| `REWARD_WINDOW_M` | `5` | Rolling window length (interactions) for reward-history edge evaluation |

### Topic

| Setting | Default | Description |
|---|---|---|
| `TOPIC_LABEL` | `"Migrationspolitik"` | Short identifier used in console output and `events.jsonl` |
| `TOPIC_TEXT` | *(question string)* | The discussion question passed to agents as the opening moderator message |

A single fixed topic is required for Q-value coherence: the +1/−1 stance dimension must refer to the same opinion object throughout the run. Multi-topic schedules are deliberately not supported — see `PLAN_banisch_opinion.md` for the rationale.

---

## Conventions

**Language** — all LLM prompts and agent outputs are in German. Code, docstrings, comments, and logs are in English.

**Config** — all tunables live in `config.py`. No magic numbers in source files.

**Asymmetric update** — only the expresser's Q-value is updated per interaction. The responder's Q-value is unchanged. This is the core SFT mechanism; do not add a symmetric update without revisiting the theoretical alignment.

**Q-update consistency** — `update_q_value()` takes the *actual drawn stance* (`expressed`) as an explicit parameter, not the modal argmax. Always pass the value returned by `softmax_opinion()` for that interaction.

**Edge drop rule** — threshold-based: edge is severed when *either* agent's valuation falls to or below `STRENGTH_FLOOR`. Valuation is updated from the rolling mean of the agent's `reward_a` history on that edge (window `REWARD_WINDOW_M`). Active only when `GRAPH_DYNAMIC = True`.

**`select_responder`** — the partner-selection function is kept as a standalone swappable unit. The virtual-worlds multi-platform extension (Jacob & Banisch 2023) slots in here by replacing the neighbour set, not by changing the main loop.

**Logging** — `SimulationLogger` writes to `logs/run_<timestamp>/`. Pass SFT metrics via `extra_metrics` and Q-snapshots via `opinion_states` to `snapshot_network()`.

### What not to change without good reason

- `classify_reward()` prompt — minimal context is deliberate; adding persona/transcript context re-couples expression and evaluation
- `respond()` position-taking instruction — a legible stance is required for reward detection to work
- `memory/scoring.py` weights (0.3/0.3/0.4) — tuned composite retrieval
- The `BEWERTUNG` / `MEINUNGSABGLEICH` output labels in `evaluate()` — parsed by prefix match (applies if `evaluate()` is reinstated)
- `LEARNING_RATE` and `OPINION_BETA` together govern the convergence rate — change them together and re-run the phase-transition baseline

---

## Extension: endogenous tie formation

The main experiments hold the graph fixed (`GRAPH_DYNAMIC = False`) to isolate opinion dynamics from structural dynamics. The extension chapter activates tie rewiring by setting `GRAPH_DYNAMIC = True`:

- After each interaction, `reward_a` is appended to the expresser's rolling history deque on that edge (`EdgeData.reward_history`, window `REWARD_WINDOW_M`). The rolling mean drives the expresser's strength update via `update_edge()`.
- Edges are severed when either agent's valuation reaches `STRENGTH_FLOOR`.
- Isolated agents are reconnected via `ensure_connectivity()`.

Known failure modes: (1) the graph crystallising before Q-values diverge — keep `STRENGTH_DELTA` small relative to `LEARNING_RATE`; (2) premature edge drops before the history window fills — cold-start signal defaults to 0.0 (neutral) until `REWARD_WINDOW_M` interactions have accumulated on that edge.

The further extension toward the full Jacob & Banisch (2023) virtual-worlds model (parallel real-world / virtual-worlds networks with login probability λ) is out of scope for the main experiments. It slots in by replacing the neighbour set passed to `select_responder()`.

---

## References

Banisch, S., & Olbrich, E. (2019). Opinion polarization by learning from social feedback. *The Journal of Mathematical Sociology, 43*(2), 76–103.

Jacob, D., & Banisch, S. (2023). Polarization in social media: A virtual worlds-based approach. *Journal of Artificial Societies and Social Simulation, 26*(3), 11.

Sárközi, R., Denz, T., & Lorenz-Spreen, P. (2022). Testing social feedback theory: An experiment on the effect of social feedback on opinion expression. *PLOS ONE, 17*(4).

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *UIST 2023*.
