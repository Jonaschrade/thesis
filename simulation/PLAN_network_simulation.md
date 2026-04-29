# Plan: Network Simulation Extension

## Context

The project is a thesis simulation where LLM-backed agents (grounded in real German survey data) deliberate on immigration policy. The goal is to extend this into a network simulation: agents are nodes in a graph, edges are two-way communication channels, and each round N/2 pairs discuss simultaneously. After each discussion, agents vote (using the already-implemented social feedback `evaluate()` prompt) to keep or drop their edge.

The implementation uses only the **existing agent machinery** (respond, reflect, evaluate) — no opinion stance extraction, no Q-values, no conviction tracking. The code is however structured so that adding Banisch-style opinion state later requires touching only one isolated module and a few clean extension points, without restructuring anything.

---

## Architecture Decision: Plain Python Outer Loop, Agent Primitives Unchanged

The LangGraph `build_graph()` is replaced by a plain Python simulation loop at the network level. `agent.respond()`, `agent.reflect()`, and `agent.evaluate()` are called directly — all three are standalone methods needing no graph infrastructure. The existing `graph/` module stays untouched.

---

## Per-Round Structure

```
Initialization:
  - Sample agents, build initial Watts-Strogatz graph
  - Log round-0 network snapshot

For each simulation round:
  1. Compute pairings: max-weight matching over existing edges + random fallback for unmatched agents
  2. For each pair (agent_a, agent_b):
       a. Run DISCUSSION_TURNS turn-taking loop via agent.respond()
       b. Both agents call agent.evaluate(transcript) → "continue" / "move"
       c. Update or drop edge based on votes
       d. [EXTENSION POINT] opinion module hook (no-op by default)
  3. Reconnect agents with degree 0 (friends-of-friends, then random)
  4. If round % REFLECT_EVERY == 0: agent.reflect() for all agents
  5. Log discussion events + network snapshot
```

---

## File Plan

### Files unchanged
- `agents/agent.py`, `agents/personas.py`
- `memory/store.py`, `memory/scoring.py`
- `graph/` (all files)
- `main.py`

### `config.py` — append only (no removals)
```python
# Network simulation
NUM_AGENTS_NETWORK = 20
NETWORK_MAX_ROUNDS = 30
DISCUSSION_TURNS   = 6        # LLM turns per pairwise discussion (must be even)
INITIAL_GRAPH_K    = 4        # Watts-Strogatz k (initial avg degree)
INITIAL_GRAPH_P    = 0.3      # Watts-Strogatz p (rewiring probability)
STRENGTH_CAP       = 3.0      # maximum edge strength
```

### `requirements.txt` — append `networkx`

---

### `network/` — new module

**`network/__init__.py`** — empty

---

**`network/state.py`**
```python
from dataclasses import dataclass, field
import networkx as nx

@dataclass
class EdgeData:
    strength: float = 1.0       # grows on mutual "continue", drops reset it
    rounds_active: int = 0      # how many discussions this pair has had

@dataclass
class NetworkState:
    agents: dict                     # name -> Agent
    graph: nx.Graph                  # nodes=agent names, edge attr "data": EdgeData
    round: int = 0
    max_rounds: int = 30
    idle_agent: str | None = None    # agent sitting out this round (odd N)
    # EXTENSION POINT: opinion_states: dict | None = None
    # When adding Banisch opinion tracking, add:
    #   opinion_states: dict = field(default_factory=dict)  # name -> AgentOpinionState
```

---

**`network/matching.py`** — pairing logic
- `compute_pairings(state) -> list[tuple[str, str]]`
  1. If `len(agents)` is odd: rotate sit-out agent via `state.round % len(agents)`
  2. `nx.max_weight_matching(subgraph, maxcardinality=True)` with edge weight = `EdgeData.strength`
  3. Pair leftover unmatched agents randomly → new introductory edges at `strength=0.5`
  4. Return full list of pairs for this round
- `reconnect_isolated(state, agent_name)` — tries friends-of-friends first, then random non-neighbour; adds edge at `strength=0.5`

---

**`network/discussion.py`** — pairwise runner
```python
def run_discussion(agent_a, agent_b, topic, turns=DISCUSSION_TURNS) -> dict:
    transcript = [{"speaker": "Moderator", "content": topic}]
    for i in range(turns):
        speaker = agent_a if i % 2 == 0 else agent_b
        last = transcript[-1]
        reply = speaker.respond(last["content"], last["speaker"])
        transcript.append({"speaker": speaker.name, "content": reply})
    convo = transcript[1:]
    eval_a = agent_a.evaluate(convo)
    eval_b = agent_b.evaluate(convo)
    return {
        "turns":    convo,
        "vote_a":   eval_a["vote"],   "reason_a": eval_a["reason"],
        "vote_b":   eval_b["vote"],   "reason_b": eval_b["reason"],
        # EXTENSION POINT: "reward_a" / "reward_b" floats for Q-update
        # Map vote → reward here when adding Banisch opinion model:
        #   "reward_a": 1.0 if eval_a["vote"] == "continue" else -1.0
    }
```

---

**`network/edges.py`** — edge lifecycle
```python
STRENGTH_BONUS = 0.2

def update_edge(state, agent_a: str, agent_b: str, vote_a: str, vote_b: str) -> bool:
    """Update edge after a discussion. Returns True if edge survives.

    Drop rule: either agent votes 'move' → edge removed.
    Strengthen: both vote 'continue' → strength += STRENGTH_BONUS.
    """
    if vote_a == "move" or vote_b == "move":
        state.graph.remove_edge(agent_a, agent_b)
        return False
    edge = state.graph[agent_a][agent_b]["data"]
    edge.strength = min(STRENGTH_CAP, edge.strength + STRENGTH_BONUS)
    edge.rounds_active += 1
    return True

def ensure_connectivity(state) -> None:
    """Reconnect any agent with degree 0."""
    for name in list(state.agents):
        if state.graph.degree(name) == 0:
            reconnect_isolated(state, name)
```

---

**`network/logger.py`** — structured output
- `SimulationLogger(run_id=None)` creates `logs/run_<timestamp>/`
- `log_discussion(round, a, b, result)` → appends to `events.jsonl`
- `log_edge_event(round, type, a, b, **kwargs)` → appends to `events.jsonl`
- `log_reflection(round, agent, insights)` → appends to `events.jsonl`
- `snapshot_network(state, extra_metrics=None)` → writes `network_rounds/round_NNNN.json`
  - Always includes: edge list, density, n_components, avg_degree
  - `extra_metrics` dict is merged in if provided — this is where Banisch metrics (n_d, σ², etc.) slot in without changing the logger

---

**`network/opinion.py`** — **EXTENSION POINT, not implemented now**

This file does not exist yet. When adding Banisch-style opinion tracking, create it here:
```python
# network/opinion.py  (future)
# - AgentOpinionState dataclass: q_pos, q_neg, public_opinion, conviction
# - init_opinion_states(agents, topic) -> dict
# - update_opinion_states(opinion_states, discussion_result, alpha)
# - compute_metrics(graph, opinion_states) -> dict  # n_d, σ², congruent_links_pct
```
Wire it into `main_network.py` by passing `extra_metrics=compute_metrics(...)` to `snapshot_network()`.
No other file needs to change.

---

### `main_network.py` — new entry point

```python
TOPIC = (
    "Deutschland nimmt jedes Jahr Hunderttausende Migranten auf. "
    "Sollte Deutschland die Grenzen für Nicht-EU-Ausländer dauerhaft schließen?"
)

def main():
    llm = OllamaLLM(...)
    logger = SimulationLogger()

    personas = sample_personas(NUM_AGENTS_NETWORK, llm)
    agents = {p["name"]: Agent(p["name"], p["persona"], llm) for p in personas}

    G = nx.watts_strogatz_graph(len(agents), k=INITIAL_GRAPH_K, p=INITIAL_GRAPH_P)
    name_list = list(agents.keys())
    G = nx.relabel_nodes(G, {i: name_list[i] for i in range(len(name_list))})
    for u, v in G.edges():
        G[u][v]["data"] = EdgeData()

    state = NetworkState(agents=agents, graph=G, max_rounds=NETWORK_MAX_ROUNDS)
    logger.snapshot_network(state)

    for round_n in range(1, NETWORK_MAX_ROUNDS + 1):
        state.round = round_n
        pairings = compute_pairings(state)

        for agent_a_name, agent_b_name in pairings:
            result = run_discussion(agents[agent_a_name], agents[agent_b_name], TOPIC)
            logger.log_discussion(round_n, agent_a_name, agent_b_name, result)
            survived = update_edge(state, agent_a_name, agent_b_name,
                                   result["vote_a"], result["vote_b"])
            logger.log_edge_event(round_n,
                                  "edge_maintained" if survived else "edge_dropped",
                                  agent_a_name, agent_b_name)

        ensure_connectivity(state)

        if round_n % REFLECT_EVERY == 0:
            for agent in agents.values():
                agent.reflect()

        # extra_metrics=None now; pass compute_metrics(...) here when opinion.py is added
        logger.snapshot_network(state, extra_metrics=None)
```

---

## Edge-Drop Rule

**Either agent votes "move" → edge dropped.** (Asymmetric veto — realistic, and produces richer turnover for analysis.) Edge strength grows only when both vote "continue" (+0.2, capped at 3.0). Strength influences pairing priority via the matching weight but is not itself a drop threshold.

---

## Initial Topology

**Watts-Strogatz small-world graph** (`k=4, p=0.3`): interpretable, well-studied, and produces the community clustering that Banisch & Olbrich identify as the structural precondition for polarization — making it a theoretically motivated baseline even without the opinion model active.

---

## Extension Points Summary

When adding the Banisch opinion state model later:

| Where | What to add | What to change |
|---|---|---|
| `network/opinion.py` | Create from scratch: `AgentOpinionState`, init, update, metrics | Nothing else needed |
| `network/discussion.py` | Uncomment `reward_a/reward_b` fields in return dict | 2 lines |
| `main_network.py` | Import + call `init_opinion_states()`, pass `extra_metrics` to `snapshot_network()` | ~5 lines |
| `network/state.py` | Uncomment `opinion_states` field | 1 line |

No existing logic needs modification.

---

## Extension Perspective: Banisch & Olbrich Opinion State

Banisch & Olbrich (2019) model opinion dynamics as a Q-learning process: each agent holds two private Q-values — `q_pos` and `q_neg` — representing how rewarding it has been to express a pro or contra stance in past interactions. The difference `ΔQ_i = q_pos − q_neg` is the agent's *conviction*, and the publicly expressed opinion is simply `argmax(q_pos, q_neg)`. After each interaction, the Q-value for the expressed opinion is updated toward the social reward received: +1 if the interaction partner agreed, −1 if they disagreed.

**Mapping to this simulation.** The existing `evaluate()` call already produces exactly the reward signal Banisch's model requires. An agent voting `"weiter"` has experienced social validation — the equivalent of agreement (r = +1). An agent voting `"wechseln"` has found little resonance — the equivalent of disagreement (r = −1). The only missing pieces are (a) a numeric opinion state per agent and (b) the Q-update rule applied after each discussion.

**What `network/opinion.py` would contain.** A dataclass `AgentOpinionState` stores `q_pos`, `q_neg` (initialised uniformly in [−0.5, 0.5]), and exposes `public_opinion: int` (+1 or −1) and `conviction: float` (ΔQ). A function `init_opinion_states(agents, topic)` bootstraps this by making one LLM call per agent — a forced-choice question on the discussion topic (e.g. *"Antworte ausschließlich mit 'ja' oder 'nein'"*) — to set the initial public opinion, then draws random Q-values consistent with that sign. After each discussion, `update_opinion_states(opinion_states, discussion_result, alpha=0.05)` maps each agent's vote to ±1 and applies the standard Q-learning update:

```
Q(expressed_opinion) ← (1 − α) · Q(expressed_opinion) + α · reward
```

A second LLM call re-extracts the public opinion after the update (since conviction may have shifted enough to flip `argmax`), or — more efficiently — the flip is inferred purely from whether `ΔQ` crossed zero, avoiding an extra LLM call entirely.

**Polarization metrics.** With opinion states available, `compute_metrics(graph, opinion_states)` can compute the two key measures from Jacob & Banisch (2023):

- **Total effective antagonism** `n_d = (1/|E|) Σ_{(i,j)∈E} ½(1 − o_i · o_j)` — the fraction of edges connecting agents with opposing opinions. `n_d → 0` signals *structural polarization* (opinion clusters align with network communities); `n_d ≈ 0.5` signals *un-structural polarization* (opposing opinions co-exist within communities).
- **Dispersion** `σ² = Var(ΔQ_i)` — variance of conviction across the population. High dispersion with low n_d is the hallmark of structural echo chambers forming.

Both metrics are returned as a dict and passed to `snapshot_network(state, extra_metrics=...)`, so they appear in every round's JSON snapshot without any change to the logger or the core simulation loop.

**What does not change.** The discussion runner, edge lifecycle, matching logic, and logger are all untouched. The network co-evolution mechanism — agents dropping edges when they feel socially unrewarded — becomes doubly grounded: it drives both the Q-update (weakening conviction) and the structural change (severing the relationship). This is precisely the co-evolution extension Banisch & Olbrich (2019) proposed as future work in their conclusion.

---

## Implementation Order

1. `config.py` — 6 new constants
2. `requirements.txt` — add `networkx`
3. `network/state.py`
4. `network/edges.py`
5. `network/logger.py`
6. `network/discussion.py`
7. `network/matching.py`
8. `main_network.py`

---

## Verification

1. Set `NUM_AGENTS_NETWORK=4`, `NETWORK_MAX_ROUNDS=3`, `DISCUSSION_TURNS=2`
2. Run `python main_network.py` → 2 pairs/round, 3 round snapshots in `logs/`
3. Inspect `events.jsonl` — verify turns, votes, edge events present
4. Inspect `network_rounds/round_0003.json` — verify edge list, density, component count
5. Force `evaluate()` to always return `"move"` → verify all edges drop and reconnect fires
6. Scale to `NUM_AGENTS_NETWORK=20`, 5 rounds; verify no crashes, memory per agent grows
