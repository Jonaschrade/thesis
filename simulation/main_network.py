"""
Network simulation entry point.

Runs the full network simulation: a Watts-Strogatz small-world graph of
agents who hold pairwise discussions each round and vote — using the existing
social-feedback ``evaluate()`` prompt — on whether to maintain or sever their
communication edge.

Round structure
---------------
1. Compute pairings via max-weight matching over existing edges (with random
   fallback for unmatched agents).
2. Each pair runs ``DISCUSSION_TURNS`` LLM turns, then both agents cast a
   social-reward vote ("weiter" / "wechseln").
3. Edges are strengthened (mutual "continue") or removed (any "move").
4. Isolated agents are reconnected before the next round.
5. Every ``REFLECT_EVERY`` rounds all agents reflect on their recent memories.
6. The network state is snapshotted to ``logs/run_<timestamp>/``.

Configuration
-------------
All tunable parameters live in ``config.py``.  For a quick smoke test, set::

    NUM_AGENTS_NETWORK = 4
    NETWORK_MAX_ROUNDS = 3
    DISCUSSION_TURNS   = 2

To change the discussion topic, edit the ``TOPIC`` constant below.

Extension point
---------------
To activate Banisch & Olbrich opinion tracking, import and wire up
``network/opinion.py`` here (see the inline comments marked EXTENSION POINT).
No other file needs to change.
"""

import networkx as nx
from langchain_ollama import OllamaLLM

from agents.agent import Agent
from agents.personas import sample_personas
from config import (
    INITIAL_GRAPH_K,
    INITIAL_GRAPH_P,
    LLM_MODEL,
    NETWORK_MAX_ROUNDS,
    NUM_AGENTS_NETWORK,
    OLLAMA_HOST,
    REFLECT_EVERY,
)
from network.discussion import run_discussion
from network.edges import update_edge
from network.logger import SimulationLogger
from network.matching import compute_pairings, ensure_connectivity
from network.state import EdgeData, NetworkState

# ── Discussion topic ────────────────────────────────────────────────────────
TOPIC = (
    "Deutschland nimmt jedes Jahr Hunderttausende Migranten auf – "
    "doch Integration scheitert immer wieder an Sprache, Arbeit und Kultur. "
    "Sollte Deutschland die Grenzen für Nicht-EU-Ausländer dauerhaft schließen?"
)


def _build_initial_graph(agent_names: list[str]) -> nx.Graph:
    """Create a Watts-Strogatz small-world graph over the agent name list.

    Nodes are relabelled from integers to agent names so that every edge
    attribute and algorithm operates on human-readable identifiers.  Every
    edge is initialised with a fresh ``EdgeData`` instance at the default
    strength of 1.0.

    Parameters
    ----------
    agent_names:
        Ordered list of unique agent names.  The graph will have exactly
        ``len(agent_names)`` nodes.

    Returns
    -------
    nx.Graph
        A connected (or near-connected) small-world graph with ``"data"``
        edge attributes.
    """
    n = len(agent_names)
    G = nx.watts_strogatz_graph(n, k=INITIAL_GRAPH_K, p=INITIAL_GRAPH_P)
    G = nx.relabel_nodes(G, {i: agent_names[i] for i in range(n)})
    for u, v in G.edges():
        G[u][v]["data"] = EdgeData()
    return G


def main() -> None:
    """Run the full network simulation."""
    llm = OllamaLLM(model=LLM_MODEL, base_url=f"http://{OLLAMA_HOST}")
    logger = SimulationLogger()

    # ── Agent initialisation ─────────────────────────────────────────────
    print(f"\nSampling {NUM_AGENTS_NETWORK} personas...")
    personas = sample_personas(NUM_AGENTS_NETWORK, llm)
    agents: dict[str, Agent] = {
        p["name"]: Agent(name=p["name"], persona=p["persona"], llm=llm)
        for p in personas
    }

    print(f"\n{'━' * 60}")
    print("Participants")
    for name, agent in agents.items():
        print(f"  {name}: {agent.persona}")
    print(f"{'━' * 60}\n")

    # ── Network initialisation ───────────────────────────────────────────
    G = _build_initial_graph(list(agents.keys()))
    state = NetworkState(agents=agents, graph=G, max_rounds=NETWORK_MAX_ROUNDS)

    # EXTENSION POINT — Banisch opinion initialisation:
    # from network.opinion import init_opinion_states
    # state.opinion_states = init_opinion_states(agents, TOPIC)

    logger.snapshot_network(state)   # round-0 baseline
    print(f"Initial graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges "
          f"(Watts-Strogatz k={INITIAL_GRAPH_K}, p={INITIAL_GRAPH_P})\n")

    # ── Main simulation loop ─────────────────────────────────────────────
    for round_n in range(1, NETWORK_MAX_ROUNDS + 1):
        state.round = round_n
        print(f"\n{'━' * 60}")
        print(f"Round {round_n} / {NETWORK_MAX_ROUNDS}  "
              f"│  edges: {state.graph.number_of_edges()}  "
              f"│  components: {nx.number_connected_components(state.graph)}")
        if state.idle_agent:
            print(f"  (sitting out: {state.idle_agent})")
        print(f"{'━' * 60}")

        pairings = compute_pairings(state)

        for agent_a_name, agent_b_name in pairings:
            print(f"\n  ▶ {agent_a_name} ↔ {agent_b_name}")
            result = run_discussion(
                agents[agent_a_name],
                agents[agent_b_name],
                TOPIC,
            )
            logger.log_discussion(round_n, agent_a_name, agent_b_name, result)

            survived = update_edge(
                state,
                agent_a_name,
                agent_b_name,
                result["vote_a"],
                result["vote_b"],
            )

            event_type = "edge_maintained" if survived else "edge_dropped"
            logger.log_edge_event(round_n, event_type, agent_a_name, agent_b_name)

            status = "✔ maintained" if survived else "✘ dropped"
            print(f"    {agent_a_name} → {result['vote_a']}  |  "
                  f"{agent_b_name} → {result['vote_b']}  |  edge {status}")

        # ── Reconnect isolated agents ────────────────────────────────────
        ensure_connectivity(state)

        # ── Reflection phase ─────────────────────────────────────────────
        if round_n % REFLECT_EVERY == 0:
            print(f"\n── Reflection phase (round {round_n}) ──")
            for agent in agents.values():
                agent.reflect()
                logger.log_reflection(round_n, agent.name)

        # EXTENSION POINT — Banisch opinion update:
        # from network.opinion import update_opinion_states, compute_metrics
        # update_opinion_states(state.opinion_states, last_results, alpha=0.05)
        # extra = compute_metrics(state.graph, state.opinion_states)
        # logger.snapshot_network(state, extra_metrics=extra)

        logger.snapshot_network(state)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n\n{'━' * 60}")
    print("Simulation complete")
    print(f"  Rounds run    : {NETWORK_MAX_ROUNDS}")
    print(f"  Final edges   : {state.graph.number_of_edges()}")
    print(f"  Components    : {nx.number_connected_components(state.graph)}")
    print(f"  Logs written  : {logger.run_dir}")
    print(f"{'━' * 60}\n")


if __name__ == "__main__":
    main()
