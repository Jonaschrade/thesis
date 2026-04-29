"""
Agent pairing and network reconnection for the network simulation.

Each round, agents are matched into N/2 pairs for pairwise discussions.
The pairing algorithm has two steps:

1. **Max-weight matching over existing edges.**  NetworkX's implementation
   of Edmonds' blossom algorithm finds the maximum-weight matching on the
   current edge set, using ``EdgeData.strength`` as the weight.  This
   ensures that well-established relationships are given priority.

2. **Random fallback for unmatched agents.**  Agents left unmatched after
   step 1 (because the graph is sparse or disconnected) are paired at
   random.  A new introductory edge at ``strength=0.5`` is added to the
   graph for each such pair so that the next round's matching can consider
   it.

Odd agent counts are handled by rotating one agent out each round.

Reconnection
------------
After each round's edge updates, agents whose degree has dropped to zero
are reconnected by ``ensure_connectivity``.  The reconnection strategy
prefers friends-of-friends (second-degree neighbours in the current graph)
to preserve the small-world clustering structure; it falls back to a
uniformly random non-neighbour if no second-degree candidate is available.
"""

from __future__ import annotations

import random

import networkx as nx

from network.state import EdgeData, NetworkState


def compute_pairings(state: NetworkState) -> list[tuple[str, str]]:
    """Compute agent pairings for the current round.

    Agents are matched preferentially over their existing edges using a
    max-weight matching algorithm (Edmonds' blossom).  Any agents left
    unmatched receive a randomly chosen partner and a new introductory edge.

    If the total number of agents is odd, one agent is rotated out each
    round and recorded in ``state.idle_agent``.  The sit-out index cycles
    deterministically via ``state.round % len(agents)`` so that no single
    agent is systematically excluded.

    Parameters
    ----------
    state:
        The current ``NetworkState``.  ``state.graph`` and
        ``state.idle_agent`` may be mutated.

    Returns
    -------
    list[tuple[str, str]]
        A list of ``(agent_a_name, agent_b_name)`` pairs covering every
        active agent exactly once.
    """
    # Use a sorted list for deterministic sit-out rotation
    all_agents = sorted(state.agents.keys())

    if len(all_agents) % 2 == 1:
        sit_out_idx = state.round % len(all_agents)
        state.idle_agent = all_agents[sit_out_idx]
        active = [a for a in all_agents if a != state.idle_agent]
    else:
        state.idle_agent = None
        active = all_agents

    # Build a helper graph with 'weight' = edge strength for the matching algo
    H = nx.Graph()
    H.add_nodes_from(active)
    subgraph = state.graph.subgraph(active)
    for u, v in subgraph.edges():
        H.add_edge(u, v, weight=state.graph[u][v]["data"].strength)

    # Step 1: max-weight matching over existing edges
    matched = nx.max_weight_matching(H, maxcardinality=True)
    matched_pairs = [tuple(pair) for pair in matched]
    matched_agents = {a for pair in matched_pairs for a in pair}

    # Step 2: pair leftover agents randomly; add introductory edges
    unmatched = [a for a in active if a not in matched_agents]
    random.shuffle(unmatched)
    new_pairs: list[tuple[str, str]] = []
    for i in range(0, len(unmatched) - 1, 2):
        a, b = unmatched[i], unmatched[i + 1]
        if not state.graph.has_edge(a, b):
            state.graph.add_edge(a, b, data=EdgeData(strength=0.5, rounds_active=0))
        new_pairs.append((a, b))

    return matched_pairs + new_pairs


def reconnect_isolated(state: NetworkState, agent_name: str) -> None:
    """Connect a degree-zero agent to a new partner.

    The reconnection strategy is:

    1. **Friends-of-friends** — collect second-degree neighbours of the
       agent's *current* neighbours (i.e. neighbours' neighbours).  If any
       of those are not yet directly connected to ``agent_name``, one is
       chosen at random and a new introductory edge is added.
    2. **Random fallback** — if no FoF candidate exists (e.g. the agent is
       completely isolated), a uniformly random agent that is not already a
       direct neighbour is chosen instead.

    The new edge is created at ``strength=0.5`` (neutral introductory level)
    so it does not immediately dominate the matching weight of established
    relationships.

    Parameters
    ----------
    state:
        The current ``NetworkState``.  ``state.graph`` is mutated in place.
    agent_name:
        Name of the agent to reconnect.
    """
    current_neighbours = set(state.graph.neighbors(agent_name))

    # Collect friends-of-friends not already adjacent
    fof: set[str] = set()
    for nb in current_neighbours:
        fof.update(state.graph.neighbors(nb))
    fof -= current_neighbours
    fof.discard(agent_name)

    # Candidates not yet directly connected
    all_others = set(state.agents.keys()) - {agent_name} - current_neighbours
    candidates = list(fof & all_others) or list(all_others)

    if not candidates:
        return  # every other agent is already a neighbour — nothing to do

    partner = random.choice(candidates)
    state.graph.add_edge(agent_name, partner, data=EdgeData(strength=0.5, rounds_active=0))


def ensure_connectivity(state: NetworkState) -> None:
    """Reconnect any agent whose degree has dropped to zero.

    Called once per round after all edge updates.  Iterates over every
    agent and invokes ``reconnect_isolated`` for those with no remaining
    edges, ensuring no agent is permanently excluded from future rounds.

    Parameters
    ----------
    state:
        The current ``NetworkState``.  ``state.graph`` may be mutated.
    """
    for name in list(state.agents.keys()):
        if state.graph.degree(name) == 0:
            reconnect_isolated(state, name)
