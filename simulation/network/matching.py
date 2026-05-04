"""
Agent pairing and network reconnection for the network simulation.

Each round, agents are matched into pairs for pairwise discussions using a
single step:

**Max-weight matching over existing edges.**  NetworkX's implementation of
Edmonds' blossom algorithm finds the maximum-weight matching on the current
edge set, using the sum of ``EdgeData.strengths`` (both agents' internal
valuations) as the weight.  This ensures that mutually valued relationships
are given priority.

Agents left unmatched (because the graph is sparse or disconnected) pause
that discussion round.  Their edge strengths and memories are unchanged.

Odd agent counts are handled by rotating one agent out each round.

Reconnection
------------
After each round's edge updates, agents whose degree has dropped to zero
are reconnected by ``ensure_connectivity``.  A uniformly random partner is
chosen from all other agents and a new introductory edge at ``strength=0.5``
is added.
"""

from __future__ import annotations

import random

import networkx as nx

from network.state import EdgeData, NetworkState


def compute_pairings(state: NetworkState) -> list[tuple[str, str]]:
    """Compute agent pairings for the current round.

    Agents are matched over their existing edges using a max-weight matching
    algorithm (Edmonds' blossom).  Agents left unmatched pause the round —
    their edge strengths and memories are unchanged.

    If the total number of agents is odd, one agent is rotated out each
    round and recorded in ``state.idle_agent``.  The sit-out index cycles
    deterministically via ``state.round % len(agents)`` so that no single
    agent is systematically excluded.

    Parameters
    ----------
    state:
        The current ``NetworkState``.  ``state.idle_agent`` may be mutated.

    Returns
    -------
    list[tuple[str, str]]
        A list of ``(agent_a_name, agent_b_name)`` pairs for agents that
        will hold a discussion this round.  Unmatched agents are omitted.
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
        H.add_edge(u, v, weight=sum(state.graph[u][v]["data"].strengths.values()))

    matched = nx.max_weight_matching(H, maxcardinality=True)
    return [tuple(pair) for pair in matched]


def reconnect_isolated(state: NetworkState, agent_name: str) -> None:
    """Connect a degree-zero agent to a uniformly random other agent.

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
    candidates = [a for a in state.agents if a != agent_name]

    if not candidates:
        return

    partner = random.choice(candidates)
    state.graph.add_edge(agent_name, partner, data=EdgeData(strengths={agent_name: 0.5, partner: 0.5}))


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
