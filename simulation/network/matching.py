"""
Agent partner selection and network reconnection for the network simulation.

The primary entry point is ``select_responder``, which implements the
asymmetric interaction rule from Jacob & Banisch (2023): given an expresser
and its graph neighbours, it draws a responder with an optional homophily
bias h.  At h = 0 the draw is uniform (replicating Banisch & Olbrich 2019);
at h > 0, neighbours whose Q-gap is similar to the expresser's receive higher
weight, representing the tendency to seek out like-minded interlocutors.

``compute_pairings`` and ``ensure_connectivity`` are retained for the
``GRAPH_DYNAMIC = True`` extension (endogenous tie rewiring).
"""

from __future__ import annotations

import math
import random
from collections import deque

import networkx as nx

from config import REWARD_WINDOW_M
from network.state import EdgeData, NetworkState


def select_responder(
    expresser: str,
    neighbours: list[str],
    opinion_states: dict,
    h: float,
) -> str:
    """Draw a responder from the expresser's neighbours with homophily bias h.

    Called once per interaction in the main loop of ``main_network.py`` and
    ``main_pairwise.py`` to implement the asymmetric SFT interaction rule.

    Implements the partner-selection mechanism from Jacob & Banisch (2023):
    interaction probability is weighted by conviction similarity, controlled
    by h.  Keeping this as a standalone function means the virtual-worlds
    multi-platform extension can swap in a different draw (e.g. cross-platform
    neighbours) without touching the interaction loop.

    Weight formula:  w_j = exp(−h · |Δq_i − Δq_j|)

    where Δq = q_pos − q_neg is each agent's signed conviction.

    Parameters
    ----------
    expresser:
        Name of the agent expressing an opinion this interaction.
    neighbours:
        Adjacency list of the expresser (already filtered to non-empty).
    opinion_states:
        Mapping of agent name to ``AgentOpinionState``.
    h:
        Homophily parameter ≥ 0.
        h = 0 → uniform draw (Banisch & Olbrich 2019).
        h > 0 → neighbours with similar q_gap weighted higher.

    Returns
    -------
    str
        Name of the selected responder.
    """
    if h == 0.0:
        return random.choice(neighbours)

    expresser_gap = opinion_states[expresser].q_gap
    weights = [
        math.exp(-h * abs(expresser_gap - opinion_states[n].q_gap))
        for n in neighbours
    ]
    total = sum(weights)
    r = random.random() * total
    cumulative = 0.0
    for name, w in zip(neighbours, weights):
        cumulative += w
        if r <= cumulative:
            return name
    return neighbours[-1]


def compute_pairings(state: NetworkState) -> list[tuple[str, str]]:
    """Compute agent pairings for the current round.

    Not called in the default asymmetric interaction mode.  Reserved for the
    ``GRAPH_DYNAMIC = True`` extension if a symmetric global-matching round
    structure is needed alongside endogenous tie rewiring.

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
    state.graph.add_edge(agent_name, partner, data=EdgeData(
        strengths={agent_name: 0.5, partner: 0.5},
        reward_history={
            agent_name: deque(maxlen=REWARD_WINDOW_M),
            partner:     deque(maxlen=REWARD_WINDOW_M),
        },
    ))


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
