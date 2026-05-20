"""
Edge lifecycle management for the network simulation.

Provides ``update_edge``, which applies a concordance score from a pairwise
discussion to each agent's internal valuation of the edge.  Each agent
maintains their own value independently; the edge is severed as soon as
*either* agent's value falls to or below ``STRENGTH_FLOOR``.

This module is only active when ``GRAPH_DYNAMIC = True`` in config.  In the
default fixed-graph mode (main SFT experiments), ``update_edge`` is not
called and the graph structure set at initialisation is preserved for the
full run.

Note
----
``ensure_connectivity`` and ``reconnect_isolated`` live in
``network/matching.py`` because reconnection is topologically a pairing
operation.  Import them from there rather than here.
"""

from __future__ import annotations

from config import STRENGTH_CAP, STRENGTH_DELTA, STRENGTH_FLOOR
from network.state import NetworkState


def update_edge(
    state: NetworkState,
    agent_a: str,
    agent_b: str,
    score_a: float,
    score_b: float,
) -> bool:
    """Adjust each agent's internal edge valuation from their own concordance score.

    Each agent's value is updated independently: ``score_a`` adjusts
    ``agent_a``'s valuation and ``score_b`` adjusts ``agent_b``'s.  Both
    values are capped at ``STRENGTH_CAP``.  The edge is removed immediately
    if *either* agent's value falls to or below ``STRENGTH_FLOOR`` — one
    agent's dissatisfaction is sufficient to sever the channel.

    ``EdgeData.rounds_active`` is incremented whenever the edge survives.

    Parameters
    ----------
    state:
        The current ``NetworkState``.  The graph is mutated in place.
    agent_a:
        Name of the first agent in the pair.
    agent_b:
        Name of the second agent in the pair.
    score_a:
        Concordance score from ``agent_a.evaluate()`` in [−1.0, 1.0].
        Positive = agreement, negative = disagreement.
    score_b:
        Concordance score from ``agent_b.evaluate()``.  Same range.

    Returns
    -------
    bool
        ``True`` if the edge survived, ``False`` if it was removed.
    """
    edge = state.graph[agent_a][agent_b]["data"]

    edge.strengths[agent_a] = max(0.0, min(STRENGTH_CAP,
        edge.strengths[agent_a] + score_a * STRENGTH_DELTA))
    edge.strengths[agent_b] = max(0.0, min(STRENGTH_CAP,
        edge.strengths[agent_b] + score_b * STRENGTH_DELTA))

    if edge.strengths[agent_a] <= STRENGTH_FLOOR or edge.strengths[agent_b] <= STRENGTH_FLOOR:
        state.graph.remove_edge(agent_a, agent_b)
        return False

    edge.rounds_active += 1
    return True
