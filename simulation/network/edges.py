"""
Edge lifecycle management for the network simulation.

Provides ``update_edge``, which applies the social-reward vote from a
pairwise discussion to the corresponding edge: either strengthening it
(both agents voted "continue") or removing it (either agent voted "move").

The asymmetric drop rule — either agent can veto continuation — reflects the
social reality that a relationship only persists if both parties are willing.
It also produces richer network turnover, which is desirable for studying
dynamic polarisation.

Note
----
``ensure_connectivity`` and ``reconnect_isolated`` live in
``network/matching.py`` because reconnection is topologically a pairing
operation.  Import them from there rather than here.
"""

from __future__ import annotations

from config import STRENGTH_CAP
from network.state import NetworkState


def update_edge(
    state: NetworkState,
    agent_a: str,
    agent_b: str,
    vote_a: str,
    vote_b: str,
) -> bool:
    """Apply the post-discussion votes to the edge between two agents.

    Drop rule (asymmetric veto)
        If *either* agent voted ``"move"``, the edge is removed from the
        graph immediately.  The agent may be reconnected later by
        ``ensure_connectivity``.

    Strengthen rule
        If *both* agents voted ``"continue"``, ``EdgeData.strength`` is
        incremented by a fixed bonus (0.2) up to ``STRENGTH_CAP``, and
        ``EdgeData.rounds_active`` is incremented.

    Parameters
    ----------
    state:
        The current ``NetworkState``.  The graph is mutated in place.
    agent_a:
        Name of the first agent in the pair.
    agent_b:
        Name of the second agent in the pair.
    vote_a:
        Social-reward vote from ``agent_a.evaluate()``.  Either
        ``"continue"`` or ``"move"``.
    vote_b:
        Social-reward vote from ``agent_b.evaluate()``.  Same format.

    Returns
    -------
    bool
        ``True`` if the edge survived, ``False`` if it was removed.
    """
    _STRENGTH_BONUS = 0.2

    if vote_a == "move" or vote_b == "move":
        state.graph.remove_edge(agent_a, agent_b)
        return False

    edge = state.graph[agent_a][agent_b]["data"]
    edge.strength = min(STRENGTH_CAP, edge.strength + _STRENGTH_BONUS)
    edge.rounds_active += 1
    return True
