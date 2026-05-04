"""
Edge lifecycle management for the network simulation.

Provides ``update_edge``, which applies the social-feedback score from a
pairwise discussion to the corresponding edge: strengthening it when agents
agree (positive score) or weakening it when they disagree (negative score).
An edge is removed as soon as its strength falls at or below ``STRENGTH_FLOOR``.

This replaces the former binary vote ("continue" / "move") with a continuous
reward signal.  In Banisch & Olbrich (2019) the reward r = o_i · o_j is binary
(±1) because public opinions are discrete.  The continuous score used here is a
deliberate extension: partial agreement yields a fractional adjustment rather
than a hard switch, allowing for richer edge dynamics.

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
    """Adjust edge strength based on the social-feedback scores from both agents.

    The combined score — the average of both agents' concordance ratings — is
    multiplied by ``STRENGTH_DELTA`` and added to the current edge strength.
    Agreement (positive combined score) strengthens the edge; disagreement
    (negative combined score) weakens it.  The edge is removed immediately if
    strength falls at or below ``STRENGTH_FLOOR``.

    Strength is capped at ``STRENGTH_CAP`` to prevent runaway accumulation.
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
    combined = (score_a + score_b) / 2.0

    edge = state.graph[agent_a][agent_b]["data"]
    edge.strength += combined * STRENGTH_DELTA
    edge.strength = max(0.0, min(STRENGTH_CAP, edge.strength))

    if edge.strength <= STRENGTH_FLOOR:
        state.graph.remove_edge(agent_a, agent_b)
        return False

    edge.rounds_active += 1
    return True
