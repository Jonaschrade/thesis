"""
Edge lifecycle management for the network simulation.

Provides ``update_edge``, which records the reward each agent received from
the current interaction and derives an evaluation signal from the rolling mean
of their per-edge reward history (window length ``REWARD_WINDOW_M``).  The
signal drives a STRENGTH_DELTA step on that agent's internal edge valuation;
the edge is severed as soon as *either* agent's valuation falls to or below
``STRENGTH_FLOOR``.

Asymmetric vs. symmetric mode
------------------------------
Pass only ``reward_a`` (expresser) to run in asymmetric mode: the responder's
history stays empty and contributes a neutral signal of 0.0, so only the
expresser's accumulated experience can trigger a drop.  Pass both ``reward_a``
and ``reward_b`` to activate symmetric evaluation once that path is wired up
in the caller.

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
    reward_a: float | None = None,
    reward_b: float | None = None,
) -> bool:
    """Record interaction rewards and update each agent's edge valuation.

    Appends the supplied reward(s) to the corresponding agent's rolling
    history window on this edge, then derives an evaluation signal as the
    mean reward over that window.  ``None`` means the agent does not receive
    a reward this interaction (asymmetric mode); their history is unchanged
    and the resulting signal is 0.0 (neutral — no valuation change).

    Each agent's valuation is updated independently:
        strength += mean_reward × STRENGTH_DELTA
    Both values are clamped to [0, STRENGTH_CAP].  The edge is removed
    immediately if *either* agent's valuation falls to or below
    ``STRENGTH_FLOOR`` — one agent's accumulated dissatisfaction is
    sufficient to sever the channel.

    ``EdgeData.rounds_active`` is incremented whenever the edge survives.

    Parameters
    ----------
    state:
        The current ``NetworkState``.  The graph is mutated in place.
    agent_a:
        Name of the expresser (first agent in the pair).
    agent_b:
        Name of the responder (second agent in the pair).
    reward_a:
        Reward received by ``agent_a`` this interaction ∈ [−1, 1], or
        ``None`` to skip (asymmetric mode).
    reward_b:
        Reward received by ``agent_b`` this interaction ∈ [−1, 1], or
        ``None`` to skip (asymmetric mode, current default).

    Returns
    -------
    bool
        ``True`` if the edge survived, ``False`` if it was removed.
    """
    edge = state.graph[agent_a][agent_b]["data"]

    if reward_a is not None:
        edge.reward_history[agent_a].append(reward_a)
    if reward_b is not None:
        edge.reward_history[agent_b].append(reward_b)

    def _signal(history) -> float:
        return sum(history) / len(history) if history else 0.0

    signal_a = _signal(edge.reward_history[agent_a])
    signal_b = _signal(edge.reward_history[agent_b])

    edge.strengths[agent_a] = max(0.0, min(STRENGTH_CAP,
        edge.strengths[agent_a] + signal_a * STRENGTH_DELTA))
    edge.strengths[agent_b] = max(0.0, min(STRENGTH_CAP,
        edge.strengths[agent_b] + signal_b * STRENGTH_DELTA))

    if edge.strengths[agent_a] <= STRENGTH_FLOOR or edge.strengths[agent_b] <= STRENGTH_FLOOR:
        state.graph.remove_edge(agent_a, agent_b)
        return False

    edge.rounds_active += 1
    return True
