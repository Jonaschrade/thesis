"""
Data classes for the network simulation state.

NetworkState is the single mutable object passed through every round of the
simulation.  It holds the agent registry, the NetworkX graph, and round
bookkeeping.  EdgeData is stored as a per-edge attribute under the key "data"
on every edge of the graph.

Extension point
---------------
When adding Banisch & Olbrich (2019) opinion tracking, uncomment the
``opinion_states`` field and populate it via ``network/opinion.py``.  No other
file in this module needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx


@dataclass
class EdgeData:
    """Metadata stored on every edge of the communication network.

    Attributes
    ----------
    strength:
        A floating-point weight that grows each time both endpoints vote
        "continue" after a discussion (capped at ``STRENGTH_CAP`` from
        config).  Used as the matching weight during pairing so that
        well-established relationships are preferred over new ones.
    rounds_active:
        Counter incremented after every discussion in which the edge
        survives.  Useful for post-hoc analysis of relationship duration.
    """

    strength: float = 1.0
    rounds_active: int = 0


@dataclass
class NetworkState:
    """Complete mutable state of one network simulation run.

    Attributes
    ----------
    agents:
        Mapping from agent name (str) to the corresponding ``Agent``
        instance.  Treat as read-only after initialisation.
    graph:
        Undirected NetworkX graph whose nodes are agent names and whose
        edges carry an ``EdgeData`` instance under the attribute key
        ``"data"``.
    round:
        Current simulation round, incremented by ``main_network.py``
        before each round's processing begins.
    max_rounds:
        Total number of rounds to run.  The outer loop exits when
        ``round >= max_rounds``.
    idle_agent:
        Name of the agent sitting out the current round when the total
        agent count is odd.  ``None`` when the count is even.

    Extension point
    ---------------
    Banisch opinion state — uncomment when ``network/opinion.py`` is added::

        opinion_states: dict = field(default_factory=dict)
        # maps agent name -> AgentOpinionState (q_pos, q_neg, public_opinion)
    """

    agents: dict
    graph: nx.Graph
    round: int = 0
    max_rounds: int = 30
    idle_agent: str | None = None
    # opinion_states: dict = field(default_factory=dict)
