"""
Data classes for the network simulation state.

NetworkState is the single mutable object passed through every round of the
simulation.  It holds the agent registry, the NetworkX graph, agent Q-value
opinion states, and round bookkeeping.  EdgeData is stored as a per-edge
attribute under the key "data" on every edge of the graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx


@dataclass
class EdgeData:
    """Metadata stored on every edge of the communication network.

    Attributes
    ----------
    strengths:
        Per-agent internal valuation of the relationship, keyed by agent
        name.  Each agent's value is updated independently after each
        discussion by their own concordance score × ``STRENGTH_DELTA``.
        The edge is severed as soon as *either* agent's value falls to or
        below ``STRENGTH_FLOOR``.  The matching weight is the sum of both
        values so that well-established (mutually valued) relationships are
        preferred in pairing.
    rounds_active:
        Counter incremented after every discussion in which the edge
        survives.  Useful for post-hoc analysis of relationship duration.
    """

    strengths: dict = field(default_factory=dict)  # {agent_name: float}
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
        Name of the agent sitting out the current round when odd-count
        symmetric pairing is used (``compute_pairings()``).  Always
        ``None`` in the default asymmetric interaction mode.
    opinion_states:
        Mapping from agent name to ``AgentOpinionState`` (Q-values and
        expressed opinion).  Populated by ``network/opinion.py`` at the
        start of the simulation and updated after every discussion.
    """

    agents: dict
    graph: nx.Graph
    round: int = 0
    max_rounds: int = 30
    idle_agent: str | None = None
    opinion_states: dict = field(default_factory=dict)
    # maps agent name -> AgentOpinionState (q_pos, q_neg, expressed_opinion)
