"""
Structured logging for the network simulation.

All simulation events are written to a timestamped run directory under
``logs/``.  Two output formats are produced:

* **``events.jsonl``** — one JSON object per line, covering every discussion,
  edge lifecycle event, and reflection trigger.  Suitable for streaming
  analysis or replay.

* **``network_rounds/round_NNNN.json``** — a full snapshot of the network
  state after each round: edge list with metadata, basic graph metrics, and
  an optional ``extra_metrics`` dict for extension data (e.g. Banisch
  polarisation measures).

Extension point
---------------
Pass ``extra_metrics=compute_metrics(state.graph, state.opinion_states)`` to
``snapshot_network`` once ``network/opinion.py`` is implemented.  The metrics
will be merged into the ``"metrics"`` section of the snapshot without any
change to this module.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import networkx as nx

from network.state import NetworkState


class SimulationLogger:
    """Writes structured simulation output to a per-run directory.

    Output directory layout::

        logs/
        └── run_<timestamp>/
            ├── events.jsonl          # one JSON record per event
            └── network_rounds/
                ├── round_0000.json   # pre-simulation snapshot
                ├── round_0001.json
                └── ...

    Parameters
    ----------
    run_id:
        Optional string identifier for the run directory name.  Defaults to
        the Unix timestamp at construction time.
    """

    def __init__(self, run_id: str | None = None) -> None:
        ts = run_id or str(int(time.time()))
        self.run_dir = Path("logs") / f"run_{ts}"
        self.rounds_dir = self.run_dir / "network_rounds"
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"

    # ── internal helpers ────────────────────────────────────────────────────

    def _write(self, record: dict) -> None:
        """Append a JSON record to the events log."""
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── public logging methods ──────────────────────────────────────────────

    def log_personas(self, agents: dict) -> None:
        """Write all agent personas to ``personas.json`` in the run directory.

        Called once at startup so that every run has a persistent record of
        which agents participated and what their persona descriptions were.
        This allows post-hoc correlation of agent behaviour with persona
        attributes even if the console output is lost.

        Output path: ``logs/run_<timestamp>/personas.json``

        Parameters
        ----------
        agents:
            Mapping of agent name to ``Agent`` instance, as used throughout
            the simulation.  Each entry's ``name`` and ``persona`` attributes
            are written; all other state is ignored.
        """
        records = [
            {"name": name, "persona": agent.persona}
            for name, agent in agents.items()
        ]
        path = self.run_dir / "personas.json"
        path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def log_discussion(
        self,
        round_n: int,
        agent_a: str,
        agent_b: str,
        result: dict,
    ) -> None:
        """Log the outcome of one pairwise discussion.

        Parameters
        ----------
        round_n:
            Simulation round in which the discussion took place.
        agent_a:
            Name of the first participant.
        agent_b:
            Name of the second participant.
        result:
            The dict returned by ``network.discussion.run_discussion``.
            Keys ``transcript``, ``topic_label``, ``score_a``, ``reason_a``,
            ``score_b``, ``reason_b`` are included verbatim.
        """
        self._write({
            "type":    "discussion",
            "round":   round_n,
            "agent_a": agent_a,
            "agent_b": agent_b,
            **result,
        })

    def log_edge_event(
        self,
        round_n: int,
        event_type: str,
        agent_a: str,
        agent_b: str,
        **kwargs,
    ) -> None:
        """Log an edge lifecycle event.

        Parameters
        ----------
        round_n:
            Current simulation round.
        event_type:
            One of ``"edge_maintained"``, ``"edge_dropped"``, or
            ``"edge_added"`` (for reconnection events).
        agent_a:
            Name of one endpoint.
        agent_b:
            Name of the other endpoint.
        **kwargs:
            Any additional fields to include in the record (e.g.
            ``strength=1.2``).
        """
        self._write({
            "type":    event_type,
            "round":   round_n,
            "agent_a": agent_a,
            "agent_b": agent_b,
            **kwargs,
        })

    def log_reflection(self, round_n: int, agent_name: str) -> None:
        """Log that reflection was triggered for an agent.

        The content of the reflection (insights) is printed to stdout by
        ``Agent.reflect()`` directly.  This event records only the trigger,
        allowing downstream analysis to correlate reflection rounds with
        network changes.

        Parameters
        ----------
        round_n:
            Current simulation round.
        agent_name:
            Name of the agent that reflected.
        """
        self._write({
            "type":  "reflection",
            "round": round_n,
            "agent": agent_name,
        })

    def snapshot_network(
        self,
        state: NetworkState,
        extra_metrics: dict | None = None,
    ) -> None:
        """Write a full network snapshot for the current round.

        The snapshot contains the complete edge list (with strength and
        activity count), basic graph metrics, and any additional metrics
        supplied by the caller.

        Output path: ``network_rounds/round_NNNN.json``

        Parameters
        ----------
        state:
            The current ``NetworkState``.
        extra_metrics:
            Optional dict of additional metrics to merge into the
            ``"metrics"`` section.  Reserved for Banisch opinion-state
            metrics (``n_d``, ``dispersion``, etc.) once
            ``network/opinion.py`` is implemented.  Pass ``None`` (default)
            to omit.
        """
        edges = [
            {
                "a":             u,
                "b":             v,
                "strengths":     state.graph[u][v]["data"].strengths,
                "rounds_active": state.graph[u][v]["data"].rounds_active,
            }
            for u, v in state.graph.edges()
        ]

        degrees = dict(state.graph.degree())
        n_agents = max(1, len(state.agents))

        metrics: dict = {
            "density":      nx.density(state.graph),
            "n_components": nx.number_connected_components(state.graph),
            "avg_degree":   sum(degrees.values()) / n_agents,
            "n_edges":      state.graph.number_of_edges(),
        }
        if extra_metrics:
            metrics.update(extra_metrics)

        snapshot = {
            "round":      state.round,
            "idle_agent": state.idle_agent,
            "nodes":      list(state.agents.keys()),
            "edges":      edges,
            "metrics":    metrics,
        }

        path = self.rounds_dir / f"round_{state.round:04d}.json"
        path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
