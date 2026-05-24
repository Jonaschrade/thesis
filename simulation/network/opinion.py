"""
Agent opinion state and Q-value dynamics for Social Feedback Theory (SFT).

Implements the core reinforcement-learning mechanism from Banisch & Olbrich
(2019): each agent holds Q-values over the two opinion stances (+1 / −1),
expresses a stance drawn by softmax with inverse temperature β, and updates
that Q-value after each interaction using a temporal-difference rule.

Public API
----------
AgentOpinionState             dataclass: Q-values + preferred_opinion (argmax) + q_gap
init_opinion_states()         initialise one state per agent
softmax_opinion()             stochastic expression draw (β-parameterised)
update_q_value()              apply TD update given the expressed stance
opinion_states_to_dict()      serialise for logging (preferred + optional expressed)
compute_polarization_metrics()  population-level SFT metrics

Terminology note
----------------
``preferred_opinion``  — deterministic argmax over Q-values; the stance the
                         agent *would prefer* given its current Q-values.
                         Used for metrics, logging, and console output.
``expressed opinion``  — the stance *actually drawn* by ``softmax_opinion()``
                         in a specific interaction.  Can differ from the
                         preferred opinion when conviction (q_gap) is low.
Both are recorded in the per-round snapshot; only the expressed stance drives
the Q-value TD update.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AgentOpinionState:
    """SFT internal state: Q-values over the two opinion stances.

    Attributes
    ----------
    q_pos:
        Q(+1) — perceived social value of expressing the pro-stance.
        Updated whenever the agent expresses +1 and receives feedback.
    q_neg:
        Q(−1) — perceived social value of expressing the contra-stance.
        Updated whenever the agent expresses −1 and receives feedback.

    Both values start at 0.0 so neither stance is initially preferred,
    and the social environment drives divergence over rounds.
    """

    q_pos: float = 0.0
    q_neg: float = 0.0

    @property
    def preferred_opinion(self) -> int:
        """Preferred opinion: argmax over Q-values; ties resolve in favour of +1.

        This is the deterministic indicator used for metrics and logging.
        It reflects the stance the agent *would prefer* given its current
        Q-values, but is not necessarily the stance expressed in any given
        interaction.  The actual expressed stance is drawn stochastically by
        ``softmax_opinion()`` and can differ from this property when
        conviction (q_gap) is low.
        """
        return 1 if self.q_pos >= self.q_neg else -1

    @property
    def q_gap(self) -> float:
        """Signed confidence: positive = leaning pro, negative = leaning contra."""
        return self.q_pos - self.q_neg


def init_opinion_states(agent_names: list[str]) -> dict[str, AgentOpinionState]:
    """Initialise one neutral AgentOpinionState for each agent."""
    return {name: AgentOpinionState() for name in agent_names}


def update_q_value(
    opinion: AgentOpinionState,
    expressed: int,
    reward: float,
    alpha: float,
) -> None:
    """Temporal-difference update for the Q-value of the expressed stance.

    Implements the SFT update rule from Banisch & Olbrich (2019):

        Q(o_i) ← (1 − α) · Q(o_i) + α · r

    Only the Q-value of ``expressed`` is updated; the other Q-value is left
    unchanged — consistent with SFT's assumption that unexpressed opinions
    receive no feedback signal.

    ``expressed`` must be the actual stance drawn by ``softmax_opinion()`` for
    this interaction, not the modal argmax.  The two can differ when conviction
    is low (q_gap near 0) and must be kept in sync for the trajectory to be
    interpretable.

    Parameters
    ----------
    opinion:
        The agent's current opinion state.  Mutated in place.
    expressed:
        The stance (+1 or −1) the agent expressed in this interaction.
    reward:
        Social feedback scalar r ∈ [−1.0, 1.0].  Positive = agreement.
    alpha:
        Learning rate α (LEARNING_RATE in config.py).
    """
    if expressed == 1:
        opinion.q_pos = (1 - alpha) * opinion.q_pos + alpha * reward
    else:
        opinion.q_neg = (1 - alpha) * opinion.q_neg + alpha * reward


def softmax_opinion(opinion: AgentOpinionState, beta: float) -> int:
    """Draw an expressed opinion via softmax with inverse temperature β.

    Uses the logistic (sigmoid) form, which is equivalent to a two-class
    softmax and numerically stable:

        p(+1) = σ(β · (q_pos − q_neg)) = 1 / (1 + exp(−β · Δq))

    β = 0  → p(+1) = 0.5  (uniform random — fixes tied-Q init artifact)
    β > 0  → p(+1) > 0.5 when q_pos > q_neg
    β → ∞  → deterministic argmax

    Parameters
    ----------
    opinion:
        Current Q-value state.
    beta:
        Inverse temperature β ≥ 0.

    Returns
    -------
    int
        +1 or −1.
    """
    import random
    p_pos = 1.0 / (1.0 + math.exp(-beta * opinion.q_gap))
    return 1 if random.random() < p_pos else -1


def opinion_states_to_dict(
    opinion_states: dict[str, AgentOpinionState],
    expressed_stances: dict[str, int] | None = None,
) -> dict[str, dict]:
    """Serialise opinion states for JSON logging.

    Parameters
    ----------
    opinion_states:
        Mapping of agent name to AgentOpinionState.
    expressed_stances:
        Optional mapping of agent name to the actual softmax-drawn stance for
        the current round.  When provided, each agent entry gains an
        ``"expressed"`` key with the interaction-level draw alongside the
        deterministic ``"preferred"`` (argmax) indicator.  Agents absent from
        this dict (e.g. never selected as expresser in the round) receive no
        ``"expressed"`` key in their entry.
    """
    result = {}
    for name, s in opinion_states.items():
        entry: dict = {
            "q_pos":     round(s.q_pos, 4),
            "q_neg":     round(s.q_neg, 4),
            "preferred": s.preferred_opinion,
        }
        if expressed_stances is not None and name in expressed_stances:
            entry["expressed"] = expressed_stances[name]
        entry["q_gap"] = round(s.q_gap, 4)
        result[name] = entry
    return result


def compute_polarization_metrics(
    opinion_states: dict[str, AgentOpinionState],
) -> dict:
    """Population-level polarization metrics aligned with Banisch & Olbrich (2019).

    Parameters
    ----------
    opinion_states:
        Mapping of agent name to AgentOpinionState.

    Returns
    -------
    dict with keys:
        n_pos : int
            Agents whose preferred opinion is +1 (argmax).
        n_neg : int
            Agents whose preferred opinion is −1 (argmax).
        dispersion : float
            Variance of preferred opinions in {−1, +1}.  Ranges from 0
            (full consensus) to 1 (maximally split population).
        mean_q_gap : float
            Mean |Q(+1) − Q(−1)| across agents — average certainty of
            preferred stance.  High values indicate committed opinions.
    """
    if not opinion_states:
        return {}

    opinions = [s.preferred_opinion for s in opinion_states.values()]
    n = len(opinions)
    n_pos = sum(1 for o in opinions if o == 1)
    n_neg = n - n_pos
    mean_opinion = sum(opinions) / n
    dispersion = sum((o - mean_opinion) ** 2 for o in opinions) / n
    mean_q_gap = sum(abs(s.q_gap) for s in opinion_states.values()) / n

    return {
        "n_pos":      n_pos,
        "n_neg":      n_neg,
        "dispersion": round(dispersion, 4),
        "mean_q_gap": round(mean_q_gap, 4),
    }
