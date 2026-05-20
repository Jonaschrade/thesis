"""
Pairwise discussion runner for the network simulation.

A single discussion consists of ``DISCUSSION_TURNS`` alternating LLM calls
between two agents.  Two reward signals are then computed:

* ``reward_a`` / ``reward_b`` — scalar rewards derived by each agent
  calling ``classify_reward()`` on the *partner's* last message.  These
  drive the SFT Q-value TD update and are structurally independent of the
  agents' own response generation.

* ``score_a`` / ``score_b`` — full-transcript concordance scores from
  ``evaluate()``.  These are used only when ``GRAPH_DYNAMIC = True`` to
  adjust each agent's internal edge valuation.

Keeping the two signals separate preserves the causal chain required for
the SFT interpretability claim: Q-value trajectories are driven by the
classifier, not by the agents' self-assessment.
"""

from __future__ import annotations

from agents.agent import Agent


def run_discussion(
    agent_a: Agent,
    agent_b: Agent,
    topic: str,
    topic_label: str = "",
    turns_per_agent: int = 1,
    verbose: bool = True,
    opinion_a: int | None = None,
    opinion_b: int | None = None,
) -> dict:
    """Run a pairwise discussion between two agents and collect reward signals.

    The discussion begins with a moderator message containing ``topic``.
    Agents then alternate for ``turns_per_agent × 2`` total LLM turns.
    Each agent's ``respond()`` call is conditioned on their SFT expressed
    opinion (``opinion_a`` / ``opinion_b``) when provided.

    After the last turn, two independent reward signals are computed:

    1. **Classifier reward** — each agent calls ``classify_reward()`` on the
       *partner's* last message.  No persona or full-transcript context is
       used, keeping generation and evaluation causally separate.

    2. **Evaluate score** — each agent calls ``evaluate()`` on the full
       transcript.  Retained for edge-dynamics mode (``GRAPH_DYNAMIC = True``).

    Parameters
    ----------
    agent_a:
        The first participant.  Speaks first (turn 0, 2, 4, …).
    agent_b:
        The second participant.  Speaks second (turn 1, 3, 5, …).
    topic:
        The opening moderator message used to seed the conversation.
    topic_label:
        Short identifier for the topic.
    turns_per_agent:
        Number of turns each agent takes.  Defaults to 1 (one asymmetric
        exchange); pass a higher value for multi-exchange discussions.
    verbose:
        When ``True``, each reply is printed to stdout.
    opinion_a:
        SFT expressed opinion for agent_a (+1 or −1).  Passed to
        ``respond()`` to anchor the agent's stance.  None disables anchoring.
    opinion_b:
        SFT expressed opinion for agent_b.  Same semantics as ``opinion_a``.

    Returns
    -------
    dict with keys:
        ``turns`` : list[dict]
            Conversation transcript as ``{"speaker": str, "content": str}``
            dicts, excluding the moderator opening.
        ``reward_a`` : float
            Classifier reward for agent_a from agent_b's last reaction ∈ [−1, 1].
            Used for agent_a's Q-value TD update.
        ``reward_b`` : float
            Classifier reward for agent_b from agent_a's last reaction ∈ [−1, 1].
        ``score_a`` : float
            Full-transcript concordance score from agent_a ∈ [−1, 1].
            Used for edge-dynamics when GRAPH_DYNAMIC = True.
        ``reason_a`` : str
            One-sentence concordance description from agent_a.
        ``score_b`` : float
            Full-transcript concordance score from agent_b.
        ``reason_b`` : str
            One-sentence concordance description from agent_b.
        ``topic_label`` : str
            The label passed in via ``topic_label``.
    """
    transcript: list[dict] = [{"speaker": "Moderator", "content": topic}]
    total_turns = turns_per_agent * 2

    for i in range(total_turns):
        if i % 2 == 0:
            speaker = agent_a
            expressed = opinion_a
        else:
            speaker = agent_b
            expressed = opinion_b

        last = transcript[-1]
        reply = speaker.respond(last["content"], last["speaker"], expressed_opinion=expressed)
        if verbose:
            print(f"\n{speaker.name}: {reply}")
        transcript.append({"speaker": speaker.name, "content": reply})

    # Exclude the moderator opening before evaluate() and reward extraction
    convo = transcript[1:]

    # Classifier reward: each agent classifies the partner's last message.
    # Extracted without persona or full-transcript context.
    last_b_msg = next(
        t["content"] for t in reversed(convo) if t["speaker"] == agent_b.name
    )
    last_a_msg = next(
        t["content"] for t in reversed(convo) if t["speaker"] == agent_a.name
    )
    reward_a = agent_a.classify_reward(last_b_msg)
    reward_b = agent_b.classify_reward(last_a_msg)

    print(f"\n  🎯 Reward  {agent_a.name}: {reward_a:+.2f}  |  {agent_b.name}: {reward_b:+.2f}")

    # Full-transcript evaluation for edge-dynamics mode
    eval_a = agent_a.evaluate(convo)
    eval_b = agent_b.evaluate(convo)

    return {
        "topic_label": topic_label,
        "turns":       convo,
        "reward_a":    reward_a,
        "reward_b":    reward_b,
        "score_a":     eval_a["score"],
        "reason_a":    eval_a["reason"],
        "score_b":     eval_b["score"],
        "reason_b":    eval_b["reason"],
    }
