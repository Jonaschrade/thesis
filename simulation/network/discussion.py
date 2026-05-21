"""
Pairwise discussion runner for the network simulation.

A single discussion consists of alternating LLM calls between two agents,
followed by reward classification.  The reward signal drives two mechanisms:

* **Q-value update** — ``reward_a`` is used for the expresser's SFT TD update
  (``reward_b`` is computed but currently unused in the Q-update, which is
  asymmetric by design).

* **Edge evaluation** — when ``GRAPH_DYNAMIC = True``, ``reward_a`` is
  recorded in the edge's rolling reward history and drives the expresser's
  internal valuation of the channel (see ``network/edges.py``).  In the
  current asymmetric implementation only ``reward_a`` is passed to
  ``update_edge``; ``reward_b`` is available for a future symmetric extension.

The classifier reward is kept causally separate from ``respond()`` (no shared
prompt context) to avoid self-scoring contamination (Chuang et al. 2024).
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

    After the last turn, classifier rewards are computed:

    * **Classifier reward** — each agent calls ``classify_reward()`` on the
      *partner's* last message.  No persona or full-transcript context is
      used, keeping generation and evaluation causally separate.

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
            Drives agent_a's Q-value TD update and edge valuation (asymmetric mode).
        ``reward_b`` : float
            Classifier reward for agent_b from agent_a's last reaction ∈ [−1, 1].
            Available for a future symmetric edge-evaluation extension.
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

    # Exclude the moderator opening from reward extraction
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

    return {
        "topic_label": topic_label,
        "turns":       convo,
        "reward_a":    reward_a,
        "reward_b":    reward_b,
    }
