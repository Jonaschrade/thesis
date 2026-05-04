"""
Pairwise discussion runner for the network simulation.

A single discussion consists of ``DISCUSSION_TURNS`` alternating LLM calls
between two agents, followed by a social-feedback evaluation from each agent.
The discussion is self-contained: each agent sees only the current pair's
transcript (not the global conversation history), and draws on its own
persistent memory for context.

The caller receives a plain dict with the transcript, both agents' concordance
scores, and both agents' stated reasons.  The edge lifecycle module
(``network/edges.py``) consumes the scores to adjust edge strength.

Extension point
---------------
When adding Banisch opinion tracking, pass ``score_a`` and ``score_b`` directly
as the reward signal (they are already continuous values in [−1.0, 1.0]).
The ``update_opinion_states`` function in ``network/opinion.py`` will consume
those fields.  No structural change to this module is required.
"""

from __future__ import annotations

from agents.agent import Agent
from config import DISCUSSION_TURNS


def run_discussion(
    agent_a: Agent,
    agent_b: Agent,
    topic: str,
    topic_label: str = "",
    turns_per_agent: int = DISCUSSION_TURNS,
    verbose: bool = True,
) -> dict:
    """Run a pairwise discussion between two agents and collect concordance scores.

    The discussion begins with a moderator message containing ``topic``.
    Agents then alternate for ``turns_per_agent × 2`` total LLM turns
    (agent_a speaks on even indices, agent_b on odd indices), so each agent
    contributes exactly ``turns_per_agent`` replies.  After the last turn,
    both agents independently evaluate the conversation via their
    ``evaluate()`` method, which returns a concordance score in [−1.0, 1.0].

    Each agent's ``respond()`` call stores the interaction in its own memory
    store, so the discussion is reflected in future retrieval even though
    neither agent has access to the other's memories.

    Parameters
    ----------
    agent_a:
        The first participant.  Speaks first (turn 0, 2, 4, …).
    agent_b:
        The second participant.  Speaks second (turn 1, 3, 5, …).
    topic:
        The opening moderator message used to seed the conversation.
    topic_label:
        Short identifier for the topic (e.g. a dict key from ``config.TOPICS``).
        Passed through to the returned dict so it appears in ``events.jsonl``.
        Defaults to an empty string when not needed.
    turns_per_agent:
        Number of turns each agent takes.  Total LLM calls for the turn
        loop = ``turns_per_agent × 2``.  Defaults to ``DISCUSSION_TURNS``
        from config.
    verbose:
        When ``True`` (default), each agent reply is printed to stdout as
        ``"\\n{agent_name}: {reply}"``.  Set to ``False`` to suppress
        per-turn output (e.g. in automated batch runs).

    Returns
    -------
    dict with keys:
        ``turns`` : list[dict]
            Conversation transcript as ``{"speaker": str, "content": str}``
            dicts, excluding the moderator opening.
        ``score_a`` : float
            Concordance score from agent_a in [−1.0, 1.0].
            Positive = agreement, negative = disagreement.
        ``reason_a`` : str
            One-sentence concordance description from agent_a.
        ``score_b`` : float
            Concordance score from agent_b.  Same range.
        ``reason_b`` : str
            One-sentence concordance description from agent_b.
        ``topic_label`` : str
            The label passed in via ``topic_label``.

    Extension point
    ---------------
    ``score_a`` and ``score_b`` are already the continuous reward signal
    r ∈ [−1.0, 1.0] that ``network/opinion.py`` will need for Q-value updates.
    Pass them directly to ``update_opinion_states`` when that module is added.
    """
    transcript: list[dict] = [{"speaker": "Moderator", "content": topic}]
    total_turns = turns_per_agent * 2

    for i in range(total_turns):
        speaker = agent_a if i % 2 == 0 else agent_b
        last = transcript[-1]
        reply = speaker.respond(last["content"], last["speaker"])
        if verbose:
            print(f"\n{speaker.name}: {reply}")
        transcript.append({"speaker": speaker.name, "content": reply})

    # Exclude the moderator opening before passing to evaluate()
    convo = transcript[1:]

    eval_a = agent_a.evaluate(convo)
    eval_b = agent_b.evaluate(convo)

    return {
        "topic_label": topic_label,
        "turns":       convo,
        "score_a":     eval_a["score"],
        "reason_a":    eval_a["reason"],
        "score_b":     eval_b["score"],
        "reason_b":    eval_b["reason"],
    }
