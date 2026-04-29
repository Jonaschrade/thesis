"""
Pairwise discussion runner for the network simulation.

A single discussion consists of ``DISCUSSION_TURNS`` alternating LLM calls
between two agents, followed by a social-reward evaluation from each agent.
The discussion is self-contained: each agent sees only the current pair's
transcript (not the global conversation history), and draws on its own
persistent memory for context.

The caller receives a plain dict with the transcript, both agents' votes, and
both agents' stated reasons.  The edge lifecycle module (``network/edges.py``)
consumes the votes to decide whether the edge survives.

Extension point
---------------
When adding Banisch opinion tracking, uncomment the ``reward_a`` / ``reward_b``
fields in the returned dict and map the vote to ±1 there.  The
``update_opinion_states`` function in ``network/opinion.py`` will consume those
fields.  No structural change to this module is required.
"""

from __future__ import annotations

from agents.agent import Agent
from config import DISCUSSION_TURNS


def run_discussion(
    agent_a: Agent,
    agent_b: Agent,
    topic: str,
    turns: int = DISCUSSION_TURNS,
    verbose: bool = True,
) -> dict:
    """Run a pairwise discussion between two agents and collect edge votes.

    The discussion begins with a moderator message containing ``topic``.
    Agents then alternate for ``turns`` LLM turns (agent_a speaks on even
    indices, agent_b on odd indices).  After the last turn, both agents
    independently evaluate the conversation via their ``evaluate()`` method
    and vote ``"continue"`` or ``"move"``.

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
    turns:
        Total number of agent turns.  Should be even so that both agents
        speak the same number of times.  Defaults to ``DISCUSSION_TURNS``
        from config.
    verbose:
        When ``True`` (default), each agent reply is printed to stdout as
        ``"\\n{agent_name}: {reply}"`` — matching the output format of the
        original LangGraph-based pairwise runner.  Set to ``False`` to
        suppress per-turn output (e.g. in automated batch runs).

    Returns
    -------
    dict with keys:
        ``turns`` : list[dict]
            Conversation transcript as ``{"speaker": str, "content": str}``
            dicts, excluding the moderator opening.
        ``vote_a`` : str
            ``"continue"`` or ``"move"`` — social-reward vote from agent_a.
        ``reason_a`` : str
            One-sentence reason given by agent_a for their vote.
        ``vote_b`` : str
            ``"continue"`` or ``"move"`` — social-reward vote from agent_b.
        ``reason_b`` : str
            One-sentence reason given by agent_b for their vote.

    Extension point
    ---------------
    Uncomment the following two keys when ``network/opinion.py`` is added::

        "reward_a": 1.0 if eval_a["vote"] == "continue" else -1.0,
        "reward_b": 1.0 if eval_b["vote"] == "continue" else -1.0,
    """
    transcript: list[dict] = [{"speaker": "Moderator", "content": topic}]

    for i in range(turns):
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
        "turns":    convo,
        "vote_a":   eval_a["vote"],
        "reason_a": eval_a["reason"],
        "vote_b":   eval_b["vote"],
        "reason_b": eval_b["reason"],
        # "reward_a": 1.0 if eval_a["vote"] == "continue" else -1.0,
        # "reward_b": 1.0 if eval_b["vote"] == "continue" else -1.0,
    }
