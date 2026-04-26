"""
LangGraph shared state definition.

Kept in a separate module to avoid circular imports between builder and router.
All graph nodes receive and return a dict conforming to this schema.
"""

from typing import TypedDict, List


class State(TypedDict):
    messages:           List[dict]  # {"speaker": str, "content": str} — full simulation history
    conversation_start: int         # index into messages where the current conversation begins
    next_speaker:       str         # name of the agent whose turn it is next
    turn:               int         # total number of agent turns taken so far
    round:              int         # increments after every agent has spoken once
    max_rounds:         int         # conversation ends when round >= max_rounds
    evaluations:        List[dict]  # [{"agent": str, "vote": "continue"|"move", "reason": str}]
