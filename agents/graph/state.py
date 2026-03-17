"""
LangGraph shared state definition.
Kept in separate file to avoid circular imports between builder and router.
"""
from typing import TypedDict, List

class State(TypedDict):
    messages: List[dict] # list of {"speaker": str, "content": str}
    next_speaker: str # name of agent whose turn it is
    turn: int # current turn counter
    round: int # increments after all agents have spoken once
    max_rounds: int # conversation ends when round >= max_rounds,
    evaluations: List[dict] # [{"agent": str, "vote": "continue|move", "reason": str}]


