"""
Routing logic: deicdes whether to continue or end conversation.
"""

from langgraph.graph import END
from graph.state import State
from config import EVAL_EVERY


def router(state: State) -> str:
    """
    Return next node name, or END when max_rounds is reached.
    """

    if state["round"] >= state["max_rounds"]:
        return END
    
    return state["next_speaker"]

def make_last_agent_router(first_agent_name: str):
    """Factory — returns a router bound to the first agent's name."""
    def last_agent_router(state: State) -> str:
        """
        Last agent routes to evaluator (if eval is due) or back to first, or ends conversation if max_rounds is reached.
        """
        if state["round"] >= state["max_rounds"]:
            return END
        if state["round"] % EVAL_EVERY == 0 and state["round"] > 0:
            return "evaluator"
        return first_agent_name
    return last_agent_router

def eval_router(state: State) -> str: 
    """
    After evaluation: stop if any agent voted move, else continue.
    """
    if any (v["vote"] =="move" for v in state["evaluations"]):
        print("\n  ✋ Conversation stopped by agent vote.")
        return END
    return state["next_speaker"]