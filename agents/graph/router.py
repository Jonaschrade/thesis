"""
Routing logic: decides whether to continue, reflect, evaluate, or end the conversation.

Three routing functions are used at different points in the graph:
    router               — after each non-last agent: continue or END
    make_reflector_router — after the reflector: evaluate or loop back
    eval_router          — after the evaluator: continue or END
"""

from langgraph.graph import END
from graph.state import State
from config import EVAL_EVERY


def router(state: State) -> str:
    """Return the next speaker's name, or END when max_rounds is reached."""
    if state["round"] >= state["max_rounds"]:
        return END
    return state["next_speaker"]


def make_reflector_router(first_agent_name: str):
    """Factory returning a router that fires the evaluator every EVAL_EVERY rounds.

    Args:
        first_agent_name: Name of agents[0], used as the loop-back destination.
    """
    def reflector_router(state: State) -> str:
        if state["round"] % EVAL_EVERY == 0 and state["round"] > 0:
            return "evaluator"
        return first_agent_name
    return reflector_router


def eval_router(state: State) -> str:
    """End the conversation if any agent voted 'move', otherwise continue.

    A single 'move' vote is enough to stop; unanimous agreement is not required.
    """
    if any(v["vote"] == "move" for v in state["evaluations"]):
        print("\n  ✋ Conversation stopped by agent vote.")
        return END
    return state["next_speaker"]
