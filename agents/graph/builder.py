"""
Graph builder: wires agents into a LangGraph conversation loop.

build_graph() accepts any list of Agent objects and automatically:
    - Creates one node per agent.
    - Routes each agent to the next in list order (wrapping around).
    - Inserts a reflector node after every full round.
    - Inserts an evaluator node every EVAL_EVERY rounds.
    - Ends the conversation when max_rounds is reached or any agent votes 'move'.
"""

from langgraph.graph import StateGraph, END

from agents.agent import Agent
from graph.router import router, make_reflector_router, eval_router
from graph.state import State
from config import EVAL_EVERY, REFLECT_EVERY


def make_node(agent: Agent, next_agent_name: str, num_agents: int):
    """Factory returning a LangGraph node function bound to agent.

    The returned node reads the last message, generates the agent's reply,
    advances the turn and round counters, and returns the updated state.

    Args:
        agent:           The Agent instance that owns this node.
        next_agent_name: Name of the agent to speak next.
        num_agents:      Total number of agents (used to compute round number).
    """
    def node(state: State) -> State:
        last = state["messages"][-1]

        new_turn = state["turn"] + 1
        # Round increments only after every agent has spoken once
        new_round = new_turn // num_agents
        reply = agent.respond(last["content"], last["speaker"])
        print(f"\n{agent.name}: {reply}")

        return {
            **state,
            "messages":     state["messages"] + [{"speaker": agent.name, "content": reply}],
            "next_speaker": next_agent_name,
            "turn":         new_turn,
            "round":        new_round,
            "evaluations":  [],
        }
    return node


def make_reflector(agents: list[Agent]):
    """Factory returning a node that triggers simultaneous reflection for all agents.

    Reflection only fires when the current round is a non-zero multiple of
    REFLECT_EVERY, so not every round causes reflection overhead.
    """
    def reflector(state: State) -> State:
        current_round = state["round"]
        if current_round > 0 and current_round % REFLECT_EVERY == 0:
            print("\n\n── Reflection phase ──")
            for agent in agents:
                agent.reflect()
        return state
    return reflector


def last_to_reflector(state: State) -> str:
    """Route the last agent's output to the reflector, or END if max_rounds is reached.

    This is a dedicated router for the last agent in the list so that every
    full round passes through the reflector before looping back.
    """
    if state["round"] >= state["max_rounds"]:
        return END
    return "reflector"


def make_evaluator(agents: list[Agent]):
    """Factory returning a node that collects a continue/move vote from every agent.

    Only messages from the current conversation (state["conversation_start"]:)
    are passed to each agent's evaluate() so that prior conversation history
    does not pollute the social-reward signal.
    """
    def evaluator(state: State) -> State:
        print("\n\n── Evaluation phase ──")
        current_transcript = state["messages"][state["conversation_start"]:]
        votes = [agent.evaluate(current_transcript) for agent in agents]
        return {**state, "evaluations": votes}
    return evaluator


def build_graph(agents: list[Agent]):
    """Build and compile the LangGraph conversation graph for the given agents.

    Conversation order follows list order, cycling back to agents[0] after the
    last agent speaks.  The graph topology is:

        agent_0 → agent_1 → … → agent_n → reflector → [evaluator] → agent_0

    The evaluator step is inserted every EVAL_EVERY rounds; any 'move' vote
    there terminates the conversation.

    Args:
        agents: List of Agent instances (minimum 2).

    Returns:
        A compiled LangGraph runnable.
    """
    g = StateGraph(State)
    n = len(agents)

    last_agent  = agents[-1]
    first_agent = agents[0]

    for i, agent in enumerate(agents):
        next_agent = agents[(i + 1) % n]
        g.add_node(agent.name, make_node(agent, next_agent.name, n))

    g.add_node("reflector", make_reflector(agents))
    g.add_node("evaluator", make_evaluator(agents))

    # All agents except the last route normally (respecting max_rounds)
    for agent in agents[:-1]:
        next_agent = agents[(agents.index(agent) + 1) % n]
        g.add_conditional_edges(
            agent.name,
            router,
            {next_agent.name: next_agent.name, END: END}
        )

    # Last agent always goes to the reflector (or END)
    g.add_conditional_edges(
        last_agent.name,
        last_to_reflector,
        {"reflector": "reflector", END: END}
    )

    # Reflector routes to evaluator (if due) or back to the first agent
    g.add_conditional_edges(
        "reflector",
        make_reflector_router(first_agent.name),
        {"evaluator": "evaluator", first_agent.name: first_agent.name}
    )

    # After evaluation: continue to first agent or end the conversation
    g.add_conditional_edges(
        "evaluator",
        eval_router,
        {first_agent.name: first_agent.name, END: END}
    )

    g.set_entry_point(agents[0].name)
    return g.compile()
