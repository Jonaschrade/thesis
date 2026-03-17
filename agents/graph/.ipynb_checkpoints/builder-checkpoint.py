"""
Graph builder: wires agents into LangGraph conversation loop.

build_graph() accepts any list of Agent objects and automatically:
    - Creates one node per agent.
    - Routes each agent to the next in the list (wrapping around).
    - Applies the round-limit router on every edge.
"""

from langgraph.graph import StateGraph, END

from agents.agent import Agent
from graph.router import router, make_last_agent_router, eval_router
from graph.state import State
from config import EVAL_EVERY, REFLECT_EVERY

def make_node(agent: Agent, next_agent_name: str, num_agents: int):
    """
    Factory that returns a LangGraph node function bound to `agent`.
    The node reads the last message, generates reply, and updates state.
    """

    def node(state: State) -> State:
        last = state["messages"][-1]
        
        new_turn=state["turn"]+1
        new_round=new_turn//num_agents # increment round after all agents have spoken
        reply=agent.respond(last["content"], last["speaker"])
        print(f"\n{agent.name}: {reply}")

        if new_round>0 and new_round % REFLECT_EVERY == 0 and new_round!=agent._last_reflected_round:
            agent._last_reflected_round=new_round
            agent.reflect()


            
        return {
            **state,
            "messages":     state["messages"] + [{"speaker": agent.name, "content": reply}],
            "next_speaker": next_agent_name,
            "turn":         new_turn,
            "round":        new_round,
            "evaluations":  [],
        }
    return node

def make_evaluator(agents: list[Agent]):
    """
    Node that collects vote from every agent after full round.
    """
    def evaluator(state: State) -> State:
        print("\n\n── Evaluation phase ──")
        votes=[agent.evaluate(state["messages"]) for agent in agents]
        return {**state, "evaluations": votes}
    return evaluator


def build_graph(agents: list[Agent]):
    """
    Build and compile a LangGraph graph for given list of agents.

    Conversation order follows list order, cycling back to agents[0] after last agent speaks.

    Args:
        agents: list of Agent istances (minimum 2)

    Returns:
        A compiled LangGraph runnable.
    """

    g = StateGraph(State)
    n = len(agents)

    
    last_agent=agents[-1]
    first_agent=agents[0]
    last_router = make_last_agent_router(first_agent.name)

    for i, agent in enumerate (agents):
        next_agent = agents[(i+1) % n]
        g.add_node(agent.name, make_node(agent, next_agent.name, n))

    # All agents except last one route normally
    for agent in agents[:-1]: 
        next_agent = agents[(agents.index(agent)+1) % n]
        g.add_conditional_edges(
            agent.name,
            router,
            {next_agent.name: next_agent.name, END: END}
        )
    
    # Last agent uses factory-produced router
    g.add_conditional_edges(
        last_agent.name,
        last_router,
        {"evaluator": "evaluator", first_agent.name: first_agent.name, END: END}
    )

    # After evaluation continue or end
    g.add_node("evaluator", make_evaluator(agents))
    g.add_conditional_edges(
        "evaluator",
        eval_router,
        {first_agent.name: first_agent.name, END: END}

    )

    # The first agent in list always responds first 
    g.set_entry_point(agents[0].name)
    return g.compile()