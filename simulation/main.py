"""
Entry point.

Samples personas from the pool, instantiates agents, prints participant
summaries, and runs the LangGraph conversation loop.

To change the number of agents, edit NUM_AGENTS in config.py.
To change the LLM model or conversation thresholds, edit config.py.
"""

from langchain_ollama import OllamaLLM

from agents.agent import Agent
from agents.personas import sample_personas
from config import DEFAULT_MAX_ROUNDS, LLM_MODEL, NUM_AGENTS, OLLAMA_HOST

from graph.builder import build_graph

def main():
    llm = OllamaLLM(model=LLM_MODEL, base_url=f"http://{OLLAMA_HOST}")

    print(f"\nSampling {NUM_AGENTS} personas...")
    personas = sample_personas(NUM_AGENTS, llm)

    agents = [
        Agent(name=p["name"], persona=p["persona"], llm=llm) for p in personas
    ]

    print(f"\n{'━' * 50}")
    print("Participants")
    for a in agents:
        print(f"  {a.name}: {a.persona}")
    print(f"{'━' * 50}")

    graph = build_graph(agents)

    opening = (
        "Deutschland nimmt jedes Jahr Hunderttausende Migranten auf – "
        "doch Integration scheitert immer wieder an Sprache, Arbeit und Kultur. "
        "Sollte Deutschland die Grenzen für Nicht-EU-Ausländer dauerhaft schließen?"
    )

    print(f"\n Moderator: {opening}\n")

    # agents[0] always speaks first; the graph cycles through agents in list order
    graph.invoke({
        "messages":           [{"speaker": "Moderator", "content": opening}],
        "conversation_start": 0,
        "next_speaker":       agents[0].name,
        "turn":               0,
        "round":              0,
        "max_rounds":         DEFAULT_MAX_ROUNDS,
        "evaluations":        [],
    })

    print("\n\n --Conversation complete--")

if __name__ == "__main__":
    main()
