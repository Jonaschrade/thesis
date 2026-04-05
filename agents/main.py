"""
Entry point: Instantiates agents and runs conversation graph.

To change personas, edit agents/personas.py
To change llm model or thresholds, edit config.py
"""

from langchain_ollama import OllamaLLM

from agents.agent import Agent
from agents.personas import PERSONAS
from config import DEFAULT_MAX_ROUNDS, LLM_MODEL, OLLAMA_HOST

from graph.builder import build_graph

def main():
    llm = OllamaLLM(model=LLM_MODEL, base_url = f"http://{OLLAMA_HOST}")

    agents = [
        Agent(name = p["name"], persona= p["persona"], llm = llm) for p in PERSONAS
    ]

    graph = build_graph(agents)

    # Moderator opens conversation; Alice (agents[0]) responds first

    opening = ("With the recent surge in migration flows into Europe and the growing pressure on Germany's housing, labor market, and social services, what is the most sustainable and socially cohesive way for Germany to manage and integrate immigration over the next decade?")

    print(f"\n Moderator: {opening}\n")

    graph.invoke({
        "messages": [{"speaker": "Moderator", "content": opening}],
        "next_speaker": agents[0].name,
        "turn": 0,
        "round":0,
        "max_rounds": DEFAULT_MAX_ROUNDS,
        "evaluations": []
    })

    print("\n\n --Conversation complete--")

if __name__ == "__main__":
    main()