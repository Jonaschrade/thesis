"""
Entry point — pairwise simulation mode.

Two agents hold a sequential round-based conversation, reflecting and
evaluating at configurable intervals.  This is a direct Python reimplementation
of the former LangGraph-based loop; the console output is identical but the
graph machinery has been removed entirely.

Round structure
---------------
Each of the ``DEFAULT_MAX_ROUNDS`` rounds both agents speak once (in list
order).  After the round:

* Every ``REFLECT_EVERY`` rounds all agents reflect on their recent memories.
* Every ``EVAL_EVERY`` rounds all agents cast a social-reward vote
  (``weiter`` / ``wechseln``).  A single ``wechseln`` vote ends the
  conversation early.

Configuration
-------------
All tunable parameters live in ``config.py``.
To run the network simulation instead, see ``main_network.py``.
"""

from langchain_ollama import OllamaLLM

from agents.agent import Agent
from agents.personas import sample_personas
from config import (
    DEFAULT_MAX_ROUNDS,
    EVAL_EVERY,
    LLM_MODEL,
    NUM_AGENTS,
    OLLAMA_HOST,
    REFLECT_EVERY,
)

_OPENING = (
    "Deutschland nimmt jedes Jahr Hunderttausende Migranten auf – "
    "doch Integration scheitert immer wieder an Sprache, Arbeit und Kultur. "
    "Sollte Deutschland die Grenzen für Nicht-EU-Ausländer dauerhaft schließen?"
)


def main() -> None:
    """Run the two-agent pairwise conversation."""
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

    print(f"\n Moderator: {_OPENING}\n")

    # Full message history; agents[0] always speaks first each round
    messages: list[dict] = [{"speaker": "Moderator", "content": _OPENING}]

    for round_n in range(1, DEFAULT_MAX_ROUNDS + 1):

        # ── Each agent speaks once per round ─────────────────────────────
        for agent in agents:
            last = messages[-1]
            reply = agent.respond(last["content"], last["speaker"])
            print(f"\n{agent.name}: {reply}")
            messages.append({"speaker": agent.name, "content": reply})

        # ── Reflection phase ──────────────────────────────────────────────
        if round_n % REFLECT_EVERY == 0:
            print("\n\n── Reflection phase ──")
            for agent in agents:
                agent.reflect()

        # ── Evaluation phase ──────────────────────────────────────────────
        if round_n % EVAL_EVERY == 0:
            print("\n\n── Evaluation phase ──")
            transcript = messages[1:]  # pass full history, excluding moderator opening
            votes = [agent.evaluate(transcript) for agent in agents]
            if any(v["vote"] == "move" for v in votes):
                print("\n  ✋ Conversation stopped by agent vote.")
                break

    print("\n\n --Conversation complete--")


if __name__ == "__main__":
    main()
