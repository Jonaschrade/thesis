"""
Entry point — pairwise simulation mode.

Runs the same round structure as the network simulation but with exactly two
agents and no graph machinery.  Each simulation round is one complete discussion
of ``DISCUSSION_TURNS`` exchanges.  Conversation continuity across rounds is
maintained through each agent's persistent memory store.

Round structure
---------------
1. Determine the active topic from the ``TOPICS`` schedule.
2. Run one discussion (``DISCUSSION_TURNS`` exchanges via ``run_discussion()``).
3. Each agent's internal edge valuation is adjusted by their own concordance
   score × ``STRENGTH_DELTA``.  The conversation ends early if either agent's
   valuation falls to or below ``STRENGTH_FLOOR``.
4. Every ``REFLECT_EVERY`` rounds all agents reflect on their recent memories.

Configuration
-------------
All tunable parameters live in ``config.py``.
To run the network simulation instead, see ``main_network.py``.
"""

from langchain_ollama import OllamaLLM

from agents.agent import Agent
from agents.personas import sample_personas
from config import (
    LLM_MODEL,
    NETWORK_MAX_ROUNDS,
    OLLAMA_HOST,
    REFLECT_EVERY,
    STRENGTH_CAP,
    STRENGTH_DELTA,
    STRENGTH_FLOOR,
    TOPICS,
)
from network.discussion import run_discussion
from network.logger import SimulationLogger

# ── Topic schedule (mirrors main_network.py logic) ───────────────────────────
_TOPIC_LABELS = list(TOPICS.keys())
_BLOCK_SIZE   = NETWORK_MAX_ROUNDS // len(_TOPIC_LABELS)


def _topic_for_round(round_n: int) -> tuple[str, str]:
    """Return (label, text) for the given simulation round."""
    idx = min((round_n - 1) // _BLOCK_SIZE, len(_TOPIC_LABELS) - 1)
    label = _TOPIC_LABELS[idx]
    return label, TOPICS[label]


def main() -> None:
    """Run the two-agent pairwise simulation."""
    llm = OllamaLLM(model=LLM_MODEL, base_url=f"http://{OLLAMA_HOST}")
    logger = SimulationLogger()

    print(f"\nSampling 2 personas...")
    personas = sample_personas(2, llm)
    agents = [Agent(name=p["name"], persona=p["persona"], llm=llm) for p in personas]

    print(f"\n{'━' * 50}")
    print("Participants")
    for a in agents:
        print(f"  {a.name}: {a.persona}")
    print(f"{'━' * 50}\n")

    logger.log_personas({a.name: a for a in agents})

    # Per-agent internal valuations of the relationship (mirrors network/edges.py)
    strength = {agents[0].name: 1.0, agents[1].name: 1.0}

    for round_n in range(1, NETWORK_MAX_ROUNDS + 1):
        topic_label, topic_text = _topic_for_round(round_n)
        a, b = agents[0].name, agents[1].name

        print(f"\n{'━' * 50}")
        print(f"Round {round_n} / {NETWORK_MAX_ROUNDS}  │  topic: {topic_label}  │  "
              f"strengths: {a} {strength[a]:.2f} / {b} {strength[b]:.2f}")
        print(f"{'━' * 50}")

        # ── Discussion ────────────────────────────────────────────────────
        result = run_discussion(agents[0], agents[1], topic_text, topic_label=topic_label)

        # ── Per-agent strength update (mirrors network/edges.py) ──────────
        strength[a] = max(0.0, min(STRENGTH_CAP, strength[a] + result["score_a"] * STRENGTH_DELTA))
        strength[b] = max(0.0, min(STRENGTH_CAP, strength[b] + result["score_b"] * STRENGTH_DELTA))

        print(f"\n  {a} {result['score_a']:+.2f} → {strength[a]:.2f}  │  "
              f"{b} {result['score_b']:+.2f} → {strength[b]:.2f}")

        if strength[a] <= STRENGTH_FLOOR or strength[b] <= STRENGTH_FLOOR:
            print("\n  ✋ Conversation ended: relationship strength reached zero.")
            break

        # ── Reflection phase ──────────────────────────────────────────────
        if round_n % REFLECT_EVERY == 0:
            print(f"\n── Reflection phase (round {round_n}) ──")
            for agent in agents:
                agent.reflect()

    print("\n\n --Simulation complete--")


if __name__ == "__main__":
    main()
