"""
Entry point — pairwise simulation mode.

Implements the same SFT mechanisms as the network simulation for exactly two
agents with no graph.  Each round consists of two asymmetric interactions:
first agent A expresses to agent B, then agent B expresses to agent A, giving
each agent one expected Q-value update per round.

Mechanisms present
------------------
- SFT Q-values: both agents hold ``AgentOpinionState``; ``softmax_opinion(β)``
  draws the expressed stance each interaction.
- Asymmetric update: ``update_q_value()`` is called on the expresser only;
  the responder's Q-values are unchanged for that interaction.
- Classifier reward: ``classify_reward()`` on the partner's last message
  drives the Q-update.  ``evaluate()`` provides the full-transcript score.
- Reflection: every ``REFLECT_EVERY`` rounds.

Relationship strength (GRAPH_DYNAMIC = True)
---------------------------------------------
When ``GRAPH_DYNAMIC`` is enabled, ``evaluate()`` scores also adjust per-agent
relationship strength by ``score × STRENGTH_DELTA``.  The simulation ends
early if either agent's strength reaches ``STRENGTH_FLOOR`` — the pairwise
analogue of edge severance in network mode.  With ``GRAPH_DYNAMIC = False``
(default), only Q-dynamics run and there is no early exit.

Note on homophily h
-------------------
``HOMOPHILY_H`` has no effect in pairwise mode: with exactly two agents there
is only one possible responder, so the weighted draw degenerates to a fixed
choice.

Round structure
---------------
For each round (2 asymmetric interactions: A→B, then B→A):
  For each interaction:
    1. Softmax draw    expressed = softmax(β) over expresser's Q-values
    2. Exchange        expresser speaks → responder reacts  (1 exchange)
    3. Reward          classify_reward(responder's message) → r
    4. Q-update        Q(expressed) ← (1−α)·Q(expressed) + α·r  [expresser only]
    5. Evaluate        evaluate() → score_a/b  [always]
    6. Strength        [GRAPH_DYNAMIC only] strength update; exit if ≤ STRENGTH_FLOOR
  After both interactions:
    7. Reflection      if round % REFLECT_EVERY == 0

Configuration
-------------
All tunable parameters live in ``config.py``.
"""

from langchain_ollama import OllamaLLM

from agents.agent import Agent
from agents.personas import sample_personas
from config import (
    GRAPH_DYNAMIC,
    LEARNING_RATE,
    LLM_MODEL,
    NETWORK_MAX_ROUNDS,
    OLLAMA_HOST,
    OPINION_BETA,
    REFLECT_EVERY,
    STRENGTH_CAP,
    STRENGTH_DELTA,
    STRENGTH_FLOOR,
    TOPIC_LABEL,
    TOPIC_TEXT,
)
from network.discussion import run_discussion
from network.logger import SimulationLogger
from network.opinion import (
    compute_polarization_metrics,
    init_opinion_states,
    softmax_opinion,
    update_q_value,
)


def main() -> None:
    """Run the two-agent pairwise simulation with full SFT Q-value tracking."""
    llm = OllamaLLM(model=LLM_MODEL, base_url=f"http://{OLLAMA_HOST}")
    logger = SimulationLogger()

    # ── Agent initialisation ─────────────────────────────────────────────
    print("\nSampling 2 personas...")
    personas = sample_personas(2, llm)
    agents = [Agent(name=p["name"], persona=p["persona"], llm=llm) for p in personas]
    a_name, b_name = agents[0].name, agents[1].name

    print(f"\n{'━' * 50}")
    print("Participants")
    for ag in agents:
        print(f"  {ag.name}: {ag.persona}")
    print(f"{'━' * 50}\n")

    logger.log_personas({ag.name: ag for ag in agents})

    # ── SFT opinion state initialisation ────────────────────────────────
    opinion_states = init_opinion_states([a_name, b_name])

    # ── Relationship strength (GRAPH_DYNAMIC only) ───────────────────────
    strength = {a_name: 1.0, b_name: 1.0}

    print(f"β={OPINION_BETA}  α={LEARNING_RATE}  dynamic={GRAPH_DYNAMIC}\n")

    # ── Main simulation loop ─────────────────────────────────────────────
    terminated_early = False
    for round_n in range(1, NETWORK_MAX_ROUNDS + 1):
        n_pos = sum(1 for s in opinion_states.values() if s.expressed_opinion == 1)
        strength_str = (
            f"  │  strengths: {a_name} {strength[a_name]:.2f} / {b_name} {strength[b_name]:.2f}"
            if GRAPH_DYNAMIC else ""
        )

        print(f"\n{'━' * 50}")
        print(
            f"Round {round_n} / {NETWORK_MAX_ROUNDS}  "
            f"│  topic: {TOPIC_LABEL}  "
            f"│  opinions: +{n_pos} / −{2 - n_pos}"
            f"{strength_str}"
        )
        print(f"{'━' * 50}")

        # Two interactions per round: A→B, then B→A
        for expresser, responder in [(agents[0], agents[1]), (agents[1], agents[0])]:

            # 1. Softmax draw
            expressed_a = softmax_opinion(opinion_states[expresser.name], OPINION_BETA)
            expressed_b = softmax_opinion(opinion_states[responder.name], OPINION_BETA)

            print(
                f"\n  ▶ {expresser.name} → {responder.name}  "
                f"(stances: {expressed_a:+d} / {expressed_b:+d})"
            )

            # 2–5. Exchange, reward, Q-update, evaluate
            result = run_discussion(
                expresser,
                responder,
                TOPIC_TEXT,
                topic_label=TOPIC_LABEL,
                turns_per_agent=1,
                opinion_a=expressed_a,
                opinion_b=expressed_b,
            )
            logger.log_discussion(round_n, expresser.name, responder.name, result)

            # 4. Q-update: expresser only
            update_q_value(
                opinion_states[expresser.name],
                expressed_a,
                result["reward_a"],
                LEARNING_RATE,
            )

            q = opinion_states[expresser.name]
            print(
                f"    Q-gap {expresser.name}: {q.q_gap:+.3f} "
                f"(modal→{q.expressed_opinion:+d})"
            )

            # 6. Strength update (GRAPH_DYNAMIC only)
            if GRAPH_DYNAMIC:
                strength[expresser.name] = max(0.0, min(
                    STRENGTH_CAP,
                    strength[expresser.name] + result["score_a"] * STRENGTH_DELTA,
                ))
                strength[responder.name] = max(0.0, min(
                    STRENGTH_CAP,
                    strength[responder.name] + result["score_b"] * STRENGTH_DELTA,
                ))
                print(
                    f"    Strength  {expresser.name}: {strength[expresser.name]:.2f}  "
                    f"│  {responder.name}: {strength[responder.name]:.2f}"
                )
                if (strength[expresser.name] <= STRENGTH_FLOOR
                        or strength[responder.name] <= STRENGTH_FLOOR):
                    print("\n  ✋ Conversation ended: relationship strength reached floor.")
                    terminated_early = True
                    break

        if terminated_early:
            break

        # ── Reflection phase ─────────────────────────────────────────────
        if round_n % REFLECT_EVERY == 0:
            print(f"\n── Reflection phase (round {round_n}) ──")
            for ag in agents:
                ag.reflect()
                logger.log_reflection(round_n, ag.name)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n\n{'━' * 50}")
    print("Simulation complete")

    final_pol = compute_polarization_metrics(opinion_states)
    print(
        f"  Opinions +1   : {final_pol.get('n_pos', '?')}  /  "
        f"Opinions −1: {final_pol.get('n_neg', '?')}"
    )
    print(f"  Dispersion    : {final_pol.get('dispersion', '?')}")
    print(f"  Mean |Q-gap|  : {final_pol.get('mean_q_gap', '?')}")
    if GRAPH_DYNAMIC:
        print(
            f"  Final strengths: {a_name} {strength[a_name]:.2f} "
            f"/ {b_name} {strength[b_name]:.2f}"
        )
    print(f"  Logs written  : {logger.run_dir}")
    print(f"{'━' * 50}\n")


if __name__ == "__main__":
    main()
