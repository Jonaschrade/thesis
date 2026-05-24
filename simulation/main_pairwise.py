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
  drives both the Q-update and the strength update.
- Reflection: every ``REFLECT_EVERY`` rounds.

``INTERACTIONS_PER_ROUND`` controls the number of asymmetric interactions per
round — the expresser is drawn uniformly at random each time (mirroring network
mode).  With two agents the responder is always the other agent.

Relationship strength (GRAPH_DYNAMIC = True)
---------------------------------------------
When ``GRAPH_DYNAMIC`` is enabled, the expresser's ``reward_a`` adjusts the
expresser's relationship strength by ``reward_a × STRENGTH_DELTA`` each
interaction (asymmetric — only the expresser evaluates the channel).  The
simulation ends early if either agent's strength reaches ``STRENGTH_FLOOR``.
With ``GRAPH_DYNAMIC = False`` (default), only Q-dynamics run.

Note: pairwise mode applies rewards directly per interaction rather than
through a rolling window (no ``EdgeData``/deques).  For symmetric strength
updates, pass ``reward_b`` for the responder — currently commented out.

Note on homophily h
-------------------
``HOMOPHILY_H`` has no effect in pairwise mode: with exactly two agents there
is only one possible responder, so the weighted draw degenerates to a fixed
choice.

Round structure
---------------
For each round (INTERACTIONS_PER_ROUND asymmetric interactions; expresser drawn uniformly):
  For each interaction:
    1. Softmax draw    expressed = softmax(β) over expresser's Q-values
    2. Exchange        expresser speaks → responder reacts  (1 exchange)
    3. Reward          classify_reward(responder's message) → reward_a
    4. Q-update        Q(expressed) ← (1−α)·Q(expressed) + α·reward_a  [expresser only]
    5. Strength        [GRAPH_DYNAMIC only] strength[expresser] += reward_a × STRENGTH_DELTA
  After all interactions:
    6. Reflection      if round % REFLECT_EVERY == 0

Configuration
-------------
All tunable parameters live in ``config.py``.
"""

import random

from langchain_ollama import OllamaLLM

from agents.agent import Agent
from agents.personas import sample_personas
from config import (
    GRAPH_DYNAMIC,
    INTERACTIONS_PER_ROUND,
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
        n_pos = sum(1 for s in opinion_states.values() if s.preferred_opinion == 1)
        n_neg = len(agents) - n_pos
        strength_str = (
            f"  │  strengths: {a_name} {strength[a_name]:.2f} / {b_name} {strength[b_name]:.2f}"
            if GRAPH_DYNAMIC else ""
        )

        print(f"\n{'━' * 50}")
        print(
            f"Round {round_n} / {NETWORK_MAX_ROUNDS}  "
            f"│  topic: {TOPIC_LABEL}  "
            f"│  opinions: +{n_pos} / −{n_neg}"
            f"{strength_str}"
        )
        print(f"{'━' * 50}")

        # INTERACTIONS_PER_ROUND asymmetric interactions; expresser drawn uniformly each time
        for interaction_i in range(INTERACTIONS_PER_ROUND):
            expresser = random.choice(agents)
            responder = agents[1] if expresser is agents[0] else agents[0]

            # 1. Softmax draw
            expressed_a = softmax_opinion(opinion_states[expresser.name], OPINION_BETA)
            expressed_b = softmax_opinion(opinion_states[responder.name], OPINION_BETA)

            print(
                f"\n  [{interaction_i + 1}/{INTERACTIONS_PER_ROUND}]  "
                f"{expresser.name} → {responder.name}  "
                f"(stances: {expressed_a:+d} / {expressed_b:+d})"
            )

            # 2–4. Exchange, reward, Q-update
            result = run_discussion(
                expresser,
                responder,
                TOPIC_TEXT,
                topic_label=TOPIC_LABEL,
                turns_per_agent=1,
                opinion_a=expressed_a,
                opinion_b=expressed_b,
            )
            result["preferred_a"] = opinion_states[expresser.name].preferred_opinion
            result["preferred_b"] = opinion_states[responder.name].preferred_opinion
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
                f"(expressed→{expressed_a:+d}, preferred→{q.preferred_opinion:+d})"
            )

            # 5. Strength update (GRAPH_DYNAMIC only) — asymmetric: expresser only
            if GRAPH_DYNAMIC:
                strength[expresser.name] = max(0.0, min(
                    STRENGTH_CAP,
                    strength[expresser.name] + result["reward_a"] * STRENGTH_DELTA,
                ))
                # strength[responder.name] unchanged (asymmetric mode)
                # enable for symmetric: + result["reward_b"] * STRENGTH_DELTA
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
    print(f"  Interactions  : {NETWORK_MAX_ROUNDS * INTERACTIONS_PER_ROUND}")

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
