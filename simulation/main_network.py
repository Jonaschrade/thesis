"""
Network simulation entry point.

Implements the asymmetric interaction model from Jacob & Banisch (2023) on a
single Stochastic Block Model (SBM) graph.  Agent opinion dynamics are
governed by Social Feedback Theory (Banisch & Olbrich 2019): each agent holds
Q-values over opinion stances (+1 / −1), expresses a stance drawn by softmax
with inverse temperature β, and updates that Q-value via a TD rule after each
interaction.

Interaction rule (asymmetric)
------------------------------
The unit of dynamics is a single asymmetric interaction:

  1. Draw one expresser uniformly from agents with at least one neighbour.
  2. Draw one responder from the expresser's neighbours with homophily bias h
     (h = 0 recovers uniform, replicating Banisch & Olbrich 2019).
  3. Expresser's stance is drawn by softmax(β).
  4. One exchange: expresser speaks, responder reacts.
  5. classify_reward() on the responder's reaction → reward r.
  6. Q-update for the expresser only: Q(o_i) ← (1−α)·Q(o_i) + α·r.

A "round" groups INTERACTIONS_PER_ROUND such events for snapshotting.

Experimental variables
-----------------------
SBM_P_INTER   — between-community coupling; sweep for polarization phase transition
HOMOPHILY_H   — partner-selection bias; 0 = uniform (2019 baseline), >0 = 2023 model
OPINION_BETA  — softmax inverse temperature; β=0 is 50/50, β→∞ is argmax

Configuration
-------------
All tunable parameters live in ``config.py``.
"""

import random
from collections import deque

import networkx as nx
from langchain_ollama import OllamaLLM

from agents.agent import Agent
from agents.personas import sample_personas
from config import (
    GRAPH_DYNAMIC,
    HOMOPHILY_H,
    INTERACTIONS_PER_ROUND,
    LEARNING_RATE,
    LLM_MODEL,
    NETWORK_MAX_ROUNDS,
    NUM_AGENTS,
    OLLAMA_HOST,
    OPINION_BETA,
    REFLECT_EVERY,
    REWARD_WINDOW_M,
    SBM_NUM_COMMUNITIES,
    SBM_P_INTER,
    SBM_P_INTRA,
    STRENGTH_CAP,
    STRENGTH_DELTA,
    STRENGTH_FLOOR,
    TOPIC_LABEL,
    TOPIC_TEXT,
)
from network.discussion import run_discussion
from network.edges import update_edge
from network.logger import SimulationLogger
from network.matching import ensure_connectivity, select_responder
from network.opinion import (
    compute_polarization_metrics,
    init_opinion_states,
    opinion_states_to_dict,
    softmax_opinion,
    update_q_value,
)
from network.state import EdgeData, NetworkState


def _distribute_sizes(n: int, k: int) -> list[int]:
    """Divide n agents as evenly as possible into k communities."""
    base, remainder = divmod(n, k)
    return [base + (1 if i < remainder else 0) for i in range(k)]


def _build_initial_graph(agent_names: list[str]) -> nx.Graph:
    """Create a Stochastic Block Model graph over the agent name list.

    Nodes are relabelled from integers to agent names.  Community membership
    is stored as a node attribute ``"community"`` for post-hoc analysis.
    Every edge is initialised with a fresh ``EdgeData`` instance at strength 1.0.

    ``SBM_P_INTER`` is the primary experimental variable: sweep it to reproduce
    SFT's polarization-to-consensus phase transition.
    """
    n = len(agent_names)
    sizes = _distribute_sizes(n, SBM_NUM_COMMUNITIES)
    p_matrix = [
        [SBM_P_INTRA if i == j else SBM_P_INTER for j in range(SBM_NUM_COMMUNITIES)]
        for i in range(SBM_NUM_COMMUNITIES)
    ]

    G_int = nx.stochastic_block_model(sizes, p_matrix)
    G = nx.relabel_nodes(G_int, {i: agent_names[i] for i in range(n)})

    offset = 0
    for comm_idx, size in enumerate(sizes):
        for i in range(offset, offset + size):
            G.nodes[agent_names[i]]["community"] = comm_idx
        offset += size

    for u, v in G.edges():
        G[u][v]["data"] = EdgeData(
            strengths={u: 1.0, v: 1.0},
            reward_history={
                u: deque(maxlen= REWARD_WINDOW_M),
                v: deque(maxlen=REWARD_WINDOW_M),
            },
        )

    return G


def main() -> None:
    """Run the full network simulation."""
    llm = OllamaLLM(model=LLM_MODEL, base_url=f"http://{OLLAMA_HOST}")
    logger = SimulationLogger()

    # ── Agent initialisation ─────────────────────────────────────────────
    print(f"\nSampling {NUM_AGENTS} personas...")
    personas = sample_personas(NUM_AGENTS, llm)
    agents: dict[str, Agent] = {
        p["name"]: Agent(name=p["name"], persona=p["persona"], llm=llm)
        for p in personas
    }

    print(f"\n{'━' * 60}")
    print("Participants")
    for name, agent in agents.items():
        print(f"  {name}: {agent.persona}")
    print(f"{'━' * 60}\n")

    logger.log_personas(agents)

    # ── Network initialisation ───────────────────────────────────────────
    G = _build_initial_graph(list(agents.keys()))
    state = NetworkState(agents=agents, graph=G, max_rounds=NETWORK_MAX_ROUNDS)
    state.opinion_states = init_opinion_states(list(agents.keys()))

    logger.snapshot_network(state)   # round-0 baseline

    community_sizes = _distribute_sizes(NUM_AGENTS, SBM_NUM_COMMUNITIES)
    print(
        f"Initial graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges  "
        f"(SBM: {SBM_NUM_COMMUNITIES} communities {community_sizes}, "
        f"p_intra={SBM_P_INTRA}, p_inter={SBM_P_INTER})\n"
        f"α={LEARNING_RATE}  β={OPINION_BETA}  h={HOMOPHILY_H}  "
        f"interactions/round={INTERACTIONS_PER_ROUND}  dynamic={GRAPH_DYNAMIC}\n"
    )

    # last message each agent sent to a specific partner, keyed by (speaker, listener)
    last_message_to: dict[tuple[str, str], str] = {}

    # ── Main simulation loop ─────────────────────────────────────────────
    for round_n in range(1, NETWORK_MAX_ROUNDS + 1):
        state.round = round_n

        n_pos = sum(1 for s in state.opinion_states.values() if s.preferred_opinion == 1)
        n_neg = NUM_AGENTS - n_pos

        print(f"\n{'━' * 60}")
        print(
            f"Round {round_n} / {NETWORK_MAX_ROUNDS}  "
            f"│  topic: {TOPIC_LABEL}  "
            f"│  edges: {state.graph.number_of_edges()}  "
            f"│  opinions: +{n_pos} / −{n_neg}"
        )
        print(f"{'━' * 60}")

        # ── INTERACTIONS_PER_ROUND asymmetric interactions ───────────────
        expressed_stances: dict[str, int] = {}  # last expressed stance per agent this round
        for interaction_i in range(INTERACTIONS_PER_ROUND):

            # 1. Draw expresser uniformly from agents with at least one neighbour
            eligible = [n for n in agents if state.graph.degree(n) > 0]
            if not eligible:
                break
            expresser_name = random.choice(eligible)

            # 2. Draw responder with homophily bias h
            neighbours = list(state.graph.neighbors(expresser_name))
            responder_name = select_responder(
                expresser_name, neighbours, state.opinion_states, HOMOPHILY_H
            )

            # 3. Draw expressed stances via softmax(β)
            expressed_a = softmax_opinion(state.opinion_states[expresser_name], OPINION_BETA)
            expressed_b = softmax_opinion(state.opinion_states[responder_name], OPINION_BETA)

            print(f"\n  [{interaction_i + 1}/{INTERACTIONS_PER_ROUND}]  "
                  f"{expresser_name} → {responder_name}  "
                  f"(stances: {expressed_a:+d} / {expressed_b:+d})")

            # 4. One exchange: expresser speaks, responder reacts
            result = run_discussion(
                agents[expresser_name],
                agents[responder_name],
                TOPIC_TEXT,
                topic_label=TOPIC_LABEL,
                turns_per_agent=1,
                opinion_a=expressed_a,
                opinion_b=expressed_b,
                prior_b_message=last_message_to.get((responder_name, expresser_name)),
            )

            # record each agent's last utterance so future exchanges can continue from it
            for turn in result["turns"]:
                other = responder_name if turn["speaker"] == expresser_name else expresser_name
                last_message_to[(turn["speaker"], other)] = turn["content"]
            result["preferred_a"] = state.opinion_states[expresser_name].preferred_opinion
            result["preferred_b"] = state.opinion_states[responder_name].preferred_opinion
            logger.log_discussion(round_n, expresser_name, responder_name, result)

            # 5. Q-update: expresser only
            update_q_value(
                state.opinion_states[expresser_name],
                expressed_a,
                result["reward_a"],
                LEARNING_RATE,
            )
            expressed_stances[expresser_name] = expressed_a

            q = state.opinion_states[expresser_name]
            print(
                f"    Q-gap {expresser_name}: {q.q_gap:+.3f} "
                f"(expressed→{expressed_a:+d}, preferred→{q.preferred_opinion:+d})"
            )

            # 6. Edge dynamics (GRAPH_DYNAMIC only)
            if GRAPH_DYNAMIC and state.graph.has_edge(expresser_name, responder_name):
                survived = update_edge(
                    state,
                    expresser_name,
                    responder_name,
                    reward_a=result["reward_a"],
                    # reward_b=result["reward_b"],  # enable for symmetric mode
                )
                event_type = "edge_maintained" if survived else "edge_dropped"
                logger.log_edge_event(round_n, event_type, expresser_name, responder_name)

        # ── Reconnect isolated agents (GRAPH_DYNAMIC only) ───────────────
        if GRAPH_DYNAMIC:
            ensure_connectivity(state)

        # ── Reflection phase ─────────────────────────────────────────────
        if round_n % REFLECT_EVERY == 0:
            print(f"\n── Reflection phase (round {round_n}) ──")
            for agent in agents.values():
                agent.reflect()
                logger.log_reflection(round_n, agent.name)

        # ── Snapshot ─────────────────────────────────────────────────────
        pol_metrics = compute_polarization_metrics(state.opinion_states)
        logger.snapshot_network(
            state,
            extra_metrics=pol_metrics,
            opinion_states=opinion_states_to_dict(state.opinion_states, expressed_stances),
        )

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n\n{'━' * 60}")
    print("Simulation complete")
    print(f"  Rounds run    : {NETWORK_MAX_ROUNDS}")
    print(f"  Interactions  : {NETWORK_MAX_ROUNDS * INTERACTIONS_PER_ROUND}")
    print(f"  Final edges   : {state.graph.number_of_edges()}")
    print(f"  Components    : {nx.number_connected_components(state.graph)}")

    final_pol = compute_polarization_metrics(state.opinion_states)
    print(f"  Opinions +1   : {final_pol.get('n_pos', '?')}  /  "
          f"Opinions −1: {final_pol.get('n_neg', '?')}")
    print(f"  Dispersion    : {final_pol.get('dispersion', '?')}")
    print(f"  Mean |Q-gap|  : {final_pol.get('mean_q_gap', '?')}")
    print(f"  Logs written  : {logger.run_dir}")
    print(f"{'━' * 60}\n")


if __name__ == "__main__":
    main()
