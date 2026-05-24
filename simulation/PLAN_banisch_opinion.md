# Banisch & Olbrich Opinion State — Implementation Notes

This document records the design decisions made when implementing the SFT Q-value layer. It supersedes the original plan, which proposed a simpler wiring of the existing `evaluate()` score into a Q-update. The implementation differs in two significant ways: the reward source, and the opinion initialisation strategy.

---

## Theoretical background

Banisch & Olbrich (2019) model opinion dynamics as a Q-learning process. Each agent holds two private Q-values — `q_pos` and `q_neg` — representing how rewarding it has been to express a pro or contra stance in past interactions. The *preferred* opinion is `argmax(q_pos, q_neg)` — the deterministic indicator of which stance the agent currently favours; the *expressed* opinion in any given interaction is drawn stochastically by softmax with inverse temperature β and may differ from the preferred opinion when conviction is low. After each interaction the Q-value for the expressed (softmax-drawn) stance is updated toward the social reward received:

```
Q(o_i) ← (1 − α) · Q(o_i) + α · r
```

where r ∈ {−1, +1} in the original binary formulation.

---

## What was built and where

| Component | File | Notes |
|---|---|---|
| `AgentOpinionState` dataclass | `network/opinion.py` | `q_pos`, `q_neg`; `preferred_opinion` property (argmax — stance preferred given current Q-values, not necessarily the stance expressed in a given interaction); `q_gap` property |
| `init_opinion_states()` | `network/opinion.py` | Initialises all agents at Q = (0, 0); no LLM call at startup |
| `update_q_value(expressed, reward, α)` | `network/opinion.py` | TD update for the Q-value of the *actual drawn stance* (`expressed` passed explicitly) |
| `softmax_opinion(β)` | `network/opinion.py` | Stochastic draw: β=0 → 50/50; β→∞ → argmax; uses logistic form |
| `compute_polarization_metrics()` | `network/opinion.py` | `n_pos`, `n_neg`, `dispersion`, `mean_q_gap` |
| `opinion_states_to_dict()` | `network/opinion.py` | Serialises Q-trajectories for JSON logging |
| `opinion_states` field | `network/state.py` | `dict[str, AgentOpinionState]` on `NetworkState` |
| Reward classification | `agents/agent.py` | `classify_reward(reaction_text)` — new separate method |
| Opinion-conditioned response | `agents/agent.py` | `respond(..., expressed_opinion=None)` — optional stance anchor |
| Discussion wiring | `network/discussion.py` | Passes `opinion_a/b` to `respond()`; calls `classify_reward()` after turns |
| Main loop wiring (network) | `main_network.py` | asymmetric draw → `softmax_opinion(β)` → `run_discussion(turns=1)` → `update_q_value(expressed, …)` → snapshot |
| Main loop wiring (pairwise) | `main_pairwise.py` | same SFT mechanisms; `INTERACTIONS_PER_ROUND` interactions/round (expresser drawn uniformly); no graph or community |
| Homophily selection | `network/matching.py` | `select_responder(h)` — h=0 uniform, h>0 conviction-similarity weighted |

---

## Key design decisions and divergences from the original plan

### 1. Reward source: classifier, not `evaluate()`

**Original plan:** pass `score_a/b` from `evaluate()` directly as the reward.

**Actual implementation:** a separate `classify_reward(reaction_text)` method, called on the partner's last message only, with no persona or full-transcript context.

**Why:** `evaluate()` is called by the agent that just spoke, on the transcript it was part of generating. This self-scoring couples expression and evaluation — the generating LLM scores its own interactional outcome in a shared context, contaminating the causal chain. Sárközi et al. (2022) specifically found that feedback processing is the critical link; keeping it causally independent from expression is necessary for the interpretability claim to hold.

`classify_reward()` uses a minimal prompt: the partner's last message, no persona, no history. This means the reward is a clean function of what the partner said, not of what the scoring agent expected to hear.

`evaluate()` is retained in the `Agent` class [legacy] but is no longer called by any active simulation path. Its role in edge dynamics has been superseded by the reward-history mechanism (`network/edges.py`): edge valuation now derives from the rolling mean of `classify_reward()` outputs accumulated in `EdgeData.reward_history`, eliminating a separate LLM concordance judgment per interaction.

### 2. Opinion initialisation: neutral, not LLM-bootstrapped

**Original plan:** call the LLM for each agent to get a `ja/nein` answer and initialise Q-values with a small random offset in the stated direction.

**Actual implementation:** all agents initialise at Q(+1) = Q(−1) = 0.0.

**Why:** LLM-bootstrapped opinions introduce model bias (the left-lean and truth-bias documented by Chuang et al. 2024) at the very first time step, before any social interaction. This conflates model bias with social dynamics and makes the Q-trajectories harder to interpret. Starting at zero means any divergence in Q-values is entirely caused by the social feedback received, which is exactly what the interpretability check measures. Agents may still start expressing different stances by round 2 due to randomness in the softmax draw and asymmetric early rewards.

### 3. Graph initialisation: SBM instead of Watts-Strogatz

The original plan did not specify a graph topology change. The implementation switches to a Stochastic Block Model (SBM) with `SBM_NUM_COMMUNITIES` blocks and a tunable `SBM_P_INTER` coupling parameter. This is necessary because the phase-transition baseline sanity check (Banisch & Olbrich 2019) requires sweeping community modularity as a controlled variable. Watts-Strogatz rewiring probability does not give a clean inter-group coupling knob.

### 4. Asymmetric interaction rule: one expresser, one Q-update

**Previous implementation:** symmetric pairs matched by max-weight matching; both agents' Q-values updated after every discussion.

**Actual implementation:** each interaction draws one expresser uniformly at random, one responder from the expresser's neighbourhood, runs one exchange (`turns_per_agent=1`), and updates only the expresser's Q-value.

**Why:** Jacob & Banisch (2023) establish that the one-directional update is what produces the asymmetric social-feedback dynamics: agents shift their opinion based on the reaction they receive when speaking, not when listening. The symmetric bilateral update obscures this directionality and changes the phase-transition behaviour. Using `turns_per_agent=1` in `run_discussion()` matches the single-exchange base unit of the model; `INTERACTIONS_PER_ROUND` (default = `NUM_AGENTS`) defines how many such events constitute a snapshot round.

### 5. Softmax inverse temperature β replaces temperature τ

**Previous implementation:** `softmax_opinion(temperature)` — τ=0 collapses to argmax, τ>0 adds noise.

**Actual implementation:** `softmax_opinion(beta)` — uses the logistic form `p(+1) = σ(β · Δq)`.

**Why:** Banisch uses inverse temperature throughout. More importantly, the parameterisation change fixes the initialisation artifact: at β=0 (or equivalently β·Δq=0 when Q-values are tied at 0), `σ(0) = 0.5`, so agents are split 50/50 at initialisation without any random jitter. The old τ=0 path returned argmax and sent every agent to +1. The `expressed` stance is now passed explicitly to `update_q_value()` so that the Q-update targets the stance that was actually drawn, even if it differed from the modal argmax.

### 6. Homophily partner-selection bias h

**Previous implementation:** responder drawn uniformly from graph neighbours.

**Actual implementation:** `select_responder(expresser, neighbours, opinion_states, h)` weights each neighbour by `exp(−h · |Δq_i − Δq_j|)`.

**Why:** Jacob & Banisch (2023) show that homophilic partner selection is the mechanism by which community structure amplifies into stable echo chambers. At h=0 the function is exactly uniform, recovering the 2019 baseline. At h>0, similar-conviction neighbours are more likely to be selected. Keeping this as a standalone swappable function means the virtual-worlds extension (replacing the neighbour set with a cross-platform adjacency list) requires no changes to the main loop.

---

## Output format

Every `round_NNNN.json` snapshot now contains:

```json
{
  "metrics": {
    "density": 0.33,
    "n_components": 1,
    "avg_degree": 2.0,
    "n_edges": 4,
    "n_pos": 3,
    "n_neg": 1,
    "dispersion": 0.75,
    "mean_q_gap": 0.12
  },
  "opinion_states": {
    "Anna": {"q_pos": 0.18, "q_neg": 0.04, "preferred": 1, "expressed": 1, "q_gap": 0.14},
    "Ben":  {"q_pos": -0.02, "q_neg": 0.11, "preferred": -1, "expressed": -1, "q_gap": -0.13}
  }
}
```

`opinion_states` records the full Q-trajectory for every agent, enabling the interpretability check: regressing Q-gap trajectories against observed opinion switches to test whether the SFT layer genuinely governs the LLM's expressed positions.  `"preferred"` is the deterministic argmax (`preferred_opinion` property); `"expressed"` is the actual softmax-drawn stance from the agent's last interaction as expresser in the round (absent if not selected as expresser).

Each `discussion` record in `events.jsonl` also carries all four stance fields for the expresser (a) and responder (b):

| Field | Meaning |
|---|---|
| `expressed_a` / `expressed_b` | Actual softmax-drawn stances for this interaction |
| `preferred_a` / `preferred_b` | Deterministic argmax (pre-Q-update) going into this interaction |

This allows per-interaction analysis without cross-referencing the round snapshot.

---

## What remains for the extension chapter

Setting `GRAPH_DYNAMIC = True` in `config.py` activates the homophilic tie-formation mechanism:

- Each interaction appends `reward_a` to the expresser's rolling history deque on that edge (`EdgeData.reward_history`, window length `REWARD_WINDOW_M` from `config.py`)
- The rolling mean of that history drives the expresser's valuation update via `update_edge()` (asymmetric — responder's side unchanged; enable symmetric mode by passing `reward_b` at the call site)
- Edges are severed when either agent's valuation reaches `STRENGTH_FLOOR`
- `ensure_connectivity()` reconnects isolated agents

This corresponds to Jacob & Banisch's (2023) virtual-worlds setup where structural co-evolution is driven by the same reward signal as opinion learning — `classify_reward()` now serves dual duty: Q-value update and edge evaluation. Known design risks: (1) graph crystallising before Q-values diverge — keep `STRENGTH_DELTA` small relative to `LEARNING_RATE`; (2) premature edge drops before the history window fills — cold-start signal defaults to 0.0 (neutral).

---

## References

Banisch, S., & Olbrich, E. (2019). Opinion polarization by learning from social feedback. *The Journal of Mathematical Sociology, 43*(2), 76–103.

Jacob, D., & Banisch, S. (2023). Polarization in social media: A virtual worlds-based approach. *Journal of Artificial Societies and Social Simulation, 26*(3), 11.

Sárközi, R., Denz, T., & Lorenz-Spreen, P. (2022). Testing social feedback theory: An experiment on the effect of social feedback on opinion expression. *PLOS ONE, 17*(4).

Chuang, Y.-S., Goyal, A., Harlalka, N., Suresh, S., Hawkins, R., Yang, S., ... & Yang, D. (2024). Simulating opinion dynamics with networks of LLM-based agents. *NAACL 2024*.
