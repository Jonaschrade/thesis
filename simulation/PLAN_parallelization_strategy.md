# Plan: Parallelization Strategy

## Context

The simulation is currently fully sequential: within each round, pairs are processed one after another, and within each discussion, LLM calls are issued one at a time. For a 100-agent simulation (50 pairs/round, DISCUSSION_TURNS=5, NETWORK_MAX_ROUNDS=20) the sequential runtime is approximately **63 hours**. This plan reduces that to under 4 hours on a 10 GB H100 slice.

---

## Where Time Is Spent

```
Per discussion (2 agents, DISCUSSION_TURNS=5 exchanges):
  respond() calls  : 5 × 2 = 10 LLM calls
  evaluate() calls :         0 LLM calls  (removed — edge eval now uses reward history)
  Total            :        10 LLM calls × 17.5 s avg = 175 s ≈ 2.9 min

Per round (50 pairs, N=100):
  Discussions      : 50 × 175 s = 8 750 s ≈ 2.4 hours  (sequential)
  Reflections      : (every 2nd round) 100 agents × ~4 LLM calls × 17.5 s = 7 000 s ≈ 1.9 hours total
  Snapshots/logging: < 1 s                               (negligible)

Total sequential   : 20 rounds × 2.4 h + 10 reflect rounds × 1.9 h ÷ 10 ≈ 51 hours
```

---

## Parallelism Opportunities

| Opportunity | Independence | Leverage |
|---|---|---|
| **Discussions across pairs** | Fully independent (separate agent memory stores, separate edges) | Very high — 50× in theory |
| **Reflections across agents** | Each agent's memory store is isolated | Up to 100× |
| **Turns within a discussion** | Strictly sequential (B's turn depends on A's reply) | Not parallelisable |

Note: `evaluate()` was previously listed as a 2× parallelism opportunity within each discussion. It has been removed from `run_discussion()` — edge evaluation now uses `classify_reward()` outputs accumulated in `EdgeData.reward_history`, saving 2 LLM calls per interaction without any parallelism work required.

---

## Hardware Budget: 10 GB H100 Slice

| Model | Q4 VRAM | Remaining KV cache | Est. concurrent requests |
|---|---|---|---|
| `qwen2.5:14b` | ~9.0 GB | ~1.0 GB | 1–2 |
| `qwen2.5:7b` | ~4.5 GB | ~5.5 GB | 4–6 |
| `qwen2.5:3b` | ~2.0 GB | ~8.0 GB | 10–12 |

**Practical recommendation**: switch to `qwen2.5:7b` on the H100. The quality loss is small for opinion-bearing German responses; the parallelism gain is 4–6×. For a pilot run use `qwen2.5:3b` to validate correctness at full speed.

---

## Two-Phase Implementation Plan

**Phase 1** — Thread-level parallelism, Ollama backend unchanged.  
Works immediately; bottleneck shifts to Ollama's sequential GPU scheduling.  
Speedup: **2–3×** (Ollama batches poorly, but I/O overlap helps).

**Phase 2** — Switch LLM backend to vLLM.  
vLLM serves an OpenAI-compatible endpoint and batches concurrent requests at the GPU level via continuous batching. Combines with Phase 1 threads.  
Speedup: **8–15×** over baseline (depending on model size and KV cache).

Both phases are additive. Phase 1 can be deployed immediately; Phase 2 requires starting a vLLM server.

---

## Phase 1: Thread-Level Parallelism

### Changes overview

```
config.py           — add MAX_DISCUSSION_WORKERS, MAX_REFLECT_WORKERS
network/logger.py   — add threading.Lock to _write()
main_network.py     — wrap pair loop in ThreadPoolExecutor; parallelize reflection phase
```

No changes to `agents/`, `memory/`, `network/discussion.py`, `network/edges.py`, `network/matching.py`, `network/state.py`.

---

### `config.py` additions

```python
# ── Parallelism ────────────────────────────────────────────────────────────────
# Maximum number of concurrent pairwise discussions per round.
# With Ollama (sequential GPU): 2–4 is the practical ceiling before requests queue.
# With vLLM (continuous batching): raise to match VRAM-limited concurrent request count.
MAX_DISCUSSION_WORKERS = 4   # concurrent pair discussions
MAX_REFLECT_WORKERS    = 8   # concurrent agent reflections
```

---

### `network/logger.py` — thread-safe write

The only shared mutable state accessed from multiple threads is `events.jsonl`. Add a lock to `_write()`. All other logger methods write to per-round files called only from the main thread.

```python
import threading

class SimulationLogger:
    def __init__(self, run_id: str | None = None) -> None:
        ts = run_id or str(int(time.time()))
        self.run_dir   = Path("logs") / f"run_{ts}"
        self.rounds_dir = self.run_dir / "network_rounds"
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self._lock = threading.Lock()          # guards events.jsonl appends

    def _write(self, record: dict) -> None:
        with self._lock:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

No other method needs changes: `log_discussion`, `log_edge_event`, `log_reflection` all call `_write`, so they inherit the lock. `snapshot_network` writes to per-round files — called sequentially from the main thread, no lock needed.

---

### `network/discussion.py` — no intra-discussion parallelism needed

`evaluate()` has been removed from `run_discussion()`. The only per-discussion parallelism opportunity that remains is the two `classify_reward()` calls — these are independent and could run concurrently, but each is a single short LLM call and the gain is minimal compared to the inter-discussion parallelism below. No changes to `discussion.py` are required for Phase 1.

---

### `main_network.py` — parallel pair loop and reflection

The pair loop submits each `run_discussion()` to a thread pool. Results are collected only after all futures complete, then edges are updated sequentially in the main thread (NetworkX graph mutations are not thread-safe).

The reflection phase submits each `agent.reflect()` to a separate pool.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import MAX_DISCUSSION_WORKERS, MAX_REFLECT_WORKERS

# ... (inside main(), replacing the pair loop) ...

for round_n in range(1, NETWORK_MAX_ROUNDS + 1):
    state.round = round_n
    topic_label, topic_text = _topic_for_round(round_n)
    # ... round header print ...

    pairings = compute_pairings(state)

    # ── Parallel discussions ─────────────────────────────────────────────
    # Submit all pair discussions concurrently. NetworkX graph reads
    # (agent objects, memory retrieval) are safe across threads. Graph
    # mutations (update_edge, ensure_connectivity) happen after all futures
    # resolve, in the main thread.
    futures = {}
    with ThreadPoolExecutor(max_workers=MAX_DISCUSSION_WORKERS) as pool:
        for agent_a_name, agent_b_name in pairings:
            fut = pool.submit(
                run_discussion,
                agents[agent_a_name],
                agents[agent_b_name],
                topic_text,
                topic_label=topic_label,
            )
            futures[fut] = (agent_a_name, agent_b_name)

    # ── Sequential edge updates (graph mutation — not thread-safe) ───────
    for fut, (agent_a_name, agent_b_name) in futures.items():
        result = fut.result()
        logger.log_discussion(round_n, agent_a_name, agent_b_name, result)

        survived = update_edge(
            state, agent_a_name, agent_b_name,
            reward_a=result["reward_a"],
            # reward_b=result["reward_b"],  # enable for symmetric mode
        )
        event_type = "edge_maintained" if survived else "edge_dropped"
        logger.log_edge_event(round_n, event_type, agent_a_name, agent_b_name)

        status = "✔ maintained" if survived else "✘ dropped"
        if survived:
            edge = state.graph[agent_a_name][agent_b_name]["data"]
            str_a = edge.strengths[agent_a_name]
            str_b = edge.strengths[agent_b_name]
            strength_info = f"  |  strengths {str_a:.2f} / {str_b:.2f}"
        else:
            strength_info = ""
        print(f"  ▶ {agent_a_name} ↔ {agent_b_name}  "
              f"reward_a={result['reward_a']:+.2f}"
              f"{strength_info}  |  edge {status}")

    ensure_connectivity(state)

    # ── Parallel reflection ──────────────────────────────────────────────
    if round_n % REFLECT_EVERY == 0:
        print(f"\n── Reflection phase (round {round_n}) ──")
        with ThreadPoolExecutor(max_workers=MAX_REFLECT_WORKERS) as pool:
            reflect_futures = {
                pool.submit(agent.reflect): name
                for name, agent in agents.items()
            }
            for fut in as_completed(reflect_futures):
                name = reflect_futures[fut]
                fut.result()  # re-raises any exception
                logger.log_reflection(round_n, name)

    logger.snapshot_network(state)
```

---

### Thread-safety audit

| Resource | Concurrent access? | Safe? | Reason |
|---|---|---|---|
| `Agent.llm` (OllamaLLM) | Multiple threads | Yes | Stateless HTTP client; each call is independent |
| `Agent.memory` (ChromaDB) | Per-agent — no sharing | Yes | Each agent has its own collection |
| `Agent.respond()` | Different agents only | Yes | Writes to calling agent's own store |
| `Agent.evaluate()` | Different agents only | Yes | No `_store` call; reads transcript arg only |
| `Agent.reflect()` | Different agents only | Yes | Writes to calling agent's own store |
| `NetworkState.graph` | Main thread only (during mutations) | Yes | Reads in futures (during respond) are safe; mutations happen after all futures complete |
| `SimulationLogger._write()` | Multiple threads | Yes | Protected by `threading.Lock` |
| `SimulationLogger.snapshot_network()` | Main thread only | Yes | Called after future collection |

**Critical invariant**: `update_edge()` and `ensure_connectivity()` must never be called from inside a thread pool worker. They mutate `state.graph`. The pattern above — collect all futures, then update edges sequentially — preserves this.

---

## Phase 2: vLLM Backend

### Why vLLM

Ollama processes one request at a time per model. Ten threads with Ollama means ten sequential GPU calls — threads wait in queue and actual speedup is limited by Ollama's server-side serialization. vLLM uses **continuous batching**: incoming requests are dynamically batched mid-generation, fully utilising the GPU across multiple concurrent requests. This is the difference between 2–3× and 8–15× speedup.

### Server setup (JupyterHub terminal)

```bash
pip install vllm

# 7b model — recommended for 10 GB slice
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --port 8000 &

# Or 14b model — tight fit, lower concurrency
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 \
    --dtype float16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.95 \
    --port 8000 &
```

vLLM exposes an OpenAI-compatible REST API at `http://localhost:8000/v1`.

### `config.py` additions for Phase 2

```python
# ── LLM backend ───────────────────────────────────────────────────────────────
LLM_BACKEND  = "vllm"                   # "ollama" | "vllm"
VLLM_HOST    = "127.0.0.1:8000"
VLLM_MODEL   = "Qwen/Qwen2.5-7B-Instruct"

# Raise these when using vLLM (continuous batching supports more concurrent requests)
MAX_DISCUSSION_WORKERS = 10   # raise from 4 to match model concurrency
MAX_REFLECT_WORKERS    = 20
```

### `agents/agent.py` — backend switch

Replace `OllamaLLM` with an OpenAI-compatible client when `LLM_BACKEND == "vllm"`. The simplest approach keeps the same `Agent` interface by wrapping the call in a compatibility shim — no changes to `run_discussion`, `main_network.py`, or anything else.

```python
from openai import OpenAI
from config import LLM_BACKEND, VLLM_HOST, VLLM_MODEL, EMBED_MODEL, OLLAMA_HOST

class Agent:
    def __init__(self, name: str, persona: str, llm):
        self.name    = name
        self.persona = persona
        self.llm     = llm      # OllamaLLM or None (vLLM path uses _llm_invoke)
        self.memory  = MemoryStore(name)

        if LLM_BACKEND == "vllm":
            self._vllm = OpenAI(
                base_url=f"http://{VLLM_HOST}/v1",
                api_key="none",    # vLLM does not require a real key
            )
        else:
            self._vllm = None

    def _llm_invoke(self, prompt: str) -> str:
        """Single LLM call, routing to vLLM or Ollama based on config."""
        if self._vllm is not None:
            resp = self._vllm.chat.completions.create(
                model=VLLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512,
            )
            return resp.choices[0].message.content.strip()
        return self.llm.invoke(prompt).strip()
```

Then replace every `self.llm.invoke(...)` call in `respond()`, `reflect()`, `evaluate()`, and `_score_importance()` with `self._llm_invoke(...)`.

The `main_network.py` and `main.py` entry points pass `llm=llm` on `Agent` construction. When `LLM_BACKEND == "vllm"`, the `OllamaLLM` instance is still constructed but ignored — or the construction can be skipped:

```python
# main_network.py
if LLM_BACKEND == "vllm":
    llm = None     # vLLM path uses OpenAI client internally
else:
    llm = OllamaLLM(model=LLM_MODEL, base_url=f"http://{OLLAMA_HOST}")
```

### Embedding with vLLM

vLLM does not serve the `nomic-embed-text` embedding model. Two options:

1. **Keep Ollama for embeddings only** — run Ollama alongside vLLM; `Agent._embed()` still calls the Ollama embeddings endpoint. This is the simplest approach: embeddings are fast (< 100 ms) and not the bottleneck.

2. **Use a HuggingFace sentence-transformers model directly** — avoids Ollama entirely:
   ```python
   from sentence_transformers import SentenceTransformer
   _embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
   
   def _embed(self, text: str) -> list:
       return _embed_model.encode(text).tolist()
   ```
   The model runs on CPU and does not consume VRAM. Load it once at module level.

**Recommendation**: keep Ollama for embeddings (Option 1). It requires no code change to `_embed()` and avoids adding a dependency.

---

## Runtime Estimates After Parallelization

Configuration: N=100, DISCUSSION_TURNS=5, NETWORK_MAX_ROUNDS=20, REFLECT_EVERY=2

| Phase | Backend | Workers | Concurrency | Est. runtime |
|---|---|---|---|---|
| Baseline (sequential) | Ollama 14b | 1 | 1 | ~51 hours |
| Phase 1 only | Ollama 14b | 4 | ~2 effective | ~20–25 hours |
| Phase 1 only | Ollama 7b | 6 | ~3 effective | ~15–18 hours |
| Phase 1 + 2 | vLLM 14b Q4 | 4 | 4 batched | ~13 hours |
| Phase 1 + 2 | vLLM 7b | 10 | 8–10 batched | **3–5 hours** |
| Phase 1 + 2 | vLLM 3b | 20 | 12–15 batched | **2–3 hours** |

Note: estimates updated for 10 LLM calls/discussion (was 12 — `evaluate()` removed).

The 7b model is the practical target: 4–6 hours is feasible for overnight runs; quality is sufficient for concordance detection on clear opinion statements.

---

## Implementation Order

1. `config.py` — add parallelism constants (Phase 1)
2. `network/logger.py` — add `threading.Lock` (Phase 1, required before any parallel writes)
3. `main_network.py` — parallel pair loop and reflection (Phase 1)
4. Smoke test: `NUM_AGENTS=4, NETWORK_MAX_ROUNDS=3` — verify output identical to sequential
5. `config.py` — add vLLM constants (Phase 2)
6. `agents/agent.py` — add `_llm_invoke()` and vLLM client (Phase 2)
7. Start vLLM server, set `LLM_BACKEND="vllm"`, rerun smoke test

---

## Verification Checklist

- [ ] Sequential run and Phase 1 run produce identical `events.jsonl` structure (same fields, valid JSON)
- [ ] `network_rounds/round_NNNN.json` snapshots are structurally identical before and after parallelization
- [ ] No `RuntimeError` or `ChroaDB` corruption under concurrent access (run with `NUM_AGENTS=20, MAX_DISCUSSION_WORKERS=10`)
- [ ] `events.jsonl` is not corrupted (no interleaved partial lines) — verify with `jq -c '.' events.jsonl`
- [ ] Edge update order is deterministic within a round (futures resolved in pairings order, not completion order)
- [ ] After switching to vLLM: `evaluate()` scores are plausible (not systematically 0.0 or parse-failing)
- [ ] GPU utilisation visible in `nvidia-smi` during concurrent discussion phase (~80–95% with vLLM 7b)
