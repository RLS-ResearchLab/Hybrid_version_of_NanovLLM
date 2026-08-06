<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Hybrid Nano-vLLM — Qwen3.5-35B-A3B Serving Engine

*A research project under the [RLS Research Lab](https://github.com/RLS-ResearchLab) umbrella, mentored by Ghassen Fatnassi.*

A from-scratch extension of [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)
that serves **Qwen3.5-35B-A3B**, a *hybrid* architecture mixing linear-attention
(Gated Delta Rule), full attention, and Mixture-of-Experts — validated end-to-end
against a numerical reference implementation on the real 34.66B-parameter checkpoint.

---

## Why this project exists

Standard LLM-serving engines — vLLM, SGLang, nano-vLLM — are built around one
core assumption: every layer needs the same kind of per-sequence memory, a
**paged KV cache** that grows linearly with context length. That assumption
holds for a dense transformer. It does not hold for Qwen3.5-35B-A3B.

Qwen3.5-35B-A3B's 40 decoder layers alternate two fundamentally different
attention mechanisms, repeated in a fixed 4-layer block 10 times:

- **3× Gated Delta Rule (GDR) linear attention** — a recurrent, per-token
  delta-rule scan (causal depthwise conv1d, L2-normalized Q/K, a learned
  per-head decay gate, float32 state accumulation). This layer type carries a
  **fixed-size recurrent state + convolution buffer per sequence** — memory
  that does *not* grow with context length, and is architecturally unrelated
  to a KV cache.
- **1× grouped-query full attention** — standard GQA, with partial RoPE
  (only 25% of each head's dimension is rotated) and a gated output. This
  layer type still needs the conventional paged, growing **KV cache**.

Every layer additionally routes through a **256-expert MoE FFN** (top-8
routed + 1 shared expert), adding its own batching and memory-layout
concerns on top of the dual-cache problem above.

The result: a scheduler, block manager, and CUDA-graph pipeline built for
"one growing cache per sequence" cannot serve this model without real
architectural surgery — not a config flag, not a wrapper. This project does
that surgery on nano-vLLM specifically because its ~1,200-line, auditable
codebase makes it realistic to *correctly* extend within a research
internship timeframe, rather than writing a production-scale engine from
zero, or serving the model indefinitely from an unbatched reference server.

**The bet this project tests:** that a minimal, dense-transformer-oriented
reference engine can be extended — without a wholesale rewrite of its
scheduler/batching core — to correctly serve a hybrid linear-attention +
full-attention + MoE architecture under continuous batching, preemption, and
CUDA-graph decode, and that doing so productively de-risks the design of a
bespoke, higher-performance engine built for this architecture class before
committing to that harder, from-scratch system.

This is a **correctness-and-architecture-validation** project, not a
performance project. Kernel fusion, expert-parallel MoE dispatch, and
per-layer-type CUDA graphs are explicitly deferred future work — see
[Known limitations](#known-limitations--open-work) below.

---

## What was built

| Component | File | Purpose |
|---|---|---|
| `StateManager` | `engine/state_manager.py` | Fixed-size recurrent-state + conv-buffer slot pool for GDR layers, sized to `max_num_seqs` (not context length). Allocate/free wired into the scheduler at exactly the points the KV-cache block manager already fires. |
| Hybrid model | `models/qwen3_5.py` | `Qwen35FullAttention`, `Qwen35LinearAttention`, `Qwen35MoE`, full decoder stack — config-driven throughout, so it generalizes to the real checkpoint's config without code changes. |
| Packed-batch GDR scan | `models/qwen3_5.py` | Projects once over the full flat packed-token dimension; loops only over per-sequence segments for the two boundary-sensitive operations (causal conv1d, recurrent scan). |
| TP-aware sharding | `models/qwen3_5.py`, `layers/linear.py` | Every per-head GDR tensor (`in_proj_qkv`, `in_proj_a/b`, `A_log`, `dt_bias`) shards in lockstep across tensor-parallel ranks; the scan body itself is TP-agnostic by construction. |
| Batched HTTP server | `src/server.py` | `BatchedEngine` — single background thread drives `Scheduler`/`LLMEngine`, HTTP handlers only enqueue and block on a per-request event. Gated behind `--concurrency-mode {fcfs,batched}`. |

---

## Measured results

All correctness numbers below are cosine similarity against `src/model.py`
(the numerically-validated PyTorch reference, itself checked against
HuggingFace: cosine > 0.98, top-1 match 5/6) or against an isolated,
un-batched run of the same sequence, as noted.

| Check | Result | Scope |
|---|---|---|
| Full hybrid model vs. reference, single sequence | cosine 0.999967, top-1 exact | small model |
| GDR linear attention vs. reference | cosine 1.000001 | small model |
| MoE FFN vs. reference | cosine 1.000001 | small model |
| Eager vs. CUDA-graph decode, batch sizes 1/4/8 | cosine 1.000000, top-1 exact | small model |
| KV-cache memory shrinkage from full-attention-only sizing | 4.00× | measured, matches `full_attention_interval=4` exactly |
| Real-checkpoint single-forward-pass vs. HF, 5 varied prompts | cosine 0.9982–0.9997, 5/5 argmax match | **real 35B checkpoint** |
| Real-checkpoint prefill contamination, concurrency=8 | 8/8 PASS, cosine 0.9976–0.9997 | **real 35B checkpoint** |
| Real-checkpoint decode-time slot-reuse contamination | **not detected** — 3 independent lines of evidence (static code-path analysis, fake-model teacher-forced control, real-checkpoint no-reuse control) | **real 35B checkpoint** |
| MoE top-k=8 combine step, bf16, before/after fp32 promotion | 1.3% relative error → 0.000% | measured |
| Real-checkpoint throughput, batched mode, concurrency 1/2/4/8 | 6.2 → 7.0 → 10.4 → 12.6 tok/s (correctly scales with concurrency; **absolute numbers understated** — see caveat below) | real 35B checkpoint, 2×A6000 |
| GSM8K-CoT, `temperature=0`, `top_p=1.0` | 86.50% (1141/1319) — **invalidated**, predates two decode-loop fixes; re-run pending | real 35B checkpoint |
| GSM8K subset (n=32), marker-found vs. fallback accuracy | 95.7% when a clean final answer is stated, 11.1% when it falls back to "last number in text" | real 35B checkpoint |

**Throughput caveat:** the 6.2–12.6 tok/s figures were measured before
`/v1/chat/completions` supported `ignore_eos`/`stop` passthrough, so real
EOS truncated most completions well short of the 1024-token target,
suppressing the absolute number (fixed-cost overhead dominates short
completions). The scaling *shape* is trustworthy; the *magnitude* is not.
Now fixed (`ChatRequest` accepts `ignore_eos`/`stop`; `bench_http_concurrency.py`
has `--ignore-eos`) — re-run pending on H200 for a citable number.

---

## Known limitations / open work

- **Correctness gate not yet met.** GSM8K-CoT accuracy is currently blocked
  by answer-termination behavior, not reasoning quality: the model solves
  the arithmetic correctly in the large majority of cases but sometimes
  never emits a clean, extractable final-answer statement within the token
  budget. Root cause (length vs. unproductive reasoning loops) under
  investigation.
- **Batch-composition floating-point sensitivity at decode scale.** Batching
  alone — with slot reuse structurally ruled out — produced actual
  argmax-level token divergence in 2 of 5 long real-checkpoint sequences
  tested (42–128 decode steps). This is the same bf16 accumulation
  sensitivity behind the MoE combine-step fix, but newly observed to affect
  *generated tokens*, not just cosine similarity, at decode scale. Sample
  size (n=5) is too small to state a rate; needs a larger (≥20–30 sequence)
  measurement before batched-mode output is treated as reproducible across
  concurrency levels for any application that needs it (aggregate
  correctness evaluation is far more robust to this than any single-completion
  reproducibility claim).
- **`shared_expert` bf16 residual** (~0.6–0.8% isolated, cosine-based) —
  accepted as bounded based on representation similarity only; downstream
  task (GSM8K accuracy) impact not yet ablated.
- **Tensor parallelism** — shard-selection math verified in isolation
  (CPU-only, all four TP-aware `weight_loader`s), full multi-process engine
  construction at TP≥2 on real hardware not yet run.
- **Prefix caching is disabled whenever `StateManager` is active** — a
  permanent, real throughput cost: recurrent state cannot be reconstructed
  from a cached KV prefix, so any sequence reusing cached blocks would
  otherwise compute as if it had no prior context. Trades redundant prefill
  recompute for correctness.
- **Round-robin MoE expert sharding** chosen without a measured
  expert-utilization histogram.
- **Two narrow, unfixed gaps:** `load_model()` cannot accept a checkpoint
  using a model's native fused parameter name (split/HF-style names only);
  `LLMEngine`'s `atexit`-based lifecycle keeps every engine instance alive
  for the process's lifetime (`del engine` alone never frees GPU memory).

---

## Repository layout

```
engine/          sequence.py, block_manager.py, scheduler.py,
                 model_runner.py, llm_engine.py, state_manager.py
layers/          linear.py, attention.py, layernorm.py,
                 rotary_embedding.py, embed_head.py, sampler.py, activation.py
models/
  qwen3.py       existing dense Qwen3 model — untouched
  qwen3_5.py     hybrid Qwen3.5 model (this project)
utils/           context.py, loader.py

src/
  model.py               ground-truth PyTorch reference for Qwen3.5-35B-A3B
  model_small_qwen3.5.py ~290M-param scaled-down variant, same architecture
  server.py              OpenAI-compatible server, FCFS or batched mode

tests/           correctness suites — see "Running the test suites" below
                 (incl. gsm8k_full_run.py — GSM8K-CoT correctness gate)
bench_throughput.py       concurrency-sweep throughput harness (internal engine)
bench_http_concurrency.py concurrency-sweep throughput harness (HTTP server)
```

---

## Setup

```bash
git clone <this-repo>
cd hybrid-nano-vllm
pip install -r requirements.txt          # or: pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

For the real Qwen3.5-35B-A3B checkpoint (~67GB, bf16) — no install script ships
in this repo yet, so fetch it manually, e.g. via `huggingface_hub`:

```bash
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen3.5-35B-A3B', local_dir='./qwen35_checkpoint')"
```

Point `--model` (server) or `LLM(...)` (programmatic) at that local directory.

---

## Running the engine

**Start the server (batched mode — recommended; FCFS is the legacy default):**

```bash
python -m src.server \
    --model /path/to/Qwen3.5-35B-A3B \
    --tensor-parallel-size 2 \
    --concurrency-mode batched \
    --max-num-seqs 8 \
    --enforce-eager \
    --gpu-memory-utilization 0.85
```

**Query it (OpenAI-compatible):**

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen3.5-35b-a3b",
      "messages": [{"role": "user", "content": "What is the capital of France?"}],
      "temperature": 0,
      "max_tokens": 64
    }'
```

**Programmatic (engine-internal, no HTTP layer):**

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen3.5-35B-A3B",
    tensor_parallel_size=2,
    enforce_eager=True,
    gpu_memory_utilization=0.85,
)
sampling_params = SamplingParams(temperature=0, max_tokens=256)
outputs = llm.generate(["Explain the GDR scan in one sentence."], sampling_params)
print(outputs[0]["text"])
```

---

## Running the test suites

```bash
# Component + full-model correctness vs. reference (CPU-friendly)
python tests/test_qwen35_standalone.py

# Batching / StateManager acceptance (requires CUDA + flash-attn)
python tests/test_qwen35_batching.py

# ModelRunner construction, KV-cache byte accounting
python tests/make_fake_hf_config.py
python tests/test_qwen34_model_runner.py

# Eager vs. graph-replay decode parity
python tests/cuda_graph_consistency_test.py

# Real-checkpoint decode-time slot-reuse safety
python tests/decode_stagger_contamination_check.py            # fake model, seconds
python tests/real_checkpoint_slot_reuse_check.py               # forces reuse (--max-num-seqs 3)
python tests/real_checkpoint_slot_reuse_check.py --max-num-seqs 8   # no-reuse control

# TP=2 shard-selection math (CPU-only, no GPU required)
python tests/test_tp_shard_loader.py
```

## Running the benchmarks

```bash
# Internal-engine throughput sweep, small model (mechanism validation only —
# absolute tok/s is not meaningful at this scale)
python bench_throughput.py --model tests/fake_qwen35_small \
    --prompt-len 32 --output-len 64 --max-model-len 512 --gpu-memory-utilization 0.2

# HTTP-server throughput sweep, real checkpoint, real 1024-out target
# (--tokenizer-dir is required -- used for the retok spot-check, see script header)
python bench_http_concurrency.py \
    --base-url http://localhost:8000 \
    --tokenizer-dir /path/to/Qwen3.5-35B-A3B \
    --levels 1 2 4 8 16 32 64 \
    --prompt-tokens 1024 --max-tokens 1024 \
    --ignore-eos \
    --trials 2 --warmup-trials 1

# GSM8K-CoT correctness gate (all 1319 examples, batches of 8, temperature=0,
# checkpointed -- safe to re-run after a crash, resumes from RESULTS_PATH)
python tests/gsm8k_full_run.py --stop-strings
```

---

## Ground rules for contributors

- Do not modify `models/qwen3.py` or break the existing dense-Qwen3 path —
  the hybrid model is a fully separate model file.
- Do not modify `src/model.py`. It is the numerical ground truth for every
  formula. If the port disagrees with it, `src/model.py` wins.
- Prefer small, reviewable diffs per phase over one large rewrite; don't
  start a new phase until the previous phase's acceptance criteria pass.
- Report actual validation numbers (cosine similarity, top-1/top-5 match,
  memory before/after) after each change, not just pass/fail.