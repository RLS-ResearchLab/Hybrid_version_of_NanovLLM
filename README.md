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
| Full hybrid model vs. reference, single sequence | cosine 0.999970, top-1 exact | small model |
| GDR linear attention vs. reference | cosine 1.000001 | small model |
| MoE FFN vs. reference | cosine 1.000000 | small model |
| Eager vs. CUDA-graph decode, batch sizes 1/4/8 | cosine 1.000000, top-1 exact | small model |
| KV-cache memory shrinkage from full-attention-only sizing | 4.00× | measured, matches `full_attention_interval=4` exactly |
| Real-checkpoint single-forward-pass vs. HF, 5 varied prompts | cosine 0.9982–0.9997, 5/5 argmax match | **real 35B checkpoint** |
| Real-checkpoint prefill contamination, concurrency=8 | 8/8 PASS, cosine 0.9976–0.9997 | **real 35B checkpoint** |
| Real-checkpoint decode-time slot-reuse contamination | **not detected** — 3 independent lines of evidence (static code-path analysis, fake-model teacher-forced control, real-checkpoint no-reuse control) | **real 35B checkpoint** |
| MoE top-k=8 combine step, bf16, before/after fp32 promotion | 1.3% relative error → 0.000% | measured |
| **Real-checkpoint throughput, tp=2, 1024-in/1024-out** | baseline 9.3 / 14.7 / 21.2 / 27.2 / 31.5 / 33.8 tok/s at concurrency 1/2/4/8/16/32 | **real 35B checkpoint**, 2×A6000 |
| **Same, with CUDA graphs enabled** | **35.3 / 41.7 / 47.4 / 51.6 / 54.0** tok/s at concurrency 1/2/4/8/16 — **1.7–3.8× speedup**; plateau rises 33.8 → 54.0 | **real 35B checkpoint**, 2×A6000 |
| Fused GDR kernel, end-to-end | **+1.5%** (prefill-only intervention on a decode-dominated workload) | **real 35B checkpoint** |
| **GSM8K-CoT, full 1319 examples, through the engine** | **1257/1319 = 95.30%** — **PASS** vs. the 87.5% gate | **real 35B checkpoint**, tp=2, CUDA graphs |
| GSM8K three-arm prompt ablation (n=32) | raw 81.2% / chat-think 62.5% / **chat-no-think 100%** — isolates thinking-suppression from chat-framing | **real 35B checkpoint** |
| GSM8K-CoT, `raw` prompt format, full 1319 | **58.45%** (771/1319) — FAIL. 46.5% fallback rate; 94.0% accuracy where a marker *was* found | **real 35B checkpoint**, matched baseline |
| GSM8K subset (n=32), marker-found vs. fallback accuracy | 95.7% when a clean final answer is stated, 11.1% when it falls back to "last number in text" | real 35B checkpoint |

**Correctness re-verification — complete.** An earlier validation pass found that
`Experts.gate_up_proj`/`down_proj` were constructed via raw `nn.Parameter(torch.empty(...))`
and never initialized, so two of the rows above certified agreement on a configuration where
the routed-expert path contributed nothing. Both have been re-measured with `Experts` properly
initialized: the full-model cosine moved 0.999967 → 0.999970 and the MoE FFN comparison
1.000001 → 1.000000 (now also independent of test execution order). **The numbers barely moved,
which is itself the finding** — the port and reference agreed on the real per-expert path all
along; the original results were never wrong, only certifying less than they appeared to. The
shared fixture `tests/fake_qwen35_small/model.safetensors` has been regenerated and all
MoE-routing dependents re-confirmed.

**Throughput — the Stage 2 result.** The two optimizations produce opposite outcomes, and
the same fact explains both. The workload is 1024-in/1024-out: one prefill pass against
1024 sequential decode steps.

| Intervention | End-to-end | Path touched |
|---|---|---|
| Fused GDR kernel | **+1.5%** | Prefill only — 1 of 1025 forward passes |
| **CUDA graphs** | **1.7–3.8×** | Decode — 1024 of 1025 forward passes |

The fused kernel measured 36–63× on the isolated GDR layer and 21.5× end-to-end on a
prefill-heavy small-model benchmark. On the real workload it gives 1.5%. CUDA graphs, which
collapse per-step kernel launch overhead in decode, give 1.7–3.8× on the same workload —
and raise the saturated plateau from ~33.8 to ~54.0 tok/s, which is the figure that matters
for a loaded server rather than the 3.79× single-stream number.
**Component and small-model speedups do not transfer to end-to-end throughput unless they
touch the dominant path.**

Two caveats on these figures. They are 2×A6000 (Ampere, PHB, no NVLink) and are functional
validation, not H200-predictive. And CUDA graphs are measurable only to concurrency 4 here —
above that, graph pools (1.12 → 4.08 GB as `max_num_seqs` goes 8 → 32) plus the decode-path
expert gather exceed the ~12 GB left after weights. Concurrency 16 works; 32 does not.
Unblocked on H200's 141 GB.

*Correction: the gather was previously cited here as "512 MiB at N=32, TK=8," undifferentiated
between the two expert tensors. At the real dims (`hidden_size=2048`,
`moe_intermediate_size=512`), `gate_up_proj[local_slots]` (shape `(N,TK,2·MI,H)`) is
**1024 MiB**; 512 MiB is actually `down_proj[local_slots]`'s size (shape `(N,TK,H,MI)`, half
as wide). Both are gathered every decode step, so the correct combined figure is **1536 MiB**,
3× the original number.*

**Vectorized MoE is unreachable at tp=2.** `Qwen35MoE.forward()` tests `ep_size > 1` before
`use_vectorized_moe`, so `_forward_dispatch_vectorized` runs only at `ep_size=1` — a
configuration this model cannot use. The 2.0–4.4× MoE speedup describes a code path that has
never executed in production.

**Tensor parallelism at tp=4 — resolved via GQA kv-head replication, CPU-validated, real-hardware confirmation pending.**
`num_key_value_heads = 2` is smaller than `tp_size = 4`, so the old shard-only scheme (`num_kv_heads % tp_size == 0`)
had no valid mapping and failed during construction before any weights loaded. The fix replicates each of the 2
physical kv heads onto 2 ranks instead of splitting one (`layers/linear.py`'s `local_num_kv_heads`/
`kv_head_replica_source`, shared by `Qwen35FullAttention` construction and `allocate_kv_cache`'s sizing so the
two can't independently drift). Shard-selection math, weight-loader dispatch, and real multi-process
(`mp.spawn`+gloo) construction are CPU-validated end to end. **Not yet run:** construction against real
`.safetensors` weights, real NCCL collectives at tp=4, or output agreement against an HF reference — needs 4
GPUs, unavailable in the current 2-GPU window. tp=2 regression-checked on real hardware in the meantime (A2,
real checkpoint): 5/5 prefill (cosine 0.997320–0.998977) and 5/5 decode first-token match, sitting inside the
pre-existing bf16 baseline range (0.997236–0.999786) — the GQA-replication code changes did not regress the
already-shipping tp=2 path.

**MoE weight-only INT8 quantization (W8A8) — correctness-validated, capacity issue found and resolved.**
Grouped symmetric INT8 quantization of `Experts.gate_up_proj`/`down_proj` (group_size=128, exact division of
both `hidden_size=2048` and `moe_intermediate_size=512`), applied in place after `load_model()` with the
original bf16 Parameters explicitly deleted (`moe_int8_integration.py`). Validated in stages: quantize/dequantize
reconstruction and downstream-matmul error on random weights at real dims; quantize-then-shard vs.
shard-then-quantize proven bitwise identical under EP, which is what justifies quantizing after the existing
EP-sharded load with zero changes to `load_model()`; and a 40-example GSM8K non-regression against the same
subsample A4 used, matching baseline. Decode-path integration covers `_forward_gathered_ep` only (EP decode);
prefill and the non-EP decode path are unmodified and remain bf16-only by design.

At concurrency=16 with CUDA graphs enabled, construction OOMs at the default `gpu_memory_utilization` on this
48GB A6000 — root-caused to `allocate_kv_cache()` sizing the KV cache before `capture_cudagraph()` claims its
CUDA-graph private pool (2.95 GiB at concurrency=16; grows with the largest captured graph bucket), so nothing
reserves margin for it. INT8 frees more weight memory than bf16, so it over-allocates the KV cache by
comparison — this is why it regressed to a *lower* concurrency ceiling than bf16 despite having more headroom.
Not a reference leak — ruled out directly (eager-mode construction and a full decode trial succeed cleanly at
the same weights/concurrency, well under one GPU's capacity).

**Mitigation:** lowering `gpu_memory_utilization` to reserve margin for the private pool fixes it — confirmed
at `0.60` on this hardware (down from bf16's 0.82–0.90; the margin needed scales with `max_num_seqs`, so this
number is A6000-specific tuning, not a portable constant). A5 sweep at that setting, tp=2, real checkpoint:

| Concurrency | tok/s (INT8+graphs, gmu=0.60) |
|---|---|
| 16 | 20.5 |
| 32 | 20.9 |

Single trial each, functional validation not a polished benchmark (matches this project's own scoping
convention). The qualitative result is what matters: **concurrency=32 now completes cleanly under INT8+graphs
on hardware where bf16+graphs could not** (bf16's own ceiling was 32→OOM, above). tok/s is nearly flat
16→32, unlike bf16's clearly-scaling curve — consistent with the dequantization compute cost eating into
what CUDA graphs would otherwise buy from batching; not a red flag on its own. Expected to be a non-issue on
H200's 141GB either way.

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
eval/            GSM8K-CoT correctness gate + throughput harness
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

For the real Qwen3.5-35B-A3B checkpoint (~67GB, bf16):

```bash
bash setup.sh                            # venv, deps, weights
```

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
python bench_http_concurrency.py \
    --url http://localhost:8000/v1/chat/completions \
    --concurrency 1 2 4 8 16 32 64 \
    --input-tokens 1024 --output-tokens 1024 \
    --ignore-eos \
    --requests-per-level 64

# GSM8K-CoT correctness gate (8-shot CoT, temperature=0, top_p=1.0, concurrency=8)
python -m eval.correctness.run_correctness \
    --n 1319 --stop-strings --max-tokens 706
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