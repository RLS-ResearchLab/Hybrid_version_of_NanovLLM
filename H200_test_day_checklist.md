# QLLM — H200 Test Day Checklist

**Starting point (verified on 2×A6000, 2026-08-21):** INT8 weight-only MoE quantization + a custom fused
Triton kernel + a CUDA-graph state-manager write-back fix. **204.1 tok/s at concurrency=32, tp=2** — 3.87×
the bf16 baseline (52.7 tok/s), correctness-verified multiple independent ways (kernel-math cosine, GSM8K
non-regression in both eager and graph mode with zero discordant pairs, real generated text under graph
capture). Full detail in `README.md`'s MoE section and `moe_quantization_memo.md`.

**Hardware change, not just "more of the same":** 4×H200 (141GB each) vs. 2×A6000 (48GB each, ~768GB/s).
H200 has ~6× the memory bandwidth, ~3× the memory per GPU, and — unlike Ampere — **native FP8 tensor cores**.
Several conclusions from tonight were explicitly hardware-specific and do not carry over; this checklist is
about which ones.

---

## 🔴 Decide before the window starts — topology

**This model fits on a single H200.** ~34B total params (256 experts × ~3.15M params/expert × 40 layers,
~94.7% of params are routed experts, per `config.json`) = ~68GB at bf16 or ~34GB at INT8 — comfortably under
141GB, with 70-100GB left for KV cache. That's fundamentally different from the 2×A6000 rig, where tp=2 was
*mandatory* just to fit the model at all.

- [ ] **Decide: aggregate throughput (many concurrent users) vs. per-request latency (one user, fastest
      response)?** This determines the right topology, and building the wrong one wastes the window.
  - **4× independent tp=1 replicas** — no NCCL/communication overhead at all, near-linear aggregate
    throughput scaling, and tp=1 is the simplest, least risky case in the codebase (no NCCL correctness
    surface to validate). Best if the goal is aggregate tok/s — which is what's been measured and discussed
    all session.
  - **tp=4, one replica** — lowers per-request latency, but is the still-fully-unvalidated path (see below)
    and adds real NCCL overhead every step tonight's own profiling showed can be substantial.
  - **2× tp=2 replicas** — middle ground, reuses the already-hardware-validated tp=2 path exactly, doubles
    capacity without touching the tp=4 unknowns at all. Worth considering as the safe default if the topology
    decision can't be made in advance.
- [ ] **Raise this with the mentor explicitly** — don't default into tp=4 just because 4 GPUs are available.

---

## 🔴 P0 — Must validate on real H200 hardware, nothing here transfers from A6000

- [ ] **tp=4 real-hardware validation** (only if the topology decision above picks tp=4 or a mixed setup).
      Per the pre-existing `Gpu window checklist.md`: GQA kv-head replication is CPU-validated end to end
      (shard math, weight-loader dispatch, multi-process construction), but construction against real
      `.safetensors` weights, real NCCL collectives at tp=4, and output agreement against an HF reference
      have **never run** — needs 4 GPUs, which this window is the first chance at. Treat as the single
      biggest unknown if this path is chosen.
- [ ] **Re-derive `gpu_memory_utilization` from scratch.** Every number this session (0.60, 0.52, 0.45,
      the whole "KV cache vs. graph-capture headroom" story) is a consequence of A6000's 48GB and this
      model's specific memory footprint on it. H200's 141GB changes the arithmetic completely — do not
      reuse these values as starting points beyond "start high, back off if OOM."
- [ ] **Re-run the graph-mode nsys kernel-time profile fresh.** Tonight's profile ruled out autotuning
      the fused kernel's config (GEMM was ~0.1% of GPU time on Ampere) — but that conclusion is
      Ampere-specific. H200's different compute/bandwidth ratio and native FP8 tensor cores could shift
      which kernel dominates entirely. "Ruled out on Ampere" is not "ruled out on Hopper" — don't assume,
      remeasure.
- [ ] **Re-run the capacity sweep (16/32/64/higher) fresh.** The concurrency=64 ceiling found tonight was
      explicitly a 48GB-card finding (`StateManager` + weights already ~20GB/rank before any KV cache).
      With 141GB/GPU, 64 and well beyond is plausibly fine — find the real ceiling rather than assuming
      the old one.
- [ ] **Try large-batch/large-concurrency throughput specifically.** This model's shape (~34B total / ~1.1GB
      active FFN params per token) means the MoE weight-read cost amortizes across a batch — at large enough
      concurrency, expert coverage saturates and per-token bandwidth cost keeps dropping. This regime was
      never reachable on A6000 (never got past concurrency=32-64 before hitting memory limits) — H200's
      memory headroom is the first real chance to test whether this is where a much bigger throughput number
      actually lives.

---

## 🟡 P1 — Worth having ready, not yet built

- [ ] **Recover and adapt the FP8/`wgmma`/TMA fused MoE kernel discussed this session.** A full
      warp-specialized (producer/consumer), TMA-pipelined, dynamically-FP8-quantized MoE kernel was pasted
      into this session's chat and reviewed — it targets `wgmma.mma_async`, TMA (`cp.async.bulk.tensor`),
      and native FP8 tensor cores, **none of which exist on Ampere**, so it was correctly set aside for
      tonight. On H200, all three of those are real hardware features — this is no longer throwaway code,
      it's a legitimate starting point for true W8A8 (activation quantization too, not just weights) on the
      actual target hardware. **This code was never saved to the repo** — retrieve it from this session's
      history before it's lost, and treat porting/finishing it as real, scoped work, not a rewrite from
      scratch. Needs the same rigor as everything else here: isolated kernel-math verification before
      real-engine integration, GSM8K non-regression, graph-mode correctness check — do not skip steps
      under time pressure just because the code already exists.
- [ ] **True W8A8 (activation quantization, not just weights)** — sized earlier this session as real,
      multi-step work (dynamic per-token activation quantization, outlier handling, a genuinely different
      kernel path), not a same-day patch. H200's native FP8 support is what makes this worth prioritizing
      over Ampere-era INT8-only; scope it properly rather than rushing it.
- [ ] **Migrate prefill (`_forward_dispatch_ep`) to the fused kernel.** Deferred all session as low-priority
      (~1 of 1025 forward passes on A6000's matched-settings runs) — but if H200 testing uses much longer
      generations or many more short requests, prefill's relative share changes. Worth a quick check of
      whether it's still negligible before assuming it is.

---

## 🟢 P2 — Cheap, do first if there's slack time

- [ ] **Confirm the state-manager CUDA-graph fix still applies unmodified.** `StateManager.scratch_slot_id`
      and the in-graph `index_copy_` write-back don't depend on A6000-specific numbers — should carry over
      as-is, but re-run the same GSM8K graph-mode correctness check (40/40, chat-no-think) once on real H200
      hardware before trusting it there too. Cheap, and this is exactly the kind of change (silent
      state-corruption risk if wrong) worth re-confirming on new hardware rather than assuming.
- [ ] **`qllm_measurements.tex`** — still never got the quantization numbers written in (lowest priority all
      session, still true).
- [ ] **Git history cleanup** — many small mirror-sync commits (`"debugging"`, `"..."`) accumulated over the
      2×A6000 sessions. Not urgent, your call whether to squash before H200 work starts on a cleaner base.

---

## Gotchas from the A6000 sessions, don't re-discover these on H200

- The `nanovllm` package doesn't really exist — see `SESSION_HANDOFF_2026-08-21.md`'s gotchas section
  (now gitignored, kept locally) for the two import-shim footguns this cost real time on.
- `--no-fake-config-loader` flag polarity is inconsistent between test scripts — check each script's default
  before assuming.
- `torch.compile` measured *slower* here (33.0 vs. 37.1 tok/s) — don't re-try without a specific new reason.
- CUDA graph replay hides per-kernel detail from `torch.profiler`'s CUPTI hooks on at least one driver/PyTorch
  combination tested this session — `nsys profile` + `nsys stats --report cuda_gpu_kern_sum` was the working
  fallback; confirm which tool actually works on the H200 box's driver/CUDA version before assuming either.
- A tok/s number alone never proves correctness — always read real generated text or run a gold-checkable
  benchmark (GSM8K) for any new code path before trusting its speed number, especially anything touching CUDA
  graph capture for the first time on new hardware.
- A GPU kernel's share of total profiled time is not the same as its share of wall-clock critical-path time —
  tonight's state-manager fix measured only ~1% real gain despite the profile suggesting ~8.6%, because async
  execution had likely already hidden most of that cost. Measure the actual before/after; don't trust the
  profile percentage as a speedup prediction.

---

## What to raise with the mentor

1. **Topology decision** (above) — needs an answer before the window opens, not during it.
2. **Is 300 tok/s (or higher) a 2×A6000 target or an H200 target?** This session clarified it was about
   A6000 when asked, and 204.1 tok/s was the honest ceiling found there. If the real target was always H200,
   that's a very different, much more promising starting point — worth confirming explicitly rather than
   re-deriving expectations from an A6000 number.
3. **Scope for true W8A8/FP8 on H200** — the drafted kernel (P1 above) makes this newly realistic on this
   hardware in a way it wasn't on Ampere. Worth deciding how much of the limited H200 window to commit to it
   versus consolidating the already-verified INT8 result across the new topology.
