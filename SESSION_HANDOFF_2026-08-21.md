# Session Handoff — 2026-08-21 (pre-H200 close-out + INT8 throughput optimization)

**Read this first in any new session continuing this work.** Written to survive the
conversation it was created in — don't trust prior-chat memory of this session, verify
against the actual repo state (this project has been burned by that twice already; see
`moe_quantization_memo.md`'s own history for why that habit exists).

**Hardware this session ran on:** 2×A6000 (Ampere, 48GB each, tp=2), rented GPU box
("shadecloud"). **Not H200.** Every number below is on this hardware unless stated
otherwise. A 6000-series card has ~768GB/s memory bandwidth; H200 has ~4.8TB/s (~6x) —
keep that gap in mind before extrapolating any number here to production H200 targets.

**Repo:** `RLS-ResearchLab/Hybrid_version_of_NanovLLM` (note: dir name has lowercase
`Nanov`, not `NanoV` — a real, repeated source of typo'd paths this session, check
carefully). Remote is SSH by default (`git@github.com:...`); it has intermittently
refused connections mid-session (port 22 blocked) — if `git pull` fails with
`ssh: connect to host github.com port 22: Connection refused`, switch to HTTPS:
`git remote set-url origin https://github.com/RLS-ResearchLab/Hybrid_version_of_NanovLLM.git`.

**Sync note, worth understanding before panicking about git divergence:** local edits made
in this Windows-based session get mirrored to the GPU box through some process the user
runs (not fully understood/documented, but confirmed real and not a third party — asked
directly, confirmed "it's just me running your patches"). This produces frequent small
commits with unhelpful messages (`"debugging"`, `"..."`, `"verifying some results"`) and
occasional divergent-branch states. **Before assuming a git conflict means lost/duplicate
work, diff the actual content** — every divergence hit this session turned out to be
byte-identical mirrors of the same edits, not real conflicts. `git commit` (not `--amend`)
resolves a clean "all conflicts fixed, needs commit" state safely.

---

## Status at a glance

| Item | Status |
|---|---|
| Repo/doc audit (Part A/B from session start) | ✅ Closed |
| A5-under-INT8 OOM at concurrency=16 | ✅ Root-caused, mitigated, confirmed on hardware |
| 512 vs 1024 MiB gather-buffer figure | ✅ Corrected everywhere |
| tp=2 regression (GQA replication didn't break it) | ✅ Confirmed, matches baseline |
| MQA kernel check (4:1 head ratio) | ✅ Passed |
| Critical import bug (prod server couldn't start) | ✅ Found, fixed, verified |
| Throughput fix 1 (drop fp32 intermediate in dequant) | ✅ Verified, 21.3→32.3 tok/s |
| Throughput fix 2 (fuse cast+multiply) | ✅ Verified, 32.3→37.1 tok/s |
| Throughput fix 3 (`torch.compile` on dequant) | ❌ Tried, measured **slower** (33.0 tok/s), reverted |
| Custom fused Triton MoE kernel (idea #3) | 🟡 Built, correctness-verified in isolation AND in real eager-mode GSM8K. **Graph-mode output correctness is the one open question — see "Next step" below.** |
| tp=4 real-hardware validation | ⏳ Structurally can't happen on 2 GPUs — needs a future 4-GPU window |
| Profiler hang at tp=2 | ⏳ Still just documented, not root-caused |
| `qllm_measurements.tex` quantization numbers | ⏳ Not written yet — lowest priority |

---

## UPDATE (later same session): the graph-mode check PASSED

The readout check below was run. Output for all 4 prompts read by eye: every arithmetic
answer correct (train speed, 15% calculations, rectangle area with shown reasoning),
grammatically coherent, no garbage/corruption. One prompt ("capital of France") fell into
benign repetition (known greedy-decoding artifact on an easy prompt with `temperature=0`
and no stop condition, not a correctness red flag). Combined with the eager-mode 8/8 GSM8K
result and the isolated kernel's cosine=0.999988, **three independent angles now confirm
the fused kernel + graph capture combination is genuinely correct. The 172.7 tok/s number
is trustworthy.** Next: lock this into `README.md`/`moe_quantization_memo.md` (was
deliberately held back until this point), then proceed to backlog item 2 (A5 capacity
sweep with the fused kernel, larger-n GSM8K if time allows).

---

## THE ONE THING IN FLIGHT — do this first

A real-engine run under **CUDA graph capture + the new fused kernel** measured
**172.7 tok/s** (`tests/diag_w8a8_eager_vs_graph.py`, concurrency=16, tp=2,
`gpu_memory_utilization=0.75`, `NANOVLLM_USE_FUSED_MOE_KERNEL=1`, graph mode i.e. no
`--enforce-eager`). That's 3.3x the bf16 baseline (52.7 tok/s) and 4.65x the best prior
INT8 number (37.1 tok/s) — **too good to trust without checking the actual output text**,
which was never done for this specific combination.

What's already been checked and passed:
- The kernel's math, in isolation, against synthetic weights: cosine=0.999988 (full
  gate_up+silu+down_proj pipeline vs. the real production dequant-then-matmul reference).
- The kernel wired into the real model, in **eager mode**: n=8 real-checkpoint GSM8K,
  8/8 both arms, zero flips, McNemar p=1.0 — genuinely strong evidence the kernel's math
  is right.

What's NOT checked: `cluster_q6_moe_w8a8_gsm8k.py` hardcodes `enforce_eager=True`, so the
GSM8K check above never touched CUDA graph capture. **The specific combination that
produced 172.7 tok/s — fused kernel + graph capture together — has never had its output
read.** A tok/s number can't distinguish "fast and correct" from "fast because it's
silently producing garbage/short-circuited output" — only reading actual generated text can.

**The script to run this check already exists and is ready:**
```bash
NANOVLLM_USE_FUSED_MOE_KERNEL=1 python tests/diag_fused_kernel_graph_readout.py \
    --checkpoint "$HOME/Hybrid_version_of_NanovLLM/qwen35_checkpoint" \
    2>&1 | tee tests/_cluster_day_cache/logs/diag_fused_kernel_graph_readout.log
```
It builds the engine in graph mode (default `--enforce-eager` is off) with the fused
kernel on, generates from 4 real prompts (3 arithmetic word problems + 1 factual) at
`temperature=0`, and prints `PROMPT`/`OUTPUT` pairs. **Read them.** If they're coherent,
sensible answers, that closes the last real gap and 172.7 tok/s (or whatever this run
measures) becomes a trustworthy number — genuinely excellent news, close to the user's
stated 200 tok/s goal (see reality-check below), on hardware not even expected to get
there. If they're garbage/repeated tokens/nonsense, the graph-capture combination has a
real bug that needs root-causing before any of these numbers are usable.

This was queued and waiting on the user pasting back the output when the session ended.
**Check for that output first** — it may already be sitting in the conversation this
handoff doc was written from, or may need re-running.

---

## The 200 tok/s goal — reality-checked, don't re-litigate this

The user's stated goal is 200 tok/s. Context for a future session: **this is very unlikely
on the 2×A6000 test hardware regardless of software optimization** — bf16 with CUDA
graphs, the fastest config measured all session with zero quantization overhead, tops out
at 52.7-54.0 tok/s here. That's this card's memory bandwidth, not a software ceiling.
200 tok/s is a realistic **H200** target (rough bandwidth-scaling argument: ~6x more
bandwidth → plausibly 300+ tok/s ceiling for this model on H200), not something to expect
to prove or disprove on this test hardware. If the graph-mode fused-kernel number checks
out as correct, frame it to the user/mentor as "validates the approach and gets close on
test hardware most consider inadequate for that number" — not as "hit the goal already,"
unless a future measurement genuinely clears 200 on this same 2×A6000 setup.

---

## Full throughput arc, all numbers, matched settings unless noted

Matched-settings A/B methodology throughout: `tests/diag_w8a8_eager_vs_graph.py`,
concurrency=16, `gpu_memory_utilization=0.75`, tp=2, real checkpoint, prompt-len=128,
output-len=1024 (unless noted), identical everything except the flag being tested.

| Config | tok/s | Notes |
|---|---|---|
| bf16 + CUDA graphs | 52.7 | The baseline everything else is measured against |
| INT8, original dequant (fp32 intermediate) | 21.3 | 2.47x slower than bf16 — the original, unoptimized state |
| INT8, fix 1 (bf16-direct dequant, no fp32) | 32.3 | 1.63x slower than bf16 |
| INT8, fix 2 (fused cast+multiply) | 37.1 | 1.42x slower than bf16 — **the "safe, fully-verified" production number** |
| INT8, fix 3 (`torch.compile` added) | — | Not re-measured standalone; the eager-mode fused-kernel numbers below supersede interest in this |
| INT8, fused kernel, eager mode | 21.9-29.3 | **Slower than fix 2** — diagnosed as `moe_align_block_size` dispatch overhead (~57% of time), not kernel compute (~39%). Expected to improve under graph capture, which eliminates per-op Python dispatch cost. output-len varied (1024 and 64) between the two eager runs, not directly comparable to each other. |
| INT8, fused kernel, **graph mode** | **172.7** | **UNVERIFIED FOR CORRECTNESS — see "one thing in flight" above.** If confirmed, this is the real result. |

**Capacity result (separate from throughput, already fully closed):** INT8 unlocks
concurrency=32 with CUDA graphs, which bf16 cannot do on this hardware at all (OOMs).
Confirmed via the A5 sweep at `gpu_memory_utilization=0.60`: 20.5 tok/s @ concurrency=16,
20.9 tok/s @ concurrency=32, both PASS, no OOM. (This predates the fused-kernel work —
worth re-running once the kernel path is trusted, since it changes the memory profile too.)

---

## The custom kernel work (idea #3) — what's actually built

**Origin:** `layers/fused_moe_triton_raw.py` (676 lines) turned out to already exist in
the repo — a verbatim copy of vLLM's real production fused-MoE Triton kernel, committed
2026-07-27 (three weeks before this session's quantization work), never wired up anywhere
(dead code, same situation `layers/moe_quant.py` was in before it got deleted this
session). Has `use_int8_w8a16` support built in — exactly this project's weight-only INT8
scheme — which is why this became worth pursuing rather than writing a kernel from scratch.

**What got built this session (all in `layers/`):**

- **`fused_moe_triton.py`** — adapted copy of the raw kernel, stripped of vLLM-package
  dependencies (`vllm.envs`, `vllm._custom_ops`, etc. — those are only needed for
  FP8/topk_softmax/silu_and_mul paths this project doesn't use). **Modified from vLLM's
  original**: the original applies INT8 scale ONCE, after the full K-reduction loop —
  correct only for one scale per whole output-channel row. This project's actual
  quantization is GROUPED (`group_size=128`, finer-grained, and what all the
  already-validated accuracy numbers were measured against). Rewrote the kernel to apply
  scale ONCE PER K-ITERATION instead, exact as long as `BLOCK_SIZE_K` divides
  `QUANT_GROUP_SIZE` evenly (asserted, not assumed, in `invoke_fused_moe_kernel`).

- **`moe_align_block_size.py`** — pure-PyTorch reimplementation of vLLM's
  `ops.moe_align_block_size` (a compiled CUDA extension in real vLLM, which this project
  doesn't depend on). CPU-tested exhaustively (`test_moe_align_block_size.py`, 15 cases
  including edge cases like single-expert-gets-everything and last-expert-gets-nothing) —
  all pass, checking real invariants (every token appears exactly once, block↔expert
  assignment is self-consistent), not just "runs without crashing." **This is also the
  function later found to be the eager-mode bottleneck** — see below.

- **`fused_moe_int8.py`** — the actual integration entry point,
  `fused_moe_int8_forward(x, gate_up_proj_int8, gate_up_proj_scale, down_proj_int8,
  down_proj_scale, group_size, local_slots, top_k, local_num_experts, config=None)`.
  Handles the full gate_up→silu→down_proj pipeline. Key design insight: the kernel needs
  NO pre-gathered weight buffer — it reads directly from the original small activation
  tensor and the FULL local-expert weight tensor, gathering implicitly via indices. This
  is WHY it's fast: it avoids ever materializing the `(N, TK, ...)` buffer that's been the
  root of both the original OOM and the throughput cost all session. down_proj's second
  kernel call uses a "virtual token, top_k=1" trick (each (token,k) pair reshaped to its
  own row) since its input is already per-expert after gate_up.
  Has built-in diagnostic timing (`NANOVLLM_PROFILE_FUSED_MOE=1` env var) that's how the
  eager-mode bottleneck got found.

- **Smoke tests** (`smoke_test_fused_moe_triton.py`, `smoke_test_full_moe_pipeline.py`) —
  isolated correctness+speed validation before touching production code. Results: single
  GEMM 7.51x vs. production reference; full pipeline 4.71x, cosine=0.999988.

- **Integration into `models/qwen3_5.py`** — `_forward_gathered_ep` (the decode EP path)
  now branches on `_USE_FUSED_MOE_KERNEL` (env var `NANOVLLM_USE_FUSED_MOE_KERNEL=1`,
  checked once at import time). Deliberately flag-gated, not a hard replacement — the
  original gather+dequant+einsum path is the safe fallback. **`_forward_dispatch_ep`
  (prefill) was NOT touched** — lower priority (1 of ~1025 forward passes), still uses the
  original path unconditionally regardless of the flag.

**The eager-mode surprise, and why graph mode is the real test:** first real-engine test
(eager, concurrency=16) measured 29.3 tok/s — *slower* than the 37.1 baseline, despite the
kernel itself measuring 4.71x faster in isolation. Profiling
(`NANOVLLM_PROFILE_FUSED_MOE=1`) showed why: `moe_align_block_size`'s ~8-10 sequential
small PyTorch ops (`scatter_add_`, two `cumsum`s, `argsort`, gathers, `searchsorted`) eat
~57% of the function's time in eager mode — each pays real Python/CUDA dispatch overhead,
and there are 2x more such ops per layer than the original approach had. The actual GEMM
kernels are fast and consistent with the isolated benchmark (~39% of time, cheap). CUDA
graph capture is specifically designed to eliminate exactly this kind of per-op dispatch
cost (record once, replay as one unit) — which is why the graph-mode number jumped to
172.7 tok/s. This is a plausible, not yet fully certain, explanation.

**If graph-mode output turns out to be broken** (the check still pending): the fix would
be writing `moe_align_block_size` as a Triton kernel too, instead of ~10 separate PyTorch
ops — the algorithm is already correct and tested, so this would be porting known-good
logic, not designing from scratch. Real but bounded additional work.

**Autotuning — not done, real potential upside:** every kernel call this session used one
fixed, conservative config (`BLOCK_SIZE_M=16, N=64, K=64, warps=4, stages=2`), never
tuned. `layers/fused_moe_triton_raw.py`'s own docstring area references an autotuner
script pattern (from a different/prior project, imports `model_update.fused_moe_triton`
which doesn't exist in this repo — not directly runnable here, but a good template) for
searching block-size configs per shape/concurrency. Worth doing once the current
integration is trusted — typically buys a real additional chunk beyond a first-working config.

---

## Backlog, priority order for a future session

1. **Read the graph-mode readout output** (above) — blocks trusting the 172.7 number and
   anything built on top of it.
2. **If correct:** re-run the A5 capacity sweep (concurrency 16/32/64) with the fused
   kernel + graph mode, get a real throughput-vs-concurrency curve. Re-check GSM8K at
   larger n if time allows (n=40, matching the original Q6 scope). Update
   `moe_quantization_memo.md`/`README.md` with the final numbers — both currently still
   describe the fix-1+fix-2-only state (37.1 tok/s, 1.42x gap) as current; the fused
   kernel result isn't written into either doc yet, deliberately, until verified.
3. **If broken:** port `moe_align_block_size` to Triton (bounded, well-scoped — the
   pure-Python algorithm in `layers/moe_align_block_size.py` is already correct and
   tested, this is a port, not a redesign).
4. **Autotune the kernel config** — real upside, low risk, once (2) or (3) lands.
5. **tp=4 real-hardware validation** — needs a 4-GPU window, can't happen on this box.
   MQA kernel check already passed (de-risks it), full construction/NCCL/HF-reference
   check is still fully open.
6. **Profiler hang at tp=2** — still just documented (`moe_quantization_memo.md` §4), not
   root-caused. Lower priority.
7. **`qllm_measurements.tex`** — never got the quantization numbers written in. Pure
   documentation completeness, lowest priority.
8. **Git history cleanup** — many small mirror-sync commits from this session
   (`"debugging"`, `"..."`, etc.). Not urgent; the user's call whether it's worth squashing
   shared history or leaving it as an honest record.

---

## Gotchas learned this session, don't re-discover these

- **The `nanovllm` package doesn't really exist** — it's `Hybrid_version_of_NanovLLM/`
  (repo root) with an `__init__.py` at the root that uses PEP 562 `__getattr__` to expose
  `LLM`/`SamplingParams` lazily. Every entry point fakes `sys.modules["nanovllm"]`
  pointing at repo root. **Two distinct footguns here, both hit this session:**
  1. Getting `ROOT` wrong (single- vs. double-`dirname` depending on whether the script
     lives at repo root or one level down in `tests`/`layers`) breaks checkpoint-path
     defaults and `bench_throughput`-style imports. Always double-check which one applies.
  2. Manually constructing `sys.modules["nanovllm"] = types.ModuleType(...)` creates a
     **bare, empty** module — it does NOT execute the real `__init__.py`, so
     `from nanovllm import LLM` fails (`__getattr__` was never attached). Use
     `from nanovllm.llm import LLM` / `from nanovllm.sampling_params import SamplingParams`
     (submodule imports) instead — sidesteps the whole issue, matches how every working
     script in this project actually does it.
- **`tests/cluster_q6_moe_w8a8_gsm8k.py` hardcodes `enforce_eager=True`** — it can validate
  kernel correctness but can never be used to check anything CUDA-graph-specific.
- **`--no-fake-config-loader` flag polarity is inconsistent between scripts** —
  `cluster_a5_concurrency_sweep.py` defaults `fake_config_loader=True` (needs the flag
  explicitly for real-checkpoint runs); `cluster_a2_tp_correctness.py`/`cluster_q6...`
  default it `False` (opposite). Check per-script before assuming.
- **This Windows dev machine has no `triton`, no C++ compiler (`cl.exe`), no
  `flash_attn`** — any local verification here bottoms out at whichever GPU-only import
  comes first. That's expected and doesn't indicate a real bug; confirm by checking *where*
  the import fails, not just that it fails.
- **`torch.compile` genuinely made things slower here** (33.0 vs. 37.1 tok/s) — don't
  re-try it on this code path without a specific new reason; it was tried once, measured,
  and reverted, not skipped out of caution.
- **A tok/s number alone never proves correctness** — this session's biggest number
  (172.7 tok/s) is exactly the case for why: always read actual output text for any new
  code path before trusting its speed number, especially anything touching CUDA graph
  capture for the first time.

---

## File manifest — what's new/changed this session (non-exhaustive, the highlights)

- `layers/fused_moe_triton_raw.py` — pre-existing dead code (discovered, not created)
- `layers/fused_moe_triton.py` — adapted kernel (vLLM deps stripped, grouped-scale rewrite)
- `layers/moe_align_block_size.py` + `layers/test_moe_align_block_size.py` — alignment replacement + CPU tests
- `layers/fused_moe_int8.py` — integration entry point, with built-in profiling
- `layers/smoke_test_fused_moe_triton.py`, `layers/smoke_test_full_moe_pipeline.py` — isolated validation
- `models/qwen3_5.py` — `_forward_gathered_ep` flag-gated fused-kernel branch; also this
  session's earlier critical-import-bug fix (unrelated to the kernel work, see below)
- `tests/diag_fused_kernel_graph_readout.py` — the pending correctness check
- `tests/moe_int8_quantize.py` — `dequantize_weight_int8_grouped`, two verified throughput
  fixes applied (fp32 removal, fused multiply); `torch.compile` tried and reverted, noted
  in the docstring
- `engine/model_runner.py`, `models/qwen3_5.py` — critical import-path fix (production
  server couldn't start at all before this; unrelated to kernel work, found via a
  multi-agent code review earlier this session)
- `src/server.py` — gained `--moe-w8a8`/`--moe-w8a8-group-size` flags (previously only
  reachable via internal test scripts)
- `layers/moe_quant.py` — **deleted** (confirmed dead code, duplicate of the real
  quantization path, had a correctness-principle-violating silent fallback)
- `config.py` — dropped unused `moe_w8a8_act_eps` field
- `README.md`, `moe_quantization_memo.md` — extensively updated throughout the session;
  currently reflect the fix-1+fix-2 state (37.1 tok/s) as the trusted number, NOT the
  fused-kernel result (deliberately held back pending the correctness check above)
