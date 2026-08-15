"""QLLM Stage 2, P2 investigation (per-layer-type CUDA graphs): profile a
decode step and a prefill step to establish whether kernel-launch overhead
is actually a large enough fraction of decode wall-clock for graph
restructuring to plausibly help, BEFORE building anything.

Answers, in order:
  1. Decode-step wall-clock: launch-dispatch (CPU) time vs. GPU kernel
     execution time -- two independent passes, not read off one profiled
     run (the profiler's own overhead would contaminate a timing read):
       pass A (unprofiled): host perf_counter() + torch.cuda.Event, both
         bracketing torch.cuda.synchronize() -- the actual number to trust.
       pass B (profiled): torch.profiler with CPU+CUDA activities, for the
         op-level and launch-count breakdown. Its own wall-clock is NOT
         reported as ground truth.
  2. Kernel-launch count per decode step, split into:
       - GDR_layer / FullAttn_layer / MoE_layer (via a temporary,
         restored-after class-level forward() wrap -- record_function
         labels, zero permanent change to models/qwen3_5.py)
       - state_manager.get_all / state_manager.set_all / compute_logits
         (instance-level bound-method wrap, same non-invasive mechanism)
       - graph.replay() itself (wrapped per-CUDAGraph-object)
       - everything else in ModelRunner.run() (pre-replay buffer staging,
         Python glue) -- reported by subtraction, since that code is not
         individually labeled (deliberately not editing model_runner.py's
         hot path just to instrument it).
  3. Whether runner.run()'s use_graph condition is actually True for the
     batch sizes being tested (guards against a silent eager fallback).
  4. Whether the fused-GDR Triton kernel (use_fused_gdr_kernel=True) is
     capturable inside torch.cuda.graph() at all, independent of whether
     the current decode path invokes it (it doesn't -- is_decode_shape
     forces the sequential path in decode regardless of the flag; this
     checks the underlying capturability question directly, empirically).

Requires a real CUDA GPU. Prints [SKIP] and exits cleanly otherwise, same
convention as tests/bench_fused_gdr.py.

Usage (fake small model -- self-contained, default, exercises the SAME
30:10-ish GDR:full-attention ratio as the real model via
full_attention_interval, just at 8 layers instead of 40 -- absolute launch
counts will scale roughly 5x on the real checkpoint, ratios should not):
    python tests/make_fake_hf_config.py     # if not already run
    python tests/profile_decode_launch_overhead.py

Usage (real checkpoint):
    python tests/profile_decode_launch_overhead.py \\
        --model /path/to/qwen3.5-35b-a3b --no-fake-config-loader \\
        --max-model-len 4096 --gpu-memory-utilization 0.85 \\
        --batch-sizes 1 4 8 32

Also runs the fused-GDR capturability check by default; skip with
--no-check-fused-gdr-capture if flash-linear-attention isn't installed in
this environment (the script already skips it gracefully either way).
"""

import argparse
import functools
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_WS_NAME = os.path.basename(ROOT)
if _WS_NAME != "nanovllm":
    import types
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class AttrDict(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def _fake_from_pretrained(path, *args, **kwargs):
    with open(os.path.join(path, "config.json")) as f:
        d = json.load(f)
    d = AttrDict(d)
    d.dtype = _DTYPE_MAP[d.pop("torch_dtype")]
    return d


LAUNCH_NAME_HINTS = (
    "cudaLaunchKernel", "cudaGraphLaunch", "cudaMemcpyAsync",
    "cudaMemsetAsync", "cuLaunchKernel",
)
REGION_LABELS = (
    "GDR_layer", "FullAttn_layer", "MoE_layer",
    "state_manager.get_all", "state_manager.set_all", "compute_logits",
    "graph.replay",
)


# ─── harness construction ──────────────────────────────────────────────────


def build_runner(model_dir, fake_config_loader, max_num_seqs, max_model_len,
                  gpu_memory_utilization, use_fused_gdr_kernel):
    import nanovllm.utils.loader as loader_mod

    if fake_config_loader:
        import nanovllm.config as config_mod
        config_mod.AutoConfig.from_pretrained = staticmethod(_fake_from_pretrained)
        # Matches tests/cuda_graph_consistency_test.py's Phase 4 harness: no
        # real .safetensors for the fake small config, so load_model is a
        # no-op and weights are set explicitly afterward (below). Do NOT
        # stub this when pointed at a real checkpoint (--model + explicit
        # --no-fake-config-loader) -- that path needs real weight loading.
        loader_mod.load_model = lambda model, path: None

    from nanovllm.config import Config
    from nanovllm.engine.model_runner import ModelRunner

    config = Config(
        model=model_dir,
        max_num_batched_tokens=max(2048, max_model_len),
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=False,   # graphs must be built for this profiling script
        use_fused_gdr_kernel=use_fused_gdr_kernel,
    )
    runner = ModelRunner(config, rank=0, event=[])
    return runner, config


def init_weights_and_guard(runner, fake_config_loader, seed=7):
    """Only re-inits when load_model was stubbed (fake harness) -- a real
    checkpoint already has real weights and must not be clobbered. Either
    way, the guardrail runs: assert_all_parameters_initialized has caught
    four independent uninitialized-parameter incidents in this project (see
    tests/test_utils.py's docstring) and is mandatory in any new harness."""
    from test_utils import (
        init_model_weights_with_norms_, assert_all_parameters_initialized,
        known_zero_initialized_param_names,
    )
    if fake_config_loader:
        with torch.no_grad():
            init_model_weights_with_norms_(runner.model, seed=seed)
    assert_all_parameters_initialized(
        runner.model, whitelist_zero=known_zero_initialized_param_names(runner.model)
    )
    print("  [OK] assert_all_parameters_initialized passed")


# ─── sequence / batch helpers ──────────────────────────────────────────────


def make_prefill_batch(scheduler, Sequence, bs, target_len, seed):
    """Varied sequence lengths per prompt (target_len//2 .. target_len), not
    one fixed length repeated bs times -- a recent audit in this project
    found tests using degenerate uniform inputs that silently prevented them
    from exercising the code paths they claimed to. Real torch.randint token
    ids (matches this project's own convention for synthetic prompts;
    prompts are token-id lists, not embeddings, so randn doesn't apply
    here).

    max_tokens is set far above any plausible profiling loop length and
    ignore_eos=True -- config.eos defaults to -1 here (no tokenizer/LLMEngine
    in this harness) so a real sampled token id would never equal it anyway,
    but max_tokens=64's default WOULD eventually be hit by a long enough
    --timed-iters/--profiled-iters run, silently shrinking scheduler.running
    mid-profile and corrupting the batch-size comparison (or crashing
    schedule()'s `assert scheduled_seqs` once it empties). Pin both
    explicitly so iteration counts can be raised freely from the CLI."""
    from nanovllm.sampling_params import SamplingParams
    sp = SamplingParams(max_tokens=1_000_000, ignore_eos=True)
    g = torch.Generator().manual_seed(seed)
    lo = max(4, target_len // 2)
    prompts = []
    for _ in range(bs):
        length = torch.randint(lo, target_len + 1, (1,), generator=g).item()
        prompts.append(torch.randint(100, 5000, (length,), generator=g).tolist())
    seqs = [Sequence(p, sp) for p in prompts]
    for s in seqs:
        scheduler.add(s)
    return seqs


def run_prefill_to_completion(runner, scheduler, seqs_added, max_rounds=64):
    """Chunked-prefill-safe: keeps calling schedule()/run()/postprocess()
    until every just-added sequence has moved into scheduler.running (chunked
    prefill can split one logical prefill across multiple schedule() calls
    if max_num_batched_tokens is tight)."""
    for _ in range(max_rounds):
        pending = [s for s in seqs_added if s.status.name != "RUNNING"]
        if not pending:
            break
        seqs, is_prefill = scheduler.schedule()
        assert is_prefill, "expected prefill rounds only in this helper"
        token_ids = runner.run(seqs, True)
        scheduler.postprocess(seqs, token_ids, True)
    else:
        raise RuntimeError("prefill did not complete within max_rounds -- "
                            "raise --max-num-batched-tokens or shrink --prefill-len")


def cleanup_running(scheduler, seqs):
    for s in list(seqs):
        if s in scheduler.running:
            scheduler.running.remove(s)
        scheduler.block_manager.deallocate(s)
        if scheduler.state_manager is not None:
            scheduler.state_manager.free(s)


# ─── profiling-region instrumentation (additive, fully restored after) ────


class RegionInstrumentation:
    """Temporarily wraps forward()/bound-methods with torch.profiler
    record_function labels so the trace can attribute time/launches to
    layer type and to the specific pre/post-replay glue calls in
    ModelRunner.run(), without changing a single line of production source.
    Everything is restored in __exit__, including on exception."""

    def __init__(self, runner):
        self.runner = runner
        self._class_patches = []   # (cls, attr, original)
        self._instance_patches = []  # (obj, attr, original)

    def __enter__(self):
        from nanovllm.models.qwen3_5 import (
            Qwen35LinearAttention, Qwen35FullAttention, Qwen35MoE,
        )
        self._wrap_class_forward(Qwen35LinearAttention, "GDR_layer")
        self._wrap_class_forward(Qwen35FullAttention, "FullAttn_layer")
        self._wrap_class_forward(Qwen35MoE, "MoE_layer")

        if self.runner.state_manager is not None:
            self._wrap_instance_method(self.runner.state_manager, "get_all", "state_manager.get_all")
            self._wrap_instance_method(self.runner.state_manager, "set_all", "state_manager.set_all")
        self._wrap_instance_method(self.runner.model, "compute_logits", "compute_logits")

        if hasattr(self.runner, "graphs"):
            for bs, graph in self.runner.graphs.items():
                self._wrap_instance_method(graph, "replay", "graph.replay")
        return self

    def __exit__(self, *exc):
        for obj, attr, original in reversed(self._instance_patches):
            setattr(obj, attr, original)
        for cls, attr, original in reversed(self._class_patches):
            setattr(cls, attr, original)
        return False

    def _wrap_class_forward(self, cls, label):
        original = cls.forward

        @functools.wraps(original)
        def wrapped(self_mod, *a, **kw):
            with torch.profiler.record_function(label):
                return original(self_mod, *a, **kw)

        cls.forward = wrapped
        self._class_patches.append((cls, "forward", original))

    def _wrap_instance_method(self, obj, attr, label):
        original = getattr(obj, attr)

        @functools.wraps(original)
        def wrapped(*a, **kw):
            with torch.profiler.record_function(label):
                return original(*a, **kw)

        setattr(obj, attr, wrapped)
        self._instance_patches.append((obj, attr, original))


# ─── timing (unprofiled, host+device cross-check) ──────────────────────────


def time_iters(fn, iters):
    """Host perf_counter() AND CUDA events, both bracketing
    torch.cuda.synchronize() -- the standard in this project since a prior
    benchmark needed both to rule out an async-completion measurement gap.
    Returns (host_ms_per_iter, device_ms_per_iter); caller should flag a
    large disagreement between the two rather than silently trusting one."""
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    t0 = time.perf_counter()
    start_evt.record()
    for _ in range(iters):
        fn()
    end_evt.record()
    torch.cuda.synchronize()
    host_s = time.perf_counter() - t0
    device_ms = start_evt.elapsed_time(end_evt)

    return (host_s * 1000.0) / iters, device_ms / iters


# ─── profiler-based breakdown ───────────────────────────────────────────────


def profile_iters(fn, iters, trace_path=None):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    if trace_path:
        try:
            prof.export_chrome_trace(trace_path)
        except Exception as e:
            print(f"  [WARN] could not export chrome trace to {trace_path}: {e}")
    return prof


def analyze_profile(prof, iters):
    """Best-effort numeric summary on top of key_averages(); the printed
    key_averages() table (see caller) is the ground truth if any of this
    extraction disagrees with it due to a profiler API difference across
    torch versions. Returns None (with a printed warning, not a crash) if
    the extraction itself fails.

    Deliberately does NOT report an aggregate "GPU busy time" or
    "launch-dispatch time as a fraction of wall-clock" anymore -- a real run
    proved that unreliable: key_averages() gives graph.replay (a
    record_function label wrapping torch.cuda.graph().replay()) its own row
    whose self_device_time_total is the SUM of every kernel that executed
    during that replay, AND ALSO gives each of those same kernels their own
    separate named rows (e.g. "cutlass::Kernel2...",
    "elementwise_kernel..."). Blindly summing self_device_time_total across
    every DeviceType.CUDA row therefore double(+)-counts everything inside
    a captured graph -- confirmed by a real GPU run reporting GPU busy time
    exceeding total wall-clock time and "launch-dispatch fraction" at
    100.3%, both physically impossible for a true non-overlapping CPU
    dispatch cost. The per-LABEL numbers below (regions_per_iter) don't have
    this problem -- each key is looked up individually, not summed across
    every CUDA-typed row -- so they're kept. See the caller's
    top_level_events_only=True table for a correct, non-double-counted
    aggregate CPU-side view instead."""
    try:
        from torch.autograd import DeviceType  # noqa: F401  (kept for callers' own use)

        avgs = prof.key_averages()

        launch_count = 0
        for e in avgs:
            key = e.key or ""
            if any(h.lower() in key.lower() for h in LAUNCH_NAME_HINTS):
                launch_count += e.count

        regions = {}
        for e in avgs:
            if e.key in REGION_LABELS:
                regions[e.key] = {
                    "count": e.count,
                    "cpu_us_total": e.cpu_time_total,
                    "device_us_total": e.device_time_total,
                }

        return {
            "launch_count_per_iter": launch_count / iters,
            "regions_per_iter": {
                k: {
                    "count": v["count"] / iters,
                    "cpu_ms": (v["cpu_us_total"] / iters) / 1000.0,
                    "device_ms": (v["device_us_total"] / iters) / 1000.0,
                }
                for k, v in regions.items()
            },
        }
    except Exception as e:
        print(f"  [WARN] analyze_profile extraction failed ({type(e).__name__}: {e}) -- "
              f"likely a torch-version profiler API difference. Falling back to the raw "
              f"key_averages().table() output below; read launch counts/times from there instead.")
        return None


def _top_level_table(prof, sort_by="cpu_time_total", row_limit=15):
    """Excludes nested children (e.g. the individual kernels replayed inside
    a captured CUDA graph, or the ops inside compute_logits/state_manager
    calls) -- a correct, non-double-counted view of CPU-side work, unlike
    blindly summing every row of key_averages(). Best-effort: returns None
    (caller just skips printing it) if this torch version's EventList.table
    doesn't support top_level_events_only the way this was written against."""
    try:
        return prof.key_averages().table(sort_by=sort_by, row_limit=row_limit, top_level_events_only=True)
    except Exception as e:
        print(f"  [WARN] top_level_table failed ({type(e).__name__}: {e}) -- skipping, "
              f"use the full key_averages table below instead")
        return None


def print_analysis(label, host_ms, device_ms, analysis, top_k_table=None, top_level_table=None):
    print(f"\n--- {label} ---")
    print(f"  wall-clock (unprofiled, host perf_counter):  {host_ms:.3f} ms/iter")
    print(f"  wall-clock (unprofiled, CUDA event):          {device_ms:.3f} ms/iter")
    disagreement = abs(host_ms - device_ms) / max(host_ms, device_ms, 1e-6)
    if disagreement > 0.15:
        print(f"  [WARN] host vs device timing disagree by {disagreement*100:.0f}% "
              f"-- possible async-completion gap, investigate before trusting either number")
    if analysis is not None:
        print(f"  CPU-side launch/copy API calls per iter (cudaLaunchKernel+cudaGraphLaunch+"
              f"memcpy/memset async -- note a captured CUDA graph replays as ONE "
              f"cudaGraphLaunch regardless of how many kernels it contains, so this is "
              f"NOT a per-kernel count once a graph is involved): "
              f"{analysis['launch_count_per_iter']:.1f}")
        if analysis["regions_per_iter"]:
            print("  labeled regions (count / cpu_ms / device_ms per iter -- each individually "
                  "trustworthy, not summed across rows):")
            for k, v in analysis["regions_per_iter"].items():
                print(f"    {k:<24s} count={v['count']:.1f}  cpu={v['cpu_ms']:.3f}ms  device={v['device_ms']:.3f}ms")
    if top_level_table:
        print("  top-level-only view (excludes nested children -- correct, non-double-counted "
              "CPU-side breakdown):")
        print(top_level_table)
    if top_k_table:
        print(top_k_table)


# ─── fused-GDR capturability micro-check ───────────────────────────────────


def check_fused_gdr_capturability(device="cuda"):
    print("\n--- fused-GDR-kernel capturability check ---")
    try:
        import fla  # noqa: F401
    except ImportError:
        print("  [SKIP] flash-linear-attention ('fla') not installed in this "
              "environment -- cannot test capturability here. Run this on the "
              "GPU box where use_fused_gdr_kernel is actually intended to run.")
        return None

    from nanovllm.models.qwen3_5 import Qwen35LinearAttention

    torch.manual_seed(3)
    layer = Qwen35LinearAttention(
        hidden_size=512, linear_attn_kq_heads=8, linear_attn_v_heads=16,
        linear_attn_head_dim=64, conv_kernel_size=4, rms_norm_eps=1e-6,
        use_fused_gdr_kernel=True,
    ).to(device).to(torch.bfloat16)
    layer.eval()

    T = 64
    x = torch.randn(T, 512, device=device, dtype=torch.bfloat16)
    # Single multi-token segment -> is_decode_shape=False -> fused path is
    # actually exercised (decode's is_decode_shape=True would silently skip
    # straight to the sequential path regardless of this capture attempt).
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

    try:
        with torch.no_grad():
            layer(x, cu_seqlens)  # warmup OUTSIDE capture: forces Triton
            # autotune/compile + any first-call cudaMalloc before entering
            # the graph context -- same discipline capture_cudagraph() itself
            # already uses (`_step()    # warmup` before `with torch.cuda.graph(...)`).
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(g):
                layer(x, cu_seqlens)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        print("  [OK] chunk_gated_delta_rule IS capturable inside torch.cuda.graph "
              "(after an out-of-capture warmup call).")
        return True
    except Exception as e:
        print(f"  [FAIL] fused GDR kernel is NOT capturable: {type(e).__name__}: {e}")
        print("  This would be a direct conflict with any plan to graph-capture "
              "prefill or to enable the fused path inside decode -- surface immediately.")
        return False


# ─── main ───────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "fake_qwen35_small"))
    ap.add_argument("--no-fake-config-loader", dest="fake_config_loader", action="store_false", default=True)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--prefill-len", type=int, default=256)
    ap.add_argument("--prefill-batch-size", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--warmup-iters", type=int, default=3)
    ap.add_argument("--timed-iters", type=int, default=10, help="unprofiled timing pass")
    ap.add_argument("--profiled-iters", type=int, default=5, help="profiler-wrapped pass")
    ap.add_argument("--use-fused-gdr-kernel", action="store_true", default=False,
                     help="Build the runner itself with the flag on (decode still never "
                          "invokes the fused path -- is_decode_shape gates it off "
                          "regardless -- this only changes whether the flag is threaded "
                          "through at all; use --check-fused-gdr-capture for the real "
                          "capturability question).")
    ap.add_argument("--check-fused-gdr-capture", dest="check_fused_gdr_capture",
                     action="store_true", default=True)
    ap.add_argument("--no-check-fused-gdr-capture", dest="check_fused_gdr_capture", action="store_false")
    ap.add_argument("--trace-dir", default=os.path.join(os.path.dirname(__file__), "profiling_output"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[SKIP] no CUDA GPU available in this environment")
        return

    os.makedirs(args.trace_dir, exist_ok=True)

    max_num_seqs = max(args.batch_sizes + [args.prefill_batch_size])

    print(f"Building ModelRunner (model={args.model}, max_num_seqs={max_num_seqs}, "
          f"use_fused_gdr_kernel={args.use_fused_gdr_kernel})...")
    runner, config = build_runner(
        args.model, args.fake_config_loader, max_num_seqs, args.max_model_len,
        args.gpu_memory_utilization, args.use_fused_gdr_kernel,
    )
    print(f"  [OK] constructed. graphs captured for batch sizes: "
          f"{getattr(runner, 'graph_bs', 'NONE -- enforce_eager or capture failed')}")
    init_weights_and_guard(runner, args.fake_config_loader)

    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.sequence import Sequence

    scheduler = Scheduler(config, runner.state_manager, runner)

    is_hybrid = runner.state_manager is not None
    if is_hybrid:
        layer_types = runner.model.model.layer_types
        n_full = sum(1 for t in layer_types if t == "full_attention")
        n_linear = sum(1 for t in layer_types if t == "linear_attention")
        print(f"  layer_types: {n_linear} linear_attention (GDR), {n_full} full_attention "
              f"-- ratio {n_linear}:{n_full}")

    # ── prefill profiling ──────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PREFILL STEP")
    print("=" * 78)

    # SAME seed on every single prefill call below -- warmup, timed, AND
    # profiled -- not a fresh one per iteration. Varied lengths ACROSS the
    # batch (make_prefill_batch's own per-prompt randint) are still real;
    # what must NOT vary is the total packed token count N *between*
    # iterations of this timing loop, since several ops in layers/ are
    # @torch.compile'd (layernorm.py's rms_forward, etc.) and dynamo
    # recompiles on every new shape it sees -- confirmed the hard way: an
    # earlier version of this script incremented the seed every call,
    # producing a different N each of the ~18 warmup+timed+profiled prefill
    # calls, and measured 910ms/iter wall-clock on an 8-layer, hidden=512
    # toy model at bs=4 -- two to three orders of magnitude more than
    # genuine compute for that size, and a dead match for this project's
    # own documented issue (bench_throughput.py's docstring: "dynamo
    # recompiles per new input shape... Sticking to {1,2,4} keeps total
    # distinct compiled shapes... under torch._dynamo's default
    # cache_size_limit of 8"). Holding N fixed lets warmup pay the
    # compilation cost once, off the clock, exactly like that precedent.
    PREFILL_SEED = 4242

    for w in range(args.warmup_iters):
        added = make_prefill_batch(scheduler, Sequence, args.prefill_batch_size, args.prefill_len, PREFILL_SEED)
        run_prefill_to_completion(runner, scheduler, added)
        cleanup_running(scheduler, added)

    def make_and_run_prefill():
        added = make_prefill_batch(scheduler, Sequence, args.prefill_batch_size, args.prefill_len, PREFILL_SEED)
        run_prefill_to_completion(runner, scheduler, added)
        cleanup_running(scheduler, added)

    host_ms, device_ms = time_iters(make_and_run_prefill, args.timed_iters)

    with RegionInstrumentation(runner):
        trace_path = os.path.join(args.trace_dir, "prefill_trace.json")
        prof = profile_iters(make_and_run_prefill, args.profiled_iters, trace_path)

    analysis = analyze_profile(prof, args.profiled_iters)
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
    tl_table = _top_level_table(prof)
    print_analysis(f"prefill (bs={args.prefill_batch_size}, len~{args.prefill_len})",
                    host_ms, device_ms, analysis, top_k_table=table, top_level_table=tl_table)
    print(f"  chrome trace: {trace_path}")

    # ── decode profiling, per batch size ───────────────────────────────────
    print("\n" + "=" * 78)
    print("DECODE STEP")
    print("=" * 78)

    for bs in args.batch_sizes:
        print(f"\n### batch_size={bs} ###")
        added = make_prefill_batch(scheduler, Sequence, bs, args.prefill_len, seed=2000 + bs)
        run_prefill_to_completion(runner, scheduler, added)

        decode_seqs = list(scheduler.running)
        assert len(decode_seqs) == bs, f"expected {bs} running sequences, got {len(decode_seqs)}"

        use_graph_expected = (not runner.enforce_eager and hasattr(runner, "graphs")
                               and bs <= 512)
        print(f"  use_graph condition (mirrors ModelRunner.run()): {use_graph_expected}")
        if use_graph_expected:
            chosen = next((g for g in runner.graph_bs if g >= bs), None)
            print(f"  -> would replay graph captured for padded batch size {chosen}")
        else:
            print("  -> would fall back to EAGER decode -- if unexpected, this IS "
                  "the 'silent fallback' failure mode Step 1 asked to rule out")

        def decode_step():
            seqs, is_prefill = scheduler.schedule()
            assert not is_prefill, f"expected decode round at bs={bs}"
            token_ids = runner.run(seqs, False)
            scheduler.postprocess(seqs, token_ids, False)

        for _ in range(args.warmup_iters):
            decode_step()

        host_ms, device_ms = time_iters(decode_step, args.timed_iters)

        with RegionInstrumentation(runner):
            trace_path = os.path.join(args.trace_dir, f"decode_trace_bs{bs}.json")
            prof = profile_iters(decode_step, args.profiled_iters, trace_path)

        analysis = analyze_profile(prof, args.profiled_iters)
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
        tl_table = _top_level_table(prof)
        print_analysis(f"decode (bs={bs})", host_ms, device_ms, analysis, top_k_table=table, top_level_table=tl_table)
        print(f"  chrome trace: {trace_path}")

        # Also profile eager decode at this batch size for the layer-type
        # breakdown -- under graph replay everything collapses into a
        # single CPU "graph.replay" op, so GDR_layer/FullAttn_layer/
        # MoE_layer labels only fire (are only individually visible at the
        # CPU-op level) when NOT replaying a graph.
        prev_eager = runner.enforce_eager
        runner.enforce_eager = True
        for _ in range(args.warmup_iters):
            decode_step()
        with RegionInstrumentation(runner):
            trace_path = os.path.join(args.trace_dir, f"decode_trace_bs{bs}_eager_layertype.json")
            prof_eager = profile_iters(decode_step, args.profiled_iters, trace_path)
        runner.enforce_eager = prev_eager

        analysis_eager = analyze_profile(prof_eager, args.profiled_iters)
        print(f"  [eager decode, same bs={bs}, for layer-type attribution only -- "
              f"NOT a graph-vs-eager speed comparison, warmup wasn't reset]")
        if analysis_eager is not None:
            for k, v in analysis_eager["regions_per_iter"].items():
                if k in ("GDR_layer", "FullAttn_layer", "MoE_layer"):
                    print(f"    {k:<16s} count={v['count']:.1f}  cpu={v['cpu_ms']:.3f}ms  device={v['device_ms']:.3f}ms")
        else:
            eager_table = prof_eager.key_averages().table(sort_by="cpu_time_total", row_limit=15)
            print(eager_table)

        cleanup_running(scheduler, decode_seqs)

    # ── fused-GDR capturability ─────────────────────────────────────────────
    if args.check_fused_gdr_capture:
        check_fused_gdr_capturability()

    print("\n" + "=" * 78)
    print("Done. Read the printed breakdowns above against the Step-1 questions in "
          "this file's module docstring, and inspect the exported chrome traces in "
          f"{args.trace_dir} (chrome://tracing or https://ui.perfetto.dev) for anything "
          "the automated extraction may have mis-attributed.")


if __name__ == "__main__":
    main()
