"""Prints ACTUAL decoded text from the fused-kernel + CUDA-graph-capture
combination -- the one combination never yet checked for real output
correctness (cluster_q6_moe_w8a8_gsm8k.py hardcodes enforce_eager=True, so
it validated the kernel's math but never exercised graph capture; the
172.7 tok/s run only confirmed the trial completed, not that its output was
sensible). A tok/s number cannot distinguish "fast and correct" from "fast
because it's silently producing garbage" -- reading the actual text can.

Usage:
    NANOVLLM_USE_FUSED_MOE_KERNEL=1 python tests/diag_fused_kernel_graph_readout.py
"""
import argparse
import os
import sys
import types

# tests/ itself on sys.path (bench_throughput.py-style modules some of our
# imports transitively need), plus the repo-root "nanovllm" package shim
# every real entry point in this project sets up (repo root isn't an
# installed package -- this fakes it into sys.modules the same way
# bench_throughput.py/engine/model_runner.py do).
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if "nanovllm" not in sys.modules:
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "qwen35_checkpoint"))
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    ap.add_argument("--enforce-eager", action="store_true", default=False)
    ap.add_argument("--concurrency", type=int, default=16,
                    help="How many prompts to submit at once, and max_num_seqs to build "
                         "with. Matters beyond just batch size: a different concurrency "
                         "means a DIFFERENT captured CUDA graph bucket (graph_bs in "
                         "capture_cudagraph()) -- correctness at one concurrency does not "
                         "guarantee correctness at another, so this needs pointing at "
                         "whichever bucket you actually want to verify, not left at the "
                         "default.")
    args = ap.parse_args()

    print(f"NANOVLLM_USE_FUSED_MOE_KERNEL={os.environ.get('NANOVLLM_USE_FUSED_MOE_KERNEL', '0')}  "
          f"enforce_eager={args.enforce_eager}  tp={args.tp}  concurrency={args.concurrency}", flush=True)

    # Not `from nanovllm import LLM, SamplingParams` -- the repo-root
    # __init__.py exposes those via a module-level __getattr__ (PEP 562),
    # which only works if that file's own code actually ran. The manual
    # sys.modules["nanovllm"] shim above (same pattern bench_throughput.py
    # uses) creates a bare empty module object and never executes
    # __init__.py, so that __getattr__ is never attached -- submodule
    # imports sidestep the whole issue and match how every other script in
    # this project actually does it.
    from nanovllm.llm import LLM
    from nanovllm.sampling_params import SamplingParams

    llm = LLM(
        args.checkpoint,
        tensor_parallel_size=args.tp,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.concurrency,
        max_model_len=2048,
        use_moe_w8a8=True,
        moe_w8a8_weight_group_size=128,
    )

    base_prompts = [
        "Q: If a train travels 60 miles in 1.5 hours, what is its average speed in miles per hour? A:",
        "Q: What is 15% of 200? A:",
        "Q: A rectangle has length 8 and width 5. What is its area? A:",
        "The capital of France is",
        "Q: If a car travels 90 miles in 1.5 hours, what is its average speed in miles per hour? A:",
        "Q: What is 25% of 400? A:",
        "Q: A rectangle has length 12 and width 4. What is its area? A:",
        "The capital of Japan is",
    ]
    # Cycle through the base set to fill the requested concurrency -- actually
    # submits `concurrency` simultaneous requests, exercising the real batch
    # size / graph bucket, not just a handful regardless of --concurrency.
    prompts = [base_prompts[i % len(base_prompts)] for i in range(args.concurrency)]
    sp = SamplingParams(temperature=0, max_tokens=64)

    print(f"\nGenerating {len(prompts)} prompts at once (temperature=0, deterministic) ...\n", flush=True)
    outputs = llm.generate(prompts, sp, use_tqdm=False)

    # Group by prompt text: print each DISTINCT prompt's answer once (avoids
    # flooding output at concurrency=32+), and separately check that EVERY
    # repeat of the same prompt at a different batch slot produced the
    # IDENTICAL answer -- this project has real prior history of
    # decode-time slot-reuse/cross-request contamination bugs (see
    # README.md), so this isn't a hypothetical check at higher concurrency.
    by_prompt = {}
    for p, o in zip(prompts, outputs):
        by_prompt.setdefault(p, []).append(o["text"])

    for p, texts in by_prompt.items():
        print("=" * 78)
        print(f"PROMPT: {p}")
        print(f"OUTPUT: {texts[0]!r}")
        if len(texts) > 1:
            all_same = all(t == texts[0] for t in texts)
            status = "OK, all identical" if all_same else "MISMATCH -- possible cross-request contamination"
            print(f"  ({len(texts)} copies at different batch slots: {status})")
    print("=" * 78)
    print(f"\n{len(prompts)} total requests at concurrency={args.concurrency}, "
          f"{len(by_prompt)} distinct prompts. Read the OUTPUTs for coherence, and check "
          f"none of the repeat-consistency lines above say MISMATCH.")
    print("\nRead these by eye: do they look like real, coherent answers, or "
          "garbage/repeated tokens/gibberish? That's the actual question here, "
          "not the tok/s number.")

    llm.exit()


if __name__ == "__main__":
    main()
