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

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(os.path.dirname(ROOT), "qwen35_checkpoint"))
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    ap.add_argument("--enforce-eager", action="store_true", default=False)
    args = ap.parse_args()

    print(f"NANOVLLM_USE_FUSED_MOE_KERNEL={os.environ.get('NANOVLLM_USE_FUSED_MOE_KERNEL', '0')}  "
          f"enforce_eager={args.enforce_eager}  tp={args.tp}", flush=True)

    from nanovllm import LLM, SamplingParams

    llm = LLM(
        args.checkpoint,
        tensor_parallel_size=args.tp,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=16,
        max_model_len=2048,
        use_moe_w8a8=True,
        moe_w8a8_weight_group_size=128,
    )

    prompts = [
        "Q: If a train travels 60 miles in 1.5 hours, what is its average speed in miles per hour? A:",
        "Q: What is 15% of 200? A:",
        "Q: A rectangle has length 8 and width 5. What is its area? A:",
        "The capital of France is",
    ]
    sp = SamplingParams(temperature=0, max_tokens=64)

    print("\nGenerating (temperature=0, deterministic) ...\n", flush=True)
    outputs = llm.generate(prompts, sp, use_tqdm=False)

    for p, o in zip(prompts, outputs):
        print("=" * 78)
        print(f"PROMPT: {p}")
        print(f"OUTPUT: {o['text']!r}")
    print("=" * 78)
    print("\nRead these by eye: do they look like real, coherent answers, or "
          "garbage/repeated tokens/gibberish? That's the actual question here, "
          "not the tok/s number.")

    llm.exit()


if __name__ == "__main__":
    main()
