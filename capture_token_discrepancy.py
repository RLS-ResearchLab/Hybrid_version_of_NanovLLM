#!/usr/bin/env python
"""Item 3 (today's A6000 validation session) -- token-count discrepancy
CAPTURE ONLY, no diagnosis. Direct-engine (no HTTP), matching
bench_throughput.py's exact workload -- adversarial random-vocab prompts,
temperature=1.0, ignore_eos=True -- since that's the setup under which the
~10-18% reported-vs-retokenized completion-token gap was originally
observed. The HTTP path used for today's Item 1 sweep can't supply this:
/v1/chat/completions only returns a token COUNT, never the actual
completion token IDs.

For --num-seqs sequences, run together as ONE real batched generate() call
(concurrency == --num-seqs, matching how the original discrepancy was
seen under batching): saves the engine-reported completion token IDs
(LLMEngine.generate()'s own token_ids, pre-decode) alongside an independent
re-tokenization of that SAME output's decoded text (re-encoding
tokenizer.decode(token_ids) back through tokenizer.encode), side by side,
to a JSON file. Diagnosis (why they differ) is explicitly out of scope for
this script -- that happens offline, per today's task rules.

Usage:
    python capture_token_discrepancy.py --model qwen35_checkpoint \\
        --tensor-parallel-size 2 --num-seqs 8 --prompt-len 32 --output-len 128 \\
        --out token_discrepancy_capture.json
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
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


def make_prompts(n, prompt_len, seed):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randint(100, 5000, (prompt_len,), generator=g).tolist()
        for _ in range(n)
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--tensor-parallel-size", type=int, default=2, dest="tensor_parallel_size")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9, dest="gpu_memory_utilization")
    p.add_argument("--max-model-len", type=int, default=2048, dest="max_model_len")
    p.add_argument("--max-num-batched-tokens", type=int, default=2048, dest="max_num_batched_tokens")
    p.add_argument("--num-seqs", type=int, default=8, dest="num_seqs",
                    help="Also used as max_num_seqs -- all run together in one real batched call.")
    p.add_argument("--prompt-len", type=int, default=32)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--enforce-eager", dest="enforce_eager", action="store_true", default=True)
    p.add_argument("--out", default="token_discrepancy_capture.json")
    args = p.parse_args()

    from nanovllm.llm import LLM
    from nanovllm.sampling_params import SamplingParams

    print(f"Loading engine from {args.model} (tensor_parallel_size={args.tensor_parallel_size})...")
    llm = LLM(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.num_seqs,
        enforce_eager=args.enforce_eager,
    )

    prompts = make_prompts(args.num_seqs, args.prompt_len, args.seed)
    sp = SamplingParams(max_tokens=args.output_len, temperature=1.0, ignore_eos=True)

    print(f"Running {args.num_seqs} sequences together (real batched generate(), "
          f"prompt_len={args.prompt_len}, output_len={args.output_len}, "
          f"temperature=1.0, ignore_eos=True)...")
    outputs = llm.generate(prompts, sp, use_tqdm=False)

    records = []
    for i, out in enumerate(outputs):
        reported_ids = out["token_ids"]
        decoded_text = out["text"]
        retokenized_ids = llm.tokenizer.encode(decoded_text, add_special_tokens=False)
        record = {
            "seq_index": i,
            "reported_completion_token_ids": reported_ids,
            "reported_count": len(reported_ids),
            "decoded_text": decoded_text,
            "retokenized_token_ids": retokenized_ids,
            "retokenized_count": len(retokenized_ids),
            "count_diff": len(retokenized_ids) - len(reported_ids),
        }
        records.append(record)
        print(f"  seq[{i}]: reported={len(reported_ids)} retokenized={len(retokenized_ids)} "
              f"diff={record['count_diff']:+d}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    llm.exit()
    print(f"\nSaved {len(records)} side-by-side records to {args.out}")


if __name__ == "__main__":
    main()
