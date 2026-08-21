#!/usr/bin/env python
"""Throughput measurement harness: concurrency sweep, warm-up, repeated
trials, CSV logging. Not a comparison tool -- a run against the small fake
model is pure characterization, meant to validate the harness itself (timers,
warm-up handling, whether enforce_eager vs CUDA graphs changes anything)
before trusting any number it produces against the real 35B checkpoint.

Concurrency defaults to 1, 2, 4 -- deliberately not higher. Per README.md's
open-items list, Phase 2 (continuous batching for the hybrid model) has an
unresolved item: a cosine~=0.947 result on one packed sequence that hasn't
been root-caused as numerical noise vs. a real cross-sequence contamination
bug. Measuring throughput at higher concurrency would be timing a code path
already known to be suspect. Don't pass --concurrency values above 4 until
that's closed.

Each concurrency level gets its own warm-up trials (discarded, not logged as
timed). This matters here specifically because Qwen35RMSNorm.rms_forward is
@torch.compile'd (layers/layernorm.py) -- dynamo recompiles per new input
shape, and prefill/decode at each concurrency level are new shapes it hasn't
seen. Warm-up pays that cost off the clock instead of inside a "trial".
Sticking to {1, 2, 4} also keeps total distinct compiled shapes (prefill +
decode, per level) under torch._dynamo's default cache_size_limit of 8 --
exceeding it silently falls back to eager for further shapes mid-run, which
would quietly contaminate later measurements.

Prompts are raw token-id lists (not text), so prompt_len is exact -- this
sidesteps the tokenizer's variable text->token ratio entirely and avoids any
dependency on which tokenizer happens to be attached to the model dir.

Usage (small fake model -- sanity-checking the harness itself):
    python tests/make_fake_hf_config.py
    python tests/make_fake_checkpoint.py
    python tests/make_fake_tokenizer.py
    python bench_throughput.py --model tests/fake_qwen35_small \\
        --prompt-len 32 --output-len 64 --max-model-len 512 \\
        --gpu-memory-utilization 0.2 --csv-out throughput_small.csv

Usage (real checkpoint):
    python bench_throughput.py --model /path/to/qwen3.5-35b-a3b \\
        --prompt-len <P> --output-len <O> --max-model-len <L> \\
        --gpu-memory-utilization <U> --tensor-parallel-size 2 \\
        --no-fake-config-loader --csv-out throughput_35b.csv
"""

'''
command that need to be used :
python bench_throughput.py --model tests/fake_qwen35_small     
--prompt-len 32 --output-len 64 --max-model-len 512     
--gpu-memory-utilization 0.2 --csv-out throughput_small1.csv
'''
import argparse
import csv
import json
import os
import sys
import time

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
    """Same shim used by tests/test_qwen35_*.py -- needed because
    model_type "qwen3_5_moe" isn't registered with transformers' AutoConfig."""
    with open(os.path.join(path, "config.json")) as f:
        d = json.load(f)
    d = AttrDict(d)
    d.dtype = _DTYPE_MAP[d.pop("torch_dtype")]
    return d


def build_engine(args):
    if args.fake_config_loader:
        import nanovllm.config as config_mod
        config_mod.AutoConfig.from_pretrained = staticmethod(_fake_from_pretrained)

    from nanovllm.engine.llm_engine import LLMEngine
    return LLMEngine(
        args.model,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=max(args.concurrency),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        use_fused_gdr_kernel=args.use_fused_gdr_kernel,
        use_moe_w8a8=getattr(args, "use_moe_w8a8", False),
        moe_w8a8_weight_group_size=getattr(args, "moe_w8a8_weight_group_size", 128),
    )


def make_prompts(n, prompt_len, seed):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randint(100, 5000, (prompt_len,), generator=g).tolist()
        for _ in range(n)
    ]


def run_trial(engine, concurrency, prompt_len, output_len, seed):
    from nanovllm.sampling_params import SamplingParams

    prompts = make_prompts(concurrency, prompt_len, seed)
    sp = SamplingParams(max_tokens=output_len, temperature=1.0, ignore_eos=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = engine.generate(prompts, sp, use_tqdm=False)
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0

    completion_lengths = [len(o["token_ids"]) for o in outputs]
    total_output_tokens = sum(completion_lengths)
    return {
        "wall_s": wall_s,
        "completion_lengths": completion_lengths,
        "total_output_tokens": total_output_tokens,
        "tok_s": total_output_tokens / wall_s,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True)
    p.add_argument(
        "--concurrency", type=int, nargs="+", default=[1, 2, 4],
        help="Default [1,2,4]. Do not raise past 4 until the Phase 2 batching "
             "cosine~=0.947 open item in README.md is resolved.",
    )
    p.add_argument("--prompt-len", type=int, default=32)
    p.add_argument("--output-len", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument(
        "--max-num-batched-tokens", type=int, default=512, dest="max_num_batched_tokens",
        help="Also sizes ModelRunner.warmup_model()'s internal prefill batch "
             "(num_seqs = min(max_num_batched_tokens // seq_len, max_num_seqs)) -- "
             "raising this raises warmup's peak memory, which allocate_kv_cache() "
             "subtracts from the KV-cache budget. Keep it just large enough for "
             "max(concurrency) * prompt_len, not larger.",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument(
        "--enforce-eager", dest="enforce_eager", action="store_true", default=True,
        help="Default. Eager decode, no CUDA graph capture.",
    )
    p.add_argument(
        "--cuda-graphs", dest="enforce_eager", action="store_false",
        help="Use CUDA-graph decode instead of eager mode.",
    )
    p.add_argument(
        "--use-fused-gdr-kernel", dest="use_fused_gdr_kernel", action="store_true", default=False,
        help="QLLM Stage 2: use flash-linear-attention's chunked kernel for GDR "
             "prefill instead of the sequential scan (models/qwen3_5.py, "
             "Qwen35LinearAttention). Default off -- matches the existing "
             "behavior for every caller that doesn't pass this flag. See "
             "tests/test_qwen35_fused_gdr.py for the correctness validation "
             "this should only be trusted after passing.",
    )
    p.add_argument(
        "--moe-w8a8", dest="use_moe_w8a8", action="store_true", default=False,
        help="Quantize MoE expert weights to INT8 after loading (Q4/Q6 -- correctness- "
             "and accuracy-validated, 40/40 GSM8K non-regression under chat-no-think; "
             "see moe_quantization_memo.md). Off by default. Previously only reachable "
             "via the tests/*.py wrapper scripts that build their own args object --"
             "build_engine() already reads use_moe_w8a8 via getattr, this just exposes "
             "it here too so running this file directly can use it.",
    )
    p.add_argument("--moe-w8a8-group-size", type=int, default=128, dest="moe_w8a8_weight_group_size",
                    help="Group size for INT8 grouped quantization. Only used when --moe-w8a8 is set.")
    p.add_argument("--trials", type=int, default=5, help="Timed trials per concurrency level.")
    p.add_argument(
        "--warmup-trials", type=int, default=2,
        help="Untimed trials per concurrency level, run and discarded before "
             "the timed trials -- see module docstring for why this is per-level.",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--csv-out", default="throughput_results.csv")
    p.add_argument(
        "--no-fake-config-loader", dest="fake_config_loader", action="store_false", default=True,
        help="Disable the config.json monkeypatch (only needed for the fake/small "
             "config's unregistered model_type; a real checkpoint with a "
             "transformers-registered config doesn't need it).",
    )
    args = p.parse_args()

    if max(args.concurrency) > 4:
        print(
            f"[WARNING] concurrency levels {args.concurrency} include values > 4. "
            f"Per README.md, Phase 2 batching has an unresolved cosine~=0.947 open "
            f"item -- you are about to measure a code path known to be suspect.",
            file=sys.stderr,
        )

    print(
        f"Building engine for model={args.model} "
        f"(max_num_seqs={max(args.concurrency)}, enforce_eager={args.enforce_eager})..."
    )
    engine = build_engine(args)

    fieldnames = [
        "timestamp", "model", "enforce_eager", "use_fused_gdr_kernel", "tensor_parallel_size",
        "gpu_memory_utilization", "max_model_len", "prompt_len", "output_len",
        "concurrency", "trial_index", "batch_composition_json",
        "total_output_tokens", "wall_s", "tok_s",
    ]
    write_header = not os.path.exists(args.csv_out)
    csv_file = open(args.csv_out, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    try:
        for concurrency in args.concurrency:
            print(f"\n=== concurrency={concurrency} ===")

            for w in range(args.warmup_trials):
                result = run_trial(engine, concurrency, args.prompt_len, args.output_len, seed=args.seed)
                print(
                    f"  warmup {w + 1}/{args.warmup_trials}: "
                    f"wall_s={result['wall_s']:.3f} tok/s={result['tok_s']:.1f} (discarded)"
                )

            tok_s_values = []
            for t in range(args.trials):
                result = run_trial(
                    engine, concurrency, args.prompt_len, args.output_len, seed=args.seed + t + 1
                )
                tok_s_values.append(result["tok_s"])
                print(
                    f"  trial {t + 1}/{args.trials}: wall_s={result['wall_s']:.3f} "
                    f"total_tokens={result['total_output_tokens']} "
                    f"tok/s={result['tok_s']:.1f} "
                    f"composition={result['completion_lengths']}"
                )

                writer.writerow({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "model": args.model,
                    "enforce_eager": args.enforce_eager,
                    "use_fused_gdr_kernel": args.use_fused_gdr_kernel,
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "max_model_len": args.max_model_len,
                    "prompt_len": args.prompt_len,
                    "output_len": args.output_len,
                    "concurrency": concurrency,
                    "trial_index": t,
                    "batch_composition_json": json.dumps(result["completion_lengths"]),
                    "total_output_tokens": result["total_output_tokens"],
                    "wall_s": result["wall_s"],
                    "tok_s": result["tok_s"],
                })
                csv_file.flush()

            mean_tok_s = sum(tok_s_values) / len(tok_s_values)
            print(f"  concurrency={concurrency}: mean tok/s over {args.trials} trials = {mean_tok_s:.1f}")
    finally:
        csv_file.close()
        engine.exit()

    print(f"\nResults written to {args.csv_out}")


if __name__ == "__main__":
    main()
