"""GSM8K-shaped integration check for use_fused_gdr_kernel, on the small
fake model (tests/fake_qwen35_small) -- the step this task's own template
requires (single-forward-pass comparison -> multi-sequence batched
comparison -> full GSM8K-shaped run) that had NOT been done: everything so
far exercised either a single forward call or a single prefill call, never
real multi-step autoregressive decode through the full engine (scheduler,
KV cache reuse across steps, EOS handling) on realistic variable-length
prompts.

IMPORTANT -- what this does and does NOT check: fake_qwen35_small has
RANDOM, untrained weights (see tests/make_fake_hf_config.py /
make_fake_checkpoint.py), so "GSM8K exact match" against the gold answer is
meaningless here -- a fresh random-init model has no learned math ability
regardless of this flag. What IS meaningful, and what this script actually
checks, is CONSISTENCY: does use_fused_gdr_kernel=True produce the same
generated completions (token-for-token, deterministic temperature=0) as
use_fused_gdr_kernel=False, on real GSM8K 8-shot CoT prompts, through real
multi-step decode -- not just a single forward pass. This is the small-model
stand-in for this task's GSM8K success criterion; it is NOT a substitute for
running actual GSM8K accuracy on the real checkpoint (see original task's
hypothesis: "GSM8K exact match unchanged beyond noise" -- that claim can
only be evaluated with trained weights, on the real checkpoint, later).

Structure borrowed directly from tests/gsm8k_smoke_test.py (same
build_prompt/extract_answer_detailed reuse, same temperature=0/max_tokens
sampling discipline) -- pointed at the small fake model instead of the real
checkpoint, and run once per flag value via --use-fused-gdr-kernel, writing
to a flag-labeled output file so both runs can be diffed afterward.

Usage (run BOTH, then diff -- see module-level comment at the bottom for
the exact diff command):
    python tests/gsm8k_fused_gdr_check.py
    python tests/gsm8k_fused_gdr_check.py --use-fused-gdr-kernel
"""
import argparse
import json
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_WS_NAME = os.path.basename(ROOT)
if _WS_NAME != "nanovllm" and "nanovllm" not in sys.modules:
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

sys.path.insert(0, os.path.dirname(__file__))
from gsm8k_prompt import build_prompt  # noqa: E402
from gsm8k_extract import extract_answer_detailed  # noqa: E402

CKPT_DIR = os.path.join(ROOT, "tests", "fake_qwen35_small")
CACHE_DIR = os.path.join(ROOT, "tests", "_gsm8k_cache")

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
    """Same shim as bench_throughput.py's build_engine. Confirmed root
    cause (checked directly, not guessed): transformers==5.14.1 actually
    HAS "qwen3_5_moe" registered now (Qwen3_5MoeConfig) and correctly
    reads our flat config.json's num_hidden_layers=8 onto the top-level
    object. But that config class nests a `text_config` sub-object with
    its OWN independent defaults -- including a real 40-layer,
    3-GDR+1-full-attention `layer_types` list (transformers ships the
    actual Qwen3.5-35B-A3B shape as ITS default now). Our fake
    config.json never overrides text_config, so it keeps that 40-entry
    default. config.py's own __post_init__ flattening (written generally
    to support VLM checkpoints that nest fields under text_config) then
    copies text_config.layer_types onto the top-level object because the
    top-level one is None -- landing a 40-element list next to
    num_hidden_layers=8, which is exactly what crashed
    _get_layer_types's `assert len(layers_block_type) == num_layers`.
    Bypassing AutoConfig entirely and reading config.json as a flat dict
    (no text_config, no derived defaults) sidesteps this -- same fix
    bench_throughput.py already uses successfully against this exact
    fake checkpoint, reused here rather than reinvented."""
    with open(os.path.join(path, "config.json")) as f:
        d = json.load(f)
    d = AttrDict(d)
    d.dtype = _DTYPE_MAP[d.pop("torch_dtype")]
    return d

MAX_MODEL_LEN = 2048
MAX_TOKENS = 64  # short on purpose -- consistency check, not a real eval; enough
                 # decode steps to exercise the real multi-step path without
                 # paying for 512 tokens x random-weight rambling
BATCH_SIZE = 4
NUM_EXAMPLES = 8  # 2 batches of 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-fused-gdr-kernel", dest="use_fused_gdr_kernel",
                     action="store_true", default=False)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6,
                     help="See earlier session: torch.cuda.mem_get_info()-based "
                          "budget, this GPU needed >~0.42 before subtracting "
                          "this run's own footprint. Check again if this fails.")
    args = ap.parse_args()

    tag = "on" if args.use_fused_gdr_kernel else "off"
    os.makedirs(CACHE_DIR, exist_ok=True)
    results_path = os.path.join(CACHE_DIR, f"fused_gdr_check_{tag}.jsonl")
    if os.path.exists(results_path):
        os.remove(results_path)

    import nanovllm.config as config_mod
    config_mod.AutoConfig.from_pretrained = staticmethod(_fake_from_pretrained)

    from datasets import load_dataset
    from nanovllm.llm import LLM
    from nanovllm.sampling_params import SamplingParams

    print("Loading openai/gsm8k (main, test split) ...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    subset = ds.select(range(NUM_EXAMPLES))

    print(f"Loading small fake engine from {CKPT_DIR} "
          f"(use_fused_gdr_kernel={args.use_fused_gdr_kernel}) ...")
    llm = LLM(
        CKPT_DIR,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=BATCH_SIZE,
        max_num_batched_tokens=2048,
        max_model_len=MAX_MODEL_LEN,
        use_fused_gdr_kernel=args.use_fused_gdr_kernel,
    )
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

    all_records = []
    n_batches = (NUM_EXAMPLES + BATCH_SIZE - 1) // BATCH_SIZE

    with open(results_path, "a", encoding="utf-8") as f:
        for b in range(n_batches):
            lo, hi = b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, NUM_EXAMPLES)
            batch_examples = [subset[i] for i in range(lo, hi)]
            prompts = [build_prompt(ex["question"]) for ex in batch_examples]

            print(f"\n=== Batch {b + 1}/{n_batches} (examples {lo}-{hi - 1}) ===")
            outputs = llm.generate(prompts, sp)

            for i, ex, out in zip(range(lo, hi), batch_examples, outputs):
                model_text = out["text"]
                model_result = extract_answer_detailed(model_text)
                record = {
                    "idx": i,
                    "model_output": model_text,
                    "token_ids": out["token_ids"],
                    "extracted_answer": model_result.value,
                }
                all_records.append(record)
                f.write(json.dumps(record) + "\n")
                f.flush()
                print(f"  idx={i}  extracted={model_result.value}  "
                      f"text[:80]={model_text[:80]!r}")

    print(f"\nWrote {len(all_records)} records to {results_path}")
    print("Run the OTHER flag value next, then diff:")
    print(f"  python -c \"import json; "
          f"a=[json.loads(l) for l in open('{CACHE_DIR}/fused_gdr_check_off.jsonl')]; "
          f"b=[json.loads(l) for l in open('{CACHE_DIR}/fused_gdr_check_on.jsonl')]; "
          f"diffs=[(x['idx'], x['token_ids']==y['token_ids']) for x,y in zip(a,b)]; "
          f"print('token-identical:', sum(1 for _,ok in diffs if ok), '/', len(diffs)); "
          f"[print('  DIVERGED idx', i) for i,ok in diffs if not ok]\""
          )


if __name__ == "__main__":
    main()
