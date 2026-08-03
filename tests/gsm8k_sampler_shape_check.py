"""Does the REAL, production self.sampler (layers/sampler.py's
@torch.compile-decorated Sampler, invoked inside the actual multi-process
engine -- not a bare standalone script, where torch.compile crashed with an
unrelated Triton/Inductor environment bug, see
tests/test_sampler_batch_invariance.py's failed run) pick a DIFFERENT
greedy token for the SAME row's logits depending on how many other rows
are stacked alongside it?

Built on tests/gsm8k_prefill_scale_check.py, which already confirmed the
raw PREFILL LOGITS themselves match (cosine>=0.99 + argmax) between
concurrency=1 and concurrency=8 for real GSM8K-scale prompts -- ruling out
the model computation itself. This script takes that one step further: it
captures the SAME already-consistent logits, then runs them through the
REAL self.sampler (via the new engine/model_runner.py diagnostic
sample_from_logits(), additive, mirrors run()'s exact
`self.sampler(logits, temperatures)` call) in two different stacking
arrangements -- once alone (batch=1) and once as part of the full 8-row
packed stack -- and checks whether the SAMPLED token (not just raw
.argmax()) agrees in both cases.

Usage:
    python tests/gsm8k_sampler_shape_check.py
"""
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

CKPT_DIR = os.path.join(ROOT, "qwen35_checkpoint")
MAX_MODEL_LEN = 2048
TARGET_INDICES = [0, 1, 2]
FILLER_INDICES = [3, 4, 5, 6, 7]
SELECTED_INDICES = TARGET_INDICES + FILLER_INDICES  # 8 total, concurrency=8


def _new_seq(llm, prompt: str):
    from nanovllm.engine.sequence import Sequence

    prompt_ids = llm.tokenizer.encode(prompt)
    seq = Sequence(prompt_ids)
    seq.num_scheduled_tokens = len(prompt_ids)
    llm.scheduler.block_manager.allocate(seq, 0)  # forced-uncached, same as gsm8k_prefill_scale_check.py
    if llm.model_runner.state_manager is not None:
        llm.model_runner.call("allocate_state_slot", seq)
    return seq, prompt_ids


def main():
    from datasets import load_dataset
    from nanovllm.llm import LLM

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if len(ds) != 1319:
        print(f"STOP: expected 1319 test examples, got {len(ds)}.")
        sys.exit(1)

    prompts = [build_prompt(ds[i]["question"]) for i in SELECTED_INDICES]

    llm = LLM(
        CKPT_DIR,
        enforce_eager=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_num_seqs=2 * len(prompts),
        max_model_len=MAX_MODEL_LEN,
    )

    print("\n" + "=" * 78)
    print("Capturing BASELINE (concurrency=1) and PACKED (concurrency=8) prefill logits")
    print("(same methodology as gsm8k_prefill_scale_check.py, already confirmed clean)")
    print("=" * 78)
    baseline_logits = []
    for prompt in prompts:
        seq, _ = _new_seq(llm, prompt)
        logits = llm.model_runner.call("get_prefill_logits", [seq])
        if logits.dim() == 2 and logits.shape[0] == 1:
            logits = logits.squeeze(0)
        baseline_logits.append(logits.float())

    packed_seqs = [_new_seq(llm, prompt)[0] for prompt in prompts]
    packed_logits = llm.model_runner.call("get_prefill_logits", packed_seqs)
    packed_logits = packed_logits.float()

    print("\n" + "=" * 78)
    print("SAMPLER SHAPE CHECK -- same logits, sampled ALONE (batch=1) vs STACKED (batch=8)")
    print("=" * 78)
    print(f"  {'#':>3}  {'gsm8k_idx':>9}  {'raw_argmax_base':>15}  {'raw_argmax_packed':>17}  "
          f"{'sampler_alone':>13}  {'sampler_stacked':>15}  {'ALL_MATCH':>9}")
    print("  " + "-" * 90)

    # sample_from_logits is rank0-only-local (no cross-GPU model computation,
    # just self.sampler on an already-computed logits tensor) -- call it
    # DIRECTLY on llm.model_runner (rank0's own object, in this process),
    # NOT via .call(), which broadcasts args to rank1 through a 1MB
    # shared-memory buffer that a (8, 248320) float32 tensor (~7.6MB)
    # blows straight past. Computed once outside the loop, not once per row.
    sampled_stacked_all = llm.model_runner.sample_from_logits(packed_logits.cpu(), torch.zeros(len(prompts)))

    all_match = True
    for i, idx in enumerate(SELECTED_INDICES):
        base_row = baseline_logits[i]
        packed_row = packed_logits[i]
        raw_argmax_base = int(base_row.argmax().item())
        raw_argmax_packed = int(packed_row.argmax().item())

        sampled_alone = llm.model_runner.sample_from_logits(base_row.unsqueeze(0).cpu(), torch.zeros(1))[0]
        sampled_stacked = sampled_stacked_all[i]

        match = len({raw_argmax_base, raw_argmax_packed, sampled_alone, sampled_stacked}) == 1
        if not match:
            all_match = False
        print(f"  {i:>3}  {idx:>9}  {raw_argmax_base:>15}  {raw_argmax_packed:>17}  "
              f"{sampled_alone:>13}  {sampled_stacked:>15}  {str(match):>9}")

    print()
    if all_match:
        print("RESULT: PASS -- sampler output is IDENTICAL regardless of batch-1-alone vs "
              "batch-8-stacked, for all 8 rows, and matches raw argmax throughout. The sampler "
              "is NOT the cause of gsm8k_decode_contamination_check.py's divergence.")
    else:
        print("RESULT: FAIL -- the sampler produced a DIFFERENT token for at least one row "
              "depending on batch stacking, even though the underlying logits were already "
              "confirmed consistent. This is a real, checkpoint-independent bug in "
              "layers/sampler.py's @torch.compile-decorated Sampler.forward -- likely explains "
              "gsm8k_decode_contamination_check.py's divergence.")


if __name__ == "__main__":
    main()
