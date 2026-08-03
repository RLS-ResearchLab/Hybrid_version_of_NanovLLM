"""Prefill packed-contamination check at REAL GSM8K scale -- built directly
on tests/phase6_packed_contamination_check.py's already-validated
methodology (same get_prefill_logits diagnostic, same forced
num_cached_blocks=0 allocation to bypass prefix-cache reuse entirely, same
cosine>=0.99 + argmax threshold), but with prompts that check never
exercised: phase6_packed_contamination_check.py's 8 prompts were all short
(<50 tokens) and mutually prefix-UNRELATED. The real 8-shot GSM8K prompts
this eval actually uses are 700-830 tokens each AND all 8 share the same
~500-700 token exemplar preamble verbatim (gsm8k_prompt.py's EXEMPLARS).

Why this matters: tests/gsm8k_decode_contamination_check.py (separate
baseline/packed PROCESSES, so no caching confound) found real divergence
in each target prompt's FIRST generated token. engine/scheduler.py's
postprocess() shows that first token is sampled from PREFILL logits, not a
decode step -- so that divergence could be a packed-PREFILL bug specific
to long, shared-prefix prompts (never validated at this scale), not a
decode-EP bug at all. This script isolates exactly that question, cheaply,
by reusing the existing get_prefill_logits diagnostic instead of building
new decode instrumentation.

Usage:
    python tests/gsm8k_prefill_scale_check.py
"""
import os
import sys
import types

import torch
import torch.nn.functional as F

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
COSINE_SIM_THRESHOLD = 0.99  # same threshold as reference_check_phase6.py / phase6_packed_contamination_check.py
TARGET_INDICES = [0, 1, 2]
FILLER_INDICES = [3, 4, 5, 6, 7]
SELECTED_INDICES = TARGET_INDICES + FILLER_INDICES  # 8 total, concurrency=8


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item()


def _new_seq(llm, prompt: str):
    from nanovllm.engine.sequence import Sequence

    prompt_ids = llm.tokenizer.encode(prompt)
    seq = Sequence(prompt_ids)
    seq.num_scheduled_tokens = len(prompt_ids)
    # Same forced num_cached_blocks=0 allocation phase6_packed_contamination_check.py
    # uses -- bypasses the prefix cache entirely (every call here is a genuinely
    # fresh, uncached prefill), so this test is ISOLATED to packed-prefill
    # computation itself, not confounded by cache reuse (unlike gsm8k_decode_
    # contamination_check.py's first, single-process version).
    llm.scheduler.block_manager.allocate(seq, 0)
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
    print(f"Prompts under test ({len(prompts)}), all sharing the same 8-shot exemplar preamble:")
    for idx, p in zip(SELECTED_INDICES, prompts):
        print(f"  gsm8k idx={idx}: {len(p)} chars")

    llm = LLM(
        CKPT_DIR,
        enforce_eager=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_num_seqs=2 * len(prompts),  # baseline slots + packed slots, no reuse needed
        max_model_len=MAX_MODEL_LEN,
    )

    print("\n" + "=" * 78)
    print("BASELINE -- each prompt run single-sequence (its own dedicated slot, forced-uncached)")
    print("=" * 78)
    baseline_logits = []
    for idx, prompt in zip(SELECTED_INDICES, prompts):
        seq, prompt_ids = _new_seq(llm, prompt)
        logits = llm.model_runner.call("get_prefill_logits", [seq])
        assert logits is not None, "expected rank0 to return gathered logits"
        if logits.dim() == 2 and logits.shape[0] == 1:
            logits = logits.squeeze(0)
        print(f"  gsm8k idx={idx}: {len(prompt_ids)} tokens, logits shape {tuple(logits.shape)}")
        baseline_logits.append(logits.float())

    print("\n" + "=" * 78)
    print(f"PACKED -- all {len(prompts)} prompts in ONE batch (real prepare_prefill packing, forced-uncached)")
    print("=" * 78)
    packed_seqs = [_new_seq(llm, prompt)[0] for prompt in prompts]
    packed_logits = llm.model_runner.call("get_prefill_logits", packed_seqs)
    assert packed_logits is not None, "expected rank0 to return gathered logits"
    print(f"  packed logits shape: {tuple(packed_logits.shape)}  (expect ({len(prompts)}, vocab))")
    assert packed_logits.shape[0] == len(prompts), f"expected one row per prompt, got {packed_logits.shape}"

    print("\n" + "=" * 78)
    print(f"COMPARISON -- threshold stated before running: cosine >= {COSINE_SIM_THRESHOLD} AND argmax match")
    print("=" * 78)
    print(f"  {'#':>3}  {'gsm8k_idx':>9}  {'cosine':>9}  {'argmax':>7}  {'verdict':>7}")
    print("  " + "-" * 50)

    results = []
    for i, idx in enumerate(SELECTED_INDICES):
        b = baseline_logits[i]
        p = packed_logits[i].float()
        cos = cosine(b, p)
        argmax_match = int(b.argmax().item()) == int(p.argmax().item())
        passed = cos >= COSINE_SIM_THRESHOLD and argmax_match
        results.append((idx, cos, argmax_match, passed))
        verdict = "PASS" if passed else "FAIL"
        print(f"  {i:>3}  {idx:>9}  {cos:>9.6f}  {'match' if argmax_match else 'DIFF':>7}  {verdict:>7}")

    n_pass = sum(1 for r in results if r[3])
    n_total = len(results)
    print()
    print(f"OVERALL: {n_pass}/{n_total} prompts passed (cosine >= {COSINE_SIM_THRESHOLD} AND argmax match)")
    if n_pass == n_total:
        print("VERDICT: PASS -- packed PREFILL at real GSM8K scale (long, shared-prefix prompts) is clean. "
              "The decode-contamination-check divergence is NOT explained by prefill -- points back to "
              "the decode loop specifically.")
    else:
        target_results = [r for r in results if r[0] in TARGET_INDICES]
        print(f"VERDICT: FAIL -- {n_total - n_pass} prompt(s) degraded under packing at prefill:")
        for idx, cos, argmax_match, passed in results:
            if not passed:
                print(f"    gsm8k idx={idx}: cosine={cos:.6f}  argmax_match={argmax_match}")
        print("This would mean packed PREFILL at long/shared-prefix scale has a real bug -- likely the "
              "actual source of gsm8k_decode_contamination_check.py's divergence (that check's 'position 0' "
              "token comes from prefill sampling, per scheduler.postprocess()), not decode-EP.")

    print("\nDONE")


if __name__ == "__main__":
    main()
