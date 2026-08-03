"""Decode-loop cross-sequence contamination check -- extends
tests/phase6_packed_contamination_check.py's packed-PREFILL contamination
check (cosine>=0.99 + argmax match, prefill logits only) to the FULL DECODE
LOOP, now that decode at ep_size>1 actually works
(Qwen35MoE._forward_gathered_ep -- see models/qwen3_5.py and
tests/test_moe_ep_dispatch_decode.py). The prefill-only check could not
have caught decode-time contamination, because decode-EP didn't exist when
it ran (it raised NotImplementedError at ep_size>1 until this session).

Design: 3 real GSM8K-style prompts (built via gsm8k_prompt.build_prompt --
the actual 8-shot CoT prompts the real eval uses, not toy prompts, since
packing behavior can depend on sequence length/content) are run
individually (concurrency=1) to get a baseline completion for each. Then
the SAME 3 prompts are run together with 5 filler prompts (also real GSM8K
questions) in ONE generate() call to reach concurrency=8 -- the real
eval's batch size -- and each target prompt's completion at concurrency=8
is compared against its own concurrency=1 baseline, TOKEN FOR TOKEN.

CAVEAT -- stated here, before running, not after seeing the result: exact
token-for-token equality across up to 512 autoregressive decode steps is a
STRONGER bar than the prefill contamination check used (cosine>=0.99 +
argmax match, not bitwise equality), and for a specific reason. The
prefill check already MEASURED cosine 0.9976-0.9997 (not 1.0) between
packed and solo prefill logits on this exact checkpoint: batched
matmul/layernorm reductions are not bitwise-associative, so some numeric
noise between concurrency=1 and concurrency=8 is EXPECTED even from a
genuinely correct implementation. At decode, each step's sampled token
(temperature=0 argmax) feeds back as the next step's input, so a tiny
logit perturbation at any single step can flip an argmax and send the two
trajectories on completely different paths from that point on -- a real
property of chaotic autoregressive sampling, not automatically a
contamination bug. So a divergence here is diagnostic, not an automatic
FAIL: what matters is WHERE it first happens. A divergence in the first
few tokens (or text that goes immediately incoherent) looks like real
contamination; a late, single-position divergence after many tokens of
exact agreement is consistent with the same sub-0.3%-magnitude numeric
noise the prefill check already measured and accepted, now compounding
into an argmax flip at one specific step. This script has no access to
per-step decode logits (this engine has no get_decode_logits diagnostic,
unlike get_prefill_logits) to fully disambiguate the two cases numerically
-- only the position and content of any divergence is reported.

Usage:
    python tests/gsm8k_decode_contamination_check.py
"""
import os
import sys
import types
from time import perf_counter

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
MAX_TOKENS = 512  # same as the real eval -- validating the actual config, not a scaled-down proxy
TARGET_INDICES = [0, 1, 2]        # same GSM8K test-set indices used in gsm8k_smoke_test.py
FILLER_INDICES = [3, 4, 5, 6, 7]  # 3 target + 5 filler = 8, the real eval's batch size
BATCH_SIZE = len(TARGET_INDICES) + len(FILLER_INDICES)


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    for pos, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return pos
    return len(a) if len(a) != len(b) else None  # None means fully identical


def main():
    from datasets import load_dataset
    from nanovllm.llm import LLM
    from nanovllm.sampling_params import SamplingParams

    print("Loading openai/gsm8k (main, test split) ...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded {len(ds)} examples (expect 1319).")
    if len(ds) != 1319:
        print(f"STOP: expected 1319 test examples, got {len(ds)}. Not proceeding.")
        sys.exit(1)

    target_prompts = [build_prompt(ds[i]["question"]) for i in TARGET_INDICES]
    filler_prompts = [build_prompt(ds[i]["question"]) for i in FILLER_INDICES]

    print(f"Loading engine from {CKPT_DIR} (tensor_parallel_size=2) ...")
    llm = LLM(
        CKPT_DIR,
        enforce_eager=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_num_seqs=BATCH_SIZE,
        max_model_len=MAX_MODEL_LEN,
    )
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

    print("\n" + "=" * 78)
    print(f"BASELINE -- each of {len(TARGET_INDICES)} target prompts run individually (concurrency=1)")
    print("=" * 78)
    baseline_token_ids = []
    for i, idx in enumerate(TARGET_INDICES):
        t0 = perf_counter()
        out = llm.generate([target_prompts[i]], sp, use_tqdm=False)
        dt = perf_counter() - t0
        token_ids = out[0]["token_ids"]
        baseline_token_ids.append(token_ids)
        print(f"  target #{i} (gsm8k idx={idx}): {len(token_ids)} tokens generated in {dt:.1f}s")

    print("\n" + "=" * 78)
    print(f"PACKED -- {len(TARGET_INDICES)} target + {len(FILLER_INDICES)} filler prompts, "
          f"ONE batch (concurrency={BATCH_SIZE})")
    print("=" * 78)
    all_prompts = target_prompts + filler_prompts
    t0 = perf_counter()
    packed_out = llm.generate(all_prompts, sp, use_tqdm=False)
    dt = perf_counter() - t0
    print(f"  packed batch: {dt:.1f}s total")
    packed_token_ids = [packed_out[i]["token_ids"] for i in range(len(TARGET_INDICES))]

    print("\n" + "=" * 78)
    print("COMPARISON -- token-for-token, baseline (concurrency=1) vs packed (concurrency=8)")
    print("=" * 78)
    all_exact = True
    for i, idx in enumerate(TARGET_INDICES):
        base = baseline_token_ids[i]
        packed = packed_token_ids[i]
        exact = base == packed
        print(f"\n  target #{i} (gsm8k idx={idx}): baseline_len={len(base)} packed_len={len(packed)} "
              f"EXACT_MATCH={exact}")
        if exact:
            print(f"    PASS -- token-for-token identical")
            continue

        all_exact = False
        first_diff = _first_divergence(base, packed)
        pct_through = 100 * first_diff / max(len(base), 1)
        print(f"    DIVERGENCE at position {first_diff} ({pct_through:.1f}% through the baseline completion)")
        ctx_lo = max(0, first_diff - 5)
        print(f"    baseline tokens around divergence: {base[ctx_lo:first_diff + 5]}")
        print(f"    packed   tokens around divergence: {packed[ctx_lo:first_diff + 5]}")
        print(f"    baseline text around divergence: {llm.tokenizer.decode(base[ctx_lo:first_diff + 5])!r}")
        print(f"    packed   text around divergence: {llm.tokenizer.decode(packed[ctx_lo:first_diff + 5])!r}")
        if first_diff <= 5:
            print(f"    ASSESSMENT: divergence in the first few tokens -- looks like REAL "
                  f"contamination, not benign fp reassociation. Investigate before trusting "
                  f"decode-EP's packed-batch output.")
        else:
            print(f"    ASSESSMENT: divergence after {first_diff} tokens of exact agreement -- "
                  f"consistent with the same magnitude of batched-vs-solo numeric noise the "
                  f"prefill contamination check already measured (cosine 0.9976-0.9997, not "
                  f"1.0) compounding into an argmax flip at this one step. Not conclusive proof "
                  f"of a code bug by itself -- would need per-step decode logits (not currently "
                  f"instrumented in this engine) to fully rule out a real contamination bug.")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if all_exact:
        print(f"ALL {len(TARGET_INDICES)} target prompts: EXACT token-for-token match between "
              f"concurrency=1 and concurrency={BATCH_SIZE}. No decode-time contamination detected.")
    else:
        print("At least one target prompt diverged -- see the DIVERGENCE/ASSESSMENT lines above "
              "for each before concluding this is (or isn't) a real bug.")


if __name__ == "__main__":
    main()
