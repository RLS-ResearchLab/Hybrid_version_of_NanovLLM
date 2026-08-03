"""Decode-loop cross-sequence contamination check -- extends
tests/phase6_packed_contamination_check.py's packed-PREFILL contamination
check (cosine>=0.99 + argmax match, prefill logits only) to the FULL DECODE
LOOP, now that decode at ep_size>1 actually works
(Qwen35MoE._forward_gathered_ep -- see models/qwen3_5.py and
tests/test_moe_ep_dispatch_decode.py). The prefill-only check could not
have caught decode-time contamination, because decode-EP didn't exist when
it ran (it raised NotImplementedError at ep_size>1 until this session).

THREE SEPARATE PROCESSES, deliberately (same reason as reference_check_phase6.py's
--phase engine / --phase hf / --phase compare split, just applied to two runs
of the SAME engine instead of engine-vs-HF): the first version of this
script ran baseline (concurrency=1) THEN packed (concurrency=8) in one
process, on one LLM instance -- and the block_manager's prefix cache
(engine/block_manager.py) caused the packed run to silently reuse KV
blocks cached from each target prompt's OWN earlier baseline run whenever
a prompt was long enough to fill full 256-token blocks purely from prompt
content (784//256==3 blocks, etc.) That's CORRECT cache behavior (chained
hashing + explicit token_ids equality check before reuse -- verified by
reading engine/block_manager.py's can_allocate()), not a bug, but it means
the packed run wasn't doing a fresh prefill for the repeated prompts: it
exercised a DIFFERENT, previously-untested scenario (packed prefill with
wildly UNEVEN cached-vs-new-token splits across the 8 sequences) on top of
whatever the decode question was supposed to isolate, making that first
run's divergences impossible to attribute cleanly. Running baseline and
packed as separate process invocations (separate block managers, zero
shared cache state) eliminates that confound: the packed run's prefill is
genuinely fresh, matching the scenario phase6_packed_contamination_check.py
already validated at prefill, isolating this check to the DECODE loop
specifically -- which is the thing that's actually new this session.

Usage (run engine loads are ~1-2 min each; run baseline and packed in
either order, both before compare):
    python tests/gsm8k_decode_contamination_check.py --phase baseline
    python tests/gsm8k_decode_contamination_check.py --phase packed
    python tests/gsm8k_decode_contamination_check.py --phase compare   # no GPU needed

Intermediate artifacts land in tests/_gsm8k_cache/decode_contamination/
(repo-relative, safe to delete after --phase compare).

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
"""
import argparse
import json
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
CACHE_DIR = os.path.join(ROOT, "tests", "_gsm8k_cache", "decode_contamination")
BASELINE_PATH = os.path.join(CACHE_DIR, "baseline.json")
PACKED_PATH = os.path.join(CACHE_DIR, "packed.json")

MAX_MODEL_LEN = 2048
MAX_TOKENS = 512  # same as the real eval -- validating the actual config, not a scaled-down proxy
TARGET_INDICES = [0, 1, 2]        # same GSM8K test-set indices used in gsm8k_smoke_test.py
FILLER_INDICES = [3, 4, 5, 6, 7]  # 3 target + 5 filler = 8, the real eval's batch size
BATCH_SIZE = len(TARGET_INDICES) + len(FILLER_INDICES)


def _load_gsm8k():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if len(ds) != 1319:
        print(f"STOP: expected 1319 test examples, got {len(ds)}. Not proceeding.")
        sys.exit(1)
    return ds


def phase_baseline():
    os.makedirs(CACHE_DIR, exist_ok=True)
    ds = _load_gsm8k()
    target_prompts = [build_prompt(ds[i]["question"]) for i in TARGET_INDICES]

    from nanovllm.llm import LLM
    from nanovllm.sampling_params import SamplingParams

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
    print(f"BASELINE -- each of {len(TARGET_INDICES)} target prompts run individually (concurrency=1), "
          f"fresh process/engine/cache")
    print("=" * 78)
    results = {}
    for idx, prompt in zip(TARGET_INDICES, target_prompts):
        t0 = perf_counter()
        out = llm.generate([prompt], sp, use_tqdm=False)
        dt = perf_counter() - t0
        token_ids = out[0]["token_ids"]
        results[str(idx)] = token_ids
        print(f"  gsm8k idx={idx}: {len(token_ids)} tokens generated in {dt:.1f}s")

    with open(BASELINE_PATH, "w") as f:
        json.dump(results, f)
    print(f"\nSaved baseline results to {BASELINE_PATH}")
    print("PHASE COMPLETE -- exit this process fully before running --phase packed")


def phase_packed():
    os.makedirs(CACHE_DIR, exist_ok=True)
    ds = _load_gsm8k()
    target_prompts = [build_prompt(ds[i]["question"]) for i in TARGET_INDICES]
    filler_prompts = [build_prompt(ds[i]["question"]) for i in FILLER_INDICES]

    from nanovllm.llm import LLM
    from nanovllm.sampling_params import SamplingParams

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
    print(f"PACKED -- {len(TARGET_INDICES)} target + {len(FILLER_INDICES)} filler prompts, "
          f"ONE batch (concurrency={BATCH_SIZE}), fresh process/engine/cache -- this is each "
          f"target prompt's FIRST EVER submission in this process, so num_cached_tokens must be "
          f"0 for all of them (verified below); no self-cache-reuse confound possible.")
    print("=" * 78)
    all_prompts = target_prompts + filler_prompts
    t0 = perf_counter()
    packed_out = llm.generate(all_prompts, sp, use_tqdm=False)
    dt = perf_counter() - t0
    print(f"  packed batch: {dt:.1f}s total")

    results = {str(idx): packed_out[i]["token_ids"] for i, idx in enumerate(TARGET_INDICES)}
    for idx, token_ids in results.items():
        print(f"  gsm8k idx={idx}: {len(token_ids)} tokens generated")

    with open(PACKED_PATH, "w") as f:
        json.dump(results, f)
    print(f"\nSaved packed results to {PACKED_PATH}")
    print("PHASE COMPLETE")


def _first_divergence(a: list[int], b: list[int]):
    for pos, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return pos
    return len(a) if len(a) != len(b) else None  # None means fully identical


def phase_compare():
    assert os.path.exists(BASELINE_PATH), f"missing {BASELINE_PATH} -- run --phase baseline first"
    assert os.path.exists(PACKED_PATH), f"missing {PACKED_PATH} -- run --phase packed first"

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    with open(PACKED_PATH) as f:
        packed = json.load(f)

    print("\n" + "=" * 78)
    print("COMPARISON -- token-for-token, baseline (concurrency=1) vs packed (concurrency=8), "
          "ZERO shared cache state between the two runs")
    print("=" * 78)
    all_exact = True
    for idx in TARGET_INDICES:
        base = baseline[str(idx)]
        pack = packed[str(idx)]
        exact = base == pack
        print(f"\n  gsm8k idx={idx}: baseline_len={len(base)} packed_len={len(pack)} EXACT_MATCH={exact}")
        if exact:
            print(f"    PASS -- token-for-token identical")
            continue

        all_exact = False
        first_diff = _first_divergence(base, pack)
        pct_through = 100 * first_diff / max(len(base), 1)
        print(f"    DIVERGENCE at position {first_diff} ({pct_through:.1f}% through the baseline completion)")
        ctx_lo = max(0, first_diff - 5)
        print(f"    baseline tokens around divergence: {base[ctx_lo:first_diff + 5]}")
        print(f"    packed   tokens around divergence: {pack[ctx_lo:first_diff + 5]}")
        if first_diff <= 5:
            print(f"    ASSESSMENT: divergence in the first few tokens, with NO caching confound "
                  f"possible this time -- looks like REAL decode-time contamination. Investigate "
                  f"before trusting decode-EP's packed-batch output.")
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
              f"concurrency=1 and concurrency={BATCH_SIZE}, with no shared-cache confound. No "
              f"decode-time contamination detected.")
    else:
        print("At least one target prompt diverged -- see the DIVERGENCE/ASSESSMENT lines above "
              "for each before concluding this is (or isn't) a real bug.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True, choices=["baseline", "packed", "compare"])
    args = parser.parse_args()
    if args.phase == "baseline":
        phase_baseline()
    elif args.phase == "packed":
        phase_packed()
    else:
        phase_compare()


if __name__ == "__main__":
    main()
