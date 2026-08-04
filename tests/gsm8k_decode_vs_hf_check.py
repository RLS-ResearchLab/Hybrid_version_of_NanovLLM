"""Check 1 -- decode-EP vs. HF, independent ground truth. Everything validated
about decode-EP so far (tests/gsm8k_decode_contamination_check.py) is
INTERNAL consistency: concurrency=1 vs concurrency=8 on this engine, both
runs going through the same Qwen35MoE._forward_gathered_ep code path (models/qwen3_5.py).
Two runs of the same possibly-buggy math agreeing with each other proves
determinism, not correctness. reference_check_phase6.py is closer to
independent ground truth but only checks ONE prefill step's logits, never
the full autoregressive decode loop where decode-EP actually executes.

This script closes that gap: real multi-token greedy generation from this
engine (tensor_parallel_size=2 -> ep_size=2, decode-EP active) compared
token-for-token against the dense HF reference implementation's own
greedy generate(), same checkpoint, same prompts, same temperature=0.

Deliberately small: 10 prompts (GSM8K test-set indices 0-9; the original
version of this script used just the first 3, same indices as
gsm8k_decode_contamination_check.py -- extended once the 3-prompt run
surfaced an idx=2 divergence at an empty-<think></think> "skip reasoning"
branch point, which is a sample size of 1 and can't distinguish "noise
landed on a near-tied fork once" from "the engine has a systematic
tendency to skip reasoning that HF doesn't." See the THINK-BRANCH RATES
section of --phase compare's output for that specific question -- it
matters beyond decode-EP correctness because CoT-skipping directly and
predictably hurts multi-step arithmetic accuracy, so a real asymmetry
here would be a systematic accuracy tax, not symmetric numeric noise),
30 tokens each -- enough autoregressive steps to see whether decode-EP's
math tracks an independent implementation, cheap enough to iterate on if
it doesn't. This is NOT trying to reach "The answer is N." (gsm8k_dry_run.py's
docstring puts that around 100-200 tokens); it's isolating the math, not
scoring accuracy.

PASS/FAIL BAR -- stated here, before running, not after seeing the number,
for the same reason reference_check_phase6.py states its cosine threshold
up front: this compares a genuinely different code path (EP+TP engine) to
a dense reference over up to 30 autoregressive steps, and
gsm8k_decode_contamination_check.py already established that even a
CORRECT implementation is not expected to be bitwise-identical across
concurrency levels on this checkpoint (prefill cosine 0.9976-0.9997, not
1.0 -- and at decode, temperature=0 argmax feeds each step's token back in,
so a single tiny logit perturbation can flip an argmax and send the two
trajectories down completely different paths from that point on). So:
  - First-token argmax MUST match for all 3 prompts. This is the one
    position where "different code path, same math" has no chaotic
    amplification to hide behind -- prefill logits already measured
    cosine>=0.99 in reference_check_phase6.py, so a first-token mismatch
    here would be new evidence of a decode-EP-specific bug, not noise.
  - Beyond token 0, a divergence is diagnostic, not an automatic FAIL, per
    the same logic gsm8k_decode_contamination_check.py already applied:
    an EARLY divergence (first few tokens) or text that goes immediately
    incoherent looks like a real bug; a LATE divergence after many tokens
    of exact agreement is consistent with expected cross-implementation
    numeric noise compounding into one argmax flip. Both engine and HF
    text are printed in full so this can be judged by eye, not just by
    position.

GPU MEMORY -- same reason as reference_check_phase6.py: this engine
(~42 GB/rank x2) and the dense HF reference (~70 GB) cannot both fit on
the same GPUs at once. Three separate process invocations, each exiting
fully before the next starts:

    python tests/gsm8k_decode_vs_hf_check.py --phase engine
    python tests/gsm8k_decode_vs_hf_check.py --phase hf
    python tests/gsm8k_decode_vs_hf_check.py --phase compare   # no GPU needed

Intermediate artifacts land in tests/_gsm8k_cache/decode_vs_hf/
(repo-relative, safe to delete after --phase compare).
"""
import argparse
import json
import os
import sys
import types

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
CACHE_DIR = os.path.join(ROOT, "tests", "_gsm8k_cache", "decode_vs_hf")
ENGINE_PATH = os.path.join(CACHE_DIR, "engine.json")
HF_PATH = os.path.join(CACHE_DIR, "hf.json")

MAX_MODEL_LEN = 2048
MAX_TOKENS = 30
# Extended from the original [0, 1, 2] (same GSM8K test-set indices as
# gsm8k_decode_contamination_check.py) to 10 -- one prompt's empty-<think>
# divergence (idx=2) is a sample size of 1, not enough to tell "engine has a
# systematic tendency to skip reasoning" apart from "noise landed on a
# near-tied fork once." 10 is still small but cheap (single-prompt calls,
# 30 tokens each) and gives an actual rate to compare, not an anecdote.
TARGET_INDICES = list(range(10))


def _load_gsm8k():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if len(ds) != 1319:
        print(f"STOP: expected 1319 test examples, got {len(ds)}. Not proceeding.")
        sys.exit(1)
    return ds


def phase_engine():
    os.makedirs(CACHE_DIR, exist_ok=True)
    ds = _load_gsm8k()
    prompts = [build_prompt(ds[i]["question"]) for i in TARGET_INDICES]

    from nanovllm.llm import LLM
    from nanovllm.sampling_params import SamplingParams

    print(f"Loading engine from {CKPT_DIR} (tensor_parallel_size=2, ep_size=2, decode-EP active) ...")
    llm = LLM(
        CKPT_DIR,
        enforce_eager=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_num_seqs=len(prompts),
        max_model_len=MAX_MODEL_LEN,
    )
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

    results = {}
    for idx, prompt in zip(TARGET_INDICES, prompts):
        prompt_ids = llm.tokenizer.encode(prompt)
        out = llm.generate([prompt], sp, use_tqdm=False)
        token_ids = out[0]["token_ids"]
        print(f"  gsm8k idx={idx}: prompt_tokens={len(prompt_ids)}  generated={len(token_ids)} tokens")
        results[str(idx)] = {"prompt_ids": prompt_ids, "token_ids": token_ids, "text": out[0]["text"]}

    with open(ENGINE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"\nSaved engine results to {ENGINE_PATH}")
    print("PHASE COMPLETE -- exit this process fully before running --phase hf")


def phase_hf():
    os.makedirs(CACHE_DIR, exist_ok=True)
    ds = _load_gsm8k()
    prompts = [build_prompt(ds[i]["question"]) for i in TARGET_INDICES]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        CKPT_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    results = {}
    for idx, prompt in zip(TARGET_INDICES, prompts):
        prompt_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(model.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        token_ids = out[0, input_ids.shape[1]:].tolist()
        text = tokenizer.decode(token_ids)
        print(f"  gsm8k idx={idx}: prompt_tokens={len(prompt_ids)}  generated={len(token_ids)} tokens")
        results[str(idx)] = {"prompt_ids": prompt_ids, "token_ids": token_ids, "text": text}

    with open(HF_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"\nSaved HF results to {HF_PATH}")
    print("PHASE COMPLETE")


def _first_divergence(a: list[int], b: list[int]):
    for pos, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return pos
    return len(a) if len(a) != len(b) else None  # None means fully identical (over the shared length)


def _classify_think(text: str) -> str:
    """Text-based, not token-ID-based (robust to exact tokenization of the
    <think>/</think> tags, which this script doesn't otherwise need to know).

    - "no_think_tag": <think> never appears in this budget at all.
    - "still_reasoning": <think> opened but not yet closed within MAX_TOKENS
      -- the common case for the verbose branch (its own boilerplate hadn't
      closed by token 30 in the original 3-prompt run either); NOT the same
      as "empty_think", just unresolved at this budget.
    - "empty_think": <think> immediately closed with only whitespace inside
      -- the "skip reasoning" branch idx=2 took.
    - "closed_with_content": closed within budget but had real content --
      rare at MAX_TOKENS=30 for genuine CoT, called out separately rather
      than lumped into either bucket above.
    """
    open_tag, close_tag = "<think>", "</think>"
    i = text.find(open_tag)
    if i == -1:
        return "no_think_tag"
    j = text.find(close_tag, i + len(open_tag))
    if j == -1:
        return "still_reasoning"
    inner = text[i + len(open_tag):j]
    return "empty_think" if inner.strip() == "" else "closed_with_content"


def phase_compare():
    assert os.path.exists(ENGINE_PATH), f"missing {ENGINE_PATH} -- run --phase engine first"
    assert os.path.exists(HF_PATH), f"missing {HF_PATH} -- run --phase hf first"

    with open(ENGINE_PATH, encoding="utf-8") as f:
        engine_results = json.load(f)
    with open(HF_PATH, encoding="utf-8") as f:
        hf_results = json.load(f)

    print("\n" + "=" * 78)
    print("COMPARISON -- engine (tensor_parallel_size=2, decode-EP) vs. dense HF reference, "
          "same checkpoint, same prompts, temperature=0")
    print("=" * 78)

    all_first_token_match = True
    engine_branch_counts: dict[str, int] = {}
    hf_branch_counts: dict[str, int] = {}
    for idx in TARGET_INDICES:
        er = engine_results[str(idx)]
        hr = hf_results[str(idx)]
        ids_match = er["prompt_ids"] == hr["prompt_ids"]
        engine_toks = er["token_ids"]
        hf_toks = hr["token_ids"]

        engine_branch = _classify_think(er["text"])
        hf_branch = _classify_think(hr["text"])
        engine_branch_counts[engine_branch] = engine_branch_counts.get(engine_branch, 0) + 1
        hf_branch_counts[hf_branch] = hf_branch_counts.get(hf_branch, 0) + 1

        print(f"\n  gsm8k idx={idx}  (prompt_ids match: {ids_match})")
        if not ids_match:
            print(f"    WARNING: tokenization mismatch between engine and HF for this prompt -- "
                  f"the comparison below is confounded by that, not a clean signal.")
        print(f"    think-branch: engine={engine_branch}  hf={hf_branch}"
              + ("  <-- DIFFERS" if engine_branch != hf_branch else ""))

        exact = engine_toks == hf_toks
        first_diff = _first_divergence(engine_toks, hf_toks)
        first_token_match = len(engine_toks) > 0 and len(hf_toks) > 0 and engine_toks[0] == hf_toks[0]
        all_first_token_match = all_first_token_match and first_token_match

        print(f"    engine_len={len(engine_toks)}  hf_len={len(hf_toks)}  EXACT_MATCH={exact}  "
              f"FIRST_TOKEN_MATCH={first_token_match}")
        print(f"    engine text: {er['text']!r}")
        print(f"    hf     text: {hr['text']!r}")

        if exact:
            print(f"    PASS -- token-for-token identical over {len(engine_toks)} decode steps")
            continue

        pct_through = 100 * first_diff / max(len(engine_toks), 1)
        print(f"    DIVERGENCE at position {first_diff} ({pct_through:.1f}% through)")
        ctx_lo = max(0, first_diff - 5)
        print(f"    engine tokens around divergence: {engine_toks[ctx_lo:first_diff + 5]}")
        print(f"    hf     tokens around divergence: {hf_toks[ctx_lo:first_diff + 5]}")
        if first_diff == 0:
            print(f"    ASSESSMENT: divergence at position 0 -- this is the FIRST generated "
                  f"token, sampled directly from PREFILL logits (see [PREFILL-SAMPLE DEBUG] for "
                  f"this seq_id in the --phase engine log), before decode-EP's own forward path "
                  f"(Qwen35MoE._forward_gathered_ep) ever runs. Implicates PREFILL numerics on "
                  f"this ~750-800 token GSM8K-formatted prompt specifically, NOT decode-EP -- "
                  f"reference_check_phase6.py's cosine>=0.99 result never tested a prompt this "
                  f"long/this shape, only 5-11 token prompts. Worth a direct prefill-logits "
                  f"cosine measurement on this prompt before concluding either way.")
        elif first_diff <= 5:
            print(f"    ASSESSMENT: divergence in the first few DECODE steps (position {first_diff} "
                  f">= 1, so at least one step has already gone through decode-EP's own forward "
                  f"path) against an INDEPENDENT implementation -- looks like a real decode-EP "
                  f"correctness bug. Investigate before trusting decode-EP's output.")
        else:
            print(f"    ASSESSMENT: divergence after {first_diff} tokens of exact agreement -- "
                  f"consistent with expected cross-implementation numeric noise (see this "
                  f"script's docstring) compounding into an argmax flip at one step. Read the "
                  f"two texts above by eye: if both remain coherent and mutually plausible past "
                  f"the divergence, this is not conclusive proof of a bug by itself.")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"THRESHOLD (stated before running): first-token argmax must match for all "
          f"{len(TARGET_INDICES)} prompts.")
    if all_first_token_match:
        print("VERDICT: PASS on the stated bar -- first-token argmax agrees with independent HF "
              "ground truth for every prompt. Any later divergence is diagnostic only; see the "
              "per-prompt ASSESSMENT lines above.")
    else:
        print("VERDICT: FAIL on the stated bar -- at least one prompt's first decode token "
              "disagrees with independent HF ground truth. CORRECTION vs. this script's original "
              "reasoning: the first generated token is sampled directly from PREFILL logits (see "
              "the [PREFILL-SAMPLE DEBUG] line for that seq_id in the --phase engine log) -- "
              "before Qwen35MoE._forward_gathered_ep (decode-EP) ever runs -- so this implicates "
              "PREFILL numerics specifically, NOT decode-EP. It also doesn't contradict "
              "reference_check_phase6.py's cosine>=0.99 result: that script only tested 5 short "
              "(5-11 token) prompts, never a realistic ~750-800 token GSM8K-formatted prompt like "
              "the ones here. A first-token mismatch on THIS prompt shape is new evidence worth "
              "following up with an actual prefill logits cosine-similarity measurement on the "
              "specific failing prompt(s), not assumed to be the same already-validated case.")

    print("\n" + "-" * 78)
    print(f"THINK-BRANCH RATES (out of {len(TARGET_INDICES)} prompts) -- does the engine take the "
          f"empty-<think></think> (skip-reasoning) branch at a different rate than independent HF?")
    print("-" * 78)
    engine_empty = engine_branch_counts.get("empty_think", 0)
    hf_empty = hf_branch_counts.get("empty_think", 0)
    print(f"  engine: {dict(sorted(engine_branch_counts.items()))}")
    print(f"  hf:     {dict(sorted(hf_branch_counts.items()))}")
    print(f"  empty_think count -- engine: {engine_empty}/{len(TARGET_INDICES)}   "
          f"hf: {hf_empty}/{len(TARGET_INDICES)}")
    gap = abs(engine_empty - hf_empty)
    if gap <= 1:
        print(f"  READ: engine and HF take the empty-think branch at essentially the same rate "
              f"(gap={gap}) -- consistent with the idx=2 divergence being noise landing on a "
              f"near-tied fork, as originally concluded, not a systematic engine tendency.")
    else:
        print(f"  READ: engine and HF diverge by {gap} on this small sample -- worth extending "
              f"further (more prompts, or a larger think-budget past MAX_TOKENS={MAX_TOKENS} to "
              f"resolve any 'still_reasoning' cases either way) before concluding this is noise. "
              f"A CONSISTENT engine-side tilt toward skipping reasoning is the concerning case: "
              f"CoT-skipping directly and predictably hurts multi-step arithmetic accuracy, so "
              f"this would be a real, systematic (if small) accuracy tax specific to decode-EP, "
              f"not symmetric numeric noise -- and would help explain the gsm8k_full_run.py n=32 "
              f"accuracy softness independent of Bug B's prefix-cache-disable timing cost.")
        print(f"  NOTE: n={len(TARGET_INDICES)} is still small -- a {gap}-prompt gap is NOT yet "
              f"strong statistical evidence of a systematic tendency by itself, just a reason to "
              f"look closer rather than dismiss it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True, choices=["engine", "hf", "compare"])
    args = parser.parse_args()
    if args.phase == "engine":
        phase_engine()
    elif args.phase == "hf":
        phase_hf()
    else:
        phase_compare()


if __name__ == "__main__":
    main()
