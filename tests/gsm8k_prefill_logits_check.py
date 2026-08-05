"""Follow-up on Check 1's idx=9 finding: gsm8k_decode_vs_hf_check.py found
the engine's and HF's FIRST generated token disagree for one of 10 GSM8K
prompts (idx=9 -- GSM8K test-set example 9), the only one of the 10 to fail
that check's pre-registered first-token bar. That first token is sampled
directly from PREFILL logits (see [PREFILL-SAMPLE DEBUG] in that script's
engine-phase log), before decode-EP's own forward path
(Qwen35MoE._forward_gathered_ep) ever runs -- so it implicates PREFILL
numerics specifically, not decode-EP.

reference_check_phase6.py already validated this engine's raw prefill
logits against HF (cosine>=0.99 + argmax match) -- but only on 5 SHORT
prompts (5-11 tokens each; "The capital of France is", "2 + 2 =", etc.).
It has never been checked on a REALISTIC ~750-830 token GSM8K-formatted
prompt (the 8-shot exemplar preamble dominates the length). idx=9's
mismatch could be the same already-accepted cross-implementation numeric
noise (prefill cosine 0.9976-0.9997, not 1.0, already measured on short
prompts) just landing on a near-tied argmax this one time -- or it could
mean long/realistic prompts are a genuinely different, worse regime for
this checkpoint's EP-dispatch numerics. This script is the direct way to
tell those apart: same methodology as reference_check_phase6.py, same
threshold, but GSM8K prompts 0-9 (gsm8k_prompt.build_prompt against real
GSM8K test-set questions) instead of short synthetic ones -- INCLUDING
idx=9 itself, so this directly re-tests the specific case that failed.

PASS/FAIL THRESHOLD -- stated here, before running, not after seeing the
number, same bar reference_check_phase6.py already established: cosine
similarity >= 0.99 AND top-1 (argmax) token match, both required, per
prompt.

GPU MEMORY -- same reason as reference_check_phase6.py: this engine and
the dense HF reference cannot both fit on the same GPUs at once. Three
separate process invocations, each exiting fully before the next starts:

    python tests/gsm8k_prefill_logits_check.py --phase engine
    python tests/gsm8k_prefill_logits_check.py --phase hf
    python tests/gsm8k_prefill_logits_check.py --phase compare   # no GPU needed

Intermediate artifacts land in tests/_gsm8k_cache/prefill_logits/
(repo-relative, safe to delete after --phase compare).
"""
import argparse
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
CACHE_DIR = os.path.join(ROOT, "tests", "_gsm8k_cache", "prefill_logits")
ENGINE_LOGITS_PATH = os.path.join(CACHE_DIR, "engine_logits.pt")
HF_LOGITS_PATH = os.path.join(CACHE_DIR, "hf_logits.pt")
MAX_MODEL_LEN = 2048

# Same GSM8K test-set indices as gsm8k_decode_vs_hf_check.py's Check 1 (0-9)
# -- deliberately, so idx=9 here is the EXACT SAME real question that failed
# the first-token bar there, not a different prompt that merely shares a
# number.
TARGET_INDICES = list(range(10))
DIVERGENT_IDX = 9  # the specific idx Check 1 flagged -- called out below, not just listed

# Stated before running, not after seeing the number -- same bar
# reference_check_phase6.py already established on short prompts.
COSINE_SIM_THRESHOLD = 0.99


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
    from nanovllm.engine.sequence import Sequence

    llm = LLM(
        CKPT_DIR,
        enforce_eager=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_num_seqs=len(prompts),
        max_model_len=MAX_MODEL_LEN,
    )

    results = []
    for idx, prompt in zip(TARGET_INDICES, prompts):
        prompt_ids = llm.tokenizer.encode(prompt)
        print(f"[engine] idx={idx}  prompt_tokens={len(prompt_ids)}")

        seq = Sequence(prompt_ids)
        seq.num_scheduled_tokens = len(prompt_ids)
        # Same reasoning as reference_check_phase6.py's identical lines --
        # prepare_prefill() treats an empty block_table as the warmup case
        # and skips slot_mapping entirely, which then fails store_kvcache's
        # assertion for a real sequence. Allocate real KV-cache blocks the
        # same way Scheduler.schedule() does for a fresh, uncached sequence.
        llm.scheduler.block_manager.allocate(seq, 0)
        if llm.model_runner.state_manager is not None:
            # Dispatch through .call(...), NOT state_manager.allocate(seq)
            # directly -- at tensor_parallel_size>1 each rank is a SEPARATE
            # PROCESS with its OWN independent StateManager (its own
            # warmup-contaminated slot 0), and calling .allocate() directly
            # on this process's local model_runner only zeros rank0's copy.
            # See reference_check_phase6.py's identical comment for the full
            # writeup -- confirmed empirically there, not re-derived here.
            llm.model_runner.call("allocate_state_slot", seq)

        logits = llm.model_runner.call("get_prefill_logits", [seq])
        assert logits is not None, "expected rank0 to return gathered logits"
        results.append({"idx": idx, "prompt_ids": prompt_ids, "logits": logits})

    torch.save({"results": results}, ENGINE_LOGITS_PATH)
    print(f"\n[engine] saved {len(results)} prompt(s) to {ENGINE_LOGITS_PATH}")
    print("[engine] PHASE COMPLETE -- exit this process fully before running --phase hf")


def phase_hf():
    os.makedirs(CACHE_DIR, exist_ok=True)
    ds = _load_gsm8k()
    prompts = [build_prompt(ds[i]["question"]) for i in TARGET_INDICES]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        CKPT_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    results = []
    for idx, prompt in zip(TARGET_INDICES, prompts):
        prompt_ids = tokenizer.encode(prompt)
        print(f"[hf] idx={idx}  prompt_tokens={len(prompt_ids)}")

        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(model.device)
        with torch.no_grad():
            out = model(input_ids=input_ids)

        logits_all_positions = out.logits  # (1, seq_len, vocab)
        logits = logits_all_positions[0, -1, :].float().cpu()
        results.append({"idx": idx, "prompt_ids": prompt_ids, "logits": logits})

    torch.save({"results": results}, HF_LOGITS_PATH)
    print(f"\n[hf] saved {len(results)} prompt(s) to {HF_LOGITS_PATH}")
    print("[hf] PHASE COMPLETE")


def _compare_one(idx: int, engine_result: dict, hf_result: dict) -> dict:
    engine_ids = engine_result["prompt_ids"]
    hf_ids = hf_result["prompt_ids"]
    ids_match = engine_ids == hf_ids

    engine_logits = engine_result["logits"].float()
    hf_logits = hf_result["logits"].float()
    if engine_logits.dim() == 2 and engine_logits.shape[0] == 1:
        engine_logits = engine_logits.squeeze(0)

    if engine_logits.shape != hf_logits.shape:
        return {
            "idx": idx, "ids_match": ids_match, "shape_mismatch": True,
            "engine_shape": tuple(engine_logits.shape), "hf_shape": tuple(hf_logits.shape),
            "passed": False,
        }

    cos_sim = torch.nn.functional.cosine_similarity(
        engine_logits.unsqueeze(0), hf_logits.unsqueeze(0)
    ).item()
    engine_argmax = int(engine_logits.argmax().item())
    hf_argmax = int(hf_logits.argmax().item())
    argmax_match = engine_argmax == hf_argmax
    passed = cos_sim >= COSINE_SIM_THRESHOLD and argmax_match

    # How close was the top-2 gap on EACH side, in probability space, not
    # just logit space -- a near-tied top-1/top-2 gap on either side is
    # exactly the "noise flips a near-tie" story; a wide gap on both sides
    # would mean something else is going on.
    engine_probs = torch.softmax(engine_logits, dim=-1)
    hf_probs = torch.softmax(hf_logits, dim=-1)
    engine_top2 = torch.topk(engine_probs, 2).values.tolist()
    hf_top2 = torch.topk(hf_probs, 2).values.tolist()

    return {
        "idx": idx, "ids_match": ids_match, "shape_mismatch": False,
        "cos_sim": cos_sim, "engine_argmax": engine_argmax, "hf_argmax": hf_argmax,
        "argmax_match": argmax_match, "passed": passed,
        "engine_top1_top2_gap": engine_top2[0] - engine_top2[1],
        "hf_top1_top2_gap": hf_top2[0] - hf_top2[1],
    }


def phase_compare():
    assert os.path.exists(ENGINE_LOGITS_PATH), f"missing {ENGINE_LOGITS_PATH} -- run --phase engine first"
    assert os.path.exists(HF_LOGITS_PATH), f"missing {HF_LOGITS_PATH} -- run --phase hf first"

    engine_results = torch.load(ENGINE_LOGITS_PATH, weights_only=True)["results"]
    hf_results = torch.load(HF_LOGITS_PATH, weights_only=True)["results"]

    if len(engine_results) != len(hf_results):
        print(f"FAIL: engine cache has {len(engine_results)} prompt(s), hf cache has "
              f"{len(hf_results)} -- re-run both phases against the same TARGET_INDICES.")
        return

    tokenizer = None
    outcomes = []
    print(f"{'idx':>4}  {'cosine':>9}  {'argmax':>7}  {'engine_gap':>11}  {'hf_gap':>9}  {'verdict':>7}")
    print("-" * 70)
    for er, hr in zip(engine_results, hf_results):
        idx = er["idx"]
        assert idx == hr["idx"], f"idx mismatch between cache files ({idx} vs {hr['idx']}) -- re-run both phases"
        outcome = _compare_one(idx, er, hr)
        outcomes.append(outcome)
        if outcome["shape_mismatch"]:
            print(f"{idx:>4}  {'--':>9}  {'--':>7}  {'--':>11}  {'--':>9}  {'FAIL':>7}  "
                  f"(shape mismatch: engine={outcome['engine_shape']} hf={outcome['hf_shape']})")
            continue
        verdict = "PASS" if outcome["passed"] else "FAIL"
        flag = "  <-- Check 1's divergent idx" if idx == DIVERGENT_IDX else ""
        print(f"{idx:>4}  {outcome['cos_sim']:>9.6f}  "
              f"{'match' if outcome['argmax_match'] else 'DIFF':>7}  "
              f"{outcome['engine_top1_top2_gap']:>11.4f}  {outcome['hf_top1_top2_gap']:>9.4f}  "
              f"{verdict:>7}{flag}")
        if not outcome["ids_match"]:
            print(f"      WARNING: tokenization mismatch for this prompt -- confounded, not a clean signal.")
        if not outcome["argmax_match"] and tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=True)
        if not outcome["argmax_match"]:
            hf_text = tokenizer.decode([outcome["hf_argmax"]])
            engine_text = tokenizer.decode([outcome["engine_argmax"]])
            print(f"      HF top-1: {hf_text!r} (token {outcome['hf_argmax']})   "
                  f"engine top-1: {engine_text!r} (token {outcome['engine_argmax']})")

    n_pass = sum(1 for o in outcomes if o["passed"])
    n_total = len(outcomes)
    print()
    print(f"THRESHOLD (stated before running): cosine >= {COSINE_SIM_THRESHOLD} AND argmax match, per prompt")
    print(f"OVERALL: {n_pass}/{n_total} prompts passed")

    divergent_outcome = next((o for o in outcomes if o["idx"] == DIVERGENT_IDX), None)
    print()
    print(f"idx={DIVERGENT_IDX} SPECIFICALLY (Check 1's divergent case):")
    if divergent_outcome is None or divergent_outcome.get("shape_mismatch"):
        print("  could not be evaluated -- see shape_mismatch warning above.")
    elif divergent_outcome["passed"]:
        print(f"  PASSED here (cosine={divergent_outcome['cos_sim']:.6f}, argmax matches). Prefill logits "
              f"agree at the raw-logits level even though Check 1's sampled first token diverged -- "
              f"points at something in the sample/dispatch path between logits and the emitted token "
              f"(e.g. a difference in how the two implementations break a near-tie), not the logits "
              f"themselves. Worth checking Check 1's engine/HF top-1 probabilities were actually close "
              f"if this needs to be chased further.")
    else:
        print(f"  FAILED here too (cosine={divergent_outcome['cos_sim']:.6f}, argmax "
              f"{'matches' if divergent_outcome['argmax_match'] else 'DIFFERS'}). Confirms the "
              f"disagreement is already present in the raw prefill logits, not introduced later in "
              f"sampling -- consistent with a real, reproducible prefill-numerics difference on this "
              f"prompt, not a one-off flip.")
        gap_ratio = divergent_outcome["engine_top1_top2_gap"] / max(divergent_outcome["hf_top1_top2_gap"], 1e-9)
        if min(divergent_outcome["engine_top1_top2_gap"], divergent_outcome["hf_top1_top2_gap"]) < 0.05:
            print(f"  Top1/top2 probability gap is small on at least one side (engine="
                  f"{divergent_outcome['engine_top1_top2_gap']:.4f}, hf={divergent_outcome['hf_top1_top2_gap']:.4f}) "
                  f"-- consistent with a genuinely near-tied decision that the already-known "
                  f"cross-implementation noise (cosine 0.9976-0.9997 on short prompts) could plausibly flip, "
                  f"not a large, confident disagreement.")
        else:
            print(f"  Top1/top2 probability gap is NOT small on either side (engine="
                  f"{divergent_outcome['engine_top1_top2_gap']:.4f}, hf={divergent_outcome['hf_top1_top2_gap']:.4f}) "
                  f"-- both implementations are confident, just in DIFFERENT tokens. That's a stronger signal "
                  f"than a near-tie flip; worth localizing further (e.g. reference_check_phase6_layerwise.py) "
                  f"rather than attributing to expected noise.")

    if n_pass < n_total:
        others_failed = [o["idx"] for o in outcomes if not o["passed"] and o["idx"] != DIVERGENT_IDX]
        if others_failed:
            print(f"\nOther prompt(s) besides idx={DIVERGENT_IDX} also failed: {others_failed} -- if the "
                  f"failure isn't isolated to the one case Check 1 already flagged, that's evidence for "
                  f"'long GSM8K-shaped prompts are a worse regime in general', not just one unlucky prompt.")


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
