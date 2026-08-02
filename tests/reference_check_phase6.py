"""Phase 6 reference check: compare this engine's raw prefill logits against
the official HF reference implementation's logits, for several varied
prompts, at bf16 (the real deployment dtype, not fp32), tp_size=2/ep_size=2.

PASS/FAIL THRESHOLD -- STATED HERE, BEFORE RUNNING, NOT AFTER SEEING THE
NUMBER: cosine similarity >= 0.99 AND top-1 (argmax) token match, both
required, PER PROMPT. Cosine similarity alone can hide a real disagreement
(two distributions can point in a similar overall direction while
disagreeing on which token actually wins); argmax is the more direct
predictor of generated-text correctness.

Multiple prompts (varied length/content -- short factual completion, a
GSM8K-style arithmetic word problem, a longer narrative, a bare numeric
expression) are run in ONE process per phase, not one process per prompt:
loading this engine (~42 GB/rank) or the dense HF reference (~70 GB) is the
expensive part, and re-paying that cost per prompt would be wasteful. This
directly de-risks Phase 7 (a real GSM8K run) against a prompt-dependent
issue that a single prompt's PASS wouldn't have caught.

GPU MEMORY -- run as three SEPARATE process invocations, each fully exiting
before the next starts. This engine and the dense HF reference model
cannot both fit on the same 2x47.54 GB GPUs at the same time. There is
deliberately no "run everything" default mode, to avoid the temptation to
OOM:

    python tests/reference_check_phase6.py --phase engine
    python tests/reference_check_phase6.py --phase hf
    python tests/reference_check_phase6.py --phase compare   # no GPU needed

Intermediate artifacts land in tests/_phase6_cache/ (repo-relative, safe to
delete after --phase compare; not intended to be committed).
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

CKPT_DIR = os.path.join(ROOT, "qwen35_checkpoint")
CACHE_DIR = os.path.join(ROOT, "tests", "_phase6_cache")
ENGINE_LOGITS_PATH = os.path.join(CACHE_DIR, "engine_logits.pt")
HF_LOGITS_PATH = os.path.join(CACHE_DIR, "hf_logits.pt")

# Varied length/content, deliberately: a short factual completion (the
# original single-prompt check, kept for continuity), a GSM8K-style
# arithmetic word problem (closest to Phase 7's actual eval distribution),
# a bare numeric expression, a longer narrative (stresses the 5-token
# prompt's KV-cache/state assumptions at a different length), and a
# multi-clause sentence with an embedded quotation.
PROMPTS = [
    "The capital of France is",
    "If John has 3 apples and buys 5 more, he has",
    "2 + 2 =",
    "In a small village, there lived a farmer who had",
    "The quick brown fox jumps over the lazy dog. This sentence is famous because",
]

# Stated before running, not after seeing the number.
COSINE_SIM_THRESHOLD = 0.99


def phase_engine():
    os.makedirs(CACHE_DIR, exist_ok=True)

    from nanovllm.llm import LLM
    from nanovllm.engine.sequence import Sequence

    llm = LLM(
        CKPT_DIR,
        enforce_eager=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_num_seqs=len(PROMPTS),
        max_model_len=2048,
    )

    results = []
    for prompt in PROMPTS:
        prompt_ids = llm.tokenizer.encode(prompt)
        print(f"[engine] prompt {prompt!r} -> token ids: {prompt_ids}")

        seq = Sequence(prompt_ids)
        seq.num_scheduled_tokens = len(prompt_ids)
        # prepare_prefill() (engine/model_runner.py) treats an empty
        # block_table as the warmup case and skips building slot_mapping
        # entirely, which then fails store_kvcache's
        # `slot_mapping.numel() == N` assertion for any real (non-warmup)
        # sequence. Allocate real KV-cache blocks the same way
        # Scheduler.schedule() does for a fresh, uncached sequence.
        llm.scheduler.block_manager.allocate(seq, 0)
        if llm.model_runner.state_manager is not None:
            # NOT `seq.state_slot = 0`, and NOT
            # `llm.model_runner.state_manager.allocate(seq)` directly either
            # -- ModelRunner.warmup_model() runs a REAL forward pass using
            # slot 0 and persists genuinely non-zero state via set_all()
            # (never re-zeroed afterward; its post-warmup assertion only
            # checks free/used slot-ID bookkeeping), and at
            # tensor_parallel_size>1 each rank is a SEPARATE PROCESS with
            # its OWN independent StateManager (its own warmup, its own
            # buffers). Calling .allocate(seq) directly on this process's
            # llm.model_runner only zeros rank0's copy of its slot --
            # rank1's copy is untouched, still warmup-contaminated
            # (confirmed empirically: doing it this way only partially
            # closed the gap on the first single-prompt run of this
            # script). Dispatch through .call(...) instead, the same
            # shared-memory mechanism run()/get_prefill_logits() use, so
            # allocate() actually runs on every rank's own StateManager.
            # max_num_seqs=len(PROMPTS) above means each prompt in this
            # loop gets its OWN fresh slot (no reuse/free needed between
            # iterations).
            llm.model_runner.call("allocate_state_slot", seq)

        logits = llm.model_runner.call("get_prefill_logits", [seq])
        assert logits is not None, "expected rank0 to return gathered logits"
        print(f"[engine] logits shape: {tuple(logits.shape)}  dtype: {logits.dtype}")
        results.append({"prompt": prompt, "prompt_ids": prompt_ids, "logits": logits})

    torch.save({"results": results}, ENGINE_LOGITS_PATH)
    print(f"[engine] saved {len(results)} prompt(s) to {ENGINE_LOGITS_PATH}")
    print("[engine] PHASE COMPLETE -- exit this process fully before running --phase hf")


def phase_hf():
    os.makedirs(CACHE_DIR, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        CKPT_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Confirm (not assume) a text-only call works with no pixel/image
    # inputs -- inspect the signature first, then actually try it on the
    # first prompt. If the forward() call raises over a missing required
    # image argument, that failure IS the answer to "does it require image
    # tensors."
    import inspect
    sig = inspect.signature(model.forward)
    image_params = [p for p in sig.parameters if "pixel" in p.lower() or "image" in p.lower()]
    required_image_params = [p for p in image_params if sig.parameters[p].default is inspect.Parameter.empty]
    print(f"[hf] model.forward() parameters: {list(sig.parameters.keys())}")
    print(f"[hf] image/pixel-related params: {image_params}")
    print(f"[hf] of those, REQUIRED (no default): {required_image_params}")

    results = []
    for prompt in PROMPTS:
        prompt_ids = tokenizer.encode(prompt)
        print(f"[hf] prompt {prompt!r} -> token ids: {prompt_ids}")

        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(model.device)
        with torch.no_grad():
            out = model(input_ids=input_ids)

        if not hasattr(out, "logits"):
            raise RuntimeError(
                f"[hf] unexpected output structure -- no .logits attribute. "
                f"type={type(out)}, available attrs={[a for a in dir(out) if not a.startswith('_')]}"
            )
        logits_all_positions = out.logits  # (1, seq_len, vocab)
        logits = logits_all_positions[0, -1, :].float().cpu()
        print(f"[hf] full logits shape: {tuple(logits_all_positions.shape)}, "
              f"final-position logits shape: {tuple(logits.shape)}")
        results.append({"prompt": prompt, "prompt_ids": prompt_ids, "logits": logits})

    torch.save({"results": results}, HF_LOGITS_PATH)
    print(f"[hf] saved {len(results)} prompt(s) to {HF_LOGITS_PATH}")
    print("[hf] PHASE COMPLETE")


def _compare_one(prompt: str, engine_result: dict, hf_result: dict) -> dict:
    engine_ids = engine_result["prompt_ids"]
    hf_ids = hf_result["prompt_ids"]
    ids_match = engine_ids == hf_ids

    engine_logits = engine_result["logits"].float()
    hf_logits = hf_result["logits"].float()
    if engine_logits.dim() == 2 and engine_logits.shape[0] == 1:
        # get_prefill_logits() returns one row per sequence in the batch
        # ((num_seqs, vocab)); phase_hf() already slices down to the single
        # final position ((vocab,)). Same quantity, different leading dim.
        engine_logits = engine_logits.squeeze(0)

    if engine_logits.shape != hf_logits.shape:
        return {
            "prompt": prompt, "ids_match": ids_match, "shape_mismatch": True,
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

    return {
        "prompt": prompt, "ids_match": ids_match, "shape_mismatch": False,
        "cos_sim": cos_sim, "engine_argmax": engine_argmax, "hf_argmax": hf_argmax,
        "argmax_match": argmax_match, "passed": passed,
    }


def phase_compare():
    assert os.path.exists(ENGINE_LOGITS_PATH), f"missing {ENGINE_LOGITS_PATH} -- run --phase engine first"
    assert os.path.exists(HF_LOGITS_PATH), f"missing {HF_LOGITS_PATH} -- run --phase hf first"

    engine_results = torch.load(ENGINE_LOGITS_PATH, weights_only=True)["results"]
    hf_results = torch.load(HF_LOGITS_PATH, weights_only=True)["results"]

    if len(engine_results) != len(hf_results):
        print(f"FAIL: engine cache has {len(engine_results)} prompt(s), hf cache has "
              f"{len(hf_results)} -- re-run both phases against the same PROMPTS list.")
        return

    tokenizer = None
    outcomes = []
    print(f"{'#':>3}  {'prompt':<55}  {'cosine':>9}  {'argmax':>7}  {'verdict':>7}")
    print("-" * 90)
    for i, (prompt, er, hr) in enumerate(zip(PROMPTS, engine_results, hf_results)):
        assert er["prompt"] == prompt == hr["prompt"], (
            f"prompt #{i} mismatch between cache and current PROMPTS list -- re-run both phases"
        )
        outcome = _compare_one(prompt, er, hr)
        outcomes.append(outcome)
        shown = prompt if len(prompt) <= 52 else prompt[:49] + "..."
        if outcome["shape_mismatch"]:
            print(f"{i:>3}  {shown:<55}  {'--':>9}  {'--':>7}  {'FAIL':>7}  (shape mismatch: "
                  f"engine={outcome['engine_shape']} hf={outcome['hf_shape']})")
            continue
        verdict = "PASS" if outcome["passed"] else "FAIL"
        print(f"{i:>3}  {shown:<55}  {outcome['cos_sim']:>9.6f}  "
              f"{'match' if outcome['argmax_match'] else 'DIFF':>7}  {verdict:>7}")
        if not outcome["ids_match"]:
            print(f"     WARNING: tokenization mismatch for this prompt -- its result above is "
                  f"confounded by that, not a clean signal.")
        if not outcome["argmax_match"] and tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=True)
        if not outcome["argmax_match"]:
            hf_text = tokenizer.decode([outcome["hf_argmax"]])
            engine_text = tokenizer.decode([outcome["engine_argmax"]])
            print(f"     HF top-1: {hf_text!r} (token {outcome['hf_argmax']})   "
                  f"engine top-1: {engine_text!r} (token {outcome['engine_argmax']})")

    n_pass = sum(1 for o in outcomes if o["passed"])
    n_total = len(outcomes)
    print()
    print(f"THRESHOLD (stated before running): cosine >= {COSINE_SIM_THRESHOLD} AND argmax match, per prompt")
    print(f"OVERALL: {n_pass}/{n_total} prompts passed")
    if n_pass == n_total:
        print("VERDICT: PASS -- holds across varied prompt lengths/content, not just one lucky case.")
    else:
        failed = [o["prompt"] for o in outcomes if not o["passed"]]
        print(f"VERDICT: FAIL -- {n_total - n_pass} prompt(s) did not meet the bar: {failed}")
        print("A prompt-dependent failure here is exactly what running a single prompt would have "
              "missed -- worth localizing with reference_check_phase6_layerwise.py against the "
              "specific failing prompt(s) before trusting Phase 7 to generalize.")


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
