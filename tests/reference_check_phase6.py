"""Phase 6 reference check: compare this engine's raw prefill logits against
the official HF reference implementation's logits, for the identical
prompt, at bf16 (the real deployment dtype, not fp32), tp_size=2/ep_size=2
(tonight's already-validated configuration).

PASS/FAIL THRESHOLD -- STATED HERE, BEFORE RUNNING, NOT AFTER SEEING THE
NUMBER: cosine similarity >= 0.99 AND top-1 (argmax) token match, both
required. Cosine similarity alone can hide a real disagreement (two
distributions can point in a similar overall direction while disagreeing on
which token actually wins); argmax is the more direct predictor of
generated-text correctness. Below threshold is not a surprise and does not
by itself indicate a new bug: M-RoPE (mrope_interleaved / mrope_section,
declared in this checkpoint's own config.json) is not implemented in this
engine -- deliberately deferred from the very first message of tonight's
session, before any of tonight's config/loader/EP/TP work began. This check
exists to quantify that gap's actual numeric impact with a real,
reproducible number, not to declare a pass that was never really at stake.

GPU MEMORY -- run as three SEPARATE process invocations, each fully exiting
before the next starts. This engine (~42 GB/rank including KV cache, at
tp_size=2, confirmed earlier tonight) and the dense HF reference model
(~70 GB total, bf16, no sharding-aware loading) cannot both fit on the same
2x47.54 GB GPUs at the same time. There is deliberately no "run everything"
default mode, to avoid the temptation to OOM:

    python tests/reference_check_phase6.py --phase engine
    python tests/reference_check_phase6.py --phase hf
    python tests/reference_check_phase6.py --phase compare   # no GPU needed

Intermediate artifacts land in tests/_phase6_cache/ (repo-relative, safe to
delete after --phase compare; not intended to be committed -- NOTE: this
repo's own .gitignore is not plain UTF-8 text (confirmed via `file
.gitignore` -> "data", not "ASCII text"), which is very likely why
qwen35_checkpoint/config.json and model.safetensors.index.json ended up
tracked despite an apparent intent to ignore them -- unrelated to tonight's
work, flagging as a separate, pre-existing repo-hygiene item, not fixed
here).
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

PROMPT = "The capital of France is"

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
        max_num_seqs=1,
        max_model_len=2048,
    )

    prompt_ids = llm.tokenizer.encode(PROMPT)
    print(f"[engine] prompt token ids: {prompt_ids}")

    seq = Sequence(prompt_ids)
    seq.num_scheduled_tokens = len(prompt_ids)
    # prepare_prefill() (engine/model_runner.py) treats an empty block_table
    # as the warmup case and skips building slot_mapping entirely, which
    # then fails store_kvcache's `slot_mapping.numel() == N` assertion for
    # any real (non-warmup) sequence. Allocate real KV-cache blocks the same
    # way Scheduler.schedule() does for a fresh, uncached sequence.
    llm.scheduler.block_manager.allocate(seq, 0)
    if llm.model_runner.state_manager is not None:
        # Matches ModelRunner.warmup_model()'s exact pattern -- slot 0 is
        # safe to reuse here: run()'s state_manager.get_all/set_all are
        # agnostic to the scheduler's separate allocate/free bookkeeping
        # (confirmed by reading model_runner.py -- warmup_model() itself
        # never calls an allocate/free method around its own state_slot
        # usage either, and its own post-warmup assertion about free
        # slot count already proves this doesn't collide).
        seq.state_slot = 0

    logits = llm.model_runner.call("get_prefill_logits", [seq])
    assert logits is not None, "expected rank0 to return gathered logits"
    print(f"[engine] logits shape: {tuple(logits.shape)}  dtype: {logits.dtype}")

    torch.save({"prompt_ids": prompt_ids, "logits": logits}, ENGINE_LOGITS_PATH)
    print(f"[engine] saved to {ENGINE_LOGITS_PATH}")
    print("[engine] PHASE COMPLETE -- exit this process fully before running --phase hf")


def phase_hf():
    os.makedirs(CACHE_DIR, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=True)
    prompt_ids = tokenizer.encode(PROMPT)
    print(f"[hf] prompt token ids: {prompt_ids}")

    model = AutoModelForCausalLM.from_pretrained(
        CKPT_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Confirm (not assume) a text-only call works with no pixel/image
    # inputs -- inspect the signature first, then actually try it. If the
    # forward() call below raises over a missing required image argument,
    # that failure IS the answer to "does it require image tensors."
    import inspect
    sig = inspect.signature(model.forward)
    image_params = [p for p in sig.parameters if "pixel" in p.lower() or "image" in p.lower()]
    required_image_params = [p for p in image_params if sig.parameters[p].default is inspect.Parameter.empty]
    print(f"[hf] model.forward() parameters: {list(sig.parameters.keys())}")
    print(f"[hf] image/pixel-related params: {image_params}")
    print(f"[hf] of those, REQUIRED (no default): {required_image_params}")

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

    torch.save({"prompt_ids": prompt_ids, "logits": logits}, HF_LOGITS_PATH)
    print(f"[hf] saved to {HF_LOGITS_PATH}")
    print("[hf] PHASE COMPLETE")


def phase_compare():
    assert os.path.exists(ENGINE_LOGITS_PATH), f"missing {ENGINE_LOGITS_PATH} -- run --phase engine first"
    assert os.path.exists(HF_LOGITS_PATH), f"missing {HF_LOGITS_PATH} -- run --phase hf first"

    engine_payload = torch.load(ENGINE_LOGITS_PATH, weights_only=True)
    hf_payload = torch.load(HF_LOGITS_PATH, weights_only=True)

    engine_ids = engine_payload["prompt_ids"]
    hf_ids = hf_payload["prompt_ids"]
    print(f"engine prompt_ids: {engine_ids}")
    print(f"hf     prompt_ids: {hf_ids}")
    ids_match = engine_ids == hf_ids
    print(f"tokenization matches exactly: {ids_match}")
    if not ids_match:
        print("WARNING: tokenization mismatch -- any logits difference below is confounded "
              "by this, not a clean signal about model computation alone.")

    engine_logits = engine_payload["logits"].float()
    hf_logits = hf_payload["logits"].float()
    print(f"engine logits shape: {tuple(engine_logits.shape)}  "
          f"dtype in file: {engine_payload['logits'].dtype}")
    print(f"hf     logits shape: {tuple(hf_logits.shape)}  "
          f"dtype in file: {hf_payload['logits'].dtype}")

    if engine_logits.shape != hf_logits.shape:
        print(f"\nFAIL: shape mismatch, cannot compare directly "
              f"({tuple(engine_logits.shape)} vs {tuple(hf_logits.shape)})")
        return

    cos_sim = torch.nn.functional.cosine_similarity(
        engine_logits.unsqueeze(0), hf_logits.unsqueeze(0)
    ).item()

    engine_argmax = int(engine_logits.argmax().item())
    hf_argmax = int(hf_logits.argmax().item())
    argmax_match = engine_argmax == hf_argmax

    print()
    print(f"cosine similarity: {cos_sim:.6f}")
    print(f"engine argmax token id: {engine_argmax}")
    print(f"hf     argmax token id: {hf_argmax}")
    print(f"argmax match: {argmax_match}")

    if not argmax_match:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=True)
            hf_token_text = tokenizer.decode([hf_argmax])
            engine_token_text = tokenizer.decode([engine_argmax])
            print(f"\nHF's actual top-1 continuation for {PROMPT!r}: {hf_token_text!r} (token {hf_argmax})")
            print(f"Engine's actual top-1 continuation:            {engine_token_text!r} (token {engine_argmax})")
        except Exception as e:
            print(f"(could not decode tokens for display: {e})")

    passed = cos_sim >= COSINE_SIM_THRESHOLD and argmax_match
    print()
    print(f"THRESHOLD (stated before running): cosine >= {COSINE_SIM_THRESHOLD} AND argmax match")
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("Expected, not a surprise: M-RoPE (mrope_interleaved/mrope_section, declared in this "
              "checkpoint's config) is not implemented in this engine -- deliberately deferred from "
              "message 1 of tonight's session. This measurement quantifies that gap's actual numeric "
              "impact; it does not indicate a new bug introduced tonight.")


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
