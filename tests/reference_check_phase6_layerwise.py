"""Phase 6 follow-up: localize WHERE the engine/HF divergence starts.

reference_check_phase6.py found final-logits cosine similarity -0.109
(anti-correlated) and a garbage top-1 token -- not the expected M-RoPE gap
(M-RoPE reduces to plain 1D RoPE for a text-only prompt, since
get_rope_index() assigns identical t/h/w position ids with no image/video
tokens present). This script captures the residual-stream hidden state
after the embedding layer, after each of the 40 decoder layers, and at
the final logits, for both the engine and HF, and reports per-checkpoint
cosine similarity so we can see exactly where it goes wrong instead of
guessing.

Two hypotheses this distinguishes:
  - M-RoPE / positional precision: errors compound gradually, layer over
    layer -- cosine similarity starts near 1.0 and decays smoothly.
  - A discrete bug (wrong dim order, sign error, bad mask, transposed
    weight, wrong dtype cast, ...): cosine similarity stays near 1.0 right
    up to one specific layer, then drops sharply.

Same 3-separate-process constraint as reference_check_phase6.py (both
models don't fit on the same GPUs at once):

    python tests/reference_check_phase6_layerwise.py --phase engine
    python tests/reference_check_phase6_layerwise.py --phase hf
    python tests/reference_check_phase6_layerwise.py --phase compare   # no GPU needed

Intermediate artifacts land in tests/_phase6_cache/ (same dir as
reference_check_phase6.py, different filenames -- safe to run either
script's phases in either order).
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
ENGINE_STATES_PATH = os.path.join(CACHE_DIR, "engine_layer_states.pt")
HF_STATES_PATH = os.path.join(CACHE_DIR, "hf_layer_states.pt")

PROMPT = "The capital of France is"


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

    # rank0's model object lives in this process (only rank>0 workers are
    # separate spawned processes) -- plain attribute access, no dispatch
    # needed, since this is just introspecting the layer schedule, not
    # running the model.
    layer_types = list(llm.model_runner.model.model.layer_types)
    print(f"[engine] layer_types (layer 0 is {layer_types[0]}): {layer_types}")

    seq = Sequence(prompt_ids)
    seq.num_scheduled_tokens = len(prompt_ids)
    llm.scheduler.block_manager.allocate(seq, 0)
    if llm.model_runner.state_manager is not None:
        seq.state_slot = 0

    layer_states, logits = llm.model_runner.call("get_prefill_layer_states", [seq])
    assert layer_states is not None, "expected rank0 to return gathered layer states"
    print(f"[engine] captured {len(layer_states)} states "
          f"(1 embedding + {len(layer_states) - 1} decoder layers), "
          f"each shape {tuple(layer_states[0].shape)}; final logits shape {tuple(logits.shape)}")

    torch.save({
        "prompt_ids": prompt_ids,
        "layer_types": layer_types,
        "layer_states": layer_states,
        "logits": logits,
    }, ENGINE_STATES_PATH)
    print(f"[engine] saved to {ENGINE_STATES_PATH}")
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

    input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(model.device)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)

    if not hasattr(out, "hidden_states") or out.hidden_states is None:
        raise RuntimeError(
            f"[hf] output_hidden_states=True did not produce .hidden_states -- "
            f"type={type(out)}, available attrs={[a for a in dir(out) if not a.startswith('_')]}"
        )
    # out.hidden_states: tuple of (num_layers + 1) tensors, each (1, seq_len, hidden).
    # Index 0 = embedding output, index i+1 = output of layer i. Take the
    # final-position slice to align with the engine's flat (N,...) capture
    # for this single-sequence, no-prefix-cache prompt.
    layer_states = [hs[0, -1, :].float().cpu() for hs in out.hidden_states]
    logits = out.logits[0, -1, :].float().cpu()
    print(f"[hf] captured {len(layer_states)} states "
          f"(1 embedding + {len(layer_states) - 1} decoder layers), "
          f"each shape {tuple(layer_states[0].shape)}; final logits shape {tuple(logits.shape)}")

    torch.save({
        "prompt_ids": prompt_ids,
        "layer_states": layer_states,
        "logits": logits,
    }, HF_STATES_PATH)
    print(f"[hf] saved to {HF_STATES_PATH}")
    print("[hf] PHASE COMPLETE")


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.float().unsqueeze(0), b.float().unsqueeze(0)).item()


def phase_compare():
    assert os.path.exists(ENGINE_STATES_PATH), f"missing {ENGINE_STATES_PATH} -- run --phase engine first"
    assert os.path.exists(HF_STATES_PATH), f"missing {HF_STATES_PATH} -- run --phase hf first"

    engine_payload = torch.load(ENGINE_STATES_PATH, weights_only=True)
    hf_payload = torch.load(HF_STATES_PATH, weights_only=True)

    engine_ids = engine_payload["prompt_ids"]
    hf_ids = hf_payload["prompt_ids"]
    print(f"engine prompt_ids: {engine_ids}")
    print(f"hf     prompt_ids: {hf_ids}")
    if engine_ids != hf_ids:
        print("WARNING: tokenization mismatch -- results below are confounded by this.")

    layer_types = engine_payload.get("layer_types")
    engine_states = engine_payload["layer_states"]
    hf_states = hf_payload["layer_states"]

    if len(engine_states) != len(hf_states):
        print(f"\nFAIL: layer count mismatch -- engine captured {len(engine_states)} states, "
              f"hf captured {len(hf_states)}. Can't compare 1:1.")
        return

    print(f"\n{'idx':>4}  {'label':<30}  {'cos_sim':>10}  {'max_abs_diff':>13}")
    print("-" * 64)
    trace = []    # (idx, label, cos)
    for i, (e, h) in enumerate(zip(engine_states, hf_states)):
        e_vec = e[-1, :].float() if e.dim() == 2 else e.float()
        h_vec = h.float()
        cos = _cos(e_vec, h_vec)
        max_abs = (e_vec - h_vec).abs().max().item()
        if i == 0:
            label = "embedding"
        else:
            layer_idx = i - 1
            type_tag = f" ({layer_types[layer_idx]})" if layer_types else ""
            label = f"layer {layer_idx}{type_tag}"
        trace.append((i, label, cos))
        print(f"{i:>4}  {label:<30}  {cos:>10.6f}  {max_abs:>13.4e}")

    # Final logits checkpoint, one past the last decoder layer.
    engine_logits = engine_payload["logits"].float()
    hf_logits = hf_payload["logits"].float()
    if engine_logits.dim() == 2 and engine_logits.shape[0] == 1:
        engine_logits = engine_logits.squeeze(0)
    logits_cos = _cos(engine_logits, hf_logits)
    trace.append((len(engine_states), "final logits (norm+lm_head)", logits_cos))
    print(f"{len(engine_states):>4}  {'final logits (norm+lm_head)':<30}  {logits_cos:>10.6f}  "
          f"{(engine_logits - hf_logits).abs().max().item():>13.4e}")

    # Distinguish gradual precision decay (M-RoPE-shaped) from a sharp,
    # discrete-bug-shaped drop. Report the full trace regardless -- this is
    # the evidence, not just the summary label.
    print()
    first_bad = next(((i, label, cos) for i, label, cos in trace if cos < 0.99), None)
    if first_bad is None:
        print("All checkpoints have cosine similarity >= 0.99 -- no meaningful divergence found "
              "anywhere in this trace.")
    else:
        bad_idx, bad_label, _ = first_bad
        prior = trace[:bad_idx]
        # "Gradual" = every step before the drop is itself trending downward
        # by a comparable, small amount. "Sharp" = flat near 1.0 right up
        # until one point, then a cliff.
        near_one_before = all(cos >= 0.999 for _, _, cos in prior) if prior else True
        cliff = prior and (prior[-1][2] - first_bad[2]) > 0.5
        if near_one_before and cliff:
            shape = "SHARP DROP"
            explanation = (
                f"Everything before {bad_label} is essentially bit-identical (cosine >= 0.999); "
                f"the cosine similarity falls off a cliff exactly at {bad_label}. This is the "
                f"signature of a discrete bug localized to that one step (or its direct inputs -- "
                f"state/conv_state, cu_seqlens, RoPE table, a transposed weight, wrong dtype cast) "
                f"-- NOT gradual M-RoPE/positional precision drift."
            )
        elif near_one_before:
            shape = "MODERATE DROP"
            explanation = (
                f"Everything before {bad_label} matches closely (cosine >= 0.999) but the drop at "
                f"{bad_label} isn't a full cliff to near-zero/negative. Still points at {bad_label} "
                f"as the first place computation diverges -- inspect that layer's own logic and its "
                f"inputs first."
            )
        else:
            shape = "GRADUAL DECAY"
            explanation = (
                f"Cosine similarity was already trending down before {bad_label} rather than "
                f"holding near 1.0 -- consistent with compounding precision error (e.g. the known, "
                f"deferred M-RoPE gap) rather than one discrete bug. Still worth checking {bad_label} "
                f"specifically, but this shape doesn't point at a single localized defect."
            )
        print(f"SHAPE: {shape}")
        print(explanation)


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
