"""Phase 6 follow-up: localize WHERE the engine/HF divergence starts.

reference_check_phase6.py found final-logits cosine similarity -0.109
(anti-correlated) and a garbage top-1 token -- not the expected M-RoPE gap
(M-RoPE reduces to plain 1D RoPE for a text-only prompt, since
get_rope_index() assigns identical t/h/w position ids with no image/video
tokens present). This script captures the residual-stream hidden state
after the embedding layer and after each of the 40 decoder layers, for
both the engine and HF, and reports per-layer cosine similarity so we can
see exactly which layer first goes wrong instead of guessing.

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

    seq = Sequence(prompt_ids)
    seq.num_scheduled_tokens = len(prompt_ids)
    llm.scheduler.block_manager.allocate(seq, 0)
    if llm.model_runner.state_manager is not None:
        seq.state_slot = 0

    layer_states = llm.model_runner.call("get_prefill_layer_states", [seq])
    assert layer_states is not None, "expected rank0 to return gathered layer states"
    print(f"[engine] captured {len(layer_states)} states "
          f"(1 embedding + {len(layer_states) - 1} decoder layers), "
          f"each shape {tuple(layer_states[0].shape)}")

    torch.save({"prompt_ids": prompt_ids, "layer_states": layer_states}, ENGINE_STATES_PATH)
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
    # for this single-sequence, no-prefix-cache prompt (final position ==
    # last token == what phase6's logits comparison already keys off).
    layer_states = [hs[0, -1, :].float().cpu() for hs in out.hidden_states]
    print(f"[hf] captured {len(layer_states)} states "
          f"(1 embedding + {len(layer_states) - 1} decoder layers), "
          f"each shape {tuple(layer_states[0].shape)}")

    torch.save({"prompt_ids": prompt_ids, "layer_states": layer_states}, HF_STATES_PATH)
    print(f"[hf] saved to {HF_STATES_PATH}")
    print("[hf] PHASE COMPLETE")


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

    engine_states = engine_payload["layer_states"]
    hf_states = hf_payload["layer_states"]

    if len(engine_states) != len(hf_states):
        print(f"\nFAIL: layer count mismatch -- engine captured {len(engine_states)} states, "
              f"hf captured {len(hf_states)}. Can't compare 1:1.")
        return

    # Engine's per-layer capture is the flat (N, hidden) residual stream for
    # the whole prompt; take this single sequence's final position to match
    # HF's per-position slice.
    print(f"\n{'idx':>4}  {'label':<26}  {'cos_sim':>10}  {'max_abs_diff':>13}")
    print("-" * 60)
    first_bad_idx = None
    for i, (e, h) in enumerate(zip(engine_states, hf_states)):
        e_vec = e[-1, :].float() if e.dim() == 2 else e.float()
        h_vec = h.float()
        cos = torch.nn.functional.cosine_similarity(e_vec.unsqueeze(0), h_vec.unsqueeze(0)).item()
        max_abs = (e_vec - h_vec).abs().max().item()
        label = "embedding" if i == 0 else f"layer {i - 1}"
        flag = ""
        if cos < 0.99 and first_bad_idx is None:
            first_bad_idx = i
            flag = "  <-- first drop below 0.99"
        print(f"{i:>4}  {label:<26}  {cos:>10.6f}  {max_abs:>13.4e}{flag}")

    print()
    if first_bad_idx is None:
        print("All captured points have cosine similarity >= 0.99 -- divergence (if any) is "
              "entirely in the final norm/lm_head, not the decoder stack.")
    elif first_bad_idx == 0:
        print("Divergence starts at the EMBEDDING layer itself (before any decoder layer runs) -- "
              "check tokenizer/vocab alignment or embed_tokens weight loading first.")
    else:
        label = f"layer {first_bad_idx - 1}"
        print(f"Divergence first appears at {label} (index {first_bad_idx}). Everything before it "
              f"matches; that layer's own computation (or its inputs, e.g. state/conv_state, "
              f"cu_seqlens, RoPE) is the place to look next.")


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
