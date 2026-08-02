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

A per-decoder-layer checkpoint alone conflates two very different
sublayers: self_attn/linear_attn, then a full MoE FFN pass. This script
ALSO captures each layer's raw attention/linear-attn sublayer output on
its own (engine: inlined in ModelRunner.get_prefill_layer_states; HF: a
forward hook on each layer's attention-ish submodule, auto-identified as
whichever child isn't input_layernorm/post_attention_layernorm/mlp), so
the compare phase can tell "the attention sublayer itself is wrong" apart
from "the attention sublayer is fine but MoE FFN is wrong" instead of only
seeing their combined effect.

Two hypotheses the per-decoder-layer trace distinguishes:
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

_SKIP_CHILD_NAMES = {"input_layernorm", "post_attention_layernorm", "mlp"}


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

    layer_types = list(llm.model_runner.model.model.layer_types)
    print(f"[engine] layer_types (layer 0 is {layer_types[0]}): {layer_types}")

    seq = Sequence(prompt_ids)
    seq.num_scheduled_tokens = len(prompt_ids)
    llm.scheduler.block_manager.allocate(seq, 0)
    if llm.model_runner.state_manager is not None:
        seq.state_slot = 0

    layer_states, attn_only_states, post_ln_states, logits = llm.model_runner.call("get_prefill_layer_states", [seq])
    assert layer_states is not None, "expected rank0 to return gathered layer states"
    print(f"[engine] captured {len(layer_states)} layer states + {len(attn_only_states)} "
          f"attn-only states + {len(post_ln_states)} post-input_layernorm states, "
          f"each layer_states shape {tuple(layer_states[0].shape)}; "
          f"final logits shape {tuple(logits.shape)}")

    torch.save({
        "prompt_ids": prompt_ids,
        "layer_types": layer_types,
        "layer_states": layer_states,
        "attn_only_states": attn_only_states,
        "post_ln_states": post_ln_states,
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

    decoder_layers = model.model.layers
    attn_captures = [None] * len(decoder_layers)
    # Full (not just final-position) input/output sequence per layer's
    # attention-ish submodule -- needed to feed the exact same real input
    # into a standalone re-run of that layer (diag_linear_attn_tp_isolation.py),
    # not just for a single-vector cosine-similarity summary.
    mixer_io = [None] * len(decoder_layers)
    handles = []

    def _make_hook(idx):
        # with_kwargs=True: the real model calls
        # self.linear_attn(hidden_states=hidden_states, ...) with
        # hidden_states passed as a KEYWORD arg, so the default hook
        # signature's positional-only `args` is empty -- hidden_states is
        # only reachable via `kwargs` here.
        def _hook(module, args, kwargs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            attn_captures[idx] = out[0, -1, :].detach().float().cpu()
            in_ = kwargs.get("hidden_states", args[0] if args else None)
            assert in_ is not None, f"layer {idx}: couldn't find hidden_states in hook args/kwargs"
            mixer_io[idx] = {
                "input": in_[0].detach().float().cpu(),      # (seq_len, hidden)
                "output": out[0].detach().float().cpu(),     # (seq_len, hidden)
            }
        return _hook

    mixer_names = []
    for i, layer in enumerate(decoder_layers):
        mixer_name = next(
            name for name, _ in layer.named_children() if name not in _SKIP_CHILD_NAMES
        )
        mixer_names.append(mixer_name)
        handles.append(getattr(layer, mixer_name).register_forward_hook(_make_hook(i), with_kwargs=True))
    print(f"[hf] attention-sublayer submodule name per layer (first 5): {mixer_names[:5]}")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(model.device)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)

    for h in handles:
        h.remove()

    if not hasattr(out, "hidden_states") or out.hidden_states is None:
        raise RuntimeError(
            f"[hf] output_hidden_states=True did not produce .hidden_states -- "
            f"type={type(out)}, available attrs={[a for a in dir(out) if not a.startswith('_')]}"
        )
    assert all(c is not None for c in attn_captures), "some layers' attention-sublayer hook never fired"

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
        "attn_only_states": attn_captures,
        "mixer_io": mixer_io,
        "mixer_names": mixer_names,
        "logits": logits,
    }, HF_STATES_PATH)
    print(f"[hf] saved to {HF_STATES_PATH}")
    print("[hf] PHASE COMPLETE")


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.float().unsqueeze(0), b.float().unsqueeze(0)).item()


def _last_row(t: torch.Tensor) -> torch.Tensor:
    return t[-1, :].float() if t.dim() == 2 else t.float()


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
    engine_attn = engine_payload["attn_only_states"]
    hf_attn = hf_payload["attn_only_states"]
    engine_post_ln = engine_payload.get("post_ln_states")
    hf_mixer_io = hf_payload.get("mixer_io")
    have_post_ln = engine_post_ln is not None and hf_mixer_io is not None

    if len(engine_states) != len(hf_states):
        print(f"\nFAIL: layer count mismatch -- engine captured {len(engine_states)} states, "
              f"hf captured {len(hf_states)}. Can't compare 1:1.")
        return

    header = f"\n{'idx':>4}  {'label':<30}  "
    if have_post_ln:
        header += f"{'post-LN cos':>11}  "
    header += f"{'attn-only cos':>13}  {'full-layer cos':>15}  {'max_abs_diff':>13}"
    print(header)
    print("-" * (len(header) - 1))
    trace = []    # (idx, label, full_layer_cos)
    for i, (e, h) in enumerate(zip(engine_states, hf_states)):
        e_vec = _last_row(e)
        h_vec = h.float()
        cos = _cos(e_vec, h_vec)
        max_abs = (e_vec - h_vec).abs().max().item()
        post_ln_cos_str = ""
        if i == 0:
            label = "embedding"
            attn_cos_str = "n/a"
            if have_post_ln:
                post_ln_cos_str = "n/a"
        else:
            layer_idx = i - 1
            type_tag = f" ({layer_types[layer_idx]})" if layer_types else ""
            label = f"layer {layer_idx}{type_tag}"
            a_cos = _cos(_last_row(engine_attn[layer_idx]), hf_attn[layer_idx].float())
            attn_cos_str = f"{a_cos:.6f}"
            if have_post_ln:
                pln_cos = _cos(_last_row(engine_post_ln[layer_idx]), hf_mixer_io[layer_idx]["input"].float())
                post_ln_cos_str = f"{pln_cos:.6f}"
        trace.append((i, label, cos))
        row = f"{i:>4}  {label:<30}  "
        if have_post_ln:
            row += f"{post_ln_cos_str:>11}  "
        row += f"{attn_cos_str:>13}  {cos:>15.6f}  {max_abs:>13.4e}"
        print(row)

    # Final logits checkpoint, one past the last decoder layer.
    engine_logits = engine_payload["logits"].float()
    hf_logits = hf_payload["logits"].float()
    if engine_logits.dim() == 2 and engine_logits.shape[0] == 1:
        engine_logits = engine_logits.squeeze(0)
    logits_cos = _cos(engine_logits, hf_logits)
    trace.append((len(engine_states), "final logits (norm+lm_head)", logits_cos))
    final_row = f"{len(engine_states):>4}  {'final logits (norm+lm_head)':<30}  "
    if have_post_ln:
        final_row += f"{'n/a':>11}  "
    final_row += f"{'n/a':>13}  {logits_cos:>15.6f}  {(engine_logits - hf_logits).abs().max().item():>13.4e}"
    print(final_row)

    # Bisect: does the attention sublayer alone already diverge at the
    # first bad layer, or is it fine and the MoE FFN is what breaks it?
    print()
    first_bad = next(((i, label, cos) for i, label, cos in trace if cos < 0.99), None)
    if first_bad is None:
        print("All checkpoints have cosine similarity >= 0.99 -- no meaningful divergence found "
              "anywhere in this trace.")
        return

    bad_idx, bad_label, _ = first_bad
    if bad_idx == 0:
        print(f"Divergence starts at the EMBEDDING layer itself -- check tokenizer/vocab alignment "
              f"or embed_tokens weight loading first.")
        return
    bad_layer_idx = bad_idx - 1
    attn_cos_at_bad = _cos(_last_row(engine_attn[bad_layer_idx]), hf_attn[bad_layer_idx].float())
    print(f"First divergence at {bad_label} (index {bad_idx}).")
    if have_post_ln:
        pln_cos_at_bad = _cos(_last_row(engine_post_ln[bad_layer_idx]), hf_mixer_io[bad_layer_idx]["input"].float())
        print(f"  post-input_layernorm cosine at this layer (BEFORE attention runs): {pln_cos_at_bad:.6f}")
        if pln_cos_at_bad < 0.999:
            print(f"  -> Already degraded before attention even runs. The bug is in input_layernorm "
                  f"(Qwen35RMSNorm) itself or its TP handling, not in {layer_types[bad_layer_idx] if layer_types else '(mixer)'}. "
                  f"Check Qwen35RMSNorm.weight loading (should be REPLICATED across ranks, never sharded "
                  f"-- hidden_size is never partitioned in this repo's TP scheme) and its forward formula "
                  f"against HF's Qwen3_5MoeRMSNorm.")
    print(f"  attention-sublayer-only cosine at this layer: {attn_cos_at_bad:.6f}")
    if attn_cos_at_bad < 0.99:
        print(f"  -> The attention/linear_attn sublayer ITSELF is already wrong at this layer, "
              f"before the MoE FFN even runs. Look at {layer_types[bad_layer_idx] if layer_types else '(mixer)'} "
              f"and its weight loading/forward logic.")
    else:
        print(f"  -> The attention/linear_attn sublayer output MATCHES HF closely here (cosine >= 0.99). "
              f"The divergence is introduced by the MoE FFN (routing, expert compute, or shared-expert "
              f"combine/all-reduce) at this same layer, not by attention.")

    # Full per-checkpoint shape classification, same as before.
    prior = trace[:bad_idx]
    near_one_before = all(cos >= 0.999 for _, _, cos in prior) if prior else True
    cliff = prior and (prior[-1][2] - first_bad[2]) > 0.5
    print()
    if near_one_before and cliff:
        print(f"SHAPE: SHARP DROP -- everything before {bad_label} is essentially bit-identical "
              f"(cosine >= 0.999), then falls off a cliff exactly at {bad_label}. Signature of a "
              f"discrete, localized bug -- NOT gradual M-RoPE/positional precision drift.")
    elif near_one_before:
        print(f"SHAPE: MODERATE DROP at {bad_label} (not a full cliff to near-zero/negative). "
              f"Still points at {bad_label} as the first place computation diverges.")
    else:
        print(f"SHAPE: GRADUAL DECAY -- cosine similarity was already trending down before "
              f"{bad_label} rather than holding near 1.0, consistent with compounding precision "
              f"error rather than one discrete bug.")


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
