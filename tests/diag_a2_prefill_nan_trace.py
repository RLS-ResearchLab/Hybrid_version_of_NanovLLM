"""Diagnostic follow-up to cluster_a2_tp_correctness.py's --phase compare
--dry-run-no-hf-reference failure: `non-finite prefill logits for 'The
capital of France is'`, reproduced on real GPU hardware AFTER
tests/make_fake_checkpoint.py confirmed every saved tensor is finite (no
NaN/Inf on disk) -- so the non-finite value is introduced somewhere DURING
the forward pass, not loaded from a bad checkpoint.

Uses ModelRunner.get_prefill_layer_states (the same tool
reference_check_phase6_layerwise.py uses to localize an engine/HF
divergence) to capture the residual-stream hidden state after the
embedding layer and after EVERY decoder layer, plus each layer's raw
attention/linear-attn sublayer output and post-input-layernorm input --
then reports the FIRST point (embedding, or which layer/which sublayer)
where torch.isfinite stops holding, instead of guessing.

Same prompt, same tp=1, same construction as the failing A2 run --
deterministic (temperature isn't even involved, this only touches prefill
logits), so this should reproduce the identical failure and pinpoint it.

Usage (small model, single GPU; needs tests/make_fake_checkpoint.py already
run):
    python tests/diag_a2_prefill_nan_trace.py
"""
import json
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "nanovllm" not in sys.modules:
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

CKPT_DIR = os.path.join(ROOT, "tests", "fake_qwen35_small")
PROMPT = "The capital of France is"  # exact prompt cluster_a2_tp_correctness.py's PROMPTS[0]
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class _AttrDict(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def _fake_from_pretrained(path, *args, **kwargs):
    with open(os.path.join(path, "config.json")) as f:
        d = json.load(f)
    d = _AttrDict(d)
    d.dtype = _DTYPE_MAP[d.pop("torch_dtype")]
    return d


def _report(name: str, t: torch.Tensor) -> bool:
    """Prints finite/nan/inf/min/max, returns True if t is fully finite."""
    tf = t.float()
    finite = torch.isfinite(tf).all().item()
    n_nan = torch.isnan(tf).sum().item()
    n_inf = torch.isinf(tf).sum().item()
    if finite:
        print(f"    [OK]   {name}: shape={tuple(t.shape)}  min={tf.min().item():.4g}  "
              f"max={tf.max().item():.4g}  abs_max={tf.abs().max().item():.4g}")
    else:
        print(f"    [BAD]  {name}: shape={tuple(t.shape)}  nan_count={n_nan}  inf_count={n_inf}  "
              f"finite_min={tf[torch.isfinite(tf)].min().item() if torch.isfinite(tf).any() else 'n/a'}  "
              f"finite_max={tf[torch.isfinite(tf)].max().item() if torch.isfinite(tf).any() else 'n/a'}")
    return finite


def main():
    assert torch.cuda.is_available(), "Requires CUDA"
    assert os.path.exists(os.path.join(CKPT_DIR, "model.safetensors")), (
        f"missing {CKPT_DIR}/model.safetensors -- run tests/make_fake_checkpoint.py first"
    )

    import nanovllm.config as config_mod
    config_mod.AutoConfig.from_pretrained = staticmethod(_fake_from_pretrained)

    from nanovllm.llm import LLM
    from nanovllm.engine.sequence import Sequence

    print(f"Constructing engine (tensor_parallel_size=1) from {CKPT_DIR} ...")
    llm = LLM(
        CKPT_DIR,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_num_seqs=1,
        max_model_len=2048,
    )

    layer_types = list(llm.model_runner.model.model.layer_types)
    print(f"layer_types ({len(layer_types)} layers): {layer_types}\n")

    prompt_ids = llm.tokenizer.encode(PROMPT)
    print(f"prompt: {PROMPT!r} -> token ids: {prompt_ids}\n")

    seq = Sequence(prompt_ids)
    seq.num_scheduled_tokens = len(prompt_ids)
    llm.scheduler.block_manager.allocate(seq, 0)
    if llm.model_runner.state_manager is not None:
        llm.model_runner.call("allocate_state_slot", seq)

    layer_states, attn_only_states, post_ln_states, logits = llm.model_runner.call(
        "get_prefill_layer_states", [seq]
    )
    assert layer_states is not None, "expected rank0 to return gathered layer states"

    print("=" * 78)
    print("LAYER-BY-LAYER FINITENESS TRACE")
    print("=" * 78)

    first_bad = None

    ok = _report("embedding output", layer_states[0])
    if not ok and first_bad is None:
        first_bad = "embedding"

    for i, ltype in enumerate(layer_types):
        print(f"\n  layer {i} ({ltype}):")
        ok_post_ln = _report(f"post_ln (pre-attn input)", post_ln_states[i])
        if not ok_post_ln and first_bad is None:
            first_bad = f"layer {i} post_ln (input_layernorm output / attention input)"

        ok_attn = _report(f"attn_only ({'self_attn' if ltype == 'full_attention' else 'linear_attn'} output)",
                           attn_only_states[i])
        if not ok_attn and first_bad is None:
            first_bad = f"layer {i} attn_only ({ltype} sublayer output, BEFORE MoE FFN)"

        ok_resid = _report(f"residual stream after layer {i} (post-MoE-FFN)", layer_states[i + 1])
        if not ok_resid and first_bad is None:
            first_bad = f"layer {i} residual stream (AFTER MoE FFN -- implicates MLP/MoE, not attention)"

    print(f"\n  final logits:")
    ok_logits = _report("logits", logits)
    if not ok_logits and first_bad is None:
        first_bad = "final logits (post final-norm + lm_head, everything upstream was finite)"

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if first_bad is None:
        print("Every captured checkpoint was finite -- could not reproduce the "
              "non-finite prefill logits failure with this exact prompt/config. "
              "(get_prefill_layer_states's own forward pass may differ subtly from "
              "get_prefill_logits's -- worth checking that divergence next if this happens.)")
    else:
        print(f"FIRST non-finite point: {first_bad}")
        print("Everything captured BEFORE this point was finite; use that to decide "
              "where to look next (e.g. a specific layer's attention/GDR math, or the "
              "MoE FFN -- the shared_expert path in particular, since the routed-expert "
              "path is KNOWN to be all-zero in this fixture per make_fake_checkpoint.py's "
              "own comment, and 0 * finite = 0, not NaN, so the routed side is an unlikely "
              "culprit by itself).")


if __name__ == "__main__":
    main()
