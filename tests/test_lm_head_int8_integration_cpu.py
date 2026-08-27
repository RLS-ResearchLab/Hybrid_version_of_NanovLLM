"""CPU-only correctness check for lm_head_int8_integration.py's
quantize_lm_head_inplace / apply_lm_head_int8_quantization -- exercised
without CUDA, without triton.

Sections 1-7 use minimal stand-in modules (same attribute names/shapes) for
the load-time mutation logic. Section 8 runs the REAL
layers/embed_head.py::ParallelLMHead through a real decode-context
forward() -- both the bf16 and the int8 dequant-on-read branch -- and
compares logits, since models/qwen3_5.py imports on a CPU box now (the
triton import in layers/fused_moe_int8.py was made lazy 2026-08-27).
engine/model_runner.py DOES call apply_lm_head_int8_quantization when
config.use_lm_head_int8 is set.

Does NOT validate: the forward-path THROUGHPUT question -- lm_head reads its
whole weight matrix every call, so naive dequant-then-F.linear is a
plausible bandwidth regression vs bf16-direct (see lm_head_int8_integration.py's
module docstring). That's GPU-A/B-only.

Usage:
    TORCHDYNAMO_DISABLE=1 python tests/test_lm_head_int8_integration_cpu.py
"""
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "nanovllm" not in sys.modules:
    _pkg = types.ModuleType("nanovllm")
    _pkg.__path__ = [_ROOT]
    _pkg.__file__ = os.path.join(_ROOT, "__init__.py")
    sys.modules["nanovllm"] = _pkg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lm_head_int8_integration import quantize_lm_head_inplace, apply_lm_head_int8_quantization  # noqa: E402


class _FakeParallelLMHead(nn.Module):
    """Minimal stand-in for layers/embed_head.py's real ParallelLMHead --
    same .weight Parameter name/shape quantize_lm_head_inplace actually
    reads/deletes, nothing else."""

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.02)


class _FakeCausalLM(nn.Module):
    """Minimal stand-in for Qwen35ForCausalLM -- same .lm_head attribute
    name apply_lm_head_int8_quantization's getattr(model, "lm_head", None)
    actually looks for."""

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.lm_head = _FakeParallelLMHead(vocab_size, hidden_size)


def main():
    torch.manual_seed(0)
    ok = True

    V, H, group_size = 4096, 2048, 128
    lm_head = _FakeParallelLMHead(V, H)
    w_before = lm_head.weight.data.clone()

    quantize_lm_head_inplace(lm_head, group_size)

    # ---- 1. bf16/fp32 Parameter genuinely gone ----
    has_weight_param = "weight" in lm_head._parameters and lm_head._parameters["weight"] is not None
    print(f"[1] .weight still a Parameter: {has_weight_param}")
    assert not has_weight_param, "original weight Parameter should be deleted, not merely emptied"
    try:
        lm_head.weight
        raise AssertionError("lm_head.weight should raise AttributeError after deletion")
    except AttributeError:
        pass

    # ---- 2. int8 buffers present, correct dtype/shape ----
    assert hasattr(lm_head, "weight_int8") and lm_head.weight_int8.dtype == torch.int8
    assert lm_head.weight_int8.shape == (V, H)
    assert lm_head.weight_scale.shape == (V, H // group_size)
    print(f"[2] weight_int8={tuple(lm_head.weight_int8.shape)} weight_scale={tuple(lm_head.weight_scale.shape)}")

    # ---- 3. registered as buffers, not Parameters ----
    assert "weight_int8" in lm_head._buffers and "weight_int8" not in lm_head._parameters
    print("[3] registered as buffers, not Parameters -- confirmed")

    # ---- 4. group_size recorded ----
    assert lm_head.lm_head_int8_group_size == group_size
    print(f"[4] lm_head_int8_group_size={lm_head.lm_head_int8_group_size}")

    # ---- 5. dequantized values close to the original ----
    from moe_int8_quantize import dequantize_weight_int8_grouped
    deq = dequantize_weight_int8_grouped(lm_head.weight_int8, lm_head.weight_scale, group_size, torch.float32)
    cos = F.cosine_similarity(w_before.reshape(-1), deq.reshape(-1), dim=0).item()
    print(f"[5] dequant-vs-original cosine={cos:.6f}")
    assert cos > 0.999
    ok &= cos > 0.999

    # ---- 6. apply_lm_head_int8_quantization: presence-check wrapper ----
    model = _FakeCausalLM(V, H)
    n = apply_lm_head_int8_quantization(model, group_size)
    print(f"[6] apply_lm_head_int8_quantization returned {n} (expect 1)")
    assert n == 1
    assert hasattr(model.lm_head, "weight_int8")
    ok &= (n == 1)

    # ---- 7. no-lm_head model returns 0, doesn't crash ----
    class _NoLMHead(nn.Module):
        pass
    n0 = apply_lm_head_int8_quantization(_NoLMHead(), group_size)
    print(f"[7] no-lm_head model returned {n0} (expect 0)")
    assert n0 == 0
    ok &= (n0 == 0)

    # ---- 8. REAL ParallelLMHead forward(): bf16 vs int8-dequant-on-read ----
    # Previously skipped (couldn't import the real class -- triton wall);
    # PRE-1 removed that. This exercises the actual int8 branch in
    # ParallelLMHead.forward() (layers/embed_head.py), which is what the
    # engine runs every decode step under use_lm_head_int8=True.
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29551")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("gloo")
    from nanovllm.layers.embed_head import ParallelLMHead
    from nanovllm.utils.context import set_context, reset_context

    torch.manual_seed(2)
    real_head = ParallelLMHead(V, H)
    with torch.no_grad():
        real_head.weight.normal_(0, 0.02)
    Ntok = 32
    x = torch.randn(Ntok, H) * 0.1

    set_context(False)  # decode context -- forward() skips the prefill last-token slice
    with torch.no_grad():
        logits_bf16 = real_head(x).clone()

    n_real = apply_lm_head_int8_quantization(types.SimpleNamespace(lm_head=real_head), group_size)
    assert n_real == 1 and hasattr(real_head, "weight_int8") and not hasattr(real_head, "weight")
    with torch.no_grad():
        logits_int8 = real_head(x)
    reset_context()

    cos8 = F.cosine_similarity(logits_bf16.reshape(-1), logits_int8.reshape(-1), dim=0).item()
    argmax_agree = (logits_bf16.argmax(-1) == logits_int8.argmax(-1)).float().mean().item()
    print(f"[8] real ParallelLMHead.forward(): int8-vs-bf16 logits cosine={cos8:.6f}  "
          f"argmax agreement={argmax_agree*100:.1f}% over {Ntok} tokens")
    assert cos8 > 0.999, f"real forward int8 branch cosine {cos8:.6f} too low"
    assert argmax_agree > 0.90, f"real forward argmax agreement {argmax_agree*100:.1f}% too low"
    ok &= cos8 > 0.999 and argmax_agree > 0.90

    print("\nPASS" if ok else "\nFAIL")
    print("\nScope reminder: validates the load-time quantization pass + the real "
          "ParallelLMHead.forward() int8 branch's numeric behavior -- says NOTHING about the "
          "forward-path THROUGHPUT question (naive dequant-then-F.linear vs bf16-direct), "
          "which needs a GPU matched A/B.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
