"""Verifies the Qwen35LinearAttention TP-sizing fix (total_lkh/total_lvh vs
self.lkh/self.lvh) at the real qwen35_checkpoint config, tp_size=2. Same
meta-device construction pattern as Test A (tests/test_load_model_ep.py).

Also re-runs the full_attention-layer shape check as a regression: the fix
only touched Qwen35LinearAttention, so Qwen35FullAttention's q/k/v/o must be
completely unaffected.

Usage: python tests/test_linear_attn_tp_fix.py
"""
import os
import sys
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "nanovllm" not in sys.modules:
    _pkg = types.ModuleType("nanovllm")
    _pkg.__path__ = [ROOT]
    _pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = _pkg

CKPT_DIR = os.path.join(ROOT, "qwen35_checkpoint")


def _stub_attention_module():
    import torch.nn as nn
    stub = types.ModuleType("nanovllm.layers.attention")

    class Attention(nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

        def forward(self, *a, **k):
            raise NotImplementedError

    stub.Attention = Attention
    sys.modules["nanovllm.layers.attention"] = stub


def worker(rank: int):
    dist.init_process_group("gloo", init_method="tcp://localhost:29901", rank=rank, world_size=2)
    _stub_attention_module()

    from nanovllm.config import Config
    from nanovllm.layers.linear import ColumnParallelLinear
    from nanovllm.models.qwen3_5 import Qwen35ForCausalLM

    config = Config(CKPT_DIR)
    hc = config.hf_config
    print(f"[rank{rank}] real config: linear_attn_kq_heads(getattr)="
          f"{getattr(hc, 'linear_attn_kq_heads', 16)} linear_attn_v_heads(getattr)="
          f"{getattr(hc, 'linear_attn_v_heads', 32)} linear_attn_head_dim(getattr)="
          f"{getattr(hc, 'linear_attn_head_dim', 128)}")
    # NOTE (stale as of 2026-08-28, left for history): this used to warn that
    # the real config.json's field names (linear_num_key_heads /
    # linear_num_value_heads / linear_key_head_dim) weren't being read, only
    # coincidentally-matching hardcoded defaults. That's no longer true --
    # models/qwen3_5.py's Qwen35DecoderLayer.__init__ now tries the real
    # checkpoint's field names FIRST (getattr(config, "linear_num_key_heads",
    # getattr(config, "linear_attn_kq_heads", 16)), and analogously for the
    # other two), falling back to the old names only if neither is present.
    # Verified against the real qwen35_checkpoint/config.json's text_config
    # directly. The (LKH, LVH, LHD) = (16, 32, 128) below are still real
    # checkpoint values, now confirmed actually read, not just coincidental
    # defaults.

    # No assert_all_parameters_initialized here (see tests/test_utils.py):
    # torch.device("meta") allocates no real storage -- this test only reads
    # model.model.layer_types (structural, not numeric), never forward-passes
    # this model.
    with torch.device("meta"):
        model = Qwen35ForCausalLM(hc)

    layer_types = model.model.layer_types
    linear_idx = layer_types.index("linear_attention")
    full_idx = layer_types.index("full_attention")
    print(f"[rank{rank}] layer_types sample: layer {linear_idx}=linear_attention, "
          f"layer {full_idx}=full_attention")

    la = model.model.layers[linear_idx].linear_attn
    LKH, LVH, LHD = 16, 32, 128
    TP = 2
    total_qkv_dim = (LKH + LKH + LVH) * LHD
    correct_local_qkv = total_qkv_dim // TP          # 4096
    buggy_local_qkv = ((LKH // TP) + (LKH // TP) + (LVH // TP)) * LHD // TP  # 2048 (old double-divided)

    print(f"[rank{rank}] in_proj_qkv local shape={tuple(la.in_proj_qkv.weight.shape)} "
          f"(correct={correct_local_qkv}, old-buggy-would-have-been={buggy_local_qkv})")
    assert la.in_proj_qkv.weight.shape[0] == correct_local_qkv, (
        f"in_proj_qkv wrong: got {la.in_proj_qkv.weight.shape[0]}, expected {correct_local_qkv} "
        f"(buggy value would have been {buggy_local_qkv})"
    )

    correct_local_z = (LVH * LHD) // TP  # 2048
    print(f"[rank{rank}] in_proj_z local shape={tuple(la.in_proj_z.weight.shape)} (correct={correct_local_z})")
    assert la.in_proj_z.weight.shape[0] == correct_local_z

    correct_local_a = LVH // TP  # 16
    print(f"[rank{rank}] in_proj_a: class={type(la.in_proj_a).__name__} "
          f"local shape={tuple(la.in_proj_a.weight.shape)} (correct local dim={correct_local_a})")
    assert isinstance(la.in_proj_a, ColumnParallelLinear), \
        f"in_proj_a should be ColumnParallelLinear, got {type(la.in_proj_a).__name__}"
    assert isinstance(la.in_proj_b, ColumnParallelLinear), \
        f"in_proj_b should be ColumnParallelLinear, got {type(la.in_proj_b).__name__}"
    assert la.in_proj_a.weight.shape == (correct_local_a, hc.hidden_size)
    assert la.in_proj_b.weight.shape == (correct_local_a, hc.hidden_size)
    print(f"[rank{rank}] in_proj_a/in_proj_b OK -- now ColumnParallelLinear, local shape "
          f"({correct_local_a}, {hc.hidden_size})")

    correct_local_out_in = (LVH * LHD) // TP  # 2048
    print(f"[rank{rank}] out_proj local shape={tuple(la.out_proj.weight.shape)} "
          f"(correct input dim={correct_local_out_in})")
    assert la.out_proj.weight.shape[1] == correct_local_out_in

    # A_log/dt_bias/qkv_dim/head_expand must be untouched (still local, self.lkh/self.lvh-based).
    assert la.A_log.shape[0] == LVH // TP  # 16
    assert la.qkv_dim == correct_local_qkv  # conv1d sizing target, must equal in_proj_qkv's actual local output
    assert la.conv1d.in_channels == la.qkv_dim
    print(f"[rank{rank}] A_log shape={tuple(la.A_log.shape)}, qkv_dim={la.qkv_dim}, "
          f"conv1d.in_channels={la.conv1d.in_channels} -- all consistent, local, untouched by the fix")

    # Regression: full_attention layer completely unaffected.
    fa = model.model.layers[full_idx].self_attn
    expected_q = (2 * hc.num_attention_heads * getattr(hc, "head_dim", hc.hidden_size // hc.num_attention_heads)) // TP
    print(f"[rank{rank}] REGRESSION full_attention q_proj local shape={tuple(fa.q_proj.weight.shape)} "
          f"(expected {expected_q})")
    assert fa.q_proj.weight.shape[0] == expected_q
    print(f"[rank{rank}] REGRESSION OK -- full_attention layer unaffected")

    dist.destroy_process_group()


def main():
    mp.spawn(worker, args=(), nprocs=2, join=True)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
