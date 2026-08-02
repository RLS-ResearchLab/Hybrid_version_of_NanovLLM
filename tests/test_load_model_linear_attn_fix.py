"""Re-measures load_model()'s DISPATCH+DATA correctness (same standard as
tests/test_load_model_tp.py's "Part 1") specifically against
Qwen35LinearAttention's eight now-fixed parameters: in_proj_qkv, in_proj_z,
in_proj_a, in_proj_b, out_proj, A_log, dt_bias, conv1d.weight.

A_log/dt_bias/conv1d.weight were added after real GPU server runs caught
two rounds of the same bug class: raw parameters (A_log/dt_bias are our own
nn.Parameter; conv1d.weight belongs to an nn.Conv1d, PyTorch's own internal
parameter) with no LinearBase wrapper, so no free weight_loader, sized
locally but with no weight_loader to slice the checkpoint's full-sized
tensor down -- default_weight_loader's unsliced copy_() crashed on genuine
shape mismatches (32 vs 16, then 8192 vs 4096) the first two times this ran
against the real checkpoint. Both fixed via the same explicit
_tp_dim0_weight_loader, bound the same way LinearBase binds its own.

The earlier pass in test_load_model_tp.py validated load_model()'s dispatch
logic against parameters that were, at the time, WRONGLY shaped (the
double-division / ReplicatedLinear bug) -- that's a different claim from
validating it against the CORRECT shapes now that the fix is in. Measured
fresh here, not assumed from the earlier pass.

Same two independent checks as before:
  1. DISPATCH: default_weight_loader must never be called for these five.
  2. DATA: bitwise match against an independently-instantiated oracle of the
     same (now-correct) class, fed the identical full reference tensor.

Small synthetic config (tp_size=2): linear_attn_kq_heads=2,
linear_attn_v_heads=4, linear_attn_head_dim=2 -- same values used in the
standalone bug reproduction, now expected to load correctly instead of
reproducing the bug.

Usage: python tests/test_load_model_linear_attn_fix.py
"""
import os
import sys
import tempfile
import types
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "nanovllm" not in sys.modules:
    _pkg = types.ModuleType("nanovllm")
    _pkg.__path__ = [ROOT]
    _pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = _pkg

HIDDEN = 8
LKH, LVH, LHD, CK = 2, 4, 2, 4
TP_SIZE = 2
L0 = "model.language_model.layers.0"


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


def _small_config():
    return SimpleNamespace(
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        full_attention_interval=1000,  # huge -> the only layer is linear_attention
        linear_attn_kq_heads=LKH,
        linear_attn_v_heads=LVH,
        linear_attn_head_dim=LHD,
        conv_kernel_size=CK,
        rms_norm_eps=1e-6,
        vocab_size=20,
        moe_intermediate_size=2,
        shared_expert_intermediate_size=4,
        num_experts=2,
        num_experts_per_tok=1,
        tie_word_embeddings=False,
    )


def _row_tagged(rows: int, cols: int, base: float) -> torch.Tensor:
    t = torch.empty(rows, cols)
    for i in range(rows):
        t[i, :] = float(base + i)
    return t


def _vec_tagged(n: int, base: float) -> torch.Tensor:
    return torch.tensor([float(base + i) for i in range(n)])


def _conv_tagged(channels: int, kernel_size: int, base: float) -> torch.Tensor:
    t = torch.empty(channels, 1, kernel_size)
    for c in range(channels):
        t[c, :, :] = float(base + c)
    return t


def _col_tagged(rows: int, cols: int, base: float) -> torch.Tensor:
    t = torch.empty(rows, cols)
    for j in range(cols):
        t[:, j] = float(base + j)
    return t


def _build_checkpoint(tmp_path: str):
    total_qkv_dim = (LKH + LKH + LVH) * LHD   # 16
    total_z_dim = LVH * LHD                    # 8
    total_out_in = LVH * LHD                   # 8

    ref = {
        f"{L0}.linear_attn.in_proj_qkv.weight": _row_tagged(total_qkv_dim, HIDDEN, 1000.0),
        f"{L0}.linear_attn.in_proj_z.weight": _row_tagged(total_z_dim, HIDDEN, 2000.0),
        f"{L0}.linear_attn.in_proj_a.weight": _row_tagged(LVH, HIDDEN, 3000.0),
        f"{L0}.linear_attn.in_proj_b.weight": _row_tagged(LVH, HIDDEN, 4000.0),
        f"{L0}.linear_attn.out_proj.weight": _col_tagged(HIDDEN, total_out_in, 5000.0),
        f"{L0}.linear_attn.A_log": _vec_tagged(LVH, 6000.0),
        f"{L0}.linear_attn.dt_bias": _vec_tagged(LVH, 7000.0),
        f"{L0}.linear_attn.conv1d.weight": _conv_tagged(total_qkv_dim, CK, 8000.0),
    }
    save_file(ref, tmp_path)
    return ref


def worker(rank: int, ckpt_dir: str, ref: dict):
    dist.init_process_group("gloo", init_method="tcp://localhost:29803", rank=rank, world_size=TP_SIZE)
    _stub_attention_module()

    from nanovllm.models.qwen3_5 import Qwen35ForCausalLM
    from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear
    import nanovllm.utils.loader as loader_mod

    called_via_default = []
    real_default = loader_mod.default_weight_loader

    def _tracking_default(param, loaded_weight):
        called_via_default.append(id(param))
        return real_default(param, loaded_weight)

    loader_mod.default_weight_loader = _tracking_default

    config = _small_config()
    model = Qwen35ForCausalLM(config)
    loader_mod.load_model(model, ckpt_dir)

    loader_mod.default_weight_loader = real_default

    layer0 = model.model.layers[0]
    assert layer0.layer_type == "linear_attention"
    la = layer0.linear_attn

    targets = {
        "in_proj_qkv": la.in_proj_qkv.weight,
        "in_proj_z": la.in_proj_z.weight,
        "in_proj_a": la.in_proj_a.weight,
        "in_proj_b": la.in_proj_b.weight,
        "out_proj": la.out_proj.weight,
        "A_log": la.A_log,
        "dt_bias": la.dt_bias,
        "conv1d.weight": la.conv1d.weight,
    }
    hit_default = [name for name, p in targets.items() if id(p) in called_via_default]
    assert not hit_default, (
        f"[rank{rank}] DISPATCH FAILURE: these params were loaded via default_weight_loader "
        f"instead of their own weight_loader: {hit_default}"
    )
    print(f"[rank{rank}] DISPATCH check OK -- none of {list(targets.keys())} "
          f"went through default_weight_loader")

    def check(name, real_param, oracle_module, ref_key):
        full_ref = ref[ref_key]
        oracle_param = oracle_module.weight
        oracle_param.weight_loader(oracle_param, full_ref)
        assert torch.equal(real_param.data, oracle_param.data), (
            f"[rank{rank}] DATA MISMATCH on {name}: real shape={tuple(real_param.shape)} "
            f"oracle shape={tuple(oracle_param.shape)}\n"
            f"real ={real_param.data.tolist()}\noracle={oracle_param.data.tolist()}"
        )
        print(f"[rank{rank}] DATA check OK -- {name}: shape={tuple(real_param.shape)} bitwise match vs oracle")

    # in_proj_qkv: hand-computed Q/K/V-split-then-shard, independent of the
    # method under test (same pattern as A_log/dt_bias/conv1d.weight below)
    # -- NOT a same-class "oracle" built from plain ColumnParallelLinear's
    # own default weight_loader. That oracle would itself reproduce the
    # Q/K/V-boundary-crossing bug _qkv_weight_loader exists to avoid: a
    # naive contiguous dim-0 chunk of the merged [Q_full|K_full|V_full]
    # checkpoint tensor crosses section boundaries at tp_size>1, silently
    # feeding e.g. rank 0 all of Q_full+K_full and none of V_full instead of
    # each rank's own Q/K/V head slice.
    full_qkv = ref[f"{L0}.linear_attn.in_proj_qkv.weight"]
    q_full, k_full, v_full = full_qkv.split([LKH * LHD, LKH * LHD, LVH * LHD], dim=0)
    q_local = q_full.chunk(TP_SIZE, dim=0)[rank]
    k_local = k_full.chunk(TP_SIZE, dim=0)[rank]
    v_local = v_full.chunk(TP_SIZE, dim=0)[rank]
    expected_qkv = torch.cat([q_local, k_local, v_local], dim=0)
    real_qkv = targets["in_proj_qkv"]
    assert torch.equal(real_qkv.data, expected_qkv), (
        f"[rank{rank}] DATA MISMATCH on in_proj_qkv: real shape={tuple(real_qkv.shape)} "
        f"expected shape={tuple(expected_qkv.shape)}\n"
        f"real ={real_qkv.data.tolist()}\nexpected={expected_qkv.tolist()}"
    )
    print(f"[rank{rank}] DATA check OK -- in_proj_qkv: shape={tuple(real_qkv.shape)} "
          f"bitwise match vs hand-computed Q/K/V-split-then-shard slice")

    check("in_proj_z", targets["in_proj_z"],
          ColumnParallelLinear(HIDDEN, LVH * LHD, bias=False),
          f"{L0}.linear_attn.in_proj_z.weight")
    check("in_proj_a", targets["in_proj_a"],
          ColumnParallelLinear(HIDDEN, LVH, bias=False),
          f"{L0}.linear_attn.in_proj_a.weight")
    check("in_proj_b", targets["in_proj_b"],
          ColumnParallelLinear(HIDDEN, LVH, bias=False),
          f"{L0}.linear_attn.in_proj_b.weight")
    check("out_proj", targets["out_proj"],
          RowParallelLinear(LVH * LHD, HIDDEN, bias=False),
          f"{L0}.linear_attn.out_proj.weight")

    # A_log/dt_bias: independent hand-computed expected slice (not reusing
    # _tp_head_weight_loader itself, for a check that doesn't depend on the
    # method under test being correct in order to validate it).
    shard_size = LVH // TP_SIZE
    start = rank * shard_size
    for name, ref_key in (("A_log", f"{L0}.linear_attn.A_log"), ("dt_bias", f"{L0}.linear_attn.dt_bias")):
        expected = ref[ref_key][start:start + shard_size]
        real_param = targets[name]
        assert torch.equal(real_param.data, expected), (
            f"[rank{rank}] DATA MISMATCH on {name}: real={real_param.data.tolist()} "
            f"expected={expected.tolist()}"
        )
        print(f"[rank{rank}] DATA check OK -- {name}: shape={tuple(real_param.shape)} "
              f"bitwise match vs hand-computed slice [{start}:{start+shard_size}]")

    # conv1d.weight: depthwise (groups=qkv_dim), so channel c's kernel must
    # match whatever in_proj_qkv's output channel c actually is for this
    # rank -- the SAME Q/K/V-split-then-shard as in_proj_qkv above, NOT a
    # naive single contiguous dim-0 chunk of the qkv_dim channel space (that
    # would pair each rank's Q/K/V-reassembled in_proj_qkv output channels
    # with a completely different set of channels' conv kernels).
    full_conv = ref[f"{L0}.linear_attn.conv1d.weight"]
    qc_full, kc_full, vc_full = full_conv.split([LKH * LHD, LKH * LHD, LVH * LHD], dim=0)
    qc_local = qc_full.chunk(TP_SIZE, dim=0)[rank]
    kc_local = kc_full.chunk(TP_SIZE, dim=0)[rank]
    vc_local = vc_full.chunk(TP_SIZE, dim=0)[rank]
    expected_conv = torch.cat([qc_local, kc_local, vc_local], dim=0)
    real_conv = targets["conv1d.weight"]
    assert torch.equal(real_conv.data, expected_conv), (
        f"[rank{rank}] DATA MISMATCH on conv1d.weight: real shape={tuple(real_conv.shape)} "
        f"expected shape={tuple(expected_conv.shape)}"
    )
    print(f"[rank{rank}] DATA check OK -- conv1d.weight: shape={tuple(real_conv.shape)} "
          f"bitwise match vs hand-computed Q/K/V-split-then-shard slice")

    dist.destroy_process_group()


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.safetensors")
        ref = _build_checkpoint(ckpt_path)
        mp.spawn(worker, args=(tmpdir, ref), nprocs=TP_SIZE, join=True)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
