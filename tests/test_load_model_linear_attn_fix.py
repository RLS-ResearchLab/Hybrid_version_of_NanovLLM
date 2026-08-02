"""Re-measures load_model()'s DISPATCH+DATA correctness (same standard as
tests/test_load_model_tp.py's "Part 1") specifically against
Qwen35LinearAttention's five now-fixed parameters: in_proj_qkv, in_proj_z,
in_proj_a, in_proj_b, out_proj.

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

    total_qkv_dim = (LKH + LKH + LVH) * LHD
    check("in_proj_qkv", targets["in_proj_qkv"],
          ColumnParallelLinear(HIDDEN, total_qkv_dim, bias=False),
          f"{L0}.linear_attn.in_proj_qkv.weight")
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

    dist.destroy_process_group()


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.safetensors")
        ref = _build_checkpoint(ckpt_path)
        mp.spawn(worker, args=(tmpdir, ref), nprocs=TP_SIZE, join=True)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
