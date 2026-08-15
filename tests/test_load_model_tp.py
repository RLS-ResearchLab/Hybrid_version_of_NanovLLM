"""Same question as tests/test_load_model_ep.py, now for TP: does the real
load_model() actually route Qwen35FullAttention's q/k/v/o and
Qwen35SharedExpert's gate_up_proj/down_proj through their respective
TP-aware weight_loader methods -- or does it silently fall through to
default_weight_loader.

Qwen35LinearAttention's in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/out_proj
are DELIBERATELY EXCLUDED from this test -- while building it, constructing
Qwen35LinearAttention at tp_size=2 surfaced a real, pre-existing
architectural bug independent of the loader (see test_linear_attn_tp_bug()
below for a minimal, standalone reproduction): linear_attn_kq_heads/
linear_attn_v_heads are divided by tp_size BEFORE being passed to
ColumnParallelLinear/RowParallelLinear (which divide by tp_size AGAIN
internally) for in_proj_qkv/in_proj_z/out_proj, and in_proj_a/in_proj_b use
ReplicatedLinear (a straight, unsliced copy) with an already-per-rank size.
Compare to Qwen35FullAttention.q_proj, which correctly passes the FULL
(un-divided) num_heads*head_dim straight through and lets ColumnParallelLinear
divide once. This is a construction-time bug, not a loading bug -- no
checkpoint or loader change can route around it, and it would also break the
forward pass (shape-mismatched matmuls), not just weight loading.

Two independent checks per parameter, both required, for what IS tested:
  1. DISPATCH: default_weight_loader is monkeypatched to record every param
     it's called on. After load_model() runs, none of the TP-aware target
     parameters may appear in that set -- proves load_model() actually found
     and used each parameter's own .weight_loader, not inferred from reading
     LinearBase.__init__ alone.
  2. DATA: each parameter's own weight_loader class is independently
     instantiated fresh ("oracle") and fed the identical full reference
     tensor directly (bypassing load_model() entirely) to compute the
     expected local shard. The REAL model's loaded parameter is compared
     bitwise against this oracle.

Reference tensors are tagged by ROW (ColumnParallelLinear -- shards the
output/row dim) or by COLUMN (RowParallelLinear -- shards the input/column
dim) with distinct constants, so the correct slice is bitwise-checkable and
any off-by-one/wrong-axis bug is detectable.

Usage: python tests/test_load_model_tp.py
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

sys.path.insert(0, os.path.dirname(__file__))
from test_utils import known_zero_initialized_param_names, assert_all_parameters_initialized  # noqa: E402

HIDDEN = 8
NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 4, 2, 4
SHARED_INTERMEDIATE = 4
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
        full_attention_interval=1,  # the only layer is full_attention
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        partial_rotary_factor=0.25,
        rope_theta=10_000_000.0,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        vocab_size=20,
        moe_intermediate_size=2,
        shared_expert_intermediate_size=SHARED_INTERMEDIATE,
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
    full_q = 2 * NUM_HEADS * HEAD_DIM
    full_kv = NUM_KV_HEADS * HEAD_DIM
    full_o_in = NUM_HEADS * HEAD_DIM

    ref = {
        f"{L0}.self_attn.q_proj.weight": _row_tagged(full_q, HIDDEN, 6000.0),
        f"{L0}.self_attn.k_proj.weight": _row_tagged(full_kv, HIDDEN, 7000.0),
        f"{L0}.self_attn.v_proj.weight": _row_tagged(full_kv, HIDDEN, 8000.0),
        f"{L0}.self_attn.o_proj.weight": _col_tagged(HIDDEN, full_o_in, 9000.0),
        f"{L0}.mlp.shared_expert.gate_proj.weight": _row_tagged(SHARED_INTERMEDIATE, HIDDEN, 10000.0),
        f"{L0}.mlp.shared_expert.up_proj.weight": _row_tagged(SHARED_INTERMEDIATE, HIDDEN, 11000.0),
        f"{L0}.mlp.shared_expert.down_proj.weight": _col_tagged(HIDDEN, SHARED_INTERMEDIATE, 12000.0),
    }
    save_file(ref, tmp_path)
    return ref


def worker(rank: int, ckpt_dir: str, ref: dict):
    dist.init_process_group("gloo", init_method="tcp://localhost:29801", rank=rank, world_size=TP_SIZE)
    _stub_attention_module()

    from nanovllm.models.qwen3_5 import Qwen35ForCausalLM
    from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear, MergedColumnParallelLinear
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

    # Guardrail (additive only -- see tests/test_utils.py): confirms
    # load_model() actually populated EVERY parameter, not just the ones
    # this test's DISPATCH/DATA checks below happen to sample -- this
    # checkpoint only contains q/k/v/o_proj + shared_expert tensors, so
    # anything else in the model that load_model() failed to route is
    # exactly the class of bug this guardrail exists to catch.
    assert_all_parameters_initialized(
        model, whitelist_zero=known_zero_initialized_param_names(model)
    )

    loader_mod.default_weight_loader = real_default

    layer0 = model.model.layers[0]
    assert layer0.layer_type == "full_attention"

    targets = {
        "q_proj": layer0.self_attn.q_proj.weight,
        "k_proj": layer0.self_attn.k_proj.weight,
        "v_proj": layer0.self_attn.v_proj.weight,
        "o_proj": layer0.self_attn.o_proj.weight,
        "shared_expert.gate_up_proj": layer0.mlp.shared_expert.gate_up_proj.weight,
        "shared_expert.down_proj": layer0.mlp.shared_expert.down_proj.weight,
    }
    hit_default = [name for name, p in targets.items() if id(p) in called_via_default]
    assert not hit_default, (
        f"[rank{rank}] DISPATCH FAILURE: these TP-aware params were loaded via "
        f"default_weight_loader instead of their own weight_loader: {hit_default}"
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

    check("q_proj", targets["q_proj"],
          ColumnParallelLinear(HIDDEN, 2 * NUM_HEADS * HEAD_DIM, bias=False),
          f"{L0}.self_attn.q_proj.weight")
    check("k_proj", targets["k_proj"],
          ColumnParallelLinear(HIDDEN, NUM_KV_HEADS * HEAD_DIM, bias=False),
          f"{L0}.self_attn.k_proj.weight")
    check("v_proj", targets["v_proj"],
          ColumnParallelLinear(HIDDEN, NUM_KV_HEADS * HEAD_DIM, bias=False),
          f"{L0}.self_attn.v_proj.weight")
    check("o_proj", targets["o_proj"],
          RowParallelLinear(NUM_HEADS * HEAD_DIM, HIDDEN, bias=False),
          f"{L0}.self_attn.o_proj.weight")
    check("shared_expert.down_proj", targets["shared_expert.down_proj"],
          RowParallelLinear(SHARED_INTERMEDIATE, HIDDEN, bias=False),
          f"{L0}.mlp.shared_expert.down_proj.weight")

    oracle_su = MergedColumnParallelLinear(HIDDEN, [SHARED_INTERMEDIATE, SHARED_INTERMEDIATE], bias=False)
    oracle_su.weight.weight_loader(oracle_su.weight, ref[f"{L0}.mlp.shared_expert.gate_proj.weight"], 0)
    oracle_su.weight.weight_loader(oracle_su.weight, ref[f"{L0}.mlp.shared_expert.up_proj.weight"], 1)
    real_su = targets["shared_expert.gate_up_proj"]
    assert torch.equal(real_su.data, oracle_su.weight.data), (
        f"[rank{rank}] DATA MISMATCH on shared_expert.gate_up_proj:\n"
        f"real ={real_su.data.tolist()}\noracle={oracle_su.weight.data.tolist()}"
    )
    print(f"[rank{rank}] DATA check OK -- shared_expert.gate_up_proj: "
          f"shape={tuple(real_su.shape)} bitwise match vs oracle (both shards)")

    dist.destroy_process_group()


def linear_attn_bug_worker(rank: int, results: list):
    """Minimal, standalone reproduction of the Qwen35LinearAttention TP bug
    -- construction only, no loader involved. Confirms it's a construction-
    time bug independent of load_model()."""
    dist.init_process_group("gloo", init_method="tcp://localhost:29802", rank=rank, world_size=TP_SIZE)
    _stub_attention_module()

    from nanovllm.models.qwen3_5 import Qwen35LinearAttention

    LKH, LVH, LHD, CK = 2, 4, 2, 4
    la = Qwen35LinearAttention(
        hidden_size=HIDDEN, linear_attn_kq_heads=LKH, linear_attn_v_heads=LVH,
        linear_attn_head_dim=LHD, conv_kernel_size=CK, rms_norm_eps=1e-6,
    )
    full_qkv_dim = (LKH + LKH + LVH) * LHD  # 16
    expected_correct_local_qkv = full_qkv_dim // TP_SIZE  # 8, if divided once like Qwen35FullAttention
    actual_local_qkv = la.in_proj_qkv.weight.shape[0]
    print(f"[rank{rank}] in_proj_qkv local shape={tuple(la.in_proj_qkv.weight.shape)} "
          f"(rows: correct-if-divided-once={expected_correct_local_qkv}, actual={actual_local_qkv})")
    print(f"[rank{rank}] in_proj_a local shape={tuple(la.in_proj_a.weight.shape)} "
          f"(ReplicatedLinear -- full-copy semantics, but constructed with an already-per-rank size)")

    dist.destroy_process_group()
    results.append((rank, actual_local_qkv, expected_correct_local_qkv))


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.safetensors")
        ref = _build_checkpoint(ckpt_path)
        mp.spawn(worker, args=(tmpdir, ref), nprocs=TP_SIZE, join=True)
    print("\nfull_attention + shared_expert: ALL CHECKS PASSED\n")

    print("=" * 70)
    print("Separate diagnostic: Qwen35LinearAttention TP construction bug")
    print("=" * 70)
    manager = mp.Manager()
    results = manager.list()
    mp.spawn(linear_attn_bug_worker, args=(results,), nprocs=TP_SIZE, join=True)
    for rank, actual, expected in results:
        status = "MATCHES expected (bug not present)" if actual == expected else "BUG CONFIRMED -- double-divided"
        print(f"[rank{rank}] in_proj_qkv local rows: actual={actual} vs correct-if-divided-once={expected} -- {status}")


if __name__ == "__main__":
    main()
