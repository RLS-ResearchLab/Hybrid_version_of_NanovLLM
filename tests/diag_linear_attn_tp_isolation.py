"""Isolates Qwen35LinearAttention's layer-0 forward computation to answer:
is the still-broken tp_size=2 divergence a TP-sharding-specific regression,
or does the bug predate TP entirely (wrong even at tp_size=1, unsharded)?

reference_check_phase6_layerwise.py's attn-only-vs-full-layer trace showed
layer 0 (linear_attention) still diverges from HF at tp_size=2 even after
fixing in_proj_qkv's Q/K/V-vs-tp_rank sharding. This script isolates the
SAME layer 0, SAME real checkpoint weights, SAME real input (HF's actual
captured input to that layer, from --phase hf's mixer_io capture -- true
ground truth, not synthetic), and runs it three ways entirely on CPU (no
GPU needed -- Qwen35LinearAttention itself is pure PyTorch ops, no
flash-attn dependency):

  1. tp_size=1 (single rank, unsharded -- the always-correct code path,
     nothing to get wrong about combining ranks).
  2. tp_size=2 (two gloo ranks, real production sharding/weight_loader
     logic, including RowParallelLinear's internal all_reduce in
     out_proj -- both ranks should converge to the same all-reduced
     output).
  3. Both compared against HF's own real captured output for that layer
     (ground truth), not just against each other.

Verdict:
  - tp_size=1 matches HF, tp_size=2 doesn't match tp_size=1: bug is
    TP-sharding-specific (a regression from tonight's TP-support work),
    NOT the base linear-attention math.
  - tp_size=1 does NOT match HF either: the bug predates TP entirely and
    lives in the base linear-attention forward() logic itself (or in
    something upstream of it that's already wrong in the captured input --
    unlikely here since the input is HF's OWN actual tensor, not derived
    from our engine, but noted for completeness).

Prerequisite: tests/reference_check_phase6_layerwise.py --phase hf must
have been run (with mixer_io capture) at least once -- this script reads
its cache, does not re-run the HF model.

Usage: python tests/diag_linear_attn_tp_isolation.py
"""
import json
import os
import sys
import tempfile
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "nanovllm" not in sys.modules:
    _pkg = types.ModuleType("nanovllm")
    _pkg.__path__ = [ROOT]
    _pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = _pkg

CKPT_DIR = os.path.join(ROOT, "qwen35_checkpoint")
CACHE_DIR = os.path.join(ROOT, "tests", "_phase6_cache")
HF_STATES_PATH = os.path.join(CACHE_DIR, "hf_layer_states.pt")

LAYER0_PREFIX = "model.language_model.layers.0.linear_attn."
PARAM_KEYS = [
    "in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight",
    "out_proj.weight", "A_log", "dt_bias", "conv1d.weight", "norm.weight",
]


def _real_dims():
    config = json.load(open(os.path.join(CKPT_DIR, "config.json")))
    tc = config["text_config"]
    return dict(
        hidden_size=tc["hidden_size"],
        rms_norm_eps=tc["rms_norm_eps"],
        ck=tc["linear_conv_kernel_dim"],
        lhd=tc["linear_key_head_dim"],
        lkh=tc["linear_num_key_heads"],
        lvh=tc["linear_num_value_heads"],
    )


def _load_layer0_real_tensors() -> dict[str, torch.Tensor]:
    """Read exactly layer 0's linear_attn tensors straight from the real
    checkpoint shards (via the index's weight_map) -- targeted, doesn't
    touch the other ~8000 tensors in this 40-layer/256-expert checkpoint."""
    index = json.load(open(os.path.join(CKPT_DIR, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    tensors = {}
    for suffix in PARAM_KEYS:
        full_key = LAYER0_PREFIX + suffix
        shard_file = weight_map[full_key]
        with safe_open(os.path.join(CKPT_DIR, shard_file), "pt", "cpu") as f:
            tensors[suffix] = f.get_tensor(full_key)
    return tensors


def _load_into(la, real_tensors: dict[str, torch.Tensor]):
    """Load each real tensor via that parameter's OWN weight_loader --
    same dispatch mechanism utils/loader.py uses in production (including
    this session's in_proj_qkv._qkv_weight_loader fix), not a raw copy_()."""
    from nanovllm.utils.loader import default_weight_loader
    name_map = {
        "in_proj_qkv.weight": la.in_proj_qkv.weight,
        "in_proj_z.weight": la.in_proj_z.weight,
        "in_proj_a.weight": la.in_proj_a.weight,
        "in_proj_b.weight": la.in_proj_b.weight,
        "out_proj.weight": la.out_proj.weight,
        "A_log": la.A_log,
        "dt_bias": la.dt_bias,
        "conv1d.weight": la.conv1d.weight,
        "norm.weight": la.norm.weight,
    }
    for suffix, param in name_map.items():
        loader = getattr(param, "weight_loader", default_weight_loader)
        loader(param, real_tensors[suffix])


def _build_layer(dims, device):
    from nanovllm.models.qwen3_5 import Qwen35LinearAttention
    return Qwen35LinearAttention(
        hidden_size=dims["hidden_size"],
        linear_attn_kq_heads=dims["lkh"],
        linear_attn_v_heads=dims["lvh"],
        linear_attn_head_dim=dims["lhd"],
        conv_kernel_size=dims["ck"],
        rms_norm_eps=dims["rms_norm_eps"],
    ).to(device=device, dtype=torch.bfloat16)


def _use_cuda(world_size: int) -> bool:
    """Mirror engine/model_runner.py's real backend/device choice (nccl +
    torch.cuda.set_device(rank), one real GPU per rank) whenever enough
    GPUs are actually present -- CPU/gloo execution has different bf16
    rounding behavior than real GPU tensor-core execution (most CPU bf16
    ops upcast to fp32 internally; GPU tensor cores don't), which matters
    a lot for a sequential recurrent scan with exp()/softplus() in the
    loop. Falls back to CPU/gloo (with a loud warning) only when there
    genuinely aren't enough GPUs, so this stays runnable as a smoke test
    in CPU-only environments -- but a CPU-only run is NOT a faithful
    reproduction of the real engine's execution path and any agreement or
    disagreement it finds should not be treated as the final answer.
    """
    return torch.cuda.is_available() and torch.cuda.device_count() >= world_size


def run_tp1(real_tensors: dict, hf_input: torch.Tensor) -> torch.Tensor:
    use_cuda = _use_cuda(1)
    backend = "nccl" if use_cuda else "gloo"
    if not use_cuda:
        print("WARNING: running tp1 on CPU/gloo -- not a faithful reproduction of the real "
              "engine's GPU bf16 execution. See _use_cuda's docstring.")
    device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(0)
    dist.init_process_group(backend, init_method="tcp://localhost:29911", rank=0, world_size=1)
    dims = _real_dims()
    la = _build_layer(dims, device)
    _load_into(la, {k: v.to(device) for k, v in real_tensors.items()})
    cu_seqlens = torch.tensor([0, hf_input.shape[0]], dtype=torch.int32, device=device)
    with torch.no_grad():
        out, _, _ = la(hf_input.to(device=device, dtype=torch.bfloat16), cu_seqlens)
    dist.destroy_process_group()
    return out.float().cpu()


def _tp2_worker(rank: int, tmp_path: str, results: list):
    use_cuda = _use_cuda(2)
    backend = "nccl" if use_cuda else "gloo"
    if not use_cuda and rank == 0:
        print("WARNING: running tp2 on CPU/gloo -- not a faithful reproduction of the real "
              "engine's GPU bf16 execution. See _use_cuda's docstring.")
    device = torch.device(f"cuda:{rank}") if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, init_method="tcp://localhost:29912", rank=rank, world_size=2)
    payload = torch.load(tmp_path, weights_only=True, map_location="cpu")
    real_tensors = payload["real_tensors"]
    hf_input = payload["hf_input"]

    dims = _real_dims()
    la = _build_layer(dims, device)
    _load_into(la, {k: v.to(device) for k, v in real_tensors.items()})
    cu_seqlens = torch.tensor([0, hf_input.shape[0]], dtype=torch.int32, device=device)
    with torch.no_grad():
        out, _, _ = la(hf_input.to(device=device, dtype=torch.bfloat16), cu_seqlens)
    results.append((rank, out.float().cpu()))
    dist.destroy_process_group()


def run_tp2(real_tensors: dict, hf_input: torch.Tensor):
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    torch.save({"real_tensors": real_tensors, "hf_input": hf_input}, tmp_path)
    try:
        manager = mp.Manager()
        results = manager.list()
        mp.spawn(_tp2_worker, args=(tmp_path, results), nprocs=2, join=True)
    finally:
        os.unlink(tmp_path)
    by_rank = {rank: out for rank, out in results}
    return by_rank[0], by_rank[1]


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.reshape(-1).float().unsqueeze(0), b.reshape(-1).float().unsqueeze(0)).item()


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def main():
    assert os.path.exists(HF_STATES_PATH), (
        f"missing {HF_STATES_PATH} -- run "
        f"'python tests/reference_check_phase6_layerwise.py --phase hf' first "
        f"(needs the mixer_io capture)"
    )
    hf_payload = torch.load(HF_STATES_PATH, weights_only=True)
    assert "mixer_io" in hf_payload, (
        "hf_layer_states.pt was saved before mixer_io capture was added -- "
        "re-run 'python tests/reference_check_phase6_layerwise.py --phase hf'"
    )
    layer_types = None  # not saved in hf payload; layer 0 is linear_attention per config's schedule
    mixer0 = hf_payload["mixer_io"][0]
    hf_input = mixer0["input"]      # (seq_len, hidden), real HF ground truth
    hf_output = mixer0["output"]    # (seq_len, hidden), real HF ground truth
    print(f"HF ground truth for layer 0: input shape={tuple(hf_input.shape)}, "
          f"output shape={tuple(hf_output.shape)}")

    real_tensors = _load_layer0_real_tensors()
    print(f"Loaded {len(real_tensors)} real layer-0 linear_attn tensors from the checkpoint: "
          f"{sorted(real_tensors.keys())}")

    print("\n--- Running tp_size=1 (unsharded) ---")
    tp1_out = run_tp1(real_tensors, hf_input)
    print(f"tp1 output shape={tuple(tp1_out.shape)}")

    print("\n--- Running tp_size=2 (real production sharding) ---")
    tp2_rank0, tp2_rank1 = run_tp2(real_tensors, hf_input)
    print(f"tp2 rank0 output shape={tuple(tp2_rank0.shape)}, rank1 output shape={tuple(tp2_rank1.shape)}")

    print("\n=== RESULTS ===")
    cos_tp1_vs_hf = _cos(tp1_out, hf_output)
    cos_tp2r0_vs_hf = _cos(tp2_rank0, hf_output)
    cos_tp2r0_vs_tp2r1 = _cos(tp2_rank0, tp2_rank1)
    cos_tp1_vs_tp2 = _cos(tp1_out, tp2_rank0)

    print(f"tp1_out        vs HF ground truth : cosine={cos_tp1_vs_hf:.6f}  max_abs_diff={_max_abs(tp1_out, hf_output):.4e}")
    print(f"tp2_rank0      vs HF ground truth : cosine={cos_tp2r0_vs_hf:.6f}  max_abs_diff={_max_abs(tp2_rank0, hf_output):.4e}")
    print(f"tp2_rank0      vs tp2_rank1        : cosine={cos_tp2r0_vs_tp2r1:.6f}  max_abs_diff={_max_abs(tp2_rank0, tp2_rank1):.4e}")
    print(f"tp1_out        vs tp2_rank0        : cosine={cos_tp1_vs_tp2:.6f}  max_abs_diff={_max_abs(tp1_out, tp2_rank0):.4e}")

    print("\n=== VERDICT ===")
    tp1_ok = cos_tp1_vs_hf >= 0.99
    tp2_ok = cos_tp2r0_vs_hf >= 0.99
    tp2_ranks_agree = cos_tp2r0_vs_tp2r1 >= 0.99

    if not tp2_ranks_agree:
        print("tp2's own two ranks don't even agree with EACH OTHER after out_proj's all_reduce -- "
              "something in tp_size=2 construction/loading is non-deterministic or rank-divergent "
              "in a way that shouldn't be possible if out_proj's all_reduce is working. Investigate "
              "RowParallelLinear.forward's all_reduce call and dtype consistency across ranks first.")
    elif tp1_ok and not tp2_ok:
        print("CASE 3: tp_size=1 matches HF ground truth (cosine >= 0.99) but tp_size=2 does NOT. "
              "The bug is TP-SHARDING-SPECIFIC -- a regression in how tp_size=2 shards/combines "
              "in_proj_qkv/in_proj_a/in_proj_b/conv1d/A_log/dt_bias, or a missing all_reduce "
              "somewhere, NOT the base linear-attention math (which is verified correct at tp_size=1 "
              "against real HF ground truth). This session's in_proj_qkv fix was necessary but not "
              "sufficient -- something else in the tp_size=2 path is still wrong.")
    elif not tp1_ok:
        print("CASE 4: tp_size=1 (unsharded, nothing to get wrong about combining ranks) does NOT "
              "match HF ground truth either. The bug predates TP support entirely and lives in the "
              "base Qwen35LinearAttention.forward() math itself (or its weight-loading for the "
              "non-TP-specific parameters), independent of any sharding logic. Fixing tp_size=2's "
              "sharding won't fix this on its own.")
    else:
        print("Both tp_size=1 and tp_size=2 match HF ground truth here (cosine >= 0.99) -- layer 0's "
              "linear_attn, in isolation with real weights and real HF input, is NOT the source of "
              "the divergence seen in the full engine run. The bug must be introduced by something "
              "surrounding this call in the full model (state/conv_state wiring, cu_seqlens "
              "construction, or a difference between this isolated input and what the real engine's "
              "embedding+input_layernorm actually produces) -- re-check the full-pipeline attn-only "
              "trace's input, not just this layer's own weights/math.")


if __name__ == "__main__":
    main()
