"""Phase 1 standalone correctness tests for Qwen3.5 hybrid model.

Tests core numerical logic (norms, RoPE, GDR linear attention, MoE routing)
by importing individual modules directly, avoiding the full nanovllm package
import chain which requires flash_attn/triton.

Usage:
    python tests/test_qwen35_standalone.py
"""

import sys
import os
import importlib
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set up import paths: the workspace root is treated as 'nanovllm' package
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# We need 'nanovllm' to resolve as this workspace. Create a path mapping.
# The workspace parent + renaming trick: add parent to sys.path and
# make 'nanovllm' importable by creating a package redirect.
_PARENT = os.path.dirname(ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# Map 'nanovllm' to our workspace directory name
_WS_NAME = os.path.basename(ROOT)
if _WS_NAME != "nanovllm":
    # Create a virtual package mapping
    import types
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    # Don't execute __init__.py (it imports LLM which triggers flash_attn chain)
    sys.modules["nanovllm"] = nanovllm_pkg


def init_dist():
    """Initialize torch.distributed for single-process usage."""
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("gloo")


def load_reference_module():
    """Import src/model_small_qwen3.5.py despite the dot in the filename."""
    spec = importlib.util.spec_from_file_location(
        "model_small_qwen35",
        os.path.join(ROOT, "src", "model_small_qwen3.5.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_small_config():
    """Config matching src/model_small_qwen3.5.py constants."""
    from types import SimpleNamespace
    return SimpleNamespace(
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=128,
        partial_rotary_factor=0.25,
        rope_theta=10_000_000.0,
        max_position_embeddings=4096,
        vocab_size=248320,
        rms_norm_eps=1e-6,
        full_attention_interval=4,
        linear_attn_kq_heads=8,
        linear_attn_v_heads=16,
        linear_attn_head_dim=64,
        conv_kernel_size=4,
        intermediate_size=256,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        num_experts=32,
        num_experts_per_tok=4,
        tie_word_embeddings=False,
        hidden_act="silu",
        dtype=torch.bfloat16,
    )

def cosine_sim(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()


def topk_overlap(a, b, k=5):
    a_topk = set(a.float().topk(k).indices.tolist())
    b_topk = set(b.float().topk(k).indices.tolist())
    return len(a_topk & b_topk) / k

def test_conv_state_shape_invariant(device):
    """new_conv must always be exactly (B, QKV, CK-1), even for T < CK-1
    on the very first call with no prior conv_state."""
    config = make_small_config()
    from nanovllm.models.qwen3_5 import Qwen35LinearAttention

    la = Qwen35LinearAttention(
        hidden_size=config.hidden_size,
        linear_attn_kq_heads=config.linear_attn_kq_heads,
        linear_attn_v_heads=config.linear_attn_v_heads,
        linear_attn_head_dim=config.linear_attn_head_dim,
        conv_kernel_size=config.conv_kernel_size,
        rms_norm_eps=config.rms_norm_eps,
    ).to(device).to(torch.bfloat16)

    x_one_token = torch.randn(1, 1, config.hidden_size, device=device, dtype=torch.bfloat16)
    _, _, new_conv = la(x_one_token, state=None, conv_state=None)

    expected_len = config.conv_kernel_size - 1
    assert new_conv.shape[2] == expected_len, (
        f"conv_state invariant broken: got length {new_conv.shape[2]}, "
        f"expected {expected_len}"
    )
    print(f"  [PASS] conv_state shape invariant holds for T=1, no prior state: {new_conv.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Norm Variants
# ═══════════════════════════════════════════════════════════════════════════════


def test_norm_variants(device):
    """Test Qwen35RMSNorm and Qwen35RMSNormGated against reference."""
    print("\n" + "=" * 70)
    print("TEST 1: Norm Variants vs Reference")
    print("=" * 70)

    from nanovllm.layers.layernorm import Qwen35RMSNorm, Qwen35RMSNormGated
    ref_mod = load_reference_module()

    torch.manual_seed(789)
    d = 512

    # Qwen35RMSNorm vs reference RMSNorm
    nv_norm = Qwen35RMSNorm(d).to(device).to(torch.bfloat16)
    ref_norm = ref_mod.RMSNorm(d).to(device).to(torch.bfloat16)
    ref_norm.weight.data.copy_(nv_norm.weight.data)

    x = torch.randn(2, 16, d, device=device, dtype=torch.bfloat16)
    y_nv = nv_norm(x)
    y_ref = ref_norm(x)
    cos = cosine_sim(y_nv, y_ref)
    print(f"  RMSNorm cosine: {cos:.6f}")
    assert cos > 0.9999, f"RMSNorm mismatch: {cos}"

    # Qwen35RMSNormGated vs reference RMSNormGated
    nv_gnorm = Qwen35RMSNormGated(64).to(device).to(torch.bfloat16)
    ref_gnorm = ref_mod.RMSNormGated(64).to(device).to(torch.bfloat16)
    ref_gnorm.weight.data.copy_(nv_gnorm.weight.data)

    x2 = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    gate = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    y_nv2 = nv_gnorm(x2, gate)
    y_ref2 = ref_gnorm(x2, gate)
    cos2 = cosine_sim(y_nv2, y_ref2)
    print(f"  RMSNormGated cosine: {cos2:.6f}")
    assert cos2 > 0.9999, f"RMSNormGated mismatch: {cos2}"

    print("  [PASS]")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Fused Residual Norm
# ═══════════════════════════════════════════════════════════════════════════════


def test_fused_residual_norm(device):
    """Fused add_rms_forward vs separate add + norm."""
    print("\n" + "=" * 70)
    print("TEST 2: Fused Residual Norm")
    print("=" * 70)

    from nanovllm.layers.layernorm import Qwen35RMSNorm

    torch.manual_seed(789)
    norm = Qwen35RMSNorm(512).to(device).to(torch.bfloat16)

    x = torch.randn(2, 16, 512, device=device, dtype=torch.bfloat16)
    residual = torch.randn(2, 16, 512, device=device, dtype=torch.bfloat16)

    y_fused, res_fused = norm.add_rms_forward(x.clone(), residual.clone())

    combined = (x.float() + residual.float()).to(torch.bfloat16)
    y_separate = norm.rms_forward(combined)

    cos_out = cosine_sim(y_separate, y_fused)
    cos_res = cosine_sim(combined, res_fused)
    print(f"  Output cosine: {cos_out:.6f}")
    print(f"  Residual cosine: {cos_res:.6f}")
    assert cos_out > 0.999 and cos_res > 0.999
    print("  [PASS]")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: RoPE Frequency Base
# ═══════════════════════════════════════════════════════════════════════════════


def test_rope_frequency_base(device):
    """Assert partial RoPE uses rotary_dim for frequency base."""
    print("\n" + "=" * 70)
    print("TEST 3: RoPE Frequency Base")
    print("=" * 70)

    from nanovllm.layers.rotary_embedding import PartialRotaryEmbedding

    head_size, rotary_dim, base = 128, 32, 10_000_000.0
    rope = PartialRotaryEmbedding(head_size, rotary_dim, 1024, base).to(device)

    cache = rope.cos_sin_cache
    assert cache.shape[-1] == rotary_dim

    # Verify frequencies match reference build_rope
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
    freqs = torch.outer(torch.arange(1024, dtype=torch.float), inv_freq)
    cos_expected = freqs.cos()
    cos_actual = cache[:, 0, :rotary_dim // 2].cpu()
    max_diff = (cos_actual - cos_expected).abs().max().item()
    print(f"  Max frequency diff: {max_diff:.2e}")
    assert max_diff < 1e-5

    # Test passthrough
    torch.manual_seed(42)
    q = torch.randn(4, 8, head_size, device=device)
    k = torch.randn(4, 1, head_size, device=device)
    positions = torch.arange(4, device=device)
    q_rot, k_rot = rope(positions, q, k)

    q_pass_diff = (q_rot[..., rotary_dim:] - q[..., rotary_dim:]).abs().max().item()
    q_rot_diff = (q_rot[..., :rotary_dim] - q[..., :rotary_dim]).abs().max().item()
    print(f"  Passthrough diff: {q_pass_diff:.2e} (should be ~0)")
    print(f"  Rotary diff: {q_rot_diff:.4f} (should be nonzero)")
    assert q_pass_diff < 1e-5
    assert q_rot_diff > 0.01
    print("  [PASS]")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: RoPE vs Reference
# ═══════════════════════════════════════════════════════════════════════════════


def test_rope_vs_reference(device):
    """Compare PartialRotaryEmbedding against reference apply_rope."""
    print("\n" + "=" * 70)
    print("TEST 4: RoPE vs Reference")
    print("=" * 70)

    from nanovllm.layers.rotary_embedding import PartialRotaryEmbedding
    ref_mod = load_reference_module()

    head_size, rotary_dim, base, T = 128, 32, 10_000_000.0, 8
    rope_nv = PartialRotaryEmbedding(head_size, rotary_dim, 1024, base).to(device)

    ref_cos, ref_sin = ref_mod.build_rope(T + 64, device)

    torch.manual_seed(77)
    B, NQ, NKV = 1, 8, 1
    q = torch.randn(B, NQ, T, head_size, device=device)
    k = torch.randn(B, NKV, T, head_size, device=device)

    # Reference: (B, NH, T, DH) format
    q_ref, k_ref = ref_mod.apply_rope(q, k, ref_cos, ref_sin)

    # nano-vLLM: (N, NH, DH) format with positions
    q_flat = q.permute(0, 2, 1, 3).reshape(B * T, NQ, head_size)
    k_flat = k.permute(0, 2, 1, 3).reshape(B * T, NKV, head_size)
    positions = torch.arange(T, device=device).repeat(B)
    q_nv, k_nv = rope_nv(positions, q_flat, k_flat)
    q_nv = q_nv.reshape(B, T, NQ, head_size).permute(0, 2, 1, 3)
    k_nv = k_nv.reshape(B, T, NKV, head_size).permute(0, 2, 1, 3)

    cos_q = cosine_sim(q_ref, q_nv)
    cos_k = cosine_sim(k_ref, k_nv)
    print(f"  Q cosine: {cos_q:.6f}")
    print(f"  K cosine: {cos_k:.6f}")
    assert cos_q > 0.999
    assert cos_k > 0.999
    print("  [PASS]")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: Linear Attention (GDR) vs Reference
# ═══════════════════════════════════════════════════════════════════════════════


def test_linear_attention_vs_reference(device):
    """Test Qwen35LinearAttention GDR scan against reference LinearAttn."""
    print("\n" + "=" * 70)
    print("TEST 5: Linear Attention (GDR) vs Reference")
    print("=" * 70)

    ref_mod = load_reference_module()
    config = make_small_config()

    torch.manual_seed(42)
    ref_la = ref_mod.LinearAttn().to(device).to(torch.bfloat16)

    from nanovllm.models.qwen3_5 import Qwen35LinearAttention
    nv_la = Qwen35LinearAttention(
        hidden_size=config.hidden_size,
        linear_attn_kq_heads=config.linear_attn_kq_heads,
        linear_attn_v_heads=config.linear_attn_v_heads,
        linear_attn_head_dim=config.linear_attn_head_dim,
        conv_kernel_size=config.conv_kernel_size,
        rms_norm_eps=config.rms_norm_eps,
    ).to(device).to(torch.bfloat16)

    # Copy weights
    ref_sd = dict(ref_la.named_parameters())
    copied, missed = 0, []
    for name, param in nv_la.named_parameters():
        if name in ref_sd and param.shape == ref_sd[name].shape:
            param.data.copy_(ref_sd[name].data)
            copied += 1
        else:
            missed.append(f"{name}: {param.shape}")
    print(f"  Copied {copied} params, missed: {missed}")

    torch.manual_seed(99)
    B, T, H = 1, 8, config.hidden_size
    x = torch.randn(B, T, H, device=device, dtype=torch.bfloat16)
    x_packed = x[0]  # nv_la expects packed (N, H), not (B, T, H)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

    with torch.no_grad():
        y_ref, state_ref, conv_ref = ref_la(x)
        y_nv, state_nv, conv_nv = nv_la(x_packed, cu_seqlens)

    cos_y = cosine_sim(y_ref, y_nv)
    cos_state = cosine_sim(state_ref, state_nv)
    print(f"  Output cosine: {cos_y:.6f}")
    print(f"  State cosine: {cos_state:.6f}")
    assert cos_y > 0.99, f"Output mismatch: {cos_y}"
    assert cos_state > 0.99, f"State mismatch: {cos_state}"
    print("  [PASS]")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: Linear Attention Incremental Consistency
# ═══════════════════════════════════════════════════════════════════════════════


def test_linear_attention_incremental(device):
    """Single-shot vs token-by-token incremental for GDR."""
    print("\n" + "=" * 70)
    print("TEST 6: Linear Attention Incremental Consistency")
    print("=" * 70)

    config = make_small_config()
    from nanovllm.models.qwen3_5 import Qwen35LinearAttention

    torch.manual_seed(42)
    la = Qwen35LinearAttention(
        hidden_size=config.hidden_size,
        linear_attn_kq_heads=config.linear_attn_kq_heads,
        linear_attn_v_heads=config.linear_attn_v_heads,
        linear_attn_head_dim=config.linear_attn_head_dim,
        conv_kernel_size=config.conv_kernel_size,
        rms_norm_eps=config.rms_norm_eps,
    ).to(device).to(torch.bfloat16)

    
    ref_mod = load_reference_module()
    ref_la = ref_mod.LinearAttn().to(device).to(torch.bfloat16)
    ref_sd = dict(ref_la.named_parameters())
    for name, param in la.named_parameters():
        if name in ref_sd and param.shape == ref_sd[name].shape:
            param.data.copy_(ref_sd[name].data)

    torch.manual_seed(456)
    B, T, H = 1, 10, config.hidden_size
    x = torch.randn(B, T, H, device=device, dtype=torch.bfloat16)
    x_packed = x[0]  # la expects packed (N, H), not (B, T, H)
    cu_full = torch.tensor([0, T], dtype=torch.int32, device=device)
    cu_step = torch.tensor([0, 1], dtype=torch.int32, device=device)

    with torch.no_grad():
        y_full, state_full, conv_full = la(x_packed, cu_full)

        state, conv_st = None, None
        ys = []
        for t in range(T):
            y_t, state, conv_st = la(x_packed[t:t+1], cu_step, states=state, conv_states=conv_st)
            ys.append(y_t)
        y_inc = torch.cat(ys, dim=0)

    cos_all = cosine_sim(y_full, y_inc)
    cos_last = cosine_sim(y_full[-1, :], y_inc[-1, :])
    cos_state = cosine_sim(state_full, state)
    top1 = y_full[-1, :].float().argmax().item() == y_inc[-1, :].float().argmax().item()

    print(f"  All tokens cosine: {cos_all:.6f}")
    print(f"  Last token cosine: {cos_last:.6f}")
    print(f"  State cosine: {cos_state:.6f}")
    print(f"  Top-1 match: {top1}")
    assert cos_last > 0.999, f"Last token mismatch: {cos_last}"
    assert cos_state > 0.999, f"State mismatch: {cos_state}"
    print("  [PASS]")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: MoE FFN vs Reference
# ═══════════════════════════════════════════════════════════════════════════════


def test_moe_vs_reference(device):
    """Test Qwen35MoE routing against reference MoEFFN."""
    print("\n" + "=" * 70)
    print("TEST 7: MoE FFN vs Reference")
    print("=" * 70)

    ref_mod = load_reference_module()
    config = make_small_config()

    torch.manual_seed(42)
    ref_moe = ref_mod.MoEFFN().to(device).to(torch.bfloat16)

    # FIXED (was: KNOWN FAILURE below) -- see init_reference_moe_experts_'s
    # docstring in test_utils.py (also reused by test_qwen35_full_model.py
    # and make_fake_checkpoint.py, all three hitting this exact issue):
    # src/model_small_qwen3.5.py's Experts never initializes
    # gate_up_proj/down_proj, so ref_moe.experts.{gate_up_proj,down_proj}
    # were reading back all-zero straight out of construction, and the copy
    # loop below faithfully carried that zero into nv_moe. Fixed here on the
    # constructed instance (src/model_small_qwen3.5.py stays untouched, per
    # project rules) rather than via the centralized
    # init_model_weights_with_norms_, which isinstance-checks against
    # nanovllm's own layernorm classes and would silently zero this
    # reference module's RMSNormGated.weight instead (not applicable here:
    # MoEFFN has no norms of its own, but init_reference_moe_experts_ is
    # shared across all three call sites, so it stays the uniformly-safe
    # choice).
    from test_utils import init_reference_moe_experts_
    init_reference_moe_experts_(ref_moe, seed=42)

    from nanovllm.models.qwen3_5 import Qwen35MoE
    nv_moe = Qwen35MoE(
        hidden_size=config.hidden_size,
        intermediate_size=config.moe_intermediate_size,
        shared_intermediate_size=config.shared_expert_intermediate_size,
        num_experts=config.num_experts,
        top_k=config.num_experts_per_tok,
    ).to(device).to(torch.bfloat16)

    # Copy weights (handle gate_up merge)
    ref_sd = dict(ref_moe.named_parameters())
    for nv_name, nv_param in nv_moe.named_parameters():
        if "shared_expert.gate_up_proj.weight" in nv_name:
            gate_n = nv_name.replace("gate_up_proj", "gate_proj")
            up_n = nv_name.replace("gate_up_proj", "up_proj")
            if gate_n in ref_sd and up_n in ref_sd:
                nv_param.data.copy_(torch.cat([ref_sd[gate_n].data, ref_sd[up_n].data], dim=0))
                continue
        if nv_name in ref_sd and nv_param.shape == ref_sd[nv_name].shape:
            nv_param.data.copy_(ref_sd[nv_name].data)

    # Guardrail (additive only -- see tests/test_utils.py): this copy loop
    # is a per-name/per-shape match with no "missed" tracking at all (unlike
    # test_qwen35_full_model.py's copy_weights_to_port) -- any nv_moe
    # parameter that doesn't find a matching ref_sd entry is silently left
    # as torch.empty() garbage and the forward pass below would run on it
    # without any signal. Catches exactly that gap.
    #
    # FIXED (was: KNOWN FAILURE) -- root cause was NOT this copy loop, it
    # was upstream in src/model_small_qwen3.5.py's own Experts class (never
    # explicitly initialized anywhere in the reference), which this port's
    # own Experts class mirrors exactly. Fixed above via
    # init_reference_moe_experts_(ref_moe) before this copy loop runs, so
    # nv_moe now receives real values instead of faithfully-copied zero.
    # This test's cosine now reflects the actual per-expert
    # gate_up_proj/down_proj matmul agreement, not just the shared_expert +
    # gating/dispatch-bookkeeping path (see printed result below for the
    # current number vs. the pre-fix degenerate-zero pass).
    from test_utils import known_zero_initialized_param_names, assert_all_parameters_initialized
    assert_all_parameters_initialized(
        nv_moe, whitelist_zero=known_zero_initialized_param_names(nv_moe)
    )

    torch.manual_seed(99)
    B, T, H = 1, 6, config.hidden_size
    x = torch.randn(B, T, H, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        y_ref = ref_moe(x)
        y_nv = nv_moe(x)

    cos = cosine_sim(y_ref, y_nv)
    print(f"  MoE output cosine: {cos:.6f}")
    assert cos > 0.99, f"MoE mismatch: {cos}"
    print("  [PASS]")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8: Reference Model Incremental Consistency
# ═══════════════════════════════════════════════════════════════════════════════


def test_reference_incremental(device):
    """Reference model single-shot vs incremental to validate the reference itself."""
    print("\n" + "=" * 70)
    print("TEST 8: Reference Model Incremental Consistency")
    print("=" * 70)

    ref_mod = load_reference_module()
    L = 8  # num_hidden_layers

    torch.manual_seed(42)
    ref_model = ref_mod.Qwen35MoESmall().to(device).to(torch.bfloat16)

    torch.manual_seed(123)
    T = 12
    input_ids = torch.randint(100, 5000, (1, T), device=device)

    with torch.no_grad():
        logits_full, _, _, _ = ref_model(input_ids)
        logits_full_last = logits_full[0, -1, :]

        kvs = [None] * L
        states = [None] * L
        convs = [None] * L
        for t in range(T):
            logits_t, kvs, states, convs = ref_model(
                input_ids[:, t:t+1], kvs=kvs, states=states, convs=convs
            )
        logits_inc_last = logits_t[0, -1, :]

    cos = cosine_sim(logits_full_last, logits_inc_last)
    top1 = logits_full_last.float().argmax().item() == logits_inc_last.float().argmax().item()
    print(f"  cosine={cos:.6f}, top1_match={top1}")
    assert cos > 0.999, f"Reference consistency failed: {cos}"
    print("  [PASS]")


# ═══════════════════════════════════════════════════════════════════════════════


def main():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Qwen3.5 Phase 1 Tests — device={device}")
    print("=" * 70)

    init_dist()

    test_norm_variants(device)
    test_fused_residual_norm(device)
    test_rope_frequency_base(device)
    test_rope_vs_reference(device)
    test_linear_attention_vs_reference(device)
    test_linear_attention_incremental(device)
    test_moe_vs_reference(device)
    test_reference_incremental(device)

    print("\n" + "=" * 70)
    print("ALL PHASE 1 TESTS PASSED")
    print("=" * 70)

    import torch.distributed as dist
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
