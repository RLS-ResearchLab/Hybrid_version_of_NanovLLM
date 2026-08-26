"""Qwen3.5 hybrid model for nano-vLLM.

Hybrid architecture: GDR linear-attention + GQA full-attention + MoE FFN.
Config-driven — reads all hyperparameters from hf_config, no hardcoded constants.
Numerically matches src/model_small_qwen3.5.py (the ground-truth reference).

Layer schedule:
    - Default: layer i is full-attention if (i+1) % full_attention_interval == 0
    - If config.layers_block_type is present, use that list directly
"""

import math
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

# dequantize_weight_int8_grouped lives under tests/ (moe_int8_quantize.py), not on any
# production sys.path -- every real invocation this session (src/server.py,
# bench_throughput.py run directly, or `from nanovllm.models.qwen3_5 import ...` from
# an arbitrary caller) only happened to work today because it was launched via a
# tests/*.py wrapper script, which puts tests/ on sys.path[0] as a side effect of how
# `python tests/foo.py` resolves the running script's own directory. Outside that
# accident, this top-level, unconditional import raised `ModuleNotFoundError:
# No module named 'moe_int8_quantize'` before any model could even be constructed --
# reproduced directly against src/server.py's own sys.path setup. Guard explicitly
# instead of depending on the caller's launch method.
_TESTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from moe_int8_quantize import dequantize_weight_int8_grouped

# Fused-Triton-kernel MoE path -- an alternative to the gather+dequantize+
# einsum sequence below, validated in isolation first (cosine=0.999988,
# 4.71x on the full gate_up+silu+down_proj pattern,
# layers/smoke_test_full_moe_pipeline.py, 2026-08-21) but NOT yet validated
# for CUDA graph capture, real EP combine, or real-checkpoint GSM8K
# correctness -- see _USE_FUSED_MOE_KERNEL's own comment below for exactly
# what's still open. Flag-gated via an env var rather than a proper Config
# field while still experimental; promote to Config once trusted.
from nanovllm.layers.fused_moe_int8 import fused_moe_int8_forward
_USE_FUSED_MOE_KERNEL = os.environ.get("NANOVLLM_USE_FUSED_MOE_KERNEL", "0") == "1"

# True W8A8 Hopper fused kernel -- Phase 1/2 of w8a8_activation_quant_scoping_memo.md,
# written 2026-08-22. UNVALIDATED end-to-end: moe_w8a8.cu has never compiled or run
# (no CUDA toolchain on this dev machine, confirmed empirically). This wiring is
# written blind, same discipline the tp=1 INT8 fix used, so it's ready to validate the
# moment Hopper GPU access exists rather than written only once the window opens. Do
# NOT trust NANOVLLM_USE_MOE_W8A8_HOPPER=1 to produce correct output until Phase 0
# (compile moe_w8a8.cu + run layers/smoke_test_moe_w8a8_hopper.py) and the full Phase 3
# validation chain (H200_test_day_checklist.md) have actually run on real hardware.
# Importing these modules is safe without CUDA -- the JIT compile in
# fused_moe_w8a8_hopper.py fires lazily inside fused_moe_w8a8_hopper_forward's first
# real call, not at import time, same lazy-compile pattern this file's other
# CUDA-only imports already rely on.
from nanovllm.layers.moe_align_block_size import moe_align_block_size
from nanovllm.layers.moe_w8a8_hopper_quantize import (
    quantize_activation_fp8_dynamic,
    dequantize_weight_fp8_grouped_gathered,
)
from nanovllm.layers.fused_moe_w8a8_hopper import fused_moe_w8a8_hopper_forward
_USE_MOE_W8A8_HOPPER = os.environ.get("NANOVLLM_USE_MOE_W8A8_HOPPER", "0") == "1"
# moe_w8a8.h: block_n/warp_n must be exactly (32,8) or (64,4) -- any other pair
# silently no-ops the kernel launch instead of erroring (see that file's own contract
# comment). block_m/stages default to the same "one safe config, autotune later, only
# after correctness is established" values fused_moe_w8a8_hopper_forward's own
# docstring already uses -- not chosen from any measurement, there isn't one yet.
_W8A8_HOPPER_BLOCK_M = 32
_W8A8_HOPPER_BLOCK_N = 32
_W8A8_HOPPER_WARP_N = 8
_W8A8_HOPPER_STAGES = 2

from nanovllm.layers.layernorm import Qwen35RMSNorm, Qwen35RMSNormGated
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
    MergedColumnParallelLinear,
    local_num_kv_heads,
    kv_head_replica_source,
)
from nanovllm.layers.rotary_embedding import get_partial_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


def _is_full_attention(layer_idx: int, full_attention_interval: int) -> bool:
    """Default layer-type schedule matching src/model_small_qwen3.5.py."""
    return (layer_idx + 1) % full_attention_interval == 0


def _get_layer_types(config, num_layers: int) -> list[str]:
    """Determine layer types from config. Data-driven if available, else modulo rule."""
    # Prefer explicit per-layer type list (Jamba/Zamba-style). Real
    # checkpoints call this field "layer_types", not "layers_block_type" --
    # confirmed no test fixture anywhere in the repo uses the old name, so
    # this is a straight rename, not a fallback chain.
    layers_block_type = getattr(config, "layer_types", None)
    if layers_block_type is not None:
        assert len(layers_block_type) == num_layers
        return list(layers_block_type)
    # Fall back to periodic modulo rule
    fai = getattr(config, "full_attention_interval", 4)
    return [
        "full_attention" if _is_full_attention(i, fai) else "linear_attention"
        for i in range(num_layers)
    ]


def l2norm(x, dim=-1, eps=1e-6):
    """L2 normalize along dim, matching src/model_small_qwen3.5.py."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


# ─── Full Attention (GQA with gated output) ────────────────────────────────────


class Qwen35FullAttention(nn.Module):
    """GQA attention with gated output and partial RoPE.

    q_proj emits 2 * num_q_heads * head_dim (query + gate interleaved per head).
    Output is gated: o = o * sigmoid(gate) before o_proj.
    QK-RMSNorm uses the (1+w) variant.
    Partial RoPE: only first rotary_dim channels of each head are rotated.

    Matches src/model_small_qwen3.5.py FullAttn exactly.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_dim: int,
        max_position: int,
        rope_theta: float,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        tp_rank = dist.get_rank()
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.total_num_heads = num_heads
        assert num_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = local_num_kv_heads(num_kv_heads, tp_size)
        self.head_dim = head_dim

        # q_proj outputs 2 * NQ * DH: query + gate interleaved per head
        self.q_proj = ColumnParallelLinear(
            hidden_size, 2 * num_heads * head_dim, bias=False
        )
        if num_kv_heads >= tp_size:
            # Unchanged from before local_num_kv_heads existed -- byte-for-byte
            # identical construction and weight-loading path whenever there are
            # enough physical kv heads to shard normally (every tp=1/tp=2 case
            # against the real checkpoint's num_key_value_heads=2 lands here).
            self.k_proj = ColumnParallelLinear(
                hidden_size, num_kv_heads * head_dim, bias=False
            )
            self.v_proj = ColumnParallelLinear(
                hidden_size, num_kv_heads * head_dim, bias=False
            )
        else:
            # GQA kv-head replication: too few physical kv heads to give every
            # rank a whole one (e.g. num_key_value_heads=2 over tp_size=4) --
            # replicate a full kv head onto each rank that needs it instead of
            # splitting one across them, same technique vLLM and other engines
            # use for this exact num_kv_heads < tp_size case.
            #
            # ReplicatedLinear, not ColumnParallelLinear: ColumnParallelLinear's
            # constructor divides output_size by tp_size internally, assuming
            # sharding. self.num_kv_heads * head_dim here is already the correct
            # LOCAL (per-rank, single replicated head) size -- passing it through
            # ColumnParallelLinear would divide it AGAIN, the exact double-
            # division bug this file already hit once for Qwen35LinearAttention's
            # in_proj_qkv (see that class's __init__ comment). ReplicatedLinear's
            # constructor does not divide at all, which is what's needed here.
            #
            # weight_loader is overridden afterward because ReplicatedLinear's
            # default (a straight, unsliced copy) would try to copy the FULL
            # multi-head checkpoint tensor into a single-head-sized local
            # parameter -- see _kv_replicate_weight_loader below.
            self.k_proj = ReplicatedLinear(
                hidden_size, self.num_kv_heads * head_dim, bias=False
            )
            self.v_proj = ReplicatedLinear(
                hidden_size, self.num_kv_heads * head_dim, bias=False
            )
            self.k_proj.weight.weight_loader = self._kv_replicate_weight_loader
            self.v_proj.weight.weight_loader = self._kv_replicate_weight_loader
        self.o_proj = RowParallelLinear(
            num_heads * head_dim, hidden_size, bias=False
        )

        # QK-RMSNorm with (1+w) variant
        self.q_norm = Qwen35RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(head_dim, eps=rms_norm_eps)

        # Partial rotary embedding
        self.rotary_emb = get_partial_rope(
            head_dim, rotary_dim, max_position, rope_theta
        )

        self.scaling = head_dim ** -0.5
        # Lazy import to avoid requiring flash_attn/triton at module import time
        from nanovllm.layers.attention import Attention
        self.attn = Attention(
            self.num_heads, head_dim, self.scaling, self.num_kv_heads
        )

    def _kv_replicate_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Bound to k_proj.weight/v_proj.weight only when total_num_kv_heads
        < tp_size (see __init__). loaded_weight is the checkpoint's FULL
        (total_num_kv_heads * head_dim, hidden_size) tensor; this rank
        replicates exactly ONE whole physical kv head (kv_head_replica_source
        picks which one) rather than narrowing a fraction of one -- narrowing
        here would reproduce the "half a head" bug the old
        `assert num_kv_heads % tp_size == 0` used to guard against."""
        head_dim = self.head_dim
        source_head = kv_head_replica_source(self.total_num_kv_heads, self.tp_size, self.tp_rank)
        start = source_head * head_dim
        param.data.copy_(loaded_weight.narrow(0, start, head_dim))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # q_proj output: (N, 2 * num_heads * head_dim / tp)
        q_full = self.q_proj(hidden_states)
        # Split into query and gate: view as (N, num_heads, head_dim * 2)
        q_full = q_full.view(-1, self.num_heads, self.head_dim * 2)
        q, gate = q_full.chunk(2, dim=-1)  # each (N, num_heads, head_dim)

        k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)

        # QK-RMSNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Partial RoPE
        q, k = self.rotary_emb(positions, q, k)

        # FlashAttention with paged KV cache
        o = self.attn(q, k, v)  # (N, 1, num_heads, head_dim) or (N, num_heads, head_dim)

        # Gated output: o * sigmoid(gate)
        # o from attn is (N, num_heads, head_dim) during prefill, (N, 1, num_heads, head_dim) during decode
        o_flat = o.reshape(-1, self.num_heads * self.head_dim)
        gate_flat = gate.reshape(-1, self.num_heads * self.head_dim)
        o_flat = o_flat * torch.sigmoid(gate_flat)

        return self.o_proj(o_flat)


# ─── Linear Attention (Gated Delta Rule) ────────────────────────────────────────


class Qwen35LinearAttention(nn.Module):
    """GDR (Gated Delta Rule) linear attention layer.

    Causal depthwise conv1d → SiLU → split Q/K/V → head-expand K/Q →
    L2-normalize → sequential delta-rule scan in float32.

    Accepts optional (state, conv_state) and returns updated versions.
    Works correctly for both full-sequence and single-token (decode) calls.

    Matches src/model_small_qwen3.5.py LinearAttn exactly.
    """

    def __init__(
        self,
        hidden_size: int,
        linear_attn_kq_heads: int,   # LKH
        linear_attn_v_heads: int,    # LVH
        linear_attn_head_dim: int,   # LHD
        conv_kernel_size: int,       # CK
        rms_norm_eps: float,
        use_fused_gdr_kernel: bool = False,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        tp_rank = dist.get_rank()

        # QLLM Stage 2 intervention: replace the O(T) sequential Python scan
        # below with flash-linear-attention's chunked gated-delta-rule kernel
        # for prefill (T_i > 1 segments). Decode (T_i == 1) always keeps the
        # sequential path -- see forward()'s use of is_decode_shape below.
        # Default False so nothing changes unless explicitly opted in; the
        # sequential scan remains fully intact as the flag-off path and as
        # ground truth for correctness comparisons.
        self.use_fused_gdr_kernel = use_fused_gdr_kernel
        self._fla_chunk_gated_delta_rule = None
        if use_fused_gdr_kernel:
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            except ImportError as e:
                raise ImportError(
                    "use_fused_gdr_kernel=True requires the 'flash-linear-attention' "
                    "package (import name 'fla') and a CUDA GPU with Triton support. "
                    "Install it or leave use_fused_gdr_kernel=False to use the "
                    "existing sequential GDR scan."
                ) from e
            self._fla_chunk_gated_delta_rule = chunk_gated_delta_rule

        assert linear_attn_kq_heads % tp_size == 0
        assert linear_attn_v_heads % tp_size == 0
        self.total_lkh = linear_attn_kq_heads
        self.total_lvh = linear_attn_v_heads
        self.lkh = linear_attn_kq_heads // tp_size   # per-rank head counts
        self.lvh = linear_attn_v_heads // tp_size
        self.lhd = linear_attn_head_dim
        self.ck = conv_kernel_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        # QKV dim = (LKH + LKH + LVH) * LHD -- LOCAL (per-rank) width. Consumed
        # by conv1d below and by forward()'s slicing (lkh/lvh unpacked
        # directly there) -- must stay local-valued, do not change this line.
        self.qkv_dim = (self.lkh + self.lkh + self.lvh) * self.lhd

        # Projections. ColumnParallelLinear/RowParallelLinear divide their
        # size argument by tp_size internally (see Qwen35FullAttention.q_proj
        # for the correct reference pattern: full, undivided size in). These
        # five constructor sites previously passed the already-local
        # self.lkh/self.lvh-derived value, causing a double division -- use
        # total_lkh/total_lvh (undivided) here instead; self.lkh/self.lvh
        # remain local and unchanged everywhere else (conv1d, A_log, dt_bias,
        # head_expand, forward()'s slicing).
        total_qkv_dim = (self.total_lkh + self.total_lkh + self.total_lvh) * self.lhd
        self.in_proj_qkv = ColumnParallelLinear(hidden_size, total_qkv_dim, bias=False)
        # ColumnParallelLinear's default weight_loader takes ONE naive
        # contiguous dim-0 chunk of the full output range. That's correct
        # for in_proj_z/in_proj_a/in_proj_b (each a single homogeneous
        # head-group) but WRONG here: the checkpoint stores in_proj_qkv as
        # one merged [Q_full(total_lkh*lhd) | K_full(total_lkh*lhd) |
        # V_full(total_lvh*lhd)] tensor, and forward() slices each rank's
        # LOCAL weight assuming [Q_local | K_local | V_local] layout (see
        # forward()'s q/k/v split below). A naive contiguous chunk crosses
        # the Q/K/V boundaries at tp_size>1 (e.g. tp_size=2 with this
        # checkpoint's 16/16/32-head split: rank 0's chunk is all of
        # Q_full+K_full and none of V_full; rank 1's is all of V_full and
        # none of Q/K) -- silently cross-wiring which weight rows forward()
        # treats as Q vs K vs V. Split Q/K/V first, then shard each by
        # tp_rank, matching QKVParallelLinear's approach for the
        # non-merged-single-tensor case.
        self.in_proj_qkv.weight.weight_loader = self._qkv_weight_loader
        self.in_proj_z = ColumnParallelLinear(hidden_size, self.total_lvh * self.lhd, bias=False)
        # A and B project to per-head scalars -- LOCAL output (broadcast
        # against the per-rank A_log/dt_bias inside the local delta-rule
        # scan; no cross-rank communication anywhere in forward() before
        # out_proj). ColumnParallelLinear (sharded by head, matching
        # in_proj_qkv/in_proj_z), not ReplicatedLinear (full/identical copy
        # -- wrong contract for a per-head-sharded quantity, and unable to
        # load a real checkpoint's full-size tensor without erroring).
        self.in_proj_a = ColumnParallelLinear(hidden_size, self.total_lvh, bias=False)
        self.in_proj_b = ColumnParallelLinear(hidden_size, self.total_lvh, bias=False)

        # Depthwise causal conv1d. groups=qkv_dim -> 1-to-1 per channel, so
        # this operates on exactly whatever channels in_proj_qkv produced
        # locally for this rank. nn.Conv1d creates its own .weight
        # internally (not our nn.Parameter(...) call) -- easy to miss in an
        # audit that only searches for explicit nn.Parameter(...) sites.
        # Real checkpoints store the FULL (total_qkv_dim, 1, kernel_size)
        # tensor in the SAME [Q_full|K_full|V_full] channel order
        # in_proj_qkv.weight's rows are in (conv1d is depthwise, so channel
        # c's kernel must match whatever in_proj_qkv's output channel c
        # actually is for THIS rank) -- must use the same _qkv_weight_loader
        # split-then-shard-then-concat as in_proj_qkv, not a naive
        # single-chunk loader: a plain contiguous dim-0 chunk here would
        # pair each rank's (now correctly Q/K/V-reassembled) in_proj_qkv
        # output channels with a completely different set of channels'
        # conv kernels. (_qkv_weight_loader's split/chunk/cat calls are all
        # dim=0, so they apply unchanged to this tensor's extra trailing
        # (1, kernel_size) dims.)
        self.conv1d = nn.Conv1d(
            self.qkv_dim, self.qkv_dim, conv_kernel_size,
            groups=self.qkv_dim, padding=conv_kernel_size - 1, bias=False
        )
        self.conv1d.weight.weight_loader = self._qkv_weight_loader

        # Per-head parameters for GDR gating. Raw nn.Parameter, no TP-linear
        # wrapper, so -- unlike in_proj_qkv/in_proj_z/in_proj_a/in_proj_b --
        # they don't get a weight_loader for free from LinearBase. Real
        # checkpoints store the FULL (total_lvh,)-sized tensor; each rank
        # needs the SAME contiguous row-range as in_proj_a/in_proj_b's
        # ColumnParallelLinear sharding, so head h's A_log/dt_bias lands on
        # the same rank as head h's a/b gating value (forward() broadcasts
        # them together). Bind an explicit weight_loader, same mechanism
        # LinearBase uses (self.weight.weight_loader = self.weight_loader).
        self.A_log = nn.Parameter(torch.zeros(self.lvh))
        self.dt_bias = nn.Parameter(torch.zeros(self.lvh))
        self.A_log.weight_loader = self._tp_dim0_weight_loader
        self.dt_bias.weight_loader = self._tp_dim0_weight_loader

        # Gated RMSNorm on output
        self.norm = Qwen35RMSNormGated(self.lhd, eps=rms_norm_eps)

        # Output projection
        self.out_proj = RowParallelLinear(self.total_lvh * self.lhd, hidden_size, bias=False)

        # Head expansion ratio (LKH → LVH)
        assert self.lvh % self.lkh == 0
        self.head_expand = self.lvh // self.lkh

    def _qkv_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Split the checkpoint's merged [Q_full | K_full | V_full] tensor
        into its three sections first, shard each section by tp_rank
        independently, then re-concatenate -- so the local weight ends up
        [Q_local | K_local | V_local], matching forward()'s slicing."""
        lhd = self.lhd
        q_full, k_full, v_full = loaded_weight.split(
            [self.total_lkh * lhd, self.total_lkh * lhd, self.total_lvh * lhd], dim=0
        )
        q_local = q_full.chunk(self.tp_size, dim=0)[self.tp_rank]
        k_local = k_full.chunk(self.tp_size, dim=0)[self.tp_rank]
        v_local = v_full.chunk(self.tp_size, dim=0)[self.tp_rank]
        param.data.copy_(torch.cat([q_local, k_local, v_local], dim=0))

    def _tp_dim0_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Shard along dim 0 by tp_rank -- same contiguous chunking as
        ColumnParallelLinear, applied to raw parameters (A_log, dt_bias,
        conv1d.weight) that have no LinearBase wrapper of their own but need
        to land on the same rank as the qkv/a/b channels they pair with."""
        shard_size = param.shape[0]
        start = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(0, start, shard_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        states: torch.Tensor | None = None,
        conv_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (N, H) packed across all segments (sequences) in this call
            cu_seqlens: (num_segments + 1,) int tensor, e.g. [0, 5, 14, 20] for
                three segments of length 5, 9, 6. During decode, every segment
                has length 1, i.e. cu_seqlens = [0, 1, 2, ..., num_segments].
            states: (num_segments, LVH, LHD, LHD) recurrent state per segment, or None
            conv_states: (num_segments, QKV, CK-1) conv history per segment, or None

        Returns:
            output: (N, H)
            new_states: (num_segments, LVH, LHD, LHD) detached
            new_conv_states: (num_segments, QKV, CK-1) detached
        """
        assert hidden_states.ndim == 2, "expects packed (N, H), not (B, T, H)"
        N, _ = hidden_states.shape
        num_segments = cu_seqlens.numel() - 1
        lkh, lvh, lhd, ck = self.lkh, self.lvh, self.lhd, self.ck
        dt = hidden_states.dtype

        is_decode_shape = (num_segments == N)

        # ── Project once over the whole packed N — pure per-token ops ──
        z = self.in_proj_z(hidden_states)        # (N, LVH*LHD)
        a = self.in_proj_a(hidden_states)        # (N, LVH)
        b = self.in_proj_b(hidden_states)        # (N, LVH)
        qkv = self.in_proj_qkv(hidden_states)    # (N, QKV)

        # Fused kernel is only used for real multi-token prefill segments --
        # decode (every segment T_i==1) keeps the sequential path unconditionally.
        # Rationale (see investigation report): at T_i==1 the chunked kernel
        # degenerates to one step per segment with no chunking benefit, while
        # still paying a Triton kernel-launch/dispatch cost the plain tensor
        # ops below don't have -- and decode is the latency-critical path, so
        # we don't want to risk regressing it while only prefill throughput
        # was the target of this optimization.
        #
        # ALSO LOAD-BEARING FOR CUDA GRAPH CAPTURE, NOT JUST PERFORMANCE:
        # engine/model_runner.py's capture_cudagraph() only ever captures the
        # DECODE step (is_decode_shape=True here, since its cu_seqlens_q is
        # always arange(0, bs+1)). That is currently safe only because THIS
        # line guarantees decode never takes the fused branch --
        # fla.chunk_gated_delta_rule (self._fla_chunk_gated_delta_rule, called
        # from _fused_gdr_scan below) is NOT capturable inside
        # torch.cuda.graph(): reproduced directly, with proper warmup before
        # capture, as cudaErrorStreamCaptureInvalidated. Do not flip decode to
        # the fused path, and do not extend graph capture to prefill, without
        # first re-confirming this constraint against whatever fla version is
        # installed at the time -- it may have been fixed upstream, but it is
        # not safe to assume so.
        use_fused = self.use_fused_gdr_kernel and not is_decode_shape

        y_chunks, new_states, new_conv_states = [], [], []
        fused_q, fused_k, fused_v, fused_g, fused_beta = [], [], [], [], []
        fused_seg_z, fused_S0 = [], []

        for i in range(num_segments):
            if is_decode_shape:
                start, end = i, i + 1
            else:
                start = int(cu_seqlens[i])
                end = int(cu_seqlens[i + 1])
            T_i = end - start

            seg_qkv = qkv[start:end].unsqueeze(0)      # (1, T_i, QKV)
            seg_z = z[start:end]                        # (T_i, LVH*LHD)
            seg_a = a[start:end].unsqueeze(0)           # (1, T_i, LVH)
            seg_b = b[start:end].unsqueeze(0)           # (1, T_i, LVH)

            seg_state = states[i:i + 1] if states is not None else None
            seg_conv_state = conv_states[i:i + 1] if conv_states is not None else None

            # ── Causal conv1d, scoped to this segment only ──
            qkv_t = seg_qkv.transpose(1, 2)  # (1, QKV, T_i)
            if seg_conv_state is not None and T_i == 1:
                combined = torch.cat([seg_conv_state, qkv_t], dim=2)
                seg_new_conv = combined[:, :, -(ck - 1):].detach()
                qkv_conv = F.conv1d(
                    combined, self.conv1d.weight, self.conv1d.bias,
                    padding=0, groups=self.qkv_dim
                ).transpose(1, 2)
            else:
                if seg_conv_state is not None:
                    qkv_t = torch.cat([seg_conv_state, qkv_t], dim=2)
                qkv_conv = self.conv1d(qkv_t)
                seg_new_conv = qkv_t[:, :, -(ck - 1):].detach()
                pad_needed = (ck - 1) - seg_new_conv.shape[2]
                if pad_needed > 0:
                    seg_new_conv = F.pad(seg_new_conv, (pad_needed, 0))
                offset = seg_conv_state.shape[2] if seg_conv_state is not None else 0
                qkv_conv = qkv_conv[:, :, offset:offset + T_i].transpose(1, 2)

            qkv_conv = F.silu(qkv_conv)  # (1, T_i, QKV)

            # ── Split Q/K/V, head-expand, L2-norm — per-token, safe as-is ──
            q = qkv_conv[:, :, :lkh * lhd].view(1, T_i, lkh, lhd)
            k = qkv_conv[:, :, lkh * lhd:2 * lkh * lhd].view(1, T_i, lkh, lhd)
            v = qkv_conv[:, :, 2 * lkh * lhd:].view(1, T_i, lvh, lhd)

            g = -self.A_log.float().exp() * F.softplus(seg_a.float() + self.dt_bias.float())
            beta = seg_b.sigmoid()

            q = q.repeat_interleave(self.head_expand, dim=2)
            k = k.repeat_interleave(self.head_expand, dim=2)
            scale = lhd ** -0.5
            q = l2norm(q.float()) * scale
            k = l2norm(k.float())

            g = g.float()
            beta = beta.float()
            v = v.float()

            if seg_state is None:
                S = torch.zeros(1, lvh, lhd, lhd, device=hidden_states.device, dtype=torch.float32)
            else:
                S = seg_state.float()

            new_conv_states.append(seg_new_conv.squeeze(0))

            if use_fused:
                # Defer the actual recurrence to a single batched kernel call
                # after this loop -- stash this segment's prepared (post-conv,
                # post-L2-norm, float32) tensors instead of scanning here.
                fused_q.append(q)
                fused_k.append(k)
                fused_v.append(v)
                fused_g.append(g)
                fused_beta.append(beta)
                fused_seg_z.append(seg_z)
                fused_S0.append(S)
                continue

            # ── Sequential scan — identical math to before, scoped to this segment ──
            ys = []
            for t in range(T_i):
                g_t = g[:, t, :].exp().unsqueeze(-1).unsqueeze(-1)
                beta_t = beta[:, t, :].unsqueeze(-1)
                k_t = k[:, t, :, :]
                v_t = v[:, t, :, :]
                q_t = q[:, t, :, :]
                S = S * g_t
                kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
                delta = (v_t - kv_mem) * beta_t
                S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
                ys.append((S * q_t.unsqueeze(-1)).sum(dim=-2))

            seg_new_state = S.detach()  # (1, LVH, LHD, LHD)
            seg_y = torch.stack(ys, dim=1).to(dt)  # (1, T_i, LVH, LHD)

            seg_y = seg_y.reshape(T_i * lvh, lhd)
            seg_z_flat = seg_z.reshape(T_i * lvh, lhd)
            seg_y = self.norm(seg_y, seg_z_flat)
            seg_y = seg_y.reshape(T_i, lvh * lhd)

            y_chunks.append(seg_y)
            new_states.append(seg_new_state.squeeze(0))

        if use_fused:
            y_chunks, new_states = self._fused_gdr_scan(
                fused_q, fused_k, fused_v, fused_g, fused_beta,
                fused_seg_z, fused_S0, cu_seqlens, dt, lvh, lhd,
            )

        y = torch.cat(y_chunks, dim=0)               # (N, LVH*LHD)
        new_states = torch.stack(new_states, dim=0)          # (num_segments, LVH, LHD, LHD)
        new_conv_states = torch.stack(new_conv_states, dim=0)  # (num_segments, QKV, CK-1)

        return self.out_proj(y), new_states, new_conv_states

    def _fused_gdr_scan(self, q_list, k_list, v_list, g_list, beta_list,
                         seg_z_list, S0_list, cu_seqlens, dt, lvh, lhd):
        """QLLM Stage 2: replace the per-segment sequential delta-rule scan
        with one call to flash-linear-attention's chunked gated-delta-rule
        kernel across the whole packed prefill batch.

        Verified against fla==0.5.2's actual installed source
        (.venv/Lib/site-packages/fla/ops/gated_delta_rule/chunk.py) -- NOT
        via inspect.signature (blocked: fla.ops requires `triton`, which has
        no Windows wheel at all, confirmed by a failed `pip install triton`
        on this machine) but by reading chunk_gated_delta_rule's actual
        def/docstring directly, plus reading the cumsum and output kernels
        it calls to settle a dtype question. Four points vs. the original
        recollection-based draft, in order of severity:

        1. (correctness bug, would NOT have been caught by the cosine/argmax
           test) `scale` was never passed, so it defaulted to
           `k.shape[-1] ** -0.5` = lhd**-0.5 *inside* the kernel -- but q
           passed in is ALREADY pre-scaled by lhd**-0.5 upstream (same
           convention as the sequential path). That's scale applied twice:
           output would come out uniformly scaled by lhd**-1 instead of
           lhd**-0.5. Because q's scale only ever affects the final
           `S @ q_t` readout (never the state-update recurrence itself),
           this multiplies every element of this layer's y by the same
           positive constant -- and Qwen35RMSNormGated (self.norm below)
           divides by the RMS of its input per token, which exactly cancels
           a uniform positive multiplicative factor. So this bug would have
           produced ~1.0 cosine similarity AND identical argmax against the
           sequential path while being numerically wrong -- exactly the
           "passes small-context comparison, silently wrong" failure mode
           the original task warned about for GDR/MoE. Fixed by passing
           `scale=1.0` explicitly (q already carries the scale).
        2. `cu_seqlens` must be `torch.LongTensor` (int64) per the function's
           own type hint and its docstring's example
           (`q.new_tensor([...], dtype=torch.long)`) -- was being cast to
           int32 here (that's FlashAttention's convention, not fla's;
           wrong library's convention carried over). Fixed.
        3. q/k/v/beta/initial_state cast down to `dt` (bf16) immediately
           before the call -- matching the docstring's only shown calling
           convention -- then the returned final_state is upcast back to
           float32 before storing into new_states, to preserve the existing
           state-buffer dtype contract the rest of this class (and
           StateManager) relies on.
        4. `g` (log-decay) is deliberately KEPT in float32, NOT cast down
           with the rest -- this preserves the original stated design
           (matching the sequential scan's float32-accumulation discipline
           for exactly this quantity) rather than quietly abandoning it,
           and it is empirically justified, not a guess: the kernel does
           NOT require uniform dtype across its tensor arguments.
             - `chunk_gated_delta_rule_fwd` (chunk.py:62-69) runs `g`
               through `chunk_local_cumsum`, whose `output_dtype` parameter
               DEFAULTS to `torch.float` (fla/ops/utils/cumsum.py:453) --
               i.e. the library's own default is to upcast g's cumulative
               sum to fp32 internally regardless of what dtype g arrives
               as. Feeding fp32 g in loses nothing there and matches what
               the kernel does by default anyway.
             - `chunk_fwd_kernel_o` (fla/ops/common/chunk_o.py:43-140), the
               Triton kernel that combines g with q/k/v/h to produce the
               output, loads each tensor through its OWN independent
               pointer (`b_q = tl.load(p_q, ...)`, `b_g = tl.load(p_g,
               ...)`, etc. -- chunk_o.py:106-122) with no dtype-equality
               check between them, and accumulates in a hard-coded
               `tl.float32` buffer (`b_o = tl.zeros([BT, BV],
               dtype=tl.float32)`, chunk_o.py:88) regardless of input
               dtypes. Mixed dtype across q/k/v/g is therefore a supported
               pattern at the kernel level, not an unverified assumption.
           Residual caveat: this traces the cumsum stage and the final
           output-combination kernel specifically (the two places g is
           actually consumed on the path this call takes with
           use_gate_in_kernel=False); it does not exhaustively trace every
           internal kernel fla ships. Should still be empirically
           re-confirmed on real GPU hardware -- see the test file's
           dedicated g-dtype sensitivity check at T=1024, which forces a
           bf16-g fused call as a diagnostic comparison specifically
           because compounding decay error is most visible at longer T
           (this project's own precedent: Phase 6's short-prompt tests
           passed while longer-generation tests exposed real problems, so
           a T=8-only check here would not be trustworthy evidence).

        Inputs are the SAME per-segment, post-conv, post-repeat_interleave,
        post-L2-norm tensors the sequential path already computes (q
        pre-scaled by lhd**-0.5, matching the manual loop exactly) -- this
        function only replaces the recurrence itself, not the
        projections/conv/normalization around it.
        """
        # (1, T_i, LVH, LHD)/(1, T_i, LVH) per segment -> (1, N, LVH, LHD)/(1, N, LVH)
        # q/k/v/beta cast down to model dtype -- see point 3 above.
        # g deliberately kept float32 -- see point 4 above; do not cast it.
        q = torch.cat(q_list, dim=1).to(dt)
        k = torch.cat(k_list, dim=1).to(dt)
        v = torch.cat(v_list, dim=1).to(dt)
        g = torch.cat(g_list, dim=1)
        beta = torch.cat(beta_list, dim=1).to(dt)
        # (1, LVH, LHD, LHD) per segment -> (num_segments, LVH, LHD, LHD)
        initial_state = torch.cat(S0_list, dim=0).to(dt)

        # fla's own type hint (`cu_seqlens: torch.LongTensor`) and its
        # docstring example both use int64 -- NOT int32 (that's
        # FlashAttention's convention). See point 2 above.
        cu = cu_seqlens.to(dtype=torch.int64, device=q.device)

        # use_qk_l2norm_in_kernel deliberately left False -- q/k are already
        # L2-normalized above, exactly like the sequential path. Do not also
        # ask the kernel to normalize, or k gets normalized twice.
        #
        # scale=1.0 explicitly -- q already carries the lhd**-0.5 scale from
        # upstream; letting the kernel apply its own default
        # (k.shape[-1]**-0.5) on top would double-scale. See point 1 above.
        #
        # NOT SAFE UNDER torch.cuda.graph() CAPTURE -- reproduced directly
        # (proper warmup, then capture) as cudaErrorStreamCaptureInvalidated.
        # Currently never an issue: forward()'s `use_fused` gate above
        # guarantees this call is only reached for prefill, and
        # engine/model_runner.py's capture_cudagraph() only ever captures
        # decode. See the comment on `use_fused` above for the full
        # constraint -- do not graph-capture prefill, or route decode through
        # this call, without re-verifying this against the installed fla
        # version first.
        o, final_state = self._fla_chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=1.0,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=False,
        )
        # o: (1, N, LVH, LHD) in model dtype (fla returns o.to(q.dtype));
        # final_state: (num_segments, LVH, LHD, LHD) in model dtype --
        # upcast to float32 to match the sequential path's state-buffer
        # dtype contract (see point 3 above).

        o = o.to(dt).squeeze(0)  # (N, LVH, LHD)
        new_states = [final_state[i].float().detach() for i in range(final_state.shape[0])]

        y_chunks = []
        for i in range(len(seg_z_list)):
            start, end = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
            T_i = end - start
            seg_y = o[start:end].reshape(T_i * lvh, lhd)
            seg_z_flat = seg_z_list[i].reshape(T_i * lvh, lhd)
            seg_y = self.norm(seg_y, seg_z_flat)
            y_chunks.append(seg_y.reshape(T_i, lvh * lhd))

        return y_chunks, new_states

# ─── MoE FFN ────────────────────────────────────────────────────────────────────


class Experts(nn.Module):
    """Batched expert parameters for MoE.

    Stores gate_up_proj and down_proj as batched tensors (not per-expert nn.Linear).
    Matches src/model_small_qwen3.5.py Experts class.

    Shapes:
        gate_up_proj: (num_experts, 2 * intermediate_size, hidden_size)
        down_proj:    (num_experts, hidden_size, intermediate_size)
    """

    def __init__(
        self,
        num_experts: int,
        intermediate_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )


class Qwen35SharedExpert(nn.Module):
    """Shared expert with SwiGLU activation.

    Matches src/model_small_qwen3.5.py SharedExpert.
    Uses MergedColumnParallelLinear for gate+up (TP-ready).
    """

    def __init__(
        self,
        hidden_size: int,
        shared_intermediate_size: int,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [shared_intermediate_size, shared_intermediate_size], bias=False
        )
        self.down_proj = RowParallelLinear(
            shared_intermediate_size, hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Qwen35MoE(nn.Module):
    """Mixture-of-Experts FFN with shared expert.

    Routing: top-k gate → softmax → sort-by-expert dispatch → per-expert matmul
    → unsort/combine → add sigmoid-gated shared expert.

    Matches src/model_small_qwen3.5.py MoEFFN exactly.
    No TP sharding of experts (full replication) — flagged as future work.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        shared_intermediate_size: int,
        num_experts: int,
        top_k: int,
        use_vectorized_moe: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        # QLLM Stage 2 intervention: replace _forward_dispatch's per-expert
        # Python loop with two torch._grouped_mm calls (grouped/batched GEMM)
        # for the non-EP prefill path. Default False so nothing changes
        # unless explicitly opted in; _forward_dispatch remains fully intact
        # as the flag-off path and as ground truth for correctness
        # comparisons. Scoped to ep_size==1 ONLY -- see forward()'s branch
        # and the investigation report for why _forward_dispatch_ep (the
        # EP prefill loop) is deliberately NOT touched here: extending
        # grouped_mm to the round-robin local-expert-slot indexing +
        # all_reduce combine there needs its own careful pass, flagged
        # rather than guessed at.
        self.use_vectorized_moe = use_vectorized_moe

        # ep_size==1 (the default/existing single-process and non-EP-TP case)
        # must allocate identically to before: Experts(num_experts, ...), same
        # as prior to EP support. Only ep_size>1 shrinks the local allocation
        # -- round-robin, matching utils.loader.expert_local_slot exactly
        # (expert e -> rank e % ep_size, local slot e // ep_size).
        self.ep_size = dist.get_world_size()
        self.ep_rank = dist.get_rank()
        local_num_experts = num_experts // self.ep_size if self.ep_size > 1 else num_experts
        self.experts = Experts(local_num_experts, intermediate_size, hidden_size)
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        self.shared_expert = Qwen35SharedExpert(hidden_size, shared_intermediate_size)
        self.shared_expert_gate = ReplicatedLinear(hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
        N = x.shape[0]
        is_decode = (cu_seqlens is not None) and (cu_seqlens.numel() - 1 == N)
        if is_decode:
            if self.ep_size > 1:
                return self._forward_gathered_ep(x)
            return self._forward_gathered(x)
        if self.ep_size > 1:
            return self._forward_dispatch_ep(x)
        if self.use_vectorized_moe:
            return self._forward_dispatch_vectorized(x)
        return self._forward_dispatch(x)
 
    def _forward_gathered_w8a8_hopper(self, x: torch.Tensor, w: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """True W8A8 Hopper path for _forward_gathered (ep_size=1 decode).
        UNVALIDATED -- see the import-site comment above and
        w8a8_activation_quant_scoping_memo.md.

        Unlike every INT8 branch in this file, fused_moe_w8a8_hopper_forward
        does the FULL top_k-weighted combine internally (reduce-add per
        (token,k) pair inside the kernel, per moe_w8a8.h) -- it returns the
        final (N, H) MoE output directly, not a (N, TK, H) out_e for a
        caller-side combine step to reduce.

        idx plays the same role local_slots plays in the EP branch's
        equivalent call: at ep_size=1, self.experts holds ALL experts, so a
        global expert id IS already a valid local slot -- same reasoning
        this function's INT8 siblings already rely on.

        Reads gate_up_proj_fp8_KERNEL, not gate_up_proj_fp8 -- this kernel
        needs gate/up interleaved every 8 physical rows (found 2026-08-24),
        not the contiguous layout the plain FP8 dequant branches elsewhere
        in this file use. See moe_w8a8_hopper_integration.py's
        quantize_experts_module_fp8_inplace for where the two buffers are
        produced.
        """
        n_local_experts = self.experts.gate_up_proj_fp8_kernel.shape[0]
        x_fp8, x_scale = quantize_activation_fp8_dynamic(x)
        sorted_ids, expert_ids, ntpp = moe_align_block_size(idx, _W8A8_HOPPER_BLOCK_M, n_local_experts)
        out = fused_moe_w8a8_hopper_forward(
            x_fp8, x_scale,
            self.experts.gate_up_proj_fp8_kernel, self.experts.gate_up_proj_scale_fp8_kernel,
            self.experts.down_proj_fp8, self.experts.down_proj_scale_fp8,
            sorted_ids, expert_ids, ntpp,
            w.to(torch.float32), self.top_k, n_local_experts,
            block_m=_W8A8_HOPPER_BLOCK_M, block_n=_W8A8_HOPPER_BLOCK_N,
            warp_n=_W8A8_HOPPER_WARP_N, stages=_W8A8_HOPPER_STAGES,
        )
        return out.to(x.dtype)

    def _forward_gathered(self, x: torch.Tensor) -> torch.Tensor:
        """Capture-safe MoE forward for decode. No loops, no .item(), no
        data-dependent branching -- every op's shape is static given N (batch
        size) and self.top_k, both fixed for a given call/captured graph."""
        H, TK = self.hidden_size, self.top_k
        N = x.shape[0]

        w, idx = torch.topk(self.gate(x), TK, dim=-1)          # (N, TK)
        w = F.softmax(w, dim=-1).to(x.dtype)

        if _USE_MOE_W8A8_HOPPER and hasattr(self.experts, "gate_up_proj_fp8"):
            # Early return, deliberately not interleaved into the int8/bf16
            # chain below -- this kernel already produces the final combined
            # output (see helper docstring), and keeping the existing chain's
            # indentation/structure untouched avoids risking a transcription
            # error in code that was already reasoned through and is not
            # re-testable here.
            moe_out = self._forward_gathered_w8a8_hopper(x, w, idx)
            sg = torch.sigmoid(self.shared_expert_gate(x))
            return moe_out + sg * self.shared_expert(x)

        # Advanced indexing: dynamic VALUES, static SHAPE (N, TK, ...) --
        # safe under graph capture, same principle as StateManager's
        # index_select on state_slot_ids. At ep_size=1 every global expert id
        # IS a local slot directly (self.experts holds ALL experts, no
        # sharding) -- unlike _forward_gathered_ep, idx needs no // ep_size,
        # no ownership mask, no all_reduce.
        if _USE_FUSED_MOE_KERNEL and hasattr(self.experts, "gate_up_proj_int8"):
            # Fused-Triton-kernel path -- NANOVLLM_USE_FUSED_MOE_KERNEL=1.
            # Mirrors _forward_gathered_ep's fused branch: that call never
            # touches ep_rank/owned_mask/all_reduce internally (those only
            # apply to the EP combine step below the kernel call, which
            # doesn't exist at ep_size=1) -- so the kernel itself is already
            # EP-agnostic, this just wires it up here too, passing `idx`
            # directly where the EP branch passes `local_slots`. Added
            # alongside the plain-dequant tp=1 fix below, same "unblock
            # tp=1, don't touch EP" scoping.
            fused_out_e = fused_moe_int8_forward(
                x,
                self.experts.gate_up_proj_int8, self.experts.gate_up_proj_scale,
                self.experts.down_proj_int8, self.experts.down_proj_scale,
                self.experts.moe_w8a8_group_size,
                idx, TK, self.experts.gate_up_proj_int8.shape[0],
            )
            gate_up = down = None  # unused below when this branch is taken
        elif hasattr(self.experts, "gate_up_proj_int8"):
            # Plain int8 dequant branch, added to unblock use_moe_w8a8=True
            # at tensor_parallel_size=1 (previously EP-only, ep_size>1 --
            # moe_int8_integration.py deletes the bf16 Parameters
            # unconditionally regardless of ep_size, so this path would
            # AttributeError on the plain bf16 read below without this
            # branch). Mirrors _forward_gathered_ep's dequant branch exactly,
            # minus the ownership mask/all_reduce that ep_size=1 doesn't need.
            G = self.experts.moe_w8a8_group_size
            gu_i8 = self.experts.gate_up_proj_int8[idx]            # (N, TK, 2*MI, H) int8
            gu_sc = self.experts.gate_up_proj_scale[idx]
            dp_i8 = self.experts.down_proj_int8[idx]               # (N, TK, H, MI) int8
            dp_sc = self.experts.down_proj_scale[idx]
            gate_up = dequantize_weight_int8_grouped(gu_i8, gu_sc, G, x.dtype)
            down = dequantize_weight_int8_grouped(dp_i8, dp_sc, G, x.dtype)
            fused_out_e = None
        else:
            gate_up = self.experts.gate_up_proj[idx]                # (N, TK, 2*MI, H)
            down = self.experts.down_proj[idx]                      # (N, TK, H, MI)
            fused_out_e = None
        if fused_out_e is not None:
            out_e = fused_out_e
        else:
            gw, uw = gate_up.chunk(2, dim=2)                    # each (N, TK, MI, H)

            # Per-token, per-selected-expert projection -- equivalent to the
            # dispatch loop's `xt @ gw.t()` / `xt @ uw.t()`, just batched
            # over (N, TK) instead of looped over experts with a token
            # subset each.
            h_gate = torch.einsum('nkmh,nh->nkm', gw, x)        # (N, TK, MI)
            h_up = torch.einsum('nkmh,nh->nkm', uw, x)          # (N, TK, MI)
            h = F.silu(h_gate) * h_up                            # (N, TK, MI)

            # Equivalent to the dispatch loop's `h @ down_proj[e].t()`.
            out_e = torch.einsum('nkhm,nkm->nkh', down, h)      # (N, TK, H)
        # NOTE: unlike _forward_dispatch/_forward_dispatch_ep, this combine
        # is NOT promoted to fp32 -- same unpromoted bf16-sum-across-top_k
        # pattern that measurably caused ~1.3% relative error there before
        # the fix (see _forward_dispatch's comment), so this decode path
        # likely has the same issue. Not fixed here: this path wasn't
        # ablation-tested, only _forward_dispatch/_forward_dispatch_ep were.
        out = (out_e * w.unsqueeze(-1)).sum(dim=1)               # (N, H)
 
        sg = torch.sigmoid(self.shared_expert_gate(x))
        out = out + sg * self.shared_expert(x)
        return out

    def _forward_gathered_ep_w8a8_hopper(self, x: torch.Tensor, w: torch.Tensor,
                                          local_slots: torch.Tensor, owned_mask: torch.Tensor) -> torch.Tensor:
        """True W8A8 Hopper path for _forward_gathered_ep (ep_size>1 decode).
        UNVALIDATED -- see the import-site comment and the scoping memo.

        fused_moe_w8a8_hopper_forward's internal combine reduce-adds
        topk_weight-scaled per-(token,k) contributions directly into `out`
        -- there is no separate (N, TK, H) out_e for this rank's combine
        step to mask-then-sum the way every other EP branch here does. So
        the masking has to happen BEFORE the kernel call instead of after:
        zero this rank's topk weight for every (token,k) pair it does NOT
        own, so its contribution to the kernel's internal reduce-add is
        exactly zero (the local_slots gather for those pairs still reads
        some OTHER expert's weights, same wasted-but-discarded-compute
        trade-off _forward_gathered_ep's own docstring already accepts --
        multiplying by a zero weight discards it regardless of what was
        gathered), then all_reduce (SUM) the per-rank partial result across
        ranks -- structurally the same zero-then-sum combine every other EP
        branch uses, just relocated to before the kernel because this
        kernel owns the reduce-add step internally rather than exposing it.

        Precision caveat, stated plainly because it's new here: every other
        EP branch promotes the combine to fp32 BEFORE summing top_k
        contributions (measured to matter -- see _forward_dispatch's own
        comment on the ~1.3% relative error this fixed). This kernel's
        `out` is declared __nv_bfloat16 (moe_w8a8.h) with no fp32 output
        option, so the per-(token,k) reduce-add itself happens in whatever
        precision the kernel internally accumulates in -- unconfirmed, the
        kernel has never run. This function casts to fp32 immediately after
        the kernel call, before all_reduce, matching the existing
        discipline as closely as the kernel's fixed output dtype allows --
        but the summation that matters most already happened inside the
        kernel at bf16-out precision, not fp32, unlike every other branch
        here. Flag this explicitly in Phase 3's accuracy validation rather
        than assuming it's equivalent.

        Reads gate_up_proj_fp8_KERNEL (interleaved), not gate_up_proj_fp8
        (contiguous) -- same reasoning as _forward_gathered_w8a8_hopper's
        docstring.
        """
        n_local_experts = self.experts.gate_up_proj_fp8_kernel.shape[0]
        x_fp8, x_scale = quantize_activation_fp8_dynamic(x)
        w_masked = w.to(torch.float32) * owned_mask.to(torch.float32)
        sorted_ids, expert_ids, ntpp = moe_align_block_size(local_slots, _W8A8_HOPPER_BLOCK_M, n_local_experts)
        local_out = fused_moe_w8a8_hopper_forward(
            x_fp8, x_scale,
            self.experts.gate_up_proj_fp8_kernel, self.experts.gate_up_proj_scale_fp8_kernel,
            self.experts.down_proj_fp8, self.experts.down_proj_scale_fp8,
            sorted_ids, expert_ids, ntpp,
            w_masked, self.top_k, n_local_experts,
            block_m=_W8A8_HOPPER_BLOCK_M, block_n=_W8A8_HOPPER_BLOCK_N,
            warp_n=_W8A8_HOPPER_WARP_N, stages=_W8A8_HOPPER_STAGES,
        ).to(torch.float32)
        dist.all_reduce(local_out)
        return local_out.to(x.dtype)

    def _forward_gathered_ep(self, x: torch.Tensor) -> torch.Tensor:
        """Expert-parallel, capture-safe MoE forward for decode -- the EP
        counterpart of _forward_gathered(). Fills the gap _forward_gathered's
        docstring used to flag as out of scope (NotImplementedError at
        ep_size>1): this is the decode-time twin of _forward_dispatch_ep(),
        using the SAME round-robin ownership rule (global expert e is owned
        by rank e % ep_size, at local slot e // ep_size in this rank's
        sharded Experts tensor) and the same all_reduce combine, but keeping
        _forward_gathered's capture-safe shape discipline: no loops, no
        .item(), no data-dependent branching -- every op's shape is static
        given N (batch size) and self.top_k.

        Unlike the non-EP _forward_gathered(), self.experts only holds this
        rank's LOCAL shard (local_num_experts = num_experts // ep_size), so a
        global expert id can't index it directly. idx // ep_size is always a
        valid in-bounds local slot regardless of ownership (every global id
        is < num_experts = local_num_experts * ep_size, so
        idx // ep_size < local_num_experts unconditionally) -- for (token, k)
        pairs this rank does NOT own, that indexes into some OTHER expert's
        weights, computing a value that is deliberately never used: it's
        multiplied by owned_mask (0 for non-owned pairs) before the sum, so
        it contributes nothing to the result. This trades some wasted
        compute on non-owned pairs for a fully static-shape, branch-free
        gather -- the same trade-off _forward_gathered already makes for
        every (token, k) pair regardless of EP.

        Combine (weighted sum across top_k, then all_reduce) is promoted to
        fp32, mirroring _forward_dispatch_ep's measured-correct discipline --
        deliberately NOT copying the plain _forward_gathered's
        unpromoted-bf16 combine (its own comment flags that as an unfixed,
        measured ~1.3% relative error on the non-EP decode path). No reason
        to carry a known bug into new code when the fix is already
        established and cheap on the prefill EP path.

        Same reassociation caveat as _forward_dispatch_ep: bitwise-equivalent
        to a dense fp32 reference for top_k==2 (sum of exactly 2 finite
        floats is exactly commutative regardless of which rank computed
        which term); not a general guarantee for top_k > 2 spread across
        more than 2 ranks.

        Sets self._last_ep_token_id_roundtrip the same way
        _forward_dispatch_ep does (from the *same* ownership mask the real
        computation uses), for the same independent-verification purpose.
        """
        TK, P = self.top_k, self.ep_size
        N = x.shape[0]

        w, idx = torch.topk(self.gate(x), TK, dim=-1)            # (N, TK) global expert ids
        w = F.softmax(w, dim=-1).to(x.dtype)

        local_slots = idx // P                                    # (N, TK) -- always in-bounds, see docstring
        owned_mask = (idx % P) == self.ep_rank                    # (N, TK) bool

        if _USE_MOE_W8A8_HOPPER and hasattr(self.experts, "gate_up_proj_fp8"):
            # Early return, deliberately not interleaved into the int8/bf16
            # chain below -- same reasoning as _forward_gathered's identical
            # early-return: this kernel already produces the final combined,
            # all_reduce'd output (see helper docstring), and keeping the
            # existing chain untouched avoids risking a transcription error
            # in code that was already reasoned through and is not
            # re-testable on this machine.
            local_out = self._forward_gathered_ep_w8a8_hopper(x, w, local_slots, owned_mask)

            touched = owned_mask.any(dim=1)                             # (N,) bool
            token_id_local = torch.where(
                touched,
                torch.arange(N, device=x.device, dtype=torch.int64),
                torch.full((N,), -1, dtype=torch.int64, device=x.device),
            )
            dist.all_reduce(token_id_local, op=dist.ReduceOp.MAX)
            self._last_ep_token_id_roundtrip = token_id_local

            sg = torch.sigmoid(self.shared_expert_gate(x))
            return local_out + sg * self.shared_expert(x)

        if _USE_FUSED_MOE_KERNEL and hasattr(self.experts, "gate_up_proj_int8"):
            # Fused-Triton-kernel path -- NANOVLLM_USE_FUSED_MOE_KERNEL=1.
            # Replaces gather+dequantize+einsum entirely (no (N,TK,...)
            # gathered-and-dequantized weight buffer ever materialized --
            # that buffer IS the thing today's default path pays for, and
            # this avoids it structurally, not just by shrinking it).
            # STILL OPEN as of 2026-08-21, do not treat this branch as
            # equally trusted to the one below yet: (1) CUDA graph capture
            # compatibility unverified -- moe_align_block_size's
            # torch.argsort/cumsum/searchsorted are all static-shape given
            # fixed (N, TK)/local_num_experts, expected graph-safe, but
            # "expected" is not "confirmed under torch.cuda.graph()"; (2) no
            # real-checkpoint GSM8K check run through this path yet; (3) no
            # real engine tok/s number, only isolated microbenchmarks
            # (layers/smoke_test_full_moe_pipeline.py: cosine=0.999988,
            # 4.71x vs. the branch below, single-GPU, synthetic weights).
            out_e = fused_moe_int8_forward(
                x,
                self.experts.gate_up_proj_int8, self.experts.gate_up_proj_scale,
                self.experts.down_proj_int8, self.experts.down_proj_scale,
                self.experts.moe_w8a8_group_size,
                local_slots, TK, self.experts.gate_up_proj_int8.shape[0],
            )
        else:
            if hasattr(self.experts, "gate_up_proj_int8"):
                gu_i8 = self.experts.gate_up_proj_int8[local_slots]           # (N, TK, 2*MI, H) int8
                gu_sc = self.experts.gate_up_proj_scale[local_slots]          # (N, TK, 2*MI, H//G)
                dp_i8 = self.experts.down_proj_int8[local_slots]              # (N, TK, H, MI) int8
                dp_sc = self.experts.down_proj_scale[local_slots]             # (N, TK, H, MI//G)
                G = self.experts.moe_w8a8_group_size
                gate_up = dequantize_weight_int8_grouped(gu_i8, gu_sc, G, x.dtype)
                down = dequantize_weight_int8_grouped(dp_i8, dp_sc, G, x.dtype)
            else:
                gate_up = self.experts.gate_up_proj[local_slots]          # (N, TK, 2*MI, H)
                down = self.experts.down_proj[local_slots]                # (N, TK, H, MI)

            gw, uw = gate_up.chunk(2, dim=2)                        # each (N, TK, MI, H)

            h_gate = torch.einsum('nkmh,nh->nkm', gw, x)               # (N, TK, MI)
            h_up = torch.einsum('nkmh,nh->nkm', uw, x)                 # (N, TK, MI)
            h = F.silu(h_gate) * h_up                                  # (N, TK, MI)
            out_e = torch.einsum('nkhm,nkm->nkh', down, h)             # (N, TK, H)

        # Weighted-multiply/local-accumulate/all_reduce combine always
        # happens in fp32 -- see docstring.
        combine_dtype = torch.float32
        mask = owned_mask.to(combine_dtype).unsqueeze(-1)          # (N, TK, 1)
        weighted = out_e.to(combine_dtype) * w.to(combine_dtype).unsqueeze(-1) * mask
        local_out = weighted.sum(dim=1)                             # (N, H), fp32

        dist.all_reduce(local_out)
        local_out = local_out.to(x.dtype)

        # Token-id round-trip check, using the exact same owned_mask this
        # rank's real computation used. A token is "touched" by this rank if
        # it owns at least one of that token's top_k pairs; every token has
        # top_k>=1 pairs, each assigned to exactly one rank, so every token
        # is touched by exactly one rank for each of its pairs -- MAX-combine
        # across ranks recovers arange(N) with no leftover -1 sentinels.
        touched = owned_mask.any(dim=1)                             # (N,) bool
        token_id_local = torch.where(
            touched,
            torch.arange(N, device=x.device, dtype=torch.int64),
            torch.full((N,), -1, dtype=torch.int64, device=x.device),
        )
        dist.all_reduce(token_id_local, op=dist.ReduceOp.MAX)
        self._last_ep_token_id_roundtrip = token_id_local

        sg = torch.sigmoid(self.shared_expert_gate(x))
        out = local_out + sg * self.shared_expert(x)
        return out

    def _forward_dispatch(self, x: torch.Tensor) -> torch.Tensor:
        """Original sort-by-expert dispatch loop -- unchanged. Used for
        prefill (and any eager, non-graph-captured call), where N can be
        large and gathering full per-token expert weight matrices would be
        far too much memory."""
        original_shape = x.shape
        H = self.hidden_size
        NE = self.num_experts
        TK = self.top_k
 
        xf = x.reshape(-1, H)
        N = xf.shape[0]
 
        w, idx = torch.topk(self.gate(xf), TK, dim=-1)
        w = F.softmax(w, dim=-1).to(x.dtype)
 
        flat_idx = idx.reshape(-1)
        flat_w = w.reshape(-1)
        token_rep = xf.unsqueeze(1).expand(N, TK, H).reshape(N * TK, H)
 
        sort_order = torch.argsort(flat_idx, stable=True)
        sorted_idx = flat_idx[sort_order]
        sorted_tokens = token_rep[sort_order]
        sorted_weights = flat_w[sort_order]
        expert_counts = torch.zeros(NE, dtype=torch.long, device=x.device)
        expert_counts.scatter_add_(0, sorted_idx, torch.ones_like(sorted_idx))
 
        expert_offsets = torch.cat([
            torch.zeros(1, device=x.device, dtype=torch.long),
            expert_counts.cumsum(0)[:-1]
        ])
 
        # Weighted-sum-across-top_k combine always happens in fp32, regardless
        # of x.dtype -- expert FFN matmuls (h) stay in x.dtype, unchanged,
        # only this combine step is promoted, then cast back down before
        # shared_expert. Measured (not assumed): for bf16 x, this took the
        # routed-expert-only output from ~1.3% relative error (bf16-throughout
        # combine) to exactly bitwise zero vs an fp32 reference, for top_k=8 --
        # bf16 inputs carry ~8 mantissa bits; summing top_k=8 of them in an
        # fp32 (24-bit) accumulator needs only 8+log2(8)=11 bits of headroom,
        # comfortably inside fp32's 24, so the sum is exact regardless of
        # summation order/reassociation. For x already fp32 this is a no-op
        # (combine_dtype == x.dtype either way) -- verified no regression on
        # the top_k=2/top_k=8 fp32 checkpoints.
        combine_dtype = torch.float32
        sorted_out = torch.zeros(N * TK, H, device=x.device, dtype=combine_dtype)
        for e in range(NE):
            cnt = expert_counts[e].item()
            if cnt == 0:
                continue
            start = expert_offsets[e].item()
            xt = sorted_tokens[start:start + cnt]
            # int8 branch, added to unblock use_moe_w8a8=True at
            # tensor_parallel_size=1 -- mirrors _forward_dispatch_ep's
            # dequant branch exactly (see there for why: moe_int8_integration.py
            # deletes the bf16 Parameters unconditionally regardless of
            # ep_size, so the plain bf16 read below would AttributeError
            # without this branch).
            if hasattr(self.experts, "gate_up_proj_int8"):
                G = self.experts.moe_w8a8_group_size
                gate_up_e = dequantize_weight_int8_grouped(
                    self.experts.gate_up_proj_int8[e],
                    self.experts.gate_up_proj_scale[e],
                    G, x.dtype,
                )
                down_e = dequantize_weight_int8_grouped(
                    self.experts.down_proj_int8[e],
                    self.experts.down_proj_scale[e],
                    G, x.dtype,
                )
            elif hasattr(self.experts, "gate_up_proj_fp8"):
                # Plain FP8 dequant branch -- W8A8 Hopper's decode-only
                # branches (_forward_gathered/_forward_gathered_ep) never
                # gave prefill an FP8-aware path at all, so
                # use_moe_w8a8_hopper=True deleted the bf16 Parameters and
                # left this function reading them anyway -- AttributeError
                # on the very first prefill call, before decode is ever
                # reached. Mirrors the int8 branch above exactly; no fused
                # kernel for prefill here either, same deferral precedent
                # the int8 fused kernel already established.
                G = self.experts.moe_w8a8_hopper_group_size
                gate_up_e = dequantize_weight_fp8_grouped_gathered(
                    self.experts.gate_up_proj_fp8[e],
                    self.experts.gate_up_proj_scale_fp8[e],
                    G, x.dtype,
                )
                down_e = dequantize_weight_fp8_grouped_gathered(
                    self.experts.down_proj_fp8[e],
                    self.experts.down_proj_scale_fp8[e],
                    G, x.dtype,
                )
            else:
                gate_up_e = self.experts.gate_up_proj[e]
                down_e = self.experts.down_proj[e]
            gw, uw = gate_up_e.chunk(2, 0)
            h = F.silu(xt @ gw.t()) * (xt @ uw.t())
            h = h @ down_e.t()
            w_slice = sorted_weights[start:start + cnt].unsqueeze(-1).to(combine_dtype)
            sorted_out[start:start + cnt] = w_slice * h.to(combine_dtype)

        unsort_order = torch.argsort(sort_order, stable=True)
        out = sorted_out[unsort_order].reshape(N, TK, H).sum(dim=1).to(x.dtype)

        sg = torch.sigmoid(self.shared_expert_gate(xf))
        out = out + sg * self.shared_expert(xf)
        return out.view(original_shape)

    def _forward_dispatch_vectorized(self, x: torch.Tensor) -> torch.Tensor:
        """QLLM Stage 2: same routing/sort/combine as _forward_dispatch, but
        the per-expert Python loop (one matmul per expert) is replaced by
        two torch._grouped_mm calls -- one grouped GEMM for gate_up_proj,
        one for down_proj, batching every active expert's computation into
        a single call each instead of looping.

        Every detail below was verified empirically against the actual
        installed torch build (torch._grouped_mm has no docstring; the
        schema and behavior were probed directly with synthetic tensors,
        not assumed from general knowledge of similar APIs) before writing
        this, per the investigation report:

        - Real schema (confirmed via torch.ops.aten._grouped_mm.default
          ._schema): `_grouped_mm(self, mat2, offs=None, bias=None,
          out_dtype=None)`.
        - `offs` convention (confirmed via value-level probe with
          distinguishable per-group scale factors, not just shape checks):
          CUMULATIVE END-of-group counts across ALL groups in mat2's batch
          dim -- i.e. `expert_counts.cumsum(0)`, NOT the exclusive-START
          `expert_offsets` _forward_dispatch computes for its own loop.
          Must be int32 -- int64 raises a clear "Offsets have to be int32"
          error (fails loudly, not silently), so offs is cast explicitly
          below.
        - `mat2` layout requirement (confirmed via value-level probe):
          `(num_groups, in_features, out_features)` -- the TRANSPOSE of how
          Experts.gate_up_proj/down_proj are actually stored
          ((E, out_features, in_features), matching nn.Linear's convention).
          Confirmed via utils/loader.py's shard_experts_tensor /
          load_model: checkpoints are read directly into this (E, out, in)
          layout with no transpose anywhere in the loading path, so this is
          the checkpoint-native layout, not just this file's convention --
          changing the STORED layout would mean touching weight loading.
          Instead, .transpose(-1, -2) is applied at call time -- confirmed
          empirically that torch._grouped_mm accepts this non-contiguous
          transposed VIEW directly at realistic MoE dims (hidden=256,
          intermediate=512 in the probe) with no .contiguous() copy needed
          (contiguous copies were only required in a small K=N=4/16 probe,
          which hit an unrelated small-tensor alignment constraint --
          "strides should be multiple of 16 bytes" -- not a general
          transposed-view limitation).
        - Zero-token experts (an expert selected by no token in this batch)
          are handled correctly: confirmed via a value-level probe with an
          explicit zero-width group (offs[i] == offs[i-1]) sitting between
          two nonzero groups with very different weight scales -- the
          zero-width expert's weights were correctly never touched.

        Precision: the fp32-promoted combine step below is copied VERBATIM
        in placement from _forward_dispatch's (see that method's comment for
        the full measured justification -- ~1.3% relative error before this
        promotion was added, exactly zero after, for top_k=8 bf16 inputs).
        Both expert matmuls (gate_up_proj and down_proj) stay in x.dtype
        (bf16), matching _forward_dispatch's per-expert `h = ... ; h = h @
        down_proj[e].t()` exactly -- promotion to fp32 happens ONLY at the
        weighted-combine step, same as before. Deliberately NOT using
        _grouped_mm's own `out_dtype` parameter to request fp32 output
        directly from the second grouped_mm call: that would skip the
        bf16-rounding step _forward_dispatch's reference behavior actually
        has (compute in bf16, round to bf16, THEN promote to fp32 for the
        combine) in favor of a different, more-precise-but-unverified-and-
        unrequested behavior. Preserving the exact existing behavior here,
        not quietly improving on it.

        Scoped to ep_size==1 only -- see forward()'s branch and __init__'s
        comment.
        """
        original_shape = x.shape
        H = self.hidden_size
        NE = self.num_experts
        TK = self.top_k

        xf = x.reshape(-1, H)
        N = xf.shape[0]

        w, idx = torch.topk(self.gate(xf), TK, dim=-1)
        w = F.softmax(w, dim=-1).to(x.dtype)

        flat_idx = idx.reshape(-1)
        flat_w = w.reshape(-1)
        token_rep = xf.unsqueeze(1).expand(N, TK, H).reshape(N * TK, H)

        sort_order = torch.argsort(flat_idx, stable=True)
        sorted_idx = flat_idx[sort_order]
        sorted_tokens = token_rep[sort_order]
        sorted_weights = flat_w[sort_order]
        expert_counts = torch.zeros(NE, dtype=torch.long, device=x.device)
        expert_counts.scatter_add_(0, sorted_idx, torch.ones_like(sorted_idx))

        # Cumulative END counts, int32 -- see docstring. Length == NE
        # (mat2's batch dim), including zero-width entries for untouched
        # experts -- confirmed handled correctly, see docstring.
        offs = expert_counts.cumsum(0).to(torch.int32)

        # Transposed VIEWs, not copies -- see docstring.
        gate_up_t = self.experts.gate_up_proj.transpose(-1, -2)  # (NE, H, 2*MI)
        down_t = self.experts.down_proj.transpose(-1, -2)        # (NE, MI, H)

        gate_up_out = torch._grouped_mm(sorted_tokens, gate_up_t, offs=offs)  # (N*TK, 2*MI), x.dtype
        gate, up = gate_up_out.chunk(2, dim=-1)
        h = F.silu(gate) * up                                                  # (N*TK, MI), x.dtype
        h = torch._grouped_mm(h, down_t, offs=offs)                            # (N*TK, H), x.dtype

        # fp32-promoted combine -- see docstring, matches _forward_dispatch
        # exactly.
        combine_dtype = torch.float32
        sorted_out = sorted_weights.unsqueeze(-1).to(combine_dtype) * h.to(combine_dtype)

        unsort_order = torch.argsort(sort_order, stable=True)
        out = sorted_out[unsort_order].reshape(N, TK, H).sum(dim=1).to(x.dtype)

        sg = torch.sigmoid(self.shared_expert_gate(xf))
        out = out + sg * self.shared_expert(xf)
        return out.view(original_shape)

    def _forward_dispatch_ep(self, x: torch.Tensor) -> torch.Tensor:
        """Expert-parallel prefill forward.

        This is NOT an all_to_all design. Every rank in the EP group (== the
        TP group -- this codebase has no separate data-parallel dimension, so
        every TP rank already processes bit-identical activations; confirmed
        via engine/model_runner.py's shared-memory broadcast of the same
        seqs/input_ids to every rank) already has the full x locally. No
        token movement is needed: each rank locally selects the (token, k)
        pairs whose expert it owns (round-robin -- expert e belongs to rank
        e % ep_size at local slot e // ep_size, matching
        utils.loader.expert_local_slot exactly), computes only those pairs
        against its own local (sharded) Experts tensor, and an all_reduce
        sums the disjoint per-rank partial contributions into the true total
        -- the same combine mechanism RowParallelLinear already uses
        correctly for shared_expert.

        Bitwise-equivalent to the dense _forward_dispatch reference for
        top_k==2: each token's final value is a sum of exactly 2 finite
        floats (its two selected experts' contributions); IEEE754 addition
        of 2 finite floats is exactly commutative and zero-padding never
        perturbs a float via addition, so which rank(s) computed which of
        the 2 terms doesn't change the bit pattern of their sum. (Not a
        general guarantee for top_k > 2 spread across more ranks, where
        reassociation of 3+ terms can matter -- untested beyond top_k==2.)

        Sets self._last_ep_token_id_roundtrip (test/debug instrumentation,
        not used by the numeric output) from the *same* ownership mask and
        token-index tensor the real computation uses, so a caller can verify
        indexing correctness rode through the real communication step rather
        than checking a side-channel duplicate.
        """
        original_shape = x.shape
        H = self.hidden_size
        TK = self.top_k
        P = self.ep_size

        xf = x.reshape(-1, H)
        N = xf.shape[0]

        w, idx = torch.topk(self.gate(xf), TK, dim=-1)
        w = F.softmax(w, dim=-1).to(x.dtype)

        flat_idx = idx.reshape(-1)                                          # (N*TK,) global expert id
        flat_w = w.reshape(-1)                                              # (N*TK,)
        token_rep = xf.unsqueeze(1).expand(N, TK, H).reshape(N * TK, H)
        token_of_pair = torch.arange(N, device=x.device).unsqueeze(1).expand(N, TK).reshape(-1)

        owned = (flat_idx % P) == self.ep_rank                              # (N*TK,) bool
        local_slots = flat_idx[owned] // P
        owned_tokens = token_rep[owned]
        owned_weights = flat_w[owned]
        owned_token_idx = token_of_pair[owned]

        # Weighted-multiply/local-accumulate/all_reduce combine always happens
        # in fp32, regardless of x.dtype -- symmetric with _forward_dispatch's
        # fp32 combine, same measured justification (see there).
        combine_dtype = torch.float32
        local_out = torch.zeros(N, H, device=x.device, dtype=combine_dtype)
        for local_e in range(self.experts.num_experts):
            sel = local_slots == local_e
            cnt = int(sel.sum())
            if cnt == 0:
                continue
            xt = owned_tokens[sel]
            
            if hasattr(self.experts, "gate_up_proj_int8"):
                G = self.experts.moe_w8a8_group_size
                gate_up_e = dequantize_weight_int8_grouped(
                    self.experts.gate_up_proj_int8[local_e],
                    self.experts.gate_up_proj_scale[local_e],
                    G, x.dtype,
                )
                down_e = dequantize_weight_int8_grouped(
                    self.experts.down_proj_int8[local_e],
                    self.experts.down_proj_scale[local_e],
                    G, x.dtype,
                )
            elif hasattr(self.experts, "gate_up_proj_fp8"):
                # Same fix, same reason as _forward_dispatch's twin branch --
                # see that one's comment.
                G = self.experts.moe_w8a8_hopper_group_size
                gate_up_e = dequantize_weight_fp8_grouped_gathered(
                    self.experts.gate_up_proj_fp8[local_e],
                    self.experts.gate_up_proj_scale_fp8[local_e],
                    G, x.dtype,
                )
                down_e = dequantize_weight_fp8_grouped_gathered(
                    self.experts.down_proj_fp8[local_e],
                    self.experts.down_proj_scale_fp8[local_e],
                    G, x.dtype,
                )
            else:
                gate_up_e = self.experts.gate_up_proj[local_e]
                down_e = self.experts.down_proj[local_e]
            gw, uw = gate_up_e.chunk(2, 0)
            h = F.silu(xt @ gw.t()) * (xt @ uw.t())
            h = h @ down_e.t()
            w_slice = owned_weights[sel].unsqueeze(-1).to(combine_dtype)
            contribution = w_slice * h.to(combine_dtype)
            local_out.index_add_(0, owned_token_idx[sel], contribution)

        dist.all_reduce(local_out)
        local_out = local_out.to(x.dtype)

        token_id_local = torch.full((N,), -1, dtype=torch.int64, device=x.device)
        token_id_local[owned_token_idx] = owned_token_idx
        dist.all_reduce(token_id_local, op=dist.ReduceOp.MAX)
        self._last_ep_token_id_roundtrip = token_id_local

        sg = torch.sigmoid(self.shared_expert_gate(xf))
        out = local_out + sg * self.shared_expert(xf)
        return out.view(original_shape)


# ─── Decoder Layer ───────────────────────────────────────────────────────────────


class Qwen35DecoderLayer(nn.Module):
    """Single decoder layer: either full-attention or linear-attention + MoE FFN.

    Layer type is determined by layer_type string ('full_attention' or 'linear_attention').
    Uses Qwen35RMSNorm (1+w variant) with fused residual add.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        layer_type: str,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.is_full = (layer_type == "full_attention")

        hidden_size = config.hidden_size
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        # Norms
        self.input_layernorm = Qwen35RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(hidden_size, eps=rms_norm_eps)

        # Attention
        if self.is_full:
            head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)
            rotary_dim = int(head_dim * getattr(config, "partial_rotary_factor", 0.25))
            self.self_attn = Qwen35FullAttention(
                hidden_size=hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                max_position=getattr(config, "max_position_embeddings", 131072),
                rope_theta=getattr(config, "rope_theta", 10_000_000.0),
                rms_norm_eps=rms_norm_eps,
            )
        else:
           
            self.linear_attn = Qwen35LinearAttention(
                hidden_size=hidden_size,
                linear_attn_kq_heads=getattr(
                    config, "linear_num_key_heads", getattr(config, "linear_attn_kq_heads", 16)
                ),
                linear_attn_v_heads=getattr(
                    config, "linear_num_value_heads", getattr(config, "linear_attn_v_heads", 32)
                ),
                linear_attn_head_dim=getattr(
                    config, "linear_key_head_dim", getattr(config, "linear_attn_head_dim", 128)
                ),
                conv_kernel_size=getattr(
                    config, "linear_conv_kernel_dim", getattr(config, "conv_kernel_size", 4)
                ),
                rms_norm_eps=rms_norm_eps,
                use_fused_gdr_kernel=getattr(config, "use_fused_gdr_kernel", False),
            )

        # MoE FFN
        self.mlp = Qwen35MoE(
            hidden_size=hidden_size,
            intermediate_size=getattr(config, "moe_intermediate_size", getattr(config, "intermediate_size", None)),
            shared_intermediate_size=getattr(config, "shared_expert_intermediate_size", getattr(config, "intermediate_size", None)),
            num_experts=getattr(config, "num_experts", 256),
            top_k=getattr(config, "num_experts_per_tok", 8),
            use_vectorized_moe=getattr(config, "use_vectorized_moe", False),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        conv_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Returns:
            hidden_states, residual, new_state (or None), new_conv_state (or None)
        """
        # Pre-attention norm with fused residual
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Attention
        new_state = None
        new_conv = None
        if self.is_full:
            hidden_states = self.self_attn(positions, hidden_states)
        else:
            assert cu_seqlens is not None, "linear-attention layers require cu_seqlens"
            hidden_states, new_state, new_conv = self.linear_attn(
                hidden_states, cu_seqlens, states=state, conv_states=conv_state
            )
           
        # Post-attention norm with fused residual
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MoE FFN
        hidden_states = self.mlp(hidden_states, cu_seqlens)

        return hidden_states, residual, new_state, new_conv


# ─── Full Model ──────────────────────────────────────────────────────────────────


class Qwen35Model(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        num_layers = config.num_hidden_layers
        self.layer_types = _get_layer_types(config, num_layers)
        self.linear_layer_indices = [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen35DecoderLayer(config, i, self.layer_types[i])
            for i in range(num_layers)
        ])
        self.norm = Qwen35RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        states: list | None = None,
        conv_states: list | None = None,
    ) -> tuple[torch.Tensor, list, list]:
        """
        Args:
            input_ids: (N,) flat token ids
            positions: (N,) position ids
            states: list of per-linear-layer states or None
            conv_states: list of per-linear-layer conv states or None

        Returns:
            hidden_states, new_states, new_conv_states
        """
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        num_layers = len(self.layers)
        if states is None:
            states = [None] * num_layers
        if conv_states is None:
            conv_states = [None] * num_layers

        new_states = []
        new_conv_states = []

        for i, layer in enumerate(self.layers):
            hidden_states, residual, ns, nc = layer(
                positions, hidden_states, residual, cu_seqlens,
                state=states[i], conv_state=conv_states[i],
            )
            new_states.append(ns)
            new_conv_states.append(nc)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, new_states, new_conv_states


class Qwen35ForCausalLM(nn.Module):
    """Qwen3.5 hybrid causal LM for nano-vLLM.

    packed_modules_mapping only maps keys that actually differ between
    HF checkpoint names and internal parameter names.
    """
    packed_modules_mapping = {
        # Shared expert gate/up packing
        "shared_expert.gate_proj": ("shared_expert.gate_up_proj", 0),
        "shared_expert.up_proj": ("shared_expert.gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen35Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        states: list | None = None,
        conv_states: list | None = None,
    ) -> tuple[torch.Tensor, list, list]:
        return self.model(input_ids, positions, cu_seqlens, states, conv_states)

    def compute_logits(
        self,
        hidden_states_and_caches,
    ) -> torch.Tensor:
        """Accepts either bare hidden_states or (hidden_states, states, conv_states) tuple."""
        if isinstance(hidden_states_and_caches, tuple):
            hidden_states = hidden_states_and_caches[0]
        else:
            hidden_states = hidden_states_and_caches
        return self.lm_head(hidden_states)
