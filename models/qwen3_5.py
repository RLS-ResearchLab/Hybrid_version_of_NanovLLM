"""Qwen3.5 hybrid model for nano-vLLM.

Hybrid architecture: GDR linear-attention + GQA full-attention + MoE FFN.
Config-driven — reads all hyperparameters from hf_config, no hardcoded constants.
Numerically matches src/model_small_qwen3.5.py (the ground-truth reference).

Layer schedule:
    - Default: layer i is full-attention if (i+1) % full_attention_interval == 0
    - If config.layers_block_type is present, use that list directly
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.layernorm import Qwen35RMSNorm, Qwen35RMSNormGated
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
    MergedColumnParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_partial_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


def _is_full_attention(layer_idx: int, full_attention_interval: int) -> bool:
    """Default layer-type schedule matching src/model_small_qwen3.5.py."""
    return (layer_idx + 1) % full_attention_interval == 0


def _get_layer_types(config, num_layers: int) -> list[str]:
    """Determine layer types from config. Data-driven if available, else modulo rule."""
    # Prefer explicit per-layer type list (Jamba/Zamba-style)
    layers_block_type = getattr(config, "layers_block_type", None)
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
        self.total_num_heads = num_heads
        assert num_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert num_kv_heads % tp_size == 0
        self.num_kv_heads = num_kv_heads // tp_size
        self.head_dim = head_dim

        # q_proj outputs 2 * NQ * DH: query + gate interleaved per head
        self.q_proj = ColumnParallelLinear(
            hidden_size, 2 * num_heads * head_dim, bias=False
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim, bias=False
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim, bias=False
        )
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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lkh = linear_attn_kq_heads
        self.lvh = linear_attn_v_heads
        self.lhd = linear_attn_head_dim
        self.ck = conv_kernel_size

        # QKV dim = (LKH + LKH + LVH) * LHD
        self.qkv_dim = (self.lkh + self.lkh + self.lvh) * self.lhd

        # Projections
        self.in_proj_qkv = ColumnParallelLinear(hidden_size, self.qkv_dim, bias=False)
        self.in_proj_z = ColumnParallelLinear(hidden_size, self.lvh * self.lhd, bias=False)
        # A and B project to per-head scalars — LVH-sized output.
        # Structured for future TP sharding along head dimension.
        self.in_proj_a = ReplicatedLinear(hidden_size, self.lvh, bias=False)
        self.in_proj_b = ReplicatedLinear(hidden_size, self.lvh, bias=False)

        # Depthwise causal conv1d
        self.conv1d = nn.Conv1d(
            self.qkv_dim, self.qkv_dim, conv_kernel_size,
            groups=self.qkv_dim, padding=conv_kernel_size - 1, bias=False
        )

        # Per-head parameters for GDR gating
        self.A_log = nn.Parameter(torch.zeros(self.lvh))
        self.dt_bias = nn.Parameter(torch.zeros(self.lvh))

        # Gated RMSNorm on output
        self.norm = Qwen35RMSNormGated(self.lhd, eps=rms_norm_eps)

        # Output projection
        self.out_proj = RowParallelLinear(self.lvh * self.lhd, hidden_size, bias=False)

        # Head expansion ratio (LKH → LVH)
        assert self.lvh % self.lkh == 0
        self.head_expand = self.lvh // self.lkh

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

        # ── Project once over the whole packed N — pure per-token ops ──
        z = self.in_proj_z(hidden_states)        # (N, LVH*LHD)
        a = self.in_proj_a(hidden_states)        # (N, LVH)
        b = self.in_proj_b(hidden_states)        # (N, LVH)
        qkv = self.in_proj_qkv(hidden_states)    # (N, QKV)

        y_chunks, new_states, new_conv_states = [], [], []

        for i in range(num_segments):
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
            new_conv_states.append(seg_new_conv.squeeze(0))

        y = torch.cat(y_chunks, dim=0)               # (N, LVH*LHD)
        new_states = torch.stack(new_states, dim=0)          # (num_segments, LVH, LHD, LHD)
        new_conv_states = torch.stack(new_conv_states, dim=0)  # (num_segments, QKV, CK-1)

        return self.out_proj(y), new_states, new_conv_states

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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = Experts(num_experts, intermediate_size, hidden_size)
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        self.shared_expert = Qwen35SharedExpert(hidden_size, shared_intermediate_size)
        self.shared_expert_gate = ReplicatedLinear(hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        H = self.hidden_size
        NE = self.num_experts
        TK = self.top_k

        xf = x.reshape(-1, H)
        N = xf.shape[0]

        # Top-k routing with softmax
        w, idx = torch.topk(self.gate(xf), TK, dim=-1)  # (N, TK)
        w = F.softmax(w, dim=-1).to(x.dtype)

        # Sort-by-expert dispatch
        flat_idx = idx.reshape(-1)
        flat_w = w.reshape(-1)
        token_rep = xf.unsqueeze(1).expand(N, TK, H).reshape(N * TK, H)

        sort_order = torch.argsort(flat_idx, stable=True)
        sorted_idx = flat_idx[sort_order]
        sorted_tokens = token_rep[sort_order]
        sorted_weights = flat_w[sort_order]

        expert_counts = torch.bincount(sorted_idx, minlength=NE)
        expert_offsets = torch.cat([
            torch.zeros(1, device=x.device, dtype=torch.long),
            expert_counts.cumsum(0)[:-1]
        ])

        # Per-expert matmul loop
        sorted_out = torch.zeros(N * TK, H, device=x.device, dtype=x.dtype)
        for e in range(NE):
            cnt = expert_counts[e].item()
            if cnt == 0:
                continue
            start = expert_offsets[e].item()
            xt = sorted_tokens[start:start + cnt]
            gw, uw = self.experts.gate_up_proj[e].chunk(2, 0)
            h = F.silu(xt @ gw.t()) * (xt @ uw.t())
            h = h @ self.experts.down_proj[e].t()
            sorted_out[start:start + cnt] = sorted_weights[start:start + cnt].unsqueeze(-1) * h

        # Unsort and combine
        unsort_order = torch.argsort(sort_order, stable=True)
        out = sorted_out[unsort_order].reshape(N, TK, H).sum(dim=1)

        # Sigmoid-gated shared expert
        sg = torch.sigmoid(self.shared_expert_gate(xf))
        out = out + sg * self.shared_expert(xf)

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
                linear_attn_kq_heads=getattr(config, "linear_attn_kq_heads", 16),
                linear_attn_v_heads=getattr(config, "linear_attn_v_heads", 32),
                linear_attn_head_dim=getattr(config, "linear_attn_head_dim", 128),
                conv_kernel_size=getattr(config, "conv_kernel_size", 4),
                rms_norm_eps=rms_norm_eps,
            )

        # MoE FFN
        self.mlp = Qwen35MoE(
            hidden_size=hidden_size,
            intermediate_size=getattr(config, "moe_intermediate_size", config.intermediate_size),
            shared_intermediate_size=getattr(config, "shared_expert_intermediate_size", config.intermediate_size),
            num_experts=getattr(config, "num_experts", 256),
            top_k=getattr(config, "num_experts_per_tok", 8),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
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
            # LinearAttention needs 3D input (B, T, H)
            hidden_states, new_state, new_conv = self.linear_attn(
                hidden_states, state=state, conv_state=conv_state
            )
           
        # Post-attention norm with fused residual
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MoE FFN
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual, new_state, new_conv


# ─── Full Model ──────────────────────────────────────────────────────────────────


class Qwen35Model(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        num_layers = config.num_hidden_layers
        self.layer_types = _get_layer_types(config, num_layers)

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
