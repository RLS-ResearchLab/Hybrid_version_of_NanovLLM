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
        state: torch.Tensor | None = None,
        conv_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, T, H) or (N, H) flattened
            state: (B, LVH, LHD, LHD) recurrent state or None
            conv_state: (B, QKV, CK-1) conv history or None

        Returns:
            output: (B, T, H)
            new_state: (B, LVH, LHD, LHD) detached
            new_conv_state: (B, QKV, CK-1) detached
        """
        # Ensure 3D input
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(0)
        B, T, _ = hidden_states.shape

        # Projections
        z = self.in_proj_z(hidden_states)       # (B, T, LVH*LHD)
        a = self.in_proj_a(hidden_states)       # (B, T, LVH)
        b = self.in_proj_b(hidden_states)       # (B, T, LVH)
        qkv = self.in_proj_qkv(hidden_states)   # (B, T, QKV)

        QKV = self.qkv_dim

        # Causal conv1d with state management
        qkv_t = qkv.transpose(1, 2)  # (B, QKV, T)
        if conv_state is not None and T == 1:
            # Decode path: single token, use conv state directly
            combined = torch.cat([conv_state, qkv_t], dim=2)  # (B, QKV, CK)
            new_conv = combined[:, :, -(self.ck - 1):].detach()
            qkv_conv = F.conv1d(
                combined, self.conv1d.weight, self.conv1d.bias,
                padding=0, groups=QKV
            ).transpose(1, 2)  # (B, 1, QKV)
        else:
            # Prefill path (or first call without state)
            if conv_state is not None:
                qkv_t = torch.cat([conv_state, qkv_t], dim=2)
            qkv_conv = self.conv1d(qkv_t)
            new_conv = qkv_t[:, :, -(self.ck - 1):].detach()
            offset = conv_state.shape[2] if conv_state is not None else 0
            qkv_conv = qkv_conv[:, :, offset:offset + T].transpose(1, 2)  # (B, T, QKV)

        qkv_conv = F.silu(qkv_conv)

        # Split Q, K, V
        lkh, lvh, lhd = self.lkh, self.lvh, self.lhd
        q = qkv_conv[:, :, :lkh * lhd].view(B, T, lkh, lhd)
        k = qkv_conv[:, :, lkh * lhd:2 * lkh * lhd].view(B, T, lkh, lhd)
        v = qkv_conv[:, :, 2 * lkh * lhd:].view(B, T, lvh, lhd)

        # GDR gating: g = -exp(A_log) * softplus(a + dt_bias)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        beta = b.sigmoid()

        # Head expansion: LKH → LVH
        q = q.repeat_interleave(self.head_expand, dim=2)
        k = k.repeat_interleave(self.head_expand, dim=2)

        # L2-normalize Q and K, scale Q
        scale = lhd ** -0.5
        q = l2norm(q.float()) * scale
        k = l2norm(k.float())

        # GDR scan in float32
        dt = hidden_states.dtype
        g = g.float()
        beta = beta.float()
        v = v.float()

        if state is None:
            S = torch.zeros(B, lvh, lhd, lhd, device=hidden_states.device, dtype=torch.float32)
        else:
            S = state.float()

        ys = []
        for t in range(T):
            g_t = g[:, t, :].exp().unsqueeze(-1).unsqueeze(-1)   # (B, LVH, 1, 1)
            beta_t = beta[:, t, :].unsqueeze(-1)                 # (B, LVH, 1)
            k_t = k[:, t, :, :]                                   # (B, LVH, LHD)
            v_t = v[:, t, :, :]                                   # (B, LVH, LHD)
            q_t = q[:, t, :, :]                                   # (B, LVH, LHD)
            S = S * g_t                                            # decay state
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)          # (B, LVH, LHD)
            delta = (v_t - kv_mem) * beta_t                        # (B, LVH, LHD)
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)       # rank-1 update
            ys.append((S * q_t.unsqueeze(-1)).sum(dim=-2))         # (B, LVH, LHD)

        new_state = S.detach()
        y = torch.stack(ys, dim=1).to(dt)  # (B, T, LVH, LHD) → cast back to model dtype

        # Gated RMSNorm
        y = y.reshape(B * T * lvh, lhd)
        z_flat = z.reshape(B * T * lvh, lhd)
        y = self.norm(y, z_flat)
        y = y.reshape(B, T, lvh * lhd)

        return self.out_proj(y), new_state, new_conv


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
            # Flatten back if needed (for consistency with the rest of the pipeline)
            if hidden_states.ndim == 3 and hidden_states.shape[0] == 1:
                hidden_states = hidden_states.squeeze(0)

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
                positions, hidden_states, residual,
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
        states: list | None = None,
        conv_states: list | None = None,
    ) -> tuple[torch.Tensor, list, list]:
        return self.model(input_ids, positions, states, conv_states)

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
