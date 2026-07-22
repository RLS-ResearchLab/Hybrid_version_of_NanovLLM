from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb


class PartialRotaryEmbedding(nn.Module):
    """Apply RoPE to only the first rotary_dim channels of each head.

    Unlike RotaryEmbedding which requires rotary_dim == head_size,
    this supports partial rotary (rotary_dim < head_size). The remaining
    (head_size - rotary_dim) channels pass through unchanged.

    Frequency base is computed from rotary_dim, not head_size.
    Matches src/model_small_qwen3.5.py build_rope / apply_rope:
        rot = int(DH * PROT)
        inv = 1 / theta^(arange(0, rot, 2) / rot)    ← note: divided by rot, not DH
        qr, qp = q[..., :rot], q[..., rot:]
        ...
        cat([qr*c + rot_half(qr)*s, qp], -1)
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        assert rotary_dim <= head_size, (
            f"rotary_dim ({rotary_dim}) must be <= head_size ({head_size})"
        )
        # Frequency base computed from rotary_dim, not head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        # Split into rotary and passthrough parts
        q_rot, q_pass = query[..., :self.rotary_dim], query[..., self.rotary_dim:]
        k_rot, k_pass = key[..., :self.rotary_dim], key[..., self.rotary_dim:]
        q_rot = apply_rotary_emb(q_rot, cos, sin)
        k_rot = apply_rotary_emb(k_rot, cos, sin)
        query = torch.cat((q_rot, q_pass), dim=-1)
        key = torch.cat((k_rot, k_pass), dim=-1)
        return query, key


@lru_cache(1)
def get_partial_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = PartialRotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
