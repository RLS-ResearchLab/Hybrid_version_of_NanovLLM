import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Out-of-place throughout: when x arrives as float32, x.float() is a
        # documented PyTorch no-op that returns the SAME tensor object (no
        # copy), so an in-place .mul_ here would silently mutate the
        # caller's own input tensor. Not just a paranoia guard -- this
        # module is used in fp32 test/CPU configs (see add_rms_forward's
        # analogous bug, which this mirrors).
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x.to(orig_dtype) * self.weight
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Out-of-place throughout -- see rms_forward's comment. The in-place
        # version of this method had a real bug: when x arrives as float32,
        # `x.float()` and the later `x.to(orig_dtype)` are both no-ops (same
        # tensor, no copy), so `residual = x.to(orig_dtype)` aliased `x`
        # rather than snapshotting it -- the subsequent in-place
        # rsqrt/weight-scale mutations then overwrote `residual` too, so
        # callers got back the fully-normalized-and-weighted value as
        # `residual` instead of the pre-norm sum needed for the next
        # layer's skip connection. Only manifests when the working dtype is
        # float32 (bf16/fp16 inputs make x.float() a real copy, which
        # happened to mask this). Qwen35RMSNorm.add_rms_forward already
        # used this same out-of-place shape and was never affected.
        orig_dtype = x.dtype
        x = x.float() + residual.float()
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x.to(orig_dtype) * self.weight
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)


class Qwen35RMSNorm(nn.Module):
    """RMSNorm with (1 + weight) scaling, weight initialized to zeros.

    Matches src/model_small_qwen3.5.py RMSNorm:
        x_float = x.float()
        x_normed = x_float * rsqrt(mean(x_float^2) + eps)
        return (x_normed * (1.0 + weight.float())).to(original_dtype)
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * (1.0 + self.weight.float())).to(dt)

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(dt)
        x = x * x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x = (x * (1.0 + self.weight.float())).to(dt)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)


class Qwen35RMSNormGated(nn.Module):
    """Gated RMSNorm: norm(x) -> weight * x -> x * silu(gate).

    Matches src/model_small_qwen3.5.py RMSNormGated exactly,
    including the casting order: cast to dt before weight multiply,
    then back to float for silu.

        x_float = x.float()
        x_normed = x_float * rsqrt(mean(x_float^2) + eps)
        x = (weight.to(dt) * x_normed.to(dt)).float()
        return (x * silu(gate.float())).to(dt)
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x = (self.weight.to(dt) * x.to(dt)).float()
        return (x * F.silu(gate.float())).to(dt)

    # DECODE-ONLY compiled variant, added 2026-08-28 -- forward() above stays
    # UNCOMPILED because the same nn.Module instance is also called from the
    # sequential per-segment PREFILL scan (models/qwen3_5.py's
    # Qwen35LinearAttention, seg_y = self.norm(seg_y, seg_z_flat)) with
    # arbitrary, continuously-varying per-segment token counts -- compiling
    # the shared method would expose it to effectively unbounded distinct
    # shapes there, defeating the whole point of a bounded shape budget (see
    # below) and likely exhausting torch._dynamo's cache_size_limit on real
    # traffic. This method is called ONLY from
    # _forward_decode_batched/_forward_decode_fused_gdr
    # (models/qwen3_5.py), which run exclusively inside the captured
    # CUDA-graph region at exactly one of the graph's fixed bucket sizes
    # (never an arbitrary live batch count -- padding rows fill the rest).
    # max_num_seqs=64 is a confirmed-fixed ceiling (this project will never
    # scale concurrency higher), so the bucket list
    # [1,2,4,8,16,32,48,64] is fixed at exactly 8 distinct shapes forever --
    # this adds at most 8 compiled variants to torch._dynamo's shared
    # budget, not an open-ended amount.
    #
    # That budget itself was the site of a real, measured regression once
    # before: tests/moe_int8_quantize.py's dequantize_weight_int8_grouped
    # had @torch.compile tried 2026-08-21 and reverted same day (37.1 ->
    # 33.0 tok/s), suspected cause being torch._dynamo's then-default
    # cache_size_limit=8 exhausted by that function's own per-bucket shape
    # variation competing with RMSNorm/Qwen35RMSNorm's compiled methods for
    # the same limit. SMELL-11 (2026-08-27/28) later bumped it 8->64.
    # GPU-UNVALIDATED -- confirm the `[DYNAMO] recompile_limit=...`
    # diagnostic still shows no "hit config.recompile_limit" warning after
    # adding this (and the decode-only l2norm variant, models/qwen3_5.py)
    # on top of the existing compiled functions sharing this budget.
    @torch.compile
    def forward_decode_compiled(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(x, gate)
