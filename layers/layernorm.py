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
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
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
