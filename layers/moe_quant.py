import torch


def quantize_activation_per_row(x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric int8 activation quantization.

    Returns:
        qx: int8 tensor, same shape as x
        scales: float tensor of shape (x.shape[0],)
    """
    assert x.ndim == 2, f"expected 2D activations, got shape={tuple(x.shape)}"
    max_abs = x.abs().amax(dim=1)
    scales = (max_abs / 127.0).clamp_min(eps)
    qx = torch.round(x / scales.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return qx, scales


def quantize_weight_per_group(
    w: torch.Tensor,
    group_size: int = 128,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Per-output-row, per-input-group symmetric int8 weight quantization.

    Weight layout is (out_features, in_features).

    Returns:
        qw: int8 tensor, same shape as w
        scales: float tensor of shape (out_features, num_groups)
        group_size_used: effective group size (<= in_features)
    """
    assert w.ndim == 2, f"expected 2D weights, got shape={tuple(w.shape)}"
    out_features, in_features = w.shape
    g = min(max(int(group_size), 1), in_features)
    if in_features % g != 0:
        # Keep grouping simple/deterministic: fallback to one group per row.
        g = in_features
    num_groups = in_features // g
    wv = w.view(out_features, num_groups, g)
    max_abs = wv.abs().amax(dim=2)
    scales = (max_abs / 127.0).clamp_min(eps)
    qw = torch.round(wv / scales.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return qw.view_as(w), scales, g


def dequantize_weight(qw: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    """Dequantize per-group int8 weights back to float."""
    out_features, in_features = qw.shape
    num_groups = in_features // group_size
    qwv = qw.view(out_features, num_groups, group_size).float()
    return (qwv * scales.unsqueeze(-1)).reshape_as(qw)


def linear_w8a8(
    x: torch.Tensor,
    w: torch.Tensor,
    weight_group_size: int = 128,
    act_eps: float = 1e-6,
) -> torch.Tensor:
    """Functional W8A8 linear path (quantize x/w -> dequantized matmul).

    This prioritizes correctness and easy integration over kernel-level speed.
    """
    qx, xs = quantize_activation_per_row(x.float(), eps=act_eps)
    qw, ws, g = quantize_weight_per_group(w.float(), group_size=weight_group_size, eps=act_eps)
    x_hat = qx.float() * xs.unsqueeze(-1)
    w_hat = dequantize_weight(qw, ws, g)
    return x_hat @ w_hat.t()
