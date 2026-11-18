"""PyTorch reference implementations = the ground truth for every test.

Every kernel in kernels/ must have exactly the same signature and return
values as its reference here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from quant.w4 import GROUP_SIZE, dequantize_w4


def rmsnorm_residual(x: torch.Tensor, residual: torch.Tensor | None, weight: torch.Tensor,
                     eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """h = x + residual (or x if residual is None); out = RMSNorm(h) * weight.

    x, residual: [M, N] fp16; weight: [N] fp16.
    Returns (out [M, N] fp16, h [M, N] fp16) -- h is the new residual stream.
    Matches HF LlamaRMSNorm numerics (fp32 statistics).
    """
    h = x if residual is None else x + residual
    hf = h.float()
    var = hf.pow(2).mean(dim=-1, keepdim=True)
    out = (hf * torch.rsqrt(var + eps)).to(x.dtype) * weight
    return out, h


def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x: [M, K] fp16, w: [N, K] fp16 -> [M, N] fp16. (cuBLAS)"""
    return F.linear(x, w)


def w4a16_linear(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor,
                 group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Unfused baseline: dequantize the whole weight to fp16, then cuBLAS."""
    return F.linear(x, dequantize_w4(qweight, scales, group_size))
