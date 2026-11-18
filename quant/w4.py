"""W4A16 reference format (the canonical layout every kernel receives).

Weight W: [N, K] fp16 (nn.Linear layout: out_features x in_features).

Quantization: symmetric int4, per group of `group_size` along K.
    scale[n, g] = max(|W[n, g*G:(g+1)*G]|) / 7
    q = clamp(round(W / scale), -8, 7) + 8          # stored as 0..15

Packing: 8 nibbles per int32 along K.
    qweight[n, j] bit [4i, 4i+4) holds q[n, 8*j + i],  i = 0..7
    qweight: int32 [N, K // 8]
    scales:  fp16  [N, K // G]

Dequant:  W_hat[n, k] = (((qweight[n, k // 8] >> (4 * (k % 8))) & 0xF) - 8) * scales[n, k // G]

If your kernel wants a different layout (e.g. pre-shuffled for coalesced
loads, Day 10), keep this file as the source of truth and convert in your
kernel module's `prepare(args)` function. See specs/op_specs.py.
"""
from __future__ import annotations

import torch

GROUP_SIZE = 128
PACK = 8  # int4 values per int32


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """q: integer tensor [N, K] with values in 0..15 -> int32 [N, K // 8]."""
    N, K = q.shape
    assert K % PACK == 0, f"K={K} must be divisible by {PACK}"
    q = q.to(torch.int64).reshape(N, K // PACK, PACK)
    shifts = torch.arange(PACK, device=q.device, dtype=torch.int64) * 4
    packed = (q << shifts).sum(dim=-1)  # fits in uint32 range
    packed = torch.where(packed >= 2**31, packed - 2**32, packed)
    return packed.to(torch.int32).contiguous()


def unpack_int4(qweight: torch.Tensor) -> torch.Tensor:
    """int32 [N, K // 8] -> int64 [N, K] with values in 0..15."""
    N, Kp = qweight.shape
    shifts = torch.arange(PACK, device=qweight.device, dtype=torch.int64) * 4
    q = (qweight.to(torch.int64)[..., None] >> shifts) & 0xF
    return q.reshape(N, Kp * PACK)


def quantize_w4(w: torch.Tensor, group_size: int = GROUP_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """w: [N, K] float -> (qweight int32 [N, K//8], scales fp16 [N, K//group_size])."""
    N, K = w.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    assert group_size % PACK == 0
    wg = w.float().reshape(N, K // group_size, group_size)
    scales = (wg.abs().amax(dim=-1) / 7.0).clamp(min=1e-8)
    q = torch.round(wg / scales[..., None]).clamp(-8, 7) + 8
    qweight = pack_int4(q.reshape(N, K))
    return qweight, scales.to(torch.float16).contiguous()


def dequantize_w4(qweight: torch.Tensor, scales: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """-> fp16 [N, K]."""
    q = unpack_int4(qweight) - 8
    N, K = q.shape
    w = q.reshape(N, K // group_size, group_size).float() * scales.float()[..., None]
    return w.reshape(N, K).to(torch.float16)
