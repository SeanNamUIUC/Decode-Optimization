"""Day 3 -- fused RMSNorm + residual add.  (YOU write this one.)

Contract (must match reference.ops.rmsnorm_residual exactly):
    rmsnorm_residual(x, residual, weight, eps) -> (out, h)
    x, residual: [M, N] fp16 contiguous (residual may be None)
    weight:      [N] fp16
    h   = x + residual            (new residual stream, fp16)
    out = h * rsqrt(mean(h^2) + eps) * weight   (statistics in fp32)

Goal: one read of x/residual/weight, one write of out/h.
Hint: one program per row; N=2048 fits in one BLOCK (next_power_of_2).
"""
import torch
import triton
import triton.language as tl


# TODO(you): write the kernel
# @triton.jit
# def _rmsnorm_residual_kernel(x_ptr, res_ptr, w_ptr, out_ptr, h_ptr,
#                              N, stride, eps, HAS_RESIDUAL: tl.constexpr,
#                              BLOCK: tl.constexpr):
#     row = tl.program_id(0)
#     ...


def rmsnorm_residual(x: torch.Tensor, residual: torch.Tensor | None, weight: torch.Tensor,
                     eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("kernels/rmsnorm.py: write the Triton kernel")
