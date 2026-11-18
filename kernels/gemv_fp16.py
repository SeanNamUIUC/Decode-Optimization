"""Days 4-5 -- fp16 GEMV for decode (M = 1..8).  (YOU write this one.)

Contract (must match reference.ops.linear):
    gemv(x, w) -> y
    x: [M, K] fp16, w: [N, K] fp16 (nn.Linear layout), y: [M, N] fp16
    accumulate in fp32

v1 idea: grid over N blocks; each program loads a [BLOCK_N, BLOCK_K] tile of w,
loops over K, and for each of the M rows does multiply + tl.sum.
(tl.dot needs M >= 16 -- compare against padding M to 16.)

v2 ideas: autotune BLOCK_N/BLOCK_K/num_warps/num_stages, split-K for small N.
Add new versions as new functions and register them in specs/op_specs.py.
"""
import torch
import triton
import triton.language as tl


# TODO(you): write the kernel
# @triton.jit
# def _gemv_kernel(x_ptr, w_ptr, y_ptr, M, N, K, ...,
#                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
#     pid_n = tl.program_id(0)
#     ...


def gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("kernels/gemv_fp16.py: write the Triton kernel")
