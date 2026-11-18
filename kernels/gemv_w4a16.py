"""Days 8-11 -- MAIN KERNEL: fused int4 dequant + GEMV.  (YOU write this one.)

Contract (must match reference.ops.w4a16_linear):
    gemv_w4a16(x, qweight, scales, group_size) -> y
    x:       [M, K] fp16
    qweight: int32 [N, K // 8]   (layout documented in quant/w4.py)
    scales:  fp16  [N, K // group_size]
    y:       [M, N] fp16, accumulate in fp32

v1 (Day 8): load packed int32, unpack in registers ((q >> 4*i) & 0xF) - 8,
            multiply by the group's scale, FMA with x.
v2 (Day 10): change the weight layout for coalesced loads -> do it in
            `prepare` below (runs once, outside timing).
v3 (Day 11): split-K + per-shape autotune; batch-dependent dispatch.
"""
import torch
import triton
import triton.language as tl


# Optional: re-layout weights once before timing (Day 10).
# Must not use args[0] (x). Return args in the same order.
# def prepare(args):
#     x, qweight, scales, group_size = args
#     return (x, my_shuffle(qweight), scales, group_size)


# TODO(you): write the kernel
# @triton.jit
# def _gemv_w4a16_kernel(x_ptr, qw_ptr, s_ptr, y_ptr, M, N, K, ...,
#                        GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
#     ...


def gemv_w4a16(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor,
               group_size: int = 128) -> torch.Tensor:
    raise NotImplementedError("kernels/gemv_w4a16.py: write the Triton kernel")
