"""Swap HF Llama modules for modules that call our kernels.

Linear modes:
  none         -- untouched (fp16 cuBLAS)
  fp16         -- Fp16GemvLinear: kernels.gemv_fp16 for decode (M <= SMALL_M)
  w4a16_torch  -- int4 weights, unfused dequant + cuBLAS (accuracy reference)
  w4a16        -- W4A16Linear: kernels.gemv_w4a16 for decode (M <= SMALL_M)

Prefill (M > SMALL_M) falls back to cuBLAS so only decode uses the GEMV
kernels -- that is what TPOT measures.

patch_norm=True also swaps every *RMSNorm for kernels.rmsnorm (residual=None:
HF adds the residual outside the norm; fusing it needs a custom decoder layer,
a stretch goal).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant.w4 import GROUP_SIZE, dequantize_w4, quantize_w4

SMALL_M = 16


class Fp16GemvLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        assert linear.bias is None
        from kernels.gemv_fp16 import gemv
        self._gemv = gemv
        self.weight = linear.weight

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        y = self._gemv(x2, self.weight) if x2.shape[0] <= SMALL_M else F.linear(x2, self.weight)
        return y.reshape(*shape[:-1], -1)


class W4A16Linear(nn.Module):
    def __init__(self, linear: nn.Linear, use_kernel: bool = True, group_size: int = GROUP_SIZE):
        super().__init__()
        assert linear.bias is None
        self.group_size = group_size
        self.use_kernel = use_kernel
        qweight, scales = quantize_w4(linear.weight.data, group_size)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        if use_kernel:
            import kernels.gemv_w4a16 as k
            self._kernel = k.gemv_w4a16
            prepare = getattr(k, "prepare", None)
            if prepare is not None:
                _, qp, sp, _ = prepare((None, qweight, scales, group_size))
            else:
                qp, sp = qweight, scales
            self.register_buffer("qweight_k", qp)
            self.register_buffer("scales_k", sp)

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if self.use_kernel and x2.shape[0] <= SMALL_M:
            y = self._kernel(x2, self.qweight_k, self.scales_k, self.group_size)
        else:
            y = F.linear(x2, dequantize_w4(self.qweight, self.scales, self.group_size))
        return y.reshape(*shape[:-1], -1)


class FusedRMSNorm(nn.Module):
    def __init__(self, norm: nn.Module):
        super().__init__()
        from kernels.rmsnorm import rmsnorm_residual
        self._kernel = rmsnorm_residual
        self.weight = norm.weight
        self.eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5))

    def forward(self, x):
        shape = x.shape
        out, _ = self._kernel(x.reshape(-1, shape[-1]), None, self.weight, self.eps)
        return out.reshape(shape)


def _replace(root: nn.Module, pred, make) -> int:
    count = 0
    for name, child in list(root.named_children()):
        if pred(child):
            setattr(root, name, make(child))
            count += 1
        else:
            count += _replace(child, pred, make)
    return count


@torch.no_grad()
def patch_model(model: nn.Module, linear_mode: str = "none", patch_norm: bool = False) -> nn.Module:
    layers = model.model.layers  # decoder layers only; lm_head / embeddings untouched
    is_linear = lambda m: isinstance(m, nn.Linear)
    if linear_mode == "fp16":
        n = _replace(layers, is_linear, Fp16GemvLinear)
    elif linear_mode == "w4a16":
        n = _replace(layers, is_linear, lambda m: W4A16Linear(m, use_kernel=True))
    elif linear_mode == "w4a16_torch":
        n = _replace(layers, is_linear, lambda m: W4A16Linear(m, use_kernel=False))
    elif linear_mode == "none":
        n = 0
    else:
        raise ValueError(linear_mode)
    m = 0
    if patch_norm:
        m = _replace(model, lambda mod: type(mod).__name__.endswith("RMSNorm"), FusedRMSNorm)
    print(f"[patch] linear_mode={linear_mode}: {n} linears, {m} norms replaced")
    torch.cuda.empty_cache()
    return model
