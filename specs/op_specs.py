"""One OpSpec per operation: what to run, how to check it, how to score it.

To add a new kernel version, add one entry to the op's `impls` dict, e.g.
    "triton_v2": "kernels.gemv_w4a16:gemv_w4a16_v2"
Tests and benchmarks pick it up automatically.

Impl strings are "module:function". If that module also defines
`prepare(args) -> args`, it runs once per case OUTSIDE timing (use it for
weight re-layout). It must not depend on the activation x (args[0]).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from quant.w4 import GROUP_SIZE, quantize_w4
from reference import ops as ref_ops
from specs.model_shapes import ModelConfig, linear_cases, rmsnorm_cases

FP16 = 2  # bytes


@dataclass
class Impl:
    fn: Callable
    prepare: Callable | None = None


@dataclass
class OpSpec:
    name: str
    cases: Callable[[ModelConfig, Sequence[int]], list[dict]]
    make_inputs: Callable[[dict], tuple]
    ref: Callable
    bytes_moved: Callable[[dict], int]
    flops: Callable[[dict], int]
    impls: dict[str, str | Callable] = field(default_factory=dict)
    atol: float = 1e-2
    rtol: float = 1e-2

    def load_impls(self) -> dict[str, Impl]:
        loaded = {}
        for name, target in self.impls.items():
            if callable(target):
                loaded[name] = Impl(target)
                continue
            mod_name, fn_name = target.split(":")
            mod = importlib.import_module(mod_name)
            loaded[name] = Impl(getattr(mod, fn_name), getattr(mod, "prepare", None))
        return loaded


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------- rmsnorm
def _rmsnorm_inputs(c: dict) -> tuple:
    g = _gen()
    M, N = c["M"], c["N"]
    x = torch.randn(M, N, device="cuda", dtype=torch.float16, generator=g)
    res = torch.randn(M, N, device="cuda", dtype=torch.float16, generator=g)
    w = (1 + 0.1 * torch.randn(N, device="cuda", generator=g)).to(torch.float16)
    return (x, res, w, 1e-5)


RMSNORM = OpSpec(
    name="rmsnorm_residual",
    cases=rmsnorm_cases,
    make_inputs=_rmsnorm_inputs,
    ref=ref_ops.rmsnorm_residual,
    # read x, residual, weight; write out, new residual
    bytes_moved=lambda c: (4 * c["M"] * c["N"] + c["N"]) * FP16,
    flops=lambda c: 4 * c["M"] * c["N"],
    impls={
        "torch": ref_ops.rmsnorm_residual,
        "triton": "kernels.rmsnorm:rmsnorm_residual",
    },
)


# ---------------------------------------------------------------- fp16 GEMV
def _linear_inputs(c: dict) -> tuple:
    g = _gen()
    M, K, N = c["M"], c["K"], c["N"]
    x = torch.randn(M, K, device="cuda", dtype=torch.float16, generator=g)
    w = (torch.randn(N, K, device="cuda", generator=g) / K**0.5).to(torch.float16)
    return (x, w)


GEMV_FP16 = OpSpec(
    name="gemv_fp16",
    cases=linear_cases,
    make_inputs=_linear_inputs,
    ref=ref_ops.linear,
    bytes_moved=lambda c: (c["N"] * c["K"] + c["M"] * c["K"] + c["M"] * c["N"]) * FP16,
    flops=lambda c: 2 * c["M"] * c["N"] * c["K"],
    impls={
        "cublas": ref_ops.linear,
        "triton_v1": "kernels.gemv_fp16:gemv",
    },
)


# ---------------------------------------------------------------- W4A16 GEMV
def _w4_inputs(c: dict) -> tuple:
    x, w = _linear_inputs(c)
    qweight, scales = quantize_w4(w, GROUP_SIZE)
    return (x, qweight, scales, GROUP_SIZE)


def _w4_bytes(c: dict) -> int:
    M, K, N = c["M"], c["K"], c["N"]
    weight = N * K // 2
    scales = N * (K // GROUP_SIZE) * FP16
    return weight + scales + (M * K + M * N) * FP16


GEMV_W4A16 = OpSpec(
    name="gemv_w4a16",
    cases=linear_cases,
    make_inputs=_w4_inputs,
    ref=ref_ops.w4a16_linear,
    bytes_moved=_w4_bytes,
    flops=lambda c: 2 * c["M"] * c["N"] * c["K"],
    impls={
        "torch_unfused": ref_ops.w4a16_linear,
        "triton_v1": "kernels.gemv_w4a16:gemv_w4a16",
    },
    atol=2e-2,
    rtol=2e-2,
)


ALL_SPECS = {s.name: s for s in (RMSNORM, GEMV_FP16, GEMV_W4A16)}
