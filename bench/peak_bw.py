"""Measure achievable DRAM bandwidth on this GPU (device-to-device copy).

Use this measured number -- not the spec sheet -- as the 100% line for
%peak in every benchmark.

    python -m bench.peak_bw          # writes results/peak_bw.json
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from bench.harness import bench_us

OUT = Path(__file__).resolve().parents[1] / "results" / "peak_bw.json"


def main(size_mib: int = 1024) -> float:
    n = size_mib * 1024 * 1024 // 2
    src = torch.empty(n, device="cuda", dtype=torch.float16).normal_()
    dst = torch.empty_like(src)
    t_us = bench_us(lambda: dst.copy_(src), warmup=10, rep=50)
    gbps = 2 * src.numel() * 2 / (t_us * 1e-6) / 1e9  # read + write
    OUT.parent.mkdir(exist_ok=True)
    info = {"device": torch.cuda.get_device_name(), "size_mib": size_mib, "peak_gbps": round(gbps, 1)}
    OUT.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return gbps


if __name__ == "__main__":
    main()
