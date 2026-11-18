"""Shared benchmark harness: correctness check -> timing -> roofline numbers -> CSV."""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Callable, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def bench_us(fn: Callable, warmup: int = 25, rep: int = 100) -> float:
    """Median runtime in microseconds."""
    try:
        from triton.testing import do_bench
        try:
            ms = do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
        except TypeError:  # older triton
            ms = do_bench(fn, warmup=warmup, rep=rep)
        return ms * 1e3
    except ImportError:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(rep):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        return times[len(times) // 2] * 1e3


def load_peak_gbps() -> float | None:
    p = RESULTS / "peak_bw.json"
    if p.exists():
        return json.loads(p.read_text())["peak_gbps"]
    print("[warn] results/peak_bw.json not found -> run `python -m bench.peak_bw` first. %peak left empty.")
    return None


def run_spec(spec, cfg, batches: Sequence[int], peak_gbps: float | None,
             only_impls: Sequence[str] | None = None, check: bool = True) -> list[dict]:
    from specs.model_shapes import case_label

    rows = []
    impls = spec.load_impls()
    if only_impls:
        impls = {k: v for k, v in impls.items() if k in only_impls}
    for case in spec.cases(cfg, batches):
        args = spec.make_inputs(case)
        ref_out = spec.ref(*args)
        for name, impl in impls.items():
            row = {"op": spec.name, "impl": name, "case": case_label(case), **case}
            try:
                run_args = impl.prepare(args) if impl.prepare else args
                out = impl.fn(*run_args)
            except NotImplementedError:
                continue  # kernel not written yet
            if check:
                try:
                    torch.testing.assert_close(out, ref_out, atol=spec.atol, rtol=spec.rtol)
                    row["correct"] = True
                except AssertionError as e:
                    row["correct"] = False
                    print(f"[FAIL] {spec.name}/{name} {row['case']}: {str(e).splitlines()[0]}")
            t_us = bench_us(lambda: impl.fn(*run_args))
            gbps = spec.bytes_moved(case) / (t_us * 1e-6) / 1e9
            row.update(
                time_us=round(t_us, 2),
                gbps=round(gbps, 1),
                pct_peak=round(100 * gbps / peak_gbps, 1) if peak_gbps else "",
                tflops=round(spec.flops(case) / (t_us * 1e-6) / 1e12, 3),
            )
            rows.append(row)
    return rows


def save_csv(rows: list[dict], tag: str = "bench") -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{tag}_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def print_table(rows: list[dict]) -> None:
    cols = ["op", "impl", "case", "time_us", "gbps", "pct_peak", "correct"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols} if rows else {}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
