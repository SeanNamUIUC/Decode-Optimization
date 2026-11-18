"""Run every (or selected) OpSpec and save one CSV.

    python -m bench.run_all                              # all ops, batches 1/2/4/8
    python -m bench.run_all --ops gemv_w4a16 --batches 1 4
    python -m bench.run_all --ops gemv_fp16 --impls cublas triton_v1
"""
from __future__ import annotations

import argparse

from bench.harness import load_peak_gbps, print_table, run_spec, save_csv
from specs.model_shapes import DEFAULT_BATCHES, LLAMA_3_2_1B
from specs.op_specs import ALL_SPECS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ops", nargs="*", default=list(ALL_SPECS), choices=list(ALL_SPECS))
    p.add_argument("--impls", nargs="*", default=None)
    p.add_argument("--batches", nargs="*", type=int, default=list(DEFAULT_BATCHES))
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--tag", default="bench")
    a = p.parse_args()

    peak = load_peak_gbps()
    rows = []
    for name in a.ops:
        rows += run_spec(ALL_SPECS[name], LLAMA_3_2_1B, a.batches, peak, a.impls, check=not a.no_check)
    print_table(rows)
    if rows:
        print(f"\nsaved -> {save_csv(rows, a.tag)}")


if __name__ == "__main__":
    main()
