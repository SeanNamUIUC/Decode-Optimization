"""Plots for the README from a bench CSV.

    python -m bench.plot                       # newest results/bench_*.csv
    python -m bench.plot results/bench_X.csv

Writes per op:
  results/<op>_pct_peak.png   -- %peak bandwidth per layer, one bar per impl (M=1)
  results/<op>_vs_batch.png   -- time vs batch per impl, one panel per layer
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


def newest_csv() -> Path:
    files = sorted(RESULTS.glob("bench_*.csv"))
    if not files:
        sys.exit("no results/bench_*.csv yet -- run `python -m bench.run_all` first")
    return files[-1]


def plot_pct_peak(df: pd.DataFrame, op: str) -> None:
    d = df[(df.op == op) & (df.M == df.M.min())]
    if d.empty or d.pct_peak.isna().all():
        return
    t = d.pivot_table(index="layer", columns="impl", values="pct_peak")
    ax = t.plot.bar(figsize=(9, 4), rot=0)
    ax.set_ylabel("% of measured peak DRAM bandwidth")
    ax.set_ylim(0, 100)
    ax.set_title(f"{op}: bandwidth efficiency (M={int(d.M.min())})")
    plt.tight_layout()
    plt.savefig(RESULTS / f"{op}_pct_peak.png", dpi=150)
    plt.close()


def plot_vs_batch(df: pd.DataFrame, op: str) -> None:
    d = df[df.op == op]
    layers = list(dict.fromkeys(d.layer))
    fig, axes = plt.subplots(1, len(layers), figsize=(3 * len(layers), 3), squeeze=False)
    for ax, layer in zip(axes[0], layers):
        for impl, g in d[d.layer == layer].groupby("impl"):
            ax.plot(g.M, g.time_us, marker="o", label=impl)
        ax.set_title(layer)
        ax.set_xlabel("batch (M)")
    axes[0][0].set_ylabel("time (us)")
    axes[0][-1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(RESULTS / f"{op}_vs_batch.png", dpi=150)
    plt.close()


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_csv()
    df = pd.read_csv(path)
    df["pct_peak"] = pd.to_numeric(df["pct_peak"], errors="coerce")
    for op in df.op.unique():
        plot_pct_peak(df, op)
        plot_vs_batch(df, op)
    print(f"plots from {path.name} -> {RESULTS}")


if __name__ == "__main__":
    main()
