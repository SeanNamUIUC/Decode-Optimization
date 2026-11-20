"""Where do the bytes go in one decode step? (analytic, before measuring)

For each part of Llama-3.2-1B, count the DRAM bytes one decode step must move
(weights, KV cache, activations) and the lower-bound time at peak bandwidth.
Small-batch decode is memory-bound, so bytes ~ time. Confirm with nsys
(profiling/nsys_decode.sh) before trusting the percentages.

    python -m analysis.decode_breakdown                     # batch 1, ctx 512
    python -m analysis.decode_breakdown --batch 8 --ctx 2048
    python -m analysis.decode_breakdown --w4a16             # linears in int4
    python -m analysis.decode_breakdown --markdown          # table for README
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from specs.model_shapes import LLAMA_3_2_1B, ModelConfig, linear_layers

FP16 = 2
SPEC_PEAK_GBPS = 896.0  # RTX 5070 Ti spec sheet; overridden by results/peak_bw.json
GROUP = 128


def weight_bytes(K: int, N: int, w4a16: bool) -> int:
    if w4a16:
        return N * K // 2 + N * (K // GROUP) * FP16  # int4 + fp16 scales
    return N * K * FP16


def breakdown(cfg: ModelConfig, batch: int, ctx: int, w4a16: bool) -> list[tuple[str, int]]:
    """(part, bytes for the WHOLE model per decode step). Activations counted
    as an ideal fused read+write; they are tiny next to weights."""
    B, H, L = batch, cfg.hidden, cfg.n_layers
    kv_dim = cfg.n_kv_heads * cfg.head_dim
    act = lambda n: B * n * FP16  # one [B, n] fp16 activation

    parts: dict[str, int] = {}
    for name, (K, N) in linear_layers(cfg).items():
        group = "attn linears (q,k,v,o)" if name in ("q_proj", "k_proj", "v_proj", "o_proj") \
            else "mlp linears (gate,up,down)"
        parts[group] = parts.get(group, 0) + L * (weight_bytes(K, N, w4a16) + act(K) + act(N))

    # attention core: read K and V cache for ctx tokens, write the new token's K,V
    kv_read = 2 * B * ctx * kv_dim * FP16
    kv_write = 2 * B * kv_dim * FP16
    q_and_out = act(cfg.n_heads * cfg.head_dim) * 2
    parts["attention (KV cache)"] = L * (kv_read + kv_write + q_and_out)

    # 2 RMSNorm per layer + final norm: read x (+residual), weight; write out (+h)
    norm = act(H) * 4 + H * FP16
    parts["RMSNorm + residual"] = (2 * L + 1) * norm

    # SiLU(gate) * up: read 2 x [B, I], write [B, I]
    parts["SiLU * up (elementwise)"] = L * act(cfg.intermediate) * 3

    # lm_head stays fp16 (tied with the embedding); embedding lookup reads B rows
    parts["lm_head (fp16, tied)"] = weight_bytes(H, cfg.vocab, False) + act(H) + B * cfg.vocab * FP16
    parts["embedding lookup"] = act(H) * 2
    return sorted(parts.items(), key=lambda kv: -kv[1])


def load_peak() -> tuple[float, str]:
    p = Path(__file__).resolve().parents[1] / "results" / "peak_bw.json"
    if p.exists():
        return json.loads(p.read_text())["peak_gbps"], "measured"
    return SPEC_PEAK_GBPS, "spec"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--w4a16", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()

    peak, src = load_peak()
    rows = breakdown(LLAMA_3_2_1B, a.batch, a.ctx, a.w4a16)
    total = sum(b for _, b in rows)
    title = (f"decode step, batch={a.batch}, ctx={a.ctx}, "
             f"linears={'W4A16' if a.w4a16 else 'fp16'}, peak={peak:.0f} GB/s ({src})")

    if a.markdown:
        print(f"**{title}**\n")
        print("| part | MB per step | share | lower-bound time (ms) |")
        print("| --- | ---: | ---: | ---: |")
        for name, b in rows:
            print(f"| {name} | {b / 1e6:,.1f} | {100 * b / total:.1f}% | {b / (peak * 1e9) * 1e3:.3f} |")
        print(f"| **total** | **{total / 1e6:,.1f}** | 100% | **{total / (peak * 1e9) * 1e3:.3f}** |")
        return

    print(title)
    print(f"{'part':30s} {'MB':>10s} {'share':>7s} {'min ms':>8s}")
    for name, b in rows:
        print(f"{name:30s} {b / 1e6:10.1f} {100 * b / total:6.1f}% {b / (peak * 1e9) * 1e3:8.3f}")
    print(f"{'TOTAL':30s} {total / 1e6:10.1f} {'100%':>7s} {total / (peak * 1e9) * 1e3:8.3f}")
    print(f"\n=> TPOT lower bound ~{total / (peak * 1e9) * 1e3:.2f} ms "
          f"(memory-bound; real TPOT adds launch overhead and non-ideal kernels)")


if __name__ == "__main__":
    main()