#!/usr/bin/env bash
# Day 1 / Day 17 -- where does decode time go?
#   bash profiling/nsys_decode.sh                 # baseline
#   bash profiling/nsys_decode.sh w4a16 --norm    # patched
set -euo pipefail
LINEAR=${1:-none}; shift || true
OUT=results/nsys_decode_${LINEAR}
mkdir -p results
nsys profile -t cuda,nvtx --cuda-graph-trace=node -o "$OUT" --force-overwrite true \
  python -m e2e.measure_tpot --linear "$LINEAR" "$@" --batches 1 --ctx 512 --new-tokens 64 --runs 1
# per-kernel time summary (copy the top rows into NOTES.md)
nsys stats --report cuda_gpu_kern_sum --format csv -o "$OUT" "$OUT.nsys-rep"
echo "-> ${OUT}_cuda_gpu_kern_sum.csv"
