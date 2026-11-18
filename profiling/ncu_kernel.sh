#!/usr/bin/env bash
# Deep-dive one kernel with Nsight Compute.
#   bash profiling/ncu_kernel.sh <kernel-name-regex> <op> [impl]
#   bash profiling/ncu_kernel.sh _gemv_w4a16 gemv_w4a16 triton_v1
# Open the .ncu-rep in the Nsight Compute GUI; check Memory Workload Analysis
# (achieved DRAM throughput) and Warp State Statistics (stall reasons).
set -euo pipefail
KERNEL=$1; OP=$2; IMPL=${3:-triton_v1}
mkdir -p results
ncu --set full -k "regex:${KERNEL}" -c 3 -o "results/ncu_${OP}_${IMPL}" --force-overwrite \
  python -m bench.run_all --ops "$OP" --impls "$IMPL" --batches 1 --no-check --tag ncu
