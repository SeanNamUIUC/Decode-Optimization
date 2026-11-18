# decode-opt: W4A16 decode kernels on RTX 5070 Ti

> Llama-3.2-1B, batch 1–8 decode. Triton kernels (fused RMSNorm, fp16 GEMV, fused int4-dequant GEMV) → TPOT −X% vs. HF + CUDA Graphs.
> *(fill in the headline number on Day 18)*

## Results (TL;DR)
| setup | batch 1 TPOT (ms) | batch 8 TPOT (ms) | perplexity |
|---|---|---|---|
| HF fp16 + CUDA Graph | | | |
| + my W4A16 GEMV | | | |

## 1. Why decode is memory-bound
roofline: bytes per token vs. FLOPs per token, measured peak BW (`results/peak_bw.json`).

## 2. Kernels
### 2.1 Fused RMSNorm + residual
### 2.2 fp16 GEMV (v1 → v2)
### 2.3 W4A16 GEMV (v1 → v3) — main
format, layout, split-K, autotune, dispatch; %peak per layer; ncu findings.

## 3. End-to-end
TPOT before/after, nsys breakdown before/after, why kernel speedup ≠ TPOT speedup.

## 4. Limits & next steps
decode attention (split-K / FlashDecoding), Hopper/Blackwell datacenter features (wgmma, TMA, tcgen05), vLLM integration.

## Reproduce
```bash
# CUDA 12.8+, PyTorch built for cu128 (sm_120 support)
pip install -r requirements.txt
huggingface-cli login                 # Llama-3.2-1B is gated

python -m bench.peak_bw               # measured 100% line
pytest -q                             # correctness (skips kernels not written yet)
python -m bench.run_all               # microbenchmarks -> results/bench_*.csv
python -m bench.plot                  # plots -> results/*.png
python -m e2e.measure_tpot --linear none
python -m e2e.measure_tpot --linear w4a16 --norm
python -m e2e.perplexity --linear w4a16_torch
bash profiling/nsys_decode.sh w4a16 --norm
```

## Layout
```
kernels/     Triton kernels (rmsnorm, gemv_fp16, gemv_w4a16)
quant/       W4A16 format: quantize, pack, dequant (source of truth)
reference/   PyTorch ground truth
specs/       model shapes + OpSpec registry (add a kernel version = one line)
bench/       harness, peak BW, run_all, plots
tests/       correctness on every real decode shape
e2e/         HF module patching, TPOT, perplexity
profiling/   nsys / ncu scripts
results/     CSVs and plots (committed)
NOTES.md     daily lab notebook
```
