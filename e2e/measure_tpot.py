"""Day 1 / Day 16 -- TTFT and TPOT with static KV cache + CUDA Graphs.

    python -m e2e.measure_tpot                                   # baseline
    python -m e2e.measure_tpot --linear w4a16 --norm             # your kernels
    python -m e2e.measure_tpot --batches 1 4 8 --ctx 512 2048 --no-compile

Method: T(1) = generate 1 new token (~prefill = TTFT),
        T(n) = generate n new tokens,
        TPOT = (T(n) - T(1)) / (n - 1).
Results are appended to results/tpot.csv.

Without CUDA Graphs, decode at batch 1 is dominated by kernel-launch
overhead and kernel improvements barely show -- keep compile on.
Llama-3.2-1B on HF is gated: `huggingface-cli login` first, or pass an
ungated mirror with --model.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from e2e.patch_hf import patch_model

RESULTS = Path(__file__).resolve().parents[1] / "results"


def build(model_name: str, linear: str, norm: bool, compile_: bool):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda().eval()
    patch_model(model, linear, norm)
    model.generation_config.cache_implementation = "static"
    if compile_:
        torch._dynamo.config.cache_size_limit = 64
        model.forward = torch.compile(model.forward, mode="reduce-overhead")
    return model


@torch.no_grad()
def timed_generate(model, ids, n: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, min_new_tokens=n,
                   do_sample=False, pad_token_id=model.config.eos_token_id)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def measure(model, batch: int, ctx: int, new_tokens: int, runs: int) -> dict:
    ids = torch.randint(100, 20000, (batch, ctx), device="cuda")
    for _ in range(2):  # warmup: compile + CUDA graph capture for both lengths
        timed_generate(model, ids, 1)
        timed_generate(model, ids, new_tokens)
    t1 = sorted(timed_generate(model, ids, 1) for _ in range(runs))[runs // 2]
    tn = sorted(timed_generate(model, ids, new_tokens) for _ in range(runs))[runs // 2]
    tpot = (tn - t1) / (new_tokens - 1)
    return {"ttft_ms": round(t1 * 1e3, 2), "tpot_ms": round(tpot * 1e3, 3),
            "decode_tok_s": round(batch / tpot, 1)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--linear", default="none", choices=["none", "fp16", "w4a16_torch", "w4a16"])
    p.add_argument("--norm", action="store_true", help="also use kernels.rmsnorm")
    p.add_argument("--batches", nargs="*", type=int, default=[1, 4, 8])
    p.add_argument("--ctx", nargs="*", type=int, default=[512, 2048])
    p.add_argument("--new-tokens", type=int, default=128)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--no-compile", action="store_true")
    a = p.parse_args()

    model = build(a.model, a.linear, a.norm, not a.no_compile)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "tpot.csv"
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = None
        for B in a.batches:
            for L in a.ctx:
                r = {"time": f"{dt.datetime.now():%Y-%m-%d %H:%M}", "model": a.model, "linear": a.linear,
                     "norm": a.norm, "compiled": not a.no_compile, "batch": B, "ctx": L,
                     "new_tokens": a.new_tokens, **measure(model, B, L, a.new_tokens, a.runs)}
                print(r)
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(r))
                    if new:
                        w.writeheader()
                w.writerow(r)
    print(f"appended -> {out}")


if __name__ == "__main__":
    main()
