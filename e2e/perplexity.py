"""Day 16 -- accuracy cost of W4A16 on a slice of wikitext-2 (test split).

    python -m e2e.perplexity --linear none
    python -m e2e.perplexity --linear w4a16_torch     # quantization alone
    python -m e2e.perplexity --linear w4a16           # + your kernel (decode path only;
                                                       # prefill uses the same int4 weights)
"""
from __future__ import annotations

import argparse
import math

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from e2e.patch_hf import patch_model


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--linear", default="none", choices=["none", "fp16", "w4a16_torch", "w4a16"])
    p.add_argument("--tokens", type=int, default=32768)
    p.add_argument("--seq", type=int, default=1024)
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    patch_model(model, a.linear)

    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[:, : a.tokens].cuda()

    nll, count = 0.0, 0
    for s in range(0, ids.shape[1] - 1, a.seq):
        chunk = ids[:, s : s + a.seq + 1]
        if chunk.shape[1] < 2:
            break
        logits = model(chunk[:, :-1]).logits.float()
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                 chunk[:, 1:].reshape(-1), reduction="sum")
        nll += loss.item()
        count += chunk.shape[1] - 1
    print(f"linear={a.linear}  tokens={count}  perplexity={math.exp(nll / count):.3f}")


if __name__ == "__main__":
    main()
