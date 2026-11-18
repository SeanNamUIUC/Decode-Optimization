"""Model configs and the decode-time shapes we benchmark.

Shapes follow HF Llama's actual module layout (separate q/k/v, gate/up),
so microbenchmarks match exactly what e2e/patch_hf.py replaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden: int
    intermediate: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    n_layers: int
    vocab: int


LLAMA_3_2_1B = ModelConfig(
    name="meta-llama/Llama-3.2-1B",
    hidden=2048,
    intermediate=8192,
    n_heads=32,
    n_kv_heads=8,
    head_dim=64,
    n_layers=16,
    vocab=128256,
)

DEFAULT_BATCHES = (1, 2, 4, 8)


def linear_layers(cfg: ModelConfig, include_lm_head: bool = False) -> dict[str, tuple[int, int]]:
    """name -> (K = in_features, N = out_features) for one decoder layer."""
    q = cfg.n_heads * cfg.head_dim
    kv = cfg.n_kv_heads * cfg.head_dim
    layers = {
        "q_proj": (cfg.hidden, q),
        "k_proj": (cfg.hidden, kv),
        "v_proj": (cfg.hidden, kv),
        "o_proj": (q, cfg.hidden),
        "gate_proj": (cfg.hidden, cfg.intermediate),
        "up_proj": (cfg.hidden, cfg.intermediate),
        "down_proj": (cfg.intermediate, cfg.hidden),
    }
    if include_lm_head:
        layers["lm_head"] = (cfg.hidden, cfg.vocab)
    return layers


def linear_cases(cfg: ModelConfig, batches: Sequence[int] = DEFAULT_BATCHES) -> list[dict]:
    """Decode linear cases: M = batch (1 token per sequence)."""
    cases = []
    for layer, (K, N) in linear_layers(cfg).items():
        for M in batches:
            cases.append({"layer": layer, "M": M, "K": K, "N": N})
    return cases


def rmsnorm_cases(cfg: ModelConfig, batches: Sequence[int] = DEFAULT_BATCHES) -> list[dict]:
    return [{"layer": "rmsnorm", "M": M, "N": cfg.hidden} for M in batches]


def case_label(case: dict) -> str:
    keys = [k for k in ("layer", "M", "K", "N") if k in case]
    return " ".join(f"{k}={case[k]}" for k in keys)
