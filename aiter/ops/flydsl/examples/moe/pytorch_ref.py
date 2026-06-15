# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""PyTorch reference for fused Mixture-of-Experts (MoE).

Standard token-choice fused MoE with two per-expert projections and a gated
activation. For ``T`` tokens, model dim ``D``, intermediate dim ``I``, ``E``
experts and ``topk`` selected experts per token:

    scores  = hidden @ gate           # router logits  [T, E]  (here: random)
    g       = softmax(scores)          # [T, E]
    w, ids  = topk(g, topk)            # per-token expert weights + indices
    for each selected expert e:
        gate_up = x @ w1[e].T          # [., 2*I]  (w1 is [E, 2*I, D])
        h       = silu(gate) * up      # gated activation, split halves
        y_e     = h @ w2[e].T          # [., D]    (w2 is [E, D, I])
    out     = sum_topk(w * y_e)        # weighted combine -> [T, D]

All matmuls accumulate in fp32; the result is cast back to the input dtype.
This is the standalone "source" formulation. The FlyDSL kernels in
``flydsl_run.py`` compute the same math via a quantized two-stage path; see the
README for the modeling notes.

Runs the reference over every case in ``config.json``::

    python pytorch_ref.py
    python pytorch_ref.py --json ref_out.json --case a4w4_t16_e256
    python -m aiter.ops.flydsl.examples.moe.pytorch_ref
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Make ``aiter`` importable when this file is executed directly (not via -m).
try:
    from aiter.ops.flydsl.examples import _common
except ModuleNotFoundError:  # pragma: no cover - direct-execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
    from aiter.ops.flydsl.examples import _common

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"

_ACT = {"silu": F.silu, "gelu": F.gelu}


def route_topk(scores: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax router followed by top-k selection with renormalized weights."""
    gate = torch.softmax(scores.float(), dim=-1)
    weights, ids = torch.topk(gate, topk, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, ids


def moe_ref(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    """Compute fused MoE with fp32 accumulation, returned in input dtype.

    Args:
        hidden_states: ``[T, D]`` input tokens.
        w1: ``[E, 2*I, D]`` gate/up projection (gate and up concatenated).
        w2: ``[E, D, I]`` down projection.
        topk_weights: ``[T, topk]`` combine weights.
        topk_ids: ``[T, topk]`` selected expert indices.
        activation: gate activation, ``"silu"`` or ``"gelu"``.

    Returns:
        ``[T, D]`` tensor in ``hidden_states``'s dtype.
    """
    if activation not in _ACT:
        raise ValueError(f"unsupported activation {activation!r}; expected silu|gelu")
    compute = torch.float32
    in_dtype = hidden_states.dtype
    hs = hidden_states.to(compute)
    w1 = w1.to(compute)
    w2 = w2.to(compute)

    t, d = hs.shape
    topk = topk_weights.shape[1]
    inter_dim = w2.shape[2]
    act_fn = _ACT[activation]

    x = hs.view(t, 1, d).repeat(1, topk, 1)
    out = torch.zeros((t, topk, d), dtype=compute, device=hs.device)
    for e in range(w1.shape[0]):
        mask = topk_ids == e
        if mask.sum():
            sub = x[mask]
            gate_up = sub @ w1[e].transpose(0, 1)
            gate, up = gate_up.split([inter_dim, inter_dim], dim=-1)
            act_out = act_fn(gate) * up
            out[mask] = act_out @ w2[e].transpose(0, 1)

    combined = (out * topk_weights.view(t, topk, 1).to(compute)).sum(dim=1)
    return combined.to(in_dtype)


def run_case(
    case: dict, *, dtype: str, seed: int, device: str, activation: str
) -> dict:
    t = case["tokens"]
    d = case["model_dim"]
    i = case["inter_dim"]
    e = case["experts"]
    topk = case["topk"]
    activation = case.get("activation", activation)

    torch_dtype = _common.parse_dtype(dtype)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    def randn(*shape):
        return torch.randn(shape, generator=gen, device=device, dtype=torch_dtype) / 10.0

    hidden = randn(t, d)
    w1 = randn(e, 2 * i, d)
    w2 = randn(e, d, i)
    scores = torch.randn((t, e), generator=gen, device=device, dtype=torch_dtype)
    topk_weights, topk_ids = route_topk(scores, topk)

    out = moe_ref(hidden, w1, w2, topk_weights, topk_ids, activation=activation)
    return {
        "name": case["name"],
        "shape": {"tokens": t, "model_dim": d, "inter_dim": i, "experts": e, "topk": topk},
        "output": _common.tensor_stats(out),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the fused MoE PyTorch reference.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to config.json")
    parser.add_argument("--json", default=None, help="write run outputs here")
    parser.add_argument("--seed", type=int, default=_common.DEFAULT_INPUT_SEED)
    parser.add_argument(
        "--case", action="append", default=None, help="only run case(s) by name (repeatable)"
    )
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = _common.load_config(args.config)
    dtype = config.get("dtype", "bf16")
    activation = config.get("default_params", {}).get("activation", "silu")
    cases = config.get("cases", [])
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c["name"] in wanted]
    if not cases:
        print("[examples] no matching cases in config; nothing to do.")
        return 0

    print(f"[examples] moe PyTorch reference  device={device} dtype={dtype}")
    results = [
        run_case(c, dtype=dtype, seed=args.seed, device=device, activation=activation)
        for c in cases
    ]

    rows = []
    for r in results:
        s, o = r["shape"], r["output"]
        rows.append(
            {
                "name": r["name"],
                "tokens": s["tokens"],
                "dims": f"D{s['model_dim']}/I{s['inter_dim']}/E{s['experts']}/k{s['topk']}",
                "out_shape": "x".join(str(x) for x in o["shape"]),
                "min": o["min"],
                "max": o["max"],
                "mean": o["mean"],
                "std": o["std"],
            }
        )
    _common.print_table(
        rows,
        [
            ("name", "CASE"),
            ("tokens", "TOKENS"),
            ("dims", "DIMS"),
            ("out_shape", "OUT"),
            ("min", "MIN"),
            ("max", "MAX"),
            ("mean", "MEAN"),
            ("std", "STD"),
        ],
    )

    if args.json:
        _common.dump_json(
            args.json,
            {
                "op": config.get("op", "moe"),
                "kind": "pytorch_ref",
                "dtype": dtype,
                "environment": _common.environment_info(),
                "results": results,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
