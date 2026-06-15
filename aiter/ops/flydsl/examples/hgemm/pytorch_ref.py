# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""PyTorch reference for HGEMM.

Math (index form), for ``a`` of shape ``[M, K]`` and ``b`` of shape ``[N, K]``::

    out[m, n] = sum_k a[m, k] * b[n, k]          # i.e. out = a @ b.T

Accumulation is done in fp32 and the result is cast back to the input dtype.
The ``b`` operand is laid out as ``[N, K]`` to match the ``flydsl_hgemm``
wrapper contract.

Runs the reference over every shape in ``config.json``::

    python pytorch_ref.py
    python pytorch_ref.py --json ref_out.json --case square_m256_n256_k512_default
    python -m aiter.ops.flydsl.examples.hgemm.pytorch_ref
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make ``aiter`` importable when this file is executed directly (not via -m).
try:
    from aiter.ops.flydsl.examples import _common
except ModuleNotFoundError:  # pragma: no cover - direct-execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
    from aiter.ops.flydsl.examples import _common

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"


def hgemm_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute ``a @ b.T`` with fp32 accumulation, returned in ``a``'s dtype.

    Args:
        a: ``[M, K]`` tensor (fp16/bf16/fp32).
        b: ``[N, K]`` tensor with the same dtype as ``a``.

    Returns:
        ``[M, N]`` tensor in ``a``'s dtype.
    """
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"expected 2D inputs, got a.dim={a.dim()} b.dim={b.dim()}")
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"contraction dims must match: a={tuple(a.shape)} b={tuple(b.shape)}"
        )
    if a.dtype != b.dtype:
        raise ValueError(f"dtype mismatch: a={a.dtype} b={b.dtype}")
    out = torch.mm(a.float(), b.float().t())
    return out.to(a.dtype)


def run_case(case: dict, *, dtype: str, seed: int, device: str) -> dict:
    m, n, k = case["m"], case["n"], case["k"]
    torch_dtype = _common.parse_dtype(dtype)
    a = _common.make_matrix(m, k, torch_dtype, seed=seed, device=device)
    b = _common.make_matrix(n, k, torch_dtype, seed=seed, device=device)
    out = hgemm_ref(a, b)
    return {
        "name": case["name"],
        "shape": {"m": m, "n": n, "k": k},
        "output": _common.tensor_stats(out),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the HGEMM PyTorch reference.")
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
    cases = config.get("cases", [])
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c["name"] in wanted]
    if not cases:
        print("[examples] no matching cases in config; nothing to do.")
        return 0

    print(f"[examples] hgemm PyTorch reference  device={device} dtype={dtype}")
    results = [run_case(c, dtype=dtype, seed=args.seed, device=device) for c in cases]

    rows = []
    for r in results:
        s, o = r["shape"], r["output"]
        rows.append(
            {
                "name": r["name"],
                "shape": f"{s['m']}x{s['n']}x{s['k']}",
                "out_shape": "x".join(str(d) for d in o["shape"]),
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
            ("shape", "SHAPE(MxNxK)"),
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
                "op": config.get("op", "hgemm"),
                "kind": "pytorch_ref",
                "dtype": dtype,
                "environment": _common.environment_info(),
                "results": results,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
