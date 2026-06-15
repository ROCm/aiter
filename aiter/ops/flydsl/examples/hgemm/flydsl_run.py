# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run the FlyDSL HGEMM kernel over the shapes in ``config.json``.

Standalone runnable::

    python flydsl_run.py
    python flydsl_run.py --time --json flydsl_out.json
    python -m aiter.ops.flydsl.examples.hgemm.flydsl_run

Prints output stats per case, or skips with a clear message and exits 0 when
ROCm/CUDA or the optional ``flydsl`` package is unavailable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``aiter`` importable when this file is executed directly (not via -m).
try:
    from aiter.ops.flydsl.examples import _common
except ModuleNotFoundError:  # pragma: no cover - direct-execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
    from aiter.ops.flydsl.examples import _common

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"

TILING_KEYS = {
    "tile_m",
    "tile_n",
    "tile_k",
    "split_k",
    "pack_n",
    "block_m_warps",
    "block_n_warps",
    "block_k_warps",
    "b_to_lds",
    "stages",
}


def _tiling_for(case: dict, default_tiling: dict) -> dict:
    return {**default_tiling, **{k: v for k, v in case.items() if k in TILING_KEYS}}


def run_case(
    case: dict,
    *,
    dtype: str,
    default_tiling: dict,
    seed: int,
    measure: bool,
    warmup: int,
    iters: int,
) -> dict:
    import torch

    from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm

    m, n, k = case["m"], case["n"], case["k"]
    tiling = _tiling_for(case, default_tiling)
    torch_dtype = _common.parse_dtype(dtype)
    a = _common.make_matrix(m, k, torch_dtype, seed=seed)
    b = _common.make_matrix(n, k, torch_dtype, seed=seed)

    out = flydsl_hgemm(a, b, **tiling)
    torch.cuda.synchronize()

    result = {
        "name": case["name"],
        "shape": {"m": m, "n": n, "k": k},
        "tiling": tiling,
        "output": _common.tensor_stats(out),
    }
    if measure:
        timing = _common.measure_time(
            lambda: flydsl_hgemm(a, b, **tiling),
            label="flydsl",
            warmup=warmup,
            iters=iters,
        )
        result["timing"] = {
            "median_ms": timing.median_ms,
            "min_ms": timing.min_ms,
            "max_ms": timing.max_ms,
            "tflops": _common.gemm_tflops(m, n, k, timing.median_ms),
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the FlyDSL HGEMM kernel.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to config.json")
    parser.add_argument("--json", default=None, help="write run outputs here")
    parser.add_argument("--seed", type=int, default=_common.DEFAULT_INPUT_SEED)
    parser.add_argument("--time", action="store_true", help="also measure median time / TFLOPs")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--case", action="append", default=None, help="only run case(s) by name (repeatable)"
    )
    args = parser.parse_args(argv)

    if not _common.require_backends():
        return 0

    config = _common.load_config(args.config)
    dtype = config.get("dtype", "bf16")
    default_tiling = config.get("default_tiling", {})
    cases = config.get("cases", [])
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c["name"] in wanted]
    if not cases:
        print("[examples] no matching cases in config; nothing to do.")
        return 0

    print(f"[examples] hgemm FlyDSL kernel  dtype={dtype}")
    results = []
    for case in cases:
        print("=" * 70)
        print(f"[hgemm] case={case['name']} shape=({case['m']}, {case['n']}, {case['k']})")
        results.append(
            run_case(
                case,
                dtype=dtype,
                default_tiling=default_tiling,
                seed=args.seed,
                measure=args.time,
                warmup=args.warmup,
                iters=args.iters,
            )
        )

    rows = []
    for r in results:
        s, o = r["shape"], r["output"]
        row = {
            "name": r["name"],
            "shape": f"{s['m']}x{s['n']}x{s['k']}",
            "out_shape": "x".join(str(d) for d in o["shape"]),
            "min": o["min"],
            "max": o["max"],
            "mean": o["mean"],
            "std": o["std"],
        }
        if "timing" in r:
            row["ms"] = r["timing"]["median_ms"]
            row["TFLOPs"] = r["timing"]["tflops"]
        rows.append(row)

    columns = [
        ("name", "CASE"),
        ("shape", "SHAPE(MxNxK)"),
        ("out_shape", "OUT"),
        ("min", "MIN"),
        ("max", "MAX"),
        ("mean", "MEAN"),
        ("std", "STD"),
    ]
    if args.time:
        columns += [("ms", "MEDIAN_ms"), ("TFLOPs", "TFLOPs")]
    print(f"\n{'=' * 70}")
    _common.print_table(rows, columns)

    if args.json:
        _common.dump_json(
            args.json,
            {
                "op": config.get("op", "hgemm"),
                "kind": "flydsl_run",
                "dtype": dtype,
                "environment": _common.environment_info(),
                "results": results,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
