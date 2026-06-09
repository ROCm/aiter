# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BREAKDOWN = THIS_DIR / "bench_sonic_moe_breakdown.py"


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep SonicMoE/AITER fused_moe routing and runtime knobs."
    )
    parser.add_argument("--shape", action="append", default=None, help="T,H,I,E,K")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--routing", default="topk,rounded,balanced")
    parser.add_argument("--block-m", default="auto,32,64,128")
    parser.add_argument("--dispatch-policy", default="0")
    parser.add_argument("--ksplit", default="", help="Comma-separated AITER_KSPLIT values")
    parser.add_argument("--use-nt", default="", help="Comma-separated AITER_USE_NT values")
    parser.add_argument(
        "--online-tune",
        action="store_true",
        help="Set AITER_ONLINE_TUNE=1 in child runs. This can compile/profile many kernels.",
    )
    parser.add_argument(
        "--ck-sorting",
        default="",
        help="Comma-separated AITER_USE_CK_MOE_SORTING values; each run is a fresh process.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--rounding-tile", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-log", action="store_true")
    return parser.parse_args()


def _env_values(values: str) -> list[str | None]:
    if not values:
        return [None]
    return _csv(values)


def _run_one(
    shape: str,
    routing: str,
    block_m: str,
    dispatch_policy: str,
    ksplit: str | None,
    use_nt: str | None,
    ck_sorting: str | None,
    args: argparse.Namespace,
) -> tuple[dict, str]:
    env = os.environ.copy()
    if ksplit is not None:
        env["AITER_KSPLIT"] = ksplit
    if use_nt is not None:
        env["AITER_USE_NT"] = use_nt
    if ck_sorting is not None:
        env["AITER_USE_CK_MOE_SORTING"] = ck_sorting
    if args.online_tune:
        env["AITER_ONLINE_TUNE"] = "1"

    cmd = [
        sys.executable,
        str(BREAKDOWN),
        "--shape",
        shape,
        "--dtype",
        args.dtype,
        "--routing",
        routing,
        "--block-size-m",
        block_m,
        "--dispatch-policy",
        dispatch_policy,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--rounding-tile",
        str(args.rounding_tile),
        "--seed",
        str(args.seed),
        "--skip-wrapper",
    ]
    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    result = {
        "shape": shape,
        "routing": routing,
        "block_size_m": block_m,
        "dispatch_policy": dispatch_policy,
        "ksplit": ksplit or "default",
        "use_nt": use_nt or "default",
        "ck_sorting": ck_sorting or "default",
        "returncode": proc.returncode,
    }
    for line in proc.stdout.splitlines():
        if line.startswith("result_json="):
            result.update(json.loads(line.split("=", 1)[1]))
            break
    return result, proc.stdout


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    args = parse_args()
    shapes = args.shape or [
        "32768,4096,512,128,8",
        "32768,4096,1024,128,8",
    ]
    routings = _csv(args.routing)
    block_ms = _csv(args.block_m)
    dispatch_policies = _csv(args.dispatch_policy)
    ksplit_values = _env_values(args.ksplit)
    use_nt_values = _env_values(args.use_nt)
    ck_sorting_values = _env_values(args.ck_sorting)

    results: list[dict] = []
    for combo in product(
        shapes,
        routings,
        block_ms,
        dispatch_policies,
        ksplit_values,
        use_nt_values,
        ck_sorting_values,
    ):
        shape, routing, block_m, dispatch_policy, ksplit, use_nt, ck_sorting = combo
        print(
            "RUN "
            f"shape={shape} routing={routing} block_m={block_m} "
            f"dispatch_policy={dispatch_policy} ksplit={ksplit or 'default'} "
            f"use_nt={use_nt or 'default'} ck_sorting={ck_sorting or 'default'}",
            flush=True,
        )
        result, log = _run_one(
            shape,
            routing,
            block_m,
            dispatch_policy,
            ksplit,
            use_nt,
            ck_sorting,
            args,
        )
        results.append(result)
        if args.keep_log or result.get("returncode") != 0 or "direct_fused_moe_ms" not in result:
            print(log, flush=True)
        else:
            print(
                "  "
                f"direct_ms={_fmt(result.get('direct_fused_moe_ms'))} "
                f"tflops={_fmt(result.get('direct_fused_moe_expert_tflops'), 2)} "
                f"tile_eff={_fmt(result.get('tile_efficiency'), 4)} "
                f"moved={result.get('route_moved', 'NA')}",
                flush=True,
            )

    ok = [row for row in results if "direct_fused_moe_ms" in row]
    ok.sort(key=lambda row: float(row["direct_fused_moe_ms"]))
    print("\nBEST")
    print(
        "rank shape routing block_m policy ksplit use_nt ck_sorting "
        "direct_ms tflops tile_eff moved"
    )
    for idx, row in enumerate(ok[:20], start=1):
        print(
            f"{idx} "
            f"{row.get('shape')} "
            f"{row.get('routing')} "
            f"{row.get('block_size_m')} "
            f"{row.get('dispatch_policy')} "
            f"{row.get('ksplit')} "
            f"{row.get('use_nt')} "
            f"{row.get('ck_sorting')} "
            f"{_fmt(row.get('direct_fused_moe_ms'))} "
            f"{_fmt(row.get('direct_fused_moe_expert_tflops'), 2)} "
            f"{_fmt(row.get('tile_efficiency'), 4)} "
            f"{row.get('route_moved', 'NA')}"
        )


if __name__ == "__main__":
    main()
