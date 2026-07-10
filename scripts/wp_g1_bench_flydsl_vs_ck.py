#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""WP-G1 Phase B: Benchmark FlyDSL preshuffle GEMM vs CK across standard shapes.

Runs both backends on the same shapes and reports throughput (TFLOPS),
latency (us), and speedup. Use --shapes-csv to supply a custom shape set
or rely on the built-in defaults.

Usage:
    python scripts/wp_g1_bench_flydsl_vs_ck.py
    python scripts/wp_g1_bench_flydsl_vs_ck.py --shapes-csv my_shapes.csv
    python scripts/wp_g1_bench_flydsl_vs_ck.py --output results.csv
"""

import argparse
import os
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.gemm_op_a8w8 import (
    gemm_a8w8_ck,
    _select_flydsl_preshuffle_kernel,
    gemm_a8w8_bpreshuffle_flydsl,
    get_GEMM_config_with_quant_type,
)
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

DEFAULT_SHAPES = [
    # qkv_proj
    (1, 1280, 8192),
    (32, 1280, 8192),
    (64, 1280, 8192),
    (128, 1280, 8192),
    (256, 1280, 8192),
    (512, 1280, 8192),
    (1024, 1280, 8192),
    (2048, 1280, 8192),
    (4096, 1280, 8192),
    (8192, 1280, 8192),
    # attn_out
    (1, 8192, 1024),
    (32, 8192, 1024),
    (64, 8192, 1024),
    (128, 8192, 1024),
    (256, 8192, 1024),
    (512, 8192, 1024),
    (1024, 8192, 1024),
    (2048, 8192, 1024),
    (4096, 8192, 1024),
    (8192, 8192, 1024),
    # fc shapes
    (16, 7424, 8192),
    (32, 7424, 8192),
    (64, 7424, 8192),
    (128, 7424, 8192),
    (4096, 7424, 8192),
    (8192, 7424, 8192),
]

NUM_WARMUP = 10
NUM_ITERS = 100


def torch_ref(x, weight, x_scale, w_scale, dtype):
    x_fp = x.to(torch.float32) * x_scale
    w_fp = weight.to(torch.float32) * w_scale
    return F.linear(x_fp, w_fp).to(dtype)


def bench_kernel(fn, num_warmup=NUM_WARMUP, num_iters=NUM_ITERS):
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iters * 1000  # us


def run_ck(x, weight, x_scale, w_scale, out, splitK=0):
    gemm_a8w8_ck(x, weight, x_scale, w_scale, out, None, splitK)
    return out


def run_flydsl(x, weight_shuffled, x_scale, w_scale, out, ki_name):
    gemm_a8w8_bpreshuffle_flydsl(
        x, weight_shuffled, x_scale, w_scale, out, {"kernelName": ki_name}
    )
    return out


def bench_shape(m, n, k, dtype=dtypes.bf16, quant_dtype=dtypes.fp8):
    x = torch.randn((m, k), dtype=dtype, device="cuda")
    weight = torch.randn((n, k), dtype=dtype, device="cuda")
    x, x_scale = aiter.pertoken_quant(x, quant_dtype=quant_dtype)
    weight, w_scale = aiter.pertoken_quant(weight, quant_dtype=quant_dtype)
    weight_shuffled = shuffle_weight(weight, layout=(16, 16))

    ref = torch_ref(x, weight, x_scale, w_scale, dtype)
    flops = 2.0 * m * n * k

    result = {"M": m, "N": n, "K": k}

    # --- CK ---
    ck_config = get_GEMM_config_with_quant_type(
        m, n, k, quant_dtype, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_FILE
    )
    splitK = int(ck_config["splitK"]) if ck_config is not None else 0
    out_ck = torch.empty(m, n, dtype=dtype, device="cuda")
    try:
        run_ck(x, weight, x_scale, w_scale, out_ck, splitK)
        ck_us = bench_kernel(
            lambda: run_ck(x, weight, x_scale, w_scale, out_ck, splitK)
        )
        ck_tflops = flops / ck_us * 1e-6
        ck_err = torch.isclose(ref, out_ck, rtol=1e-2, atol=1e-2)
        ck_err_ratio = 1.0 - ck_err.float().mean().item()
    except RuntimeError as e:
        print(f"  CK failed for ({m},{n},{k}): {e}")
        ck_us = float("inf")
        ck_tflops = 0.0
        ck_err_ratio = -1.0
    result.update({
        "ck_us": round(ck_us, 2),
        "ck_tflops": round(ck_tflops, 2),
        "ck_err": round(ck_err_ratio, 4),
    })

    # --- FlyDSL ---
    # Prefer tuned config from bpreshuffle CSV; fall back to heuristic
    bp_config = get_GEMM_config_with_quant_type(
        m, n, k, quant_dtype, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE
    )
    fly_kernel_name = None
    if bp_config is not None and bp_config.get("libtype") == "flydsl":
        fly_kernel_name = bp_config["kernelName"]
    else:
        ki = _select_flydsl_preshuffle_kernel(m, n, k)
        if ki is not None:
            fly_kernel_name = ki.name
    if fly_kernel_name is not None:
        out_fly = torch.empty(m, n, dtype=dtype, device="cuda")
        try:
            run_flydsl(x, weight_shuffled, x_scale, w_scale, out_fly, fly_kernel_name)
            fly_us = bench_kernel(
                lambda: run_flydsl(
                    x, weight_shuffled, x_scale, w_scale, out_fly, fly_kernel_name
                )
            )
            fly_tflops = flops / fly_us * 1e-6
            fly_err = torch.isclose(ref, out_fly, rtol=1e-2, atol=1e-2)
            fly_err_ratio = 1.0 - fly_err.float().mean().item()
        except RuntimeError as e:
            print(f"  FlyDSL failed for ({m},{n},{k}): {e}")
            fly_us = float("inf")
            fly_tflops = 0.0
            fly_err_ratio = -1.0
        result.update({
            "fly_us": round(fly_us, 2),
            "fly_tflops": round(fly_tflops, 2),
            "fly_err": round(fly_err_ratio, 4),
            "fly_kernel": fly_kernel_name,
        })

        if ck_us > 0 and fly_us > 0 and ck_us != float("inf"):
            speedup = ck_us / fly_us
            result["speedup"] = round(speedup, 3)
            winner = "FlyDSL" if speedup > 1.0 else "CK"
        else:
            result["speedup"] = None
            winner = "N/A"
    else:
        result.update({
            "fly_us": None,
            "fly_tflops": None,
            "fly_err": None,
            "fly_kernel": "no_fit",
            "speedup": None,
        })
        winner = "CK (no FlyDSL fit)"

    print(
        f"  M={m:<6} N={n:<6} K={k:<6} | "
        f"CK={result['ck_us']:<8} FlyDSL={result.get('fly_us', 'N/A'):<8} | "
        f"speedup={result.get('speedup', 'N/A'):<8} winner={winner}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="WP-G1 FlyDSL vs CK benchmark")
    parser.add_argument(
        "--shapes-csv",
        type=str,
        default=None,
        help="CSV with M,N,K columns (one shape per row)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Save results to CSV",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Output dtype",
    )
    args = parser.parse_args()

    if not is_flydsl_available():
        print("ERROR: FlyDSL not available. Cannot benchmark.")
        return 1

    dtype = dtypes.bf16 if args.dtype == "bf16" else dtypes.fp16

    if args.shapes_csv:
        df = pd.read_csv(args.shapes_csv)
        shapes = list(zip(df["M"].tolist(), df["N"].tolist(), df["K"].tolist()))
    else:
        shapes = DEFAULT_SHAPES

    print(f"=== WP-G1 Benchmark: FlyDSL vs CK ({get_gfx()}) ===")
    print(f"Shapes: {len(shapes)}, dtype: {args.dtype}")
    print(f"Warmup: {NUM_WARMUP}, Iters: {NUM_ITERS}")
    print()

    results = []
    for m, n, k in shapes:
        result = bench_shape(m, n, k, dtype=dtype)
        results.append(result)
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    print()
    print(df.to_markdown(index=False))

    wins = df[df["speedup"].notna()]
    if len(wins) > 0:
        fly_wins = (wins["speedup"] > 1.0).sum()
        ck_wins = (wins["speedup"] <= 1.0).sum()
        geo_mean = wins["speedup"].prod() ** (1.0 / len(wins))
        print(f"\nFlyDSL wins: {fly_wins}/{len(wins)}, CK wins: {ck_wins}/{len(wins)}")
        print(f"Geometric mean speedup: {geo_mean:.3f}x")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
