# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for the fused residual + gate[indices] * x kernel against the eager
expression, which materialises the gathered gate and the product at [M, N].
"""

import argparse
import sys

import torch
import triton

from aiter.ops.triton.fusions.fused_indexed_gated_residual import (
    fused_indexed_gated_residual,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.fusions.test_fused_indexed_gated_residual import make

_PROVIDERS = ("fused", "eager")
# [M, N] arrays moved: fused reads residual and x and writes out; eager also
# writes the gathered gate, and writes and re-reads the product.
_ARRAYS = {"fused": 3, "eager": 7}
# MiniMax-H3: hidden 5376; 114660 tokens for a 15 s 768p video, 57330 per rank
# with two-way sequence parallelism.
_HIDDEN = 5376
_TOKENS = (1024, 4096, 16384, 57330, 114660)


def bench_fn_impl(M, N, provider, metric, args):
    residual, x, gate, indices = make(
        M, N, args.G, torch.bfloat16, "cuda", runs=not args.scattered
    )
    mem = _ARRAYS[provider] * M * N * residual.element_size()

    if provider == "fused":
        out = torch.empty_like(residual)

        def fn():
            return fused_indexed_gated_residual(residual, x, gate, indices, out=out)

    else:

        def fn():
            return residual + gate[indices] * x

    ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
    if metric == "time":
        return ms * 1000  # us
    if metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9  # GB/s
    raise ValueError(f"unknown metric {metric}")


def run_benchmark(args):
    providers = _PROVIDERS if args.provider == "all" else (args.provider,)
    metrics = ("time", "bandwidth") if args.metric == "all" else (args.metric,)
    line_vals = [f"{p}_{m}" for m in metrics for p in providers]
    x_vals = [(args.M, args.N)] if args.M else [(m, args.N) for m in _TOKENS]

    benchmark = triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_vals,
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-")][
            : len(line_vals)
        ],
        ylabel="",
        plot_name=get_caller_name_no_ext()
        + ("_scattered" if args.scattered else "_runs"),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(M, N, provider):
        name, metric = provider.rsplit("_", 1)
        return bench_fn_impl(M, N, name, metric, args)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark fused indexed gated residual",
        description="Benchmark residual + gate[indices] * x, fused against eager",
        allow_abbrev=False,
    )
    parser.add_argument("-M", type=int, default=None, help="Number of tokens")
    parser.add_argument("-N", type=int, default=_HIDDEN, help="Hidden size")
    parser.add_argument("-G", type=int, default=9, help="Gate table rows")
    parser.add_argument(
        "--scattered",
        action="store_true",
        help="Random index per token instead of runs of one index",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        choices=[*_PROVIDERS, "all"],
        help="fused kernel, eager expression, or both",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=["all", "time", "bandwidth"],
        help="Metric to report (default: all)",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels",
    )
    parser.add_argument(
        "-o", action="store_true", default=False, help="Write results to a CSV file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for fused_indexed_gated_residual...")
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
