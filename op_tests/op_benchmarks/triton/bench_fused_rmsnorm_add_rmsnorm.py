# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark fused RMSNorm, residual add, RMSNorm for Muse shapes.

Use bench_rmsnorm.py and bench_fused_rmsnorm_add.py --add-residual to benchmark
the existing RMSNorm and RMSNorm-plus-add operators separately.
"""

import argparse

import torch
import triton

from aiter.ops.triton.normalization.fused_rmsnorm_add_rmsnorm import (
    fused_rmsnorm_add_rmsnorm,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext
from op_tests.triton_tests.normalization.test_fused_rmsnorm_add_rmsnorm import (
    generate_fused_rmsnorm_add_rmsnorm_inputs,
)

POST_EPS = 1e-8
PRE_EPS = 1e-5


def get_x_vals():
    # Muse-Glimmer-30B hidden width: decode batches through prefill chunks.
    return [(m, 6656) for m in (1, 8, 32, 64, 128, 256, 1024, 4096)]


def bench_fused_rmsnorm_add_rmsnorm_fn(M, N, metric):
    x, residual, post_weight, pre_weight = generate_fused_rmsnorm_add_rmsnorm_inputs(
        M, N
    )
    residual_out = torch.empty_like(x)

    fn = lambda: fused_rmsnorm_add_rmsnorm(
        x, residual, post_weight, pre_weight, POST_EPS, PRE_EPS, residual_out
    )
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    if metric == "time":
        return ms
    if metric == "bandwidth":
        bytes_moved = (4 * M * N + 2 * N) * x.element_size()
        return bytes_moved / (ms * 1e-3) * 1e-9
    raise ValueError(f"Unknown metric: {metric}")


def run_benchmark(args):
    x_vals = [tuple(args.shape)] if args.shape is not None else get_x_vals()
    unit = "Time_(ms)" if args.metric == "time" else "Bandwidth_(GB/s)"
    benchmark = triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals,
        line_arg="unit",
        line_vals=[unit],
        line_names=[unit],
        styles=[("green", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fused_rmsnorm_add_rmsnorm(M, N, metric, **kwargs):
        return bench_fused_rmsnorm_add_rmsnorm_fn(M, N, metric)

    bench_fused_rmsnorm_add_rmsnorm.run(
        save_path="." if args.o else None, print_data=True, show_plots=False
    )


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="Benchmark FusedRMSNormAddRMSNorm",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("M", "N"),
        help="user-defined shape to benchmark",
    )
    parser.add_argument(
        "--metric",
        choices=["time", "bandwidth"],
        default="time",
        help="metric to report",
    )
    parser.add_argument("-o", action="store_true", help="write results to CSV")
    parsed = parser.parse_args(args=args)
    if parsed.shape is not None and min(parsed.shape) <= 0:
        parser.error("--shape M N requires positive dimensions")
    return parsed


def main(args: list[str] | None = None):
    parsed = parse_args(args)
    run_benchmark(parsed)


if __name__ == "__main__":
    main()
