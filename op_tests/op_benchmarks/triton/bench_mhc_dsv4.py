# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for DSV4 MHC kernels: mhc_pre_dsv4, mhc_post_dsv4, mhc_head_dsv4.

Usage:
    python bench_mhc_dsv4.py
    python bench_mhc_dsv4.py --op head
    python bench_mhc_dsv4.py --op post
    python bench_mhc_dsv4.py -metric bandwidth
"""

import argparse

import torch
import triton

from aiter.ops.triton.fusions.mhc import mhc_head_dsv4, mhc_post_dsv4, mhc_pre_dsv4
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

# DSV4 representative shapes: (M, n, C)
_SHAPES = [
    (64, 4, 1024),
    (256, 4, 1024),
    (1024, 4, 1024),
    (4096, 4, 1024),
    (16384, 4, 1024),
    (64, 4, 512),
    (1024, 4, 512),
]


def _make_pre_inputs(M, n, C, dtype, device="cuda"):
    rows = 2 * n + n * n
    residual = torch.randn(M, n, C, dtype=dtype, device=device) * 0.1
    fn = torch.randn(rows, n * C, dtype=torch.float32, device=device) * 0.02
    scale = torch.ones(3, dtype=torch.float32, device=device)
    base = torch.zeros(rows, dtype=torch.float32, device=device)
    return residual, fn, scale, base


def _make_head_inputs(M, n, C, dtype, device="cuda"):
    residual = torch.randn(M, n, C, dtype=dtype, device=device) * 0.1
    fn = torch.randn(n, n * C, dtype=torch.float32, device=device) * 0.02
    scale = torch.ones(1, dtype=torch.float32, device=device)
    base = torch.zeros(n, dtype=torch.float32, device=device)
    return residual, fn, scale, base


def benchmark(args):
    op = args.op
    unit = "ms" if args.metric == "time" else "GB/s"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    x_vals = [(M, n, C) for M, n, C in _SHAPES]

    config = triton.testing.Benchmark(
        x_names=["M", "n", "C"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=[op],
        line_names=[f"{op} ({unit})"],
        styles=[("blue", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(M, n, C, provider):
        if op == "pre":
            residual, fn, scale, base = _make_pre_inputs(M, n, C, dtype)
            fn_b = lambda: mhc_pre_dsv4(residual, fn, scale, base)
            rows = 2 * n + n * n
            # reads: residual + fn + scale + base; writes: post_mix + comb_mix + layer_input
            mem = (
                M * n * C * residual.element_size()
                + rows * n * C * 4
                + 3 * 4
                + rows * 4
                + M * n * 4  # post_mix fp32
                + M * n * n * 4  # comb_mix fp32
                + M * C * residual.element_size()  # layer_input
            )
        elif op == "post":
            residual, fn, scale, base = _make_pre_inputs(M, n, C, dtype)
            post_mix = torch.randn(M, n, 1, dtype=torch.float32, device="cuda")
            comb_mix = torch.randn(M, n, n, dtype=torch.float32, device="cuda")
            layer_input = torch.randn(M, C, dtype=dtype, device="cuda")
            fn_b = lambda: mhc_post_dsv4(layer_input, residual, post_mix, comb_mix)
            # reads: layer_input + residual + post_mix + comb_mix; writes: output
            mem = (
                M * C * residual.element_size()
                + M * n * C * residual.element_size()
                + M * n * 4
                + M * n * n * 4
                + M * n * C * residual.element_size()
            )
        else:  # head
            residual, fn, scale, base = _make_head_inputs(M, n, C, dtype)
            fn_b = lambda: mhc_head_dsv4(residual, fn, scale, base)
            # reads: residual + fn + scale + base; writes: output
            mem = (
                M * n * C * residual.element_size()
                + n * n * C * 4
                + 1 * 4
                + n * 4
                + M * C * residual.element_size()
            )

        ms = triton.testing.do_bench(fn_b, warmup=25, rep=100)
        if args.metric == "time":
            return ms
        return mem * 1e-9 / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MHC DSV4 kernels", allow_abbrev=False
    )
    parser.add_argument(
        "--op",
        choices=["pre", "post", "head"],
        default="pre",
        help="Which DSV4 op to benchmark (default: pre)",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16"],
        default="bf16",
    )
    parser.add_argument(
        "-metric",
        nargs="?",
        const="time",
        choices=["time", "bandwidth"],
        default="time",
    )
    parser.add_argument("-o", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    benchmark(args)


if __name__ == "__main__":
    main()
