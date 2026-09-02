# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark for the fast_transpose_2d Triton kernel (_transpose_2d_kernel)."""

import argparse

import torch
import triton

from aiter.ops.triton.quant.fast_transpose import fast_transpose_2d
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext


def benchmark(args):
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp8": torch.float8_e4m3fnuz,
    }[args.dtype]
    x_vals = [(m, n) for m in [1024, 4096, 8192] for n in [1024, 4096, 8192]]
    unit = "ms" if args.metric == "time" else "GB/s"

    config = triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=["triton"],
        line_names=[f"fast_transpose ({unit})"],
        styles=[("green", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(M, N, provider):
        x = torch.randn(M, N, device="cuda").to(dtype)
        ms = triton.testing.do_bench(lambda: fast_transpose_2d(x), warmup=25, rep=100)
        if args.metric == "time":
            return ms
        # read M*N + write N*M elements
        gb = 2 * M * N * x.element_size() * 1e-9
        return gb / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark fast_transpose", allow_abbrev=False
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp8"]
    )
    parser.add_argument(
        "-metric",
        nargs="?",
        const="bandwidth",
        choices=["time", "bandwidth"],
        default="bandwidth",
    )
    parser.add_argument("-o", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    benchmark(args)


if __name__ == "__main__":
    main()
