# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark for MXFP8 convert kernels.

``to`` exercises ``_convert_to_mxfp8_kernel``; ``from`` exercises
``_convert_from_mxfp8_kernel``. use_asm is left at the wrapper default
(auto-selected: ASM on gfx950, portable path elsewhere).
"""

import argparse

import torch
import triton

from aiter.ops.triton.quant.quant_mxfp8 import convert_from_mxfp8, convert_to_mxfp8
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext


def benchmark(args):
    in_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    fp8_dtype = torch.float8_e4m3fn
    # M, N must tile to (block_m=64, block_n=64).
    x_vals = [(m, n) for m in [4096, 8192] for n in [4096, 8192, 16384]]
    unit = "ms" if args.metric == "time" else "GB/s"

    config = triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=["to", "from"],
        line_names=[f"to ({unit})", f"from ({unit})"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(M, N, provider):
        x = torch.randn(M, N, device="cuda", dtype=in_dtype)
        if provider == "to":
            fn = lambda: convert_to_mxfp8(x, fp8_dtype, quant_block_size=32)
        else:
            y, s = convert_to_mxfp8(x, fp8_dtype, quant_block_size=32)
            fn = lambda: convert_from_mxfp8(y, s, in_dtype, quant_block_size=32)
        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        if args.metric == "time":
            return ms
        # bytes: read input + write output (fp8 for `to`, high-precision for `from`)
        gb = M * N * (in_dtype.itemsize + 1) * 1e-9
        return gb / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = argparse.ArgumentParser(prog="Benchmark MXFP8 convert", allow_abbrev=False)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
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
