# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmarks for fast transpose Triton kernels."""

import argparse
from functools import partial

import torch
import triton

from aiter.ops.triton.quant.fast_transpose import (
    fast_transpose_2d,
    transpose_packed_fp4,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext


def _torch_transpose_2d(x: torch.Tensor) -> torch.Tensor:
    return x.t().contiguous()


def benchmark_fast_transpose(args):
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp8": torch.float8_e4m3fnuz,
    }[args.dtype]
    x_vals = (
        [tuple(args.shape)]
        if args.shape is not None
        else [(m, n) for m in [1024, 4096, 8192] for n in [1024, 4096, 8192]]
    )
    unit = "ms" if args.metric == "time" else "GB/s"

    config = triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=[f"fast_transpose ({unit})", f"torch t().contiguous() ({unit})"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(M, N, provider):
        x = torch.randn(M, N, device="cuda").to(dtype)
        fn = partial(
            _torch_transpose_2d if provider == "torch" else fast_transpose_2d, x
        )
        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        if args.metric == "time":
            return ms
        # read M*N + write N*M elements
        gb = 2 * M * N * x.element_size() * 1e-9
        return gb / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def _torch_transpose_packed_fp4(data_fp4: torch.Tensor) -> torch.Tensor:
    low_nibbles = data_fp4 & 0x0F
    high_nibbles = (data_fp4 >> 4) & 0x0F
    unpacked = torch.stack((low_nibbles, high_nibbles), dim=-1).flatten(-2)
    transposed = unpacked.t().contiguous()
    return transposed[:, 0::2] | (transposed[:, 1::2] << 4)


def benchmark_packed_fp4_transpose(args):
    shapes = (
        [tuple(args.shape)]
        if args.shape is not None
        else [(6144, 4096), (4096, 4096), (24576, 4096), (4096, 12288)]
    )
    unit = "ms" if args.metric == "time" else "GB/s"
    config = triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=shapes,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["AITER", "PyTorch"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel=unit,
        plot_name=f"{get_caller_name_no_ext()}_packed_fp4",
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(M, N, provider):
        if M % 2 != 0 or N % 2 != 0:
            raise ValueError(f"logical M and N must be even, got ({M}, {N})")
        data_fp4 = torch.randint(0, 256, (M, N // 2), dtype=torch.uint8, device="cuda")
        fn = partial(
            (
                transpose_packed_fp4
                if provider == "triton"
                else _torch_transpose_packed_fp4
            ),
            data_fp4,
        )
        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        if args.metric == "time":
            return ms
        bytes_read_and_written = M * N
        return bytes_read_and_written * 1e-9 / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark fast_transpose", allow_abbrev=False
    )
    parser.add_argument(
        "--op",
        choices=["fast", "packed_fp4"],
        default="fast",
        help="transpose operation to benchmark",
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp8"]
    )
    parser.add_argument("--shape", type=int, nargs=2, metavar=("M", "N"))
    parser.add_argument(
        "-metric",
        "--metric",
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
    if args.op == "packed_fp4":
        benchmark_packed_fp4_transpose(args)
    else:
        benchmark_fast_transpose(args)


if __name__ == "__main__":
    main()
