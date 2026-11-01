# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import torch
import triton

from aiter.ops.triton.normalization.fused_reduce_add_rmsnorm import (
    fused_reduce_add_rmsnorm,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext


def get_x_vals():
    return [(M, 5120) for M in (1, 8, 16, 32, 64, 128, 256, 1024, 8192)]


def run_benchmark(args):
    x_vals = [tuple(args.shape)] if args.shape else get_x_vals()
    ylabel = {"time": "Time_(us)", "bandwidth": "Bandwidth_(TB/s)"}[args.metric]
    benchmark = triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals,
        line_arg="unit",
        line_vals=[ylabel],
        line_names=[ylabel],
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fused_reduce_add_rmsnorm(M, N, metric, **kwargs):
        dtype = torch.bfloat16
        if args.num_ksplit > 1:
            x = torch.randn((args.num_ksplit, M, N), dtype=torch.float32, device="cuda")
        else:
            x = torch.randn((M, N), dtype=dtype, device="cuda")
        residual = torch.randn((M, N), dtype=dtype, device="cuda")
        weight = torch.randn((N,), dtype=dtype, device="cuda")
        fn = lambda: fused_reduce_add_rmsnorm(
            x, residual, weight, 1e-6, gemma_norm=args.gemma_norm
        )
        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        # Read x, the residual and the norm weight; write out and residual_out.
        nbytes = (
            x.numel() * x.element_size()
            + 3 * M * N * residual.element_size()
            + weight.numel() * weight.element_size()
        )
        if metric == "time":
            return ms * 1e3
        return nbytes / (ms * 1e-3) / 1e12

    bench_fused_reduce_add_rmsnorm.run(save_path=None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark fused split-K reduce + residual add + RMSNorm",
        allow_abbrev=False,
    )
    parser.add_argument("--shape", type=int, nargs=2, metavar=("M", "N"))
    parser.add_argument(
        "--num-ksplit", type=int, default=3, help="1 = a reduced GEMM output"
    )
    parser.add_argument("--gemma-norm", action="store_true")
    parser.add_argument("--metric", choices=["time", "bandwidth"], default="time")
    return parser.parse_args()


def main():
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
