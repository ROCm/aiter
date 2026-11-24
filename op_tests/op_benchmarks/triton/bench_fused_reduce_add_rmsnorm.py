# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import torch
import triton

from aiter.ops.triton.normalization.fused_reduce_add_rmsnorm import (
    fused_reduce_add_rmsnorm,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)

_POOL_KEYS = ["x", "residual", "weight"]


def get_x_vals():
    return [(M, 5120) for M in (1, 8, 16, 32, 64, 128, 256, 1024, 8192)]


def _generate_inputs(M, N, num_ksplit):
    dtype = torch.bfloat16
    if num_ksplit > 1:
        x = torch.randn((num_ksplit, M, N), dtype=torch.float32, device="cuda")
    else:
        x = torch.randn((M, N), dtype=dtype, device="cuda")
    residual = torch.randn((M, N), dtype=dtype, device="cuda")
    weight = torch.randn((N,), dtype=dtype, device="cuda")
    return x, residual, weight


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
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

    @triton.testing.perf_report([benchmark])
    def bench_fused_reduce_add_rmsnorm(M, N, metric, **kwargs):
        bencher.build_pool(
            [functools.partial(_generate_inputs, M, N, args.num_ksplit)],
            [_POOL_KEYS],
        )

        def fn(x, residual, weight):
            fused_reduce_add_rmsnorm(
                x, residual, weight, 1e-6, gemma_norm=args.gemma_norm
            )

        ms = bencher.bench(fn)
        # Read x, the residual and the norm weight; write out and residual_out.
        elem = torch.bfloat16.itemsize
        x_bytes = (
            args.num_ksplit * M * N * torch.float32.itemsize
            if args.num_ksplit > 1
            else M * N * elem
        )
        nbytes = x_bytes + 3 * M * N * elem + N * elem
        if metric == "time":
            return ms * 1e3
        return nbytes / (ms * 1e-3) / 1e12

    bench_fused_reduce_add_rmsnorm.run(save_path=None, print_data=True)


def parse_args():
    parser = get_parser(kernel_name="fused split-K reduce + residual add + RMSNorm")
    parser.add_argument("--shape", type=int, nargs=2, metavar=("M", "N"))
    parser.add_argument(
        "--num-ksplit", type=int, default=3, help="1 = a reduced GEMM output"
    )
    parser.add_argument("--gemma-norm", action="store_true")
    override_parser_action(
        parser, action="--metric", choices=["time", "bandwidth"], default="time"
    )
    return parser.parse_args()


def main():
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
