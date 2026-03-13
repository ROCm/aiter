# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for Gluon A16W16 GEMM kernel (gfx1250).

Usage:
    python bench_gemm_a16w16_gfx1250.py                          # default shapes
    python bench_gemm_a16w16_gfx1250.py --shape 128 256 512      # single shape
    python bench_gemm_a16w16_gfx1250.py --metric time             # latency
    python bench_gemm_a16w16_gfx1250.py --metric bandwidth        # memory bandwidth
    python bench_gemm_a16w16_gfx1250.py --activation silu         # fused activation
    python bench_gemm_a16w16_gfx1250.py --layout TT               # transposed layout
    python bench_gemm_a16w16_gfx1250.py -o                        # save CSV
"""

import sys
import math
import torch
import triton
from typing import Optional

from aiter.ops.gluon.gemm.basic.gemm_a16w16_gfx1250 import gemm_a16w16_gfx1250
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16_gfx1250 import generate_inputs
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    add_argparse_ff,
    get_ff_args,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_benchmark_object,
    get_shape_benchmark_object,
    print_vgpr,
    get_caller_name_no_ext,
)


def get_default_shapes():
    """Default benchmark shapes — all dimensions <= 1000."""
    return [
        [1, 128, 128],
        [16, 128, 128],
        [32, 128, 128],
        [64, 128, 128],
        [128, 128, 128],
        [256, 128, 128],
        [512, 128, 128],
        [1, 256, 512],
        [32, 256, 512],
        [64, 256, 512],
        [128, 256, 512],
        [256, 256, 512],
        [512, 512, 512],
        [128, 512, 256],
        [256, 512, 256],
        [512, 256, 128],
        [1000, 1000, 1000],
        [64, 1000, 1000],
        [128, 1000, 1000],
        [256, 1000, 1000],
    ]


def bench_gemm_fn(
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    activation: Optional[str] = None,
    bias: bool = True,
    **kwargs,
):
    c_dtype = torch.bfloat16
    x, w, bias_tensor = generate_inputs(M, N, K, c_dtype, layout=layout, bias=bias)
    y = torch.empty((M, N), dtype=c_dtype, device="cuda")

    flops = 2.0 * M * N * K
    if activation is not None:
        flops += M * N
    mem_read = (M * K) * x.element_size() + (N * K) * w.element_size()
    mem_write = (M * N) * y.element_size()
    mem = mem_read + mem_write

    ms = triton.testing.do_bench(
        lambda: gemm_a16w16_gfx1250(
            x, w,
            bias=bias_tensor,
            dtype=c_dtype,
            y=y,
            activation=activation,
        ),
        warmup=25,
        rep=100,
    )

    if metric == "time":
        return ms
    elif metric == "throughput":
        return flops / ms * 1e-9
    elif metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9
    else:
        raise ValueError(f"Unknown metric: {metric}")


def run_model_benchmark(args):
    benchmark = get_model_benchmark_object(get_caller_name_no_ext(), args)

    @triton.testing.perf_report([benchmark])
    def bench(M, hidden_dim, intermediate_dim, metric, layer, **kwargs):
        if layer == "fc1":
            if args.no_glu:
                N, K = intermediate_dim, hidden_dim
            else:
                N, K = intermediate_dim * 2, hidden_dim
            N = math.ceil(N / args.tp)
        elif layer == "fc2":
            N, K = hidden_dim, intermediate_dim
            K = math.ceil(K / args.tp)

        return bench_gemm_fn(
            M, N, K, metric, args.layout, activation=args.activation
        )

    bench.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args):
    if not args.shape and not getattr(args, "M", None):
        benchmark = triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=get_default_shapes(),
            x_log=True,
            y_log=True,
            line_arg="unit",
            line_vals=[{
                "throughput": "TFLOPS",
                "time": "Time_(ms)",
                "bandwidth": "Bandwidth_(GB/s)",
            }[args.metric]],
            line_names=[{
                "throughput": "TFLOPS",
                "time": "Time_(ms)",
                "bandwidth": "Bandwidth_(GB/s)",
            }[args.metric]],
            styles=[("green", "-")],
            ylabel={
                "throughput": "Throughput (TFLOPS)",
                "time": "Time (ms)",
                "bandwidth": "Bandwidth (GB/s)",
            }[args.metric],
            plot_name=get_caller_name_no_ext(),
            args={"metric": args.metric},
        )
    else:
        benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)

    @triton.testing.perf_report([benchmark])
    def bench(M, N, K, metric, **kwargs):
        N = math.ceil(N / args.tp)
        return bench_gemm_fn(M, N, K, metric, args.layout, activation=args.activation)

    bench.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"
    if args.model:
        unsupported_args = []
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise Exception(
                    f"Argument '{arg}' is not supported for benchmarking with the --model flag."
                )
        run_model_benchmark(args)
    else:
        unsupported_args = ["fc1", "fc2", "no_glu"]
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise Exception(
                    f"Argument '{arg}' is not supported for benchmarking without the --model flag."
                )
        run_shape_benchmark(args)


def parse_args():
    parser = get_parser(kernel_name="Gluon A16W16 GEMM (gfx1250)")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        help="Fused activation: gelu, gelu_tanh, silu, silu_exp2, relu",
    )
    return get_ff_args(parser)


def main():
    args, defaults = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for Gluon A16W16 kernel...")
        fun = lambda: run_benchmark(args, defaults)
        print_vgpr(fun, get_caller_name_no_ext())
        return 0
    run_benchmark(args, defaults)


if __name__ == "__main__":
    sys.exit(main())
