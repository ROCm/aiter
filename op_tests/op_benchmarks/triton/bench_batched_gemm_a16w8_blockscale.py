# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import sys
import torch
import triton
import math
from op_tests.triton_tests.gemm.batched.test_batched_gemm_a16w8_blockscale import (
    generate_batched_gemm_a16w8_inputs,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    add_argparse_ff,
    get_ff_args,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_benchmark_object,
    get_shape_benchmark_object,
    batched_model_benchmark_shapes,
    print_vgpr,
    get_caller_name_no_ext,
)
from aiter.ops.triton.gemm.batched.batched_gemm_a16w8 import batched_gemm_a16w8

# Block shape for blockscale quantization
BLOCK_SHAPE_N = 128
BLOCK_SHAPE_K = 128


def bench_gemm_fn(
    batch: int,
    M: int,
    N: int,
    K: int,
    metric: str,
    prequant: bool = False,
    with_bias: bool = False,
):
    """Benchmark function for batched GEMM A16W8."""
    c_dtype = torch.bfloat16
    x, w, w_scale, bias, y = generate_batched_gemm_a16w8_inputs(
        batch, M, N, K, BLOCK_SHAPE_N, BLOCK_SHAPE_K, dtype=c_dtype, with_bias=with_bias, output=True
    )
    
    # FLOPS calculation
    flops = 2.0 * M * N * K * batch
    
    # Memory transfer calculation
    mem_read = x.numel() * x.element_size() + w.numel() * w.element_size()
    mem_read += w_scale.numel() * w_scale.element_size()
    if bias is not None:
        mem_read += bias.numel() * bias.element_size()
    mem_write = y.numel() * y.element_size()
    mem = mem_read + mem_write

    # Benchmark Triton kernel
    ms = triton.testing.do_bench(
        lambda: batched_gemm_a16w8(x, w, w_scale, bias, c_dtype, y, prequant=prequant),
        warmup=25,
        rep=100,
    )

    # Return metric
    if metric == "time":
        return ms
    elif metric == "throughput":
        tflops = flops / ms * 1e-9
        return tflops
    elif metric == "bandwidth":
        bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
        return bandwidth
    else:
        raise ValueError("Unknown metric: " + metric)


def run_model_benchmark(args):
    """Run benchmarks based on model configurations."""
    benchmark = get_model_benchmark_object(
        plot_name=get_caller_name_no_ext(),
        args=args,
        x_names=["M", "hidden_dim", "intermediate_dim", "batch", "model_name"],
        model_benchmark_shapes_fn=batched_model_benchmark_shapes,
    )

    @triton.testing.perf_report([benchmark])
    def bench_batched_gemm_a16w8_model(
        M, hidden_dim, intermediate_dim, batch, metric, layer, **kwargs
    ):
        if layer == "fc1":
            if args.no_glu:
                N, K = intermediate_dim, hidden_dim
            else:
                N, K = intermediate_dim * 2, hidden_dim
            # Divide N by tensor parallel
            N = math.ceil(N / args.tp)
        elif layer == "fc2":
            N, K = hidden_dim, intermediate_dim
            # Divide K by tensor parallel
            K = math.ceil(K / args.tp)

        return bench_gemm_fn(batch, M, N, K, metric, args.prequant, args.bias)

    bench_batched_gemm_a16w8_model.run(
        save_path="." if args.o else None, print_data=True
    )


def run_shape_benchmark(args):
    """Run benchmarks based on specific shapes."""
    benchmark = get_shape_benchmark_object(
        plot_name=get_caller_name_no_ext(),
        args=args,
        x_names=["batch", "M", "N", "K"],
    )

    @triton.testing.perf_report([benchmark])
    def bench_batched_gemm_a16w8_shape(
        batch,
        M,
        N,
        K,
        metric,
        **kwargs,
    ):
        return bench_gemm_fn(batch, M, N, K, metric, args.prequant, args.bias)

    bench_batched_gemm_a16w8_shape.run(
        save_path="." if args.o else None, print_data=True
    )


def run_benchmark(args, defaults):
    """Main benchmark runner."""
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
        unsupported_args = [
            "fc1",
            "fc2",
            "no_glu",
            "tp",
        ]
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise Exception(
                    f"Argument '{arg}' is not supported for benchmarking without the --model flag."
                )
        run_shape_benchmark(args)


def parse_args():
    """Parse command-line arguments."""
    parser = get_parser("Batched FP16/BF16 x FP8 GEMM with Blockscale")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "-B",
        type=int,
        required=False,
        help="Batch size to be used when using --model flag.",
    )
    parser.add_argument(
        "--prequant",
        action="store_true",
        help="Enable prequantization of activations to FP8.",
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        help="Include bias in the computation.",
    )
    return get_ff_args(parser)


def main():
    """Main entry point."""
    args, defaults = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(args, defaults)  # noqa: E731
        print_vgpr(fun, get_caller_name_no_ext())
        return 0
    run_benchmark(args, defaults)


if __name__ == "__main__":
    sys.exit(main())
