# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math
from collections.abc import Callable

import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale
from aiter.test_common import checkAllclose
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_ff_args,
    get_parser,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    get_model_benchmark_object,
    get_shape_benchmark_object,
    print_vgpr,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale_ue8m0 import (
    e8m0_to_fp32,
    generate_gemm_a8w8_blockscale_ue8m0_inputs,
    get_x_vals,
    run_torch,
)

# DeepSeek-V4.1-Flash dense projections, one row per M.
DEFAULT_M = (1, 4, 64, 1024, 4096)


def get_default_shapes(args):
    ms = (args.M,) if args.M is not None else DEFAULT_M
    return [(M, N, K) for N, K in get_x_vals() for M in ms]


def bench_gemm_fn(
    M: int, N: int, K: int, metric: str, impl: Callable, test: bool, scale_fmt: str
):
    c_dtype = torch.bfloat16
    _, x_q, x_scale, w, w_scale = generate_gemm_a8w8_blockscale_ue8m0_inputs(M, N, K)
    y = torch.empty((M, N), dtype=c_dtype, device=x_q.device)

    ref = run_torch(x_q, x_scale, w, w_scale, c_dtype) if test else None
    if scale_fmt == "fp32":
        # Same numbers, fp32 scales: exercises the generic several-scale-steps
        # per K tile path instead of the tl.dot_scaled one.
        x_scale, w_scale = e8m0_to_fp32(x_scale), e8m0_to_fp32(w_scale)

    if test:
        out = impl(x_q, w, x_scale, w_scale, c_dtype, y)
        checkAllclose(ref, out, msg=f"M={M},N={N},K={K}")

    # flops
    flops = 2.0 * M * N * K
    # memory transfer
    mem_read = (M * K) * x_q.element_size() + (N * K) * w.element_size()
    mem_write = (M * N) * 2  # TODO: Fix for c_dtype != bf16
    mem = mem_read + mem_write

    ms = triton.testing.do_bench(
        lambda: impl(x_q, w, x_scale, w_scale, c_dtype, y),
        warmup=25,
        rep=100,
    )

    if metric == "time":
        return ms
    elif metric == "throughput":
        return flops / ms * 1e-9
    elif metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9  # GB/s
    else:
        raise ValueError("Unknown metric: " + metric)


def run_model_benchmark(args, impl):
    """
    Runs benchmark given a --model argument.
    """
    benchmark = get_model_benchmark_object(get_caller_name_no_ext(), args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a8w8_blockscale_ue8m0(
        M, hidden_dim, intermediate_dim, metric, layer, model_name=None, **kwargs
    ):
        if layer == "fc1":
            if args.no_glu:
                N, K = intermediate_dim, hidden_dim
            else:
                N, K = intermediate_dim * 2, hidden_dim
            N = math.ceil(N / args.tp)
        elif layer == "fc2":
            N, K = hidden_dim, intermediate_dim
            K = math.ceil(K / args.tp)

        return bench_gemm_fn(M, N, K, metric, impl, args.test, args.scale_fmt)

    bench_gemm_a8w8_blockscale_ue8m0.run(
        save_path="." if args.o else None, print_data=True
    )


def run_shape_benchmark(args, impl):
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)
    if not args.shape:
        benchmark.x_vals = get_default_shapes(args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a8w8_blockscale_ue8m0(M, N, K, metric, model_name=None, **kwargs):
        # Divide N by tensor parallel
        N = math.ceil(N / args.tp)
        return bench_gemm_fn(M, N, K, metric, impl, args.test, args.scale_fmt)

    bench_gemm_a8w8_blockscale_ue8m0.run(
        save_path="." if args.o else None, print_data=True
    )


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"
    impl = gemm_a8w8_blockscale
    if args.model:
        run_model_benchmark(args, impl)
    else:
        unsupported_args = [
            "fc1",
            "fc2",
            "no_glu",
        ]
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise RuntimeError(
                    f"Argument '{arg}' is not supported for benchmarking without the --model flag."
                )
        run_shape_benchmark(args, impl)


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="A8W8 GEMM Blockscale 32x32 ue8m0")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "--scale-fmt",
        type=str,
        choices=["ue8m0", "fp32"],
        default="ue8m0",
        help="Block-scale dtype of both operands",
    )
    parser.add_argument(
        "-test",
        action="store_true",
        help="Run a correctness check for each benchmarked shape against a "
        "torch reference (mirrors "
        "op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale_ue8m0.py).",
    )
    return get_ff_args(parser, args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args, defaults = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args, defaults)
        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(parsed_args, defaults)


if __name__ == "__main__":
    main()
