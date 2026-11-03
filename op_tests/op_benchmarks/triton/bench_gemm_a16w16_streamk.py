# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import math

import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_a16w16_streamk import gemm_a16w16_streamk
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_do_bench_graph_kwargs,
    get_ff_args,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
    get_model_benchmark_object,
    get_shape_benchmark_object,
    print_vgpr,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
    generate_gemm_a16w16_inputs,
)


def _generate_inputs(M, N, K, c_dtype, layout):
    """x, w and a caller-supplied y, which are the kwargs of gemm_a16w16_streamk."""
    x, w, _, _, y = generate_gemm_a16w16_inputs(
        M, N, K, c_dtype, layout=layout, output=True, bias=False
    )
    return x, w, y


def bench_gemm_fn(
    kwargs_rot_bufs: dict,
    kwargs_do_bench_graph: dict,
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    **kwargs,
):
    c_dtype = torch.bfloat16

    _p_gen = functools.partial(_generate_inputs, M, N, K, c_dtype, layout)
    _out_vals_keys = ["x", "w", "y"]  # matches gemm_a16w16_streamk's kwargs

    # Rotating buffers, so the operands are read cold instead of out of L2.
    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [_out_vals_keys], **kwargs_rot_bufs
    )

    flops = 2.0 * M * N * K
    mem_read = (M * K) * rot_bufs[0]["x"].element_size() + (N * K) * rot_bufs[0][
        "w"
    ].element_size()
    mem_write = (M * N) * rot_bufs[0]["y"].element_size()
    mem = mem_read + mem_write

    ms = do_bench_aiter_triton(gemm_a16w16_streamk, rot_bufs, **kwargs_do_bench_graph)

    if metric == "time":
        return ms
    elif metric == "throughput":
        return flops / ms * 1e-9  # TFLOP/s
    elif metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9  # GB/s
    else:
        raise ValueError("Unknown metric: " + metric)


def run_model_benchmark(args):
    """Runs the benchmark for the layers of a --model."""
    benchmark = get_model_benchmark_object(get_caller_name_no_ext(), args)

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a16w16_streamk(
        M, hidden_dim, intermediate_dim, metric, layer, **kwargs
    ):
        if layer == "fc1":
            N = intermediate_dim if args.no_glu else intermediate_dim * 2
            K = hidden_dim
            N = math.ceil(N / args.tp)  # tensor parallel splits fc1 on N
        elif layer == "fc2":
            N, K = hidden_dim, intermediate_dim
            K = math.ceil(K / args.tp)  # and fc2 on K
        return bench_gemm_fn(
            kwargs_rot_bufs,
            kwargs_do_bench_graph,
            M,
            N,
            K,
            metric,
            args.layout,
        )

    bench_gemm_a16w16_streamk.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args):
    """Runs the benchmark for the given tensor shapes."""
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a16w16_streamk(M, N, K, metric, **kwargs):
        N = math.ceil(N / args.tp)
        return bench_gemm_fn(
            kwargs_rot_bufs,
            kwargs_do_bench_graph,
            M,
            N,
            K,
            metric,
            args.layout,
        )

    bench_gemm_a16w16_streamk.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"

    if args.model:
        run_model_benchmark(args)
    else:
        unsupported_args = ["fc1", "fc2", "no_glu"]
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise RuntimeError(
                    f"Argument '{arg}' is not supported for benchmarking without the "
                    f"--model flag."
                )
        run_shape_benchmark(args)


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="A16W16 WG cluster StreamK GEMM")
    parser = add_argparse_ff(parser)
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
