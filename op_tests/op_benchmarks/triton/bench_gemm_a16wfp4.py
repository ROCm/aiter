# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the BF16 x MXFP4 GEMM (gemm_a16wfp4), with or without the split-K atomic add."""

import functools
import math
import sys

import torch
import triton

from aiter import logger
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
from aiter.ops.triton.utils._triton import arch_info
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_do_bench_graph_kwargs,
    get_ff_args,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    config_launch,
    get_caller_name_no_ext,
    get_model_benchmark_object,
    get_shape_benchmark_object,
    metric_value,
    print_vgpr,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a16wfp4 import (
    generate_gemm_a16wfp4_inputs,
)


def make_bench_case(
    bencher: RotatingBufferBench,
    M: int,
    N: int,
    K: int,
    layout: str,
    atomic: bool,
):
    """(launch, rot_bufs, flops, mem) of one shape.

    With ``atomic`` the split-K partials are added into an fp32 output.
    ``launch(config, **bufs)`` takes a config; None is the installed one.
    The tuning scripts in aiter/ops/triton/utils/_triton/tuning use this function.
    """
    out_dtype = torch.float32 if atomic else torch.bfloat16
    _p_gen = functools.partial(
        generate_gemm_a16wfp4_inputs,
        M,
        N,
        K,
        output=True,
        atomic_add=atomic,
        dtype=torch.bfloat16,
        layout=layout,
        shuffle=False,
    )
    # (x, w, w_shuffled, x_scales, w_scales, w_scales_shuffled, y)
    _out_vals_keys = ["x", "w", "_", "_", "w_scales", "_", "y"]

    rot_bufs = bencher.build_pool([_p_gen], [_out_vals_keys])

    launch = config_launch(
        functools.partial(gemm_a16wfp4, atomic_add=atomic, dtype=out_dtype)
    )

    # flops
    flops = 2.0 * M * N * K
    # memory transfer. w is e2m1, packed two values per byte (K // 2 wide); the
    # scale tensor keeps its own numel (one scale per 32 K-elements).
    bufs = rot_bufs[0]
    mem_read = (M * K) * bufs["x"].element_size() + (N * K // 2) * bufs[
        "w"
    ].element_size()
    mem_read += bufs["w_scales"].numel() * bufs["w_scales"].element_size()
    mem_write = (M * N) * bufs["y"].element_size()
    mem = mem_read + mem_write

    return launch, rot_bufs, flops, mem


def bench_gemm_fn(
    bencher: RotatingBufferBench,
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    atomic: bool,
):
    launch, _, flops, mem = make_bench_case(bencher, M, N, K, layout, atomic)
    ms = bencher.bench(functools.partial(launch, None))
    return metric_value(metric, ms, flops, mem)


def run_model_benchmark(args):
    benchmark = get_model_benchmark_object(get_caller_name_no_ext(), args)

    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a16wfp4(
        M, hidden_dim, intermediate_dim, metric, layer, model_name=None, **kwargs
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

        return bench_gemm_fn(
            bencher,
            M,
            N,
            K,
            metric,
            args.layout,
            args.atomic,
        )

    bench_gemm_a16wfp4.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args):
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)

    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a16wfp4(M, N, K, metric, model_name=None, **kwargs):
        # Divide N by tensor parallel
        N = math.ceil(N / args.tp)
        return bench_gemm_fn(
            bencher,
            M,
            N,
            K,
            metric,
            args.layout,
            args.atomic,
        )

    bench_gemm_a16wfp4.run(save_path="." if args.o else None, print_data=True)


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
                    f"Argument '{arg}' is not supported for benchmarking without the --model flag."
                )
        run_shape_benchmark(args)


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="A16WFP4 GEMM")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "--atomic",
        action="store_true",
        default=False,
        help="Add the split-K partials into an fp32 output with atomics.",
    )
    return get_ff_args(parser, args=args)


def main(args: list[str] | None = None) -> int:
    if not arch_info.is_fp4_avail():
        logger.error("MXFP4 is not available on this architecture")
        return 1
    parsed_args, defaults = parse_args(args=args)
    if parsed_args.print_vgpr:
        logger.info("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args, defaults)
        print_vgpr(fun, get_caller_name_no_ext())
        return 0
    run_benchmark(parsed_args, defaults)
    return 0


if __name__ == "__main__":
    sys.exit(main())
