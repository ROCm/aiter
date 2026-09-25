# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark native group32 FP8 GEMM through its production config dispatch."""

import functools
import math

import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale_group32 import (
    gemm_a8w8_blockscale_group32,
)
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
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale_group32 import (
    generate_inputs,
    run_torch,
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def bench_gemm_fn(m, n, k, metric, args):
    dtype = DTYPES[args.dtype]
    x, w, xs, ws = generate_inputs(m, n, k, args.weight_group_rows)
    y = torch.empty((m, n), dtype=dtype, device=x.device)
    run = functools.partial(
        gemm_a8w8_blockscale_group32,
        x,
        w,
        xs,
        ws,
        dtype=dtype,
        y=y,
        weight_group_rows=args.weight_group_rows,
    )
    if args.test:
        expected = run_torch(x, w, xs, ws, args.weight_group_rows)
        torch.testing.assert_close(
            run(),
            expected.to(dtype),
            rtol={"bf16": 0.016, "fp16": 0.002, "fp32": 3e-5}[args.dtype],
            atol=5e-5 * expected.abs().max().item(),
        )
    ms = (
        triton.testing.do_bench_cudagraph(run, rep=100)
        if args.cudagraph
        else triton.testing.do_bench(run, warmup=25, rep=100)
    )
    if metric == "time":
        return ms
    if metric == "throughput":
        return 2.0 * m * n * k / ms * 1e-9
    if metric == "bandwidth":
        num_bytes = sum(t.numel() * t.element_size() for t in (x, w, xs, ws, y))
        return num_bytes / ms * 1e-6
    raise ValueError(f"Unknown metric: {metric}")


def run_benchmark(args):
    name = get_caller_name_no_ext()
    benchmark = (
        get_model_benchmark_object(name, args)
        if args.model
        else get_shape_benchmark_object(name, args)
    )

    @triton.testing.perf_report([benchmark])
    def bench(
        M,
        metric,
        N=None,
        K=None,
        hidden_dim=None,
        intermediate_dim=None,
        layer=None,
        **kwargs,
    ):
        if args.model:
            if layer == "fc1":
                N = intermediate_dim * (1 if args.no_glu else 2)
                K = hidden_dim
            else:
                N, K = hidden_dim, math.ceil(intermediate_dim / args.tp)
        if not args.model or layer == "fc1":
            N = math.ceil(N / args.tp)
        return bench_gemm_fn(M, N, K, metric, args)

    bench.run(save_path="." if args.o else None, print_data=True)


def parse_args(argv=None):
    parser = add_argparse_ff(get_parser(kernel_name="Native group32 A8W8 GEMM"))
    parser.add_argument("--weight-group-rows", type=int, choices=(1, 32), default=32)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    parser.add_argument(
        "--cudagraph", action="store_true", help="Time CUDA graph replay"
    )
    parser.add_argument(
        "-test", action="store_true", help="Check against FP64 reference"
    )
    args, _ = get_ff_args(parser, args=argv)
    if args.shape is not None and (len(args.shape) != 3 or args.model):
        parser.error("Use --shape M N K or --model with an optional -M")
    if args.layout != "TN":
        parser.error("Native group32 weights require --layout TN")
    if args.tp < 1:
        parser.error("-tp must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.print_vgpr:
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
