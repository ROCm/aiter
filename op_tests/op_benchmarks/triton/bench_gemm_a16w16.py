# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Unified benchmark for A16W16 GEMM (triton + gluon backends).

Backend selection (similar to pytest -k for tests):
    python bench_gemm_a16w16.py                           # auto-detect (gluon on gfx1250, triton elsewhere)
    python bench_gemm_a16w16.py --backend triton          # triton only
    python bench_gemm_a16w16.py --backend gluon           # gluon only (asserts gfx1250)

Other options:
    python bench_gemm_a16w16.py --shape 128 256 512       # single shape
    python bench_gemm_a16w16.py --metric time             # latency
    python bench_gemm_a16w16.py --activation silu         # fused activation
    python bench_gemm_a16w16.py --layout TT               # transposed layout
    python bench_gemm_a16w16.py -o                        # save CSV
"""

import sys
import torch
import triton
import math
from aiter.ops.triton.gemm.basic.gemm_a16w16 import (
    gemm_a16w16,
    _resolve_backend,
)
from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import gemm_a16w16_atomic
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
    generate_gemm_a16w16_inputs,
)
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
from typing import Optional


def bench_gemm_fn(
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    backend: str,
    atomic: bool = False,
    activation: Optional[str] = None,
    warmup: int = 25,
    rep: int = 100,
    **kwargs,
):
    c_dtype = torch.bfloat16
    x, w, bias, out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, c_dtype, layout=layout, output=True, bias=True
    )
    # flops
    flops = 2.0 * M * N * K
    if activation is not None:
        flops += M * N
    # memory transfer
    mem_read = (M * K) * x.element_size() + (N * K) * w.element_size()
    mem_write = (M * N) * x.element_size()
    mem = mem_read + mem_write

    if atomic:
        assert backend != "gluon", "Atomic kernel is triton-only"
        assert (
            activation is None
        ), "Atomic kernel does not currently support fused activation"
        y = y.to(torch.float32).zero_()
        ms = triton.testing.do_bench(
            lambda: gemm_a16w16_atomic(x, w, torch.float32, y),
            warmup=warmup,
            rep=rep,
        )
    else:
        ms = triton.testing.do_bench(
            lambda: gemm_a16w16(
                x, w, bias, c_dtype, y, activation=activation, backend=backend,
            ),
            warmup=warmup,
            rep=rep,
        )

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


def run_model_benchmark(args, backend):
    """
    Runs benchmark given a --model argument.
    """
    benchmark = get_model_benchmark_object(get_caller_name_no_ext(), args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a16w16(M, hidden_dim, intermediate_dim, metric, layer, **kwargs):
        """
        Fc1:
             M      K                  K           N          M       N
        A = (B, hidden_dim) @ W = (hidden_dim, 2*int_dim) -> (B, 2*int_dim) -> gating -> (B, int_dim)

        Fc2:
             M     K               K          N          M       N
        A = (B, int_dim) @ W = (int_dim, hidden_dim) -> (B, hidden_dim)

        Tensor parallel splits across int_dim (N for fc1, K for fc2)
        """
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
            M, N, K, metric, args.layout, backend,
            atomic=args.atomic, activation=args.activation,
            warmup=args.warmup, rep=args.rep,
        )

    bench_gemm_a16w16.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args, backend):
    """
    Runs a benchmark with given tensor shapes.
    """
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a16w16(M, N, K, metric, **kwargs):
        # Divide N by tensor parallel
        N = math.ceil(N / args.tp)
        return bench_gemm_fn(
            M, N, K, metric, args.layout, backend, atomic=args.atomic,
            warmup=args.warmup, rep=args.rep,
        )

    bench_gemm_a16w16.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"

    backend = _resolve_backend(args.backend)
    print(f"Using backend: {backend}")

    if args.model:
        unsupported_args = []
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise Exception(
                    f"Argument '{arg}' is not supported for benchmarking with the --model flag."
                )
        run_model_benchmark(args, backend)
    else:
        unsupported_args = [
            "fc1",
            "fc2",
            "no_glu",
        ]
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise Exception(
                    f"Argument '{arg}' is not supported for benchmarking without the --model flag."
                )
        run_shape_benchmark(args, backend)


def parse_args():
    parser = get_parser(kernel_name="A16W16 GEMM")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "--atomic",
        action="store_true",
        default=False,
        help="Use the atomic kernel (split-k with atomic_add) instead of the standard a16w16 kernel.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        help="Activation function to apply to the output. One of ('gelu', 'gelu_tanh', 'silu', 'silu_exp2', 'relu').",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["triton", "gluon"],
        default=None,
        help="Backend to use. Default: auto-detect (gluon on gfx1250, triton elsewhere).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="do_bench warmup budget in ms (default: 25). Use ~1 on simulators.",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=100,
        help="do_bench repetition budget in ms (default: 100). Use ~1 on simulators.",
    )
    return get_ff_args(parser)


def main():
    args, defaults = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(args, defaults)  # noqa: E731
        print_vgpr(fun, get_caller_name_no_ext())
        return 0
    run_benchmark(args, defaults)


if __name__ == "__main__":
    sys.exit(main())
