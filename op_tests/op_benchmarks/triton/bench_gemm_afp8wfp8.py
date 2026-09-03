# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math
import warnings

import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import (
    gemm_afp8wfp8,
    gemm_afp8wfp8_preshuffle,
)
from aiter.ops.triton.utils._triton import arch_info
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_ff_args,
    get_parser,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_benchmark_object,
    get_shape_benchmark_object,
    print_vgpr,
)
from op_tests.triton_tests.gemm.basic.test_gemm_afp8wfp8 import generate_inputs

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def bench_gemm_fn(
    M: int,
    N: int,
    K: int,
    metric: str,
    dtype: torch.dtype,
    preshuffle: bool,
    backend: str | None,
    x_scale_group_size: int,
    transpose_x_scale: bool,
    cudagraph: bool = False,
):
    # The 128x128 e8m0 weight-scale layout only exists for shapes that tile it
    # exactly. The shared default sweep (and some model dims) include shapes that
    # do not -- report them as nan rather than aborting the whole sweep.
    if N % 128 != 0 or K % 128 != 0:
        warnings.warn(
            f"Skipping M={M} N={N} K={K}: the 128x128 W-scale layout needs "
            "N % 128 == 0 and K % 128 == 0."
        )
        return float("nan")

    torch.manual_seed(0)
    x, _, w_kernel, _, x_scales_kernel, w_scales = generate_inputs(
        M,
        N,
        K,
        shuffle=preshuffle,
        x_scale_group_size=x_scale_group_size,
        transpose_x_scale=transpose_x_scale,
    )
    y = torch.empty((M, N), dtype=dtype, device=x.device)

    # flops
    flops = 2.0 * M * N * K
    # memory transfer
    mem_read = x.numel() * x.element_size() + w_kernel.numel() * w_kernel.element_size()
    mem_read += (
        x_scales_kernel.numel() * x_scales_kernel.element_size()
        + w_scales.numel() * w_scales.element_size()
    )
    mem_write = y.numel() * y.element_size()
    mem = mem_read + mem_write

    kwargs = {
        "dtype": dtype,
        "y": y,
        "x_scale_group_size": x_scale_group_size,
        "is_x_scale_transposed": transpose_x_scale,
    }
    if preshuffle:
        fn = lambda: gemm_afp8wfp8_preshuffle(
            x, w_kernel, x_scales_kernel, w_scales, backend=backend, **kwargs
        )
    else:
        # Only the preshuffle wrapper has a gluon backend.
        fn = lambda: gemm_afp8wfp8(x, w_kernel, x_scales_kernel, w_scales, **kwargs)

    bench_fn = (
        triton.testing.do_bench_cudagraph if cudagraph else triton.testing.do_bench
    )
    bench_kwargs = {} if cudagraph else {"warmup": 25, "rep": 100}
    ms = bench_fn(fn, **bench_kwargs)

    # Return exactly one scalar depending on which metric is active
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


def _bench_args(args):
    return {
        "dtype": DTYPE_MAP[args.dtype],
        "preshuffle": args.preshuffle,
        "backend": args.backend,
        "x_scale_group_size": args.x_scale_group_size,
        "transpose_x_scale": args.transpose_x_scale,
        "cudagraph": args.cudagraph,
    }


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"
    if args.model:
        unsupported_args = ["layout"]
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise RuntimeError(
                    f"Argument '{arg}' is not supported for benchmarking with the --model flag."
                )
        run_model_benchmark(args)
    else:
        unsupported_args = [
            "fc1",
            "fc2",
            "no_glu",
            "layout",
        ]
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise RuntimeError(
                    f"Argument '{arg}' is not supported for benchmarking without the --model flag."
                )
        run_shape_benchmark(args)


def run_model_benchmark(args):
    benchmark = get_model_benchmark_object("GEMM AFP8 x WFP8 Benchmark", args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_afp8wfp8(
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

        return bench_gemm_fn(M, N, K, metric, **_bench_args(args))

    bench_gemm_afp8wfp8.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args):
    benchmark = get_shape_benchmark_object("GEMM AFP8 x WFP8 Benchmark", args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_afp8wfp8(M, N, K, metric, model_name=None, **kwargs):
        return bench_gemm_fn(M, N, K, metric, **_bench_args(args))

    bench_gemm_afp8wfp8.run(save_path="." if args.o else None, print_data=True)


def parse_args(args: list[str] | None = None):
    parser = get_parser("AFP8 x WFP8 GEMM")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "--shuffle",
        "--preshuffle",
        action="store_true",
        dest="preshuffle",
        help="Preshuffle the weight (layout=(16, 16)) and use gemm_afp8wfp8_preshuffle.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["triton", "gluon"],
        default=None,
        help="Backend for the preshuffle kernel. Default auto-detects (gluon on gfx1250, else triton). Ignored without --shuffle.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=list(DTYPE_MAP.keys()),
        default="bf16",
        help="Output dtype.",
    )
    parser.add_argument(
        "--x_scale_group_size",
        type=int,
        choices=[32, 128],
        default=128,
        help="K elements per activation scale: 128 for blockscale activations, 32 for MX.",
    )
    parser.add_argument(
        "--no_transpose_x_scale",
        action="store_false",
        dest="transpose_x_scale",
        help="Hand the kernel row-major activation scales. Default is the column-major layout per_group_quant_hip(transpose_scale=True) emits.",
    )
    parser.set_defaults(transpose_x_scale=True)
    parser.add_argument(
        "--cudagraph",
        action="store_true",
        default=False,
        help="Use do_bench_cudagraph instead of do_bench to reduce CPU overhead for bandwidth-bound kernels.",
    )
    return get_ff_args(parser, args=args)


def main(args: list[str] | None = None) -> None:
    assert arch_info.is_fp8_avail(), "FP8 is not available on this architecture"

    parsed_args, defaults = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args, defaults)
        print_vgpr(fun, "GEMM")
        return
    run_benchmark(parsed_args, defaults)


if __name__ == "__main__":
    main()
