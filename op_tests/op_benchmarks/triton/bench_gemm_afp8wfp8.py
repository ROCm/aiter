# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark compact-scale FP8 GEMM and its preshuffled weight path.

For the PR group32 sweep, use --shape-set group32 --x-scale-group-size 32
--w-scale-group-size 32 32 --no-transpose-x-scale --cudagraph.
Use --test to validate an FP64 reference before timing each shape.
"""

import csv
import math
import warnings
from pathlib import Path

import torch
import triton

from aiter.ops.shuffle import shuffle_weight
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

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def generate_inputs(M, N, K, preshuffle, a_group, w_group, transpose):
    x = torch.randn((M, K), device="cuda").to(torch.float8_e4m3fn)
    w = torch.randn((N, K), device="cuda").to(torch.float8_e4m3fn)
    xs = torch.randint(125, 130, (M, K // a_group), dtype=torch.uint8, device="cuda")
    ws = torch.randint(
        125,
        130,
        (triton.cdiv(N, w_group[0]), K // w_group[1]),
        dtype=torch.uint8,
        device="cuda",
    )
    xsk = xs.T.contiguous().reshape(xs.shape) if transpose else xs
    wk = shuffle_weight(w.view(torch.uint8), layout=(16, 16)) if preshuffle else w
    return x, w, wk, xs, xsk, ws


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
    w_scale_group_size: tuple[int, int] = (128, 128),
    test: bool = False,
):
    group_n, group_k = w_scale_group_size
    if K % x_scale_group_size or K % group_k or (preshuffle and N % 128):
        warnings.warn(
            f"Skipping M={M} N={N} K={K}: incompatible scale groups or preshuffle alignment."
        )
        return float("nan")

    torch.manual_seed(0)
    x, w, w_kernel, xs, x_scales_kernel, w_scales = generate_inputs(
        M, N, K, preshuffle, x_scale_group_size, w_scale_group_size, transpose_x_scale
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
        fn = lambda: gemm_afp8wfp8(
            x,
            w_kernel,
            x_scales_kernel,
            w_scales,
            w_scale_group_size=w_scale_group_size,
            **kwargs,
        )

    if test:
        a = x.double() * torch.exp2(xs.double() - 127).repeat_interleave(
            x_scale_group_size, 1
        )
        b = w.double() * torch.exp2(w_scales.double() - 127).repeat_interleave(
            group_n, 0
        )[:N].repeat_interleave(group_k, 1)
        expected = a @ b.T
        torch.testing.assert_close(
            fn(),
            expected.to(dtype),
            rtol={torch.bfloat16: 0.016, torch.float16: 0.002, torch.float32: 3e-5}[
                dtype
            ],
            atol=5e-5 * expected.abs().max().item(),
        )

    bench_fn = (
        triton.testing.do_bench_cudagraph if cudagraph else triton.testing.do_bench
    )
    bench_kwargs = {"rep": 100} if cudagraph else {"warmup": 25, "rep": 100}
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
        "w_scale_group_size": tuple(args.w_scale_group_size),
        "test": args.test,
    }


def run_benchmark(args, defaults):
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

    if args.shape_set == "group32":
        shape_file = (
            Path(__file__).resolve().parents[3]
            / "aiter/configs/model_configs/dsv41_a8w8_blockscale_group32_tuned_gemm.csv"
        )
        with shape_file.open() as source:
            benchmark.x_vals = list(
                dict.fromkeys(
                    tuple(int(row[key]) for key in ("M", "N", "K"))
                    for row in csv.DictReader(source)
                    if args.M is None or int(row["M"]) == args.M
                )
            )
        if not benchmark.x_vals:
            raise ValueError("No group32 shapes match the requested M")

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
        "--x-scale-group-size",
        type=int,
        choices=[32, 128],
        default=128,
        help="K elements per activation scale: 128 for blockscale activations, 32 for MX.",
    )
    parser.add_argument(
        "--no_transpose_x_scale",
        "--no-transpose-x-scale",
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
    parser.add_argument(
        "--w-scale-group-size",
        "--w_scale_group_size",
        type=int,
        nargs=2,
        metavar=("N_GROUP", "K_GROUP"),
        default=(128, 128),
        help="Weight scale block: 1 32, 32 32, or 128 128.",
    )
    parser.add_argument(
        "-test",
        "--test",
        action="store_true",
        help="Check an FP64 reference before timing.",
    )
    parser.add_argument(
        "--shape-set",
        choices=("default", "group32"),
        default="default",
        help="Use the default sweep or all 1224 PR group32 shapes; scale options remain explicit.",
    )
    parsed, defaults = get_ff_args(parser, args=args)
    if parsed.shape_set == "group32" and (
        parsed.shape or parsed.model or parsed.tp != 1
    ):
        parser.error(
            "--shape-set group32 uses literal CSV shapes; cannot combine with --shape, --model, or -tp other than 1"
        )
    if parsed.shape is not None and (len(parsed.shape) != 3 or parsed.model):
        parser.error("Use --shape M N K or --model with an optional -M")
    if parsed.tp < 1:
        parser.error("-tp must be positive")
    if tuple(parsed.w_scale_group_size) not in ((1, 32), (32, 32), (128, 128)):
        parser.error("Weight scale block must be 1 32, 32 32, or 128 128")
    if parsed.preshuffle and tuple(parsed.w_scale_group_size) != (128, 128):
        parser.error("Preshuffle requires 128x128 weight scales")
    return parsed, defaults


def main(args: list[str] | None = None) -> None:
    parsed_args, defaults = parse_args(args=args)
    assert arch_info.is_fp8_avail(), "FP8 is not available on this architecture"
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args, defaults)
        print_vgpr(fun, "GEMM")
        return
    run_benchmark(parsed_args, defaults)


if __name__ == "__main__":
    main()
