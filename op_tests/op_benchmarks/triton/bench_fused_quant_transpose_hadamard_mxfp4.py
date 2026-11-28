# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark fused quantization, transpose and H16 MXFP4."""

import functools
import sys

import torch
import triton

from aiter.ops.triton.quant import fused_quant_transpose_hadamard_mxfp4
from aiter.ops.triton.utils._triton import arch_info
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.quant.test_fused_quant_transpose_hadamard_mxfp4 import (
    _signs,
)

_BLOCK_SIZE = 32


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _generate_rot_inputs(M: int, N: int, dtype: str):
    return (torch.randn((M, N), dtype=_dtype(dtype), device="cuda"),)


def _generate_fixed_inputs():
    return (_signs(torch.device("cuda")),)


_ROT_KEYS = ["x"]
_FIXED_KEYS = ["sign"]


def _default_shapes() -> list[tuple[int, int]]:
    return [
        (512, 4096),
        (2048, 4096),
        (4096, 4096),
        (6144, 4096),
        (12288, 4096),
        (24576, 4096),
        (4096, 12288),
    ]


def run_benchmark(args) -> None:
    if arch_info.get_arch() != "gfx950":
        raise RuntimeError(
            "fused quantization, transpose and Hadamard MXFP4 requires gfx950"
        )
    shapes = [tuple(args.shape)] if args.shape is not None else _default_shapes()
    providers = args.provider.split(",")
    known_providers = {"fused_rtn", "fused_rtn_layout", "fused_sr"}
    unknown_providers = set(providers) - known_providers
    if unknown_providers:
        raise ValueError(f"unknown providers: {sorted(unknown_providers)}")
    if args.metric == "utilization" and args.peak_bandwidth_gbps is None:
        raise ValueError("--peak-bandwidth-gbps is required for utilization")

    units = {
        "time": "Time (ms)",
        "bandwidth": "Effective bandwidth (GB/s)",
        "utilization": "Peak-bandwidth utilization (%)",
    }
    styles = {
        "fused_rtn": ("green", "-"),
        "fused_rtn_layout": ("green", "--"),
        "fused_sr": ("red", "-"),
    }
    benchmark = triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=shapes,
        x_log=True,
        y_log=args.metric != "utilization",
        line_arg="provider",
        line_vals=providers,
        line_names=providers,
        styles=[styles[provider] for provider in providers],
        ylabel=units[args.metric],
        plot_name=get_caller_name_no_ext(),
        args={
            "dtype": args.dtype,
            "metric": args.metric,
            "peak_bandwidth_gbps": args.peak_bandwidth_gbps,
        },
    )

    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

    @triton.testing.perf_report([benchmark])
    def bench_fused_quant_transpose_hadamard_mxfp4(
        M,
        N,
        provider,
        dtype,
        metric,
        peak_bandwidth_gbps,
    ):
        if provider == "fused_rtn":

            def fn(x, sign):
                return fused_quant_transpose_hadamard_mxfp4(x, sign)

        elif provider == "fused_rtn_layout":

            def fn(x, sign):
                return fused_quant_transpose_hadamard_mxfp4(
                    x,
                    sign,
                    swizzle_scale=True,
                    shuffle_col=True,
                )

        elif provider == "fused_sr":

            def fn(x, sign):
                return fused_quant_transpose_hadamard_mxfp4(
                    x,
                    sign,
                    use_sr_row=True,
                    use_sr_transposed=True,
                    philox_seed=1234,
                )

        else:
            raise AssertionError(f"unvalidated provider: {provider}")

        bencher.build_pool(
            [functools.partial(_generate_rot_inputs, M, N, dtype)],
            [_ROT_KEYS],
            [_generate_fixed_inputs],
            [_FIXED_KEYS],
        )
        elapsed_ms = bencher.bench(fn)

        # Useful traffic for the fused contract: one dense read plus both
        # packed payloads and both raw E8M0 scale grids.
        numel = M * N
        total_bytes = numel * _dtype(dtype).itemsize + numel + numel // 16
        bandwidth_gbps = total_bytes / (elapsed_ms * 1e-3) * 1e-9
        if metric == "time":
            return elapsed_ms
        if metric == "bandwidth":
            return bandwidth_gbps
        if metric == "utilization":
            return 100.0 * bandwidth_gbps / peak_bandwidth_gbps
        raise ValueError(f"unknown metric: {metric}")

    bench_fused_quant_transpose_hadamard_mxfp4.run(
        save_path="." if args.output else None,
        print_data=True,
    )


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="fused quantization, transpose and H16 MXFP4")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("M", "N"),
        help="Single 32-aligned shape to benchmark.",
    )
    parser.add_argument(
        "--provider",
        default="fused_rtn,fused_rtn_layout,fused_sr",
        help="Comma-separated providers: fused_rtn, fused_rtn_layout, fused_sr.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    override_parser_action(
        parser,
        action="--metric",
        choices=["time", "bandwidth", "utilization"],
        default="bandwidth",
    )
    parser.add_argument(
        "--peak-bandwidth-gbps",
        type=float,
        help="Hardware peak bandwidth used by --metric utilization.",
    )
    parser.add_argument("-o", "--output", action="store_true")
    return parser.parse_args(args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args)
    if parsed_args.shape is not None and any(
        dimension <= 0 or dimension % _BLOCK_SIZE != 0
        for dimension in parsed_args.shape
    ):
        raise ValueError("--shape dimensions must be positive multiples of 32")
    run_benchmark(parsed_args)


if __name__ == "__main__":
    sys.exit(main())
