# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark 32x32 block-scaled MXFP4 quantization."""

import argparse
import sys

import torch
import triton

from aiter.ops.triton.quant import dynamic_mxfp4_quant_blockscale
from aiter.utility.fp4_utils import (
    e8m0_to_f32,
    f32_to_mx_e8m0_scale,
    f32_to_mxfp4,
)
from aiter.utility.mx_types import MxDtypeInt, MxScaleRoundModeInt
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)

_BLOCK_SIZE = 32


def _torch_mxfp4_quant_blockscale(x: torch.Tensor):
    """Semantically equivalent eager-PyTorch baseline."""
    M, N = x.shape
    tiles = (
        x.float()
        .reshape(M // _BLOCK_SIZE, _BLOCK_SIZE, N // _BLOCK_SIZE, _BLOCK_SIZE)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    amax = tiles.abs().amax(dim=(-2, -1))
    scales = f32_to_mx_e8m0_scale(
        amax,
        mode=MxScaleRoundModeInt.Even,
        dtype=MxDtypeInt.FP4_E2M1,
    ).view(torch.uint8)
    scales = scales.clamp_max(254)
    scale_f32 = e8m0_to_f32(scales).float()

    scaled_tiles = tiles / scale_f32[:, :, None, None]
    scaled = scaled_tiles.permute(0, 2, 1, 3).reshape(M, N)
    packed = f32_to_mxfp4(scaled).view(torch.uint8)
    return packed, scales


def _get_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


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


def run_benchmark(args):
    shapes = [tuple(args.shape)] if args.shape is not None else _default_shapes()
    providers = args.provider.split(",")
    unknown_providers = set(providers) - {"aiter", "torch"}
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
        "aiter": ("green", "-"),
        "torch": ("blue", "-"),
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

    @triton.testing.perf_report([benchmark])
    def bench_quant_mxfp4_blockscale(
        M,
        N,
        provider,
        dtype,
        metric,
        peak_bandwidth_gbps,
    ):
        x = torch.randn((M, N), dtype=_get_dtype(dtype), device="cuda")
        fn = (
            (lambda: dynamic_mxfp4_quant_blockscale(x))
            if provider == "aiter"
            else (lambda: _torch_mxfp4_quant_blockscale(x))
        )
        elapsed_ms = triton.testing.do_bench(fn, warmup=25, rep=100)

        # Minimum semantic traffic: read input, write packed payload and scale grid.
        total_bytes = x.numel() * x.element_size() + M * N // 2 + M * N // 1024
        bandwidth_gbps = total_bytes / (elapsed_ms * 1e-3) * 1e-9
        if metric == "time":
            return elapsed_ms
        if metric == "bandwidth":
            return bandwidth_gbps
        if metric == "utilization":
            return 100.0 * bandwidth_gbps / peak_bandwidth_gbps
        raise ValueError(f"unknown metric: {metric}")

    bench_quant_mxfp4_blockscale.run(
        save_path="." if args.output else None,
        print_data=True,
    )


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="Benchmark block-scaled MXFP4 quantization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("M", "N"),
        help="Single 32-aligned shape to benchmark.",
    )
    parser.add_argument(
        "--provider",
        default="aiter,torch",
        help="Comma-separated providers from: aiter,torch.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="Input dtype.",
    )
    parser.add_argument(
        "--metric",
        choices=["time", "bandwidth", "utilization"],
        default="bandwidth",
        help="Metric to report.",
    )
    parser.add_argument(
        "--peak-bandwidth-gbps",
        type=float,
        help="Hardware peak bandwidth used by --metric utilization.",
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store_true",
        help="Write performance results to CSV.",
    )
    return parser.parse_args(args=args)


def main(args: list[str] | None = None):
    parsed_args = parse_args(args)
    if parsed_args.shape is not None and any(
        dim <= 0 or dim % _BLOCK_SIZE != 0 for dim in parsed_args.shape
    ):
        raise ValueError("--shape dimensions must be positive multiples of 32")
    run_benchmark(parsed_args)


if __name__ == "__main__":
    sys.exit(main())
