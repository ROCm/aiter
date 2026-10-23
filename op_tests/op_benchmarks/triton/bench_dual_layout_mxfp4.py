# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark fused dual-layout H16 MXFP4 quantization."""

import argparse
import sys

import torch
import triton

from aiter.ops.triton.quant import dual_layout_quant_mxfp4, dynamic_mxfp4_quant
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.shuffle import shuffle_scale_gemm, shuffle_weight
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)

_BLOCK_SIZE = 32
_HADAMARD_SIZE = 16


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _hadamard16(device: torch.device) -> torch.Tensor:
    matrix = torch.ones((1, 1), dtype=torch.float32, device=device)
    while matrix.shape[0] < _HADAMARD_SIZE:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / (_HADAMARD_SIZE**0.5)


def _decomposed_rtn(
    x: torch.Tensor,
    sign: torch.Tensor,
    hadamard: torch.Tensor,
):
    M, N = x.shape
    row = dynamic_mxfp4_quant(x)
    transposed_blocks = x.T.float().reshape(N, M // _HADAMARD_SIZE, _HADAMARD_SIZE)
    transposed = (transposed_blocks * sign.float().reshape(1, 1, -1)) @ hadamard
    transposed = transposed.reshape(N, M)
    return (*row, *dynamic_mxfp4_quant(transposed))


def _decomposed_rtn_layout(
    x: torch.Tensor,
    sign: torch.Tensor,
    hadamard: torch.Tensor,
):
    row, row_scale, col, col_scale = _decomposed_rtn(x, sign, hadamard)
    return (
        row,
        shuffle_scale_gemm(
            row_scale,
            arch="gfx950",
            preshuffle_factor=32,
            scale_kwidth=8,
        ),
        shuffle_weight(col, arch="gfx950", layout=(16, 16)),
        shuffle_scale_gemm(
            col_scale,
            arch="gfx950",
            preshuffle_factor=32,
            scale_kwidth=8,
        ),
    )


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
        raise RuntimeError("dual-layout MXFP4 quantization requires gfx950")
    shapes = [tuple(args.shape)] if args.shape is not None else _default_shapes()
    providers = args.provider.split(",")
    known_providers = {
        "fused_rtn",
        "fused_rtn_layout",
        "fused_sr",
        "decomposed_rtn",
        "decomposed_rtn_layout",
    }
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
        "decomposed_rtn": ("blue", "-"),
        "decomposed_rtn_layout": ("blue", "--"),
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
    def bench_dual_layout_mxfp4(
        M,
        N,
        provider,
        dtype,
        metric,
        peak_bandwidth_gbps,
    ):
        x = torch.randn((M, N), dtype=_dtype(dtype), device="cuda")
        sign = torch.where(
            torch.arange(_HADAMARD_SIZE, device=x.device) % 2 == 0,
            -torch.ones(_HADAMARD_SIZE, device=x.device),
            torch.ones(_HADAMARD_SIZE, device=x.device),
        ).to(torch.bfloat16)
        hadamard = _hadamard16(x.device)

        if provider == "fused_rtn":

            def fn():
                return dual_layout_quant_mxfp4(x, sign)

        elif provider == "fused_rtn_layout":

            def fn():
                return dual_layout_quant_mxfp4(
                    x,
                    sign,
                    swizzle_scale=True,
                    shuffle_col=True,
                )

        elif provider == "fused_sr":

            def fn():
                return dual_layout_quant_mxfp4(
                    x,
                    sign,
                    use_sr_row=True,
                    use_sr_transposed=True,
                    philox_seed=1234,
                )

        elif provider == "decomposed_rtn":

            def fn():
                return _decomposed_rtn(x, sign, hadamard)

        elif provider == "decomposed_rtn_layout":

            def fn():
                return _decomposed_rtn_layout(x, sign, hadamard)

        else:
            raise AssertionError(f"unvalidated provider: {provider}")

        elapsed_ms = triton.testing.do_bench(fn, warmup=25, rep=100)

        # Useful traffic for the fused contract: one dense read plus both
        # packed payloads and both raw E8M0 scale grids.
        total_bytes = x.numel() * x.element_size() + x.numel() + x.numel() // 16
        bandwidth_gbps = total_bytes / (elapsed_ms * 1e-3) * 1e-9
        if metric == "time":
            return elapsed_ms
        if metric == "bandwidth":
            return bandwidth_gbps
        if metric == "utilization":
            return 100.0 * bandwidth_gbps / peak_bandwidth_gbps
        raise ValueError(f"unknown metric: {metric}")

    bench_dual_layout_mxfp4.run(
        save_path="." if args.output else None,
        print_data=True,
    )


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="Benchmark dual-layout H16 MXFP4 quantization",
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
        default="fused_rtn,fused_rtn_layout,fused_sr,decomposed_rtn,decomposed_rtn_layout",
        help="Comma-separated fused or decomposed providers.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument(
        "--metric",
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
