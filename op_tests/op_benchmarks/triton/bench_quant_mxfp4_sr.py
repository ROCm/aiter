# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark gfx950 MXFP4 stochastic quantization against existing RTN paths."""

import argparse
import sys

import torch
import triton

from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.mx_types import MxScaleRoundModeInt
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _shapes(args) -> list[tuple[int, int]]:
    if args.shape is not None:
        return [tuple(args.shape)]
    return [
        (1, 4096),
        (2048, 4096),
        (8192, 4096),
        (2048, 14336),
        (8192, 14336),
    ]


def run_benchmark(args) -> None:
    providers = args.provider.split(",")
    unknown = set(providers) - {"triton_rtn", "triton_sr", "hip_even"}
    if unknown:
        raise ValueError(f"unsupported providers: {sorted(unknown)}")
    if args.dtype == "fp32" and "hip_even" in providers:
        providers.remove("hip_even")
        print("Skipping hip_even because quant_mxfp4_hip only supports bf16 input")
    if not providers:
        raise ValueError("no providers support the requested dtype")
    palette = [
        ("green", "-"),
        ("blue", "-"),
        ("red", "-"),
    ]
    benchmark = triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=_shapes(args),
        line_arg="provider",
        line_vals=providers,
        line_names=providers,
        styles=palette[: len(providers)],
        ylabel="Time (ms)" if args.metric == "time" else "Bandwidth (GB/s)",
        plot_name=get_caller_name_no_ext(),
        args={"dtype": args.dtype, "metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_quant_mxfp4_sr(M, N, provider, dtype, metric, **_):
        x = torch.randn((M, N), dtype=_dtype(dtype), device="cuda")

        if provider == "triton_rtn":

            def run():
                return dynamic_mxfp4_quant(x)

        elif provider == "triton_sr":

            def run():
                return dynamic_mxfp4_quant(
                    x,
                    use_sr=True,
                    philox_seed=1234,
                )

        elif provider == "hip_even":
            from aiter.ops.quant import quant_mxfp4_hip

            def run():
                return quant_mxfp4_hip(
                    x,
                    group_size=32,
                    round_mode=MxScaleRoundModeInt.Even,
                )

        else:
            raise AssertionError(f"unvalidated provider: {provider}")

        milliseconds = triton.testing.do_bench(run, warmup=25, rep=100)
        if metric == "time":
            return milliseconds
        total_bytes = x.numel() * x.element_size() + M * (N // 2) + M * (N // 32)
        return total_bytes / (milliseconds * 1e-3) * 1e-9

    bench_quant_mxfp4_sr.run(
        save_path="." if args.output else None,
        print_data=True,
    )


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        prog="Benchmark MXFP4 stochastic quantization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--shape", type=int, nargs=2, metavar=("M", "N"))
    parser.add_argument(
        "--provider",
        default="triton_rtn,triton_sr,hip_even",
        help="Comma-separated providers: triton_rtn,triton_sr,hip_even",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument(
        "--metric",
        choices=["time", "bandwidth"],
        default="time",
    )
    parser.add_argument("-o", "--output", action="store_true")
    return parser.parse_args(args)


def main(args=None) -> None:
    run_benchmark(parse_args(args))


if __name__ == "__main__":
    sys.exit(main())
