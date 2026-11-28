# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import sys

import torch
import triton

from aiter.ops.triton.quant import (
    dequant_hadamard_quant_mxfp4,
    dynamic_mxfp4_quant,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)


def get_default_shapes() -> list[tuple[int, int]]:
    return [
        (512, 4096),
        (2048, 4096),
        (4096, 4096),
        (6144, 4096),
        (4096, 12288),
        (12288, 4096),
    ]


def _generate_rot_inputs(M: int, K: int):
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    return dynamic_mxfp4_quant(x)


def _generate_fixed_inputs():
    return (torch.ones(16, dtype=torch.float32, device="cuda"),)


_ROT_KEYS = ["packed", "scales"]
_FIXED_KEYS = ["sign"]


def run_benchmark(args):
    shapes = [tuple(args.shape)] if args.shape else get_default_shapes()
    providers = args.provider.split(",")
    known_providers = {"fused", "fused_layout", "fused_sr"}
    unknown_providers = set(providers) - known_providers
    if unknown_providers:
        raise ValueError(f"unknown providers: {sorted(unknown_providers)}")

    styles = {
        "fused": ("green", "-"),
        "fused_layout": ("green", "--"),
        "fused_sr": ("red", "-"),
    }
    benchmark = triton.testing.Benchmark(
        x_names=["M", "K"],
        x_vals=shapes,
        line_arg="provider",
        line_vals=providers,
        line_names=providers,
        styles=[styles[provider] for provider in providers],
        ylabel="Time (ms)" if args.metric == "time" else "Bandwidth (GB/s)",
        plot_name="dequant-hadamard-quant-mxfp4",
        args={"metric": args.metric},
    )

    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

    @triton.testing.perf_report([benchmark])
    def bench(M, K, provider, metric):
        torch.manual_seed(17)

        if provider == "fused":

            def fn(packed, scales, sign):
                return dequant_hadamard_quant_mxfp4(packed, scales, sign)

        elif provider == "fused_layout":

            def fn(packed, scales, sign):
                return dequant_hadamard_quant_mxfp4(
                    packed,
                    scales,
                    sign,
                    swizzle_scale=True,
                    shuffle_data=True,
                )

        elif provider == "fused_sr":

            def fn(packed, scales, sign):
                return dequant_hadamard_quant_mxfp4(
                    packed,
                    scales,
                    sign,
                    use_sr=True,
                    philox_seed=1234,
                )

        else:
            raise AssertionError(f"unvalidated provider: {provider}")

        bencher.build_pool(
            [functools.partial(_generate_rot_inputs, M, K)],
            [_ROT_KEYS],
            [_generate_fixed_inputs],
            [_FIXED_KEYS],
        )
        ms = bencher.bench(fn)
        if metric == "time":
            return ms
        # Useful traffic: packed input, its scales, the sign vector, and both
        # outputs. Layout variants permute the same number of bytes.
        input_bytes = M * (K // 2) + M * (K // 32) + 16 * torch.float32.itemsize
        output_bytes = K * (M // 2) + K * (M // 32)
        return (input_bytes + output_bytes) / (ms * 1e-3) * 1e-9

    bench.run(save_path="." if args.output else None, print_data=True)


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="fused MXFP4 dequant-transpose-H16-requant")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("M", "K"),
        help="single logical input shape",
    )
    parser.add_argument(
        "--provider",
        default="fused,fused_layout,fused_sr",
        help="comma-separated providers: fused, fused_layout, fused_sr",
    )
    override_parser_action(
        parser, action="--metric", choices=["time", "bandwidth"], default="time"
    )
    parser.add_argument("-o", "--output", action="store_true")
    return parser.parse_args(args)


def main(args: list[str] | None = None):
    run_benchmark(parse_args(args))


if __name__ == "__main__":
    sys.exit(main())
