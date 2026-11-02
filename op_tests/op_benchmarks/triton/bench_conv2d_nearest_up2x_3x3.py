# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for conv2d_nearest_up2x_3x3 against a nearest x2 upsample followed by
the 3x3 conv, with eager F.conv2d and with aiter conv2d.
"""

import functools
import sys

import torch
import torch.nn.functional as F
import triton

from aiter.ops.triton.conv.conv2d import conv2d
from aiter.ops.triton.conv.conv2d_nearest_up2x_3x3 import conv2d_nearest_up2x_3x3
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.conv.test_conv2d_nearest_up2x_3x3 import (
    generate_nearest_up2x_conv3x3_inputs,
    torch_nearest_up2x_conv3x3_eager,
)

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# triton: this op. eager: F.interpolate + F.conv2d. aiter_unfused: F.interpolate
# + aiter conv2d.
_PROVIDERS = ("triton", "eager", "aiter_unfused")
# (N, C, K, H, W) with H, W before the upsample: Wan VAE decode upsample convs
# (per-frame Conv2d calls, N = frames per call). Wan2.1 T2V 1.3B at 480 x 832
# and Wan2.2 TI2V 5B at 704 x 1280.
_SHAPES = [
    (2, 384, 192, 60, 104),
    (4, 384, 192, 120, 208),
    (4, 192, 96, 240, 416),
    (2, 1024, 1024, 44, 80),
    (4, 1024, 1024, 88, 160),
    (4, 512, 512, 176, 320),
]


def get_benchmark_shapes(args):
    """Return [(N, C, K, H, W), ...] for the current CLI args."""
    if args.shape:
        return [tuple(args.shape)]
    return _SHAPES


def bench_fn_impl(N, C, K, H, W, provider, metric, args):
    dtype = arg_to_torch_dtype[args.dtype]
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)
    _p_gen = functools.partial(
        generate_nearest_up2x_conv3x3_inputs, N, C, K, H, W, dtype, True
    )
    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [["x", "weight", "bias"]], **kwargs_rot_bufs
    )

    if provider == "triton":

        def _run(x, weight, bias):
            conv2d_nearest_up2x_3x3(x, weight, bias)

    elif provider == "eager":

        def _run(x, weight, bias):
            torch_nearest_up2x_conv3x3_eager(x, weight, bias)

    else:

        def _run(x, weight, bias):
            xu = F.interpolate(x, scale_factor=2.0, mode="nearest-exact")
            conv2d(xu, weight, bias, (1, 1), (1, 1))

    ms = do_bench_aiter_triton(_run, rot_bufs, **kwargs_do_bench_graph)
    if metric == "time":
        return ms  # ms
    if metric == "throughput":
        # Multiply-adds of the conv on the upsampled tensor, for all providers.
        flops = 2.0 * N * K * C * 9 * (2 * H) * (2 * W)
        return flops / (ms * 1e-3) * 1e-12  # TFLOPS
    raise ValueError(f"unknown metric {metric}")


def run_benchmark(args):
    providers = _PROVIDERS if args.provider == "all" else (args.provider,)
    metrics = ("time", "throughput") if args.metric == "all" else (args.metric,)
    line_vals = [f"{p}_{m}" for m in metrics for p in providers]

    benchmark = triton.testing.Benchmark(
        x_names=["N", "C", "K", "H", "W"],
        x_vals=get_benchmark_shapes(args),
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_vals,
        styles=[
            ("red", "-"),
            ("blue", "-"),
            ("green", "-"),
            ("yellow", "-"),
            ("cyan", "-"),
            ("magenta", "-"),
        ][: len(line_vals)],
        ylabel="",
        plot_name=get_caller_name_no_ext() + f"_{args.dtype}",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(N, C, K, H, W, provider):
        name, metric = provider.rsplit("_", 1)
        return bench_fn_impl(N, C, K, H, W, name, metric, args)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = get_parser(kernel_name="nearest x2 upsample + 3x3 conv2d")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=5,
        default=None,
        metavar=("N", "C", "K", "H", "W"),
        help="One shape (H, W before the upsample); default: Wan VAE upsample convs",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        choices=[*_PROVIDERS, "all"],
        help="This op, upsample + eager F.conv2d, upsample + aiter conv2d, or all",
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=list(arg_to_torch_dtype)
    )
    override_parser_action(
        parser,
        action="--metric",
        choices=["all", "time", "throughput"],
        default="all",
    )
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels",
    )
    parser.add_argument(
        "-o", action="store_true", default=False, help="Write results to a CSV file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for conv2d_nearest_up2x_3x3 Triton kernels...")
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
