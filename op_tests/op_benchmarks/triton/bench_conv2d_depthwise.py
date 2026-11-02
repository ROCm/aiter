# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for conv2d_depthwise against eager F.conv2d(groups=C), plain and with
the SANA GLUMBConv GLU.
"""

import functools
import sys

import torch
import triton

from aiter.ops.triton.conv.conv2d_depthwise import conv2d_depthwise
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
from op_tests.triton_tests.conv.test_conv2d_depthwise import (
    generate_conv2d_depthwise_inputs,
    torch_conv2d_depthwise_eager,
)

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

_PROVIDERS = ("triton", "eager")
# (N, C, H, W, k, glu): SANA 1600M / 600M GLUMBConv conv_depth at 1024 px (plain
# and GLU), Stable Cascade stage C and stage B ResBlock depthwise convs.
_SHAPES = [
    (1, 11200, 32, 32, 3, 0),
    (2, 11200, 32, 32, 3, 0),
    (2, 11200, 32, 32, 3, 1),
    (2, 5760, 32, 32, 3, 1),
    (2, 2048, 24, 24, 3, 0),
    (2, 1280, 64, 64, 3, 0),
    (2, 640, 128, 128, 3, 0),
    (2, 320, 256, 256, 3, 0),
]


def get_benchmark_shapes(args):
    """Return [(N, C, H, W, k, glu), ...] for the current CLI args."""
    if args.shape:
        return [tuple(args.shape) + (int(args.glu),)]
    return _SHAPES


def bench_fn_impl(N, C, H, W, k, glu, provider, metric, args):
    dtype = arg_to_torch_dtype[args.dtype]
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)
    _p_gen = functools.partial(
        generate_conv2d_depthwise_inputs, N, C, H, W, k, k, dtype, True
    )
    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [["x", "weight", "bias"]], **kwargs_rot_bufs
    )
    x = rot_bufs[0]["x"]
    gate = "silu" if glu else None
    pad = k // 2
    # x is read once; the output (C or C / 2 channels) is written once.
    mem = x.numel() * x.element_size() * (1.5 if glu else 2)

    if provider == "triton":

        def _run(x, weight, bias):
            conv2d_depthwise(x, weight, bias, 1, pad, 1, None, gate)

    else:

        def _run(x, weight, bias):
            torch_conv2d_depthwise_eager(x, weight, bias, 1, pad, 1, None, gate)

    ms = do_bench_aiter_triton(_run, rot_bufs, **kwargs_do_bench_graph)
    if metric == "time":
        return ms * 1000  # us
    if metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9  # GB/s
    raise ValueError(f"unknown metric {metric}")


def run_benchmark(args):
    providers = _PROVIDERS if args.provider == "all" else (args.provider,)
    metrics = ("time", "bandwidth") if args.metric == "all" else (args.metric,)
    line_vals = [f"{p}_{m}" for m in metrics for p in providers]

    benchmark = triton.testing.Benchmark(
        x_names=["N", "C", "H", "W", "k", "glu"],
        x_vals=get_benchmark_shapes(args),
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_vals,
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-")][
            : len(line_vals)
        ],
        ylabel="",
        plot_name=get_caller_name_no_ext() + f"_{args.dtype}",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(N, C, H, W, k, glu, provider):
        name, metric = provider.rsplit("_", 1)
        return bench_fn_impl(N, C, H, W, k, glu, name, metric, args)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = get_parser(kernel_name="depthwise conv2d")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=5,
        default=None,
        metavar=("N", "C", "H", "W", "k"),
        help="One shape (stride 1, same padding); default: SANA and Stable Cascade shapes",
    )
    parser.add_argument("--glu", action="store_true", help="SANA GLU epilogue")
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        choices=[*_PROVIDERS, "all"],
        help="Triton kernel, eager F.conv2d, or both",
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=list(arg_to_torch_dtype)
    )
    override_parser_action(
        parser, action="--metric", choices=["all", "time", "bandwidth"], default="all"
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
        print("Retrieving VGPR usage for conv2d_depthwise Triton kernels...")
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
