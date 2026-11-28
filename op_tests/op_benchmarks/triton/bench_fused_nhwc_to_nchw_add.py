# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for fused_nhwc_to_nchw_add against the eager diffusers UNet
attention exit (reshape, permute, contiguous, add).
"""

import functools
import sys

import torch
import triton

from aiter.ops.triton.fusions.fused_nhwc_to_nchw_add import fused_nhwc_to_nchw_add
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
from op_tests.triton_tests.fusions.test_fused_nhwc_to_nchw_add import (
    generate_nhwc_to_nchw_add_inputs,
    torch_nhwc_to_nchw_add,
)

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

_PROVIDERS = ("triton", "eager")
# (B, C, H, W, residual): SDXL 1024 px attention exits (proj_out is a Linear,
# so the residual add follows the permute) and SD1.5 512 px exits (proj_out is
# a 1x1 conv after the permute, so the layout change only). CFG batch 2.
_SHAPES = [
    (2, 640, 64, 64, 1),
    (2, 1280, 32, 32, 1),
    (2, 320, 64, 64, 0),
    (2, 640, 32, 32, 0),
    (2, 1280, 16, 16, 0),
    (2, 1280, 8, 8, 0),
]


def get_benchmark_shapes(args):
    """Return [(B, C, H, W, residual), ...] for the current CLI args."""
    if args.shape:
        return [tuple(args.shape) + (int(not args.no_residual),)]
    return _SHAPES


def bench_fn_impl(B, C, H, W, residual, provider, metric, args):
    dtype = arg_to_torch_dtype[args.dtype]
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)
    _p_gen = functools.partial(
        generate_nhwc_to_nchw_add_inputs, B, C, H, W, dtype, bool(residual)
    )
    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [["x", "residual"]], **kwargs_rot_bufs
    )
    x = rot_bufs[0]["x"]
    # x (and the residual) are read once and the output is written once.
    mem = (3 if residual else 2) * x.numel() * x.element_size()

    if provider == "triton":

        def _run(x, residual):
            fused_nhwc_to_nchw_add(x, residual, H, W)

    else:

        def _run(x, residual):
            torch_nhwc_to_nchw_add(x, residual, H, W)

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
        x_names=["B", "C", "H", "W", "residual"],
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
    def bench_fn(B, C, H, W, residual, provider):
        name, metric = provider.rsplit("_", 1)
        return bench_fn_impl(B, C, H, W, residual, name, metric, args)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = get_parser(kernel_name="NHWC to NCHW + residual")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=4,
        default=None,
        metavar=("B", "C", "H", "W"),
        help="One shape; default: SDXL and SD1.5 attention exits",
    )
    parser.add_argument("--no_residual", action="store_true", help="Layout only")
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        choices=[*_PROVIDERS, "all"],
        help="Triton kernel, eager PyTorch, or both",
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
        print("Retrieving VGPR usage for fused_nhwc_to_nchw_add kernels...")
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
