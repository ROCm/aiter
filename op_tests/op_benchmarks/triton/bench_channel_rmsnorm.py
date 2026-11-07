# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for channel_rmsnorm (RMS norm over dim 1 + activation) against the
eager diffusers WanRMS_norm + SiLU.
"""

import functools
import sys

import torch
import triton

from aiter.ops.triton.normalization.channel_rmsnorm import channel_rmsnorm
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
from op_tests.triton_tests.normalization.test_channel_rmsnorm import (
    generate_channel_rmsnorm_inputs,
    torch_channel_rmsnorm_ref,
)

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

_PROVIDERS = ("triton", "eager")
# (N, C, T, H, W) inputs of WanRMS_norm in the Wan2.1 (C <= 384) and Wan2.2 VAE decoders.
_WAN_SHAPES = [
    (1, 96, 4, 480, 832),
    (1, 192, 4, 240, 416),
    (1, 384, 1, 60, 104),
    (1, 384, 2, 120, 208),
    (1, 256, 4, 352, 640),
    (1, 512, 4, 176, 320),
    (1, 1024, 1, 44, 80),
    (1, 1024, 2, 88, 160),
]


def get_benchmark_shapes(args):
    """Return [(N, C, T, H, W), ...] for the current CLI args."""
    if args.shape:
        return [tuple(args.shape)]
    return _WAN_SHAPES


def bench_channel_rmsnorm_fn(N, C, T, H, W, provider, metric, args):
    dtype = arg_to_torch_dtype[args.dtype]
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)
    activation = None if args.activation == "none" else args.activation
    _p_gen = functools.partial(
        generate_channel_rmsnorm_inputs, (N, C, T, H, W), dtype, args.bias
    )
    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [["x", "gamma", "bias"]], **kwargs_rot_bufs
    )
    x = rot_bufs[0]["x"]
    # The kernel reads x once and writes the output once.
    mem = 2 * x.numel() * x.element_size()

    if provider == "triton":

        def _run(x, gamma, bias):
            channel_rmsnorm(x, gamma, bias, activation)

    else:

        def _run(x, gamma, bias):
            torch_channel_rmsnorm_ref(x, gamma, bias, activation)

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
        x_names=["N", "C", "T", "H", "W"],
        x_vals=get_benchmark_shapes(args),
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_vals,
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-")][
            : len(line_vals)
        ],
        ylabel="",
        plot_name=get_caller_name_no_ext() + f"_{args.dtype}_{args.activation}",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(N, C, T, H, W, provider):
        name, metric = provider.rsplit("_", 1)
        return bench_channel_rmsnorm_fn(N, C, T, H, W, name, metric, args)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = get_parser(kernel_name="channel RMS norm")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=5,
        default=None,
        metavar=("N", "C", "T", "H", "W"),
        help="One input shape; default: the Wan VAE decoder shapes",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        choices=[*_PROVIDERS, "all"],
        help="Triton kernel, eager PyTorch, or both",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="silu",
        choices=["silu", "none"],
        help="Activation after the norm",
    )
    parser.add_argument("--bias", action="store_true", help="Add a bias")
    parser.add_argument(
        "--dtype", type=str, default="fp32", choices=list(arg_to_torch_dtype)
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
        print("Retrieving VGPR usage for channel_rmsnorm Triton kernels...")
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
