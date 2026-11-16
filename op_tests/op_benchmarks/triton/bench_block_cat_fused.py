# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for the fused cat([leaky_relu(l12[..., :W1]), l12[..., W1:] * l4]) op.
"""

import argparse
import sys

import torch
import triton

from aiter.ops.triton.fusions.block_cat_fused import block_cat_fused
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.fusions.test_block_cat_fused import (
    eager_block_cat,
    generate_block_cat_fused_inputs,
)

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

_PROVIDERS = ("fused", "eager")
_ROWS = (16384, 65536, 262144, 960000)


def get_benchmark_shapes(args):
    """Return [(M, W12), ...] for the current CLI args."""
    if args.M:
        return [(args.M, args.W12)]
    return [(m, args.W12) for m in _ROWS]


def _bytes_moved(M, W12, W1, elem_size, mode):
    W4 = W12 - W1
    # fwd reads l12 + l4 and writes out; bwd reads grad_out + l12 + l4 and
    # writes grad_l12 + grad_l4
    n = 2 * W12 + W4
    if mode == "fwd_bwd":
        n += 3 * W12 + 2 * W4
    return M * n * elem_size


def bench_block_cat_fused_fn(M, W12, provider, metric, args):
    dtype = arg_to_torch_dtype[args.dtype]
    W1 = args.W1
    l12, l4 = generate_block_cat_fused_inputs((M, W12), W1, dtype)
    op = block_cat_fused if provider == "fused" else eager_block_cat
    grad_to_none = None

    if args.mode == "fwd":

        def fn():
            return op(l12, l4, args.slope, W1)

    else:
        l12.requires_grad_(True)
        l4.requires_grad_(True)
        grad_out = torch.randn_like(l12)
        grad_to_none = [l12, l4]

        def fn():
            op(l12, l4, args.slope, W1).backward(grad_out)

    ms = triton.testing.do_bench(
        fn, warmup=args.warmup, rep=args.rep, grad_to_none=grad_to_none
    )
    if metric == "time":
        return ms * 1000  # us
    if metric == "bandwidth":
        mem = _bytes_moved(M, W12, W1, l12.element_size(), args.mode)
        return mem / (ms * 1e-3) * 1e-9  # GB/s
    raise ValueError(f"unknown metric {metric}")


def run_benchmark(args):
    providers = _PROVIDERS if args.provider == "all" else (args.provider,)
    metrics = ("time", "bandwidth") if args.metric == "all" else (args.metric,)
    line_vals = [f"{p}_{m}" for m in metrics for p in providers]

    benchmark = triton.testing.Benchmark(
        x_names=["M", "W12"],
        x_vals=get_benchmark_shapes(args),
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_vals,
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-")][
            : len(line_vals)
        ],
        ylabel="",
        plot_name=get_caller_name_no_ext() + f"_{args.mode}_W1_{args.W1}_{args.dtype}",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(M, W12, provider):
        name, metric = provider.rsplit("_", 1)
        return bench_block_cat_fused_fn(M, W12, name, metric, args)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark block_cat_fused",
        description="Benchmark the Triton fused block_cat op against the eager chain",
        allow_abbrev=False,
    )
    parser.add_argument("-M", type=int, default=None, help="Number of rows")
    parser.add_argument("--W12", type=int, default=576, help="Width of l12")
    parser.add_argument(
        "--W1", type=int, default=128, help="Width of the leaky_relu half"
    )
    parser.add_argument("--slope", type=float, default=0.01)
    parser.add_argument(
        "--mode",
        type=str,
        default="fwd_bwd",
        choices=["fwd", "fwd_bwd"],
        help="Forward only, or forward plus backward",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        choices=[*_PROVIDERS, "all"],
        help="fused op, eager cat/leaky_relu/mul chain, or both",
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=list(arg_to_torch_dtype)
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=["all", "time", "bandwidth"],
        help="Metric to report (default: all)",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
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
        print("Retrieving VGPR usage for block_cat_fused Triton kernels...")
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
