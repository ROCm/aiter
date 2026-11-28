# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for the fused cat([leaky_relu(l12[..., :W1]), l12[..., W1:] * l4]) op.
"""

import functools
import sys

import torch
import triton

from aiter.ops.triton.fusions.block_cat_fused import block_cat_fused
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.fusions.test_block_cat_fused import (
    generate_block_cat_fused_inputs,
)

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

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


def _generate_fwd_bwd_inputs(M, W12, W1, dtype):
    l12, l4 = generate_block_cat_fused_inputs((M, W12), W1, dtype)
    l12.requires_grad_(True)
    l4.requires_grad_(True)
    return l12, l4, torch.randn_like(l12)


def bench_block_cat_fused_ms(bencher, M, W12, args):
    dtype = arg_to_torch_dtype[args.dtype]
    W1 = args.W1

    if args.mode == "fwd":
        gen = functools.partial(generate_block_cat_fused_inputs, (M, W12), W1, dtype)
        keys = ["l12", "l4"]

        def fn(l12, l4):
            block_cat_fused(l12, l4, args.slope, W1)

    else:
        gen = functools.partial(_generate_fwd_bwd_inputs, M, W12, W1, dtype)
        keys = ["l12", "l4", "grad_out"]

        # autograd.grad returns new gradients and does not add into .grad, so
        # graph replays do not time a gradient accumulation.
        def fn(l12, l4, grad_out):
            out = block_cat_fused(l12, l4, args.slope, W1)
            torch.autograd.grad(out, (l12, l4), grad_out)

    bencher.build_pool([gen], [keys])
    return bencher.bench(fn)


def run_benchmark(args):
    metrics = ("time", "bandwidth") if args.metric == "all" else (args.metric,)
    line_vals = [f"fused_{m}" for m in metrics]

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

    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )
    # Time each shape once and report every metric from that time.
    timings = {}

    @triton.testing.perf_report([benchmark])
    def bench_fn(M, W12, provider):
        metric = provider.rsplit("_", 1)[1]
        if (M, W12) not in timings:
            timings[(M, W12)] = bench_block_cat_fused_ms(bencher, M, W12, args)
        ms = timings[(M, W12)]
        if metric == "time":
            return ms * 1000  # us
        elem_size = arg_to_torch_dtype[args.dtype].itemsize
        mem = _bytes_moved(M, W12, args.W1, elem_size, args.mode)
        return mem / (ms * 1e-3) * 1e-9  # GB/s

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = get_parser(kernel_name="block_cat_fused")
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
        "--dtype", type=str, default="bf16", choices=list(arg_to_torch_dtype)
    )
    override_parser_action(
        parser,
        action="--metric",
        choices=["all", "time", "bandwidth"],
        default="all",
        help="Metric to report",
    )
    # At the largest row count one fwd_bwd input set is about 3 GB. A floor of
    # 4 pool entries would make that 12 GB. Inputs this large do not fit in
    # the cache anyway, so a floor of 2 is enough.
    override_parser_action(parser, action="--rot-bufs-min-pool-len", default=2)
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
