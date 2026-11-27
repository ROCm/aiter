# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark for lse_merge, which merges attention partials over disjoint key ranges.

The default shapes are those that mha_v4(kv_splits="auto") makes on MI355X, and one large
explicit split. The time is from CUDA-graph replay over rotating buffers.
"""

import functools
import sys

import torch
import triton

from aiter.ops.triton.attention.utils import lse_merge
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
    positive_int,
    pow2_int,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
)

# (batch, sequence, heads, splits)
DEFAULT_SHAPES = [
    (1, 4352, 3, 4),
    (1, 8192, 3, 2),
    (1, 3586, 1, 7),
    (1, 5000, 2, 5),
    (2, 4352, 1, 4),
    (1, 16384, 8, 4),
]
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_OUT_VALS_KEYS = ["partial_out", "partial_lse", "out", "lse"]


def _inputs(batch, sequence, heads, splits, head_dim, dtype):
    partial_out = torch.randn(
        (batch * splits, sequence, heads, head_dim), dtype=dtype, device="cuda"
    )
    partial_lse = torch.randn(
        (batch * splits, heads, sequence), dtype=torch.float32, device="cuda"
    )
    out = torch.empty((batch, sequence, heads, head_dim), dtype=dtype, device="cuda")
    lse = torch.empty((batch, heads, sequence), dtype=torch.float32, device="cuda")
    return partial_out, partial_lse, out, lse


def _bytes_moved(batch, sequence, heads, splits, head_dim, dtype, write_lse):
    """Each partial row and its LSE are read once, and each merged row is written once."""
    row = head_dim * dtype.itemsize
    lse = torch.float32.itemsize
    return (
        batch
        * sequence
        * heads
        * (splits * (row + lse) + row + (lse if write_lse else 0))
    )


def benchmark(args):
    dtype = DTYPES[args.dtype]
    unit = "us" if args.metric == "time" else "GB/s"
    config = triton.testing.Benchmark(
        x_names=["batch", "sequence", "heads", "splits"],
        x_vals=[tuple(args.shape)] if args.shape else DEFAULT_SHAPES,
        line_arg="write_lse",
        line_vals=[False, True],
        line_names=["out", "out + lse"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={},
    )
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report([config])
    def bench_lse_merge(batch, sequence, heads, splits, write_lse):
        gen = functools.partial(
            _inputs, batch, sequence, heads, splits, args.head_dim, dtype
        )
        rot_bufs = generate_rotating_buffers_pool(
            [gen], [_OUT_VALS_KEYS], **kwargs_rot_bufs
        )

        def merge(partial_out, partial_lse, out, lse):
            lse_merge(partial_out, partial_lse, out, lse if write_lse else None)

        ms = do_bench_aiter_triton(merge, rot_bufs, **kwargs_do_bench_graph)
        if args.metric == "time":
            return ms * 1e3
        moved = _bytes_moved(
            batch, sequence, heads, splits, args.head_dim, dtype, write_lse
        )
        return moved * 1e-9 / (ms * 1e-3)

    bench_lse_merge.run(
        save_path="." if args.o else None, print_data=True, show_plots=False
    )


def parse_args():
    parser = get_parser(kernel_name="LSE merge")
    parser.add_argument(
        "--shape",
        type=positive_int,
        nargs=4,
        metavar=("BATCH", "SEQUENCE", "HEADS", "SPLITS"),
        help="One shape to benchmark instead of the default shapes",
    )
    parser.add_argument("--head-dim", type=pow2_int, default=128, help="Head dimension")
    parser.add_argument("--dtype", type=str, default="bf16", choices=list(DTYPES))
    override_parser_action(
        parser, action="--metric", choices=["time", "bandwidth"], default="time"
    )
    parser.add_argument(
        "-o", action="store_true", default=False, help="Write results to CSV"
    )
    args = parser.parse_args()
    if args.model:
        parser.error("--model is not supported; use --shape")
    return args


def main():
    args = parse_args()
    torch.manual_seed(0)
    benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
