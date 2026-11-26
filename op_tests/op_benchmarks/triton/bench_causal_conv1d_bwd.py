# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark of causal_conv1d_split_qkv_bwd_triton_fn (the training backward).

The default cases are Kimi-K3 KDA short convs (width 4): the TP8 shard
(q, k, v = 1536 channels each, x row stride 6288) and TP1 (12288 channels each).
Times come from RotatingBufferBench: CUDA-graph replays over a rotating pool of inputs.

Usage examples
--------------
python bench_causal_conv1d_bwd.py
python bench_causal_conv1d_bwd.py --seqlens 4096 4096 --k-dim 12288 --v-dim 12288 --row-stride 36864
"""

import types
from functools import partial

import torch

from aiter import logger
from aiter.ops.triton.gated_delta_net import causal_conv1d_split_qkv_bwd_triton_fn
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.gated_delta_net.test_causal_conv1d_bwd import (
    _make_inputs,
)

DTYPE = torch.bfloat16
INPUT_KEYS = ["dq", "dk", "dv", "x", "weight", "query_start_loc"]

# name: (seqlens, k_dim, v_dim, x row stride)
DEFAULT_CASES = {
    "TP8, 1 x 8192, row stride 6288": ([8192], 1536, 1536, 6288),
    "TP8, 2 x 4096, row stride 6288": ([4096] * 2, 1536, 1536, 6288),
    "TP8, varlen 8 x ~2048, gathered": (
        [2048, 1500, 2600, 2048, 1, 3000, 2000, 1185],
        1536,
        1536,
        4608,
    ),
    "TP8, 1 x 32768, gathered": ([32768], 1536, 1536, 4608),
    "TP1, 1 x 4096, gathered": ([4096], 12288, 12288, 36864),
    "TP1, 1 x 8192, gathered": ([8192], 12288, 12288, 36864),
}


def _gen_inputs(seqlens, k_dim, v_dim, row_stride):
    inp = _make_inputs(
        seqlens,
        k_dim=k_dim,
        v_dim=v_dim,
        row_extra=row_stride - (2 * k_dim + v_dim),
        x_dtype=DTYPE,
        w_dtype=torch.float32,
    )
    return tuple(inp[k] for k in INPUT_KEYS)


def run(args):
    if args.seqlens:
        cases = {"custom": (args.seqlens, args.k_dim, args.v_dim, args.row_stride)}
    else:
        cases = DEFAULT_CASES
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )
    logger.info("%-40s %10s %10s", "case", "time (us)", "TB/s")
    for name, (seqlens, k_dim, v_dim, row_stride) in cases.items():
        dim = 2 * k_dim + v_dim
        gen = partial(_gen_inputs, seqlens, k_dim, v_dim, max(row_stride, dim))
        bencher.build_pool([gen], [INPUT_KEYS])
        # The first (warmup) call stores the schedule; graph replays reuse it.
        fn = partial(
            causal_conv1d_split_qkv_bwd_triton_fn,
            bias=None,
            k_dim=k_dim,
            v_dim=v_dim,
            metadata=types.SimpleNamespace(nums_dict={}),
        )
        us = bencher.bench(fn) * 1e3
        # x and the output gradient read once, dx written once.
        nbytes = 3 * sum(seqlens) * dim * DTYPE.itemsize
        logger.info("%-40s %10.2f %10.2f", name, us, nbytes / us / 1e6)


def parse_args():
    parser = get_parser("causal_conv1d_split_qkv_bwd_triton_fn")
    parser.add_argument("--seqlens", type=int, nargs="+", default=None)
    parser.add_argument("--row-stride", type=int, default=6288)
    parser.add_argument("--k-dim", type=int, default=1536)
    parser.add_argument("--v-dim", type=int, default=1536)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
