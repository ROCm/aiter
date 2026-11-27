# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark of causal_conv1d_split_qkv_triton_tile_fn on x with unit channel stride.

Compares the sliding-window path with the 2D-tiled kernel (AITER_CONV1D_SPLIT_QKV_WINDOW=0), called as vLLM calls
it (block_m=8, prebuilt schedule). The default cases are Kimi-K3 TP8 prefill steps (q, k, v = 1536 channels each).
Times come from RotatingBufferBench: CUDA-graph replays over a rotating pool of inputs.

Usage examples
--------------
python bench_causal_conv1d_prefill.py
python bench_causal_conv1d_prefill.py --seqlens 1112 1112 1112 1112 --row-stride 4608
python bench_causal_conv1d_prefill.py --k-dim 3072 --v-dim 3072 --row-stride 12288 --seqlens 3100
"""

import itertools
import os
import types
from functools import partial

import torch

import aiter.ops.triton.gated_delta_net.causal_conv1d_prefill as conv_prefill
from aiter import logger
from aiter.ops.prefill_batch_metadata import build_causal_conv_prefill_metadata
from aiter.ops.triton.gated_delta_net import causal_conv1d_split_qkv_triton_tile_fn
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.gated_delta_net.test_causal_conv1d_prefill import (
    WIDTH,
    _make_inputs,
)

DTYPE = torch.bfloat16
BLOCK_M = 8
INPUT_KEYS = [
    "x",
    "weight",
    "conv_states",
    "query_start_loc",
    "cache_indices",
    "has_initial_state",
]

DEFAULT_CASES = {
    "one continuing chunk, row stride 6288": ([7680], [True], 6288),
    "continuing + new chunk, gathered": ([5010, 2680], [True, False], 4608),
    "7 new prompts, gathered": ([1112] * 7, [False] * 7, 4608),
    "60 new prompts of 128, gathered": ([128] * 60, [False] * 60, 4608),
    "one prompt of 1024, row stride 6288": ([1024], [False], 6288),
}


def _gen_inputs(seqlens, has_init, k_dim, v_dim, row_stride):
    # Conv states in [slot, dim, cols] with unit dim stride, as vLLM stores them.
    return _make_inputs(
        seqlens,
        has_init,
        k_dim,
        v_dim,
        row_extra=row_stride - (2 * k_dim + v_dim),
        state_cols=WIDTH - 1 + 4,
        sd_layout=True,
        n_slots=2 * len(seqlens),
        pad_seqs=(),
        w_dtype=torch.float32,
        state_dtype=DTYPE,
        seed=0,
    )


def _prebuilt_metadata(seqlens):
    # vLLM passes a plain object with a prebuilt nums_dict schedule (usable in graph capture).
    query_start_loc = torch.tensor(
        [0, *itertools.accumulate(seqlens)], dtype=torch.int32, device="cuda"
    )
    grid = build_causal_conv_prefill_metadata(
        seqlens, query_start_loc=query_start_loc, block_sizes=(BLOCK_M,)
    ).get_chunk_grid(BLOCK_M)
    return types.SimpleNamespace(
        nums_dict={
            BLOCK_M: {
                "tot": grid.total_chunks,
                "batch_ptr": grid.sequence_ids,
                "token_chunk_offset_ptr": grid.chunk_ids,
            }
        }
    )


def _time_us(bencher, fn, window):
    os.environ["AITER_CONV1D_SPLIT_QKV_WINDOW"] = "1" if window else "0"
    return bencher.bench(fn) * 1e3


def run(args):
    if args.seqlens:
        cases = {"custom": (args.seqlens, [True] * len(args.seqlens), args.row_stride)}
    else:
        cases = DEFAULT_CASES
    dim = 2 * args.k_dim + args.v_dim
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )
    logger.info(
        "%-40s %12s %12s %8s %10s",
        "case",
        "2D tile (us)",
        "window (us)",
        "speedup",
        "TB/s",
    )
    for name, (seqlens, has_init, row_stride) in cases.items():
        gen = partial(
            _gen_inputs, seqlens, has_init, args.k_dim, args.v_dim, max(row_stride, dim)
        )
        pool = bencher.build_pool([gen], [INPUT_KEYS])
        fn = partial(
            causal_conv1d_split_qkv_triton_tile_fn,
            bias=None,
            k_dim=args.k_dim,
            v_dim=args.v_dim,
            block_m=BLOCK_M,
            metadata=_prebuilt_metadata(seqlens),
            preserve_input_dtype=True,
        )
        old_us = _time_us(bencher, fn, window=False)
        os.environ["AITER_CONV1D_SPLIT_QKV_WINDOW"] = "1"
        window_config = conv_prefill._window_config(
            pool[0]["x"],
            pool[0]["weight"],
            pool[0]["conv_states"],
            args.k_dim,
            args.v_dim,
            conv_prefill.PAD_SLOT_ID,
        )
        if window_config is None:
            logger.info("%-40s %12.2f %12s", name, old_us, "n/a (2D)")
            continue
        new_us = _time_us(bencher, fn, window=True)
        # x read once and q/k/v written once; weights and conv states are small.
        nbytes = 2 * sum(seqlens) * dim * DTYPE.itemsize
        logger.info(
            "%-40s %12.2f %12.2f %7.1fx %10.2f",
            name,
            old_us,
            new_us,
            old_us / new_us,
            nbytes / new_us / 1e6,
        )
    os.environ.pop("AITER_CONV1D_SPLIT_QKV_WINDOW", None)


def parse_args():
    parser = get_parser("causal_conv1d_split_qkv_triton_tile_fn (prefill)")
    parser.add_argument("--seqlens", type=int, nargs="+", default=None)
    parser.add_argument("--row-stride", type=int, default=6288)
    parser.add_argument("--k-dim", type=int, default=1536)
    parser.add_argument("--v-dim", type=int, default=1536)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
