# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark of causal_conv1d_split_qkv_triton_tile_fn on x with unit channel stride.

Compares the sliding-window path with the 2D-tiled kernel (AITER_CONV1D_SPLIT_QKV_WINDOW=0), called as vLLM calls
it (block_m=8, prebuilt schedule). The default cases are Kimi-K3 TP8 prefill steps (q, k, v = 1536 channels each).
Times are CUDA-graph replays without a cache flush, so x can stay in the last-level cache, as after the projection
GEMM in the model.

Usage examples
--------------
python bench_causal_conv1d_prefill.py
python bench_causal_conv1d_prefill.py --seqlens 1112 1112 1112 1112 --row-stride 4608
python bench_causal_conv1d_prefill.py --k-dim 3072 --v-dim 3072 --row-stride 12288 --seqlens 3100
"""

import itertools
import os
import types

import torch
import triton

from aiter import logger
from aiter.ops.prefill_batch_metadata import build_causal_conv_prefill_metadata
from aiter.ops.triton.gated_delta_net import causal_conv1d_split_qkv_triton_tile_fn
from op_tests.op_benchmarks.triton.utils.argparse import get_parser

DEVICE = "cuda"
DTYPE = torch.bfloat16
WIDTH = 4
BLOCK_M = 8

DEFAULT_CASES = {
    "one continuing chunk, row stride 6288": ([7680], [True], 6288),
    "continuing + new chunk, gathered": ([5010, 2680], [True, False], 4608),
    "7 new prompts, gathered": ([1112] * 7, [False] * 7, 4608),
    "60 new prompts of 128, gathered": ([128] * 60, [False] * 60, 4608),
    "one prompt of 1024, row stride 6288": ([1024], [False], 6288),
}


def _make_inputs(seqlens, has_init, k_dim, v_dim, row_stride):
    dim = 2 * k_dim + v_dim
    T = sum(seqlens)
    n = len(seqlens)
    torch.manual_seed(0)
    proj = torch.randn((T, row_stride), dtype=DTYPE, device=DEVICE)
    query_start_loc = torch.tensor(
        [0, *itertools.accumulate(seqlens)], dtype=torch.int32, device=DEVICE
    )
    # vLLM passes a plain object with a prebuilt nums_dict schedule (usable in graph capture).
    grid = build_causal_conv_prefill_metadata(
        seqlens, query_start_loc=query_start_loc, block_sizes=(BLOCK_M,)
    ).get_chunk_grid(BLOCK_M)
    metadata = types.SimpleNamespace(
        nums_dict={
            BLOCK_M: {
                "tot": grid.total_chunks,
                "batch_ptr": grid.sequence_ids,
                "token_chunk_offset_ptr": grid.chunk_ids,
            }
        }
    )
    return {
        "x": proj[:, :dim].t(),
        "weight": torch.randn((dim, WIDTH), dtype=torch.float32, device=DEVICE) * 0.5,
        "bias": None,
        "conv_states": torch.randn(
            (2 * n, WIDTH - 1 + 4, dim), dtype=DTYPE, device=DEVICE
        ).transpose(1, 2),
        "query_start_loc": query_start_loc,
        "k_dim": k_dim,
        "v_dim": v_dim,
        "cache_indices": torch.arange(n, dtype=torch.int32, device=DEVICE),
        "has_initial_state": torch.tensor(has_init, dtype=torch.bool, device=DEVICE),
        "block_m": BLOCK_M,
        "metadata": metadata,
        "preserve_input_dtype": True,
    }


def _time_us(inputs, window):
    os.environ["AITER_CONV1D_SPLIT_QKV_WINDOW"] = "1" if window else "0"
    causal_conv1d_split_qkv_triton_tile_fn(**inputs)  # compile outside the capture
    ms = triton.testing.do_bench_cudagraph(
        lambda: causal_conv1d_split_qkv_triton_tile_fn(**inputs), rep=100
    )
    return ms * 1e3


def run(args):
    if args.seqlens:
        cases = {"custom": (args.seqlens, [True] * len(args.seqlens), args.row_stride)}
    else:
        cases = DEFAULT_CASES
    dim = 2 * args.k_dim + args.v_dim
    logger.info(
        "%-40s %12s %12s %8s %10s",
        "case",
        "2D tile (us)",
        "window (us)",
        "speedup",
        "TB/s",
    )
    for name, (seqlens, has_init, row_stride) in cases.items():
        inputs = _make_inputs(
            seqlens, has_init, args.k_dim, args.v_dim, max(row_stride, dim)
        )
        old_us = _time_us(inputs, window=False)
        new_us = _time_us(inputs, window=True)
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
