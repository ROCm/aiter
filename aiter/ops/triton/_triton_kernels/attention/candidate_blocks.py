# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kernel bodies for the hierarchical (two-level) indexer candidate selection."""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_NEG_INF = tl.constexpr(float("-inf"))

_candidate_block_scores_repr = make_kernel_repr(
    "_candidate_block_scores_kernel",
    ["BLOCK_SIZE", "BLOCKS_PER_TILE", "CL_IS_TENSOR"],
)

_candidate_mask_scatter_repr = make_kernel_repr(
    "_candidate_mask_scatter_kernel",
    ["BLOCK_SIZE", "TOPK_TILE"],
)

_apply_candidate_mask_repr = make_kernel_repr(
    "_apply_candidate_mask_kernel",
    ["TILE_W"],
)


@triton.jit(repr=_candidate_block_scores_repr)
def _candidate_block_scores_kernel(
    logits_ptr,  # fp32 [T, L]
    cl_ptr,  # int32 [T] (unused when CL_IS_TENSOR is False)
    out_ptr,  # fp32 [T, num_blocks]
    cl_scalar,  # int32, broadcast compress_len when CL_IS_TENSOR is False
    L,
    num_blocks,
    stride_lt: tl.int64,
    stride_ll: tl.int64,
    stride_cl,
    stride_ot: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    PAD_BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_TILE: tl.constexpr,
    CL_IS_TENSOR: tl.constexpr,
):
    """score[t, b] = max over the positions of block b; the partial last
    block of the query is pinned to +inf."""
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)

    blk = pid_b * BLOCKS_PER_TILE + tl.arange(0, BLOCKS_PER_TILE)
    blk_ok = blk < num_blocks

    lane = tl.arange(0, PAD_BLOCK_SIZE)
    pos = blk[:, None].to(tl.int64) * BLOCK_SIZE + lane[None, :]
    load_ok = blk_ok[:, None] & (lane[None, :] < BLOCK_SIZE) & (pos < L)

    vals = tl.load(
        logits_ptr + pid_t.to(tl.int64) * stride_lt + pos * stride_ll,
        mask=load_ok,
        other=_NEG_INF,
    )
    scores = tl.max(vals, axis=1)

    if CL_IS_TENSOR:
        cl = tl.load(cl_ptr + pid_t * stride_cl)
    else:
        cl = cl_scalar
    # floor-div: compress_len == 0 must pin nothing, and tl integer division
    # truncates toward zero instead of flooring.
    last = tl.where(cl > 0, (cl - 1) // BLOCK_SIZE, -1)
    scores = tl.where(blk == last, float("inf"), scores)

    tl.store(out_ptr + pid_t.to(tl.int64) * stride_ot + blk, scores, mask=blk_ok)


@triton.jit(repr=_candidate_mask_scatter_repr)
def _candidate_mask_scatter_kernel(
    idx_ptr,  # int32/int64 [T, K]
    scores_ptr,  # fp32 [T, num_blocks]
    mask_ptr,  # int8 [T, width], pre-zeroed
    K,
    num_blocks,
    width,
    stride_it: tl.int64,
    stride_st: tl.int64,
    stride_mt: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    PAD_BLOCK_SIZE: tl.constexpr,
    TOPK_TILE: tl.constexpr,
):
    """Expand selected block ids to a position mask; picks scoring -inf are dropped."""
    pid_t = tl.program_id(0)
    pid_k = tl.program_id(1)

    k = pid_k * TOPK_TILE + tl.arange(0, TOPK_TILE)
    k_ok = k < K
    blk = tl.load(idx_ptr + pid_t.to(tl.int64) * stride_it + k, mask=k_ok, other=0)
    blk = blk.to(tl.int32)

    valid = k_ok & (blk >= 0) & (blk < num_blocks)
    sc = tl.load(
        scores_ptr + pid_t.to(tl.int64) * stride_st + blk, mask=valid, other=_NEG_INF
    )
    keep = valid & (sc > _NEG_INF)

    lane = tl.arange(0, PAD_BLOCK_SIZE)
    pos = blk[:, None].to(tl.int64) * BLOCK_SIZE + lane[None, :]
    store_ok = keep[:, None] & (lane[None, :] < BLOCK_SIZE) & (pos < width)

    tl.store(
        mask_ptr + pid_t.to(tl.int64) * stride_mt + pos,
        tl.full((TOPK_TILE, PAD_BLOCK_SIZE), 1, tl.int8),
        mask=store_ok,
    )


@triton.jit(repr=_apply_candidate_mask_repr)
def _apply_candidate_mask_kernel(
    logits_ptr,  # fp32 [T, width], modified in place
    mask_ptr,  # int8 [T, width]
    width,
    stride_lt: tl.int64,
    stride_mt: tl.int64,
    TILE_W: tl.constexpr,
):
    """logits[~mask] = -inf, writing only the cleared lanes."""
    pid_t = tl.program_id(0)
    pid_w = tl.program_id(1)

    off = pid_w * TILE_W + tl.arange(0, TILE_W)
    in_row = off < width
    m = tl.load(mask_ptr + pid_t.to(tl.int64) * stride_mt + off, mask=in_row, other=1)
    tl.store(
        logits_ptr + pid_t.to(tl.int64) * stride_lt + off,
        tl.full((TILE_W,), _NEG_INF, tl.float32),
        mask=in_row & (m == 0),
    )
