# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_topk_hist_repr = make_kernel_repr("moe_topk_align_hist", ["BLOCK"])
_topk_prefix_repr = make_kernel_repr(
    "moe_topk_align_prefix", ["BLOCK_M", "NUM_EXPERTS"]
)
_topk_scatter_repr = make_kernel_repr(
    "moe_topk_align_scatter", ["BLOCK_M", "BLOCK"]
)


@triton.jit(repr=_topk_hist_repr)
def _topk_hist_kernel(
    TopkIds,
    Hist,
    route_count: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < route_count
    expert = tl.load(TopkIds + offs, mask=mask, other=0)
    tl.atomic_add(Hist + expert, 1, mask=mask, sem="relaxed")


@triton.jit(repr=_topk_prefix_repr)
def _topk_prefix_kernel(
    Hist,
    RawOffsets,
    TileOffsets,
    Cursor,
    TotalBlocks,
    BlockPidMap,
    BLOCK_M: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_E)
    mask = offs < NUM_EXPERTS
    hist = tl.load(Hist + offs, mask=mask, other=0)
    raw_cumsum = tl.cumsum(hist, 0)
    tile_count = tl.cdiv(hist, BLOCK_M)
    tile_cumsum = tl.cumsum(tile_count, 0)
    raw_start = raw_cumsum - hist
    tile_start = tile_cumsum - tile_count

    tl.store(RawOffsets + offs, raw_start, mask=mask)
    tl.store(RawOffsets + offs + 1, raw_cumsum, mask=mask)
    tl.store(TileOffsets + offs, tile_start, mask=mask)
    tl.store(TileOffsets + offs + 1, tile_cumsum, mask=mask)
    tl.store(Cursor + offs, raw_start, mask=mask)
    tl.store(TotalBlocks, tl.max(tile_cumsum, axis=0))

    block_offs = tl.arange(0, BLOCK_B)
    tl.store(BlockPidMap + block_offs, -1, mask=block_offs < MAX_BLOCKS)


@triton.jit
def _topk_blockmap_kernel(
    TileOffsets,
    BlockPidMap,
    NUM_EXPERTS: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    expert = tl.program_id(0)
    offs = tl.arange(0, BLOCK_T)
    tile_start = tl.load(TileOffsets + expert)
    tile_end = tl.load(TileOffsets + expert + 1)
    tile = tile_start + offs
    mask = (expert < NUM_EXPERTS) & (tile < tile_end) & (tile < MAX_BLOCKS)
    block_id = offs.to(tl.int32)
    value = (block_id << 16) | expert
    tl.store(BlockPidMap + tile, value, mask=mask)


@triton.jit(repr=_topk_scatter_repr)
def _topk_scatter_kernel(
    TopkIds,
    TopkWeights,
    Cursor,
    RawOffsets,
    TileOffsets,
    BlockPidMap,
    SortedRoutes,
    SortedWeights,
    Inverse,
    route_count: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < route_count
    expert = tl.load(TopkIds + offs, mask=mask, other=0)
    pos = tl.atomic_add(Cursor + expert, 1, mask=mask, sem="relaxed")
    expert_start = tl.load(RawOffsets + expert, mask=mask, other=0)
    local_pos = pos - expert_start
    tile_local = local_pos // BLOCK_M
    tile_pos = tl.load(TileOffsets + expert, mask=mask, other=0) + tile_local
    starts_tile = (local_pos % BLOCK_M) == 0
    block_value = (tile_local.to(tl.int32) << 16) | expert
    weight = tl.load(TopkWeights + offs, mask=mask, other=0.0)
    tl.store(BlockPidMap + tile_pos, block_value, mask=mask & starts_tile)
    tl.store(SortedRoutes + pos, offs, mask=mask)
    tl.store(SortedWeights + pos, weight, mask=mask)
    tl.store(Inverse + offs, pos, mask=mask)
