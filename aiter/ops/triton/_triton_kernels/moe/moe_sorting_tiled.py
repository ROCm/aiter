# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit(
    repr=make_kernel_repr(
        "_histogram", ["T", "K", "E", "NT", "TILE", "BI", "BE", "num_warps"]
    )
)
def _histogram(
    Ids,
    Counts,
    T: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    NT: tl.constexpr,
    TILE: tl.constexpr,
    BI: tl.constexpr,
    BE: tl.constexpr,
):
    tile = tl.program_id(0)
    offsets = tl.arange(0, BI)
    positions = tile * TILE * K + offsets
    ids = tl.load(
        Ids + positions, mask=(offsets < TILE * K) & (positions < T * K), other=E
    )
    histogram = tl.histogram(ids, BE)
    expert = tl.arange(0, BE)
    tl.store(Counts + expert * NT + tile, histogram, mask=expert < E)


@triton.jit(
    repr=make_kernel_repr("_prefix", ["T", "E", "NT", "UNIT", "BE", "BN", "num_warps"])
)
def _prefix(
    Counts,
    TileOffsets,
    ExpertOffsets,
    ExpertCounts,
    NumValid,
    T: tl.constexpr,
    E: tl.constexpr,
    NT: tl.constexpr,
    UNIT: tl.constexpr,
    BE: tl.constexpr,
    BN: tl.constexpr,
):
    expert = tl.arange(0, BE)
    tile = tl.arange(0, BN)
    counts = tl.load(
        Counts + expert[:, None] * NT + tile[None, :],
        mask=(expert[:, None] < E) & (tile[None, :] < NT),
        other=0,
    )
    tile_ends = tl.cumsum(counts, axis=1)
    totals = tl.sum(counts, axis=1)
    padded = tl.cdiv(totals, UNIT) * UNIT
    expert_ends = tl.cumsum(padded, axis=0)
    expert_starts = expert_ends - padded
    starts = tile_ends - counts + expert_starts[:, None]
    tl.store(
        TileOffsets + expert[:, None] * NT + tile[None, :],
        starts,
        mask=(expert[:, None] < E) & (tile[None, :] < NT),
    )
    tl.store(ExpertOffsets + expert, expert_starts, mask=expert < E)
    tl.store(ExpertCounts + expert, totals, mask=expert < E)
    tl.store(NumValid, tl.sum(padded, axis=0))
    tl.store(NumValid + 1, T)


@triton.jit(
    repr=make_kernel_repr(
        "_scatter",
        ["T", "K", "E", "NT", "UNIT", "TILE", "BI", "AUX", "num_warps", "num_stages"],
    )
)
def _scatter(
    Ids,
    Weights,
    TileOffsets,
    ExpertOffsets,
    ExpertCounts,
    SortedIds,
    SortedWeights,
    SortedExperts,
    MIndices,
    Reverse,
    T: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    NT: tl.constexpr,
    UNIT: tl.constexpr,
    TILE: tl.constexpr,
    BI: tl.constexpr,
    AUX: tl.constexpr,
):
    expert = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tl.arange(0, BI)
    positions = tile * TILE * K + offsets
    valid = (offsets < TILE * K) & (positions < T * K)
    ids = tl.load(Ids + positions, mask=valid, other=-1)
    selected = valid & (ids == expert)
    local_end = tl.cumsum(selected.to(tl.int32), axis=0)
    tile_start = tl.load(TileOffsets + expert * NT + tile)
    destination = tile_start + local_end - 1
    token = positions // K
    slot = positions % K
    packed = token | (slot << 24)
    weight = tl.load(Weights + positions, mask=selected, other=0)
    tl.store(SortedIds + destination, packed, mask=selected)
    tl.store(SortedWeights + destination, weight, mask=selected)
    if AUX:
        tl.store(MIndices + destination, token, mask=selected)
        tl.store(Reverse + positions, destination, mask=selected)

    expert_start = tl.load(ExpertOffsets + expert)
    count = tl.load(ExpertCounts + expert)
    padded_count = tl.cdiv(count, UNIT) * UNIT
    # Disjoint metadata writes. No atomics and no device-wide synchronization.
    block_index = tile * BI + offsets
    tl.store(
        SortedExperts + expert_start // UNIT + block_index,
        expert,
        mask=block_index < padded_count // UNIT,
    )
    if tile == 0:
        pad_pos = expert_start + count + offsets
        is_pad = offsets < padded_count - count
        tl.store(SortedIds + pad_pos, T | (K << 24), mask=is_pad)
        tl.store(SortedWeights + pad_pos, 0.0, mask=is_pad)
        if AUX:
            tl.store(MIndices + pad_pos, T, mask=is_pad)
