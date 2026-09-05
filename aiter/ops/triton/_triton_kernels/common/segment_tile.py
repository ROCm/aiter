# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared device helpers for segmented (ragged-batch) tile dispatch.

Kernels that launch a flat grid over a ragged row dimension — e.g. segment-M
quantization, grouped GEMM, MoE dispatch — need to map a flat program id back
to its owning segment and the corresponding row range. This module provides a
reusable JIT device function for that lookup.
"""

import triton
import triton.language as tl


@triton.jit
def _find_segment_tile_range(
    pid, batch_size, seg_indptr, scales_seg_indptr_ptr, BLOCK_SIZE: tl.constexpr
):
    """Map a flat M-axis tile id to its segment and row range.

    Given a flat program id ``pid`` (the index of a tile along the M
    dimension of a ragged-batch tensor), returns the contiguous row interval
    ``[m_range_start, m_range_end)`` that the tile should process and the
    index ``bid`` of the segment (batch element) it belongs to.

    Args:
        pid:                   Flat M-axis tile index (``tl.program_id(0)``).
        batch_size:            Number of segments.
        seg_indptr:            ``[batch_size + 1]`` cumulative row offsets.
        scales_seg_indptr_ptr: ``[batch_size + 1]`` cumulative tile counts
                               (``scales_seg_indptr[i+1] - scales_seg_indptr[i]``
                               = number of BLOCK_SIZE-row tiles in segment i).
        BLOCK_SIZE:            Tile height (constexpr).

    Returns:
        ``(m_range_start, m_range_end, bid)``
    """
    # Binary search: find bid s.t. scales_seg_indptr[bid] <= pid < scales_seg_indptr[bid+1].
    # O(log batch_size) vs a linear scan.
    lo = 0
    hi = batch_size
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if tl.load(scales_seg_indptr_ptr + mid) <= pid:
            lo = mid
        else:
            hi = mid
    bid = lo
    idx_start = tl.load(scales_seg_indptr_ptr + bid)

    m_range_start = tl.load(seg_indptr + bid) + (pid - idx_start) * BLOCK_SIZE
    m_range_end = min(tl.load(seg_indptr + bid + 1), m_range_start + BLOCK_SIZE)
    return m_range_start, m_range_end, bid
