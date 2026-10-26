# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-token gather from a small table of row vectors.

Diffusion transformer blocks carry one index per token into a table with one
row per (modality, timestep) combination. ``fused_rmsnorm_indexed_adaln`` reads
its shift/scale rows this way when a branch opens, and
``fused_indexed_gated_residual`` its gate rows when the branch closes.
"""

import triton
import triton.language as tl


@triton.jit
def _load_block_indices(idx_ptr, row_block, rows, row_mask, BLOCK_M: tl.constexpr):
    """Table indices of one row block, its first index, and whether they agree."""
    # The block's first row always exists, so its index is a safe fill for the
    # tail lanes -- they must not make a uniform block look mixed.
    first = tl.load(idx_ptr + row_block * BLOCK_M)
    idx = tl.load(idx_ptr + rows, mask=row_mask, other=first)
    return idx, first, tl.min(idx) == tl.max(idx)


@triton.jit
def _gather_indexed_rows(
    table_ptr,
    stride_row,
    idx,
    first,
    uniform,
    cols,
    row_mask,
    col_mask,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``table[idx[m], cols]`` as an fp32 [BLOCK_M, BLOCK_N] tile."""
    if uniform:
        # A packed sequence is laid out in runs of one modality, so a block
        # usually shares one index: load that row once and broadcast it.
        row = tl.load(
            table_ptr + first.to(tl.int64) * stride_row + cols,
            mask=col_mask,
            other=0.0,
        )
        vals = tl.broadcast_to(row.to(tl.float32)[None, :], (BLOCK_M, BLOCK_N))
    else:
        vals = tl.load(
            table_ptr + idx.to(tl.int64)[:, None] * stride_row + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
    return vals
