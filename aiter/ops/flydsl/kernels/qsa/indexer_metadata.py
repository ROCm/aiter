# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-query lengths for prepared-Q selection over request page tables."""

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel(name="qsa_indexer_row_metadata", known_block_size=[256, 1, 1])
def _kernel(
    REQUESTS: fx.Tensor,
    POSITIONS: fx.Tensor,
    CONTEXT: fx.Tensor,
    LENGTHS: fx.Tensor,
    SEQUENCES: fx.Tensor,
    STATS: fx.Tensor,
    rows: fx.Int32,
    requests: fx.Int32,
    width: fx.Int32,
    LONG: fx.Constexpr[bool],
):
    tid = fx.thread_idx.x
    row = fx.block_idx.x if LONG else fx.block_idx.x * 256 + tid
    if row < rows:
        if fx.const_expr(LONG):
            STATS[row * 512 + tid * 2] = fx.Int32(-1)
            STATS[row * 512 + tid * 2 + 1] = fx.Int32(-1)
        else:
            STATS[row * 2] = fx.Int32(-1)
            STATS[row * 2 + 1] = fx.Int32(-1)
        if not LONG or tid == 0:
            request = fx.Int32(REQUESTS[row])
            sequence = fx.Int32(0)
            if (request >= 0) & (request < requests):
                sequence = fx.Int32(CONTEXT[request])
            visible = fx.min(
                fx.max(fx.Int32(POSITIONS[row]) + 1, fx.Int32(0)), sequence
            )
            LENGTHS[row] = fx.min(fx.max(visible // 4, fx.Int32(0)), width)
            SEQUENCES[row] = sequence


@flyc.jit
def _launch(
    REQUESTS: fx.Tensor,
    POSITIONS: fx.Tensor,
    CONTEXT: fx.Tensor,
    LENGTHS: fx.Tensor,
    SEQUENCES: fx.Tensor,
    STATS: fx.Tensor,
    rows: fx.Int32,
    requests: fx.Int32,
    width: fx.Int32,
    ctas: fx.Int32,
    LONG: fx.Constexpr[bool],
    stream: fx.Stream,
):
    if ctas > 0:
        _kernel(
            REQUESTS,
            POSITIONS,
            CONTEXT,
            LENGTHS,
            SEQUENCES,
            STATS,
            rows,
            requests,
            width,
            LONG,
        ).launch(grid=(ctas, 1, 1), block=(256, 1, 1), stream=stream)
