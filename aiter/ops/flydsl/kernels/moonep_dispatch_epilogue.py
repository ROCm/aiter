# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Local duplicate expansion and padding zero-fill for MoonEP dispatch."""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import T, range_constexpr
from flydsl.expr.typing import Stream

from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)


def make_moonep_dispatch_epilogue_jit(
    *,
    hidden_dim: int,
    num_dispatch_rows: int,
    num_groups: int,
    block_num: int = 128,
    warp_num_per_block: int = 4,
):
    """Build the correctness-first local epilogue launcher.

    The first kernel expands negative-encoded rows from ``duplicate_src``.
    The second zeros the planner-provided padding ranges.  The two row sets are
    disjoint, but separate launches keep the contracts simple while bringing
    up gfx950 correctness; a pipelined implementation can replace this later.
    """

    if hidden_dim <= 0 or hidden_dim % 8 != 0:
        raise ValueError("hidden_dim must be positive and divisible by 8")
    if num_dispatch_rows <= 0 or num_groups <= 0:
        raise ValueError("row and group counts must be positive")
    if block_num <= 0 or warp_num_per_block <= 0:
        raise ValueError("launch geometry must be positive")

    hidden_bytes = hidden_dim * 2
    hidden_i32 = hidden_bytes // 4
    block_threads = warp_num_per_block * 64
    global_warps = block_num * warp_num_per_block
    suffix = (
        f"h{hidden_dim}_nvs{num_dispatch_rows}_g{num_groups}"
        f"_b{block_num}_w{warp_num_per_block}"
    )

    @flyc.kernel(
        name=f"moonep_expand_duplicates_{suffix}",
        known_block_size=[block_threads, 1, 1],
    )
    def expand_duplicates(
        addr_hidden: fx.Int64,  # BF16 [NvS, H]
        addr_duplicate_src: fx.Int64,  # INT32 [NvS], -1 for primary/padding
    ):
        tid = fx.thread_idx.x
        lane = tid & 63
        warp = tid >> 6
        global_warp = fx.block_idx.x * warp_num_per_block + warp
        duplicate_src_rsrc = create_buffer_resource_from_addr(addr_duplicate_src)

        for dst_row in range(global_warp, num_dispatch_rows, global_warps):
            src_row = buffer_load(
                duplicate_src_rsrc, dst_row, vec_width=1, dtype=T.i32()
            )
            is_duplicate = src_row >= 0
            src_addr = addr_hidden + fx.Int64(src_row) * hidden_bytes
            dst_addr = addr_hidden + fx.Int64(dst_row) * hidden_bytes
            src_rsrc = create_buffer_resource_from_addr(src_addr)
            dst_rsrc = create_buffer_resource_from_addr(dst_addr)
            lane_i32_off = lane * 4
            copy_end = is_duplicate.select(hidden_i32, lane_i32_off)
            for i32_off in range(lane_i32_off, copy_end, 64 * 4):
                value = buffer_load(
                    src_rsrc, i32_off, vec_width=4, dtype=T.i32()
                )
                buffer_store(value, dst_rsrc, i32_off)

    @flyc.kernel(
        name=f"moonep_zero_padding_{suffix}",
        known_block_size=[block_threads, 1, 1],
    )
    def zero_padding(
        addr_hidden: fx.Int64,  # BF16 [NvS, H]
        addr_weights: fx.Int64,  # FP32 [NvS]
        addr_zero_fill_ranges: fx.Int64,  # INT32 [G, 2]
    ):
        tid = fx.thread_idx.x
        lane = tid & 63
        warp = tid >> 6
        global_warp = fx.block_idx.x * warp_num_per_block + warp
        ranges_rsrc = create_buffer_resource_from_addr(addr_zero_fill_ranges)
        weights_rsrc = create_buffer_resource_from_addr(addr_weights)

        for group in range(global_warp, num_groups, global_warps):
            start = buffer_load(
                ranges_rsrc, group * 2, vec_width=1, dtype=T.i32()
            )
            count = buffer_load(
                ranges_rsrc, group * 2 + 1, vec_width=1, dtype=T.i32()
            )
            for pad_idx in range(0, count, 1):
                row = start + pad_idx
                row_addr = addr_hidden + fx.Int64(row) * hidden_bytes
                row_rsrc = create_buffer_resource_from_addr(row_addr)
                lane_i32_off = lane * 4
                for i32_off in range(lane_i32_off, hidden_i32, 64 * 4):
                    for scalar in range_constexpr(4):
                        buffer_store(
                            fx.Int32(0), row_rsrc, i32_off + scalar
                        )
                if lane == 0:
                    buffer_store(fx.Int32(0), weights_rsrc, row)

    @flyc.jit
    def launch(
        addr_hidden: fx.Int64,
        addr_weights: fx.Int64,
        addr_duplicate_src: fx.Int64,
        addr_zero_fill_ranges: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        expand_duplicates(addr_hidden, addr_duplicate_src).launch(
            grid=(block_num, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )
        zero_padding(
            addr_hidden, addr_weights, addr_zero_fill_ranges
        ).launch(
            grid=(block_num, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


__all__ = ["make_moonep_dispatch_epilogue_jit"]
