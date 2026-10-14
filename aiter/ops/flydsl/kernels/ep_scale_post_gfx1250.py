# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Whole-row e8m0 scale send that follows a scale-post gemm2 scatter.

The gemm2 epilogue leaves each tile's 8-byte scale slices in a local tile-major
plane ([M tile][N tile][row] x 8 B) and each row's scatter index in a side
array. This kernel writes every valid row's N/32 scale bytes to the slot's
scale plane in one run, instead of the N/tile_n partial writes a row would take
from the gemm2 tiles themselves.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr.typing import Int32, Int64, T
from flydsl.expr.typing import Vector as Vec

from .buffer_ops import buffer_load, create_buffer_resource_from_addr

BLOCK = 256
# A row is scale_row/16 lanes of 16 bytes; rounding the lane count up to a
# power of two keeps rows from straddling a wave.
LANES_PER_ROW = 16


@functools.cache
def make_ep_scale_post(
    *,
    scale_row,
    tile_m,
    n_experts,
    scale_plane_off,
    slot_stride,
    world_size,
):
    chunks = scale_row // 16
    if scale_row % 16 or chunks > LANES_PER_ROW:
        raise ValueError(f"scale_row={scale_row} needs a 16-byte multiple <= 256")
    tile_bytes = tile_m * 8

    @flyc.kernel(
        name=f"ep_scale_post_r{scale_row}m{tile_m}",
        known_block_size=[BLOCK, 1, 1],
    )
    def ep_scale_post(
        arena: Int64,
        addr_stage: Int64,
        addr_idx: Int64,
        addr_tile_map: Int64,
        n_rows: Int32,
    ):
        gid = fx.Int32(fx.block_idx.x) * fx.Int32(BLOCK) + fx.Int32(fx.thread_idx.x)
        row = gid // fx.Int32(LANES_PER_ROW)
        chunk = gid % fx.Int32(LANES_PER_ROW)
        in_range = (chunk < fx.Int32(chunks)) & (row < n_rows)
        # Every load is independent of the others and issues up front; lanes
        # past the end read row 0 / chunk 0 and are dropped below.
        safe_row = in_range.select(row, fx.Int32(0))
        safe_chunk = in_range.select(chunk, fx.Int32(0))
        rows_end = fx.Int32(
            buffer_load(
                create_buffer_resource_from_addr(addr_tile_map),
                fx.Int32(n_experts - 1),
                vec_width=1,
                dtype=T.i32,
            )
        )
        idx = fx.Int32(
            buffer_load(
                create_buffer_resource_from_addr(addr_idx),
                safe_row,
                vec_width=1,
                dtype=T.i32,
            )
        )
        # A 16-byte chunk is the 8-byte slices of two consecutive N tiles.
        m_tile = safe_row // fx.Int32(tile_m)
        off0 = (
            m_tile * fx.Int32(scale_row * tile_m)
            + safe_chunk * fx.Int32(2 * tile_bytes)
            + (safe_row - m_tile * fx.Int32(tile_m)) * fx.Int32(8)
        )
        stage = create_buffer_resource_from_addr(addr_stage)
        d0 = Vec(buffer_load(stage, off0 // 4, vec_width=2, dtype=T.i32))
        d1 = Vec(
            buffer_load(
                stage, (off0 + fx.Int32(tile_bytes)) // 4, vec_width=2, dtype=T.i32
            )
        )
        win = cco.Window(arena)
        lsa0 = fx.Int64(win.lsa_ptr(fx.Int32(0), 0))
        lsa1 = fx.Int64(win.lsa_ptr(fx.Int32(1), 0))
        oob = fx.Int32((lsa1 - lsa0) // fx.Int64(slot_stride)) * fx.Int32(world_size)
        # Padding M tiles never ran, so their index entries are stale.
        keep = (
            in_range
            & (m_tile * fx.Int32(tile_m) < rows_end)
            & (idx >= fx.Int32(0))
            & (idx < oob)
        )
        if keep:
            dst = fx.inttoptr(
                fx.PointerType.get(
                    elem_ty=fx.Int32.ir_type,
                    address_space=fx.AddressSpace.Global,
                    alignment=16,
                ),
                lsa0 + fx.Int64(scale_plane_off),
            )
            fx.ptr_store(
                Vec.from_elements(
                    [
                        fx.Int32(d0[0]),
                        fx.Int32(d0[1]),
                        fx.Int32(d1[0]),
                        fx.Int32(d1[1]),
                    ],
                    fx.Int32,
                ),
                dst
                + (
                    fx.Int64(idx) * fx.Int64(slot_stride // 4)
                    + fx.Int64(chunk * fx.Int32(4))
                ),
            )

    @flyc.jit
    def run(
        arena: Int64,
        addr_stage: Int64,
        addr_idx: Int64,
        addr_tile_map: Int64,
        n_rows: Int32,
        n_blocks: Int32,
        stream=fx.Stream(None),  # noqa: B008
    ):
        ep_scale_post(arena, addr_stage, addr_idx, addr_tile_map, n_rows).launch(
            grid=(n_blocks, 1, 1),
            block=[BLOCK, 1, 1],
            stream=stream,
        )

    return run
