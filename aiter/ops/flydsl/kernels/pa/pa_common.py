# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared helpers for paged-attention (PA) decode kernels.

Extracted verbatim from pa_decode_fp8 / pa_decode_swa / pa_metadata, which
carried byte-identical copies.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.pa.utils import copy_load

# PA Q-tiling constants (identical across all PA decode kernels).
MFMA_N = 16
Q_ELEMS_PER_LANE = 8
Q_CHUNKS_PER_LANE = Q_ELEMS_PER_LANE // 4


def _compute_block_base_dw_i64(phys_block, block_stride, head_offset):
    phys_block_i64 = fx.Int64(phys_block)
    block_stride_i64 = fx.Int64(block_stride)
    head_offset_i64 = fx.Int64(head_offset)
    return (phys_block_i64 * block_stride_i64 + head_offset_i64) >> fx.Int64(2)


@flyc.jit
def _prefetch_q_chunks(
    q_rsrc,
    q_base,
    lane16id,
    *,
    query_load_is_bf16,
    q_lanes_per_head,
):
    # bf16/f16 + in-kernel query_scale path.  Each lane owns 8 Q elements,
    # loaded as 2 × vec_width=4 buffer loads (4 bf16/f16 elems per load = 8 B,
    # element offset += 4 per iter).  After FP8 packing each load produces
    # one i32 word, so the per-lane store is `vec<2, i32>` = 8 B = 1 i64.
    q_load_lane = lane16id
    if const_expr(q_lanes_per_head < MFMA_N):
        q_load_lane = arith.select(lane16id < fx.Int32(q_lanes_per_head), lane16id, fx.Int32(0))
    q_elem = q_base + q_load_lane * fx.Int32(Q_ELEMS_PER_LANE)
    q_chunks = []
    for qwi in range_constexpr(Q_CHUNKS_PER_LANE):
        q_chunks.append(
            buffer_ops.buffer_load(
                q_rsrc,
                q_elem + fx.Int32(qwi * 4),
                vec_width=4,
                dtype=fx.BFloat16 if query_load_is_bf16 else fx.Float16,
            )
        )
    return q_chunks


@flyc.jit
def _prefetch_q_chunks_tile(
    q_tiles,
    q_copy_atom,
    q_reg,
    q_base,
    lane16id,
    *,
    q_lanes_per_head,
):
    q_load_lane = lane16id
    if const_expr(q_lanes_per_head < MFMA_N):
        q_load_lane = (lane16id < fx.Int32(q_lanes_per_head)).select(lane16id, fx.Int32(0))
    q_elem = q_base + q_load_lane * fx.Int32(Q_ELEMS_PER_LANE)
    q_tile = q_elem // fx.Int32(4)
    q_chunks = []
    for qwi in range_constexpr(Q_CHUNKS_PER_LANE):
        q_chunks.append(copy_load(q_tiles, q_tile + fx.Int32(qwi), q_copy_atom, q_reg))
    return q_chunks
