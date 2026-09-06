# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""16x16x128 FP8 primitives for the gfx950 paged-MQA indexer fast path."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.typing import T

from .. import buffer_ops

Vec = fx.Vector

NUM_HEADS = 64
HEAD_DIM = 128
KV_BLOCK_SIZE = 64
NEXT_N_MAX = 2
MFMA_M = 16
MFMA_N = 16
MFMA_K = 128
M_TILES = NUM_HEADS // MFMA_M
DREG = 4
B_RING = KV_BLOCK_SIZE // MFMA_N
# Per physical page: 4 tiles × (2× dwordx4 K + 1× f32 scale).
PAGE_VMEM_LOADS = B_RING * 3
# Predicated logit stores: 2 query rows × 4 tiles. Hot-loop pages are full, so
# these 8 VM ops land after the next-page loads and can stay in flight.
PAGE_VMEM_STORES = NEXT_N_MAX * B_RING

_NEUTRAL_E8M0 = 0x7F7F7F7F


def wait_vmcnt(n):
    """Explicit s_waitcnt vmcnt(n). Leaves n VM ops outstanding."""
    rocdl.sched_barrier(0)
    rocdl.s_waitcnt(vmcnt=int(n))
    rocdl.sched_barrier(0)


def uceildiv(a, b):
    a, b = fx.Int32(a), fx.Int32(b)
    return fx.Int32((fx.Uint32(a) + fx.Uint32(b) - 1) // fx.Uint32(b))


def imin(a, b):
    a, b = fx.Int32(a), fx.Int32(b)
    return (a <= b).select(a, b)


def guarded_store(pred, store_fn):
    """Predicated side effect (FlyDSL ``@flyc.jit`` branch on a runtime mask)."""

    @flyc.jit
    def _guarded(_pred=pred, _store=store_fn):
        if _pred:
            _store()

    _guarded()


def _concat_i32x4(lo, hi):
    lo, hi = Vec(lo), Vec(hi)
    return Vec.from_elements(
        [Vec(lo)[i].ir_value() for i in range_constexpr(4)]
        + [Vec(hi)[i].ir_value() for i in range_constexpr(4)],
        fx.Int32,
    )


def load_q_pack(q_i32, byte_base, lane_div_16):
    """Load one row's lane-owned contiguous K32 segment as a 256-bit operand."""
    off = (byte_base + lane_div_16 * 32) // fx.Int32(4)
    lo = buffer_ops.buffer_load(q_i32.rsrc, off, vec_width=4, dtype=T.i32)
    hi = buffer_ops.buffer_load(q_i32.rsrc, off + 4, vec_width=4, dtype=T.i32)
    return _concat_i32x4(lo, hi)


def load_preshuffled_k_pack(
    kv_i32,
    physical,
    tile_in_page,
    lane_mod_16,
    lane_div_16,
    *,
    index_dim,
):
    """Load a shuffle_weight(16,16) K column for 16x16x128 scaled MFMA.

    shuffle_weight stores ``[K/32, K-half, token16, byte16]``. ``lane_div_16``
    selects the lane's K32 segment and the two 16-byte halves form i32x8.
    """
    block_i32 = fx.Int32(KV_BLOCK_SIZE * index_dim // 4)
    lane_k_bytes = MFMA_K // (64 // MFMA_N)
    base = (
        physical * block_i32
        + fx.Int32(tile_in_page * HEAD_DIM * MFMA_N // 4)
        + lane_div_16 * fx.Int32(lane_k_bytes * MFMA_N // 4)
        + lane_mod_16 * 4
    )
    lo = buffer_ops.buffer_load(
        kv_i32.rsrc, base, vec_width=4, dtype=T.i32, cache_modifier=2
    )
    hi = buffer_ops.buffer_load(
        kv_i32.rsrc, base + fx.Int32(MFMA_N * 16 // 4),
        vec_width=4, dtype=T.i32, cache_modifier=2,
    )
    return _concat_i32x4(lo, hi)


def load_kv_scale(kv_i32, physical, token_in_page, *, index_dim):
    block_i32 = fx.Int32(KV_BLOCK_SIZE * index_dim // 4)
    off = physical * block_i32 + fx.Int32(KV_BLOCK_SIZE * HEAD_DIM // 4) + token_in_page
    return fx.Float32(
        buffer_ops.buffer_load(
            kv_i32.rsrc, off, vec_width=1, dtype=T.f32, cache_modifier=2
        )
    )


def mfma_scores(a_tiles, b_pack):
    """Return four 4-f32 score fragments, one per 16-head M tile."""
    result_type = Vec.make_type(DREG, fx.Float32)
    neutral = arith.constant(_NEUTRAL_E8M0, type=T.i32)
    scores = []
    for mi in range_constexpr(M_TILES):
        acc = Vec.filled(DREG, 0.0, fx.Float32)
        acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            result_type,
            [a_tiles[mi], b_pack, acc, 0, 0, 0, neutral, 0, neutral],
        )
        scores.append(acc)
    return scores


def reduce_scores(scores, weights, kv_scale):
    """ReLU, weighted H reduction, positive KV scale, then wave reduction."""
    zero = fx.Float32(0.0)
    total = zero
    for mi in range_constexpr(M_TILES):
        frag = Vec(scores[mi])
        for ii in range_constexpr(DREG):
            total = total + fx.Float32(frag[ii]).maximumf(zero) * weights[mi][ii]
    total = total * kv_scale
    total = total + total.shuffle_xor(16, 64)
    total = total + total.shuffle_xor(32, 64)
    return total


def schedule_mfma_valu_pairs():
    """Pair each 32-cycle MFMA with the prior tile's 12-op VALU fragment."""
    for _ in range_constexpr(M_TILES):
        rocdl.sched_group_barrier(0x008, 1, 0)
        rocdl.sched_group_barrier(0x002, DREG * 3, 0)
