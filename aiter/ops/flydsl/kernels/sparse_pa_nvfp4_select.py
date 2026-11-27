# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# The windowed threshold-narrowing selection of ``_select_window`` is ported from
# amd-kernels (inferact/amd-kernels, SPDX-License-Identifier: Apache-2.0, Copyright
# contributors to the amd-kernels project):
# indexer/indexer_nvfp4_d128_p128_top16/topk/top16_indices_p128_plain_wave4.py
# (``row_top16_w4_one``), as run in the prologue of
# attention/nvfp4_sparse_decode_gqa16_d128_p128_top16/kernel_select.py.

"""Top-16 block selection inside the decode-attention workgroup (FlyDSL, gfx950).

``row_top16_prologue`` picks a decode row's 16 highest-scoring index blocks from its fp32
score row in the first instructions of ``sparse_pa_nvfp4_decode``'s workgroup, so a
scorer -> attention chain needs no selector launch in between.

Order and ties are the canonical selector contract: descending score, ties to the larger
block id, NaN above +inf, the ``init_blocks`` leading and ``local_blocks`` trailing live
blocks forced in (the trailing ones outrank the leading), ids -1 past the row's live count.
This fused path implements that larger-id tie-break exactly. ``topk_per_row_small_k`` (the
aiter selector the chain otherwise runs) matches it on untied rows but deviates at its chunk
cut: when more chunks tie at the k-th place than there are places it drops chunks by lane id,
which is not monotone in the column, so the two can return different, equally-scored blocks
on the rows that tie at the 16th place (~0.1% of rows at 256 rows in the PR's measurement).
Both orders are deterministic and independent of the row's place in the batch.

Algorithm for a row of at most ``SELECT_BLOCKS`` (= 4 waves x 7 strips x 64) live blocks,
all of it in registers and wave ballots, four barriers:

  1. every lane keeps its 7 strips' order keys (the float's bits, negatives flipped, so
     signed-int order is float order) and the maximum m of them; the four waves' maxima
     are combined per lane through LDS;
  2. the 16th largest lane maximum bounds the answer from below. Its top 16 key bits are
     found by a bitwise ballot search (``count(m >= t) >= 16``); a guess of the top six
     bits from lane 31's maximum skips ten of the sixteen steps when it brackets;
  3. every key >= the bound is a candidate (Gaussian rows: ~18; more than 64 sends the row
     to the exact path below). Candidates are compacted into LDS with ballot prefix sums;
  4. a candidate's rank is the number of candidates ordered above it by (key, block id),
     split over four waves x 16 lanes and summed with permlane swaps; rank < 16 stores its
     id to LDS slot ``rank``, and 16 lanes write the ids to ``topk_idx`` in one store.

Longer rows run the same narrowing per window of 1792 blocks, merged with the running
top 16. A row with fewer than 16 live blocks or more than 64 candidates at the bound takes
the exact path: every live block counts the blocks above it, reading the row from memory,
O(live) per block. That is slow (hundreds of us at thousands of live blocks) and only
reached by degenerate rows (heavy ties right at the 16th score, e.g. a constant row) or
short contexts, where it is cheap.
"""

import struct

import flydsl.expr as fx
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.sparse_pa_nvfp4_common import _permlane_swap
from aiter.ops.flydsl.kernels.tensor_shim import (
    buf_copy_load,
    buf_copy_store,
    ptr_buf_tensor,
)

TOPK = 16
SELECT_STRIPS = 7  # strips of 64 blocks per wave, one window
SELECT_BLOCKS = 4 * SELECT_STRIPS * 64  # blocks of one window (1792 = 229K tokens)
CAND_DWORDS = 2 * (
    64 + 8 + 64
)  # LDS: candidate (block, key) pairs: 64 + rank padding + 64 trash slots
TMP_DWORDS = 260  # LDS: lane maxima of the four waves, per-wave counts, then the 16 ids
SEL_DWORDS = 4 * TOPK  # LDS: the running top 16 of the window loop, (id, key) quads
BOUND_BITS = 16  # key bits of the 16th lane maximum that bound the candidates
INT_MIN = -(1 << 31)
INT_MAX = (1 << 31) - 1
KEY_NEG_INF = (0xFF800000 ^ 0x7FFFFFFF) - (1 << 32)  # signed key of -inf


def _fkey(value: float) -> int:
    bits = struct.unpack("<i", struct.pack("<f", value))[0]
    return bits ^ ((bits >> 31) & 0x7FFFFFFF)


# the forced-block pins of topk_per_row_small_k: above every real score, init above local
INIT_PIN_KEY = _fkey(1e30)
LOCAL_PIN_KEY = _fkey(1e29)


def _s32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v


def _ballot(c):
    return fx.Int64(fx.rocdl.ballot(T.i64, c))


def _popc(m):
    """Set bits of a wave mask as i32 (two 32-bit counts keep the compare scalar)."""
    lo = fx.math.ctpop(m.to(fx.Int32)).to(fx.Int32)
    hi = fx.math.ctpop((m >> 32).to(fx.Int32)).to(fx.Int32)
    return fx.Int32(lo) + fx.Int32(hi)


def _below(mask, base):
    """The set bits of ``mask`` below this lane, plus ``base`` (v_mbcnt)."""
    lo = mask.to(fx.Int32)
    hi = (mask >> 32).to(fx.Int32)
    r = _rocdl_ops.mbcnt_lo(T.i32, lo.ir_value(), fx.Int32(base).ir_value())
    return fx.Int32(_rocdl_ops.mbcnt_hi(T.i32, hi.ir_value(), r))


def _imax(a, b):
    return (a > b).select(a, b)


def _one(c):
    return c.select(fx.Int32(1), fx.Int32(0))


def _comp(k, i):
    """(key, id) as one i64, the key in the high word: the signed compare is the selection order."""
    return (k.to(fx.Int64) << 32) | (i.to(fx.Int64) & 0xFFFFFFFF)


def _xlane4_sum(x):
    """Sum over lanes L, L ^ 16, L ^ 32, L ^ 48, on every lane."""
    a, b = _permlane_swap(fx.rocdl.permlane32_swap, x, x)
    s = a + b
    a, b = _permlane_swap(fx.rocdl.permlane16_swap, s, s)
    return a + b


def _order_key(bits):
    """Order key of fp32 bits: negatives' magnitude bits flipped (signed-int order is the
    float order, -0.0 one step below 0.0 as in ``topk_per_row_small_k``), NaN on top."""
    key = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    is_nan = (bits & 0x7FFFFFFF) > 0x7F800000
    return is_nan.select(fx.Int32(INT_MAX), key)


def _pinned_key(bits, col, init_blocks, local_start, pins):
    key = _order_key(bits)
    if const_expr(pins):
        key = (col < init_blocks).select(fx.Int32(INIT_PIN_KEY), key)
        key = (col >= local_start).select(fx.Int32(LOCAL_PIN_KEY), key)
    return key


def first_window(tx, score_row, width):
    """The first window's raw score words, issued before the row's length is loaded (it
    only masks them). ``score_row``: the row's i64 address; ``width``: readable columns.
    """
    lane, wave = tx % 64, tx // 64
    span = (width < SELECT_BLOCKS).select(width, fx.Int32(SELECT_BLOCKS))
    srs = ptr_buf_tensor(
        score_row, fx.Int32, unit_elems=1, num_records_bytes=span.to(fx.Int64) * 4
    )
    return [
        fx.Int32(buf_copy_load(srs, (j * 4 + wave) * 64 + lane))
        for j in range_constexpr(SELECT_STRIPS)
    ]


@ASTRewriter.transform
def _select_window(tx, nblk, init_blocks, local_blocks, pre, cand, tmp, pins):
    """The windowed path over the first window ``pre`` (``first_window``) of a row of
    ``nblk`` <= SELECT_BLOCKS live blocks. Returns ``bad`` (1: the exact path must run); when
    0 the 16 ids are in ``tmp[0..15]`` behind the last barrier."""
    lane, wave = tx % 64, tx // 64

    def sync():
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()

    def view(ptr, n_dw, w):
        return fx.logical_divide(
            fx.make_view(fx.recast_iter(fx.Int32, ptr), fx.make_layout(n_dw, 1)),
            fx.make_layout(w, 1),
        )

    pairs, quads = view(cand, CAND_DWORDS, 2), view(cand, CAND_DWORDS, 4)
    atom64 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)
    atom128 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

    def pair_put(i, blk, key):
        r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
        r.store(Vec.from_elements([blk, key], fx.Int32))
        fx.copy(atom64, r, fx.slice(pairs, (None, i)))

    def quad_get(i):
        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        fx.copy(atom128, fx.slice(quads, (None, i)), r)
        v = Vec(r.load())
        return [fx.Int32(v[e]) for e in range_constexpr(4)]

    local_start = (nblk > local_blocks).select(nblk - local_blocks, fx.Int32(0))
    # candidate slots past the live count compare below every candidate
    if tx < 64:
        pair_put(tx, fx.Int32(0), fx.Int32(INT_MIN))
    keys = []
    for j in range_constexpr(SELECT_STRIPS):
        col = (j * 4 + wave) * 64 + lane
        keys.append(
            (col < nblk).select(
                _pinned_key(pre[j], col, init_blocks, local_start, pins),
                fx.Int32(INT_MIN),
            )
        )
    m = keys[0]
    for j in range_constexpr(1, SELECT_STRIPS):
        m = _imax(m, keys[j])
    tmp[tx] = m
    sync()
    m = _imax(
        _imax(fx.Int32(tmp[lane]), fx.Int32(tmp[64 + lane])),
        _imax(fx.Int32(tmp[128 + lane]), fx.Int32(tmp[192 + lane])),
    )
    # both endpoint counts prove this six-bit interval holds the 16th lane maximum; a failed
    # guess runs the whole 16-bit search
    seed = fx.Int32(fx.rocdl.readlane(T.i32, m.ir_value(), fx.Int32(31).ir_value()))
    seed = (seed ^ INT_MIN) & _s32(0xFFFFFFFF << 26 & 0xFFFFFFFF)
    upper = seed + _s32(1 << 26)
    lo_count = _popc(_ballot(m >= (seed ^ INT_MIN)))
    hi_count = _popc(_ballot(m >= (upper ^ INT_MIN)))
    valid = (lo_count >= TOPK) & (hi_count < TOPK) & (upper != 0)
    p = fx.Int32(0)
    if valid:
        p = seed
        for d in range_constexpr(6, BOUND_BITS):
            t = p | _s32(1 << (31 - d))
            count = _popc(_ballot(m >= (t ^ INT_MIN)))
            p = (count >= TOPK).select(t, p)
    else:
        for d in range_constexpr(BOUND_BITS):
            t = p | _s32(1 << (31 - d))
            count = _popc(_ballot(m >= (t ^ INT_MIN)))
            p = (count >= TOPK).select(t, p)
    ta = _imax(p ^ INT_MIN, fx.Int32(INT_MIN + 1))
    oks = [_one(k >= ta) for k in keys]
    cnt = oks[0]
    for j in range_constexpr(1, SELECT_STRIPS):
        cnt = cnt + oks[j]
    prefix, wtotal = fx.Int32(0), fx.Int32(0)
    for b in range_constexpr(SELECT_STRIPS.bit_length()):
        mask = _ballot((cnt & (1 << b)) != 0)
        prefix = prefix + (_below(mask, 0) << b)
        wtotal = wtotal + (_popc(mask) << b)
    if lane == 0:
        tmp[256 + wave] = wtotal
    sync()
    offset, total = prefix, fx.Int32(0)
    for w in range_constexpr(4):
        wt = fx.Int32(tmp[256 + w])
        offset = offset + (wave > w).select(wt, fx.Int32(0))
        total = total + wt
    # every wave sees the same ta and total
    bad = ((total > 64) | (ta <= KEY_NEG_INF)).select(fx.Int32(1), fx.Int32(0))
    if bad == 0:
        for j in range_constexpr(SELECT_STRIPS):
            if oks[j] != 0:
                pair_put(offset, (j * 4 + wave) * 64 + lane, keys[j])
            offset = offset + oks[j]
        sync()
        # candidate c's rank is split over lanes c % 16 + 16 q of wave c // 16 (16 candidates each)
        c = wave * 16 + lane % 16
        q = lane // 16
        mi, mk = fx.Int32(cand[2 * c]), fx.Int32(cand[2 * c + 1])
        mw = _comp(mk, mi)
        rank = fx.Int32(0)
        if total <= 32:
            es = [quad_get(q * 4 + u) for u in range_constexpr(4)]
            fx.rocdl.sched_barrier(0)
            for e in es:
                rank = rank + _one(_comp(e[1], e[0]) > mw)
                rank = rank + _one(_comp(e[3], e[2]) > mw)
        else:
            es = [quad_get(q * 8 + u) for u in range_constexpr(8)]
            fx.rocdl.sched_barrier(0)
            for e in es:
                rank = rank + _one(_comp(e[1], e[0]) > mw)
                rank = rank + _one(_comp(e[3], e[2]) > mw)
        rank = _xlane4_sum(rank)
        if (q == 0) & (c < total) & (rank < TOPK):
            tmp[rank] = mi
        sync()
    return bad


@ASTRewriter.transform
def _select_loop(
    tx, score_row, width, nblk, init_blocks, local_blocks, pins, pre, cand, sel, tmp
):
    """Rows of more than SELECT_BLOCKS live blocks: windows of 4 waves x 7 strips x 64
    blocks, each narrowed as ``_select_window`` does and merged with the running top 16 in
    ``sel`` (a window's candidates must clear the running 16th key as well). Returns ``bad``;
    when 0 the ids are in ``tmp[0..15]``."""
    lane, wave = tx % 64, tx // 64
    nlive = (nblk < width).select(nblk, width)
    local_start = (nblk > local_blocks).select(nblk - local_blocks, fx.Int32(0))
    srs = ptr_buf_tensor(
        score_row, fx.Int32, unit_elems=1, num_records_bytes=nlive.to(fx.Int64) * 4
    )

    def sync():
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()

    def view(ptr, n_dw, w):
        return fx.logical_divide(
            fx.make_view(fx.recast_iter(fx.Int32, ptr), fx.make_layout(n_dw, 1)),
            fx.make_layout(w, 1),
        )

    pairs, quads = view(cand, CAND_DWORDS, 2), view(cand, CAND_DWORDS, 4)
    atom64 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)
    atom128 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

    def pair_put(i, blk, key):
        r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
        r.store(Vec.from_elements([blk, key], fx.Int32))
        fx.copy(atom64, r, fx.slice(pairs, (None, i)))

    def quad_get(i):
        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        fx.copy(atom128, fx.slice(quads, (None, i)), r)
        v = Vec(r.load())
        return [fx.Int32(v[e]) for e in range_constexpr(4)]

    def reduce_waves(v):
        tmp[tx] = v
        sync()
        return _imax(
            _imax(fx.Int32(tmp[lane]), fx.Int32(tmp[64 + lane])),
            _imax(fx.Int32(tmp[128 + lane]), fx.Int32(tmp[192 + lane])),
        )

    nrun, tprev = fx.Int32(0), fx.Int32(INT_MIN + 1)
    wbase, bad = fx.Int32(0), fx.Int32(0)
    while (wbase < nlive) & (bad == 0):
        keys = []
        for j in range_constexpr(SELECT_STRIPS):
            col = wbase + (j * 4 + wave) * 64 + lane
            raw = pre[j]
            if wbase != 0:
                raw = fx.Int32(buf_copy_load(srs, col))
            keys.append(
                (col < nlive).select(
                    _pinned_key(raw, col, init_blocks, local_start, pins),
                    fx.Int32(INT_MIN),
                )
            )
        m = keys[0]
        for j in range_constexpr(1, SELECT_STRIPS):
            m = _imax(m, keys[j])
        m = reduce_waves(m)
        # The bound: 16 key bits of this window's 16th lane maximum while fewer than 16 keys
        # are held; afterwards the running 16th key, and, when 16 lane maxima reach it (scores
        # rising toward the row's end), all 32 bits of the window's own 16th lane maximum.
        tw = fx.Int32(INT_MIN)
        if nrun < TOPK:
            p = fx.Int32(0)
            for d in range_constexpr(BOUND_BITS):
                t = p | _s32(1 << (31 - d))
                count = _popc(_ballot(m >= (t ^ INT_MIN)))
                p = (count >= TOPK).select(t, p)
            tw = p ^ INT_MIN
        else:
            if _popc(_ballot(m >= tprev)) >= TOPK:
                p = fx.Int32(0)
                bit = fx.Int32(31)
                while bit >= 0:
                    t = p | (fx.Int32(1) << bit)
                    p = (_popc(_ballot(m >= (t ^ INT_MIN))) >= TOPK).select(t, p)
                    bit = bit - 1
                tw = p ^ INT_MIN
        ta = _imax(tw, tprev)
        oks = [_one(k >= ta) for k in keys]
        cnt = oks[0]
        for j in range_constexpr(1, SELECT_STRIPS):
            cnt = cnt + oks[j]
        prefix, wtotal = fx.Int32(0), fx.Int32(0)
        for b in range_constexpr(SELECT_STRIPS.bit_length()):
            mask = _ballot((cnt & (1 << b)) != 0)
            prefix = prefix + (_below(mask, 0) << b)
            wtotal = wtotal + (_popc(mask) << b)
        if lane == 0:
            tmp[256 + wave] = wtotal
        sync()
        offset, total = nrun + prefix, nrun
        for w in range_constexpr(4):
            wt = fx.Int32(tmp[256 + w])
            offset = offset + (wave > w).select(wt, fx.Int32(0))
            total = total + wt
        bad = ((total > 64) | (ta <= KEY_NEG_INF)).select(fx.Int32(1), fx.Int32(0))
        if bad == 0:
            if wave == 0:
                if lane < nrun:
                    pair_put(lane, fx.Int32(sel[4 * lane]), fx.Int32(sel[4 * lane + 1]))
                if lane < 8:
                    pair_put(total + lane, fx.Int32(0), fx.Int32(INT_MIN))
            for j in range_constexpr(SELECT_STRIPS):
                if oks[j] != 0:
                    pair_put(offset, wbase + (j * 4 + wave) * 64 + lane, keys[j])
                offset = offset + oks[j]
            sync()
            mi, mk = fx.Int32(cand[2 * lane]), fx.Int32(cand[2 * lane + 1])
            mw = _comp(mk, mi)
            rank = fx.Int32(0)
            # each wave compares two groups of eight candidates; the partial ranks are summed
            for g in range_constexpr(2):
                start = wave * 8 + g * 32
                if total > start:
                    es = [quad_get(start // 2 + u) for u in range_constexpr(4)]
                    fx.rocdl.sched_barrier(0)
                    for e in es:
                        rank = rank + _one(_comp(e[1], e[0]) > mw)
                        rank = rank + _one(_comp(e[3], e[2]) > mw)
            tmp[tx] = rank
            sync()
            if wave == 0:
                r = (
                    fx.Int32(tmp[lane])
                    + fx.Int32(tmp[64 + lane])
                    + fx.Int32(tmp[128 + lane])
                    + fx.Int32(tmp[192 + lane])
                )
                if (lane < total) & (r < TOPK):
                    sel[4 * r] = mi
                    sel[4 * r + 1] = mk
            sync()
            nrun = (total < TOPK).select(total, fx.Int32(TOPK))
            tprev = (nrun >= TOPK).select(
                fx.Int32(sel[4 * 15 + 1]), fx.Int32(INT_MIN + 1)
            )
        wbase = wbase + 4 * SELECT_STRIPS * 64
    if (bad == 0) & (tx < TOPK):
        tmp[tx] = (tx < nrun).select(fx.Int32(sel[4 * tx]), fx.Int32(-1))
    sync()
    return bad


@ASTRewriter.transform
def _select_exact(tx, score_row, width, nblk, init_blocks, local_blocks, pins, tmp):
    """The rare path (fewer than 16 live blocks, or more than 64 candidates at the bound):
    every live block counts the blocks ordered above it, reading the row from memory, so
    O(live) per block. The ids (-1 filled) end up in ``tmp[0..15]``."""
    live = (nblk < width).select(nblk, width)
    local_start = (nblk > local_blocks).select(nblk - local_blocks, fx.Int32(0))
    srs = ptr_buf_tensor(
        score_row, fx.Int32, unit_elems=1, num_records_bytes=width.to(fx.Int64) * 4
    )
    if tx < TOPK:
        tmp[tx] = fx.Int32(-1)
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    fx.gpu.barrier()
    n_steps = (live + 255) // 256
    for step in range(n_steps):
        c = tx + step * 256
        if c < live:
            mw = _comp(
                _pinned_key(
                    fx.Int32(buf_copy_load(srs, c)), c, init_blocks, local_start, pins
                ),
                c,
            )
            rank = fx.Int32(0)
            for j in range(live):
                jj = fx.Int32(j)
                kj = _pinned_key(
                    fx.Int32(buf_copy_load(srs, jj)), jj, init_blocks, local_start, pins
                )
                rank = rank + _one(_comp(kj, jj) > mw)
            if rank < TOPK:
                tmp[rank] = c
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    fx.gpu.barrier()


@ASTRewriter.transform
def row_top16_prologue(
    tx,
    score_row,
    width,
    pre,
    nblk,
    init_blocks,
    local_blocks,
    pins,
    cand,
    sel,
    tmp,
    out_row,
):
    """The top-16 of one decode row, run by every thread of a 256-thread workgroup before
    its own work. The ids (-1 past the live count) are in ``tmp[0..15]`` behind this
    function's final barrier and are also stored to ``out_row`` (the i64 address of 16
    int32; every workgroup of a row stores the same bytes, and only final values: a
    sibling may read them back).

    ``score_row``: the row's i64 address (fp32, ``width`` readable columns, columns past
    the row's live blocks are never trusted); ``pre``: ``first_window`` of it, which the
    caller issues before it knows the row's length; ``nblk``: the row's live blocks,
    ``ceil((seq - ql + row_in_req + 1) / 128)``; ``cand`` (CAND_DWORDS), ``sel``
    (SEL_DWORDS) and ``tmp`` (TMP_DWORDS): LDS the caller leaves alone until it has read
    the ids."""
    bad = fx.Int32(0)
    if nblk <= SELECT_BLOCKS:
        bad = _select_window(tx, nblk, init_blocks, local_blocks, pre, cand, tmp, pins)
    else:
        bad = _select_loop(
            tx,
            score_row,
            width,
            nblk,
            init_blocks,
            local_blocks,
            pins,
            pre,
            cand,
            sel,
            tmp,
        )
    if bad != 0:
        _select_exact(tx, score_row, width, nblk, init_blocks, local_blocks, pins, tmp)
    if tx < TOPK:
        outr = ptr_buf_tensor(
            out_row, fx.Int32, unit_elems=1, num_records_bytes=fx.Int64(TOPK * 4)
        )
        buf_copy_store(outr, tx, fx.Int32(tmp[tx]))
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    fx.gpu.barrier()
