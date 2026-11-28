# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-row top-16 of ragged fp32 score rows, one wave per row (FlyDSL, gfx950).

The wide-row, k=16 build behind ``topk_per_row_small_k``: one 64-lane wave selects a whole row, with no
workgroup barrier, which is what a block-sparse indexer's prefill needs -- many rows of a few hundred to a few
thousand scores each.

Contract: ``scores`` are ``[rows, width]`` fp32 with inner stride 1, ``row_lens`` the live columns of each row
(``None``: every row is ``width`` wide), and the output ``[rows, 16]`` int32 column ids, best first, ``-1`` after a
row's live columns. The ``init`` leading and ``local`` trailing live columns of a row are forced into the selection
(local outranks init), applied to the keys in the kernel so the scores stay untouched.

Order: key = the fp32 score's order-preserving bits (NaN as -inf, -0 below +0), ties to the larger column.

A row is read in windows of STRIPS 64-wide strips (lane l holds columns 64 j + l); a window's candidates are the
keys at or above the 16th largest lane maximum (and the running 16th of earlier windows), compacted into LDS and
ranked against each other. A window whose candidates overflow one wave, or whose NaNs / -infs could reach the top
16, takes an exact bit search for its 16th key and the index cut.

At most 8192 columns a row: the tie-break holds the column in 13 bits.

Technique and kernel structure from amd-kernels (``indexer_nvfp4_d128_p128_top16/topk``, Apache-2.0), rewritten
in FlyDSL; nothing is imported from it.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops

TOPK = 16
STRIP_TILES = (
    4,
    8,
    16,
    20,
    26,
)  # window tiles (64-column strips), chosen by the row width
MAX_SLOTS = 128  # 8192 columns: the 13 index bits of the key tie-break
MAX_WIDTH = MAX_SLOTS * 64
CAP = 64  # candidates one rank pass holds (one per lane)
IDX_BITS = 13  # column index bits (slots * 64 <= 8192)
INT_MIN = -(1 << 31)
KEY_NEG_INF = (0xFF800000 ^ 0x7FFFFFFF) - (1 << 32)  # signed key of -inf (NaN's)
KEY_POS_INF = 0x7F800000
KEY_INIT = 0x7149F2CA  # fp32 bits of 1e30 (a positive float's key is its bits)
KEY_LOCAL = 0x6FA18F08  # fp32 bits of 1e29
# Bits of the 16th lane maximum that bound the candidates (swept 12-16 over 4-16K rows, 0-256K context).
BOUND_BITS = 14
# Strips of the first window loaded before the row's length is known; the rest only up to the row's live end.
FIRST_STRIPS = 4


def _s32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v


def _key(bits):
    """Signed key in AITER's order (the float's bits, negatives flipped); NaNs land above +inf / below -inf and are
    folded to -inf only where they could matter (``_fold_nan``)."""
    return bits ^ ((bits >> 31) & 0x7FFFFFFF)


def _fold_nan(k):
    return ((k > KEY_POS_INF) | (k < KEY_NEG_INF)).select(fx.Int32(KEY_NEG_INF), k)


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
    """(key, index) as one i64, the key in the high word: the signed compare is AITER's order."""
    return (k.to(fx.Int64) << 32) | (i.to(fx.Int64) & 0xFFFFFFFF)


def _rsrc(base, nbytes):
    """A buffer over ``nbytes`` from the 64-bit address ``base``: every tensor is rebased per row (bases in i64, the
    operands widened before they multiply), so offsets stay small at any tensor size."""
    return buffer_ops.create_buffer_resource_from_addr(
        base, num_records_bytes=fx.Int64(nbytes)
    )


# LDS dwords of one row's candidate pairs (``row_top16``'s ``cand``).
CAND_DWORDS = 2 * (CAP + 8 + 64)
SEL_DWORDS = 4 * TOPK  # LDS dwords of one row's running top 16 (``sel``)


# A module-level device helper: its dynamic ifs / whiles need the AST rewrite to become scf ops. ``flyc.jit``
# applies that rewrite, and when called inside a kernel's trace (below) it inlines the rewritten body into the
# kernel's IR rather than emitting a separate launchable.
@flyc.jit
def row_top16(
    strips,
    lane,
    score_row,
    width,
    len_fn,
    cand_ptr,
    cand_total,
    sel_ptr,
    sel_total,
    cbase,
    sbase,
    out_row,
    sync,
    init_blocks,
    local_blocks,
    bound_bits=BOUND_BITS,
):
    """One wave's top 16 of one score row (``strips`` x 64 columns per window): ``score_row`` the row's i64 address,
    ``width`` (i32) its readable columns, ``len_fn()`` its live length (called once the first strips are in flight),
    the row's LDS at pair ``cbase`` of ``cand_ptr`` and quad ``sbase`` of ``sel_ptr`` (dwords ``cand_total`` /
    ``sel_total`` in all), the 16 ids (-1 past the row's count) to ``out_row``; ``sync()`` orders the wave's LDS
    writes before its reads of other lanes' (a workgroup barrier when the wave is the workgroup, else a wave-local
    wait)."""
    STRIPS = strips
    COUNT_BITS = STRIPS.bit_length()  # a lane's count of kept keys is <= STRIPS
    FIRST = min(FIRST_STRIPS, STRIPS)
    BITS = bound_bits
    WIN = STRIPS * 64
    RANK_BLOCK = 8
    # running / window candidates, rank-loop padding, then one discard slot per lane; an overflowing window rebuilds
    # its candidates exactly
    TRASH = CAP + RANK_BLOCK

    def at(base, i):
        return i if isinstance(base, int) and base == 0 else i + base

    # this row's scores, bounded by its width: a window past it reads 0s
    srs = _rsrc(score_row, width.to(fx.Int64) * 4)
    pos = lane

    def load_strips(srs, wbase, j0, j1):
        """Strips j0..j1-1 of a window's score bits (past the row: masked as
        keys, or 0)."""
        return [
            fx.Int32(
                buffer_ops.buffer_load(
                    srs,
                    wbase + (j * 64) + pos,
                    vec_width=1,
                    dtype=fx.Int32,
                )
            )
            for j in range_constexpr(j0, j1)
        ]

    def load_window(srs, wbase):
        return Vec.from_elements(load_strips(srs, wbase, 0, STRIPS), fx.Int32)

    # the first window's first strips go out before the row's length is known
    head = load_strips(srs, fx.Int32(0), 0, FIRST)
    row_len = len_fn()
    nlive = (row_len < width).select(row_len, width)
    # a negative length is a pad row: no live column, and no negative buffer bound or candidate count
    nlive = (nlive > fx.Int32(0)).select(nlive, fx.Int32(0))
    # later windows: the row's live columns only (no dead-tail reads)
    live_end = nlive
    srs_live = _rsrc(score_row, live_end.to(fx.Int64) * 4)
    # the rest of the first window: the row's live strips only
    bits = Vec.from_elements(
        head + load_strips(srs_live, fx.Int32(0), FIRST, STRIPS), fx.Int32
    )

    def view(ptr, n_dw, w):
        return fx.logical_divide(
            fx.make_view(fx.recast_iter(fx.Int32, ptr), fx.make_layout(n_dw, 1)),
            fx.make_layout(w, 1),
        )

    pairs = view(cand_ptr, cand_total, 2)
    quads = view(cand_ptr, cand_total, 4)
    sels = view(sel_ptr, sel_total, 4)
    qbase = cbase // 2 if isinstance(cbase, int) else cbase >> 1
    atom64 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)
    atom128 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

    def lds_put(atom, v, w, vals):
        r = fx.make_rmem_tensor(fx.make_layout(w, 1), fx.Int32)
        r.store(Vec.from_elements(vals, fx.Int32))
        fx.copy(atom, r, v)

    def lds_get(atom, v, w):
        r = fx.make_rmem_tensor(fx.make_layout(w, 1), fx.Int32)
        fx.copy(atom, v, r)
        x = Vec(r.load())
        return [fx.Int32(x[e]) for e in range_constexpr(w)]

    def put_pair(pos, i, k):
        lds_put(atom64, fx.slice(pairs, (None, at(cbase, pos))), 2, [i, k])

    def wave_sum(c):
        """Sum over the wave of a per-lane count in [0, STRIPS]."""
        t = fx.Int32(0)
        for b in range_constexpr(COUNT_BITS):
            t = t + (_popc(_ballot((c & (1 << b)) != 0)) << b)
        return t

    def compact(keys, wbase, base, pred):
        """The keys ``pred`` keeps, appended at ``base`` in lane-major order
        (only if they fit below CAP); returns the new end either way."""
        oks = [
            _one(pred(keys[j], wbase + (j * 64) + lane))
            for j in range_constexpr(STRIPS)
        ]
        cnt = oks[0]
        for j in range_constexpr(1, STRIPS):
            cnt = cnt + oks[j]
        pos = fx.Int32(base)
        end = fx.Int32(base)
        for b in range_constexpr(6):  # cnt <= 32
            bm = _ballot((cnt & (1 << b)) != 0)
            pos = pos + (_below(bm, 0) << b)
            end = end + (_popc(bm) << b)
        # the wave-uniform count includes every kept key; an overflow skips the writes and the caller rebuilds
        if end <= CAP:
            for j in range_constexpr(STRIPS):
                put_pair(
                    (oks[j] != 0).select(pos, TRASH + lane),
                    wbase + (j * 64) + lane,
                    keys[j],
                )
                pos = pos + oks[j]
        return end

    def exact_cut(keys, wbase):
        """The window's 16th key v and the index cut ie among the keys tied
        with it: (k > v) | (k == v & i >= ie) keeps exactly its top 16."""
        pe = fx.Int32(0)
        b = fx.Int32(31)
        while b >= 0:
            t = (pe | (fx.Int32(1) << b)) ^ INT_MIN
            c = fx.Int32(0)
            for j in range_constexpr(STRIPS):
                c = c + _one(keys[j] >= t)
            pe = (wave_sum(c) >= TOPK).select(t ^ INT_MIN, pe)
            b = b - 1
        v = pe ^ INT_MIN
        c = fx.Int32(0)
        for j in range_constexpr(STRIPS):
            c = c + _one(keys[j] > v)
        need = TOPK - wave_sum(c)
        ie = fx.Int32(0)
        b = fx.Int32(IDX_BITS - 1)
        while b >= 0:
            t = ie | (fx.Int32(1) << b)
            c = fx.Int32(0)
            for j in range_constexpr(STRIPS):
                c = c + _one((keys[j] == v) & ((wbase + (j * 64) + lane) >= t))
            ie = (wave_sum(c) >= need).select(t, ie)
            b = b - 1
        return v, ie

    def lane_max_bound(m):
        p = fx.Int32(0)
        for d in range_constexpr(BITS):
            t = p | _s32(1 << (31 - d))
            count = _popc(_ballot(m >= (t ^ INT_MIN)))
            p = (count >= TOPK).select(t, p)
        return p ^ INT_MIN

    def window(bits, wbase, nrun, tprev):
        """One window of keys merged into the running top 16 (sel)."""
        keys = []
        for j in range_constexpr(STRIPS):
            col = wbase + (j * 64) + lane
            live = col < nlive
            kj = _key(bits[j])
            kj = (col < init_blocks).select(fx.Int32(KEY_INIT), kj)
            kj = (col >= nlive - local_blocks).select(fx.Int32(KEY_LOCAL), kj)
            keys.append(live.select(kj, fx.Int32(INT_MIN)))
        m = keys[0]
        for j in range_constexpr(1, STRIPS):
            m = _imax(m, keys[j])
        # the first window is bounded by its lane maxima, a later one by the
        # running 16th
        tw = fx.Int32(INT_MIN)
        if nrun < TOPK:
            tw = lane_max_bound(m)
        ta = _imax(tw, tprev)
        total = compact(keys, wbase, nrun, lambda k, i: k >= ta)
        # NaN keys are not folded yet: exact only while 16 keys above -inf
        # bound the window and no NaN sits above +inf
        nan_hi = _ballot(m > KEY_POS_INF) != fx.Int64(0)
        if (total > CAP) | (ta <= KEY_NEG_INF) | nan_hi:
            wl = nlive - wbase
            if wl <= CAP - nrun:
                # every live column of the window is a candidate (strip 0)
                put_pair(
                    (lane < wl).select(nrun + lane, TRASH + lane),
                    wbase + lane,
                    _fold_nan(keys[0]),
                )
                total = nrun + wl
            else:
                # NaNs as -inf, re-bounded by the lane maxima; the exact
                # search only if that still overflows
                fixed = []
                for j in range_constexpr(STRIPS):
                    live = (wbase + (j * 64) + lane) < nlive
                    fixed.append(live.select(_fold_nan(keys[j]), fx.Int32(INT_MIN)))
                mf = fixed[0]
                for j in range_constexpr(1, STRIPS):
                    mf = _imax(mf, fixed[j])
                tf = _imax(lane_max_bound(mf), tprev)
                total = compact(fixed, wbase, nrun, lambda k, i: k >= tf)
                if total > CAP:
                    v, ie = exact_cut(fixed, wbase)
                    total = compact(
                        fixed,
                        wbase,
                        nrun,
                        lambda k, i: (k > v) | ((k == v) & (i >= ie)),
                    )
        if lane < RANK_BLOCK:
            put_pair(total + lane, fx.Int32(0), fx.Int32(INT_MIN))
        sync()
        mi, mk = lds_get(atom64, fx.slice(pairs, (None, at(cbase, lane))), 2)
        mw = _comp(mk, mi)
        rank = fx.Int32(0)
        for g in range_constexpr(CAP // RANK_BLOCK):
            if total > RANK_BLOCK * g:
                es = [
                    lds_get(
                        atom128,
                        fx.slice(quads, (None, at(qbase, RANK_BLOCK // 2 * g + u))),
                        4,
                    )
                    for u in range_constexpr(RANK_BLOCK // 2)
                ]
                # all of a block's reads in flight before the compares
                fx.rocdl.sched_barrier(0)
                for e in es:
                    rank = rank + _one(_comp(e[1], e[0]) > mw)
                    rank = rank + _one(_comp(e[3], e[2]) > mw)
        if (lane < total) & (rank < TOPK):
            lds_put(
                atom128, fx.slice(sels, (None, at(sbase, rank))), 4, [mi, mk, 0, mi]
            )
        return (total < TOPK).select(total, fx.Int32(TOPK))

    wbase = fx.Int32(0)
    nrun = fx.Int32(0)
    tprev = fx.Int32(INT_MIN + 1)  # dead keys are INT_MIN
    while (wbase == 0) | (wbase < nlive):
        if wbase > 0:
            bits = load_window(srs_live, wbase)
            # the running top 16 back into cand; its 16th key bounds the window
            sync()
            r = lds_get(
                atom128, fx.slice(sels, (None, at(sbase, lane & (TOPK - 1)))), 4
            )
            if lane < nrun:
                put_pair(lane, r[0], r[1])
            k15 = fx.Int32(fx.rocdl.readlane(T.i32, r[1], fx.Int32(TOPK - 1)))
            tprev = (nrun >= TOPK).select(k15, fx.Int32(INT_MIN + 1))
            sync()
        nrun = window(
            [fx.Int32(bits[j]) for j in range_constexpr(STRIPS)],
            wbase,
            nrun,
            tprev,
        )
        wbase = wbase + WIN

    # outputs
    sync()
    r = lds_get(atom128, fx.slice(sels, (None, at(sbase, lane & (TOPK - 1)))), 4)
    ids = (lane < nrun).select(r[0], fx.Int32(-1))
    buffer_ops.buffer_store(
        ids,
        _rsrc(out_row, TOPK * 4),
        lane,
        mask=lane < TOPK,
    )


@functools.cache
def build_topk_per_row_small_k_wave_module(strips: int, full_rows: bool):
    """One window tile (``strips``); the row count, width, ``init`` / ``local`` and strides are launch arguments.

    ``full_rows`` compiles every row's length in as the width, so the kernel reads no length tensor.
    """
    assert strips in STRIP_TILES
    STRIPS = strips
    CAND = CAND_DWORDS // 2
    name = f"topk_per_row_small_k_wave_s{STRIPS}" + ("_full" if full_rows else "")

    @fx.struct
    class Shared:
        cand: fx.Array[fx.Int32, 2 * CAND, 16]
        sel: fx.Array[fx.Int32, SEL_DWORDS, 16]

    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def kernel(
        arg_score: fx.Int64,
        i64_s_row: fx.Int64,
        i32_width: fx.Int32,
        arg_lens: fx.Int64,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        arg_out: fx.Int64,
        i64_o_row: fx.Int64,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        n = fx.block_idx.x
        lane = fx.thread_idx.x
        n64 = fx.Int32(n).to(fx.Int64)

        def len_fn():
            if const_expr(full_rows):
                return i32_width
            return fx.Int32(
                buffer_ops.buffer_load(
                    _rsrc(arg_lens + n64 * 4, fx.Int64(4)),
                    fx.Int32(0),
                    vec_width=1,
                    is_scalar=True,
                )
            )

        row_top16(
            STRIPS,
            lane,
            arg_score + n64 * i64_s_row * 4,
            i32_width,
            len_fn,
            smem.cand.ptr,
            2 * CAND,
            smem.sel.ptr,
            SEL_DWORDS,
            0,
            0,
            arg_out + n64 * i64_o_row * 4,
            fx.gpu.barrier,
            i32_init,
            i32_local,
        )

    @flyc.jit
    def launch(
        arg_score: fx.Int64,
        i64_s_row: fx.Int64,
        i32_width: fx.Int32,
        arg_lens: fx.Int64,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        arg_out: fx.Int64,
        i64_o_row: fx.Int64,
        i32_rows: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_score,
            i64_s_row,
            i32_width,
            arg_lens,
            i32_init,
            i32_local,
            arg_out,
            i64_o_row,
        ).launch(
            grid=(fx.Int64(i32_rows), 1, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    # the score / length / output addresses arrive in SGPRs: the first loads do not wait for the kernel-argument load
    launch.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": True,
            "amdgpu-kernarg-preload-count": 16,
        }
    }
    return launch


def wave_strips(width: int) -> int:
    """The window tile for rows of up to ``width`` columns: the narrowest that holds the row in one window, the
    widest for anything longer (further windows continue the row)."""
    return next((s for s in STRIP_TILES if width <= s * 64), STRIP_TILES[-1])
