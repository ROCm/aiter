# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 prefill index top-16: the block ids of ragged fp32 score rows (FlyDSL, gfx950).

Input: the scores ``score_prefill_flydsl*`` writes, ``[heads, rows, width]`` fp32 (the raw ``max(q . k)`` of each
(row, block), no sentinels), and ``n_valid_column_per_row`` (live blocks of each row, head-major
``[heads * rows]``). Output: ``[heads, rows, 16]`` int32 block ids, best first, ``-1`` after a row's live blocks.
The ``init`` leading and ``local`` trailing live blocks of a row are forced into the selection, as in the
selector the sentinels 1e30 (init) / 1e29 (local, wins over init) give: they are applied to the keys in the
kernel, so the scores stay untouched.

Order: key = the fp32 score's order-preserving bits (NaN as -inf, -0 below +0), ties to the larger block index.

One wave per (row, head). The row is read in windows of STRIPS 64-wide strips (lane l holds blocks 64 j + l); a
window's candidates are the keys at or above the 16th largest lane maximum (and the running 16th of earlier
windows), compacted into LDS and ranked against each other. A window whose candidates overflow one wave, or
whose NaNs / -infs could reach the top 16, takes an exact bit search for its 16th key and the index cut.

Technique and kernel structure from amd-kernels (``indexer_nvfp4_d128_p128_top16/topk``, Apache-2.0).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

BLK = 128  # tokens per block
TOPK = 16
PREFILL_STRIPS = (
    4,
    8,
    16,
    20,
    26,
)  # window tiles (64-block strips), chosen by max_seq_len
MAX_SLOTS = 128  # 8192 blocks: the 13 index bits of the key tie-break
CAP = 64  # candidates one rank pass holds (one per lane)
IDX_BITS = 13  # block index bits (slots * 64 <= 8192)
INT_MIN = -(1 << 31)
KEY_NEG_INF = (0xFF800000 ^ 0x7FFFFFFF) - (1 << 32)  # signed key of -inf (NaN's)
KEY_POS_INF = 0x7F800000
KEY_INIT = 0x7149F2CA  # fp32 bits of 1e30 (a positive float's key is its bits)
KEY_LOCAL = 0x6FA18F08  # fp32 bits of 1e29
# Bits of the 16th lane maximum that bound the candidates (swept 12-16 over 4-16K rows, 0-256K context).
PREFILL_BOUND_BITS = 14
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


def cand_dwords(strips: int) -> int:
    """LDS dwords of one row's candidate pairs (``row_top16``'s ``cand``)."""
    return 2 * (CAP + 8 + 64)


SEL_DWORDS = 4 * TOPK  # LDS dwords of one row's running top 16 (``sel``)


def row_top16(
    strips,
    lane,
    score_row,
    width,
    nblk_fn,
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
    bound_bits=PREFILL_BOUND_BITS,
):
    """One wave's top 16 of one score row (``strips`` x 64 blocks per window): ``score_row`` the row's i64 address,
    ``width`` (i32) its readable blocks, ``nblk_fn()`` its block count (called once the first strips are in flight),
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
    assert cand_dwords(strips) == 2 * (TRASH + 64)

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
    nblk = nblk_fn()
    nlive = (nblk < width).select(nblk, width)
    # later windows: the row's live blocks only (no dead-tail reads)
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
            blk = wbase + (j * 64) + lane
            live = blk < nlive
            kj = _key(bits[j])
            kj = (blk < init_blocks).select(fx.Int32(KEY_INIT), kj)
            kj = (blk >= nblk - local_blocks).select(fx.Int32(KEY_LOCAL), kj)
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
                # every live block of the window is a candidate (strip 0)
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
    blk = (lane < nrun).select(r[0], fx.Int32(-1))
    buffer_ops.buffer_store(
        blk,
        _rsrc(out_row, TOPK * 4),
        lane,
        mask=lane < TOPK,
    )


# A module-level helper is not rewritten with the kernel that calls it: its dynamic ifs / whiles become scf ops only
# through the kernel's rewriter, applied here by hand.
row_top16 = ASTRewriter.transform(row_top16)


def compile_topk_prefill(strips: int):
    """One window tile (``strips``); shapes, ``init`` / ``local`` are launch arguments."""
    assert strips in PREFILL_STRIPS
    STRIPS = strips
    CAND = cand_dwords(STRIPS) // 2
    name = f"minimax_m3_index_top16_prefill_w{STRIPS}"

    @fx.struct
    class Shared:
        cand: fx.Array[fx.Int32, 2 * CAND, 16]
        sel: fx.Array[fx.Int32, SEL_DWORDS, 16]

    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def kernel(
        arg_score: fx.Int64,
        i64_s_head: fx.Int64,
        i64_s_row: fx.Int64,
        i32_width: fx.Int32,
        arg_nvp: fx.Int64,
        i32_rows: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        arg_topk: fx.Int64,
        i64_t_head: fx.Int64,
        i64_t_row: fx.Int64,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        n, h = fx.block_idx.x, fx.block_idx.y
        lane = fx.thread_idx.x
        n64 = fx.Int32(n).to(fx.Int64)
        h64 = fx.Int32(h).to(fx.Int64)

        def nblk_fn():
            # n_valid is head-major: row (h, n) at h * rows + n
            idx = h64 * fx.Int32(i32_rows).to(fx.Int64) + n64
            return fx.Int32(
                buffer_ops.buffer_load(
                    _rsrc(arg_nvp + idx * 4, fx.Int64(4)),
                    fx.Int32(0),
                    vec_width=1,
                    is_scalar=True,
                )
            )

        row_top16(
            STRIPS,
            lane,
            arg_score + (h64 * i64_s_head + n64 * i64_s_row) * 4,
            i32_width,
            nblk_fn,
            smem.cand.ptr,
            2 * CAND,
            smem.sel.ptr,
            SEL_DWORDS,
            0,
            0,
            arg_topk + (h64 * i64_t_head + n64 * i64_t_row) * 4,
            fx.gpu.barrier,
            i32_init,
            i32_local,
        )

    @flyc.jit
    def launch(
        arg_score: fx.Int64,
        i64_s_head: fx.Int64,
        i64_s_row: fx.Int64,
        i32_width: fx.Int32,
        arg_nvp: fx.Int64,
        i32_rows: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        arg_topk: fx.Int64,
        i64_t_head: fx.Int64,
        i64_t_row: fx.Int64,
        i32_heads: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_score,
            i64_s_head,
            i64_s_row,
            i32_width,
            arg_nvp,
            i32_rows,
            i32_init,
            i32_local,
            arg_topk,
            i64_t_head,
            i64_t_row,
        ).launch(
            grid=(fx.Int64(i32_rows), fx.Int64(i32_heads), 1),
            block=(64, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    # the score / table / length addresses arrive in SGPRs: the first loads do not wait for the kernel-argument load
    launch.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": True,
            "amdgpu-kernarg-preload-count": 16,
        }
    }
    return launch


@functools.cache
def _get_kernel(strips: int, device_index: int):
    # device_index only keys the cache: the launcher memoises a handle bound to the device it compiled on
    return compile_topk_prefill(strips)


def _pow2_ceil(n: int) -> int:
    return 1 << (n - 1).bit_length() if n >= 1 else 1


def minimax_m3_index_topk_prefill_flydsl(
    score: torch.Tensor,  # [heads, rows, width] fp32, raw scores (no sentinels)
    n_valid_column_per_row: torch.Tensor,  # [heads * rows] int32, live blocks of each row
    topk_idx: torch.Tensor,  # [heads, rows, 16] int32 out
    max_seq_len: int,
    init_blocks: int = 0,
    local_blocks: int = 0,
) -> torch.Tensor:
    """Top-16 block ids of ragged prefill score rows (gfx950); see the module docstring for the contract.

    ``score.stride(2)`` must be 1; reads of a row stay below ``min(n_valid, width)`` blocks. ``max_seq_len`` only
    picks the window tile; longer rows continue through further windows. At most 8192 live blocks a row.
    """
    require_gfx950()
    heads, rows, width = score.shape
    if score.dtype != torch.float32 or score.stride(2) != 1:
        raise ValueError("score must be fp32 with a unit last stride")
    if (
        topk_idx.dtype != torch.int32
        or topk_idx.shape != (heads, rows, TOPK)
        or topk_idx.stride(2) != 1
    ):
        raise ValueError(
            f"topk_idx must be int32 [{heads}, {rows}, {TOPK}] with a unit last stride"
        )
    nv = n_valid_column_per_row
    if not (
        score.is_cuda and nv.device == score.device and topk_idx.device == score.device
    ):
        raise ValueError(
            "score, n_valid_column_per_row and topk_idx must be on one GPU"
        )
    if nv.dtype != torch.int32 or not nv.is_contiguous() or nv.numel() < heads * rows:
        raise ValueError(
            "n_valid_column_per_row must be contiguous int32 with heads * rows values"
        )
    if (
        _pow2_ceil(-(-max_seq_len // BLK)) > MAX_SLOTS * 64
        or -(-max_seq_len // BLK) > MAX_SLOTS * 64
    ):
        raise ValueError("at most 8192 blocks per row")
    if init_blocks < 0 or local_blocks < 0:
        raise ValueError("init_blocks and local_blocks must be >= 0")
    if rows == 0 or heads == 0:
        return topk_idx
    strips = next(
        (s for s in PREFILL_STRIPS if max_seq_len <= s * BLK * 64), PREFILL_STRIPS[-1]
    )
    dev = (
        score.device.index
        if score.device.index is not None
        else torch.cuda.current_device()
    )
    _run_compiled(
        _get_kernel(strips, dev),
        score.data_ptr(),
        score.stride(0),
        score.stride(1),
        width,
        nv.data_ptr(),
        rows,
        init_blocks,
        local_blocks,
        topk_idx.data_ptr(),
        topk_idx.stride(0),
        topk_idx.stride(1),
        heads,
        torch.cuda.current_stream(score.device),
    )
    return topk_idx


def require_gfx950():
    from aiter.jit.utils.chip_info import get_gfx_runtime

    arch = get_gfx_runtime()
    if arch != "gfx950":
        raise RuntimeError(
            f"minimax_m3_index_topk_prefill_flydsl requires gfx950, got {arch}"
        )
