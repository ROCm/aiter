# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3 prefill index top-k fused with the MSA page sort_topk (FlyDSL, gfx950).

Ported from vLLM's m3_index_topk_prefill_indices (the prefill part of
models/minimax_m3/amd/ops/index_topk_decode/prefill_kernel.py), same results
bit for bit.

One 64-lane wave per (row, head) returns the row's 16 highest-scoring block ids
in AITER's pa_sparse_block_topk order, walking the row in windows of STRIPS x
64 blocks: a bound (the running 16th key, or the 16th largest lane maximum)
keeps few candidates, which are compacted into LDS and ranked exactly.

sort_topk_q = Q > 1 (aiter's Gluon MSA at BLOCK_M = 16 Q): one workgroup per (Q
consecutive rows, head), wave w the group's row w, and an epilogue that writes
the group's page sort_topk exactly as aiter's msa_sort_topk_kernel does (uniq / count /
tokblk / pos), so msa_fwd(sort_topk=...) needs no sort_topk pre-pass.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

# Legacy buffer_ops: the kernel is vLLM's FlyDSL 0.4 kernel, moved as is.
from . import buffer_ops

# The first 16 dwords of kernel arguments arrive in SGPRs, so a kernel's first
# loads do not wait for s_load.
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}

BLK = 128  # tokens per block
TOPK = 16
PREFILL_STRIPS = (4, 8, 16, 20, 26)
CAP = 64  # candidates one rank pass holds (one per lane)
IDX_BITS = 13  # block index bits (slots * 64 <= 8192)
INT_MIN = -(1 << 31)
KEY_NEG_INF = (0xFF800000 ^ 0x7FFFFFFF) - (1 << 32)  # signed key of -inf (NaN's)
KEY_POS_INF = 0x7F800000


def _s32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v


def _key(bits):
    """Signed key in AITER's order; NaNs land above +inf or below -inf."""
    return bits ^ ((bits >> 31) & 0x7FFFFFFF)


def _fold_nan(k):
    return ((k > KEY_POS_INF) | (k < KEY_NEG_INF)).select(fx.Int32(KEY_NEG_INF), k)


def _ballot(c):
    return fx.Int64(fx.rocdl.ballot(T.i64, c))


def _popc(m):
    """Set bits of a wave mask; two 32-bit counts keep the compare scalar."""
    lo = fx.math.ctpop(m.to(fx.Int32)).to(fx.Int32)
    hi = fx.math.ctpop((m >> 32).to(fx.Int32)).to(fx.Int32)
    return fx.Int32(lo) + fx.Int32(hi)


def _below(mask, base):
    """The set bits of `mask` below this lane, plus `base` (v_mbcnt)."""
    lo = mask.to(fx.Int32)
    hi = (mask >> 32).to(fx.Int32)
    r = fx.rocdl.mbcnt_lo(T.i32, lo.ir_value(), fx.Int32(base).ir_value())
    return fx.Int32(fx.rocdl.mbcnt_hi(T.i32, hi.ir_value(), r))


def _imax(a, b):
    return (a > b).select(a, b)


def _one(c):
    return c.select(fx.Int32(1), fx.Int32(0))


def _comp(k, i):
    """(key, index) as one i64 whose signed order is AITER's."""
    return (k.to(fx.Int64) << 32) | (i.to(fx.Int64) & 0xFFFFFFFF)


def build_fused_m3_index_topk_prefill_indices_sort_topk_module(strips: int, sort_topk_q: int = 1):
    """The launcher for windows of strips x 64 blocks and sort_topk_q rows per
    workgroup (1: top-k only)."""
    assert strips in PREFILL_STRIPS
    assert sort_topk_q in (1, 2, 4)
    STRIPS = strips
    UQ = sort_topk_q  # rows per workgroup
    SORT_TOPK = UQ > 1
    C = UQ * TOPK  # sort_topk candidates
    WAVES = UQ  # one wave per row
    MW = WAVES > 1  # several waves per workgroup
    NTHR = 64 * WAVES
    WIN = STRIPS * 64
    RANK_BLOCK = 8
    # Candidates, rank-loop padding, then one discard slot per lane.
    TRASH = CAP + RANK_BLOCK
    CAND = TRASH + 64
    name = f"fused_m3_index_topk_prefill_indices_sort_topk_w{STRIPS}_q{UQ}"

    @fx.struct
    class Shared:
        cand: fx.Array[fx.Int32, WAVES * 2 * CAND, 16]
        sel: fx.Array[fx.Int32, WAVES * 4 * TOPK, 16]

    @flyc.kernel(name=name, known_block_size=[NTHR, 1, 1])
    def kernel(
        arg_score: fx.Int64,
        i64_s_head: fx.Int64,
        i64_s_row: fx.Int64,
        i32_width: fx.Int32,
        arg_nvp: fx.Int64,
        arg_topk: fx.Int64,
        i64_t_head: fx.Int64,
        i64_t_row: fx.Int64,
        i32_rows: fx.Int32,
        i32_groups: fx.Int32,
        arg_bt: fx.Int64,  # sort_topk: int32 block table [reqs, i32_bt_stride]
        i32_bt_stride: fx.Int32,
        i32_bt_numel: fx.Int32,
        arg_qreq: fx.Int64,  # sort_topk: int32 [rows] request / position of each row
        arg_qpos: fx.Int64,
        arg_uniq: fx.Int64,  # sort_topk outputs, as aiter's msa_sort_topk_kernel
        arg_count: fx.Int64,
        arg_tokblk: fx.Int64,
        arg_upos: fx.Int64,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        g, h = fx.block_idx.x, fx.block_idx.y
        if const_expr(MW):
            tid = fx.Int32(fx.thread_idx.x)
            lane = tid & 63
            # Wave index, scalar.
            wv = fx.Int32(fx.rocdl.readfirstlane(T.i32, (tid >> 6).ir_value()))
            c_off = wv * CAND  # this wave's candidate pairs
            s_off = wv * TOPK  # this wave's running top 16
        else:
            lane = fx.thread_idx.x
            c_off = 0
            s_off = 0
        if const_expr(SORT_TOPK):
            n = fx.Int32(g) * UQ + wv  # this wave's row
            row_ok = n < i32_rows
            n_ld = row_ok.select(n, i32_rows - 1)  # loads stay in bounds
        else:
            n = g
            n_ld = g
        n64 = fx.Int32(n_ld).to(fx.Int64)
        h64 = fx.Int32(h).to(fx.Int64)

        def rsrc(base, nbytes):
            """Buffer at an i64 base; per-row rebasing keeps offsets small."""
            return buffer_ops.create_buffer_resource_from_addr(
                base, num_records_bytes=fx.Int64(nbytes)
            )

        # Reads past the row's width return 0.
        score_row = arg_score + (h64 * i64_s_head + n64 * i64_s_row) * 4
        srs = rsrc(score_row, i32_width.to(fx.Int64) * 4)

        def load_window(srs, wbase):
            return Vec.from_elements(
                [
                    fx.Int32(
                        buffer_ops.buffer_load(
                            srs,
                            wbase + (j * 64) + lane,
                            vec_width=1,
                            dtype=fx.Int32,
                        )
                    )
                    for j in range_constexpr(STRIPS)
                ],
                fx.Int32,
            )

        # The first window is loaded before the row's length is known.
        bits = load_window(srs, 0)

        def scalar_at(base, idx64, nbytes):
            return fx.Int32(
                buffer_ops.buffer_load(
                    rsrc(base + idx64 * 4, nbytes),
                    fx.Int32(0),
                    vec_width=1,
                    is_scalar=True,
                )
            )

        nblk = scalar_at(arg_nvp, n64, fx.Int64(4))
        if const_expr(SORT_TOPK):
            nblk = row_ok.select(nblk, fx.Int32(0))
        nlive = (nblk < i32_width).select(nblk, i32_width)
        # The window loop's trip count: the group's longest row under a sort_topk
        # (the barriers inside must match across the waves), else the row's.
        nloop = nlive
        if const_expr(SORT_TOPK):
            for u in range_constexpr(UQ):
                ru = fx.Int32(g) * UQ + u
                oku = ru < i32_rows
                vu = scalar_at(arg_nvp, oku.select(ru, i32_rows - 1).to(fx.Int64), fx.Int64(4))
                vu = oku.select(vu, fx.Int32(0))
                vu = (vu < i32_width).select(vu, i32_width)
                nloop = (vu > nloop).select(vu, nloop)
        # Later windows read only the row's live blocks.
        srs_live = rsrc(score_row, nlive.to(fx.Int64) * 4)

        def view(ptr, n_dw, width):
            return fx.logical_divide(
                fx.make_view(fx.recast_iter(fx.Int32, ptr), fx.make_layout(n_dw, 1)),
                fx.make_layout(width, 1),
            )

        pairs = view(smem.cand.ptr, WAVES * 2 * CAND, 2)
        quads = view(smem.cand.ptr, WAVES * 2 * CAND, 4)
        sels = view(smem.sel.ptr, WAVES * 4 * TOPK, 4)
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
            lds_put(atom64, fx.slice(pairs, (None, c_off + pos)), 2, [i, k])

        def wave_sum(c):
            """Sum over the wave of a per-lane count in [0, 63]."""
            t = fx.Int32(0)
            for b in range_constexpr(6):
                t = t + (_popc(_ballot((c & (1 << b)) != 0)) << b)
            return t

        def compact(keys, wbase, base, pred):
            """Append the keys `pred` keeps at `base` if they fit below CAP."""
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
            # On overflow nothing is written; the caller rebuilds the candidates.
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

        def lane_max_bound(m, d1=12, d0=0, key=INT_MIN):
            """The 16th largest lane maximum m, to its top d1 bits (from d0 on,
            above the bound `key` of the top d0)."""
            p = fx.Int32(key ^ INT_MIN)
            for d in range_constexpr(d0, d1):
                t = p | _s32(1 << (31 - d))
                count = _popc(_ballot(m >= (t ^ INT_MIN)))
                p = (count >= TOPK).select(t, p)
            return p ^ INT_MIN

        def window(bits, wbase, nrun, tprev):
            """One window of keys merged into the running top 16 (sel)."""
            keys = []
            for j in range_constexpr(STRIPS):
                live = (wbase + (j * 64) + lane) < nlive
                keys.append(live.select(_key(bits[j]), fx.Int32(INT_MIN)))
            m = keys[0]
            for j in range_constexpr(1, STRIPS):
                m = _imax(m, keys[j])
            # Bound: the lane maxima in the first window, the running 16th later.
            tw = fx.Int32(INT_MIN)
            if nrun < TOPK:
                tw = lane_max_bound(m)
            ta = _imax(tw, tprev)
            total = compact(keys, wbase, nrun, lambda k, i: k >= ta)
            # Exact only if 16 keys above -inf bound it and no NaN is above +inf.
            nan_hi = _ballot(m > KEY_POS_INF) != fx.Int64(0)
            if (total > CAP) | (ta <= KEY_NEG_INF) | nan_hi:
                wl = nlive - wbase
                if const_expr(MW):
                    # A wave's windows past its row's end (sort_topk) are empty.
                    wl = (wl < 0).select(fx.Int32(0), wl)
                if wl <= CAP - nrun:
                    # Every live block of the window is a candidate.
                    put_pair(
                        (lane < wl).select(nrun + lane, TRASH + lane),
                        wbase + lane,
                        _fold_nan(keys[0]),
                    )
                    total = nrun + wl
                else:
                    # NaNs to -inf and re-bound; exact search if still too many.
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
            fx.gpu.barrier()
            mi, mk = lds_get(atom64, fx.slice(pairs, (None, c_off + lane)), 2)
            mw = _comp(mk, mi)
            rank = fx.Int32(0)
            for g in range_constexpr(CAP // RANK_BLOCK):
                if total > RANK_BLOCK * g:
                    es = [
                        lds_get(
                            atom128,
                            fx.slice(
                                quads, (None, c_off // 2 + RANK_BLOCK // 2 * g + u)
                            ),
                            4,
                        )
                        for u in range_constexpr(RANK_BLOCK // 2)
                    ]
                    # All of a block's reads in flight before the compares.
                    fx.rocdl.sched_barrier(0)
                    for e in es:
                        rank = rank + _one(_comp(e[1], e[0]) > mw)
                        rank = rank + _one(_comp(e[3], e[2]) > mw)
            if (lane < total) & (rank < TOPK):
                lds_put(
                    atom128, fx.slice(sels, (None, s_off + rank)), 4, [mi, mk, 0, mi]
                )
            return (total < TOPK).select(total, fx.Int32(TOPK))

        words = view(smem.cand.ptr, WAVES * 2 * CAND, 1)
        atom32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
        BIGP = 0x7FFFFFFF  # above every page

        def sort_topk_epilogue(blk):
            """The group's page sort_topk, as aiter's msa_sort_topk_kernel builds it:
            candidate c = (row c // 16, pick c % 16); uniq = the distinct pages
            of the valid picks, ascending; tokblk[slot, row] = the row's block
            on that page or -1; pos = the rows' positions (head 0).

            LDS (the candidate area, free once every wave passed its last
            window), in int32 words: [0, 2C) (block, page) of each candidate
            (page -1: not valid), [2C, 4C) (first page, -) of each candidate
            (BIGP: not the first of its page), [4C, 4C + C UQ) tokblk. Every
            wave runs the code (no dynamic branch on the wave); only wave 0
            writes past [0, 2C) and to global memory."""
            # Each wave its own row's 16 candidates, pages looked up.
            q_req = scalar_at(arg_qreq, n64, fx.Int64(4))
            q_pos = row_ok.select(scalar_at(arg_qpos, n64, fx.Int64(4)), fx.Int32(-1))
            valid = (lane < TOPK) & row_ok & (blk >= 0) & (blk * BLK <= q_pos)
            bt_rs = rsrc(arg_bt, i32_bt_numel.to(fx.Int64) * 4)
            pidx = valid.select(q_req * i32_bt_stride + blk, fx.Int32(0))
            page = fx.Int32(buffer_ops.buffer_load(bt_rs, pidx, vec_width=1, dtype=fx.Int32))
            page = valid.select(page, fx.Int32(-1))
            if lane < TOPK:
                lds_put(atom64, fx.slice(pairs, (None, wv * TOPK + lane)), 2, [blk, page])
            fx.gpu.barrier()
            w0 = wv == 0
            mine = lane < C
            cp = lds_get(atom64, fx.slice(pairs, (None, lane & (C - 1))), 2)
            cb, pg = cp[0], cp[1]
            ok = mine & (pg >= 0)
            # first of its page: no earlier candidate on it (two per 128-bit read)
            dup = fx.Int32(0)
            for j in range_constexpr(0, C, 2):
                e = lds_get(atom128, fx.slice(quads, (None, j // 2)), 4)
                dup = dup | _one((e[1] == pg) & (lane > j)) | _one((e[3] == pg) & (lane > j + 1))
            first = ok & (dup == 0)
            if mine & w0:
                lds_put(atom64, fx.slice(pairs, (None, C + lane)), 2,
                        [first.select(pg, fx.Int32(BIGP)), fx.Int32(0)])
            fx.gpu.barrier()
            # slot: the distinct valid pages below this one
            slot = fx.Int32(0)
            for j in range_constexpr(0, C, 2):
                e = lds_get(atom128, fx.slice(quads, (None, C // 2 + j // 2)), 4)
                slot = slot + _one(e[0] < pg) + _one(e[2] < pg)
            cnt = _popc(_ballot(first))
            base = h64 * i32_groups.to(fx.Int64) + fx.Int32(g).to(fx.Int64)
            u_rs = rsrc(arg_uniq + base * (C * 4), fx.Int64(C * 4))
            buffer_ops.buffer_store(pg, u_rs, slot, mask=first & w0)
            c_rs = rsrc(arg_count + base * 4, fx.Int64(4))
            buffer_ops.buffer_store(cnt, c_rs, fx.Int32(0), mask=(lane == 0) & w0)
            # tokblk[slot, row]: -1, then each valid candidate's block
            TB = C * UQ
            for k in range_constexpr((TB + 63) // 64):
                if (lane + 64 * k < TB) & w0:
                    lds_put(atom32, fx.slice(words, (None, 4 * C + lane + 64 * k)), 1, [fx.Int32(-1)])
            fx.gpu.barrier()
            if ok & w0:
                lds_put(atom32, fx.slice(words, (None, 4 * C + slot * UQ + (lane >> 4))), 1, [cb])
            fx.gpu.barrier()
            t_rs = rsrc(arg_tokblk + base * (TB * 4), fx.Int64(TB * 4))
            for k in range_constexpr((TB + 63) // 64):
                e = lane + 64 * k
                v = lds_get(atom32, fx.slice(words, (None, 4 * C + (e & (TB - 1)))), 1)[0]
                buffer_ops.buffer_store(v, t_rs, e, mask=(e < TB) & w0)
            # each row's position (head 0)
            o_rs = rsrc(arg_upos, i32_rows.to(fx.Int64) * 4)
            buffer_ops.buffer_store(q_pos, o_rs, fx.Int32(n_ld), mask=(lane == 0) & row_ok
                                    & (fx.Int32(h) == 0))

        # Every wave runs the same trip count (the group's longest row), so the
        # barriers match.
        gbase = fx.Int32(0)
        nrun = fx.Int32(0)
        tprev = fx.Int32(INT_MIN + 1)  # dead keys are INT_MIN
        while (gbase == 0) | (gbase < nloop):
            wbase = gbase
            if gbase > 0:
                bits = load_window(srs_live, wbase)
                # The running top 16 back into cand; its 16th key bounds the window.
                fx.gpu.barrier()
                r = lds_get(
                    atom128, fx.slice(sels, (None, s_off + (lane & (TOPK - 1)))), 4
                )
                if lane < nrun:
                    put_pair(lane, r[0], r[1])
                k15 = fx.Int32(fx.rocdl.readlane(T.i32, r[1], fx.Int32(TOPK - 1)))
                tprev = (nrun >= TOPK).select(k15, fx.Int32(INT_MIN + 1))
                fx.gpu.barrier()
            nrun = window(
                [fx.Int32(bits[j]) for j in range_constexpr(STRIPS)],
                wbase,
                nrun,
                tprev,
            )
            gbase = gbase + WIN

        out_rs = rsrc(arg_topk + (h64 * i64_t_head + n64 * i64_t_row) * 4, TOPK * 4)
        fx.gpu.barrier()
        r = lds_get(atom128, fx.slice(sels, (None, s_off + (lane & (TOPK - 1)))), 4)
        blk = (lane < nrun).select(r[0], fx.Int32(-1))
        if const_expr(SORT_TOPK):
            buffer_ops.buffer_store(blk, out_rs, lane, mask=(lane < TOPK) & row_ok)
            sort_topk_epilogue(blk)
        else:
            buffer_ops.buffer_store(blk, out_rs, lane, mask=lane < TOPK)

    @flyc.jit
    def launch(
        arg_score: fx.Int64,
        i64_s_head: fx.Int64,
        i64_s_row: fx.Int64,
        i32_width: fx.Int32,
        arg_nvp: fx.Int64,
        arg_topk: fx.Int64,
        i64_t_head: fx.Int64,
        i64_t_row: fx.Int64,
        i32_rows: fx.Int32,
        i32_heads: fx.Int32,
        i32_groups: fx.Int32,
        arg_bt: fx.Int64,
        i32_bt_stride: fx.Int32,
        i32_bt_numel: fx.Int32,
        arg_qreq: fx.Int64,
        arg_qpos: fx.Int64,
        arg_uniq: fx.Int64,
        arg_count: fx.Int64,
        arg_tokblk: fx.Int64,
        arg_upos: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            arg_score,
            i64_s_head,
            i64_s_row,
            i32_width,
            arg_nvp,
            arg_topk,
            i64_t_head,
            i64_t_row,
            i32_rows,
            i32_groups,
            arg_bt,
            i32_bt_stride,
            i32_bt_numel,
            arg_qreq,
            arg_qpos,
            arg_uniq,
            arg_count,
            arg_tokblk,
            arg_upos,
        ).launch(
            grid=(fx.Int64(i32_groups), fx.Int64(i32_heads), 1),
            block=(NTHR, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
