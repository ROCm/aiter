# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3 prefill index top-k fused with the MSA page sort_topk, Gluon, gfx950.

fused_m3_index_topk_prefill_indices_sort_topk_gfx950_gluon_kernel: one program
per (TQ consecutive query rows, index head), one warp per row, gives each row's
top-16 blocks in AITER's pa_sparse_block_topk order (as vLLM's FlyDSL
m3_index_topk_prefill_indices) and, TQ > 1, the row group's page sort_topk
(as aiter's msa_sort_topk_kernel), bit for bit.

The top-k is FlyDSL's bound-and-compact walk. Each window of STRIPS x 64
blocks (lane l holds blocks w0 + 64 j + l):
  1. folded keys: bits ^ ((bits >> 31) & 0x7FFFFFFF), NaNs to -inf's key,
     dead (past the row) INT_MIN;
  2. a bound ta no new top-16 key is below: the running 16th key once the row
     has 16, before that the 16th largest lane maximum (to 12 bits);
  3. the keys >= ta (at most CAPW, else the window's exact top 16 by a bit
     search) compacted by a prefix scan and scattered into LDS behind the
     running 16;
  4. every candidate ranked against all (key, block) pairs; ranks < 16 are the
     new running top 16, in order.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

BLK = gl.constexpr(128)
TOPK = gl.constexpr(16)
CAPW = gl.constexpr(48)  # window candidates per pass (LDS slots [16, 64))
NRANK = gl.constexpr(64)  # ranked slots: 16 running + CAPW window
NCAND = gl.constexpr(128)  # LDS slots per row: NRANK + 64 discard slots
INT_MIN = gl.constexpr(-2147483648)
KEY_POS_INF = gl.constexpr(0x7F800000)
KEY_NEG_INF = gl.constexpr(-2139095041)  # key of -inf (0xFF800000)
BIG = gl.constexpr(0x7FFFFFFF)


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _count(x):
    """Per row: set elements of a [TQ, S, 64] bool."""
    return gl.sum(gl.sum(x.to(gl.int32), axis=2), axis=1)


@gluon.jit
def _lane_bound(key, R3: gl.constexpr):
    """Per row: the 16th largest lane maximum of key [TQ, S, 64], to its top
    12 bits (rounded down)."""
    m = gl.max(key, axis=1).to(gl.int64) + 2147483648  # [TQ, 64], unsigned order
    p = gl.max(m, axis=1) * 0
    for d in gl.static_range(12):
        t = p + (1 << (31 - d))
        cnt = gl.sum((m >= t[:, None]).to(gl.int32), axis=1)
        p = gl.where(cnt >= TOPK, t, p)
    return gl.convert_layout((p - 2147483648).to(gl.int32), R3)


@gluon.jit
def _exact_top(key, idx, live, R3: gl.constexpr):
    """The window's exact top 16 by (key, block): v = the 16th largest key,
    ie = the block cut among the keys equal to v."""
    v64 = key.to(gl.int64) + 2147483648  # dead: 0
    p = _count(live).to(gl.int64) * 0
    for d in gl.static_range(32):
        t = p + (1 << (31 - d))
        p = gl.where(_count(v64 >= t[:, None, None]) >= TOPK, t, p)
    v = p[:, None, None]
    need = TOPK - _count(v64 > v)
    ie = _count(live) * 0
    for b in gl.static_range(13):
        t = ie + (1 << (12 - b))
        c = _count((v64 == v) & (idx >= t[:, None, None]) & live)
        ie = gl.where(c >= need, t, ie)
    return live & ((v64 > v) | ((v64 == v) & (idx >= ie[:, None, None])))


_fused_m3_index_topk_prefill_indices_sort_topk_gfx950_gluon_kernel_repr = make_kernel_repr(
    "fused_m3_index_topk_prefill_indices_sort_topk_gfx950_gluon_kernel",
    ["TQ", "STRIPS", "SORT_TOPK"],
)


@gluon.jit(repr=_fused_m3_index_topk_prefill_indices_sort_topk_gfx950_gluon_kernel_repr)
def fused_m3_index_topk_prefill_indices_sort_topk_gfx950_gluon_kernel(
    score_ptr,  # fp32 [H, rows, >= width]
    nvp_ptr,  # int32 [rows]: row n's visible blocks
    topk_ptr,  # int32 [H, rows, 16] out
    block_table_ptr,  # sort_topk: int32 [reqs, max_blocks]
    query_req_ptr,  # sort_topk: int32 [rows]
    query_pos_ptr,
    uniq_ptr,  # sort_topk outputs, as msa_sort_topk_kernel
    count_ptr,
    tokblk_ptr,
    pos_ptr,
    num_rows,
    width,
    stride_s_h,
    stride_s_t,
    stride_tk_h,
    stride_tk_t,
    stride_bt,
    TQ: gl.constexpr,
    STRIPS: gl.constexpr,
    SORT_TOPK: gl.constexpr,
):
    WIN: gl.constexpr = STRIPS * 64
    # window keys: lane l, register j holds block w0 + 64 j + l; warp t row t
    L3: gl.constexpr = gl.BlockedLayout([1, STRIPS, 1], [1, 1, 64], [TQ, 1, 1], [2, 1, 0])
    R3: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, L3))  # [TQ]
    # running top 16 of each row (lanes 16.. replicate)
    LB: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [TQ, 1], [1, 0])
    # ranking: candidate e on lanes, every candidate f in registers
    LP: gl.constexpr = gl.BlockedLayout([1, 1, NRANK], [1, 64, 1], [TQ, 1, 1], [2, 1, 0])
    LE: gl.constexpr = gl.SliceLayout(2, LP)
    LF: gl.constexpr = gl.SliceLayout(1, LP)
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])

    grp = gl.program_id(0)
    h = gl.program_id(1)
    t3 = gl.arange(0, TQ, layout=R3)
    row3 = grp * TQ + t3
    ok3 = row3 < num_rows
    nlive = gl.minimum(gl.load(nvp_ptr + row3, mask=ok3, other=0), width)
    n_max = gl.max(nlive, axis=0)
    s_row = score_ptr + h.to(gl.int64) * stride_s_h + row3.to(gl.int64) * stride_s_t
    j3 = gl.arange(0, STRIPS, layout=gl.SliceLayout(0, gl.SliceLayout(2, L3)))[None, :, None]
    l3 = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(1, L3)))[None, None, :]
    zero3 = gl.zeros([TQ, STRIPS, 64], gl.int32, L3)
    trash = (NRANK + l3) + zero3  # discard slot of each lane

    cand_k = gl.allocate_shared_memory(gl.int32, [TQ, NCAND], SH)
    cand_i = gl.allocate_shared_memory(gl.int32, [TQ, NCAND], SH)
    best_k_s = gl.allocate_shared_memory(gl.int32, [TQ, NCAND], SH)
    best_i_s = gl.allocate_shared_memory(gl.int32, [TQ, NCAND], SH)

    jb = gl.arange(0, TOPK, layout=gl.SliceLayout(0, LB))[None, :] + gl.zeros(
        [TQ, TOPK], gl.int32, LB
    )
    dead_k = gl.full([TQ, TOPK], INT_MIN, gl.int32, LB)
    dead_i = gl.full([TQ, TOPK], -1, gl.int32, LB)
    best_k = dead_k
    best_i = dead_i
    se = gl.arange(0, NRANK, layout=gl.SliceLayout(0, LE))[None, :]
    sf = gl.arange(0, NRANK, layout=gl.SliceLayout(0, LF))[None, :]
    le = gl.arange(0, NRANK, layout=gl.SliceLayout(0, LE))[None, :] + gl.zeros(
        [TQ, NRANK], gl.int32, LE
    )

    for w0 in range(0, n_max, WIN):
        idx = w0 + j3 * 64 + l3 + zero3
        live = idx < nlive[:, None, None]
        bits = gl.load(s_row[:, None, None] + idx, mask=live, other=0.0).to(
            gl.int32, bitcast=True
        )
        key = bits ^ ((bits >> 31) & 0x7FFFFFFF)
        key = gl.where((key > KEY_POS_INF) | (key < KEY_NEG_INF), KEY_NEG_INF, key)
        key = gl.where(live, key, INT_MIN)

        # the bound: the running 16th key, or (row not full) the lane bound
        k16 = gl.convert_layout(gl.min(best_k, axis=1), R3)
        full = k16 != INT_MIN
        ta = gl.where(full, k16, INT_MIN + 1)
        if gl.min(full.to(gl.int32), axis=0) == 0:
            ta = gl.where(full, ta, gl.maximum(_lane_bound(key, R3), ta))
        sel = key >= ta[:, None, None]
        cnt = _count(sel)
        if gl.max(cnt, axis=0) > CAPW:
            # too loose: the window's exact top 16
            sel = gl.where((cnt > CAPW)[:, None, None], _exact_top(key, idx, live, R3), sel)
            cnt = _count(sel)

        # compact the candidates behind the running 16 (prefix positions)
        si = sel.to(gl.int32)
        lane_cnt = gl.sum(si, axis=1)
        # (the scan runs on a blocked layout: lanes along dim 1)
        lc = gl.convert_layout(lane_cnt, LB)
        lane_base = gl.convert_layout(
            gl.associative_scan(lc, 1, _add) - lc, gl.SliceLayout(1, L3)
        )
        within = gl.associative_scan(si, 1, _add) - si
        slot = gl.where(sel, TOPK + lane_base[:, None, :] + within, trash)
        cand_k.scatter(gl.reshape(key, [TQ, WIN]), gl.reshape(slot, [TQ, WIN]), 1)
        cand_i.scatter(gl.reshape(idx, [TQ, WIN]), gl.reshape(slot, [TQ, WIN]), 1)
        cand_k.scatter(best_k, jb, 1)
        cand_i.scatter(best_i, jb, 1)
        best_k_s.scatter(dead_k, jb, 1)
        best_i_s.scatter(dead_i, jb, 1)
        gl.barrier()

        # rank every candidate against all: ranks < 16 are the new top 16
        cs_k = cand_k.slice(0, NRANK, 1)
        cs_i = cand_i.slice(0, NRANK, 1)
        ke = cs_k.load(LE)
        ie = cs_i.load(LE)
        kf = cs_k.load(LF)
        jf = cs_i.load(LF)
        ve = se < (TOPK + gl.convert_layout(cnt, gl.SliceLayout(1, LE))[:, None])
        vf = sf < (TOPK + gl.convert_layout(cnt, gl.SliceLayout(1, LF))[:, None])
        ke = gl.where(ve, ke, INT_MIN)
        ie = gl.where(ve, ie, -1)
        kf = gl.where(vf, kf, INT_MIN)
        jf = gl.where(vf, jf, -1)
        gt = (kf[:, None, :] > ke[:, :, None]) | (
            (kf[:, None, :] == ke[:, :, None]) & (jf[:, None, :] > ie[:, :, None])
        )
        rank = gl.sum(gt.to(gl.int32), axis=2)
        dst = gl.where((rank < TOPK) & (ke != INT_MIN), rank, NRANK + le)
        best_k_s.scatter(ke, dst, 1)
        best_i_s.scatter(ie, dst, 1)
        gl.barrier()
        best_k = best_k_s.slice(0, TOPK, 1).load(LB)
        best_i = best_i_s.slice(0, TOPK, 1).load(LB)
        gl.barrier()

    blk = gl.where(best_k == INT_MIN, -1, best_i)
    rowb = grp * TQ + gl.arange(0, TQ, layout=gl.SliceLayout(1, LB))
    gl.store(
        topk_ptr + h.to(gl.int64) * stride_tk_h + rowb.to(gl.int64)[:, None] * stride_tk_t + jb,
        blk,
        mask=(rowb < num_rows)[:, None] & (jb >= 0),
    )

    if SORT_TOPK:
        # The group's page sort_topk, as msa_sort_topk_kernel builds it (candidate
        # c = row c // 16, pick c % 16): uniq = the distinct valid pages,
        # ascending; tokblk[slot, row] = the row's block on that page or -1.
        C: gl.constexpr = TQ * TOPK
        NW: gl.constexpr = 64 * TQ
        LU: gl.constexpr = gl.BlockedLayout([1, C], [64, 1], [TQ, 1], [1, 0])  # [C e, C f]
        UE: gl.constexpr = gl.SliceLayout(1, LU)
        UF: gl.constexpr = gl.SliceLayout(0, LU)
        SH1: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
        u_blk2 = gl.allocate_shared_memory(gl.int32, [TQ, TOPK], SH)
        u_pg = gl.allocate_shared_memory(gl.int32, [C], SH1)
        u_fp = gl.allocate_shared_memory(gl.int32, [C], SH1)
        u_sl = gl.allocate_shared_memory(gl.int32, [C], SH1)
        u_blk2.store(blk)
        u_blk = u_blk2.reshape([C])
        gl.barrier()
        c = gl.arange(0, C, layout=UE)
        cb = u_blk.load(UE)
        row_c = grp * TQ + c // TOPK
        rc_ok = row_c < num_rows
        req = gl.load(query_req_ptr + row_c, mask=rc_ok, other=0).to(gl.int64)
        pos = gl.load(query_pos_ptr + row_c, mask=rc_ok, other=-1)
        pos = gl.where(rc_ok, pos, -1)
        valid = rc_ok & (cb >= 0) & (cb * BLK <= pos)
        page = gl.load(
            block_table_ptr + req * stride_bt + gl.where(valid, cb, 0), mask=valid, other=-1
        )
        page = gl.where(valid, page, -1)
        u_pg.store(page)
        gl.barrier()
        pf = u_pg.load(UF)
        cf = gl.arange(0, C, layout=UF)
        dup = gl.max(((pf[None, :] == page[:, None]) & (cf[None, :] < c[:, None])).to(gl.int32), axis=1)
        first = valid & (dup == 0)
        u_fp.store(gl.where(first, page, BIG))
        gl.barrier()
        fpf = u_fp.load(UF)
        slot = gl.sum((fpf[None, :] < page[:, None]).to(gl.int32), axis=1)
        nfirst = gl.sum(first.to(gl.int32), axis=0)
        base = (h * gl.num_programs(0) + grp).to(gl.int64)
        gl.store(uniq_ptr + base * C + slot, page, mask=first)
        gl.store(count_ptr + base + c * 0, nfirst + c * 0, mask=c == 0)
        u_sl.store(gl.where(valid, slot, -1))
        gl.barrier()
        # tokblk[s, t] (entry e = s * TQ + t): the block of row t's candidate in slot s
        E: gl.constexpr = C * TQ
        LT: gl.constexpr = gl.BlockedLayout([1, C], [64, 1], [TQ, 1], [1, 0])  # [E, C]
        e = gl.arange(0, E, layout=gl.SliceLayout(1, LT))
        slf = u_sl.load(gl.SliceLayout(0, LT))
        cbf = u_blk.load(gl.SliceLayout(0, LT))
        tf = gl.arange(0, C, layout=gl.SliceLayout(0, LT)) // TOPK
        hit = (slf[None, :] == (e // TQ)[:, None]) & (tf[None, :] == (e % TQ)[:, None])
        val = gl.max(gl.where(hit, cbf[None, :], -1), axis=1)
        gl.store(tokblk_ptr + base * E + e, val)
        if h == 0:
            gl.store(pos_ptr + row_c, pos, mask=rc_ok & (c % TOPK == 0))
