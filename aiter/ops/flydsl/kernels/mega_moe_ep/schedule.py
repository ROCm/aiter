# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""In-kernel dispatch planner for the EP MegaMoE kernel.

Ported from the TP dynamic planner (:mod:`mega_moe_tp.schedule` ``dyn_plan``) and
adapted to expert parallelism:

* the active-expert space is this rank's ``EPR`` *local* experts (bitmap ``L_DYN``,
  per-expert counts ``L_DCNT``), not the whole replicated expert set;
* the routing scanned is the dispatched ``(token, expert)`` pairs this rank *received*
  -- ``a["recv_le"]`` (received pair -> local expert id) over ``a["nrecv"]`` pairs --
  which ``dispatch_send`` fills, not the all-gathered routing;
* a local expert routed by ``R`` received tokens runs as ``ceil(R / rch)`` row chunks,
  each in ``P`` equal inter pieces (``P`` chosen to balance ``#CU`` CTAs), the unit list
  sliced per CTA ``[u_lo, u_hi)`` XCD-aware -- identical to TP.

The GEMM1+act+GEMM2 body (``unit_tile``) and the per-expert received-row gather
(``gather_recv_chunk``) are stubs filled by the compute / dispatch milestones; the
planner only needs the per-expert row counts the route scan already wrote to ``L_DCNT``.
"""

from __future__ import annotations

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ..kernels_common import ceildiv
from .common import (
    ACQ_REL,
    _ctpop,
    bld,
    bst,
    REL,
    dma4,
    dma16,
    g_add_agent,
    g_st_sys,
    i32,
    lds_atomic_add,
    lds_atomic_or,
    lds_ld_acq,
    lds_ld_i32,
    lds_st,
    lds_st_rel,
    now,
    rsrc,
    spin0,
    spin_lds_ge,
    wait_lgkm0,
    wait_vm,
    wave_rank,
)

# Device-phase profiling (matches mega_moe_ep_kernel._PROF): slots 4,5 are stamped in
# compute_units (post dispatch_wait / post dyn_plan) to split the `roles` phase.
_PROF = bool(int(os.environ.get("EP_PROF_DEV", "0")))
_PROF_SLOTS = 19  # +7 post-gemm1, +8 post-gemm2, +9/10/11 (see mega_moe_ep_kernel)
from .mega_moe_ep_config import (
    C_BARCNT,
    C_BARGEN,
    C_CNT,
    C_DYNP,
    C_ELAST,
    C_K1,
    C_K2,
    C_S1,
    C_S2,
    BAL,
    C_NACT,
    C_NCH,
    C_PLAN,
    C_UL,
    C_UNIT,
    C_UROWS,
    C_USEQ,
    ECNT1,
    ECNT2,
    ERR_COMPUTE,
    ESYNC,
    NAB,
    N_XCD,
    XCDMAP,
    NLOAD,
    RC_AFREE,
    RC_ASEQ,
    RC_G1GO,
    RC_G2GO,
    RC_WFREE,
    RC_WSEQ,
    UL_MAX,
    KernelCtx,
)

__all__ = ["build_schedule"]


def build_schedule(kc: KernelCtx) -> dict:
    """The dispatch-planner device functions of one instance (added to kc)."""
    DYN_PS, EPR, I, KS2, TOPK = kc.get("DYN_PS EPR I KS2 TOPK")
    L_CTL, L_DCH, L_DCNT, L_DYN = kc.get("L_CTL L_DCH L_DCNT L_DYN")
    L_EOFF, L_RIX, L_WSUM, NBW = kc.get("L_EOFF L_RIX L_WSUM NBW")
    L_RCTL = kc.get("L_RCTL")[0]
    const_expr, rch, N_TILES = kc.get("const_expr rch N_TILES")
    N_TILES2 = kc.get("N_TILES2")[0]
    # GEMM1/GEMM2 + the per-unit A loaders (gemm.build_gemm) and the combine epilogue
    # (communication.build_communication) are added to kc before build_schedule.
    a_loader_g1, a_loader_g2 = kc.get("a_loader_g1 a_loader_g2")
    gemm1, gemm2 = kc.get("gemm1 gemm2")
    gemm2_combine = kc.get("gemm2_combine")[0]
    # NW=8 (all-8-waves-compute, serial-comm) variants: inline-A GEMMs + the a2a send done
    # by the compute waves themselves (no concurrent dispatch/loader band).
    gemm1_nw8, gemm2_pipe_nw8, gemm2_combine_nw8 = kc.get(
        "gemm1_nw8 gemm2_pipe_nw8 gemm2_combine_nw8"
    )
    dispatch_send = kc.get("dispatch_send")[0]
    combine_send = kc.get("combine_send")[0]
    dispatch_wait = kc.get("dispatch_wait")[0]
    dispatch_wait_meta = kc.get("dispatch_wait_meta")[0]
    grid_bar = kc.get("grid_bar")[0]
    ctrl_at, spin_sys_ge = kc.get("ctrl_at spin_sys_ge")
    L_PF_ = kc.get("L_PF")[0]
    NW, NT, B_LOADER = kc.get("NW NT B_LOADER")  # per-launch warp-spec width (not globals)
    assert L_EOFF is not None and const_expr  # L_EOFF: gather offsets (future)

    @flyc.jit
    def cbar(L, tid):
        """The four compute waves' LDS generation barrier (TP ``communication.cbar``)."""
        wait_lgkm0()
        if (tid % i32(64)) == i32(0):
            g = lds_ld_acq(L, L_CTL + C_BARGEN * 4)
            old = lds_atomic_add(L, L_CTL + C_BARCNT * 4, 1, ACQ_REL)
            if old == i32(NW - 1):
                lds_st(L, L_CTL + C_BARCNT * 4, i32(0))
                lds_st_rel(L, L_CTL + C_BARGEN * 4, g + i32(1))
            else:
                cur = lds_ld_acq(L, L_CTL + C_BARGEN * 4)
                while cur == g:
                    rocdl.s_sleep(0)
                    cur = lds_ld_acq(L, L_CTL + C_BARGEN * 4)
        rocdl.sched_barrier(0)

    @flyc.jit
    def gbar(L, tid, a, P):
        """Barrier between GEMM passes, wrapped for the warp-specialized block.

        When ``P == 1`` every echunk's four passes run on the SAME CTA (echunk == bid), so
        the shared global round-trips (rowmap / inter / gout) are same-CTA: ``wait_vm``
        orders write-before-read and ``cbar`` syncs the compute waves -- the costly
        cross-CTA handshake + L1 flush/invalidate are SKIPPED (so a high-``nch`` launch that
        cannot shard pays no multi-pass overhead).

        When ``P > 1`` the n-tile shards of an echunk live on DIFFERENT CTAs, so an
        agent-scope *release* fence writes this CU's L1 back to the device-wide L2, the
        tid0-only ``grid_bar`` is the cross-CTA handshake (no block barrier -- the
        dispatch/idle bands are elsewhere), and an agent-scope *acquire* fence then
        invalidates this CU's L1 so the next pass reads the other shards' fresh data."""
        wait_vm(0)
        cbar(L, tid)
        if P > i32(1):
            fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Release)
            grid_bar(a, tid)
            cbar(L, tid)
            fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Acquire)

    @flyc.jit
    def echunk_done(L, tid, a, slot):
        """EP_ESYNC (P>1): this CTA's n-tile shard of an echunk pass is written -- release
        it device-wide (same L2 writeback as gbar) and add 1 to ctrl[slot]; the pre-add
        count lands in LDS C_ELAST for every compute wave (== P-1: this was the last)."""
        wait_vm(0)
        fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Release)
        cbar(L, tid)
        if tid == i32(0):
            lds_st(L, L_CTL + C_ELAST * 4, g_add_agent(ctrl_at(a, slot), i32(1)))
        cbar(L, tid)

    @flyc.jit
    def echunk_wait(a, lane0, slot, P):
        """EP_ESYNC (P>1): block until all P shards of an echunk pass published, then
        acquire (L1 invalidate) so this wave reads their data.  ``lane0`` polls; the
        caller syncs the rest of its waves."""
        if lane0:
            spin_sys_ge(ctrl_at(a, slot), P, a, ERR_COMPUTE)

    def unit_fields(L, u):
        f = [
            lds_ld_i32(L, L_CTL + (i32(C_UL) + u * i32(4) + i32(k)) * i32(4))
            for k in range(4)
        ]
        return f[0], f[1], f[2], f[3]

    def masked(le):
        # invalid / foreign slot: le >= EPR (and -1 wraps to a huge unsigned value)
        return fx.Int32(le).bitcast(fx.Uint32) >= fx.Uint32(EPR)

    def _wave_prefix(v, bits):
        pre, tot = i32(0), i32(0)
        for b in range_constexpr(bits):
            below, cnt = wave_rank(((v >> i32(b)) & i32(1)) == i32(1))
            pre = pre + (below << i32(b))
            tot = tot + (cnt << i32(b))
        return pre, tot

    def _pick_p(nun, C, ps):
        # pick the GEMM1 inter-piece count that best fills the CTAs within UL_MAX units
        P = i32(ps[0])
        best = ceildiv(nun * i32(ps[0]), C) * i32(KS2 // ps[0])
        for q in ps[1:]:
            cost = ceildiv(nun * i32(q), C) * i32(KS2 // q)
            better = (cost < best) & (nun * i32(q) <= C * i32(UL_MAX))
            P = better.select(i32(q), P)
            best = better.select(cost, best)
        return P

    def _xcd_rank(C):
        bid = i32(gpu.block_id("x"))
        return (C % i32(N_XCD) == i32(0)).select(
            (bid % i32(N_XCD)) * (C // i32(N_XCD)) + bid // i32(N_XCD), bid
        )

    def _walk_start(U, C):
        """First item of this CTA's stride-C walk over U echunks / units (every walk of
        one item kind -- compute, both loader bands -- must agree; P==1 relies on one CTA
        owning all passes of an echunk).  XCDMAP, one round (U <= C): XCD x's CTAs take
        the contiguous block [x*Q, x*Q + Q), Q = ceil(U / N_XCD) -- an expert's echunks
        share an XCD and the idle CTAs spread evenly over the XCDs; more rounds: the
        XCD-major rank."""
        bid = i32(gpu.block_id("x"))
        if not XCDMAP:
            return bid
        x = bid % i32(N_XCD)
        k = bid // i32(N_XCD)
        Q = ceildiv(U, i32(N_XCD))
        one = (k < Q).select(x * Q + k, U)
        return (C % i32(N_XCD) == i32(0)).select((U <= C).select(one, _xcd_rank(C)), bid)

    # ---- NW=4 GEMM pass split -------------------------------------------------------
    # A pass over NTn n-tiles per echunk is cut into shards of S n-tiles: q = NTn // S
    # full shards per echunk (one CTA item each) + the r = NTn % S remainder shards,
    # packed K per CTA item.  Items: [0, uf) full (echunk t//q, shard t%q), [uf, uc)
    # packs (echunks (t-uf)*K + j, j < K, n-tiles [q*S, NTn)).  Every echunk then has
    # Pe = q + (r > 0) shards -- the ESYNC completion target.  S / K are planned once
    # (_bal_plan) and every band (compute, A-loader, W-loader) re-derives the walk.
    def _phase_split(nch, C, NTn, P_leg):
        if not BAL:  # the legacy P-way split's largest shard, remainders unpacked
            return ceildiv(i32(NTn), P_leg), i32(1)
        S, K = i32(NTn), i32(1)  # no S fits (nch > C): whole echunks, multi-round walk
        for i in range_constexpr(NTn):
            cand = NTn - i  # descending: the smallest fitting S wins
            q, r = NTn // cand, NTn % cand
            k = cand // r if r else 1
            need = nch * i32(q) + (ceildiv(nch, i32(k)) if r else i32(0))
            ok = need <= C
            S = ok.select(i32(cand), S)
            K = ok.select(i32(k), K)
        return S, K

    def _phase(L, nch, NTn, S, K):
        q = NTn // S
        r = NTn - q * S
        uf = nch * q
        rem = r > i32(0)
        uc = uf + rem.select(ceildiv(nch, K), i32(0))
        return q, r, uf, uc, q + rem.select(i32(1), i32(0))

    def _unit(t, j, rnd, S, K, q, r, uf, NTn):
        """(echunk, nt_lo, nt_hi, qb) of slot j of item t (round rnd of a multi-round
        walk); qb = n-tiles the CTA ran earlier in the pass (ring-tile base)."""
        full = t < uf
        eq = t // q
        eg = full.select(eq, (t - uf) * K + j)
        lo = full.select((t - eq * q) * S, q * S)
        hi = full.select(lo + S, NTn)
        qb = full.select(rnd * S, j * r)
        return eg, lo, hi, qb

    def _nslots(t, uf, K):
        return (t < uf).select(i32(1), K)

    def _plan_ld(L, ph2):
        cs, ck = (C_S2, C_K2) if ph2 else (C_S1, C_K1)
        return lds_ld_i32(L, L_CTL + cs * 4), lds_ld_i32(L, L_CTL + ck * 4)

    @flyc.jit
    def _dyn_prefix(L, tid):
        if tid == i32(0):
            acc = i32(0)
            for w in range_constexpr(NBW):
                lds_st(L, L_DYN + i32((NBW + w) * 4), acc)
                acc = acc + _ctpop(lds_ld_i32(L, L_DYN + i32(w * 4)))
            lds_st(L, L_CTL + C_NACT * 4, acc)

    @flyc.jit
    def _dyn_list(L, e):
        if e < i32(EPR):
            wv = lds_ld_i32(L, L_DYN + (e >> i32(5)) * i32(4))
            if ((wv >> (e & i32(31))) & i32(1)) != i32(0):
                low = (i32(1) << (e & i32(31))) - i32(1)
                rank = lds_ld_i32(
                    L, L_DYN + (i32(NBW) + (e >> i32(5))) * i32(4)
                ) + _ctpop(wv & low)
                lds_st(L, L_DYN + (i32(2 * NBW) + rank) * i32(4), e)

    @flyc.jit
    def _chunk_table(L, tid):
        nact = lds_ld_i32(L, L_CTL + C_NACT * 4)
        w = tid // i32(64)
        lane = tid % i32(64)
        base = i32(0)
        for rnd in range_constexpr(ceildiv(EPR, NT)):
            j = tid + i32(rnd * NT)
            live = j < nact
            e = live.select(
                lds_ld_i32(
                    L, L_DYN + (i32(2 * NBW) + fx.min(j, i32(EPR - 1))) * i32(4)
                ),
                i32(0),
            )
            r = lds_ld_i32(L, L_DCNT + e * i32(4))
            # Balanced-gather: full RG-row chunks. position//RG is an EXACT partition of
            # the expert's 0..R_e-1 rows, so a chunk never exceeds rch (no row drop) and
            # fills a whole RG MFMA tile (no M-padding). This also halves the chunk count
            # vs the old sk%ce safety margin -> more CU n-tile shard room (higher P).
            ce = fx.min(fx.max(ceildiv(r, i32(rch)), i32(1)), i32(255))
            ce = live.select(ce, i32(0))
            pre, tot = _wave_prefix(ce, 8)
            if lane == i32(0):
                lds_st(L, L_WSUM + w * i32(4), tot)
            cbar(L, tid)
            o, total = base + pre, i32(0)
            for v in range_constexpr(NW):
                sv = lds_ld_i32(L, L_WSUM + i32(v * 4))
                o = o + (i32(v) < w).select(sv, i32(0))
                total = total + sv
            base = base + total
            for c_ in range(i32(0), ce, i32(1)):
                c = i32(c_)
                lds_st(
                    L, L_DCH + (o + c) * i32(4), e | (ce << i32(16)) | (c << i32(24))
                )
            cbar(L, tid)
        if tid == i32(0):
            lds_st(L, L_CTL + C_NCH * 4, base)

    @flyc.jit
    def _dyn_units(L, tid):
        if tid == i32(0):
            nun = lds_ld_i32(L, L_CTL + C_NCH * 4)
            C = i32(gpu.grid_dim.x)
            P = _pick_p(nun, C, DYN_PS)
            U = nun * P
            bid = _xcd_rank(C)
            u_lo = bid * U // C
            u_hi = (bid + i32(1)) * U // C
            icnt = i32(I) // P
            for u_ in range(u_lo, u_hi, i32(1)):
                u = i32(u_)
                j = u // P
                ul = L_CTL + (i32(C_UL) + (u - u_lo) * i32(4)) * i32(4)
                lds_st(L, ul, lds_ld_i32(L, L_DCH + j * i32(4)))
                lds_st(L, ul + i32(4), (u - j * P) * icnt)
                lds_st(L, ul + i32(8), icnt)
                lds_st(L, ul + i32(12), i32(0))
            lds_st(L, L_CTL + C_DYNP * 4, P)
            lds_st(L, L_CTL + C_UNIT * 4, i32(0))
            lds_st(L, L_CTL + (C_UNIT + 1) * 4, u_hi - u_lo)
            lds_st_rel(L, L_CTL + C_PLAN * 4, i32(1))

    @flyc.jit
    def _bal_plan(L, tid):
        if tid == i32(0):
            nch = lds_ld_i32(L, L_CTL + C_NCH * 4)
            C = i32(gpu.grid_dim.x)
            P = fx.max(fx.min(C // fx.max(nch, i32(1)), i32(N_TILES)), i32(1))
            S1, K1 = _phase_split(nch, C, N_TILES, P)
            S2, K2 = _phase_split(nch, C, N_TILES2, P)
            lds_st(L, L_CTL + C_S1 * 4, S1)
            lds_st(L, L_CTL + C_K1 * 4, K1)
            lds_st(L, L_CTL + C_S2 * 4, S2)
            lds_st(L, L_CTL + C_K2 * 4, K2)
            lds_st_rel(L, L_CTL + C_PLAN * 4, i32(1))

    @flyc.jit
    def pre_pos(L, tid, a):
        """Balanced-gather pre-pass (pre-sort): give every received (token,route) pair a
        GLOBAL per-local-expert position (0..R_e-1 rank) via an agent-scope atomic counter
        (``ecnt``, host-zeroed each launch) and bucket its recv slot straight into
        ``rowlist[le*cap + pos]`` (cap = npes*mtpr: a token routes to an expert at most
        once).  Strided across ALL CTAs x threads so each recv slot is visited exactly
        once; the gather then reads a chunk's RG rows by position -- O(RG), no rescan of
        the npes x m x topk window per echunk."""
        del L
        rle = rsrc(a["recv_le"], a["nrecv"] * i32(4))
        cap = a["npes"] * a["mtpr"]
        rlist = rsrc(a["rowlist"], i32(EPR) * cap * i32(4))
        ecnt = fx.Int64(a["ecnt"])
        blk = a["m"] * i32(TOPK)
        stride_p = a["mtpr"] * i32(TOPK)
        base = i32(gpu.block_id("x")) * i32(NT) + tid
        gstride = i32(gpu.grid_dim.x) * i32(NT)
        for c_ in range(base, a["npes"] * blk, gstride):
            c = i32(c_)
            p = c // blk
            idx = p * stride_p + (c - p * blk)
            le = fx.Int32(bld(rle, idx * i32(4), 0, T.i32, 0))
            if masked(le) == fx.Boolean(False):
                pos = g_add_agent(ecnt + fx.Int64(le) * fx.Int64(4), i32(1))
                bst(idx, rlist, (le * cap + pos) * i32(4), 0, 0)

    @flyc.jit
    def dyn_plan(L, tid, a):
        """Per-local-expert bitmap + counts -> active list -> row-chunk table -> per-CTA
        unit list [u_lo, u_hi) (XCD-aware).  The per-expert counts come straight from
        pre_pos's global ``ecnt`` (one value per local expert; pre_pos already did the
        agent-atomic scan of recv_le) -- so dyn_plan reads only EPR ints instead of
        re-scanning the whole npes x m x topk routing window (which otherwise contends for
        HBM with the overlapped payload a2a).  gbar(2) before this publishes ecnt."""
        recnt = rsrc(fx.Int64(a["ecnt"]), i32(EPR) * i32(4))
        for e_ in range(tid, i32(EPR), i32(NT)):
            e = i32(e_)
            cnt = fx.Int32(bld(recnt, e * i32(4), 0, T.i32, 0))
            lds_st(L, L_DCNT + e * i32(4), cnt)  # distinct e per thread: no race
            if cnt > i32(0):
                lds_atomic_or(  # bitmap word shared by 32 experts: atomic
                    L, L_DYN + (e >> i32(5)) * i32(4), i32(1) << (e & i32(31))
                )
        cbar(L, tid)
        _dyn_prefix(L, tid)
        cbar(L, tid)
        for it in range_constexpr(ceildiv(EPR, NT)):
            _dyn_list(L, tid + i32(it * NT))
        cbar(L, tid)
        _chunk_table(L, tid)
        cbar(L, tid)
        _bal_plan(L, tid)
        cbar(L, tid)

    @flyc.jit
    def gather_recv_chunk(L, tid, a, expert, cc, ce):
        """Collect echunk (``expert``, chunk ``cc``)'s payload rows into L_RIX: chunk cc is
        exactly positions [cc*RG, min((cc+1)*RG, R_e)) of the expert's pre-sorted bucket
        ``rowlist[expert*cap + pos]`` (pre_pos) -- a full RG-row tile, deterministic.
        L_RIX stores the recv slot idx (= source_key*topk + route); the GEMM reads payload
        row idx//topk and combine recovers route/source from idx. ``ce`` is unused now."""
        del ce
        cap = a["npes"] * a["mtpr"]
        rlist = rsrc(a["rowlist"], i32(EPR) * cap * i32(4))
        lo = cc * i32(rch)
        R = fx.min(lds_ld_i32(L, L_DCNT + expert * i32(4)) - lo, i32(rch))
        for r_ in range(tid, R, i32(NT)):
            r = i32(r_)
            idx = fx.Int32(bld(rlist, (expert * cap + lo + r) * i32(4), 0, T.i32, 0))
            lds_st(L, L_RIX + r * i32(4), idx)
        cbar(L, tid)
        return R

    @flyc.jit
    def _ring_reset_g2go(L, tid):
        """Reset the A / W rings for GEMM2 + release the loaders' PASS-2 phase (inter is
        written + flushed).  Safe: PASS 1 drained the rings and the loaders park on
        RC_G2GO, so nothing touches them."""
        if tid == i32(0):
            for b in range_constexpr(NAB):
                lds_st(L, L_RCTL + (i32(RC_ASEQ) + i32(b)) * i32(4), i32(0))
                lds_st(L, L_RCTL + (i32(RC_AFREE) + i32(b)) * i32(4), i32(0))
            for b in range_constexpr(4):
                lds_st(L, L_RCTL + (i32(RC_WSEQ) + i32(b)) * i32(4), i32(0))
                lds_st(L, L_RCTL + (i32(RC_WFREE) + i32(b)) * i32(4), i32(0))
            lds_st_rel(L, L_RCTL + RC_G2GO * i32(4), i32(1))
        cbar(L, tid)  # compute waves see the zeroed ring before consuming GEMM2 A

    @flyc.jit
    def _prof_mark(tid, a, slot):
        if const_expr(_PROF):  # noqa: SIM102 - compile-time gate, not dynamic
            if tid == i32(0):
                bst(now(), rsrc(a["dbg"]),
                    (i32(gpu.block_id("x")) * i32(_PROF_SLOTS) + i32(slot)) * i32(8), 0, 0)

    @flyc.jit
    def _gemm_passes_nw4(L, tid, a, nch, nblk, bid_e, S1, K1, q1, r1, uf1, uc1, P1,
                         S2, K2, q2, r2, uf2, uc2, P2):
        """NW=4 GEMM1 -> GEMM2 (+ combine) over the pass splits (_phase / _unit); the
        loader bands walk the identical item / slot / n-tile sequence.  ESYNC: an echunk
        pass's shards publish ECNT1 / ECNT2 (Pe shards each) unless P1 == P2 == 1 (all
        passes of an echunk on one CTA); P2 == 1 runs the fused-combine GEMM2."""
        local = (P1 == i32(1)) & (P2 == i32(1))
        # --- PASS 1 (GEMM1) -> inter[echunk] n-tile slices.
        t1 = _walk_start(uc1, nblk)
        for t_ in range(t1, uc1, nblk):
            t = i32(t_)
            rnd = (t - t1) // nblk
            for j_ in range(i32(0), _nslots(t, uf1, K1), i32(1)):
                eg, lo, hi, qb = _unit(t, i32(j_), rnd, S1, K1, q1, r1, uf1, i32(N_TILES))
                if eg < nch:
                    expert = lds_ld_i32(L, L_DCH + eg * i32(4)) & i32(0xFFFF)
                    gemm1(L, tid, a, expert, eg, lo, hi, qb)
                    if const_expr(ESYNC):  # noqa: SIM102 - compile-time gate, not dynamic
                        if local == fx.Boolean(False):
                            echunk_done(L, tid, a, i32(ECNT1) + eg)
        _prof_mark(tid, a, 7)  # post gemm1
        if const_expr(ESYNC):
            gbar(L, tid, a, i32(1))
        else:
            gbar(L, tid, a, local.select(i32(1), i32(2)))
        _ring_reset_g2go(L, tid)
        # --- PASS 2 (GEMM2) [+ fused combine at P2 == 1: whole echunks per CTA].
        t2 = _walk_start(uc2, nblk)
        if P2 == i32(1):
            for ei_ in range(t2, nch, nblk):
                ei = i32(ei_)
                ec = lds_ld_i32(L, L_DCH + ei * i32(4))
                cc = ec.shrui(i32(24))
                rows = fx.min(
                    lds_ld_i32(L, L_DCNT + (ec & i32(0xFFFF)) * i32(4)) - cc * i32(rch),
                    i32(rch),
                )
                if const_expr(ESYNC):  # noqa: SIM102 - compile-time gate, not dynamic
                    if P1 > i32(1):
                        echunk_wait(a, tid == i32(0), i32(ECNT1) + ei, P1)
                        cbar(L, tid)
                        fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Acquire)
                gemm2_combine(L, tid, a, ec & i32(0xFFFF), ei, rows,
                              (ei - t2) // nblk * i32(N_TILES2))
                if const_expr(ESYNC):  # noqa: SIM102 - compile-time gate, not dynamic
                    # sole consumer (its A-loader waited before staging): reset for reuse
                    if (P1 > i32(1)) & (tid == i32(0)):
                        g_st_sys(ctrl_at(a, i32(ECNT1) + ei), i32(0))
            _prof_mark(tid, a, 8)  # post gemm2
        else:
            for t_ in range(t2, uc2, nblk):
                t = i32(t_)
                rnd = (t - t2) // nblk
                for j_ in range(i32(0), _nslots(t, uf2, K2), i32(1)):
                    eg, lo, hi, qb = _unit(t, i32(j_), rnd, S2, K2, q2, r2, uf2,
                                           i32(N_TILES2))
                    if eg < nch:
                        ec = lds_ld_i32(L, L_DCH + eg * i32(4))
                        expert = ec & i32(0xFFFF)
                        if const_expr(ESYNC):  # noqa: SIM102 - compile-time gate
                            echunk_wait(a, tid == i32(0), i32(ECNT1) + eg, P1)
                            cbar(L, tid)
                            fx.memory_fence(syncscope="agent",
                                            ordering=fx.AtomicOrdering.Acquire)
                        gemm2(L, tid, a, expert, eg, lo, hi, rsrc(a["gout"]), qb)
                        if const_expr(ESYNC):  # noqa: SIM102 - compile-time gate
                            # the echunk's last GEMM2 shard combines it right away; every
                            # GEMM1/GEMM2 waiter of eg has passed, so reset both counters.
                            echunk_done(L, tid, a, i32(ECNT2) + eg)
                            if lds_ld_i32(L, L_CTL + C_ELAST * 4) == P2 - i32(1):
                                if tid == i32(0):
                                    g_st_sys(ctrl_at(a, i32(ECNT2) + eg), i32(0))
                                    g_st_sys(ctrl_at(a, i32(ECNT1) + eg), i32(0))
                                fx.memory_fence(syncscope="agent",
                                                ordering=fx.AtomicOrdering.Acquire)
                                cc = ec.shrui(i32(24))
                                rc = fx.min(
                                    lds_ld_i32(L, L_DCNT + expert * i32(4)) - cc * i32(rch),
                                    i32(rch),
                                )
                                combine_send(L, tid, a, rsrc(a["gout"]), eg, rc)
            _prof_mark(tid, a, 8)  # post gemm2
            if const_expr(not ESYNC):  # noqa: SIM102 - compile-time gate, not dynamic
                gbar(L, tid, a, i32(2))
                for ei_ in range(bid_e, nch, nblk):
                    ei = i32(ei_)
                    ec = lds_ld_i32(L, L_DCH + ei * i32(4))
                    cc = ec.shrui(i32(24))
                    rc = fx.min(
                        lds_ld_i32(L, L_DCNT + (ec & i32(0xFFFF)) * i32(4)) - cc * i32(rch),
                        i32(rch),
                    )
                    combine_send(L, tid, a, rsrc(a["gout"]), ei, rc)

    @flyc.jit
    def compute_units(L, tid, a):
        """Local-expert GEMM as GATHER -> GEMM1 -> GEMM2 -> COMBINE passes separated by
        device grid barriers (``gbar``).  Each CTA walks its slice of the work with a
        stride-#CU loop, so a chunk count ``nch`` (or unit count ``nch*P``) that EXCEEDS
        the grid runs in multiple waves (high bs); when it fits one wave the loop body
        runs once.  gather/combine iterate echunks (CTA bid owns echunks bid, bid+nblk,
        ...); GEMM1/GEMM2 iterate (echunk, n-tile shard) units.  The gather pass publishes
        each row's recv-slot (``rowidx``) + payload-row (``rowmap``) to GLOBAL so the later
        passes -- on different CTAs once P>1 shards an echunk -- can read them."""
        if const_expr(NW == 8):
            # NW=8 (all 8 waves compute MFMA): there is no concurrent dispatch band, so wave
            # 0 does this CTA's full a2a scatter HERE (serial), then all 8 waves proceed to
            # the planner + inline-A GEMM.  Trades the comm/compute overlap for 2 MFMA
            # waves/SIMD.  (The A-loader band is likewise gone; the GEMMs self-stage A.)
            if tid < i32(64):
                dispatch_send(L, tid, a)
            gpu.barrier()
            if const_expr(_PROF):  # noqa: SIM102 - compile-time gate, not dynamic
                if tid == i32(0):  # post dispatch_send (slot 9)
                    bst(now(), rsrc(a["dbg"]),
                        (i32(gpu.block_id("x")) * i32(_PROF_SLOTS) + i32(9)) * i32(8), 0, 0)
        if tid == i32(0):
            dispatch_wait_meta(a, a["epoch"] & i32(1), i32(0))  # wait ROUTING only (fast)
        cbar(L, tid)
        if const_expr(_PROF):  # noqa: SIM102 - compile-time gate, not dynamic
            if tid == i32(0):  # post dispatch_wait_meta (slot 4)
                bst(now(), rsrc(a["dbg"]),
                    (i32(gpu.block_id("x")) * i32(_PROF_SLOTS) + i32(4)) * i32(8), 0, 0)
        # balanced-gather pre-pass: assign global per-expert positions, then a true
        # cross-CTA barrier (the scan is distributed) publishes rowlist to every CTA.
        # All of pre_pos / dyn_plan / PASS0-gather read only the routing (recv_le), so
        # they run NOW -- overlapping the big payload a2a still in flight -- and GEMM1
        # below waits for the payload (dispatch_wait) just before it needs it.
        pre_pos(L, tid, a)
        gbar(L, tid, a, i32(2))  # forced cross-CTA (pre_pos is a distributed scan)
        dyn_plan(L, tid, a)
        if const_expr(_PROF):  # noqa: SIM102 - compile-time gate, not dynamic
            if tid == i32(0):  # post dyn_plan (slot 5)
                bst(now(), rsrc(a["dbg"]),
                    (i32(gpu.block_id("x")) * i32(_PROF_SLOTS) + i32(5)) * i32(8), 0, 0)
        nch = lds_ld_i32(L, L_CTL + C_NCH * 4)
        nblk = i32(gpu.grid_dim.x)
        # P-way CU split: shard each echunk's GEMM n-tiles across ~nblk/nch CTAs (capped
        # at N_TILES).  When nch >= nblk (high bs) P collapses to 1 and nch itself spills
        # into multiple waves via the loops below.
        P = fx.max(fx.min(nblk // fx.max(nch, i32(1)), i32(N_TILES)), i32(1))
        bid_e = _walk_start(nch, nblk)  # echunk walks (gather / P==1 GEMM2 / combine)
        bid = _walk_start(nch * P, nblk)  # (echunk, shard) unit walks (NW=8)
        S1, K1 = _plan_ld(L, False)  # NW=4 pass splits (_phase / _unit)
        S2, K2 = _plan_ld(L, True)
        q1, r1, uf1, uc1, P1 = _phase(L, nch, i32(N_TILES), S1, K1)
        q2, r2, uf2, uc2, P2 = _phase(L, nch, i32(N_TILES2), S2, K2)
        r_rowmap = rsrc(a["rowmap"])
        r_rowidx = rsrc(a["rowidx"])
        # --- PASS 0 (gather): walk this CTA's echunks, publish per-row recv-slot (rowidx,
        #     for combine) + payload-row (rowmap, for GEMM1's A-gather) to global.
        for ei_ in range(bid_e, nch, nblk):
            ei = i32(ei_)
            ec = lds_ld_i32(L, L_DCH + ei * i32(4))
            expert = ec & i32(0xFFFF)
            cc = ec.shrui(i32(24))
            ce = ec.shrui(i32(16)) & i32(0xFF)
            R = gather_recv_chunk(L, tid, a, expert, cc, ce)  # -> L_RIX, R; ends in cbar
            for r_ in range(tid, i32(rch), i32(NT)):
                r = i32(r_)
                idx = (r < R).select(lds_ld_i32(L, L_RIX + r * i32(4)), i32(0))
                off = (ei * i32(rch) + r) * i32(4)
                bst(idx, r_rowidx, off, 0, 0)
                bst((r < R).select(idx // i32(TOPK), i32(0)), r_rowmap, off, 0, 0)
            cbar(L, tid)  # done reading L_RIX before the next echunk's gather overwrites it
        if const_expr(NW == 8):
            gbar(L, tid, a, P)
        else:  # P1 == 1: each echunk's GEMM1 runs on its gather CTA (same walk start)
            gbar(L, tid, a, P1)
        if const_expr(_PROF):  # noqa: SIM102 - compile-time gate, not dynamic
            if tid == i32(0):  # post gather (slot 10)
                bst(now(), rsrc(a["dbg"]),
                    (i32(gpu.block_id("x")) * i32(_PROF_SLOTS) + i32(10)) * i32(8), 0, 0)
        # payload wait: only NOW does the kernel need the dispatched activation (GEMM1's
        # A-gather). The planning + gather above overlapped the payload a2a, so this spin
        # is short (payload mostly landed). After it, the rows for every echunk are live.
        if tid == i32(0):
            dispatch_wait(a, a["epoch"] & i32(1), i32(0))
        cbar(L, tid)
        if const_expr(_PROF):  # noqa: SIM102 - compile-time gate, not dynamic
            if tid == i32(0):  # post payload wait (slot 11)
                bst(now(), rsrc(a["dbg"]),
                    (i32(gpu.block_id("x")) * i32(_PROF_SLOTS) + i32(11)) * i32(8), 0, 0)
        if tid == i32(0):  # release the A-loader wave: payload + rowmap are now live
            lds_st_rel(L, L_RCTL + RC_G1GO * i32(4), i32(1))
        if const_expr(NW == 8):
            # --- PASS 1 (GEMM1): walk this CTA's (echunk, shard) units -> inter[echunk] slice.
            # uord = this CTA's unit ordinal in the stride-#CU walk (matches a_loader_units),
            # so the running A-ring tile index is identical on both the producer and consumer.
            for gi_ in range(bid, nch * P, nblk):
                gi = i32(gi_)
                eg = gi // P
                expert = lds_ld_i32(L, L_DCH + eg * i32(4)) & i32(0xFFFF)
                if NW == 8:
                    gemm1_nw8(L, tid, a, expert, eg, gi - eg * P, P)
                else:
                    gemm1(L, tid, a, expert, eg, gi - eg * P, P, (gi - bid) // nblk)
                if const_expr(ESYNC):  # noqa: SIM102 - compile-time gate, not dynamic
                    if P > i32(1):
                        echunk_done(L, tid, a, i32(ECNT1) + eg)
            if const_expr(_PROF):  # noqa: SIM102 - compile-time gate, not dynamic
                if tid == i32(0):  # post gemm1 (slot 7)
                    bst(now(), rsrc(a["dbg"]),
                        (i32(gpu.block_id("x")) * i32(_PROF_SLOTS) + i32(7)) * i32(8), 0, 0)
            # EP_ESYNC: GEMM2 units wait their own echunk's GEMM1 shards (echunk_wait), so
            # only the local drain is needed here; else a grid barrier at P>1.
            if const_expr(ESYNC):
                gbar(L, tid, a, i32(1))
            else:
                gbar(L, tid, a, P)
            # reset the A-ring for GEMM2 + release the A-loader's PASS-2 phase (inter is now
            # written + gbar-flushed).  Safe here: PASS-1 drained the ring (compute consumed,
            # loader produced) and the loader is parked on RC_G2GO, so nothing touches it.
            if tid == i32(0):
                for b in range_constexpr(NAB):
                    lds_st(L, L_RCTL + (i32(RC_ASEQ) + i32(b)) * i32(4), i32(0))
                    lds_st(L, L_RCTL + (i32(RC_AFREE) + i32(b)) * i32(4), i32(0))
                for b in range_constexpr(4):  # W ring (EP_WLDS): the loader parks on G2GO too
                    lds_st(L, L_RCTL + (i32(RC_WSEQ) + i32(b)) * i32(4), i32(0))
                    lds_st(L, L_RCTL + (i32(RC_WFREE) + i32(b)) * i32(4), i32(0))
                lds_st_rel(L, L_RCTL + RC_G2GO * i32(4), i32(1))
            cbar(L, tid)  # compute waves see the zeroed ring before consuming GEMM2 A
            # --- PASS 2 (GEMM2) [+ fused combine at P==1].  P==1 (high bs): each echunk's
            #     GEMM2 scatters its output straight to the source ``part`` from the cshuffle
            #     epilogue (gemm2_combine) -- no gout round-trip, and the cross-rank P2P
            #     overlaps the NEXT n-tile's MFMA.  P>1 (gout n-tile-sharded across CTAs):
            #     grid-barrier, then combine full rows in this band (pass 3, combine_send).
            if P == i32(1):
                # fused-combine GEMM2: the bf16 output scatters straight to each row's source
                # part from the epilogue (no gout). rows = valid count (padding skipped).
                for ei_ in range(bid_e, nch, nblk):
                    ei = i32(ei_)
                    ec = lds_ld_i32(L, L_DCH + ei * i32(4))
                    cc = ec.shrui(i32(24))
                    rows = fx.min(
                        lds_ld_i32(L, L_DCNT + (ec & i32(0xFFFF)) * i32(4)) - cc * i32(rch),
                        i32(rch),
                    )
                    if NW == 8:
                        gemm2_combine_nw8(L, tid, a, ec & i32(0xFFFF), ei, rows)
                    else:
                        gemm2_combine(L, tid, a, ec & i32(0xFFFF), ei, rows, (ei - bid_e) // nblk)
                if const_expr(_PROF):  # noqa: SIM102 - compile-time gate, not dynamic
                    if tid == i32(0):  # post gemm2 (slot 8)
                        bst(now(), rsrc(a["dbg"]),
                            (i32(gpu.block_id("x")) * i32(_PROF_SLOTS) + i32(8)) * i32(8),
                            0, 0)
            else:
                for gi_ in range(bid, nch * P, nblk):
                    gi = i32(gi_)
                    eg = gi // P
                    ec = lds_ld_i32(L, L_DCH + eg * i32(4))
                    expert = ec & i32(0xFFFF)
                    if const_expr(ESYNC):  # noqa: SIM102 - compile-time gate, not dynamic
                        echunk_wait(a, tid == i32(0), i32(ECNT1) + eg, P)
                        cbar(L, tid)
                        fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Acquire)
                    if NW == 8:
                        gemm2_pipe_nw8(L, tid, a, expert, eg, gi - eg * P, P, rsrc(a["gout"]))
                    else:
                        gemm2(L, tid, a, expert, eg, gi - eg * P, P,
                              rsrc(a["gout"]), (gi - bid) // nblk)
                    if const_expr(ESYNC):  # noqa: SIM102 - compile-time gate, not dynamic
                        # the echunk's last GEMM2 shard combines it right away (no grid
                        # barrier); every GEMM1/GEMM2 waiter of eg has passed, so reset both.
                        echunk_done(L, tid, a, i32(ECNT2) + eg)
                        if lds_ld_i32(L, L_CTL + C_ELAST * 4) == P - i32(1):
                            if tid == i32(0):
                                g_st_sys(ctrl_at(a, i32(ECNT2) + eg), i32(0))
                                g_st_sys(ctrl_at(a, i32(ECNT1) + eg), i32(0))
                            fx.memory_fence(syncscope="agent",
                                            ordering=fx.AtomicOrdering.Acquire)
                            cc = ec.shrui(i32(24))
                            rc = fx.min(
                                lds_ld_i32(L, L_DCNT + expert * i32(4)) - cc * i32(rch),
                                i32(rch),
                            )
                            combine_send(L, tid, a, rsrc(a["gout"]), eg, rc)
                if const_expr(_PROF):  # noqa: SIM102 - compile-time gate, not dynamic
                    if tid == i32(0):  # post gemm2 (slot 8)
                        bst(now(), rsrc(a["dbg"]),
                            (i32(gpu.block_id("x")) * i32(_PROF_SLOTS) + i32(8)) * i32(8),
                            0, 0)
                if const_expr(not ESYNC):  # noqa: SIM102 - compile-time gate, not dynamic
                    gbar(L, tid, a, P)
                    for ei_ in range(bid_e, nch, nblk):
                        ei = i32(ei_)
                        ec = lds_ld_i32(L, L_DCH + ei * i32(4))
                        cc = ec.shrui(i32(24))
                        rc = fx.min(
                            lds_ld_i32(L, L_DCNT + (ec & i32(0xFFFF)) * i32(4)) - cc * i32(rch),
                            i32(rch),
                        )
                        combine_send(L, tid, a, rsrc(a["gout"]), ei, rc)
        else:
            _gemm_passes_nw4(L, tid, a, nch, nblk, bid_e, S1, K1, q1, r1, uf1, uc1, P1,
                             S2, K2, q2, r2, uf2, uc2, P2)
        if const_expr(_PROF):  # noqa: SIM102 - compile-time gate, not dynamic
            if tid == i32(0):  # per-n-tile GEMM tick sums (gemm.py _pmark) -> slots 13..18
                for k in range_constexpr(6):
                    bst(fx.Int64(lds_ld_i32(L, L_PF_ + k * 4)), rsrc(a["dbg"]),
                        (i32(gpu.block_id("x")) * i32(_PROF_SLOTS) + i32(13 + k)) * i32(8),
                        0, 0)

    @flyc.jit
    def a_loader_units(L, tid, a):
        """Dedicated A-loader band driver (TP-style).  Two phases, each gated by a go flag
        the compute band raises once the A source is live: PASS 1 (RC_G1GO -- dispatched
        payload + PASS-0 rowmap) stages each GEMM1 unit's A, PASS 2 (RC_G2GO -- the GEMM1
        intermediate written + gbar-flushed, ring reset) stages each GEMM2 unit's inter A.
        Walks the SAME stride-#CU unit walk + P as compute_units so the running ring tile
        index matches the consumer; GEMM2 mirrors the P==1 (per-echunk) / P>1 (per-shard)
        split of compute_units PASS 2."""
        lane = tid - i32(B_LOADER)  # loader-local lane 0..NLOAD-1 (= ATileLoader override)
        nblk = i32(gpu.grid_dim.x)
        # PASS 1: GEMM1 A (dispatched payload, indexed gather).
        spin0(L, lane, L_RCTL + RC_G1GO * i32(4), i32(1))
        nch = lds_ld_i32(L, L_CTL + C_NCH * 4)
        S1, K1 = _plan_ld(L, False)
        S2, K2 = _plan_ld(L, True)
        q1, r1, uf1, uc1, P1 = _phase(L, nch, i32(N_TILES), S1, K1)
        q2, r2, uf2, uc2, P2 = _phase(L, nch, i32(N_TILES2), S2, K2)
        t1 = _walk_start(uc1, nblk)
        for t_ in range(t1, uc1, nblk):
            t = i32(t_)
            rnd = (t - t1) // nblk
            for j_ in range(i32(0), _nslots(t, uf1, K1), i32(1)):
                eg, lo, hi, qb = _unit(t, i32(j_), rnd, S1, K1, q1, r1, uf1, i32(N_TILES))
                if eg < nch:
                    a_loader_g1(L, lane, a, lds_ld_i32(L, L_DCH + eg * i32(4)) & i32(0xFFFF),
                                eg, lo, hi, qb)
        # PASS 2: GEMM2 A (the intermediate, non-indexed) -- same item walk (P2 == 1:
        # whole echunks, the fused-combine GEMM2's walk).
        spin0(L, lane, L_RCTL + RC_G2GO * i32(4), i32(1))
        cross = (P1 > i32(1)) | (P2 > i32(1))  # the echunk's GEMM1 ran on other CTAs
        t2 = _walk_start(uc2, nblk)
        for t_ in range(t2, uc2, nblk):
            t = i32(t_)
            rnd = (t - t2) // nblk
            for j_ in range(i32(0), _nslots(t, uf2, K2), i32(1)):
                eg, lo, hi, qb = _unit(t, i32(j_), rnd, S2, K2, q2, r2, uf2, i32(N_TILES2))
                if eg < nch:
                    if const_expr(ESYNC):  # noqa: SIM102 - compile-time gate, not dynamic
                        if cross:
                            echunk_wait(a, lane == i32(0), i32(ECNT1) + eg, P1)
                            fx.memory_fence(syncscope="agent",
                                            ordering=fx.AtomicOrdering.Acquire)
                    a_loader_g2(L, lane, a, eg, lo, hi - lo, qb)

    # ---- W-loader (EP_WLDS, NW=4): the two waves past the A-loader -------------------
    # Walks the SAME unit / n-tile / K-step sequence as the A-loader (running ring tile q
    # per GEMM phase) and LDS-DMAs tile q's W (TILE_N/16 n-blocks x 2 KB, BWeightLoader
    # layout) + B-scale (TILE_N/32 groups x 256 B, BScaleLoader layout) into W-ring bank
    # q % NWB, which the compute waves read instead of HBM (gemm.py _wl_*).  Bank layout:
    # chunk c = nb_local*2 + ksub at c*1 KB (lane*16 B within), scale group g at
    # WB_W + g*256 (lane*4 B within).  Each wave stages half a tile (16 x 1 KB + 4 x 256 B
    # = WL_OPS DMAs) and publishes it WL_D-1 tiles later (wait_vm keeps WL_D tiles in
    # flight); the last tile of a phase flushes.  GEMM2 starts after compute's G2GO (which
    # follows its ring reset), so the loader never touches the counters across the reset.
    H, K_ITERS = kc.get("H K_ITERS")
    K_ITERS2 = I // (H // K_ITERS)  # GEMM2 K-steps (same TILE_K)
    TILE_N, WL, NWB, L_W, WB = kc.get("TILE_N WL NWB L_W WB")
    NB_W = TILE_N // 16  # 16-col weight blocks per n-tile
    NG_S = TILE_N // 32  # 32-col scale groups per n-tile
    WB_W = NB_W * 2048  # W bytes per bank (the B-scale follows)
    WL_OPS = NB_W + NG_S // 2  # DMAs per loader wave per tile
    WL_D = max(NWB, 1)  # tiles in flight per loader wave (<= NWB: no publish deadlock)
    assert WL_OPS * (WL_D - 1) <= 63  # vmcnt range
    assert EPR * 2 * I * (H // 2) < 2**31 and EPR * H * (I // 2) < 2**31  # i32 soffset

    @flyc.jit
    def _wl_put(L, lw, lane, q, rw, rws, sw, ss, sn0, srow4):
        """Stage this wave's half of tile q (soffsets sw / ss = the tile's W / scale base)."""
        if lane == i32(0):
            spin_lds_ge(L, L_RCTL + (i32(RC_WFREE) + q % i32(NWB)) * i32(4),
                        i32(NW) * (q // i32(NWB)))
        rocdl.sched_barrier(0)
        wb = L + i32(L_W) + (q % i32(NWB)) * i32(WB)
        for j in range_constexpr(NB_W):
            c = lw * i32(NB_W) + i32(j)
            dma16(wb + c * i32(1024), rw, lane * i32(16),
                  sw + (c >> i32(1)) * sn0 + (c & i32(1)) * i32(1024))
        for j in range_constexpr(NG_S // 2):
            g = lw * i32(NG_S // 2) + i32(j)
            dma4(wb + i32(WB_W) + g * i32(256), rws, lane * i32(4), ss + g * srow4)

    @flyc.jit
    def _wl_pub(L, lane, q, last):
        """Publish tile q - (WL_D-1) (its DMAs are done once <= (WL_D-1) tiles' remain);
        at the phase's last tile drain and publish the rest."""
        if q >= i32(WL_D - 1):
            wait_vm(WL_OPS * (WL_D - 1))
            if lane == i32(0):
                t = q - i32(WL_D - 1)
                lds_atomic_add(L, L_RCTL + (i32(RC_WSEQ) + t % i32(NWB)) * i32(4), 1, REL)
        if last:
            wait_vm(0)
            if lane == i32(0):
                for d in range_constexpr(WL_D - 1):
                    t = q - i32(d)
                    if t >= i32(0):
                        lds_atomic_add(
                            L, L_RCTL + (i32(RC_WSEQ) + t % i32(NWB)) * i32(4), 1, REL)

    @flyc.jit
    def w_lds_units(L, tid, a):
        """One phase-parameterised walk (ph 0 = GEMM1 W1 [2I, H], ph 1 = GEMM2 W2 [H, I])
        on one V# per phase at the local-expert slab base; expert / n-tile / K-step offsets
        go in the (uniform) soffset.  P==1 makes nt_lo/nt_hi span all n-tiles, matching
        compute's per-echunk GEMM2."""
        wl = tid - i32(B_LOADER + NLOAD)  # 0..127
        lw = wl // i32(64)
        lane = wl % i32(64)
        nblk = i32(gpu.grid_dim.x)
        spin_lds_ge(L, L_CTL + C_PLAN * 4, i32(1))
        nch = lds_ld_i32(L, L_CTL + C_NCH * 4)
        for ph_ in range(i32(0), i32(2), i32(1)):
            g2 = i32(ph_) == i32(1)
            if g2:
                if lane == i32(0):
                    spin_lds_ge(L, L_RCTL + RC_G2GO * 4, i32(1))
                rocdl.sched_barrier(0)
            rw = rsrc(g2.select(fx.Int64(a["w2"]), fx.Int64(a["w1"])))
            rws = rsrc(g2.select(fx.Int64(a["w2s"]), fx.Int64(a["w1s"])))
            ntn = g2.select(i32(N_TILES2), i32(N_TILES))
            kn = g2.select(i32(K_ITERS2), i32(K_ITERS))
            sn0 = g2.select(i32(I * 8), i32(H * 8))
            srow4 = g2.select(i32(I // 256 * 256), i32(H // 256 * 256))
            ew = g2.select(i32(H * (I // 2)), i32(2 * I * (H // 2)))
            es = g2.select(i32(H * (I // 32)), i32(2 * I * (H // 32)))
            S = g2.select(lds_ld_i32(L, L_CTL + C_S2 * 4), lds_ld_i32(L, L_CTL + C_S1 * 4))
            K = g2.select(lds_ld_i32(L, L_CTL + C_K2 * 4), lds_ld_i32(L, L_CTL + C_K1 * 4))
            q, r, uf, uc, _ = _phase(L, nch, ntn, S, K)
            t0 = _walk_start(uc, nblk)
            for t_ in range(t0, uc, nblk):
                t = i32(t_)
                rnd = (t - t0) // nblk
                ns = _nslots(t, uf, K)
                for j_ in range(i32(0), ns, i32(1)):
                    j = i32(j_)
                    eg, nt_lo, nt_hi, qb = _unit(t, j, rnd, S, K, q, r, uf, ntn)
                    if eg < nch:
                        expert = lds_ld_i32(L, L_DCH + eg * i32(4)) & i32(0xFFFF)
                        # this CTA's last unit of the phase: flush the delayed publishes
                        lastu = (t + nblk >= uc) & ((j == ns - i32(1)) | (eg + i32(1) >= nch))
                        for nt_ in range(nt_lo, nt_hi, i32(1)):
                            nt = i32(nt_)
                            sw = expert * ew + nt * i32(NB_W) * sn0
                            ss = expert * es + nt * i32(NG_S) * srow4
                            lastn = lastu & (nt == nt_hi - i32(1))
                            for sp_ in range(i32(0), kn, i32(1)):
                                sp = i32(sp_)
                                q_ = (qb + nt - nt_lo) * kn + sp
                                _wl_put(L, lw, lane, q_, rw, rws, sw + sp * i32(2048),
                                        ss + sp * i32(256), sn0, srow4)
                                _wl_pub(L, lane, q_, lastn & (sp == kn - i32(1)))

    return locals()
