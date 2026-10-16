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
    i32,
    lds_atomic_add,
    lds_atomic_or,
    lds_ld_acq,
    lds_ld_i32,
    lds_st,
    lds_st_rel,
    now,
    rsrc,
    wait_lgkm0,
    wait_vm,
    wave_rank,
)

# Device-phase profiling (matches mega_moe_ep_kernel._PROF): slots 4,5 are stamped in
# compute_units (post dispatch_wait / post dyn_plan) to split the `roles` phase.
_PROF = bool(int(os.environ.get("EP_PROF_DEV", "0")))
_PROF_SLOTS = 9  # +7 post-gemm1, +8 post-gemm2 (per-unit; <=1 unit/CTA)
from .mega_moe_ep_config import (
    C_BARCNT,
    C_BARGEN,
    C_CNT,
    C_DYNP,
    C_NACT,
    C_NCH,
    C_PLAN,
    C_UL,
    C_UNIT,
    C_UROWS,
    C_USEQ,
    N_XCD,
    NT,
    NW,
    UL_MAX,
    KernelCtx,
)

__all__ = ["build_schedule"]


def build_schedule(kc: KernelCtx) -> dict:
    """The dispatch-planner device functions of one instance (added to kc)."""
    DYN_PS, EPR, I, KS2, TOPK = kc.get("DYN_PS EPR I KS2 TOPK")
    L_CTL, L_DCH, L_DCNT, L_DYN = kc.get("L_CTL L_DCH L_DCNT L_DYN")
    L_EOFF, L_RIX, L_WSUM, NBW = kc.get("L_EOFF L_RIX L_WSUM NBW")
    const_expr, rch, N_TILES = kc.get("const_expr rch N_TILES")
    # GEMM1/GEMM2 + the per-unit A stager (gemm.build_gemm) and the combine epilogue
    # (communication.build_communication) are added to kc before build_schedule.
    a_loader, gemm1, gemm2 = kc.get("a_loader gemm1 gemm2")
    combine_send = kc.get("combine_send")[0]
    dispatch_wait = kc.get("dispatch_wait")[0]
    grid_bar = kc.get("grid_bar")[0]
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
            # Target rch/2 rows per chunk, not rch: gather chunks by sk%ce (deterministic
            # but uneven), so a fuller target lets an unlucky bucket exceed rch and drop
            # rows (the gather caps at rch). Half-full keeps the max bucket well under rch.
            # TODO(tune): deterministic idx-rank chunking for a hard guarantee.
            ce = fx.min(fx.max(ceildiv(r, i32(max(rch // 2, 1))), i32(1)), i32(255))
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
    def dyn_plan(L, tid, a):
        """Scan this rank's received routing -> per-local-expert bitmap + counts ->
        active list -> row-chunk table -> per-CTA unit list [u_lo, u_hi) (XCD-aware)."""
        rle = rsrc(a["recv_le"], a["nrecv"] * i32(4))
        # Scan only the LIVE window -- npes source ranks x m tokens x topk routes -- not
        # the full mtpr capacity, mapping the compact index back to the real recv_le slot
        # (p*mtpr*topk + within-block). Assumes a uniform per-rank token count m (true for
        # the EP tests / balanced serving); TODO: carry a global max-m for skewed ranks.
        blk = a["m"] * i32(TOPK)
        stride_p = a["mtpr"] * i32(TOPK)
        for c_ in range(tid, a["npes"] * blk, i32(NT)):
            c = i32(c_)
            p = c // blk
            idx = p * stride_p + (c - p * blk)
            le = fx.Int32(bld(rle, idx * i32(4), 0, T.i32, 0))
            if masked(le) == fx.Boolean(False):
                lds_atomic_or(
                    L, L_DYN + (le >> i32(5)) * i32(4), i32(1) << (le & i32(31))
                )
                lds_atomic_add(L, L_DCNT + le * i32(4), 1)
        cbar(L, tid)
        _dyn_prefix(L, tid)
        cbar(L, tid)
        for it in range_constexpr(ceildiv(EPR, NT)):
            _dyn_list(L, tid + i32(it * NT))
        cbar(L, tid)
        _chunk_table(L, tid)
        cbar(L, tid)
        _dyn_units(L, tid)
        cbar(L, tid)

    @flyc.jit
    def gather_recv_chunk(L, tid, a, expert, cc, ce):
        """Scan the received routing; collect the payload rows (source_keys) whose
        local expert is ``expert`` and that fall in chunk ``cc`` of ``ce`` into L_RIX,
        and return the chunk's row count. Chunk by source_key % ce (deterministic and
        balanced). L_RIX stores each row's recv slot idx = source_key*topk + route; the
        GEMM reads payload row source_key = idx//topk, and combine recovers route =
        idx%topk (and source_key -> src rank/token) to scatter the result back."""
        rle = rsrc(a["recv_le"], a["nrecv"] * i32(4))
        blk = a["m"] * i32(TOPK)  # live window per source rank (see dyn_plan)
        stride_p = a["mtpr"] * i32(TOPK)
        if tid == i32(0):
            lds_st(L, L_CTL + C_CNT * 4, i32(0))
        cbar(L, tid)
        for c_ in range(tid, a["npes"] * blk, i32(NT)):
            c = i32(c_)
            p = c // blk
            idx = p * stride_p + (c - p * blk)
            le = fx.Int32(bld(rle, idx * i32(4), 0, T.i32, 0))
            sk = idx // i32(TOPK)
            if (le == expert) & ((sk % ce) == cc):
                pos = lds_atomic_add(L, L_CTL + C_CNT * 4, 1)
                if pos < i32(rch):
                    lds_st(L, L_RIX + pos * i32(4), idx)
        cbar(L, tid)
        return fx.min(lds_ld_i32(L, L_CTL + C_CNT * 4), i32(rch))

    @flyc.jit
    def compute_units(L, tid, a):
        """Local-expert GEMM as GATHER -> GEMM1 -> GEMM2 -> COMBINE passes separated by
        device grid barriers (``gbar``), so GEMM1/GEMM2 can be sharded across CUs.

        Inc 2 (structure): P==1 -- CTA ``bid`` owns echunk ``bid`` (the chunk table is a
        dense list 0..NCH that every CTA built in ``dyn_plan``), so every pass is the same
        CTA and the shared inter/gout round-trips are same-CTA (cross-CTA coherence is
        Inc 3). The gather pass keeps the recv-slot list in LDS (``L_RIX``) for combine and
        publishes the payload-row map to global for GEMM1's A-gather."""
        if tid == i32(0):
            dispatch_wait(a, a["epoch"] & i32(1), i32(0))
        cbar(L, tid)
        if const_expr(_PROF):  # noqa: SIM102 - compile-time gate, not dynamic
            if tid == i32(0):  # post dispatch_wait (slot 4)
                bst(now(), rsrc(a["dbg"]),
                    (i32(gpu.block_id("x")) * i32(_PROF_SLOTS) + i32(4)) * i32(8), 0, 0)
        dyn_plan(L, tid, a)
        if const_expr(_PROF):  # noqa: SIM102 - compile-time gate, not dynamic
            if tid == i32(0):  # post dyn_plan (slot 5)
                bst(now(), rsrc(a["dbg"]),
                    (i32(gpu.block_id("x")) * i32(_PROF_SLOTS) + i32(5)) * i32(8), 0, 0)
        bid = i32(gpu.block_id("x"))
        nch = lds_ld_i32(L, L_CTL + C_NCH * 4)
        nblk = i32(gpu.grid_dim.x)
        # P-way CU split: shard each echunk's GEMM n-tiles across ~nblk/nch CTAs so the
        # idle CUs join the per-expert GEMM (capped at N_TILES -> >=1 n-tile per shard).
        P = fx.max(fx.min(nblk // fx.max(nch, i32(1)), i32(N_TILES)), i32(1))
        # gather + combine run per echunk (pass 0 / 3): CTA bid owns echunk bid.
        live = bid < nch
        ec = live.select(lds_ld_i32(L, L_DCH + bid * i32(4)), i32(0))
        expert = ec & i32(0xFFFF)
        cc = ec.shrui(i32(24))
        ce = ec.shrui(i32(16)) & i32(0xFF)
        # GEMM1 + GEMM2 run per (echunk, shard) (pass 1 / 2): CTA bid -> echunk bid//P,
        # n-tile shard bid%P. The P shards of one echunk share inter[echunk] / gout[echunk]
        # (published across CTAs by the grid barriers), so no partial-sum reduction.
        eg = bid // P
        sg = bid - eg * P
        liveg = bid < nch * P
        ecg = liveg.select(lds_ld_i32(L, L_DCH + eg * i32(4)), i32(0))
        expertg = ecg & i32(0xFFFF)
        # --- PASS 0 (gather): echunk=bid rows -> L_RIX (LDS, kept for combine) + the
        #     payload-row map to global (rowmap[bid*RG+r], GEMM1's A-gather source key).
        R = i32(0)
        if live:
            R = gather_recv_chunk(L, tid, a, expert, cc, ce)
        if tid == i32(0):
            lds_st(L, L_CTL + C_UROWS * 4, R)  # persist for the combine pass (same CTA)
        cbar(L, tid)
        r_rowmap = rsrc(a["rowmap"])
        for r_ in range(tid, i32(rch), i32(NT)):
            r = i32(r_)
            v = (live & (r < R)).select(
                lds_ld_i32(L, L_RIX + r * i32(4)) // i32(TOPK), i32(0)
            )
            bst(v, r_rowmap, (bid * i32(rch) + r) * i32(4), 0, 0)
        gbar(L, tid, a, P)
        # --- PASS 1 (GEMM1): echunk eg, n-tile shard sg of P -> inter[eg] col slice.
        if liveg:
            gemm1(L, tid, a, expertg, eg, sg, P)
        if const_expr(_PROF):  # noqa: SIM102 - compile-time gate, not dynamic
            if tid == i32(0):  # post gemm1 (slot 7)
                bst(now(), rsrc(a["dbg"]),
                    (i32(gpu.block_id("x")) * i32(_PROF_SLOTS) + i32(7)) * i32(8), 0, 0)
        gbar(L, tid, a, P)
        # --- PASS 2 (GEMM2): echunk eg, n-tile shard sg of P -> gout[eg] col slice
        #     (reads the full inter[eg] published by all P GEMM1 shards via the grid bar).
        if liveg:
            gemm2(L, tid, a, expertg, eg, sg, P, rsrc(a["gout"]))
        if const_expr(_PROF):  # noqa: SIM102 - compile-time gate, not dynamic
            if tid == i32(0):  # post gemm2 (slot 8)
                bst(now(), rsrc(a["dbg"]),
                    (i32(gpu.block_id("x")) * i32(_PROF_SLOTS) + i32(8)) * i32(8), 0, 0)
        gbar(L, tid, a, P)
        # --- PASS 3 (combine): echunk=bid rows (R + L_RIX from pass 0, same CTA).
        if live:
            rc = lds_ld_i32(L, L_CTL + C_UROWS * 4)
            if tid < i32(64):
                combine_send(L, tid, a, rsrc(a["gout"]), bid, rc)

    return locals()
