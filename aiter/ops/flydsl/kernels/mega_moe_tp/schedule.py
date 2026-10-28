# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit scheduling: dynamic (active-expert row-chunk plan) and LB (two-phase GEMM1 /
GEMM2 units claimed by whichever CTA is free)."""

from __future__ import annotations

import os

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ..kernels_common import ceildiv
from .common import (
    ACQ,
    AUX_SYS,
    V4I,
    _ctpop,
    bld,
    bst,
    g_add_agent,
    g_ld_sys,
    g_st_sys,
    i32,
    lds_atomic_add,
    lds_ld,
    lds_ld_acq,
    lds_ld_i32,
    lds_st,
    lds_st_rel,
    rsrc,
    spin0,
    spin_lds_ge,
    trace_mark,
    dma16,
    uni,
    wait_lgkm0,
    wait_vm,
    wave_rank,
)
from .mega_moe_tp_config import (
    C_CDONE,
    C_CLAIM,
    C_DYNP,
    C_G1FIN,
    C_G2END,
    C_G2FREE,
    C_G2LAST,
    C_G2RDY,
    C_G2U,
    C_G2V,
    C_G2VN,
    C_GEXP,
    C_LBB,
    C_LBJ2,
    C_LBN,
    C_NACT,
    C_NCH,
    C_PLAN,
    C_RLAND,
    C_UL,
    C_UNIT,
    C_UROWS,
    C_USEQ,
    CTRL_COLC,
    CTRL_G1C,
    CTRL_INTS,
    CTRL_LBR,
    CTRL_LBSEQ,
    CTRL_SMC,
    CTRL_SMF,
    CTRL_XF,
    G1C_STRIDE,
    LRDY_STRIDE,
    N_XCD,
    NCHLB_MAX,
    NCK_MAX,
    NCTA_MAX,
    NT,
    NW,
    UL_MAX,
    UNIT_G1X,
    XQ_SHIFT,
    KernelCtx,
)


# (small-M) CTAs without GEMM1 work prefetch a neighbour's back half
DPF = os.environ.get("AITER_MEGAMOE_TP_DPF", "0") == "1"


def build_schedule(kc: KernelCtx) -> dict:
    """The schedule device functions of one instance (added to kc)."""
    DLL, DYN_PS, E, GPC, H2, I, KS2 = kc.get("DLL DYN_PS E GPC H2 I KS2")
    L_B1, L_B1S, L_CTL, L_DCH, L_DCNT = kc.get("L_B1 L_B1S L_CTL L_DCH L_DCNT")
    L_DPRE, L_DYN, L_EOFF, L_INTER = kc.get("L_DPRE L_DYN L_EOFF L_INTER")
    L_INTERS, L_RIX, L_WSUM, L_WT, LB_B = kc.get("L_INTERS L_RIX L_WSUM L_WT LB_B")
    LB_PS, LBMIX, LBMIX_DIV, LBPF, MLL = kc.get("LB_PS LBMIX LBMIX_DIV LBPF MLL")
    NBW, NCG, NCK, RG, SI_STRIDE, TMAX = kc.get("NBW NCG NCK RG SI_STRIDE TMAX")
    TOPK, ag_stage_free, ag_wait_meta = kc.get("TOPK ag_stage_free ag_wait_meta")
    ar, cbar, cg_lo, claim_addr = kc.get("ar cbar cg_lo claim_addr")
    col_claim, const_expr, ctrl_at = kc.get("col_claim const_expr ctrl_at")
    gather_routes_chunk, gemm1, gemm2 = kc.get("gather_routes_chunk gemm1 gemm2")
    lb, lrdy_at, masked = kc.get("lb lrdy_at masked")
    rch, report_chunk, spin_sys_ge = kc.get("rch report_chunk spin_sys_ge")
    unit_fields, zero_masked = kc.get("unit_fields zero_masked")
    TRACE, SM, route_regs = kc.get("TRACE SM route_regs")
    H, KS1, L_RING = kc.get("H KS1 L_RING")

    def mark(tid, a, k):
        if TRACE and (not kc.TRACE_LIGHT or k < 10 or k >= 40):
            trace_mark(a["ctrl"], CTRL_INTS, k, tid, 0)

    @flyc.jit
    def gemm2_dispatch(L, tid, a, expert, ks0, icnt, r0, rows, signal):
        for q in DYN_PS:
            if icnt == i32(I // q):
                n = KS2 // q
                gemm2(L, tid, a, expert, ks0, r0, rows, signal, n, ks0 // i32(n))

    @flyc.jit
    def sm_zero_next(tid, a):
        """every launch zeroes the small-M chunk counters of the next launch's bank (epoch
        parity): a small-M launch reads its own bank only, and other variants may run in
        between"""
        bid = i32(gpu.block_id("x"))
        nblk = i32(gpu.grid_dim.x)
        nb = (a["epoch"] + i32(1)) & i32(1)
        if tid < i32(64):
            for j_ in range(bid + tid * nblk, i32(NCHLB_MAX), nblk * i32(64)):
                g_st_sys(
                    ctrl_at(
                        a,
                        i32(CTRL_SMC)
                        + (nb * i32(NCHLB_MAX) + i32(j_)) * i32(G1C_STRIDE),
                    ),
                    i32(0),
                )
            if bid == i32(0):
                for j_ in range(tid, i32(NCTA_MAX), i32(64)):
                    g_st_sys(
                        ctrl_at(
                            a,
                            i32(CTRL_SMF)
                            + (nb * i32(NCTA_MAX) + i32(j_)) * i32(G1C_STRIDE),
                        ),
                        i32(0),
                    )

    # small-M launches: no L2 invalidate at the start (it evicts the kernarg lines, whose
    # re-load then stalls every CTA ~2 us); their reads of peer-written data are sc1 / sys
    DSTALE = os.environ.get(
        "AITER_MEGAMOE_TP_DSTALE", "none" if SM and not kc.TN else "agent"
    )

    @flyc.jit
    def _drop_stale(tid, a):
        if const_expr(DSTALE == "agent"):
            if tid < i32(64):
                fx.memory_fence(syncscope="agent", ordering=ACQ)  # drop stale L1 lines
        bid = i32(gpu.block_id("x"))
        if (tid < i32(64)) & (bid < i32(N_XCD)) & (a["epoch"] == i32(1)):
            fx.memory_fence(syncscope="one-as", ordering=ACQ)
            if tid == i32(0):
                g_st_sys(ctrl_at(a, i32(CTRL_XF) + bid), a["epoch"])

    @flyc.jit
    def _stale_dropped(tid, a):
        if (tid == i32(0)) & (a["epoch"] == i32(1)):
            x = i32(gpu.block_id("x")) % i32(N_XCD)
            spin_sys_ge(ctrl_at(a, i32(CTRL_XF) + x), a["epoch"])

    def unit_tile(L, tid, a, expert, i0, icnt, r0, rows, sig):
        mark(tid, a, 10)
        gemm1(L, tid, a, expert, i0, icnt // i32(128))
        mark(tid, a, 11)
        gemm2_dispatch(L, tid, a, expert, i0 // i32(128), icnt, r0, rows, sig)
        mark(tid, a, 12)

    @flyc.jit
    def _dyn_loop(L, tid, a):
        lane = tid % i32(64)
        w = tid // i32(64)
        ub = lds_ld_i32(L, L_CTL + C_UNIT * 4)
        ue = lds_ld_i32(L, L_CTL + (C_UNIT + 1) * 4)
        if ub == ue:
            if const_expr(SM and DPF):
                _dyn_prefetch(L, tid, a)
            report_chunk(L, lane, w, i32(NCK - 1))
        for u_ in range(ub, ue, i32(1)):
            u = i32(u_)
            expert, i0, icnt, kind = unit_fields(L, u)
            if const_expr(lb):
                R = lb_routes(L, tid, a, expert)
            else:
                cc = expert.shrui(i32(24))
                ce = expert.shrui(i32(16)) & i32(0xFF)
                expert = expert & i32(0xFFFF)
                R = gather_routes_chunk(L, tid, a, expert, cc, ce)
            if tid == i32(0):
                lds_st(L, L_CTL + C_UROWS * 4, R)
                lds_st_rel(L, L_CTL + C_USEQ * 4, u - ub + i32(1))
            if (u == ue - i32(1)) & (R == i32(0)):
                report_chunk(L, lane, w, i32(NCK - 1))
            for r0_ in range(i32(0), R, i32(RG)):
                r0 = i32(r0_)
                rows = fx.min(R - r0, i32(RG))
                if const_expr(lb):
                    lb_g1_tile(L, tid, a, expert, i0, icnt, rows)
                else:
                    sig = (u == ue - i32(1)) & (r0 + i32(RG) >= R)
                    unit_tile(L, tid, a, expert, i0, icnt, r0, rows, sig)
                cbar(L, tid)
            if const_expr(lb):  # noqa: SIM102
                if tid == i32(0):
                    g_add_agent(lb_g1c(a, kind.shrui(i32(XQ_SHIFT)), lb_bank(L)), 1)
        if const_expr(LBPF):
            lb_g2_consume(L, tid, a)
        elif const_expr(lb):
            lb_g2_phase(L, tid, a)

    @flyc.jit
    def pf_unit(L, tid, a, e, i0, icnt, w2):
        """read the back half of a unit -- its GEMM1 k-steps from KS1 / 2 on (inter rows
        [i0, i0 + icnt) of expert e) and (w2) its GEMM2 k-step -- into the MALL: a CTA
        without work of its own, while the unit's CTA streams the front half"""
        lane = tid % i32(64)
        w = uni(tid // i32(64))
        e = uni(e)
        dst = L + i32(L_RING) + w * i32(1024)
        r1 = rsrc(fx.Int64(a["w1"]) + fx.Int64(e) * fx.Int64(I * H))
        ng = icnt // i32(16)  # 16-row groups per half (gate / up)
        nk = KS1 - KS1 // 2
        for q_ in range(w, ng * i32(2 * nk), i32(NW)):
            q = i32(q_)
            g = q // i32(nk)
            kk = q - g * i32(nk) + i32(KS1 // 2)
            up = g >= ng
            row = up.select(i32(I) + i0 + (g - ng) * i32(16), i0 + g * i32(16))
            wait_vm(40)
            dma16(dst, r1, lane * i32(16), row * i32(H // 2) + kk * i32(1024))
        if const_expr(w2):
            r2 = rsrc(fx.Int64(a["w2"]) + fx.Int64(e) * fx.Int64(H * (I // 2)))
            kp = i0 // i32(128)
            for n_ in range(w, i32(H // 16), i32(NW)):
                wait_vm(40)
                dma16(dst, r2, lane * i32(16), i32(n_) * i32(16 * (I // 2)) + kp * i32(1024))
        wait_vm(0)

    @flyc.jit
    def _dyn_prefetch(L, tid, a):
        """(a CTA without a dynamic unit) pf_unit of the next CTA's"""
        C = i32(gpu.grid_dim.x)
        P = lds_ld_i32(L, L_CTL + C_DYNP * 4)
        U = lds_ld_i32(L, L_CTL + C_NCH * 4) * P
        if U > i32(0):
            u = fx.min(_xcd_rank(C) * U // C, U - i32(1))
            j = u // P
            icnt = i32(I) // P
            pf_unit(
                L, tid, a, lds_ld_i32(L, L_DCH + j * i32(4)) & i32(0xFFFF), (u - j * P) * icnt, icnt, True
            )

    @flyc.jit
    def compute_units(L, tid, a):
        if const_expr(not ar and not MLL):
            ag_wait_meta(a, tid, a["epoch"])
            cbar(L, tid)  # (wave 0 polled the routing flags)
        if const_expr(not lb):
            # the dynamic schedule's routes in registers (MLL: the meta wait)
            a["rregs"] = route_regs(tid, a)
        mark(tid, a, 2)
        _stale_dropped(tid, a)
        mark(tid, a, 24)
        if const_expr(ar and not DLL):
            zero_masked(L, tid % i32(64), a, tid // i32(64), NW)
        cbar(L, tid)
        if const_expr(MLL):  # noqa: SIM102
            if tid == i32(0):
                lds_st_rel(L, L_CTL + C_RLAND * 4, i32(1))
        if const_expr(not ar and not DLL):
            zero_masked(L, tid % i32(64), a, tid // i32(64), NW)
        if const_expr(lb):
            lb_plan(L, tid, a)
        else:
            dyn_plan(L, tid, a)
        mark(tid, a, 3)
        if const_expr(SM):
            kc.sm_compute(L, tid, a)
        else:
            _dyn_loop(L, tid, a)
        mark(tid, a, 4)
        if const_expr(DLL):  # noqa: SIM102
            if tid == i32(0):
                lds_st_rel(L, L_CTL + C_CDONE * 4, i32(1))

    @flyc.jit
    def dyn_plan(L, tid, a):
        """(compute waves) the routes per expert (LDS atomics from the registers), then wave
        0 alone: the row-chunk table (expert ascending) and this CTA's units"""
        for _, e, _, live in a["rregs"]:
            if live & (masked(e) == fx.Boolean(False)):
                lds_atomic_add(L, L_DCNT + e * i32(4), 1)
        cbar(L, tid)
        mark(tid, a, 21)
        if const_expr(PCT):
            _par_chunk_table(L, tid)
        if tid < i32(64):
            if const_expr(not PCT):
                _wave_chunk_table(L, tid)
            mark(tid, a, 23)
            if const_expr(SM):
                kc.sm_units(L, tid, a)
            else:
                _dyn_units(L, tid)
        cbar(L, tid)

    # the chunk table over the compute waves (contiguous expert groups of 64 per wave)
    PCT = os.environ.get("AITER_MEGAMOE_TP_PCT", "1") == "1"
    NG = ceildiv(E, 64)
    GPW = ceildiv(NG, NW)

    @flyc.jit
    def _par_chunk_table(L, tid):
        """(compute waves) _wave_chunk_table, wave w the expert groups [w GPW, (w+1) GPW)"""
        lane = tid % i32(64)
        w = tid // i32(64)
        ces = []
        loc = i32(0)
        for g in range_constexpr(GPW):
            e = (w * i32(GPW) + i32(g)) * i32(64) + lane
            live = e < i32(E)
            r = live.select(
                lds_ld_i32(L, L_DCNT + fx.min(e, i32(E - 1)) * i32(4)), i32(0)
            )
            ce = fx.min(ceildiv(r, i32(rch)), i32(255))
            pre, tot = _wave_prefix(ce, 8)
            ces.append((e, ce, loc + pre))
            loc = loc + tot
        if lane == i32(0):
            lds_st(L, L_WSUM + w * i32(4), loc)
        cbar(L, tid)
        base = i32(0)
        nch = i32(0)
        for v in range_constexpr(NW):
            cv = lds_ld_i32(L, L_WSUM + i32(v * 4))
            base = base + (i32(v) < w).select(cv, i32(0))
            nch = nch + cv
        for gi in range_constexpr(GPW):
            e, ce, off = ces[gi]
            for c_ in range(i32(0), ce, i32(1)):
                c = i32(c_)
                lds_st(
                    L,
                    L_DCH + (base + off + c) * i32(4),
                    e | (ce << i32(16)) | (c << i32(24)),
                )
        if tid == i32(0):
            lds_st(L, L_CTL + C_NCH * 4, nch)
        cbar(L, tid)

    @flyc.jit
    def _wave_chunk_table(L, lane):
        """(one wave) expert e -> ceil(routes / rch) row chunks, listed expert-ascending in
        L_DCH (e | chunks << 16 | chunk << 24); C_NCH = their number"""
        base = i32(0)
        for k in range_constexpr(ceildiv(E, 64)):
            e = lane + i32(k * 64)
            live = e < i32(E)
            r = live.select(
                lds_ld_i32(L, L_DCNT + fx.min(e, i32(E - 1)) * i32(4)), i32(0)
            )
            ce = fx.min(ceildiv(r, i32(rch)), i32(255))
            pre, tot = _wave_prefix(ce, 8)
            off = base + pre
            for c_ in range(i32(0), ce, i32(1)):
                c = i32(c_)
                lds_st(
                    L, L_DCH + (off + c) * i32(4), e | (ce << i32(16)) | (c << i32(24))
                )
            base = base + tot
        if lane == i32(0):
            lds_st(L, L_CTL + C_NCH * 4, base)

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
        if e < i32(E):
            wv = lds_ld_i32(L, L_DYN + (e >> i32(5)) * i32(4))
            if ((wv >> (e & i32(31))) & i32(1)) != i32(0):
                low = (i32(1) << (e & i32(31))) - i32(1)
                rank = lds_ld_i32(
                    L, L_DYN + (i32(NBW) + (e >> i32(5))) * i32(4)
                ) + _ctpop(wv & low)
                lds_st(L, L_DYN + (i32(2 * NBW) + rank) * i32(4), e)

    def _pick_p(nun, C, ps):
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

    def lb_g1c(a, j, bank):
        return ctrl_at(a, i32(CTRL_G1C) + (bank * i32(NCHLB_MAX) + j) * i32(G1C_STRIDE))

    def lb_colc(a, c, bank):
        return ctrl_at(a, i32(CTRL_COLC) + (bank * i32(NCK_MAX) + c) * i32(LRDY_STRIDE))

    def lb_lbr(a, bank, x):
        return ctrl_at(a, i32(CTRL_LBR) + (bank * i32(N_XCD) + x) * i32(LRDY_STRIDE))

    def lb_lbr_wait(a, bank, x):
        nblk = i32(gpu.grid_dim.x)
        spin_sys_ge(lb_lbr(a, bank, x), ceildiv(nblk - x, i32(N_XCD)), a)

    @flyc.jit
    def lb_none(L, tid, a, nun):
        if (nun == i32(0)) & (tid == i32(0)) & (i32(gpu.block_id("x")) == i32(0)):
            lb_lists_wait(a, lb_bank(L))
            for c in range_constexpr(NCK):
                g_st_sys(lrdy_at(a, i32(c)), a["epoch"])

    @flyc.jit
    def lb_col_done(L, a, cc, nun, bank):
        if g_add_agent(lb_colc(a, cc, bank), 1) == nun - i32(1):
            for q_ in range(cg_lo(cc), cg_lo(cc + i32(1)), i32(1)):
                g_st_sys(lrdy_at(a, i32(q_)), a["epoch"])

    @flyc.jit
    def lb_lists_wait(a, bank):
        for x in range_constexpr(N_XCD):
            lb_lbr_wait(a, bank, i32(x))

    def lb_lists(a):
        base = fx.Int64(a["xg"]) + fx.Int64(TMAX * TOPK * (I // 2 + I // 32))
        return rsrc(base), rsrc(base + fx.Int64(TMAX * TOPK * 4))

    def lb_bank(L):
        return lds_ld_i32(L, L_CTL + C_LBB * 4)

    @flyc.jit
    def lb_plan(L, tid, a):
        n = a["ttot"] * i32(TOPK)
        nblk = i32(gpu.grid_dim.x)
        per = ceildiv(n, nblk)
        lo = fx.min(i32(gpu.block_id("x")) * per, n)
        hi = fx.min(lo + per, n)
        if tid == i32(0):
            lds_st(L, L_CTL + C_LBB * 4, g_ld_sys(ctrl_at(a, i32(CTRL_LBSEQ))) & i32(1))
        rid = rsrc(a["ids"], n * i32(4))
        n4 = n // i32(4)
        for r0_ in range(i32(0), n4, i32(NT * LB_B)):
            r0 = i32(r0_)
            vs = [
                fx.Vector(
                    bld(
                        rid,
                        fx.min(r0 + tid + i32(it * NT), n4 - i32(1)) * i32(16),
                        0,
                        V4I,
                        0,
                    )
                )
                for it in range(LB_B)
            ]
            for it in range_constexpr(LB_B):
                q = r0 + tid + i32(it * NT)
                for j in range_constexpr(4):
                    e = fx.Int32(vs[it][j])
                    live, ahead = q < n4, q * i32(4) + i32(j) < lo
                    if live & (masked(e) == fx.Boolean(False)):
                        lds_atomic_add(L, L_DCNT + e * i32(4), 1)
                        if ahead:
                            lds_atomic_add(L, L_DPRE + e * i32(4), 1)
        for idx_ in range(n4 * i32(4) + tid, n, i32(NT)):
            idx = i32(idx_)
            e = fx.Int32(bld(rid, idx * i32(4), 0, T.i32, 0))
            ahead = idx < lo
            if masked(e) == fx.Boolean(False):
                lds_atomic_add(L, L_DCNT + e * i32(4), 1)
                if ahead:
                    lds_atomic_add(L, L_DPRE + e * i32(4), 1)
        cbar(L, tid)
        _lb_bitmap(L, tid)
        cbar(L, tid)
        _dyn_prefix(L, tid)
        cbar(L, tid)
        for it in range_constexpr(ceildiv(E, NT)):
            _dyn_list(L, tid + i32(it * NT))
        cbar(L, tid)
        _chunk_table(L, tid, True)
        lb_scatter(L, tid, a, lo, hi)
        _lb_units(L, tid)
        cbar(L, tid)

    @flyc.jit
    def _lb_bitmap(L, tid):
        lane = tid % i32(64)
        w = tid // i32(64)
        for rnd in range_constexpr(ceildiv(E, NT)):
            e = tid + i32(rnd * NT)
            c = (e < i32(E)).select(
                lds_ld_i32(L, L_DCNT + fx.min(e, i32(E - 1)) * i32(4)), i32(0)
            )
            b = fx.Int64(rocdl.ballot(T.i64, c > i32(0)))
            for h in range_constexpr(2):
                wi = i32(rnd * NT // 32 + h) + w * i32(2)
                if (lane == i32(0)) & (wi < i32(NBW)):
                    lds_st(
                        L,
                        L_DYN + wi * i32(4),
                        i32((b >> fx.Int64(32 * h)) & fx.Int64(0xFFFFFFFF)),
                    )

    @flyc.jit
    def lb_zero_next(tid, a, bank):
        nb = bank ^ i32(1)
        bid = i32(gpu.block_id("x"))
        nblk = i32(gpu.grid_dim.x)
        if tid < i32(64):
            for j_ in range(bid + tid * nblk, i32(NCHLB_MAX), nblk * i32(64)):
                g_st_sys(lb_g1c(a, i32(j_), nb), i32(0))
            if bid == i32(0):
                if tid < i32(NCK):
                    g_st_sys(lb_colc(a, tid, nb), i32(0))
                if (tid >= i32(NCK)) & (tid < i32(NCK + N_XCD)):
                    g_st_sys(lb_lbr(a, nb, tid - i32(NCK)), i32(0))

    def _wave_prefix(v, bits):
        pre, tot = i32(0), i32(0)
        for b in range_constexpr(bits):
            below, cnt = wave_rank(((v >> i32(b)) & i32(1)) == i32(1))
            pre = pre + (below << i32(b))
            tot = tot + (cnt << i32(b))
        return pre, tot

    @flyc.jit
    def _chunk_table(L, tid, offs=False):
        nact = lds_ld_i32(L, L_CTL + C_NACT * 4)
        w = tid // i32(64)
        lane = tid % i32(64)
        base = [i32(0), i32(0)]
        for rnd in range_constexpr(ceildiv(E, NT)):
            j = tid + i32(rnd * NT)
            live = j < nact
            e = live.select(
                lds_ld_i32(L, L_DYN + (i32(2 * NBW) + fx.min(j, i32(E - 1))) * i32(4)),
                i32(0),
            )
            r = lds_ld_i32(L, L_DCNT + e * i32(4))
            r = live.select(r, i32(0)) if offs else r
            ce = fx.min(fx.max(ceildiv(r, i32(rch)), i32(1)), i32(255))
            ce = live.select(ce, i32(0))
            sums = [_wave_prefix(ce, 8)] + (
                [_wave_prefix(r, TMAX.bit_length())] if offs else []
            )
            if lane == i32(0):
                for q in range_constexpr(len(sums)):
                    lds_st(L, L_WSUM + (w + i32(q * NW)) * i32(4), sums[q][1])
            cbar(L, tid)
            off = []
            for q in range_constexpr(len(sums)):
                o, total = base[q] + sums[q][0], i32(0)
                for v in range_constexpr(NW):
                    sv = lds_ld_i32(L, L_WSUM + i32((q * NW + v) * 4))
                    o = o + (i32(v) < w).select(sv, i32(0))
                    total = total + sv
                off.append(o)
                base[q] = base[q] + total
            if const_expr(offs):  # noqa: SIM102
                if live:
                    lds_st(L, L_EOFF + e * i32(4), off[1])
            for c_ in range(i32(0), ce, i32(1)):
                c = i32(c_)
                lds_st(
                    L,
                    L_DCH + (off[0] + c) * i32(4),
                    e | (ce << i32(16)) | (c << i32(24)),
                )
            cbar(L, tid)
        if tid == i32(0):
            lds_st(L, L_CTL + C_NCH * 4, base[0])
        if const_expr(offs):
            cbar(L, tid)

    @flyc.jit
    def lb_scatter(L, tid, a, lo, hi):
        n = a["ttot"] * i32(TOPK)
        rid = rsrc(a["ids"], n * i32(4))
        rtw = rsrc(a["tw"], n * i32(4))
        rl, rw = lb_lists(a)
        for idx_ in range(lo + tid, hi, i32(NT)):
            idx = i32(idx_)
            e = fx.Int32(bld(rid, idx * i32(4), 0, T.i32, 0))
            if masked(e) == fx.Boolean(False):
                pos = lds_atomic_add(L, L_DPRE + e * i32(4), 1)
                slot = lds_ld_i32(L, L_EOFF + e * i32(4)) + pos
                wv = fx.Int32(bld(rtw, idx * i32(4), 0, T.i32, 0))
                bst(idx, rl, slot * i32(4), 0, AUX_SYS)
                bst(wv, rw, slot * i32(4), 0, AUX_SYS)
        wait_vm(0)
        cbar(L, tid)
        if tid == i32(0):
            g_add_agent(lb_lbr(a, lb_bank(L), i32(gpu.block_id("x")) % i32(N_XCD)), 1)

    @flyc.jit
    def _lb_units(L, tid):
        if tid == i32(0):
            nun = lds_ld_i32(L, L_CTL + C_NCH * 4)
            C = i32(gpu.grid_dim.x)
            P = _pick_p(nun, C, LB_PS)
            U = nun * P
            bid = _xcd_rank(C)
            icnt = i32(I) // P
            mix = fx.Boolean(False)
            J2 = nun
            if const_expr(LBMIX):
                X = i32(2) * nun - C
                K = ceildiv(X, i32(2))
                M = i32(2) * (nun - K)
                mix = (X > i32(0)) & (X <= C // i32(LBMIX_DIV)) & (i32(H2) * X <= M)
                P = mix.select(i32(2), P)
                U = mix.select(i32(0), U)
                J2 = mix.select(nun - K, nun)
                if mix:
                    _lb_units_mix(L, bid, C, nun, K, M, X)
            lds_st(L, L_CTL + C_LBJ2 * 4, J2)
            for u_ in range(bid, U, C):
                u = i32(u_)
                j = u // P
                s = u - j * P
                ul = L_CTL + (i32(C_UL) + ((u - bid) // C) * i32(4)) * i32(4)
                lds_st(L, ul, lds_ld_i32(L, L_DCH + j * i32(4)))
                lds_st(L, ul + i32(4), s * icnt)
                lds_st(L, ul + i32(8), icnt)
                lds_st(
                    L,
                    ul + i32(12),
                    i32(UNIT_G1X) | (s << i32(8)) | (j << i32(XQ_SHIFT)),
                )
            mine = (bid < U).select(ceildiv(U - bid, C), i32(0))
            if const_expr(LBMIX):
                mine = mix.select(lds_ld_i32(L, L_CTL + (C_UNIT + 1) * 4), mine)
            lds_st(L, L_CTL + C_DYNP * 4, P)
            lds_st(L, L_CTL + C_UNIT * 4, i32(0))
            lds_st(L, L_CTL + (C_UNIT + 1) * 4, mine)
            lds_st_rel(L, L_CTL + C_PLAN * 4, i32(1))

    def lb_pieces(L, j):
        P = lds_ld_i32(L, L_CTL + C_DYNP * 4)
        if const_expr(LBMIX):
            P = (j >= lds_ld_i32(L, L_CTL + C_LBJ2 * 4)).select(i32(KS2), P)
        return P

    def _lb_ul(L, n, ent, i0, icnt, kind):
        ul = L_CTL + (i32(C_UL) + n * i32(4)) * i32(4)
        lds_st(L, ul, ent)
        lds_st(L, ul + i32(4), i0)
        lds_st(L, ul + i32(8), icnt)
        lds_st(L, ul + i32(12), kind)

    @flyc.jit
    def _lb_units_mix(L, r, C, nun, K, M, X):
        J2 = nun - K
        if r < M:
            j = r // i32(2)
            s_ = r - j * i32(2)
            _lb_ul(
                L,
                i32(0),
                lds_ld_i32(L, L_DCH + j * i32(4)),
                s_ * i32(I // 2),
                i32(I // 2),
                i32(UNIT_G1X) | (s_ << i32(8)) | (j << i32(XQ_SHIFT)),
            )
            lds_st(L, L_CTL + (C_UNIT + 1) * 4, i32(1))
        else:
            for k in range_constexpr(H2):
                f = (r - M) * i32(H2) + i32(k)
                j = J2 + f // i32(KS2)
                s_ = f - (j - J2) * i32(KS2)
                _lb_ul(
                    L,
                    i32(k),
                    lds_ld_i32(L, L_DCH + j * i32(4)),
                    s_ * i32(128),
                    i32(128),
                    i32(UNIT_G1X) | (s_ << i32(8)) | (j << i32(XQ_SHIFT)),
                )
            lds_st(L, L_CTL + (C_UNIT + 1) * 4, i32(H2))
        if r < i32(H2) * X:
            f = (C - M) * i32(H2) + r
            j = J2 + f // i32(KS2)
            s_ = f - (j - J2) * i32(KS2)
            nn = lds_ld_i32(L, L_CTL + (C_UNIT + 1) * 4)
            _lb_ul(
                L,
                nn,
                lds_ld_i32(L, L_DCH + j * i32(4)),
                s_ * i32(128),
                i32(128),
                i32(UNIT_G1X) | (s_ << i32(8)) | (j << i32(XQ_SHIFT)),
            )
            lds_st(L, L_CTL + (C_UNIT + 1) * 4, nn + i32(1))

    @flyc.jit
    def lb_routes(L, tid, a, ent):
        e = ent & i32(0xFFFF)
        r0 = ent.shrui(i32(24)) * i32(rch)
        R = fx.min(lds_ld_i32(L, L_DCNT + e * i32(4)) - r0, i32(rch))
        base = lds_ld_i32(L, L_EOFF + e * i32(4)) + r0
        if (tid < i32(N_XCD)) & (lds_ld_i32(L, L_CTL + C_LBN * 4) == i32(0)):
            lb_lbr_wait(a, lb_bank(L), tid)
        cbar(L, tid)
        if (tid == i32(0)) & (lds_ld_i32(L, L_CTL + C_LBN * 4) == i32(0)):
            lds_st(L, L_CTL + C_LBN * 4, i32(1))
        cbar(L, tid)
        rl, rw = lb_lists(a)
        if tid < R:
            lds_st(
                L,
                L_RIX + tid * i32(4),
                fx.Int32(bld(rl, (base + tid) * i32(4), 0, T.i32, AUX_SYS)),
            )
            lds_st(
                L,
                L_WT + tid * i32(4),
                fx.Int32(bld(rw, (base + tid) * i32(4), 0, T.i32, AUX_SYS)),
            )
        if tid == i32(0):
            lds_st(L, L_CTL + C_GEXP * 4, i32(0))
        cbar(L, tid)
        return R

    def lb_g1_tile(L, tid, a, ent, i0, icnt, rows):
        gemm1(L, tid, a, ent & i32(0xFFFF), i0, icnt // i32(128), rows)
        xq_export(L, tid, a, i0, icnt, i32(0), rows)

    def lb_g2_cap(U2, nblk):
        return ceildiv(U2, nblk) * i32(4) + i32(4)

    @flyc.jit
    def lb_g2_consume(L, tid, a):
        nun = lds_ld_i32(L, L_CTL + C_NCH * 4)
        nblk = i32(gpu.grid_dim.x)
        lb_none(L, tid, a, nun)
        if tid == i32(0):
            lds_st_rel(L, L_CTL + C_G1FIN * 4, i32(1))
        cap = lb_g2_cap(nun * i32(NCG), nblk)
        for it_ in range(i32(0), cap, i32(1)):
            _lb_g2_take(L, tid, a, nun, i32(it_))

    @flyc.jit
    def spin_g2(L, target):
        cur = lds_ld_acq(L, L_CTL + C_G2RDY * 4)
        end = lds_ld_acq(L, L_CTL + C_G2END * 4)
        while (cur < target) & (end == i32(0)):
            rocdl.s_sleep(0)
            cur = lds_ld_acq(L, L_CTL + C_G2RDY * 4)
            end = lds_ld_acq(L, L_CTL + C_G2END * 4)

    @flyc.jit
    def _lb_g2_take(L, tid, a, nun, n):
        if tid == i32(0):
            spin_g2(L, n + i32(1))
        cbar(L, tid)
        if lds_ld_acq(L, L_CTL + C_G2RDY * 4) > n:
            b = n % i32(2)
            u = L_CTL + (i32(C_G2U) + b * i32(4)) * i32(4)
            ent = lds_ld_i32(L, u)
            cc = lds_ld_i32(L, u + i32(4))
            R = lds_ld_i32(L, u + i32(8))
            gemm2(
                L,
                tid,
                a,
                ent & i32(0xFFFF),
                i32(0),
                i32(0),
                R,
                fx.Boolean(False),
                KS2,
                i32(0),
                cg_lo(cc) * i32(GPC),
                cg_lo(cc + i32(1)) * i32(GPC),
                buf=b,
                progress=functools.partial(_g2_last, L, tid, n),
                nxt=n,
            )
            cbar(L, tid)
            if tid == i32(0):
                lds_st_rel(L, L_CTL + C_G2FREE * 4, n + i32(1))
                lb_col_done(L, a, cc, nun, lb_bank(L))

    @flyc.jit
    def _g2_last(L, tid, n):
        if tid == i32(0):
            lds_st_rel(L, L_CTL + C_G2LAST * 4, n + i32(1))

    @flyc.jit
    def lb_g2_prefetch(L, lane, a):
        spin0(L, lane, L_CTL + C_G1FIN * 4, i32(1))
        nun = lds_ld_i32(L, L_CTL + C_NCH * 4)
        U2 = nun * i32(NCG)
        bank = lb_bank(L)
        if (lane < i32(N_XCD)) & (lds_ld_i32(L, L_CTL + C_LBN * 4) == i32(0)):
            lb_lbr_wait(a, bank, lane)
        rocdl.sched_barrier(0)
        cap = lb_g2_cap(U2, i32(gpu.grid_dim.x))
        for it_ in range(i32(0), cap, i32(1)):
            _lb_g2_fetch(L, lane, a, nun, U2, bank, i32(it_))
        if lane == i32(0):
            lds_st_rel(L, L_CTL + C_G2END * 4, i32(1))

    @flyc.jit
    def _lb_g2_fetch(L, lane, a, nun, U2, bank, n):
        if lds_ld_acq(L, L_CTL + C_G2END * 4) == i32(0):
            if (lane == i32(0)) & (n > i32(0)):
                spin_lds_ge(L, L_CTL + C_G2LAST * 4, n)
            rocdl.sched_barrier(0)
            if lane == i32(0):
                lds_st(L, L_CTL + C_G2V * 4, g_add_agent(claim_addr(a, 0), 1))
                lds_st_rel(L, L_CTL + C_G2VN * 4, n + i32(1))
            rocdl.sched_barrier(0)
            v = uni(lds_ld_acq(L, L_CTL + C_G2V * 4))
            if v >= U2:
                if lane == i32(0):
                    lds_st_rel(L, L_CTL + C_G2END * 4, i32(1))
            else:
                _lb_g2_load(L, lane, a, nun, bank, n, v)

    @flyc.jit
    def _lb_g2_load(L, lane, a, nun, bank, n, v):
        b = n % i32(2)
        if lane == i32(0):
            spin_lds_ge(L, L_CTL + C_G2FREE * 4, n - i32(1))
        rocdl.sched_barrier(0)
        cc = v // nun
        j = v - cc * nun
        ent = lds_ld_i32(L, L_DCH + j * i32(4))
        e = ent & i32(0xFFFF)
        r0 = ent.shrui(i32(24)) * i32(rch)
        R = fx.min(lds_ld_i32(L, L_DCNT + e * i32(4)) - r0, i32(rch))
        base = lds_ld_i32(L, L_EOFF + e * i32(4)) + r0
        if lane == i32(0):
            spin_sys_ge(lb_g1c(a, j, bank), lb_pieces(L, j), a)
        rocdl.sched_barrier(0)
        rix0 = i32(L_RIX) + b * i32(128 * 4)
        wt0 = i32(L_WT) + b * i32(128 * 4)
        ib0 = (b == i32(0)).select(i32(L_INTER), i32(L_B1))
        is0 = (b == i32(0)).select(i32(L_INTERS), i32(L_B1S))
        rl, rw = lb_lists(a)
        rv = []
        for k in range_constexpr(ceildiv(RG, 64)):
            r = lane + i32(k * 64)
            rc = fx.min(r, R - i32(1))
            rv.append(
                (
                    r,
                    bld(rl, (base + rc) * i32(4), 0, T.i32, AUX_SYS),
                    bld(rw, (base + rc) * i32(4), 0, T.i32, AUX_SYS),
                )
            )
        for r, iv, wv in rv:
            if r < i32(RG):
                lds_st(L, rix0 + r * i32(4), fx.Int32(iv))
                lds_st(L, wt0 + r * i32(4), fx.Int32(wv))
        wait_lgkm0()
        rx, rxs = xg_rs(a)
        u16 = I // 2 // 16
        nq = fx.max(R * i32(u16), i32(1))
        NQ = ceildiv(RG * u16, 64)
        for b0 in range_constexpr(0, NQ, 8):
            dq = []
            for k in range_constexpr(b0, min(b0 + 8, NQ)):
                q = fx.min(lane + i32(k * 64), nq - i32(1))
                row = q // i32(u16)
                c = q - row * i32(u16)
                rix = lds_ld_i32(L, rix0 + row * i32(4))
                dq.append(
                    (row, c, bld(rx, rix * i32(I // 2) + c * i32(16), 0, V4I, AUX_SYS))
                )
            for row, c, dv in dq:
                lds_st(L, ib0 + row * i32(SI_STRIDE) + c * i32(16), dv, 16)
        ns = fx.max(R * i32(I // 128), i32(1))
        ds = []
        for k in range_constexpr(ceildiv(RG * (I // 128), 64)):
            q = fx.min(lane + i32(k * 64), ns - i32(1))
            row = q // i32(I // 128)
            c = q - row * i32(I // 128)
            rix = lds_ld_i32(L, rix0 + row * i32(4))
            ds.append(
                (row, c, bld(rxs, rix * i32(I // 32) + c * i32(4), 0, T.i32, AUX_SYS))
            )
        for row, c, sv in ds:
            lds_st(L, is0 + row * i32(I // 32) + c * i32(4), fx.Int32(sv))
        wait_lgkm0()
        if lane == i32(0):
            u = L_CTL + (i32(C_G2U) + b * i32(4)) * i32(4)
            lds_st(L, u, ent)
            lds_st(L, u + i32(4), cc)
            lds_st(L, u + i32(8), R)
            lds_st(L, u + i32(12), j)
            lds_st_rel(L, L_CTL + C_G2RDY * 4, n + i32(1))
        rocdl.sched_barrier(0)

    @flyc.jit
    def lb_g2_phase(L, tid, a):
        nun = lds_ld_i32(L, L_CTL + C_NCH * 4)
        U2 = nun * i32(NCG)
        nblk = i32(gpu.grid_dim.x)
        lb_none(L, tid, a, nun)
        cap = ceildiv(U2, nblk) * i32(2) + i32(2)
        col_claim(L, tid, a)
        for it_ in range(i32(0), cap, i32(1)):
            _lb_g2_unit(L, tid, a, nun, U2, i32(it_) < cap - i32(1))

    @flyc.jit
    def _lb_g2_unit(L, tid, a, nun, U2, more):
        v = lds_ld_i32(L, L_CTL + C_CLAIM * 4)
        if v < U2:
            cc = v // nun
            _lb_g2_run(L, tid, a, nun, cc, v - cc * nun)
        if more:
            col_claim(L, tid, a)

    @flyc.jit
    def _lb_g2_run(L, tid, a, nun, cc, j):
        ent = lds_ld_i32(L, L_DCH + j * i32(4))
        bank = lb_bank(L)
        if tid == i32(0):
            spin_sys_ge(lb_g1c(a, j, bank), lb_pieces(L, j), a)
        R = lb_routes(L, tid, a, ent)
        gemm2(
            L,
            tid,
            a,
            ent & i32(0xFFFF),
            i32(0),
            i32(0),
            R,
            fx.Boolean(False),
            KS2,
            i32(0),
            cg_lo(cc) * i32(GPC),
            cg_lo(cc + i32(1)) * i32(GPC),
            pre=functools.partial(lb_import, L, tid, a, R),
        )
        cbar(L, tid)
        if tid == i32(0):
            lb_col_done(L, a, cc, nun, bank)

    @flyc.jit
    def lb_import(L, tid, a, rows):
        ag_stage_free(L, tid % i32(64), a)
        cbar(L, tid)
        u16 = I // 2 // 16
        rx, rxs = xg_rs(a)
        nq = fx.max(rows * i32(u16), i32(1))
        ns = fx.max(rows * i32(I // 128), i32(1))
        dq, ds = [], []
        for k in range_constexpr(ceildiv(RG * u16, NT)):
            q = fx.min(tid + i32(k * NT), nq - i32(1))
            row = q // i32(u16)
            c = q - row * i32(u16)
            rix = lds_ld_i32(L, L_RIX + row * i32(4))
            dq.append(
                (row, c, bld(rx, rix * i32(I // 2) + c * i32(16), 0, V4I, AUX_SYS))
            )
        for k in range_constexpr(ceildiv(RG * (I // 128), NT)):
            q = fx.min(tid + i32(k * NT), ns - i32(1))
            row = q // i32(I // 128)
            c = q - row * i32(I // 128)
            rix = lds_ld_i32(L, L_RIX + row * i32(4))
            ds.append(
                (row, c, bld(rxs, rix * i32(I // 32) + c * i32(4), 0, T.i32, AUX_SYS))
            )
        for row, c, v in dq:
            lds_st(L, i32(L_INTER) + row * i32(SI_STRIDE) + c * i32(16), v, 16)
        for row, c, sv in ds:
            lds_st(L, i32(L_INTERS) + row * i32(I // 32) + c * i32(4), fx.Int32(sv))
        cbar(L, tid)

    @flyc.jit
    def lb_finish(tid, a, L):
        lb_zero_next(tid, a, lb_bank(L))
        if (tid == i32(0)) & (i32(gpu.block_id("x")) == i32(0)):
            lb_lists_wait(a, lb_bank(L))
            g_add_agent(ctrl_at(a, i32(CTRL_LBSEQ)), 1)

    def xg_rs(a):
        rx = rsrc(a["xg"])
        rxs = rsrc(fx.Int64(a["xg"]) + fx.Int64(a["ttot"] * i32(TOPK * (I // 2))))
        return rx, rxs

    @flyc.jit
    def xq_export(L, tid, a, i0, icnt, r0, rows):
        u16 = icnt // i32(32)
        rx, rxs = xg_rs(a)
        for q_ in range(tid, rows * u16, i32(NT)):
            q = i32(q_)
            row = q // u16
            c = q - row * u16
            rix = lds_ld_i32(L, L_RIX + (r0 + row) * i32(4))
            v = lds_ld(L, i32(L_INTER) + row * i32(SI_STRIDE) + c * i32(16), V4I, 16)
            bst(v, rx, rix * i32(I // 2) + i0 // i32(2) + c * i32(16), 0, AUX_SYS)
        nsd = icnt // i32(128)
        for q_ in range(tid, rows * nsd, i32(NT)):
            q = i32(q_)
            row = q // nsd
            c = q - row * nsd
            rix = lds_ld_i32(L, L_RIX + (r0 + row) * i32(4))
            sv = lds_ld_i32(L, i32(L_INTERS) + row * i32(I // 32) + c * i32(4))
            bst(sv, rxs, rix * i32(I // 32) + i0 // i32(32) + c * i32(4), 0, AUX_SYS)
        if (icnt % i32(128)) != i32(0):  # (small-M half block: its 2 scales per row)
            for row_ in range(tid, rows, i32(NT)):
                row = i32(row_)
                rix = lds_ld_i32(L, L_RIX + (r0 + row) * i32(4))
                sv = lds_ld(L, i32(L_INTERS) + row * i32(I // 32), T.i16, 2)
                bst(sv, rxs, rix * i32(I // 32) + i0 // i32(32), 0, AUX_SYS)
        wait_vm(0)

    return locals()
