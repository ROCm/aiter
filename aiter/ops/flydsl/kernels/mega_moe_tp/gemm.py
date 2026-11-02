# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GEMM1 (gate / up + activation + quant of the intermediate into LDS) with its A
loader wave, and GEMM2 (down projection) of a unit into the route regions."""

from __future__ import annotations

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ..kernels_common import ceildiv
from ..mxfp4_gemm_common import _activation_mul_batch, _e8m0_from_amax
from .common import (
    AUX_SC1,
    REL,
    V4I,
    amax,
    bld,
    bst,
    cat8,
    dma4,
    dma16,
    fp4_pack,
    fp8x4_pack,
    i32,
    lds_atomic_add,
    lds_ld,
    lds_ld_acq,
    lds_ld_i32,
    lds_st,
    lds_st_rel,
    mfma,
    pack_bf16x2,
    rsrc,
    spin0,
    spin_lds_ge,
    swap16,
    swap32,
    uni,
    wait_lgkm0,
    wait_vm,
)
from .mega_moe_tp_config import (
    C_AFREE,
    C_ARDY,
    C_ASEQ,
    C_DONE,
    C_G2V,
    C_G2VN,
    C_LPUB,
    C_LQ,
    C_NCH,
    C_PFW,
    C_PLAN,
    C_QBASE,
    C_UL,
    C_UNIT,
    C_UROWS,
    C_USEQ,
    KCS,
    NW,
    OPS,
    SLOT,
    KernelCtx,
    _nsk2_for,
)


def build_gemm(kc: KernelCtx) -> dict:
    """The gemm device functions of one instance (added to kc)."""
    ACB, ADEPTH, AUX_RT, CGW, CH1, CH2 = kc.get("ACB ADEPTH AUX_RT CGW CH1 CH2")
    DLL, G2, GPC, H, I, KS1, L_A, L_AS = kc.get("DLL G2 GPC H I KS1 L_A L_AS")
    L_B1, L_B1S, L_CTL, L_DCH, L_INTER = kc.get("L_B1 L_B1S L_CTL L_DCH L_INTER")
    L_INTERS, L_RING, L_RIX, L_WT, MT = kc.get("L_INTERS L_RING L_RIX L_WT MT")
    MTSKIP, NA_L, NA_ROWOPS, NAB, NCG = kc.get("MTSKIP NA_L NA_ROWOPS NAB NCG")
    NCH, NCK, NSC_BLK, RG, SI_STRIDE = kc.get("NCH NCK NSC_BLK RG SI_STRIDE")
    TOPK, XB, a8, a_frag = kc.get("TOPK XB a8 a_frag")
    act_quant_store, ag_stage_free = kc.get("act_quant_store ag_stage_free")
    ag_wait_chunk, cbar, const_expr = kc.get("ag_wait_chunk cbar const_expr")
    ll_pkt, npp, nsk, route_fp8 = kc.get("ll_pkt npp nsk route_fp8")
    route_region_bytes = kc.route_region_bytes

    @flyc.jit
    def wait_a_chunk(L, lane, buf, q):
        if lane == i32(0):
            spin_lds_ge(L, L_CTL + (C_ASEQ * 4) + buf * i32(4), q + i32(1))
        rocdl.sched_barrier(0)

    @flyc.jit
    def release_a_chunk(L, lane, buf):
        if lane == i32(0):
            lds_atomic_add(L, L_CTL + C_AFREE * 4 + buf * i32(4), 1, REL)

    def g1_group(nnb):
        if const_expr(npp > 1):
            return ((nnb % i32(npp)) == i32(0)).select(i32(npp), i32(1))
        return i32(1)

    @flyc.jit
    def _gemm1_disp(L, tid, a, expert, i0, nnb, mte=None):
        if const_expr(npp > 1):
            if (nnb % i32(npp)) == i32(0):
                _gemm1(L, tid, a, expert, i0, nnb, npp, mte)
            else:
                _gemm1(L, tid, a, expert, i0, nnb, 1, mte)
        else:
            _gemm1(L, tid, a, expert, i0, nnb, 1, mte)

    @flyc.jit
    def gemm1(L, tid, a, expert, i0, nnb, rows=None):
        nnb = uni(nnb)
        if const_expr(MTSKIP and rows is not None):
            if uni(rows) <= i32(3 * 16):
                _gemm1_disp(L, tid, a, expert, i0, nnb, 3)
            else:
                _gemm1_disp(L, tid, a, expert, i0, nnb)
        else:
            _gemm1_disp(L, tid, a, expert, i0, nnb)
        wait_vm(0)
        cbar(L, tid)

    def _g1_operands(tid, a, expert, i0):
        lane = tid % i32(64)
        w = uni(tid // i32(64))
        e = uni(expert)
        rw = rsrc(fx.Int64(a["w1"]) + fx.Int64(e) * fx.Int64(2 * I * (H // 2)))
        rws = rsrc(a["w1s"])
        icol0 = uni(i0) + w * i32(32)
        vg0 = icol0 * i32(H // 2) + lane * i32(16)
        vg1 = vg0 + i32(16 * (H // 2))
        vu0 = vg0 + i32(I * (H // 2))
        vu1 = vu0 + i32(16 * (H // 2))
        vsl = lane * i32(4)
        sg0 = uni((e * i32(2 * I) + icol0) // i32(32)) * i32(CH1 * 256)
        su0 = uni((e * i32(2 * I) + i32(I) + icol0) // i32(32)) * i32(CH1 * 256)
        return lane, w, rw, rws, (vg0, vg1, vu0, vu1), vsl, sg0, su0

    def _g1_where(g, total, P):
        gc = fx.min(g, total - i32(1))
        grp = gc // i32(KS1 * P)
        r = gc - grp * i32(KS1 * P)
        kk = r // i32(P)
        return grp * i32(P) + (r - kk * i32(P)), kk

    def _g1_so(pas, kk):
        return pas * i32(128 * (H // 2)) + kk * i32(1024)

    def _g1_sso(pas, kk):
        return pas * i32((128 // 32) * CH1 * 256) + (kk // i32(2)) * i32(256)

    def _g1_chunk(L, lane, qb, ks):
        k = ks % KCS
        q = qb + i32(ks // KCS)
        buf = q % i32(NAB)
        rocdl.sched_barrier(0)
        if k == 0:
            wait_a_chunk(L, lane, buf, q)
        abuf = L + i32(L_A) + buf * i32(RG * ACB)
        asbuf = (
            L
            + i32(L_AS)
            + buf * i32(KCS * NSC_BLK * 64 * 4)
            + i32(k * NSC_BLK * 64 * 4)
        )
        return k, buf, abuf, asbuf

    def _g1_mfma(acc, o, b, sgw, suw, kh, abuf, asbuf, lane, k, MTE):
        sc = [
            (x >> i32(8 * (kh * 2 + h))) & i32(0xFF) for x in (sgw, suw) for h in (0, 1)
        ]
        # all A fragments / scales of the k-step first, then the MFMAs: the LDS reads
        # overlap instead of one exposed lgkmcnt(0) wait per 16-row block
        afs, sas = [], []
        for m in range_constexpr(MTE):
            row = i32(m * 16) + lane % i32(16)
            afs.append(a_frag(abuf, row, k, lane // i32(16)))
            sas.append(
                fx.Int32(lds_ld(asbuf, row * i32(4) + lane // i32(16), T.i8, 1))
                & i32(0xFF)
            )
        for m in range_constexpr(MTE):
            for t in range_constexpr(4):
                j = o + (t // 2) * 2 * MTE + m * 2 + t % 2
                acc[j] = mfma(b[t], afs[m], acc[j], sc[t], sas[m], a8)

    def _g1_out(L, lane, w, a, acc, grp, P, MTE):
        ag_stage_free(L, lane, a)
        for pp in range_constexpr(P):
            o = pp * 4 * MTE
            act_quant_store(
                L,
                lane,
                w,
                acc[o : o + 2 * MTE],
                acc[o + 2 * MTE : o + 4 * MTE],
                grp * i32(P) + i32(pp),
                MTE,
            )

    @flyc.jit
    def _gemm1(L, tid, a, expert, i0, nnb, P, MTE=None):
        MTE = MT if MTE is None else MTE
        lane, w, rw, rws, vs, vsl, sg0, su0 = _g1_operands(tid, a, expert, i0)
        ring = uni(L + i32(L_RING) + w * i32(nsk * SLOT))
        total = nnb * i32(KS1)
        qbase = lds_ld_i32(L, L_CTL + C_QBASE * 4 + w * i32(4))
        GS = KS1 * P
        IL = nsk * (P * KCS) // math.gcd(nsk, P * KCS)
        assert GS % IL == 0

        def issue_b(g, slot_idx):
            pas, kk = _g1_where(g, total, P)
            slot = ring + i32(slot_idx * SLOT)
            so = _g1_so(pas, kk)
            for t in range_constexpr(4):
                dma16(slot + i32(t * 1024), rw, vs[t], so, nt=True)
            sso = _g1_sso(pas, kk)
            dma4(slot + i32(4096), rws, vsl, sg0 + sso)
            dma4(slot + i32(4096 + 256), rws, vsl, su0 + sso)

        for kk in range_constexpr(nsk):
            issue_b(i32(kk), kk)
        zero = fx.full(4, 0.0, fx.Float32)
        for grp_ in range(i32(0), nnb // i32(P), i32(1)):
            grp = i32(grp_)
            init = [zero] * (4 * MTE * P)
            for g0_, st in range(
                grp * i32(GS), (grp + i32(1)) * i32(GS), i32(IL), init=init
            ):
                g0 = i32(g0_)
                acc = list(st)
                qb = qbase + grp * i32(NCH) + (g0 - grp * i32(GS)) // i32(P * KCS)
                for ks in range_constexpr(IL // P):
                    k, buf, abuf, asbuf = _g1_chunk(L, lane, qb, ks)
                    for pp in range_constexpr(P):
                        s_ = ks * P + pp
                        g = g0 + i32(s_)
                        wait_vm((nsk - 1) * OPS)
                        slot = ring + i32((s_ % nsk) * SLOT)
                        b = [
                            lds_ld(slot, i32(t * 1024) + lane * i32(16), V4I, 16)
                            for t in range(4)
                        ]
                        sgw = lds_ld_i32(slot, i32(4096) + lane * i32(4))
                        suw = lds_ld_i32(slot, i32(4096 + 256) + lane * i32(4))
                        _g1_mfma(
                            acc,
                            pp * 4 * MTE,
                            b,
                            sgw,
                            suw,
                            ks & 1,
                            abuf,
                            asbuf,
                            lane,
                            k,
                            MTE,
                        )
                        wait_lgkm0()
                        rocdl.sched_barrier(0)
                        issue_b(g + i32(nsk), s_ % nsk)
                        rocdl.sched_barrier(0)
                    if k == KCS - 1:
                        release_a_chunk(L, lane, buf)
                res = yield acc
            _g1_out(L, lane, w, a, res, grp, P, MTE)
        qend = qbase + (nnb // i32(P)) * i32(NCH)
        if lane == i32(0):
            lds_st(L, L_CTL + C_QBASE * 4 + w * i32(4), qend)

    def unit_fields(L, u):
        f = [
            lds_ld_i32(L, L_CTL + (i32(C_UL) + u * i32(4) + i32(k)) * i32(4))
            for k in range(4)
        ]
        return f[0], f[1], f[2], f[3]

    @flyc.jit
    def a_loader(L, tid, a):
        lane = tid % i32(64)
        rx = rsrc(a["ax"])
        rxs = rsrc(a["axs"])
        spin0(L, lane, L_CTL + C_PLAN * 4, i32(1))
        ub = uni(lds_ld_acq(L, L_CTL + C_UNIT * 4))
        ue = uni(lds_ld_acq(L, L_CTL + (C_UNIT + 1) * 4))
        for u_ in range(ub, ue, i32(1)):
            u = i32(u_)
            if lane == i32(0):
                spin_lds_ge(L, L_CTL + C_USEQ * 4, u - ub + i32(1))
            _a_unit(L, lane, a, rx, rxs, u)

    @flyc.jit
    def _a_unit(L, lane, a, rx, rxs, u):
        R = uni(lds_ld_acq(L, L_CTL + C_UROWS * 4))
        nnb = unit_fields(L, u)[2] // i32(128)
        for r0_ in range(i32(0), R, i32(RG)):
            r0 = i32(r0_)
            rows = fx.min(R - r0, i32(RG))
            arow = []
            for j in range_constexpr(NA_ROWOPS):
                if const_expr(a8):
                    row = i32(j * 4) + lane // i32(16)
                    col = (lane % i32(16)) ^ (row & i32(7))
                else:
                    row = i32(j * 8) + lane // i32(8)
                    col = (lane % i32(8)) ^ (row & i32(7))
                rr = (row < rows).select(row, i32(0))
                arow.append(
                    (lds_ld_i32(L, L_RIX + (r0 + rr) * i32(4)) // i32(TOPK)) * i32(XB)
                    + col * i32(16)
                )
            ascl = []
            for j in range_constexpr(NSC_BLK):
                row = i32(j * 64) + lane
                rr = (row < rows).select(row, i32(0))
                ascl.append(
                    (lds_ld_i32(L, L_RIX + (r0 + rr) * i32(4)) // i32(TOPK))
                    * i32(H // 32)
                )
            nq = (nnb // g1_group(nnb)) * i32(NCH)
            if lane == i32(0):
                lds_st(L, L_CTL + C_ARDY * 4, i32(0))
            for cidx_ in range(i32(0), nq, i32(1)):
                cidx = i32(cidx_)
                q = lds_ld_i32(L, L_CTL + C_LQ * 4)
                b = q % i32(NAB)
                if (lane == i32(0)) & (q >= i32(NAB)):
                    spin_lds_ge(
                        L,
                        L_CTL + C_AFREE * 4 + b * i32(4),
                        i32(NW) * (q // i32(NAB)),
                    )
                rocdl.sched_barrier(0)
                cc = cidx % i32(NCH)
                if cidx < i32(NCH):  # noqa: SIM102
                    if cidx >= uni(lds_ld_i32(L, L_CTL + C_ARDY * 4)):
                        ag_wait_chunk(L, lane, a, a["epoch"], cc)
                abase = L + i32(L_A) + b * i32(RG * ACB)
                for j in range_constexpr(NA_ROWOPS):
                    dma16(abase + i32(j * 1024), rx, arow[j], cc * i32(ACB))
                for k in range_constexpr(KCS):
                    for j in range_constexpr(NSC_BLK):
                        dst = (
                            L
                            + i32(L_AS)
                            + b * i32(KCS * NSC_BLK * 64 * 4)
                            + i32((k * NSC_BLK + j) * 64 * 4)
                        )
                        dma4(dst, rxs, ascl[j], cc * i32(8) + i32(k * 4), sys=True)
                _loader_advance(L, lane, q)
            wait_vm(0)
            _loader_flush(L, lane)

    @flyc.jit
    def _loader_advance(L, lane, q):
        pub = lds_ld_i32(L, L_CTL + C_LPUB * 4)
        if q + i32(1) - pub >= i32(ADEPTH):
            wait_vm(NA_L * (ADEPTH - 1))
            if lane == i32(0):
                lds_st_rel(
                    L, L_CTL + C_ASEQ * 4 + (pub % i32(NAB)) * i32(4), pub + i32(1)
                )
                lds_st(L, L_CTL + C_LPUB * 4, pub + i32(1))
        if lane == i32(0):
            lds_st(L, L_CTL + C_LQ * 4, q + i32(1))
        rocdl.sched_barrier(0)

    @flyc.jit
    def _loader_flush(L, lane):
        if lane == i32(0):
            pub = lds_ld_i32(L, L_CTL + C_LPUB * 4)
            q = lds_ld_i32(L, L_CTL + C_LQ * 4)
            for p_ in range(pub, q, i32(1)):
                p = i32(p_)
                lds_st_rel(L, L_CTL + C_ASEQ * 4 + (p % i32(NAB)) * i32(4), p + i32(1))
            lds_st(L, L_CTL + C_LPUB * 4, q)
        rocdl.sched_barrier(0)

    @flyc.jit
    def report_chunk(L, lane, w, cidx):
        if lane == i32(0):
            lds_st_rel(L, L_CTL + C_DONE * 4 + w * i32(4), cidx)

    @flyc.jit
    def maybe_report(L, lane, w, gi, signal, rlag_wait, lag=1):
        if (
            signal
            & (((gi + i32(1)) % i32(GPC)) == i32(0))
            & (gi + i32(1) > i32(lag * GPC))
        ):
            wait_vm(rlag_wait)
            report_chunk(L, lane, w, (gi // i32(GPC)) - i32(lag))

    def store_route_fp8(rs, accs, wt, rix, n0, q4, ok, oob, a):
        d, e8s = [], []
        for half in range_constexpr(2):
            f = [
                fx.Float32(fx.Vector(accs[2 * half + h])[i]) * wt
                for h in range(2)
                for i in range(4)
            ]
            am = amax(f)
            am = am.maximumf(am.shuffle_xor(i32(16), i32(64)))
            am = am.maximumf(am.shuffle_xor(i32(32), i32(64)))
            e8, qs = _e8m0_from_amax(am, max_norm=448.0)
            d += [fp8x4_pack(f[0:4], qs), fp8x4_pack(f[4:8], qs)]
            e8s.append(fx.Int32(e8) & i32(0xFF))
        d0, d1 = swap16(d[0], d[1])
        d2, d3 = swap16(d[2], d[3])
        d0, d2 = swap32(d0, d2)
        d1, d3 = swap32(d1, d3)
        lo, hi = [d0, d1], [d2, d3]
        col = n0 + q4 * i32(16)
        if const_expr(DLL):
            e8 = ((q4 >> i32(1)) == i32(0)).select(e8s[0], e8s[1])
            off = (rix * i32(H) + col) * i32(2)
            for h in range_constexpr(2):
                d = lo if h == 0 else hi
                pkt = ll_pkt(a, d[0], d[1], e8)
                bst(pkt, rs, ok.select(off + i32(16 * h), oob), 0, AUX_SC1)
            return
        ov = fx.Vector.from_elements(lo + hi, fx.Int32)
        bst(ov, rs, ok.select(rix * i32(H) + col, oob), 0, AUX_RT)
        sc = fx.Int16(e8s[0] | (e8s[1] << i32(8)))
        soff = a["ttot"] * i32(TOPK * H) + rix * i32(H // 32) + n0 // i32(32)
        bst(sc, rs, (ok & (q4 == i32(0))).select(soff, oob), 0, AUX_RT)

    def _w2_off(n0, t, k):
        return (n0 + i32(t * 16)) * i32(I // 2) + k * i32(1024)

    def g2_slot_dma(a, lane, w, slot, e, gi, k):
        n0 = (w + i32(NW) * gi) * i32(64)
        for t in range_constexpr(4):
            dma16(
                slot + i32(t * 1024),
                rsrc(a["w2"]),
                lane * i32(16),
                e * i32(H * (I // 2))
                + (n0 + i32(t * 16)) * i32(I // 2)
                + k * i32(1024),
                nt=True,
            )
        for p in range_constexpr(2):
            rb = (e * i32(H) + n0 + i32(p * 32)) // i32(32)
            dma4(
                slot + i32(4096 + p * 256),
                rsrc(a["w2s"]),
                lane * i32(4),
                (rb * i32(CH2) + k // i32(2)) * i32(256),
            )

    @flyc.jit
    def gemm2(
        L, tid, a, expert, ks0, r0, rows, signal, NKS, pidx, gi_lo=None, *args, **kw
    ):
        g = functools.partial(
            _gemm2,
            L,
            tid,
            a,
            expert,
            ks0,
            r0,
            rows,
            signal,
            NKS,
            pidx,
            gi_lo,
            *args,
            **kw,
        )
        if const_expr(MTSKIP):
            if uni(rows) <= i32(3 * 16):
                g(mte=3)
            else:
                g()
        else:
            g()

    @flyc.jit
    def _gemm2(
        L,
        tid,
        a,
        expert,
        ks0,
        r0,
        rows,
        signal,
        NKS,
        pidx,
        gi_lo=None,
        gi_hi=None,
        pre=None,
        buf=None,
        progress=None,
        mte=None,
        nxt=None,
    ):
        MTE = MT if mte is None else mte
        if const_expr(buf is None):
            ib0, is0 = i32(L_INTER), i32(L_INTERS)
            rix0, wt0 = i32(L_RIX), i32(L_WT)
        else:
            b0 = buf == i32(0)
            ib0 = b0.select(i32(L_INTER), i32(L_B1))
            is0 = b0.select(i32(L_INTERS), i32(L_B1S))
            rix0 = i32(L_RIX) + buf * i32(128 * 4)
            wt0 = i32(L_WT) + buf * i32(128 * 4)
        NSK2 = _nsk2_for(NKS, G2)
        GPI = (NSK2 * NKS // math.gcd(NSK2, NKS)) // NKS
        assert G2 % GPI == 0
        WAIT_B2 = (NSK2 - 1) * OPS
        LAG = max(1, ceildiv(NSK2, GPC * NKS))
        RLAG = GPC * NKS * OPS
        lane = tid % i32(64)
        w = uni(tid // i32(64))
        e = uni(expert)
        ks0 = uni(ks0)
        rw = rsrc(fx.Int64(a["w2"]) + fx.Int64(e) * fx.Int64(H * (I // 2)))
        rws = rsrc(a["w2s"])
        routes_bytes = route_region_bytes(a["ttot"])
        pidx = uni(pidx)
        dst = (pidx == i32(0)).select(
            fx.Int64(a["routes"]),
            fx.Int64(a["proutes"]) + fx.Int64(pidx - i32(1)) * fx.Int64(routes_bytes),
        )
        r_routes = rsrc(dst, routes_bytes)
        oob = routes_bytes
        ring = uni(L + i32(L_RING) + w * i32(nsk * SLOT))
        q4 = lane // i32(16)
        g_lo = i32(0) if gi_lo is None else uni(gi_lo)
        g_hi = i32(G2) if gi_hi is None else uni(gi_hi)

        XPF = nxt is not None

        def issue_x(q, slot_idx):
            tq = g_hi * i32(NKS)
            if q == tq:
                vn = lds_ld_acq(L, L_CTL + C_G2VN * 4)
                ok = (vn == nxt + i32(2)) & (
                    lds_ld_i32(L, L_CTL + C_G2V * 4)
                    < lds_ld_i32(L, L_CTL + C_NCH * 4) * i32(NCG)
                )
                if lane == i32(0):
                    lds_st(L, L_CTL + (C_PFW + w) * i32(4), ok.select(i32(1), i32(0)))
            pf = lds_ld_i32(L, L_CTL + (C_PFW + w) * i32(4)) == i32(1)
            nun_ = fx.max(lds_ld_i32(L, L_CTL + C_NCH * 4), i32(1))
            v = lds_ld_i32(L, L_CTL + C_G2V * 4)
            cc_ = v // nun_
            ent_ = lds_ld_i32(L, L_DCH + fx.min(v - cc_ * nun_, nun_ - i32(1)) * i32(4))
            ee = uni(pf.select(ent_ & i32(0xFFFF), e))
            qc = uni(pf.select(q - tq + cg_lo(cc_) * i32(GPC * NKS), tq - i32(1)))
            gi = qc // i32(NKS)
            g2_slot_dma(
                a,
                lane,
                w,
                ring + i32(slot_idx * SLOT),
                ee,
                gi,
                ks0 + qc - gi * i32(NKS),
            )

        def issue(q, slot_idx):
            if const_expr(XPF):
                if q >= g_hi * i32(NKS):
                    issue_x(q, slot_idx)
                else:
                    issue_n(q, slot_idx)
            else:
                issue_n(q, slot_idx)

        def issue_n(q, slot_idx):
            qc = fx.min(q, g_hi * i32(NKS) - i32(1))
            gi = qc // i32(NKS)
            k = ks0 + qc - gi * i32(NKS)
            n0 = (w + i32(NW) * gi) * i32(64)
            slot = ring + i32(slot_idx * SLOT)
            for t in range_constexpr(4):
                dma16(
                    slot + i32(t * 1024), rw, lane * i32(16), _w2_off(n0, t, k), nt=True
                )
            for p in range_constexpr(2):
                rb = (e * i32(H) + n0 + i32(p * 32)) // i32(32)
                dma4(
                    slot + i32(4096 + p * 256),
                    rws,
                    lane * i32(4),
                    (rb * i32(CH2) + k // i32(2)) * i32(256),
                )

        if const_expr(XPF):
            if lds_ld_i32(L, L_CTL + (C_PFW + w) * i32(4)) == i32(0):
                for qq in range_constexpr(NSK2):
                    issue(g_lo * i32(NKS) + i32(qq), qq)
            if lane == i32(0):
                lds_st(L, L_CTL + (C_PFW + w) * i32(4), i32(0))
        else:
            for qq in range_constexpr(NSK2):
                issue(g_lo * i32(NKS) + i32(qq), qq)
        if const_expr(pre is not None):
            pre()
        zero = fx.full(4, 0.0, fx.Float32)

        row_meta = []
        for m in range_constexpr(MTE):
            R = i32(m * 16) + lane % i32(16)
            ok = R < rows
            Rc = ok.select(R, i32(0))
            row_meta.append(
                (
                    ok,
                    fx.Float32(lds_ld(L, wt0 + (r0 + Rc) * i32(4), T.f32)),
                    lds_ld_i32(L, rix0 + (r0 + Rc) * i32(4)),
                )
            )

        def groups(gi0):
            for j in range_constexpr(GPI):
                gi = gi0 + i32(j)
                acc = [zero] * (4 * MTE)
                for k in range_constexpr(NKS):
                    sidx = (j * NKS + k) % NSK2
                    if const_expr(k < NSK2 and NKS >= NSK2):
                        if const_expr(j > 0):  # noqa: SIM114
                            wait_vm(WAIT_B2 + 2 * MTE)
                        elif gi > g_lo:
                            wait_vm(WAIT_B2 + 2 * MTE)
                        else:
                            wait_vm(WAIT_B2)
                    else:
                        wait_vm(WAIT_B2)
                    slot = ring + i32(sidx * SLOT)
                    b = [
                        lds_ld(slot, i32(t * 1024) + lane * i32(16), V4I, 16)
                        for t in range(4)
                    ]
                    s0 = lds_ld_i32(slot, i32(4096) + lane * i32(4))
                    s1 = lds_ld_i32(slot, i32(4096 + 256) + lane * i32(4))
                    sh0 = ((ks0 + i32(k)) & i32(1)) * i32(16)
                    sb = [
                        (s0 >> sh0) & i32(0xFF),
                        (s0 >> (sh0 + i32(8))) & i32(0xFF),
                        (s1 >> sh0) & i32(0xFF),
                        (s1 >> (sh0 + i32(8))) & i32(0xFF),
                    ]
                    af_l, sa_l = [], []
                    for m in range_constexpr(MTE):
                        row = i32(m * 16) + lane % i32(16)
                        ib = ib0 + row * i32(SI_STRIDE)
                        if const_expr(a8):
                            kb = ib + i32(k * 128) + q4 * i32(16)
                            af_l.append(
                                cat8(
                                    lds_ld(L, kb, V4I, 16),
                                    lds_ld(L, kb + i32(64), V4I, 16),
                                )
                            )
                        else:
                            af_l.append(
                                lds_ld(L, ib + i32(k * 64) + q4 * i32(16), V4I, 16)
                            )
                        sa_l.append(
                            fx.Int32(
                                lds_ld(
                                    L,
                                    is0 + row * i32(I // 32) + i32(k * 4) + q4,
                                    T.i8,
                                    1,
                                )
                            )
                            & i32(0xFF)
                        )
                    rocdl.sched_barrier(0)
                    rocdl.s_setprio(1)
                    for m in range_constexpr(MTE):
                        for t in range_constexpr(4):
                            acc[m * 4 + t] = mfma(
                                b[t], af_l[m], acc[m * 4 + t], sb[t], sa_l[m], a8
                            )
                    rocdl.s_setprio(0)
                    wait_lgkm0()
                    rocdl.sched_barrier(0)
                    issue(gi * i32(NKS) + i32(k + NSK2), sidx)
                    rocdl.sched_barrier(0)
                n0 = (w + i32(NW) * gi) * i32(64)
                ccol = (q4 & i32(1)) * i32(16) + (q4 >> i32(1)) * i32(8)
                for m in range_constexpr(MTE):
                    ok, wt, rix = row_meta[m]
                    if const_expr(route_fp8):
                        store_route_fp8(
                            r_routes,
                            acc[m * 4 : m * 4 + 4],
                            wt,
                            rix,
                            n0,
                            q4,
                            ok,
                            oob,
                            a,
                        )
                    else:
                        pk = []
                        for t in range_constexpr(4):
                            av = fx.Vector(acc[m * 4 + t])
                            pk.append(
                                [
                                    pack_bf16x2(
                                        fx.Float32(av[2 * h]) * wt,
                                        fx.Float32(av[2 * h + 1]) * wt,
                                    )
                                    for h in range(2)
                                ]
                            )
                        for half in range_constexpr(2):
                            ta, tb = 2 * half, 2 * half + 1
                            lo, hi = [], []
                            for h in range_constexpr(2):
                                x, y = swap16(pk[ta][h], pk[tb][h])
                                lo.append(x)
                                hi.append(y)
                            ov = fx.Vector.from_elements(lo + hi, fx.Int32)
                            col = n0 + i32(half * 32) + ccol
                            voff = (rix * i32(H) + col) * i32(2)
                            bst(ov, r_routes, ok.select(voff, oob), 0, AUX_RT)
                if const_expr(not DLL):
                    maybe_report(L, lane, w, gi, signal, LAG * RLAG, LAG)

        for gi0_ in range(g_lo, g_hi, i32(GPI)):
            if const_expr(progress is not None):  # noqa: SIM102
                if i32(gi0_) + i32(GPI) >= g_hi:
                    progress()
            groups(i32(gi0_))
        wait_vm(0)
        if const_expr(not DLL):  # noqa: SIM102
            if signal:
                report_chunk(L, lane, w, i32(NCK - 1))

    def cg_lo(cc):
        v = i32(0)
        for k in range_constexpr(1, NCG + 1):
            v = (cc >= i32(k)).select(i32(sum(CGW[:k])), v)
        return v

    return locals()


# ---- big-M fused units (compile_bigm_kernel) ----
const_expr = fx.const_expr

KCK = 2  # GEMM1 k-steps (of 128) per chunk


def bigm_lds(RG, NAB, I):
    """LDS layout (bytes) of one CTA."""
    c = {}
    o = 0

    def take(n, align=16):
        nonlocal o
        o = (o + align - 1) // align * align
        r = o
        o += n
        return r

    # A codes: [buf][row][128 B], 16 B columns swizzled
    c["LA"] = take(NAB * RG * 64 * KCK)
    c["LAS"] = take(NAB * NW * 64 * 4)  # A scales: [buf][wave][lane] words (dma4 order)
    c["LRIX"] = take(RG * 4)
    c["LWT"] = take(RG * 4)
    c["SIS"] = I // 2 + 16
    c["LINT"] = take(RG * c["SIS"])  # intermediate MXFP4 rows
    c["ISS"] = I // 32
    c["LINTS"] = take(RG * c["ISS"])  # their E8M0 scales
    c["LCTL"] = take(64)
    c["BYTES"] = (o + 15) // 16 * 16
    return c


def build_bigm_unit(H, I, TOPK, RG, NAB, BST, BS2, act, swiglu_limit, R8):
    """The fused unit device function: unit_body(L, tid, e, r0, rows, rr, rwt, rx, rxs, w1,
    w2, rws1, rws2, r_routes, oob): rows rr / rwt[r0 : r0 + rows] (route index, weight) of expert
    e -> route rows wt * GEMM2(act(GEMM1(A[route // TOPK]))) stored at route * H (bf16).
    R8: route rows as MXFP8 (e4m3 + one e8m0 scale per 32 columns): codes [rows][H] at r_routes,
    scales [rows][H / 32] after them (at oob / 2: oob = the route count * H * 2).
    Returns (LDS layout, unit_body)."""
    assert RG % 16 == 0 and RG <= 128 and H % 256 == 0 and I % 128 == 0
    MTE = RG // 16  # 16-row tiles (every wave computes every row of the unit)
    KH = H // 2  # A / w1 row bytes
    KCH = H // (128 * KCK)  # GEMM1 chunks (KCK k-steps of 128)
    RBA = 64 * KCK  # A row bytes per chunk
    LPR = RBA // 16  # DMA lanes per A row (16 B each)
    RPJ = 64 // LPR  # A rows per dma16
    PASSES = I // (NW * 32)  # GEMM1 passes
    NG2 = H // (NW * 64)  # GEMM2 column groups per wave
    KS2 = I // 128  # GEMM2 k-steps
    assert I % (NW * 32) == 0
    CH1, CH2 = ceildiv(H // 32, 8), ceildiv(I // 32, 8)
    assert KCH % NAB == 0 and KCH % BST == 0 and KS2 % BS2 == 0
    lay = bigm_lds(RG, NAB, I)
    ISS = lay["ISS"]
    LA, LAS, LRIX, LWT = lay["LA"], lay["LAS"], lay["LRIX"], lay["LWT"]
    SIS, LINT, LINTS = lay["SIS"], lay["LINT"], lay["LINTS"]
    AR = RG // NW  # A rows per wave (DMA)
    assert AR in (16, 32) and 64 % (AR * KCK) == 0
    NAJ = AR // RPJ  # dma16 per wave per chunk (RPJ rows x RBA B each)
    NA = NAJ + 1  # A ops per wave per chunk: dma16s + 1 x dma4 (scales)
    # B ops per wave per chunk: 2 k-steps x 4 tiles + gate / up scale words
    NB = 4 * KCK + 2
    # sched barrier mask after the GEMM2 B issue groups (only VMEM is held)
    SCM = 0x38F
    NB2 = 6  # GEMM2 B ops per k-step: 4 tiles + 2 scale words per 64 columns
    NST2 = 2 * MTE  # route stores per wave per group
    # steady-state GEMM1 issue order per chunk c: B(c + BST), then A(c + NAB - 1). By the
    # count alone A(c) has landed once (NA + NB) * (NAB - 2) newer ops remain, but its LDS
    # writes (direct-to-LDS loads) can still be in flight then: one chunk's ops more are
    # awaited (else a unit now and then reads stale A rows of a chunk, small units mostly)
    WAIT1 = max(min(NA + (NA + NB) * (BST - 1), (NA + NB) * (NAB - 3)), 0)
    assert WAIT1 <= 63

    def _wait2(kk):
        # GEMM2: B2 of step s was issued BS2 steps earlier; every step issues NB2 B2 ops and
        # the last k-step of a group then NST2 route stores
        n = 0
        for j in range(1, BS2 + 1):
            pos = (kk - j) % KS2
            if j < BS2:
                n += NB2
            if pos == KS2 - 1:
                n += NST2
        return min(n, 63)

    WAIT2 = [_wait2(kk) for kk in range(KS2)]

    def act_batch(gs, us):
        return _activation_mul_batch(gs, us, act=act, swiglu_limit=swiglu_limit)

    @flyc.jit
    def part_fill(L, tid, r0, rows, rr, rwt):
        gpu.barrier()
        if tid < i32(RG):
            rc = (tid < rows).select(tid, i32(0))
            lds_st(
                L,
                i32(LRIX) + tid * i32(4),
                fx.Int32(bld(rr, (r0 + rc) * i32(4), 0, T.i32)),
            )
            lds_st(
                L,
                i32(LWT) + tid * i32(4),
                fx.Int32(bld(rwt, (r0 + rc) * i32(4), 0, T.i32)),
            )
        gpu.barrier()

    @flyc.jit
    def part_g1(L, tid, e, rx, rxs, w1, rws1):
        lane = tid % i32(64)
        w = rocdl.readfirstlane(T.i32, tid // i32(64))  # the wave: its column group
        q4 = lane // i32(16)
        zero = fx.Vector.filled(4, 0.0, fx.Float32)
        # ---- A operand addressing (this wave's rows) ----
        arow = []
        for j in range_constexpr(NAJ):
            row = w * i32(AR) + i32(j * RPJ) + lane // i32(LPR)
            col = (lane % i32(LPR)) ^ (row & i32(LPR - 1))
            tok = lds_ld_i32(L, i32(LRIX) + row * i32(4)) // i32(TOPK)
            arow.append(tok * i32(KH) + col * i32(16))
        # scales: lane -> (row w * AR + lane % AR, k-step (lane // AR) % KCK)
        srow = w * i32(AR) + lane % i32(AR)
        stok = lds_ld_i32(L, i32(LRIX) + srow * i32(4)) // i32(TOPK)
        soff = stok * i32(H // 32) + ((lane // i32(AR)) % i32(KCK)) * i32(4)

        def issue_a(c, buf):
            base = L + i32(LA + buf * RG * RBA)
            for j in range_constexpr(NAJ):
                dma16(
                    base + (w * i32(AR) + i32(j * RPJ)) * i32(RBA),
                    rx,
                    arow[j],
                    c * i32(RBA),
                )
            dma4(
                L + i32(LAS + buf * NW * 256) + w * i32(256),
                rxs,
                soff,
                c * i32(4 * KCK),
            )

        # one per-lane base + compile-time offsets (row tile m, k-step, ring buffer) for the A
        # fragment / scale reads (one wave's rows are every row: the swizzle depends on the lane
        # only), instead of per-tile address registers (spilled under the register pressure)
        l16 = lane % i32(16)
        swz = l16 & i32(LPR - 1)
        abase = [
            L + i32(LA) + l16 * i32(RBA) + ((i32(k * 4) + q4) ^ swz) * i32(16)
            for k in range(KCK)
        ]
        sbase = L + i32(LAS) + l16 * i32(4) + q4

        def a_frag_m(buf, m, k):
            return lds_ld(abase[k], buf * RG * RBA + m * 16 * RBA, V4I, 16)

        def a_scale_m(buf, m, k):
            wd0 = (m * 16) // AR * 64 + k * AR + (m * 16) % AR
            return fx.Int32(lds_ld(sbase, buf * NW * 256 + wd0 * 4, T.i8, 1)) & i32(
                0xFF
            )

        # ---- GEMM1 B (w1 of expert e): pass p, chunk c ----
        rw1 = rsrc(fx.Int64(w1) + fx.Int64(e) * fx.Int64(2 * I * KH))

        def issue_b(p, c):
            icol0 = p * i32(NW * 32) + w * i32(32)
            vg0 = icol0 * i32(KH) + lane * i32(16)
            vs = (vg0, vg0 + i32(16 * KH), vg0 + i32(I * KH), vg0 + i32((I + 16) * KH))
            b = []
            for k in range_constexpr(KCK):
                for t in range_constexpr(4):
                    b.append(
                        fx.Vector(
                            bld(rw1, vs[t], (c * i32(KCK) + i32(k)) * i32(1024), V4I)
                        )
                    )
            # one scale word = 2 k-steps
            cw = (c * i32(KCK)) // i32(2)
            sg = ((e * i32(2 * I) + icol0) // i32(32)) * i32(CH1 * 256) + cw * i32(256)
            su = ((e * i32(2 * I) + i32(I) + icol0) // i32(32)) * i32(CH1 * 256)
            su = su + cw * i32(256)
            vso = lane * i32(4)
            b.append(fx.Int32(bld(rws1, vso, sg, T.i32)))
            b.append(fx.Int32(bld(rws1, vso, su, T.i32)))
            return b

        def g1_chunk(acc, bq, buf, c):
            # c: the chunk (compile time)
            sgw, suw = bq[4 * KCK], bq[4 * KCK + 1]
            for k in range_constexpr(KCK):
                kp = (c * KCK + k) % 2  # the global k-step's half of the scale word
                sc = [
                    (x >> i32(8 * (kp * 2 + h))) & i32(0xFF)
                    for x in (sgw, suw)
                    for h in (0, 1)
                ]
                afs = [a_frag_m(buf, m, k) for m in range(MTE)]
                sas = [a_scale_m(buf, m, k) for m in range(MTE)]
                for m in range_constexpr(MTE):
                    for t in range_constexpr(4):
                        j = (t // 2) * 2 * MTE + m * 2 + t % 2
                        acc[j] = mfma(bq[k * 4 + t], afs[m], acc[j], sc[t], sas[m])
            return acc

        def act_store(acc, p):
            for m in range_constexpr(MTE):
                row = i32(m * 16) + lane % i32(16)
                gs, us_ = [], []
                for t in range_constexpr(2):
                    ga = fx.Vector(acc[m * 2 + t])
                    ua = fx.Vector(acc[2 * MTE + m * 2 + t])
                    for v in range_constexpr(4):
                        gs.append(fx.Float32(ga[v]))
                        us_.append(fx.Float32(ua[v]))
                xs_ = [
                    fx.Float32(x).to(fx.BFloat16).to(fx.Float32)
                    for x in act_batch(gs, us_)
                ]
                am = amax(xs_)
                am = am.maximumf(am.shuffle_xor(i32(16), i32(64)))
                am = am.maximumf(am.shuffle_xor(i32(32), i32(64)))
                e8, qs = _e8m0_from_amax(am, max_norm=6.0)
                for t in range_constexpr(2):
                    col = p * i32(NW * 32) + w * i32(32) + i32(t * 16) + q4 * i32(4)
                    pk = fp4_pack(xs_[t * 4 : t * 4 + 4], qs)
                    lds_st(
                        L,
                        i32(LINT) + row * i32(SIS) + col // i32(2),
                        fx.Int16(pk & i32(0xFFFF)),
                        align=2,
                    )
                if q4 == i32(0):
                    lds_st(
                        L,
                        i32(LINTS) + row * i32(ISS) + p * i32(NW) + w,
                        fx.Int8(e8),
                        align=1,
                    )

        # ---- GEMM1: PASSES x KCH chunks, A ring NAB, B ring BST (carried over passes) ----
        # prologue = the steady-state issue order of the BST virtual steps before c = 0
        # (steps before the first real op of a kind issue dummy loads of the same count)
        bpre = []
        for i in range_constexpr(-max(BST, NAB - 1), 0):
            if const_expr(i + BST >= 0):
                bpre.append(issue_b(i32(0), i32(i + BST)))
            else:
                issue_b(i32(0), i32(0))
            if const_expr(i + NAB - 1 >= 0):
                issue_a(i32(i + NAB - 1), (i + NAB - 1) % NAB)
            else:
                for _ in range_constexpr(NA):
                    bld(rx, lane * i32(16), 0, V4I)
        init = [zero] * (4 * MTE) + [v for b in bpre for v in b]
        phi = i32(PASSES)
        for p_, stt in range(i32(0), phi, i32(1), init=init):
            p = i32(p_)
            acc = [zero] * (4 * MTE)
            ring = [
                list(stt[4 * MTE + s * NB : 4 * MTE + (s + 1) * NB]) for s in range(BST)
            ]
            for c in range_constexpr(KCH):
                rocdl.sched_barrier(0)
                wait_vm(WAIT1)
                gpu.barrier()
                # nothing crosses the chunk barrier: the A ring reads after every wave's DMA landed
                # (else the scheduler may hoist them above s_barrier: a race)
                rocdl.sched_barrier(0)
                acc = g1_chunk(acc, ring[c % BST], c % NAB, c)
                # refill: B of step c + BST (next pass past the end; the last pass
                # reloads its own chunks: harmless, keeps the counts static)
                cn = c + BST
                pn = p + i32(cn // KCH)
                pn = (pn < phi).select(pn, phi - i32(1))
                ring[cn % BST] = issue_b(pn, i32(cn % KCH))
                issue_a(i32((c + NAB - 1) % KCH), (c + NAB - 1) % NAB)
            act_store(acc, p)
            # (the yield's value binding is FlyDSL's loop-carried form)
            res = yield [zero] * (4 * MTE) + [  # noqa: F841
                v for s in range(BST) for v in ring[s]
            ]
        wait_vm(0)
        wait_lgkm0()  # the intermediate's LDS writes land before other waves read them
        gpu.barrier()

    @flyc.jit
    def part_g2(L, tid, e, rows, w2, rws2, r_routes, oob):
        lane = tid % i32(64)
        w = rocdl.readfirstlane(T.i32, tid // i32(64))  # the wave: its column group
        q4 = lane // i32(16)
        zero = fx.Vector.filled(4, 0.0, fx.Float32)
        # ---- GEMM2 from the LDS intermediate: NG2 groups x KS2 k-steps, B ring BS2 ----
        rw2 = rsrc(fx.Int64(w2) + fx.Int64(e) * fx.Int64(H * (I // 2)))

        def issue_b2(gi, kk):  # one 64-column group
            n0 = (w + i32(NW) * gi) * i32(64)
            b = [
                fx.Vector(
                    bld(
                        rw2,
                        lane * i32(16),
                        (n0 + i32(t * 16)) * i32(I // 2) + kk * i32(1024),
                        V4I,
                    )
                )
                for t in range(4)
            ]
            for pp in range_constexpr(2):
                rb = (e * i32(H) + n0 + i32(pp * 32)) // i32(32)
                vo = lane * i32(4)
                b.append(
                    fx.Int32(
                        bld(rws2, vo, (rb * i32(CH2) + kk // i32(2)) * i32(256), T.i32)
                    )
                )
            # the loads stay where the ring issues them (no sinking)
            rocdl.sched_barrier(SCM)
            return b

        row_meta = []
        for m in range_constexpr(MTE):
            R = i32(m * 16) + lane % i32(16)
            ok = R < rows
            Rc = ok.select(R, i32(0))
            row_meta.append(
                (
                    ok,
                    lds_ld_i32(L, i32(LWT) + Rc * i32(4)).bitcast(fx.Float32),
                    lds_ld_i32(L, i32(LRIX) + Rc * i32(4)),
                )
            )

        def epi2(acc, gi):
            # route rows of group gi (rows past the unit's: stores dropped, counts kept)
            n0 = (w + i32(NW) * gi) * i32(64)
            ccol = (q4 & i32(1)) * i32(16) + (q4 >> i32(1)) * i32(8)
            if const_expr(R8):
                for m in range_constexpr(MTE):
                    # MXFP8 route rows (the small-M engine's store_route_fp8 layout)
                    ok, wt, rix = row_meta[m]
                    d, e8s = [], []
                    for half in range_constexpr(2):
                        f = [
                            fx.Float32(fx.Vector(acc[m * 4 + 2 * half + h])[i]) * wt
                            for h in range(2)
                            for i in range(4)
                        ]
                        am = amax(f)
                        am = am.maximumf(am.shuffle_xor(i32(16), i32(64)))
                        am = am.maximumf(am.shuffle_xor(i32(32), i32(64)))
                        e8, qs = _e8m0_from_amax(am, max_norm=448.0)
                        d += [fp8x4_pack(f[0:4], qs), fp8x4_pack(f[4:8], qs)]
                        e8s.append(fx.Int32(e8) & i32(0xFF))
                    d0, d1 = swap16(d[0], d[1])
                    d2, d3 = swap16(d[2], d[3])
                    d0, d2 = swap32(d0, d2)
                    d1, d3 = swap32(d1, d3)
                    ov = fx.Vector.from_elements([d0, d1, d2, d3], fx.Int32)
                    sc = fx.Int16(e8s[0] | (e8s[1] << i32(8)))
                    bst(
                        ov,
                        r_routes,
                        ok.select(rix * i32(H) + n0 + q4 * i32(16), oob),
                        0,
                        0,
                    )
                    soff = oob // i32(2) + rix * i32(H // 32) + n0 // i32(32)
                    bst(sc, r_routes, (ok & (q4 == i32(0))).select(soff, oob), 0, 0)
            else:
                for m in range_constexpr(MTE):
                    ok, wt, rix = row_meta[m]
                    pk = []
                    for t in range_constexpr(4):
                        av = fx.Vector(acc[m * 4 + t])
                        pk.append(
                            [
                                pack_bf16x2(
                                    fx.Float32(av[2 * h]) * wt,
                                    fx.Float32(av[2 * h + 1]) * wt,
                                )
                                for h in range(2)
                            ]
                        )
                    for half in range_constexpr(2):
                        ta, tb = 2 * half, 2 * half + 1
                        lo, hi = [], []
                        for h in range_constexpr(2):
                            x, y = swap16(pk[ta][h], pk[tb][h])
                            lo.append(x)
                            hi.append(y)
                        ov = fx.Vector.from_elements(lo + hi, fx.Int32)
                        col = n0 + i32(half * 32) + ccol
                        bst(
                            ov,
                            r_routes,
                            ok.select((rix * i32(H) + col) * i32(2), oob),
                            0,
                            0,
                        )

        # prologue = the steady issue order of the BS2 steps before (0, 0): their B2 loads,
        # then (the previous group's last step) NST2 route stores, here dropped (OOB)
        b2pre = [issue_b2(i32(s_ // KS2), i32(s_ % KS2)) for s_ in range(BS2)]
        for _ in range_constexpr(NST2):
            bst(fx.Vector.filled(4, 0, fx.Int32), r_routes, oob, 0)
        ghi = i32(NG2)
        init2 = [v for b in b2pre for v in b]
        for gi_, st2 in range(i32(0), ghi, i32(1), init=init2):
            gi = i32(gi_)
            ring2 = [list(st2[s * NB2 : (s + 1) * NB2]) for s in range(BS2)]
            acc = [zero] * (4 * MTE)
            for kk in range_constexpr(KS2):
                wait_vm(WAIT2[kk])
                bq = ring2[kk % BS2]
                sh0 = (i32(kk) & i32(1)) * i32(16)
                afs, sas = [], []
                for m in range_constexpr(MTE):
                    row = i32(m * 16) + lane % i32(16)
                    afs.append(
                        lds_ld(
                            L,
                            i32(LINT) + row * i32(SIS) + i32(kk * 64) + q4 * i32(16),
                            V4I,
                            16,
                        )
                    )
                    sas.append(
                        fx.Int32(
                            lds_ld(
                                L,
                                i32(LINTS) + row * i32(ISS) + i32(kk * 4) + q4,
                                T.i8,
                                1,
                            )
                        )
                        & i32(0xFF)
                    )
                s0, s1 = bq[4], bq[5]
                sb = [
                    (s0 >> sh0) & i32(0xFF),
                    (s0 >> (sh0 + i32(8))) & i32(0xFF),
                    (s1 >> sh0) & i32(0xFF),
                    (s1 >> (sh0 + i32(8))) & i32(0xFF),
                ]
                for m in range_constexpr(MTE):
                    for t in range_constexpr(4):
                        j = m * 4 + t
                        acc[j] = mfma(bq[t], afs[m], acc[j], sb[t], sas[m])
                sn = kk + BS2
                gn = gi + i32(sn // KS2)
                gn = (gn < ghi).select(gn, ghi - i32(1))
                ring2[kk % BS2] = issue_b2(gn, i32(sn % KS2))
            epi2(acc, gi)
            res2 = yield [v for s in range(BS2) for v in ring2[s]]  # noqa: F841
        wait_vm(0)

    @flyc.jit
    def unit_body(
        L, tid, e, r0, rows, rr, rwt, rx, rxs, w1, w2, rws1, rws2, r_routes, oob
    ):
        part_fill(L, tid, r0, rows, rr, rwt)
        part_g1(L, tid, e, rx, rxs, w1, rws1)
        part_g2(L, tid, e, rows, w2, rws2, r_routes, oob)

    return lay, unit_body
