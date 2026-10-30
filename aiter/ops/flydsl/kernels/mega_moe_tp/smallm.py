# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Small-M schedule of the fused TP MegaMoE kernel (sm, up to DYN_MAX global tokens).

The dynamic schedule's units are (row chunk x inter piece) GEMM1 -> GEMM2 on one CTA: GEMM2
then reads 1 / P of every w2 row (1 KB strided pieces at P = 6), sums P partial regions per
route, and the unit sizes rarely spread evenly over the CTAs (a hot / shared expert's row
chunks pile up). Here the two GEMMs are split apart, both weight streams contiguous and the
bytes per CTA even:

  G1  the row chunks' inter blocks (I / 128 per chunk; q = blocks // CTAs): from q >= QC on
      q contiguous blocks per CTA and the r remaining ones (the last chunks) one each on
      the last r CTAs, else rounds of C blocks (block u -> CTA u % C). GEMM1 -> the MXFP4
      intermediate exported (xg, by route) -> the chunk's block count bumped (CTRL_SMC,
      banked by the launch epoch parity; every launch zeroes the next one's bank:
      schedule.sm_zero_next). The chunks of the first q * C blocks are "early".
  G2  output column shares of the row chunks (one RS chunk of CW columns each, the full
      inter dim: contiguous w2 rows), in two contiguous runs (chunk order) per CTA: the
      CTAs without an extra block take up to D shares (one block's bytes) of the early
      chunks, then every CTA an even part of the rest. Per chunk of a run: wait for its
      blocks (the w2 prologue already in flight), import its intermediate, one GEMM2 over
      the run's columns -> the route rows (one region: the push sums topk rows only).

With fewer G1 blocks than CTAs and at most one route per CTA (e.g. 8 tokens) the launch
runs the dynamic schedule's fused units instead (decided in the plan: C_SMODE).
"""

from __future__ import annotations

import functools
import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from .common import (
    ACQ,
    AUX_SYS,
    REL,
    V4F,
    bld,
    bst,
    g_add_agent,
    gptr,
    MONO,
    now,
    i32,
    lds_ld_i32,
    lds_st,
    lds_st_rel,
    rsrc,
)
from .mega_moe_tp_config import (
    C_DYNP,
    C_GEXP,
    C_NCH,
    C_PLAN,
    C_UL,
    C_UNIT,
    C_UROWS,
    C_USEQ,
    CTRL_INTS,
    CTRL_SMC,
    CTRL_SMF,
    G1C_STRIDE,
    KCS,
    NCHLB_MAX,
    NCTA_MAX,
    NW,
    UL_MAX,
    KernelCtx,
    _nsk2_for,
)
from .schedule import DPF

const_expr = fx.const_expr
# split-K partial sums of a block's last piece: PB slots x JG accumulators per step
PB, JG = 4, 4
# G1 layout: contiguous runs from this many blocks per CTA on, else rounds
QC = 3
# LDS ctl slots (unused by the dynamic schedule): the row chunk whose intermediate L_INTER
# holds; this CTA's G2 runs (first share, end share) x 2
C_SMK, C_SM2 = 66, 70
# 1: this launch runs the small-M units, 0: the dynamic schedule's (fewer G1 blocks than CTAs
# and few routes: the fused units keep the intermediate in LDS, no G1 -> G2 waits)
C_SMODE = 74
# this CTA's split-K unit was its block's last piece
C_SMLAST = 67
# the G1 block a CTA without one prefetches (schedule.pf_unit)
C_SMPF = 75
# (ROT) the chunk-table rotation: the plan's chunk j is the table's (j + C_JSH) % chunks
C_JSH = 69


def build_smallm(kc: KernelCtx) -> dict:
    """The small-M schedule device functions of one instance (added to kc)."""
    G2, GPC, H, I, KS2, L_CTL, L_DCH = kc.get("G2 GPC H I KS2 L_CTL L_DCH")
    (TOPK,) = kc.get("TOPK")
    NCK, cbar, ctrl_at = kc.get("NCK cbar ctrl_at")
    gather_routes_chunk, gemm1, gemm2 = kc.get("gather_routes_chunk gemm1 gemm2")
    lb_import, spin_sys_ge, unit_fields = kc.get("lb_import spin_sys_ge unit_fields")
    xq_export, _xcd_rank = kc.get("xq_export _xcd_rank")
    _dyn_units, _dyn_loop = kc.get("_dyn_units _dyn_loop")
    mark, TRACE = kc.get("mark TRACE")

    @flyc.jit
    def _tval(tid, a, k, v):
        if tid == i32(0):
                slot = i32(CTRL_INTS) + i32(gpu.block_id("x")) * i32(64) + k
                fx.generic_store(
                    gptr(fx.Int64(a["ctrl"]) + fx.Int64(slot * i32(4))),
                    v,
                    memory_order=MONO,
                    syncscope="agent",
                )

    def tval(tid, a, k, v):
        """trace builds: CTA value v at trace slot k (32..63: G1 / G2 call events)"""
        if TRACE and not kc.TRACE_LIGHT:
            _tval(tid, a, k, v)

    def tnow(tid, a, k):
        if TRACE and not kc.TRACE_LIGHT:
            tval(tid, a, k, fx.Int32(now()))

    KS1, MT, TMAX, _g1_out, nsk = kc.get("KS1 MT TMAX _g1_out nsk")
    (HB,) = kc.get("HB")
    GPRE = os.environ.get("AITER_MEGAMOE_TP_GPRE", "1") == "1"  # (debug)
    # (debug) half blocks after a full round while r <= C / HBF
    HBF = int(os.environ.get("AITER_MEGAMOE_TP_HBF", "3"))
    # (whole-inter G2 calls) a k-step pair's w2 scale dwords loaded once
    SDD2 = os.environ.get("AITER_MEGAMOE_TP_SDD", "1") == "1"
    # the dynamic schedule's fused units while the G1 blocks fill under DYN_HI / 256 of
    # the CTAs (debug: AITER_MEGAMOE_TP_DYNHI)
    # (128: never, given more than C / 2 blocks; measured: the small-M units win)
    DYN_HI = int(os.environ.get("AITER_MEGAMOE_TP_DYNHI", "128"))
    # split-K pieces (ascending): the K steps of a piece a multiple of the B interleave
    IL1 = nsk * KCS // math.gcd(nsk, KCS)
    SKS = [s for s in (2, 3, 4, 6, 12) if KS1 % s == 0 and (KS1 // s) % IL1 == 0]
    # split-K launches: G2 inter pieces per share (NKS = KS2 / QF k-steps a whole number of
    # GEMM2 iterations per share of GPC groups)
    QF = 1
    for q_ in (2, 3, 6):
        if KS2 % q_ == 0:
            n2 = KS2 // q_
            s2 = _nsk2_for(n2, G2)
            if GPC % (s2 * n2 // math.gcd(s2, n2) // n2) == 0:
                QF = q_
                break
    # the partial slots after the export area of xg (MegaMoeTPEngine.xg)
    XG_PART = TMAX * TOPK * (I // 2 + I // 32 + 8)

    # a G2 share: GPC groups (of NW * 64 columns), a whole number of GEMM2 iterations
    nsk2 = _nsk2_for(KS2, G2)
    assert GPC % (nsk2 * KS2 // math.gcd(nsk2, KS2) // KS2) == 0
    # G1 block bytes / G2 share bytes (2 * 128 * H vs H / NCK * I weights)
    D = max(1, round(256 * NCK / I))
    # (debug) the early-share cap per CTA without an extra block
    D = int(os.environ.get("AITER_MEGAMOE_TP_SMD", str(D)))
    # (debug) the rest run in single groups (else whole shares of GPC groups)
    RGU = GPC if os.environ.get("AITER_MEGAMOE_TP_G2G", "1") == "1" else 1
    # (debug) the plan's chunk order starts at the last expert's chunks when it has several
    # (the shared expert: every token's): its GEMM1 blocks in the first round, all CTAs in
    # step, so the reads of one weight block by its row chunks meet in L2
    ROT = os.environ.get("AITER_MEGAMOE_TP_ROT", "1") == "1"

    def _rot(L, j):
        if const_expr(ROT):
            nun = fx.max(lds_ld_i32(L, L_CTL + C_NCH * 4), i32(1))
            return (j + lds_ld_i32(L, L_CTL + C_JSH * 4)) % nun
        return j

    def smc(a, j, bank):
        return ctrl_at(a, i32(CTRL_SMC) + (bank * i32(NCHLB_MAX) + j) * i32(G1C_STRIDE))

    def smf(a, b, bank):
        return ctrl_at(a, i32(CTRL_SMF) + (bank * i32(NCTA_MAX) + b) * i32(G1C_STRIDE))

    @flyc.jit
    def sm_units(L, tid, a):
        """(after the chunk table) the dynamic schedule's units if the G1 blocks do not fill
        the CTAs and the routes are few (its P partial route rows per route stay cheap),
        else the small-M ones"""
        nb = lds_ld_i32(L, L_CTL + C_NCH * 4) * i32(KS2)
        C = i32(gpu.grid_dim.x)
        # (from at most C / 2 blocks on the small-M units split K: they fill the CTAs)
        # (more blocks: the small-M units, their G2 spread over all CTAs, win)
        if (
            (nb * i32(256) < C * i32(DYN_HI))
            & (nb * i32(2) > C)
            & (a["ttot"] * i32(TOPK) <= C)
        ):
            _dyn_units(L, tid)
            if tid == i32(0):
                lds_st(L, L_CTL + C_SMODE * 4, i32(0))
        else:
            _sm_plan(L, tid, a)

    @flyc.jit
    def _sm_plan(L, tid, a):
        """(tid 0) this CTA's G1 inter blocks as unit records (chunk entry, i0, icnt, chunk)
        and its G2 runs"""
        if tid == i32(0):
            lds_st(L, L_CTL + C_SMODE * 4, i32(1))
            nun = lds_ld_i32(L, L_CTL + C_NCH * 4)
            if const_expr(ROT):
                last = lds_ld_i32(L, L_DCH + fx.max(nun - i32(1), i32(0)) * i32(4))
                ce_l = last.shrui(i32(16)) & i32(0xFF)
                lds_st(
                    L,
                    L_CTL + C_JSH * 4,
                    ((nun > i32(0)) & (ce_l > i32(1))).select(nun - ce_l, i32(0)),
                )
            C = i32(gpu.grid_dim.x)
            x = _xcd_rank(C)
            nb = nun * i32(KS2)
            # split-K: few blocks (hot experts) run in sk K pieces each, one unit per CTA
            # (from 4 pieces on: below, the partial sums cost more than the spread gains)
            sk = i32(1)
            for s_ in SKS:
                sk = ((nb * i32(s_) <= C) & (nb * i32(4) <= C)).select(i32(s_), sk)
            nu = nb * sk
            q = nu // C
            r = nu - q * C
            # half blocks (HB): the last round's r blocks as 2 r 64-column halves (the K
            # halves on the two wave pairs) on the last 2 r CTAs: the round ends after half
            # a block (else these CTAs bound the G1 phase at a whole extra block); after a
            # full round only while at least half the CTAs stay free for the early shares
            hb = fx.Boolean(False)
            if const_expr(HB):
                hb = (
                    (sk == i32(1))
                    & (r > i32(0))
                    & (r * i32(2) <= C)
                    & ((q == i32(0)) | ((q < i32(QC)) & (r * i32(HBF) <= C)))
                )
            # partial halves (HB, one round of more than C / 2 blocks): the first
            # C - r blocks as halves on the first 2 (C - r) CTAs, every CTA one unit; the
            # chunks of the halved blocks are done after half a block ("early")
            hp = fx.Boolean(False)
            if const_expr(HB):
                hp = (sk == i32(1)) & (q == i32(0)) & (r * i32(2) > C) & (r < C)
            sh2 = (C - r) * i32(2)
            rr = hp.select(C - sh2, hb.select(r * i32(2), r))
            # G1 layout: q >= QC: q contiguous blocks per CTA (few unit boundaries), the r
            # remaining blocks (the last chunks) one each on the last r CTAs; else rounds
            # (block u -> CTA u % C, the last round on the first r CTAs). Either way the
            # chunks of the first q * C blocks are done after q blocks ("early").
            cont = (q >= i32(QC)) & (sk == i32(1))
            hi = (cont | hb | hp).select(x >= C - rr, x < r)
            n1 = hp.select(i32(1), q + hi.select(i32(1), i32(0)))
            b0 = x * q
            j0 = b0 // i32(KS2)
            nc = (q > i32(0)).select((b0 + q + i32(KS2 - 1)) // i32(KS2) - j0, i32(0))
            n = cont.select(nc + hi.select(i32(1), i32(0)), n1)
            for k_ in range(i32(0), fx.min(n, i32(UL_MAX)), i32(1)):
                k = i32(k_)
                xtra = k >= nc  # (cont) the remainder block
                u = cont.select(q * C + x - (C - r), x + k * C)
                # (hb) this CTA's half block (its last unit): half p % 2 of block
                # q * C + p // 2
                ph = (hb & hi & (k == n - i32(1))) | (hp & (hi == fx.Boolean(False)))
                p = hp.select(x, x - (C - rr))
                u = ph.select(q * C + p // i32(2), u)
                u = (hp & hi).select((C - r) + (x - sh2), u)
                bu = u // sk
                piece = u - bu * sk
                jr = bu // i32(KS2)
                bk = bu - jr * i32(KS2)
                jc = j0 + k
                lo_b = fx.max(b0, jc * i32(KS2))
                hi_b = fx.min(b0 + q, (jc + i32(1)) * i32(KS2))
                one = (cont == fx.Boolean(False)) | xtra
                j = _rot(L, one.select(jr, jc))
                ul = L_CTL + (i32(C_UL) + k * i32(4)) * i32(4)
                lds_st(L, ul, lds_ld_i32(L, L_DCH + j * i32(4)))
                lds_st(
                    L,
                    ul + i32(4),
                    one.select(bk, lo_b - jc * i32(KS2)) * i32(128)
                    + ph.select((p % i32(2)) * i32(64), i32(0)),
                )
                lds_st(
                    L,
                    ul + i32(8),
                    ph.select(i32(64), one.select(i32(1), hi_b - lo_b) * i32(128)),
                )
                # (the A loader reads the K piece from here too)
                lds_st(
                    L,
                    ul + i32(12),
                    j | (one.select(piece, i32(0)) << i32(16)) | (sk << i32(24)),
                )
            # G2: early shares (the chunks done after q blocks of every CTA) on the CTAs
            # without an extra block, up to D each; the rest even over all CTAs
            # (split-K: the G2 shares split the inter dim too, QF pieces each)
            qg = (sk == i32(1)).select(i32(1), i32(QF))
            ng2 = nun * i32(NCK) * qg
            early = (sk == i32(1)).select((q * C) // i32(KS2) * i32(NCK), i32(0))
            early = hp.select((C - r) // i32(KS2) * i32(NCK), early)
            nlo = C - rr
            cap = nlo * i32(D)
            e1 = fx.min(early, cap)
            lo = hi == fx.Boolean(False)
            xl = (cont | hb | hp).select(x, fx.max(x - r, i32(0)))
            b1 = e1 // fx.max(nlo, i32(1))
            x1 = e1 - b1 * nlo
            s1 = xl * b1 + fx.min(xl, x1)
            t1 = s1 + b1 + (xl < x1).select(i32(1), i32(0))
            # (hb) the half-block CTAs end G1 half a block later than the others start
            # their early shares: the shares after the early ones (the last round's
            # chunks) on them first, up to b1 - D / 2 each (their load then ~ the others')
            hcap = hb.select(fx.max(b1 - i32(D // 2), i32(1)) * rr, i32(0))
            eh = fx.min(ng2 - e1, hcap)
            xh = x - (C - rr)
            bh = eh // fx.max(rr, i32(1))
            xr = eh - bh * rr
            sh = e1 + xh * bh + fx.min(xh, xr)
            th = sh + bh + (xh < xr).select(i32(1), i32(0))
            # (G2G: the rest in groups, GPC per share: the last ones' finer spread)
            rest = (ng2 - e1 - eh) * i32(RGU)
            b2 = rest // C
            x2 = rest - b2 * C
            # (the remainder shares first to the CTAs without the early remainder)
            xo = hb.select((x - x1 + C) % C, x)
            s2 = (e1 + eh) * i32(RGU) + xo * b2 + fx.min(xo, x2)
            t2 = s2 + b2 + (xo < x2).select(i32(1), i32(0))
            r0s = lo.select(s1, hb.select(sh, i32(0)))
            r0t = lo.select(t1, hb.select(th, i32(0)))
            lds_st(L, L_CTL + C_SM2 * 4, r0s)
            lds_st(L, L_CTL + (C_SM2 + 1) * 4, r0t)
            lds_st(L, L_CTL + (C_SM2 + 2) * 4, s2)
            lds_st(L, L_CTL + (C_SM2 + 3) * 4, t2)
            lds_st(L, L_CTL + C_DYNP * 4, qg)
            # (a CTA without a G1 block: the block it prefetches, -1 none)
            lds_st(
                L,
                L_CTL + C_SMPF * 4,
                ((n == i32(0)) & (sk == i32(1)) & (nu > i32(0))).select(
                    (x - r) % fx.max(nu, i32(1)), i32(-1)
                ),
            )
            lds_st(L, L_CTL + C_UNIT * 4, i32(0))
            lds_st(L, L_CTL + (C_UNIT + 1) * 4, fx.min(n, i32(UL_MAX)))
            lds_st(L, L_CTL + C_SMK * 4, i32(-1))
            lds_st_rel(L, L_CTL + C_PLAN * 4, i32(1))

    def _chunk(ent):
        return ent & i32(0xFFFF), ent.shrui(i32(24)), ent.shrui(i32(16)) & i32(0xFF)

    def _part_rs(a):
        return rsrc(fx.Int64(a["xg"]) + fx.Int64(XG_PART))

    def _part_off(slot, w, j, lane):
        return ((slot * i32(NW) + w) * i32(4 * MT) + i32(j)) * i32(1024) + lane * i32(
            16
        )

    def _part_store(u, sk):
        """GEMM1 epilogue of a unit: sk > 1: this K piece's fp32 accumulators to partial
        slot u; else act + MXFP4 to L_INTER"""

        @flyc.jit
        def out(L, lane, w, a, res, grp, P, MTE):
            if sk > i32(1):
                rp = _part_rs(a)
                for j in range_constexpr(len(res)):
                    bst(res[j], rp, _part_off(u, w, j, lane), 0, AUX_SYS)
            else:
                _g1_out(L, lane, w, a, res, grp, P, MTE)

        return out

    @flyc.jit
    def _part_finish(L, tid, a, u0, sk):
        """(the last K piece of a block) the sum of its sk partial slots u0.. -> act + MXFP4
        to L_INTER"""
        lane = tid % i32(64)
        w = tid // i32(64)
        rp = _part_rs(a)
        zero = fx.full(4, 0.0, fx.Float32)
        res = []
        # JG accumulators at a time over PB slots per step: their loads in flight together
        # (each a cross-XCD miss) within the register budget
        for jg in range_constexpr(0, 4 * MT, JG):
            js = list(range(jg, min(jg + JG, 4 * MT)))
            for p_, st in range(i32(0), sk, i32(PB), init=[zero] * len(js)):
                acc = list(st)
                lds_ = []
                for pb in range_constexpr(PB):
                    p = i32(p_) + i32(pb)
                    slot = u0 + fx.min(p, sk - i32(1))
                    lds_.append(
                        (
                            p < sk,
                            [
                                bld(rp, _part_off(slot, w, j, lane), 0, V4F, AUX_SYS)
                                for j in js
                            ],
                        )
                    )
                for live, vs in lds_:
                    for i in range_constexpr(len(js)):
                        acc[i] = fx.Vector(acc[i]) + fx.Vector(live.select(vs[i], zero))
                part = yield acc
            res += list(part)
        _g1_out(L, lane, w, a, res, i32(0), 1, MT)

    @flyc.jit
    def _sm_g1(L, tid, a, u, bank):
        ent, i0, icnt, kf = unit_fields(L, u)
        j = kf & i32(0xFFFF)
        piece = kf.shrui(i32(16)) & i32(0xFF)
        sk = fx.max(kf.shrui(i32(24)) & i32(0xFF), i32(1))
        e, cc, ce = _chunk(ent)
        mark(tid, a, 10)
        tnow(tid, a, i32(32) + fx.min(u, i32(3)) * i32(2))
        R = gather_routes_chunk(L, tid, a, e, cc, ce)
        if const_expr(os.environ.get("AITER_MEGAMOE_TP_ICTEST", "0") == "1"):
            mark(tid, a, 27)
            for _ in range_constexpr(10):
                cbar(L, tid)
            mark(tid, a, 26)
            if tid == i32(0):
                lds_st(L, L_CTL + C_GEXP * 4, i32(-1))
            cbar(L, tid)
            R = gather_routes_chunk(L, tid, a, e, cc, ce)
        if tid == i32(0):
            lds_st(L, L_CTL + C_UROWS * 4, R)
            lds_st_rel(L, L_CTL + C_USEQ * 4, u + i32(1))
        mark(tid, a, 13)
        if u == i32(0):
            mark(tid, a, 25)
        # this unit's partial slot (block * sk + piece; one piece per CTA)
        uid = _xcd_rank(i32(gpu.grid_dim.x)) + u * i32(gpu.grid_dim.x)
        nk = i32(KS1) // sk
        if sk > i32(1):
            gemm1(
                L,
                tid,
                a,
                e,
                i0,
                icnt // i32(128),
                kr=(piece * nk, nk),
                out=_part_store(uid, sk),
            )
        else:  # (the static K loop of the whole K dim)
            gemm1(
                L,
                tid,
                a,
                e,
                i0,
                icnt // i32(128),
                rows=R,
                half=icnt == i32(64),
                dup=ce > i32(1),
            )
        mark(tid, a, 14)
        if u == i32(0):
            mark(tid, a, 26)
        if sk > i32(1):
            blk = uid // sk
            if tid == i32(0):
                # (the pieces' partial stores visible to the block's last piece, on any
                # XCD: release before the count, acquire after it)
                fx.memory_fence(syncscope="agent", ordering=REL)
                old = g_add_agent(smf(a, blk, bank), 1)
                last = old == sk - i32(1)
                if last:
                    fx.memory_fence(syncscope="agent", ordering=ACQ)
                lds_st(L, L_CTL + C_SMLAST * 4, last.select(i32(1), i32(0)))
            cbar(L, tid)
            if lds_ld_i32(L, L_CTL + C_SMLAST * 4) == i32(1):
                _part_finish(L, tid, a, blk * sk, sk)
                cbar(L, tid)
                mark(tid, a, 27)
                xq_export(L, tid, a, i0, icnt, i32(0), R)
                cbar(L, tid)
                if tid == i32(0):
                    g_add_agent(smc(a, j, bank), 2)
        else:
            xq_export(L, tid, a, i0, icnt, i32(0), R)
            cbar(L, tid)
            if tid == i32(0):
                g_add_agent(smc(a, j, bank), icnt // i32(64))
        mark(tid, a, 11)
        tnow(tid, a, i32(33) + fx.min(u, i32(3)) * i32(2))

    @flyc.jit
    def _sm_import(L, tid, a, j, R, bank):
        if lds_ld_i32(L, L_CTL + C_SMK * 4) != j:
            if tid == i32(0):
                spin_sys_ge(smc(a, j, bank), i32(2 * KS2), a)
            cbar(L, tid)
            mark(tid, a, 30)
            lb_import(L, tid, a, R)
            if tid == i32(0):
                lds_st(L, L_CTL + C_SMK * 4, j)
        mark(tid, a, 31)

    @flyc.jit
    def _sm_g2(L, tid, a, j, c0, c1, bank, piece, qg, ev):
        """GEMM2 of row chunk j over its column groups [c0, c1): the whole inter dim
        (qg 1), else its inter piece of KS2 / QF k-steps (partial route region piece)"""
        mark(tid, a, 28)
        tnow(tid, a, ev)
        tval(tid, a, ev + i32(2), (c1 - c0) | (j << i32(8)))
        e, cc, ce = _chunk(lds_ld_i32(L, L_DCH + j * i32(4)))
        if const_expr(GPRE):
            # (the gather inside the GEMM2 prologue: after the w2 ring prefetch is issued)
            def pre():
                R_ = gather_routes_chunk(L, tid, a, e, cc, ce)
                mark(tid, a, 29)
                _sm_import(L, tid, a, j, R_, bank)

            R = None
        else:
            R = gather_routes_chunk(L, tid, a, e, cc, ce)
            mark(tid, a, 29)
            pre = functools.partial(_sm_import, L, tid, a, j, R, bank)
        if qg == i32(1):
            gemm2(
                L,
                tid,
                a,
                e,
                i32(0),
                i32(0),
                R,
                fx.Boolean(False),
                KS2,
                i32(0),
                c0,
                c1,
                pre=pre,
                sdd=SDD2,
                odd=RGU > 1,
                dup=ce > i32(1),
            )
        else:
            gemm2(
                L,
                tid,
                a,
                e,
                piece * i32(KS2 // QF),
                i32(0),
                R,
                fx.Boolean(False),
                KS2 // QF,
                piece,
                c0,
                c1,
                pre=pre,
                aoff=piece * i32(KS2 // QF),
                odd=RGU > 1,
                dup=ce > i32(1),
            )
        cbar(L, tid)
        mark(tid, a, 15)
        tnow(tid, a, ev + i32(1))

    @flyc.jit
    def _sm_run(L, tid, a, s0, s1, bank, run):
        """the G2 shares [s0, s1) (share s: segment s // NCK = (row chunk, inter piece of
        qg), its RS column chunk s % NCK; run 1 in units of 1 / RGU share): one GEMM2 per
        segment"""
        U = RGU if run == 1 else 1
        NU = NCK * U
        qg = lds_ld_i32(L, L_CTL + C_DYNP * 4)
        g0 = s0 // i32(NU)
        ng = (s1 > s0).select((s1 + i32(NU - 1)) // i32(NU) - g0, i32(0))
        for k_ in range(i32(0), ng, i32(1)):
            g = g0 + i32(k_)
            c0 = (fx.max(s0, g * i32(NU)) - g * i32(NU)) * i32(GPC // U)
            c1 = (fx.min(s1, (g + i32(1)) * i32(NU)) - g * i32(NU)) * i32(GPC // U)
            jp = g // qg
            ev = i32(40 + 12 * run) + fx.min(i32(k_), i32(3)) * i32(3)
            _sm_g2(L, tid, a, _rot(L, jp), c0, c1, bank, g - jp * qg, qg, ev)

    @flyc.jit
    def sm_compute(L, tid, a):
        """(compute waves, after the plan)"""
        if lds_ld_i32(L, L_CTL + C_SMODE * 4) == i32(1):
            _sm_compute(L, tid, a)
        else:
            _dyn_loop(L, tid, a)

    @flyc.jit
    def _sm_compute(L, tid, a):
        """the G1 blocks, then the G2 runs"""
        bank = a["epoch"] & i32(1)  # (sm_zero_next: zeroed by the previous launch)
        ue = lds_ld_i32(L, L_CTL + (C_UNIT + 1) * 4)
        for u_ in range(i32(0), ue, i32(1)):
            _sm_g1(L, tid, a, i32(u_), bank)
        if const_expr(DPF):
            pf = lds_ld_i32(L, L_CTL + C_SMPF * 4)
            if pf >= i32(0):
                jp = pf // i32(KS2)
                ent = lds_ld_i32(L, L_DCH + _rot(L, jp) * i32(4))
                kc.pf_unit(L, tid, a, ent & i32(0xFFFF), (pf - jp * i32(KS2)) * i32(128), i32(128), False)
        for run in range_constexpr(2):
            _sm_run(
                L,
                tid,
                a,
                lds_ld_i32(L, L_CTL + (C_SM2 + 2 * run) * 4),
                lds_ld_i32(L, L_CTL + (C_SM2 + 2 * run + 1) * 4),
                bank,
                run,
            )
        mark(tid, a, 12)

    return locals()
