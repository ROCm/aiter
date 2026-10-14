# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Large-batch (big-M) GEMM of the MegaMoE TP kernel: fused units.

A unit is <= RG route rows of one expert. One CTA of NW compute waves runs, for the
unit, GEMM1 over the whole inter slice (PASSES passes of 128 inter columns: each wave
32 gate + 32 up columns), the activation + MXFP4 quant of the intermediate into LDS,
then GEMM2 (all H output columns) straight from that LDS intermediate: the
intermediate never goes to global memory. One CTA per CU (RG = 128: ~125 KB of LDS).

Operands (same conventions as gemm.py): A = the gathered MXFP4 activation rows (row
token = route // TOPK) through an NAB-deep LDS ring that the compute waves fill
themselves (LDS-DMA, XOR-swizzled 16 B columns); B = the preshuffled MXFP4 weights
straight into registers (a BST-deep ring per wave); scaled MFMA 16x16x128 f8f6f4.

vmcnt: every wave issues the same ops in the same order every step, so the wait
counts are compile-time (see the issue order notes at each wait).
"""


import os as _os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ..mxfp4_gemm_common import _activation_mul_batch, _e8m0_from_amax
from .common import (
    AUX_SYS,
    V4I,
    amax,
    bld,
    bst,
    dma4,
    dma16,
    fp4_pack,
    i32,
    lds_ld,
    lds_ld_i32,
    lds_ptr,
    lds_st,
    mfma,
    pack_bf16x2,
    rsrc,
    swap16,
    swap32,
    fp8x4_pack,
    uni,
    wait_lgkm0,
    wait_vm,
)

NW = 4
NTH = NW * 64


def _opaque(v):
    """v through a side-effecting copy: equal loads stay separate instructions (no CSE: the
    static vmcnt waits count every issued load)"""
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl.expr import arith

    return fx.Int32(_llvm.inline_asm(T.i32, [arith.unwrap(v)], "v_mov_b32 $0, $1", "=v,v", has_side_effects=True))


def bigm_lds(RG, NAB, I, NW=NW, MODE=0, KCK=2, PFD=0):
    """LDS layout (bytes) of one CTA. MODE 3 (one GEMM1 pass): the pass's intermediate columns
    reuse the A ring (written after the last chunk); MODE 4 (GEMM2 items): no A ring."""
    c = {}
    o = 0

    def take(n, align=16):
        nonlocal o
        o = (o + align - 1) // align * align
        r = o
        o += n
        return r

    c["LA"] = take(NAB * RG * 64 * KCK) if MODE != 4 else 0  # A codes: [buf][row][64 KCK B], 16 B columns swizzled
    c["LAS"] = take(NAB * NW * 64 * 4) if MODE != 4 else 0  # A scales: [buf][wave][lane] words (dma4 order)
    c["LRIX"] = take(RG * 4)
    c["LWT"] = take(RG * 4)
    if MODE == 3:
        c["SIS"] = NW * 16 + 16  # one pass: NW * 32 columns (+16: rows on different banks)
        c["LINT"] = c["LA"]
        assert RG * c["SIS"] <= NAB * RG * 64 * KCK
        c["ISS"] = NW  # scale bytes per row
        c["LINTS"] = take(RG * NW)
    else:
        c["SIS"] = I // 2 + 16
        c["LINT"] = take(RG * c["SIS"])  # intermediate MXFP4 rows
        c["ISS"] = I // 32
        c["LINTS"] = take(RG * (I // 32))  # their E8M0 scales
    c["LPF"] = take(NW * 256) if PFD else 0  # per-wave scratch of the L2 prefetch DMAs (never read)
    c["LCTL"] = take(64)
    c["BYTES"] = (o + 15) // 16 * 16
    return c


def build_bigm_unit(
    H, I, TOPK, RG=128, NAB=4, BST=3, BS2=3, act="swiglu", swiglu_limit=7.0, dbg=0, st_aux=0, ISPL=1,
    PACKED=0, NWU=NW, NRS=1, UNRP=0, SWZ=0, LRG=None, MODE=0, G2N=1, PFD=0, R8=0, KCK=2,
):
    """The fused unit device function: unit_body(L, tid, e, r0, rows, rr, rwt, rx, rxs, w1,
    w2, rws1, rws2, r_routes, oob): rows rr / rwt[r0 : r0 + rows] (route index, weight) of expert
    e -> route rows wt * GEMM2(act(GEMM1(A[route // TOPK]))) stored at route * H (bf16).
    With ISPL > 1 a unit is one of ISPL slices of the intermediate (unit_body's ih): GEMM1 of
    its I / ISPL columns, GEMM2 over them -> partial route rows (the slices are summed later).
    PACKED=1: rr holds moe_sorting ids (token | slot << 24, read at system scope: written by a
    concurrent kernel); the route index is then token * TOPK + slot.
    NRS: row splits -- NWU x NRS waves, each NWU column group's rows split over NRS waves
    (fewer accumulators per wave: two waves per SIMD fit the register file).
    LRG: rows of the LDS layout (default RG): a unit of RG < LRG rows (fewer row tiles, fewer A
    rows) in the same LDS as the LRG-row unit (both in one kernel, picked per unit by its rows).
    R8: route rows as MXFP8 (e4m3 + one e8m0 scale per 32 columns): codes [rows][H] at r_routes,
    scales [rows][H / 32] after them (at oob / 2: oob = the route count * H * 2).
    KCK: GEMM1 k-steps (of 128) per chunk: 2, or 1 for 256-row units (half the A ring per row:
    the LDS of a 256-row unit fits; one chunk = 16 row tiles x 4 column tiles x 1 k-step).
    Returns (LDS layout, unit_body)."""
    assert KCK in (1, 2)
    assert RG % 16 == 0 and RG <= (256 if KCK == 1 else 128) and H % 256 == 0 and I % 128 == 0 and RG <= (LRG or RG)
    MTE = RG // 16
    KH = H // 2  # A / w1 row bytes
    KCH = H // (128 * KCK)  # GEMM1 chunks (KCK k-steps of 128)
    RBA = 64 * KCK  # A row bytes per chunk
    LPR = RBA // 16  # DMA lanes per A row (16 B each)
    RPJ = 64 // LPR  # A rows per dma16
    NW = NWU  # column groups of the unit (each: 32 gate + 32 up columns per GEMM1 pass)
    NWT = NW * NRS  # waves (NRS row splits per column group)
    NTHU = NWT * 64  # threads of the unit
    assert MTE % NRS == 0
    MTEW = MTE // NRS  # 16-row tiles per wave
    PASSES = I // (NW * 32) // ISPL  # GEMM1 passes of this slice
    # G2W: GEMM2 groups of G2W x 64 columns per wave (fewer groups: fewer A re-reads / store bursts)
    G2W = int(_os.environ.get("BIGM_G2W", "1"))
    assert G2W in (1, 2) and (MODE != 4 or G2W == 1)
    NG2 = H // (NW * 64 * G2W)  # GEMM2 column groups per wave
    KS2 = I // 128 // ISPL  # GEMM2 k-steps of this slice
    assert I % (128 * ISPL) == 0 and I % (NW * 32 * ISPL) == 0
    CH1 = (H // 32 + 7) // 8
    CH2 = (I // 32 + 7) // 8
    assert KCH % NAB == 0 and KCH % BST == 0 and KS2 % BS2 == 0
    lay = bigm_lds(LRG or RG, NAB, I, NWT, MODE if MODE in (3, 4) else 0, KCK, PFD)
    ISS = lay["ISS"]
    P1 = MODE == 3  # LDS intermediate holds one pass (columns relative to the pass)
    LA, LAS, LRIX, LWT = lay["LA"], lay["LAS"], lay["LRIX"], lay["LWT"]
    SIS, LINT, LINTS, LCTL = lay["SIS"], lay["LINT"], lay["LINTS"], lay["LCTL"]
    LPF = lay["LPF"]
    AR = RG // NWT  # A rows per wave (DMA)
    assert AR in (8, 16, 32, 64) and AR * KCK <= 64 and 64 % (AR * KCK) == 0
    NAJ = AR // RPJ  # dma16 per wave per chunk (RPJ rows x RBA B each)
    NA = NAJ + 1  # A ops per wave per chunk: dma16s + 1 x dma4 (scales)
    # B ops per wave per chunk: 2 k-steps x 4 tiles + gate / up scale words (+ the L2 prefetch of the
    # chunk PFD ahead: one LDS-DMA touching the wave's 64 lines of it)
    NB = 4 * KCK + 2 + (1 if PFD else 0)
    # sched barriers after the GEMM issue groups (the static vmcnt waits count loads in issue order;
    # needed by the straight-line item paths): 1 nothing crosses, 2 only VMEM is held
    SCB = int(_os.environ.get("BIGM_SCB", "1" if MODE in (3, 4) else "0"))
    SCM = 0 if SCB == 1 else 0x38F
    G2OPQ = _os.environ.get("BIGM_G2OPQ", "1") == "1"
    SCB2 = _os.environ.get("BIGM_SCB2", "1") == "1"
    RSAUX = int(_os.environ.get("BIGM_RSAUX", "0"))  # route-row store cache policy (2: nt)  # sched barriers after the GEMM2 B issue groups
    NB2 = 6 * G2W  # GEMM2 B ops per k-step: 4 tiles + 2 scale words per 64 columns
    # STG (R8): a group's MXFP8 route rows staged in LDS (the A ring, idle in GEMM2; 2 buffers), then
    # stored as whole 256-column row segments: 8 + 1 stores per wave (full lines) instead of 2 x MTEW
    STG = _os.environ.get("BIGM_STG", "0") == "1" and R8 and G2W == 1 and MODE in (0, 2) and NRS == 1
    STGB = RG * NW * 64 + RG * NW * 2  # one staging buffer: codes [RG][NW x 64] + scales [RG][2 NW]
    assert not STG or (RG * NW * 64) % (NTHU * 16) == 0 and (RG * NW * 2) % (NTHU * 4) == 0
    assert not STG or lay["LRIX"] - lay["LA"] >= 2 * STGB, "STG: the A ring + scales hold 2 buffers"
    NST2 = (RG * NW * 64 // (NTHU * 16) + RG * NW * 2 // (NTHU * 4)) if STG else 2 * MTEW * G2W  # route stores per wave per group
    # steady-state GEMM1 issue order per chunk c: B(c + BST), then A(c + NAB - 1)
    WAIT1 = min(NA + (NA + NB) * (BST - 1), (NA + NB) * (NAB - 2))
    VSP = _os.environ.get("BIGM_VSP", "0") == "1" and MODE != 3
    PINB = _os.environ.get("BIGM_PINB", "1") == "1"  # sched barriers around the GEMM1 chunk barrier
    VSPM = int(_os.environ.get("BIGM_VSPM", "3"))  # MFMAs per spread VMEM load
    # (spread: a chunk's A / B loads in any order within it -> one group fewer counted for B)
    WAIT1V = min((NA + NB) * (BST - 1), (NA + NB) * (NAB - 2))
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
    # EPP: the previous group's route-row epilogue split by row tile over this group's k-steps (its
    # VALU / stores issue while this group's MFMAs execute); per step: MFMAs, epilogue slice, B2 issue
    EPP = _os.environ.get("BIGM_EPP", "0") == "1" and G2W == 1 and not STG
    MSPL = [tuple(range(kk * MTEW // KS2, (kk + 1) * MTEW // KS2)) for kk in range(KS2)]
    STK = [2 * len(MSPL[kk]) for kk in range(KS2)]  # route stores of step kk's slice

    def _wait2e(kk):
        return min(sum(STK[j % KS2] + NB2 for j in range(kk - BS2 + 1, kk)), 63)

    if EPP:
        WAIT2 = [_wait2e(kk) for kk in range(KS2)]
    ST_AUX = st_aux


    def act_batch(gs, us):
        return _activation_mul_batch(gs, us, act=act, swiglu_limit=swiglu_limit)

    @flyc.jit
    def part_fill(L, tid, r0, rows, rr, rwt):
        gpu.barrier()
        if tid < i32(RG):
            rc = (tid < rows).select(tid, i32(0))
            if const_expr(PACKED):
                v_ = fx.Int32(bld(rr, (r0 + rc) * i32(4), 0, T.i32, AUX_SYS))
                lds_st(L, i32(LRIX) + tid * i32(4), (v_ & i32(0xFFFFFF)) * i32(TOPK) + (v_ >> i32(24)))
                lds_st(L, i32(LWT) + tid * i32(4), fx.Int32(bld(rwt, (r0 + rc) * i32(4), 0, T.i32, AUX_SYS)))
            else:
                lds_st(L, i32(LRIX) + tid * i32(4), fx.Int32(bld(rr, (r0 + rc) * i32(4), 0, T.i32)))
                lds_st(L, i32(LWT) + tid * i32(4), fx.Int32(bld(rwt, (r0 + rc) * i32(4), 0, T.i32)))
        gpu.barrier()


    @flyc.jit
    def part_g1(L, tid, e, rows, rx, rxs, w1, rws1, ih, plo, phi):
        lane = tid % i32(64)
        w = rocdl.readfirstlane(T.i32, tid // i32(64))
        wc = w % i32(NW)  # column group
        rb = (w // i32(NW)) * i32(RG // NRS)  # first row of this wave's tiles
        q4 = lane // i32(16)
        zero = fx.Vector.filled(4, 0.0, fx.Float32)
        pg0 = ih * i32(PASSES)  # this slice's first GEMM1 pass (global)
        kg0 = ih * i32(KS2)  # and GEMM2 k-step
        if const_expr(MODE >= 3 and _os.environ.get("BIGM_DRAIN") == "1"):  # [dbg]
            # item units: the previous item's outstanding stores drained before the GEMM1
            # pipeline's static vmcnt waits (they assume only this item's loads are in flight)
            wait_vm(0)
            wait_lgkm0()
            gpu.barrier()
        # ---- A operand addressing (this wave's 16 rows) ----
        arow = []
        for j in range_constexpr(NAJ):
            row = w * i32(AR) + i32(j * RPJ) + lane // i32(LPR)
            col = (lane % i32(LPR)) ^ (((row >> i32(1)) if const_expr(SWZ) else row) & i32(LPR - 1))
            tok = lds_ld_i32(L, i32(LRIX) + row * i32(4)) // i32(TOPK)
            arow.append(tok * i32(KH) + col * i32(16))
        # scales: lane -> (row w * AR + lane % AR, k-step (lane // AR) % KCK)
        srow = w * i32(AR) + lane % i32(AR)
        stok = lds_ld_i32(L, i32(LRIX) + srow * i32(4)) // i32(TOPK)
        soff = stok * i32(H // 32) + ((lane // i32(AR)) % i32(KCK)) * i32(4)

        def issue_a(c, buf):
            base = L + i32(LA + buf * RG * RBA)
            for j in range_constexpr(NAJ):
                dma16(base + (w * i32(AR) + i32(j * RPJ)) * i32(RBA), rx, i32(0) if const_expr(dbg & 2) else arow[j], c * i32(RBA))
            dma4(L + i32(LAS + buf * NWT * 256) + w * i32(256), rxs, i32(0) if const_expr(dbg & 2) else soff, c * i32(4 * KCK))
            if const_expr(SCB):  # no reordering across issue groups
                rocdl.sched_barrier(SCM)

        def a_frag(buf, row, k):
            # SWZ: rows r, r + 8 (one 16-lane phase of ds_read_b128) on different banks
            col = (i32(k * 4) + q4) ^ (((row >> i32(1)) if const_expr(SWZ) else row) & i32(LPR - 1))
            return lds_ld(L, i32(LA + buf * RG * RBA) + row * i32(RBA) + col * i32(16), V4I, 16)

        def a_scale(buf, row, k):
            if const_expr(dbg & 16):  # [dbg] no A scale LDS reads
                return i32(127)
            wd = (row // i32(AR)) * i32(64) + i32(k * AR) + row % i32(AR)
            return fx.Int32(lds_ld(L, i32(LAS + buf * NWT * 256) + wd * i32(4) + q4, T.i8, 1)) & i32(0xFF)

        # AFC: one per-lane base + compile-time offsets (row tile m, k-step, ring buffer) for the A
        # fragment / scale reads (one wave's rows are every row: the swizzle depends on the lane
        # only), instead of per-tile address registers (spilled under the register pressure)
        AFC = NRS == 1 and AR % 16 == 0 and _os.environ.get("BIGM_AFC", "1") == "1"
        if const_expr(AFC):
            l16 = lane % i32(16)
            swz = ((l16 >> i32(1)) if const_expr(SWZ) else l16) & i32(LPR - 1)
            abase = [L + i32(LA) + l16 * i32(RBA) + ((i32(k * 4) + q4) ^ swz) * i32(16) for k in range(KCK)]
            sbase = L + i32(LAS) + l16 * i32(4) + q4

        def a_frag_m(buf, m, k):
            if const_expr(AFC):
                return lds_ld(abase[k], buf * RG * RBA + m * 16 * RBA, V4I, 16)
            return a_frag(buf, rb + i32(m * 16) + lane % i32(16), k)

        def a_scale_m(buf, m, k):
            if const_expr(dbg & 16):
                return i32(127)
            if const_expr(AFC):
                wd0 = (m * 16) // AR * 64 + k * AR + (m * 16) % AR
                return fx.Int32(lds_ld(sbase, buf * NWT * 256 + wd0 * 4, T.i8, 1)) & i32(0xFF)
            return a_scale(buf, rb + i32(m * 16) + lane % i32(16), k)

        # ---- GEMM1 B (w1 of expert e): pass p, chunk c ----
        rw1 = rsrc(fx.Int64(w1) + fx.Int64(e) * fx.Int64(2 * I * KH))

        def issue_b(p, c):
            icol0 = (pg0 + p) * i32(NW * 32) + wc * i32(32)
            vg0 = icol0 * i32(KH) + lane * i32(16)
            vs = (vg0, vg0 + i32(16 * KH), vg0 + i32(I * KH), vg0 + i32((I + 16) * KH))
            b = []
            for k in range_constexpr(KCK):
                for t in range_constexpr(4):
                    if const_expr(dbg & 1):
                        b.append(fx.Vector(bld(rw1, lane * i32(16) + i32(t * 1024), i32(k * 4096), V4I)))
                    else:
                        b.append(fx.Vector(bld(rw1, vs[t], (c * i32(KCK) + i32(k)) * i32(1024), V4I)))
            # one scale word = 2 k-steps (KCK 1: two chunks read it -- separate loads, opaque offset)
            cw = (c * i32(KCK)) // i32(2)
            sg = ((e * i32(2 * I) + icol0) // i32(32)) * i32(CH1 * 256) + cw * i32(256)
            su = ((e * i32(2 * I) + i32(I) + icol0) // i32(32)) * i32(CH1 * 256) + cw * i32(256)
            vso = _opaque(lane * i32(4)) if const_expr(KCK == 1) else lane * i32(4)
            b.append(fx.Int32(bld(rws1, vso, sg, T.i32)))
            b.append(fx.Int32(bld(rws1, vso, su, T.i32)))
            if const_expr(PFD):
                # lane -> block (t, k) = lane / 8, line lane % 8 of the chunk PFD ahead (next pass past
                # the end): its B lands in L2 before the ring loads it
                cq = c + i32(PFD)
                pq = fx.min(p + cq // i32(KCH), i32(PASSES - 1))
                blk = lane // i32(8)
                t_ = blk % i32(4)
                pf = ((pg0 + pq) * i32(NW * 32) + wc * i32(32)) * i32(KH) + (t_ % i32(2)) * i32(16 * KH) + (t_ // i32(2)) * i32(I * KH) + (blk // i32(4)) * i32(1024) + (lane % i32(8)) * i32(128)
                dma4(L + i32(LPF) + w * i32(256), rw1, pf, (cq % i32(KCH)) * i32(2048))
            if const_expr(SCB):
                rocdl.sched_barrier(SCM)
            return b

        APF = _os.environ.get("BIGM_APF", "0") == "1"  # [exp] both k-steps' A reads before the MFMAs

        def g1_chunk(acc, bq, buf, c=0):
            # c: the chunk (compile time; KCK 1: its k-step's half of the scale words)
            sgw, suw = bq[4 * KCK], bq[4 * KCK + 1]
            if const_expr(APF):
                afk = [[a_frag_m(buf, m, k) for m in range(MTEW)] for k in range(KCK)]
                sak = [[a_scale_m(buf, m, k) for m in range(MTEW)] for k in range(KCK)]
            for k in range_constexpr(KCK):
                kp = (c * KCK + k) % 2  # the global k-step's half of the scale word
                sc = [(x >> i32(8 * (kp * 2 + h))) & i32(0xFF) for x in (sgw, suw) for h in (0, 1)]
                if const_expr(APF):
                    afs, sas = afk[k], sak[k]
                else:
                    afs = [a_frag_m(buf, m, k) for m in range(MTEW)]
                    sas = [a_scale_m(buf, m, k) for m in range(MTEW)]
                for m in range_constexpr(MTEW):
                    for t in range_constexpr(4):
                        j = (t // 2) * 2 * MTEW + m * 2 + t % 2
                        acc[j] = mfma(bq[k * 4 + t], afs[m], acc[j], sc[t], sas[m])
            return acc

        def act_store(acc, p):
            for m in range_constexpr(MTEW):
                row = rb + i32(m * 16) + lane % i32(16)
                gs, us_ = [], []
                for t in range_constexpr(2):
                    ga, ua = fx.Vector(acc[m * 2 + t]), fx.Vector(acc[2 * MTEW + m * 2 + t])
                    for v in range_constexpr(4):
                        gs.append(fx.Float32(ga[v]))
                        us_.append(fx.Float32(ua[v]))
                xs_ = [fx.Float32(x).to(fx.BFloat16).to(fx.Float32) for x in act_batch(gs, us_)]
                am = amax(xs_)
                am = am.maximumf(am.shuffle_xor(i32(16), i32(64)))
                am = am.maximumf(am.shuffle_xor(i32(32), i32(64)))
                e8, qs = _e8m0_from_amax(am, max_norm=6.0)
                for t in range_constexpr(2):
                    col = (i32(0) if const_expr(P1) else p * i32(NW * 32)) + wc * i32(32) + i32(t * 16) + q4 * i32(4)
                    pk = fp4_pack(xs_[t * 4 : t * 4 + 4], qs)
                    lds_st(L, i32(LINT) + row * i32(SIS) + col // i32(2), fx.Int16(pk & i32(0xFFFF)), align=2)
                if q4 == i32(0):
                    lds_st(L, i32(LINTS) + row * i32(ISS) + (i32(0) if const_expr(P1) else p * i32(NW)) + wc, fx.Int8(e8), align=1)

        # ---- GEMM1: PASSES x KCH chunks, A ring NAB, B ring BST (carried over passes) ----
        # prologue = the steady-state issue order of the BST virtual steps before c = 0
        # (steps before the first real op of a kind issue dummy loads of the same count)
        bpre = []
        if const_expr(MODE == 3):
            # one pass per item, straight-line: only real chunks are loaded (a dead wrap-around load
            # would be removed by the compiler, breaking static counts), so every chunk's vmcnt wait
            # is counted from the actual trace-time issue sequence
            seq = []
            for i in range_constexpr(-max(BST, NAB - 1), 0):
                if const_expr(i + BST >= 0):
                    bpre.append(issue_b(plo, i32(i + BST)))
                    seq.append(("b", i + BST, NB))
                if const_expr(i + NAB - 1 >= 0):
                    issue_a(i32(i + NAB - 1), (i + NAB - 1) % NAB)
                    seq.append(("a", i + NAB - 1, NA))
            ring = [list(b) for b in bpre]
            acc = [zero] * (4 * MTEW)
            for c in range_constexpr(KCH):
                last = max(j for j, q in enumerate(seq) if q[1] == c)
                wait_vm(min(63, sum(q[2] for q in seq[last + 1 :])))
                if const_expr(not (dbg & 8)):
                    gpu.barrier()
                if const_expr(SCB):
                    rocdl.sched_barrier(SCM)
                acc = g1_chunk(acc, ring[c % BST], c % NAB, c)
                if const_expr(c + BST < KCH):
                    ring[c % BST] = issue_b(plo, i32(c + BST))
                    seq.append(("b", c + BST, NB))
                if const_expr(c + NAB - 1 < KCH):
                    issue_a(i32(c + NAB - 1), (c + NAB - 1) % NAB)
                    seq.append(("a", c + NAB - 1, NA))
            gpu.barrier()  # every wave's last A reads done: the intermediate reuses the A ring
            act_store(acc, plo)
        for i in range_constexpr(-max(BST, NAB - 1), 0 if MODE != 3 else -max(BST, NAB - 1)):
            if const_expr(i + BST >= 0):
                bpre.append(issue_b(plo, i32(i + BST)))
            else:
                issue_b(plo, i32(0))
            if const_expr(i + NAB - 1 >= 0):
                issue_a(i32(i + NAB - 1), (i + NAB - 1) % NAB)
            else:
                for _ in range_constexpr(NA):
                    bld(rx, lane * i32(16), 0, V4I)
        NBV = 4 * KCK + 2  # values per B issue (the prefetch DMA returns none)
        if const_expr(MODE == 3):
            pass  # done above
        elif const_expr(UNRP):
            # passes unrolled at trace time: the B ring flows straight through (no loop-carried
            # copies of it)
            ring = [list(b) for b in bpre]
            for pc in range_constexpr(PASSES):
                acc = [zero] * (4 * MTEW)
                for c in range_constexpr(KCH):
                    wait_vm(WAIT1)
                    if const_expr(not (dbg & 8)):
                        gpu.barrier()
                    acc = g1_chunk(acc, ring[c % BST], c % NAB, c)
                    cn = c + BST
                    pn = min(pc + cn // KCH, PASSES - 1)
                    ring[c % BST] = issue_b(i32(pn), i32(cn % KCH))
                    issue_a(i32((c + NAB - 1) % KCH), (c + NAB - 1) % NAB)
                act_store(acc, i32(pc))
        # VSP: the B ring has a spare slot (chunk c + BST loads into the slot chunk c - 1 freed, no
        # dependence on chunk c's MFMAs) and each chunk's loads are spread between its MFMAs
        # (sched_group_barrier): a burst of loads stalls the in-order wave at a full VMEM queue
        RS = BST + 1 if VSP else BST
        assert KCH % RS == 0
        init = [zero] * (4 * MTEW) + [v for b in bpre for v in b] + ([v for v in bpre[0]] if VSP else [])
        # (the straight-line / unrolled paths above run the passes: the loop is then empty, 0 .. 0)
        for p_, stt in range(i32(0) if (UNRP or MODE == 3) else plo, i32(0) if (UNRP or MODE == 3) else phi, i32(1), init=init):
            p = i32(p_)
            acc = [zero] * (4 * MTEW)
            ring = [list(stt[4 * MTEW + s * NBV : 4 * MTEW + (s + 1) * NBV]) for s in range(RS)]
            for c in range_constexpr(KCH):
                if const_expr(PINB):
                    rocdl.sched_barrier(0)
                wait_vm(WAIT1V if VSP else WAIT1)
                if const_expr(not (dbg & 8)):  # [dbg] dbg & 8: no per-chunk barrier
                    gpu.barrier()
                if const_expr(PINB):
                    # nothing crosses the chunk barrier: the A ring reads after every wave's DMA landed
                    # (else the scheduler may hoist them above s_barrier: a race)
                    rocdl.sched_barrier(0)
                acc = g1_chunk(acc, ring[c % RS], c % NAB, c)
                # refill: B of step c + BST (next pass past the end; the last pass
                # reloads its own chunks: harmless, keeps the counts static)
                cn = c + BST
                pn = p + i32(cn // KCH)
                pn = (pn < phi).select(pn, phi - i32(1))
                ring[cn % RS] = issue_b(pn, i32(cn % KCH))
                issue_a(i32((c + NAB - 1) % KCH), (c + NAB - 1) % NAB)
                if const_expr(VSP):
                    for _ in range_constexpr(NA + NB):
                        rocdl.sched_group_barrier(0x008, VSPM, 0)  # MFMA
                        rocdl.sched_group_barrier(0x020, 1, 0)  # VMEM read
                    rocdl.sched_group_barrier(0x008, 64, 0)
                    rocdl.sched_barrier(0)
            act_store(acc, p)
            res = yield [zero] * (4 * MTEW) + [v for s in range(RS) for v in ring[s]]
        wait_vm(0)
        wait_lgkm0()  # the intermediate's LDS writes land before other waves read them
        gpu.barrier()

    @flyc.jit
    def part_g2(L, tid, e, rows, w2, rws2, r_routes, oob, ih, glo, ghi):
        lane = tid % i32(64)
        w = rocdl.readfirstlane(T.i32, tid // i32(64))
        wc = w % i32(NW)  # column group
        rb = (w // i32(NW)) * i32(RG // NRS)  # first row of this wave's tiles
        q4 = lane // i32(16)
        zero = fx.Vector.filled(4, 0.0, fx.Float32)
        pg0 = ih * i32(PASSES)  # this slice's first GEMM1 pass (global)
        kg0 = ih * i32(KS2)  # and GEMM2 k-step
        # ---- GEMM2 from the LDS intermediate: NG2 groups x KS2 k-steps, B ring BS2 ----
        rw2 = rsrc(fx.Int64(w2) + fx.Int64(e) * fx.Int64(H * (I // 2)))

        def issue_b2(gi, kk):
            b = []
            for hw in range_constexpr(G2W):
                b += issue_b2h(gi * i32(G2W) + i32(hw), kk)
            if const_expr(SCB or SCB2):  # the loads stay where the ring issues them (no sinking)
                rocdl.sched_barrier(SCM)
            return b

        def issue_b2h(gi, kk):  # one 64-column group (gi in 64-column units)
            n0 = (wc + i32(NW) * gi) * i32(64)
            kg = kg0 + kk  # global k-step
            b = [
                fx.Vector(bld(rw2, lane * i32(16), i32(t * 1024) if const_expr(dbg & 4) else (n0 + i32(t * 16)) * i32(I // 2) + kg * i32(1024), V4I))
                for t in range(4)
            ]
            for pp in range_constexpr(2):
                rb = (e * i32(H) + n0 + i32(pp * 32)) // i32(32)
                # k-steps 2j, 2j + 1 read the same scale word: kept as separate loads (an opaque
                # offset), else CSE merges them and the static counts force a vmcnt(0) drain
                vo = _opaque(lane * i32(4)) if const_expr(G2OPQ) else lane * i32(4)
                b.append(fx.Int32(bld(rws2, vo, (rb * i32(CH2) + kg // i32(2)) * i32(256), T.i32)))
            return b

        row_meta = []
        for m in range_constexpr(MTEW):
            R = rb + i32(m * 16) + lane % i32(16)
            ok = R < rows
            Rc = ok.select(R, i32(0))
            row_meta.append(
                (
                    ok,
                    lds_ld_i32(L, i32(LWT) + Rc * i32(4)).bitcast(fx.Float32),
                    lds_ld_i32(L, i32(LRIX) + Rc * i32(4)),
                )
            )


        def epi2(acc, gi, valid, mset):
            # route rows of row tiles mset of group gi (valid False: stores dropped, counts kept)
            sbuf = L + i32(LA) + (gi & i32(1)) * i32(STGB)  # STG: the group's staging buffer
            for hw in range_constexpr(G2W):
                n0 = (wc + i32(NW) * (gi * i32(G2W) + i32(hw))) * i32(64)
                aoff = hw * 4 * MTEW
                ccol = (q4 & i32(1)) * i32(16) + (q4 >> i32(1)) * i32(8)
                for m in (mset if R8 else ()):
                    # MXFP8 route rows (the small-M engine's store_route_fp8 layout)
                    ok, wt, rix = row_meta[m]
                    ok = ok & valid
                    d, e8s = [], []
                    for half in range_constexpr(2):
                        f = [fx.Float32(fx.Vector(acc[aoff + m * 4 + 2 * half + h])[i]) * wt for h in range(2) for i in range(4)]
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
                    if const_expr(STG):
                        R = rb + i32(m * 16) + lane % i32(16)
                        lds_st(sbuf, R * i32(NW * 64) + wc * i32(64) + q4 * i32(16), ov, align=16)
                        if q4 == i32(0):
                            lds_st(sbuf + i32(RG * NW * 64), R * i32(2 * NW) + wc * i32(2), sc, align=2)
                    else:
                        bst(ov, r_routes, ok.select(rix * i32(H) + n0 + q4 * i32(16), oob), 0, RSAUX)
                        soff = oob // i32(2) + rix * i32(H // 32) + n0 // i32(32)
                        bst(sc, r_routes, (ok & (q4 == i32(0))).select(soff, oob), 0, RSAUX)
                if const_expr(STG):
                    # every wave's 64 columns of the group in LDS: whole 256-column row segments out
                    wait_lgkm0()
                    gpu.barrier()
                    gcol0 = i32(NW) * gi * i32(64)
                    for j in range_constexpr(RG * NW * 64 // (NTHU * 16)):
                        x = tid + i32(j * NTHU)
                        row = x // i32(NW * 4)
                        c16 = x - row * i32(NW * 4)
                        v = lds_ld(sbuf, row * i32(NW * 64) + c16 * i32(16), V4I, 16)
                        rixr = lds_ld_i32(L, i32(LRIX) + row * i32(4))
                        bst(v, r_routes, (row < rows).select(rixr * i32(H) + gcol0 + c16 * i32(16), oob), 0, RSAUX)
                    for j in range_constexpr(RG * NW * 2 // (NTHU * 4)):
                        x = tid + i32(j * NTHU)
                        row = x // i32(NW // 2)
                        dw = x - row * i32(NW // 2)
                        v = lds_ld_i32(sbuf, i32(RG * NW * 64) + row * i32(2 * NW) + dw * i32(4))
                        rixr = lds_ld_i32(L, i32(LRIX) + row * i32(4))
                        so = oob // i32(2) + rixr * i32(H // 32) + gcol0 // i32(32) + dw * i32(4)
                        bst(v, r_routes, (row < rows).select(so, oob), 0, RSAUX)
                for m in (() if R8 else mset):
                    ok, wt, rix = row_meta[m]
                    ok = ok & valid
                    pk = []
                    for t in range_constexpr(4):
                        av = fx.Vector(acc[aoff + m * 4 + t])
                        pk.append(
                            [pack_bf16x2(fx.Float32(av[2 * h]) * wt, fx.Float32(av[2 * h + 1]) * wt) for h in range(2)]
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
                        rixd = (rix & i32(15)) if const_expr(dbg & 64) else rix  # [dbg] 64: stores hit 16 rows (L2)
                        bst(ov, r_routes, (ok & (i32(dbg & 32) == i32(0))).select((rixd * i32(H) + col) * i32(2), oob), 0, RSAUX)

        # prologue = the steady issue order of the BS2 steps before (0, 0): their B2 loads,
        # then (the previous group's last step) NST2 route stores, here dropped (OOB)
        def pro2_epp():
            # the steady order of the BS2 virtual steps before (glo, 0): slice stores (dropped), B2
            out = []
            for s_ in range_constexpr(BS2):
                for _ in range_constexpr(STK[(s_ - BS2) % KS2]):
                    bst(fx.Vector.filled(4, 0, fx.Int32), r_routes, oob, 0)
                out.append(issue_b2(glo + i32(s_ // KS2), i32(s_ % KS2)))
            return out

        def pro2():
            out = [issue_b2(glo + i32(s_ // KS2), i32(s_ % KS2)) for s_ in range(BS2)]
            for _ in range_constexpr(NST2):
                bst(fx.Vector.filled(4, 0, fx.Int32), r_routes, oob, 0)
            return out

        assert (-1) % KS2 == KS2 - 1
        b2pre = pro2_epp() if EPP else pro2()
        init2 = [v for b in b2pre for v in b] + ([zero] * (4 * MTEW) if EPP else [])
        for gi_, st2 in range(glo, ghi, i32(1), init=init2):
            gi = i32(gi_)
            sbuf = L + i32(LA) + (gi & i32(1)) * i32(STGB)  # STG: this group's staging buffer
            ring2 = [list(st2[s * NB2 : (s + 1) * NB2]) for s in range(BS2)]
            accp = list(st2[BS2 * NB2 :]) if EPP else None  # EPP: the previous group's accumulators
            acc = [zero] * (4 * MTEW * G2W)
            for kk in range_constexpr(KS2):
                wait_vm(WAIT2[kk])
                bqa = ring2[kk % BS2]
                sh0 = ((kg0 + i32(kk)) & i32(1)) * i32(16)
                afs, sas = [], []
                for m in range_constexpr(MTEW):
                    row = rb + i32(m * 16) + lane % i32(16)
                    afs.append(lds_ld(L, i32(LINT) + row * i32(SIS) + i32(kk * 64) + q4 * i32(16), V4I, 16))
                    sas.append(
                        fx.Int32(lds_ld(L, i32(LINTS) + row * i32(I // 32) + i32(kk * 4) + q4, T.i8, 1))
                        & i32(0xFF)
                    )
                for hw in range_constexpr(G2W):
                    bq = bqa[hw * 6 : hw * 6 + 6]
                    s0, s1 = bq[4], bq[5]
                    sb = [
                        (s0 >> sh0) & i32(0xFF),
                        (s0 >> (sh0 + i32(8))) & i32(0xFF),
                        (s1 >> sh0) & i32(0xFF),
                        (s1 >> (sh0 + i32(8))) & i32(0xFF),
                    ]
                    for m in range_constexpr(MTEW):
                        for t in range_constexpr(4):
                            j = hw * 4 * MTEW + m * 4 + t
                            acc[j] = mfma(bq[t], afs[m], acc[j], sb[t], sas[m])
                if const_expr(EPP):
                    epi2(accp, gi - i32(1), gi > glo, MSPL[kk])
                sn = kk + BS2
                gn = gi + i32(sn // KS2)
                gn = (gn < ghi).select(gn, ghi - i32(1))
                ring2[kk % BS2] = issue_b2(gn, i32(sn % KS2))
            if const_expr(not EPP):
                epi2(acc, gi, gi >= glo, tuple(range(MTEW)))
            res2 = yield [v for s in range(BS2) for v in ring2[s]] + (acc if EPP else [])
        if const_expr(EPP):
            epi2(list(res2[BS2 * NB2 :]), ghi - i32(1), ghi > glo, tuple(range(MTEW)))
        wait_vm(0)

    # the global intermediate: one block per unit (rows of different units / passes never share a
    # cache line: G1 items of one unit run concurrently on different CTAs), pass-major inside:
    # codes [pass][RG rows][NW * 16 B], then scales [pass][RG rows][NW B]
    PCB = NW * 16  # code bytes of a row per pass
    PSB = NW  # scale bytes of a row per pass
    UBYTES = PASSES * RG * (PCB + PSB)
    SC0 = PASSES * RG * PCB

    @flyc.jit
    def store_inter_pass(L, tid, rows, rint, uid, pp):
        # pass pp's intermediate columns (LDS) -> the unit's block
        gpu.barrier()
        ub = uid * i32(UBYTES)
        CB = PCB // 16
        for it in range_constexpr((RG * CB + NTHU - 1) // NTHU):
            x = tid + i32(it * NTHU)
            if x < i32(RG * CB):
                r_ = x // i32(CB)
                c_ = x - r_ * i32(CB)
                if r_ < rows:
                    bst(lds_ld(L, i32(LINT) + r_ * i32(SIS) + (i32(0) if const_expr(P1) else pp * i32(PCB)) + c_ * i32(16), V4I, 16), rint,
                        ub + (pp * i32(RG) + r_) * i32(PCB) + c_ * i32(16), 0, AUX_SYS)
        for it in range_constexpr((RG + NTHU - 1) // NTHU):
            r_ = tid + i32(it * NTHU)
            if r_ < rows:
                bst(lds_ld_i32(L, i32(LINTS) + r_ * i32(ISS) + (i32(0) if const_expr(P1) else pp * i32(PSB))), rint,
                    ub + i32(SC0) + (pp * i32(RG) + r_) * i32(PSB), 0, AUX_SYS)

    @flyc.jit
    def store_inter(L, tid, rows, rint, uid):
        for pq in range_constexpr(PASSES):
            store_inter_pass(L, tid, rows, rint, uid, i32(pq))

    @flyc.jit
    def load_inter(L, tid, rows, rint, uid):
        # the unit's block -> its LDS intermediate (rows past `rows`: row 0); every load first
        ub = uid * i32(UBYTES)
        CB = PCB // 16
        NC = (PASSES * RG * CB + NTHU - 1) // NTHU
        NS = (PASSES * RG + NTHU - 1) // NTHU
        ldc, lds_ = [], []
        for it in range_constexpr(NC):
            x = fx.min(tid + i32(it * NTHU), i32(PASSES * RG * CB - 1))
            pq = x // i32(RG * CB)
            y = x - pq * i32(RG * CB)
            r_ = y // i32(CB)
            c_ = y - r_ * i32(CB)
            rs = (r_ < rows).select(r_, i32(0))
            ldc.append((x, pq, r_, c_, fx.Vector(bld(rint, ub + (pq * i32(RG) + rs) * i32(PCB) + c_ * i32(16), 0, V4I, AUX_SYS))))
        for it in range_constexpr(NS):
            x = fx.min(tid + i32(it * NTHU), i32(PASSES * RG - 1))
            pq = x // i32(RG)
            r_ = x - pq * i32(RG)
            rs = (r_ < rows).select(r_, i32(0))
            lds_.append((pq, r_, fx.Int32(bld(rint, ub + i32(SC0) + (pq * i32(RG) + rs) * i32(PSB), 0, T.i32, AUX_SYS))))
        for it in range_constexpr(NC):
            x, pq, r_, c_, v_ = ldc[it]
            lds_st(L, i32(LINT) + r_ * i32(SIS) + pq * i32(PCB) + c_ * i32(16), v_)
        for it in range_constexpr(NS):
            pq, r_, v_ = lds_[it]
            lds_st(L, i32(LINTS) + r_ * i32(I // 32) + pq * i32(PSB), v_)
        wait_lgkm0()
        gpu.barrier()

    if MODE == 3:  # G1 item: one GEMM1 pass (NW * 32 gate + up columns) -> its intermediate columns
        @flyc.jit
        def unit_body(L, tid, e, r0, rows, rr, rwt, rx, rxs, w1, w2, rws1, rws2, r_routes, oob, ih=None, rint=None, sub=None, uid=None):
            part_fill(L, tid, r0, rows, rr, rwt)
            part_g1(L, tid, e, rows, rx, rxs, w1, rws1, i32(0), sub, sub + i32(1))
            store_inter_pass(L, tid, rows, rint, uid, sub)
    elif MODE == 4:  # G2 item: GEMM2 column groups [sub * G2N, ..) of the unit from its global intermediate
        @flyc.jit
        def unit_body(L, tid, e, r0, rows, rr, rwt, rx, rxs, w1, w2, rws1, rws2, r_routes, oob, ih=None, rint=None, sub=None, uid=None):
            part_fill(L, tid, r0, rows, rr, rwt)
            load_inter(L, tid, rows, rint, uid)
            part_g2(L, tid, e, rows, w2, rws2, r_routes, oob, i32(0), sub * i32(G2N), sub * i32(G2N) + i32(G2N))
    elif MODE == 1:  # G1: GEMM1 + act -> intermediate rows in global
        @flyc.jit
        def unit_body(L, tid, e, r0, rows, rr, rwt, rx, rxs, w1, w2, rws1, rws2, r_routes, oob, ih=None, rint=None, sub=None, uid=None):
            ih = i32(0) if ih is None else ih
            part_fill(L, tid, r0, rows, rr, rwt)
            part_g1(L, tid, e, rows, rx, rxs, w1, rws1, ih, i32(0), i32(PASSES))
            store_inter(L, tid, rows, rint, uid)
    elif MODE == 2:  # G2: intermediate rows from global -> GEMM2 -> route rows
        @flyc.jit
        def unit_body(L, tid, e, r0, rows, rr, rwt, rx, rxs, w1, w2, rws1, rws2, r_routes, oob, ih=None, rint=None, sub=None, uid=None):
            ih = i32(0) if ih is None else ih
            part_fill(L, tid, r0, rows, rr, rwt)
            load_inter(L, tid, rows, rint, uid)
            part_g2(L, tid, e, rows, w2, rws2, r_routes, oob, ih, i32(0), i32(NG2))
    else:
        @flyc.jit
        def unit_body(L, tid, e, r0, rows, rr, rwt, rx, rxs, w1, w2, rws1, rws2, r_routes, oob, ih=None, rint=None, sub=None, uid=None):
            ih = i32(0) if ih is None else ih
            part_fill(L, tid, r0, rows, rr, rwt)
            part_g1(L, tid, e, rows, rx, rxs, w1, rws1, ih, i32(0), i32(PASSES))
            part_g2(L, tid, e, rows, w2, rws2, r_routes, oob, ih, i32(0), i32(NG2))

    return lay, unit_body


const_expr = fx.const_expr
