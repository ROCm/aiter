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

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ..mxfp4_gemm_common import _activation_mul_batch, _e8m0_from_amax
from .common import (
    V4I,
    amax,
    bld,
    bst,
    dma4,
    dma16,
    fp4_pack,
    fp8x4_pack,
    i32,
    lds_ld,
    lds_ld_i32,
    lds_st,
    mfma,
    opaque,
    pack_bf16x2,
    rsrc,
    swap16,
    swap32,
    wait_lgkm0,
    wait_vm,
)

NW = 4
NTH = NW * 64
KCK = 2  # GEMM1 k-steps (of 128) per chunk


def bigm_lds(RG, NAB, I, NW=NW):
    """LDS layout (bytes) of one CTA."""
    c = {}
    o = 0

    def take(n, align=16):
        nonlocal o
        o = (o + align - 1) // align * align
        r = o
        o += n
        return r

    c["LA"] = take(
        NAB * RG * 64 * KCK
    )  # A codes: [buf][row][128 B], 16 B columns swizzled
    c["LAS"] = take(NAB * NW * 64 * 4)  # A scales: [buf][wave][lane] words (dma4 order)
    c["LRIX"] = take(RG * 4)
    c["LWT"] = take(RG * 4)
    c["SIS"] = I // 2 + 16
    c["LINT"] = take(RG * c["SIS"])  # intermediate MXFP4 rows
    c["ISS"] = I // 32
    c["LINTS"] = take(RG * (I // 32))  # their E8M0 scales
    c["LCTL"] = take(64)
    c["BYTES"] = (o + 15) // 16 * 16
    return c


def build_bigm_unit(
    H,
    I,
    TOPK,
    RG=128,
    NAB=4,
    BST=3,
    BS2=3,
    act="swiglu",
    swiglu_limit=7.0,
    R8=0,
):
    """The fused unit device function: unit_body(L, tid, e, r0, rows, rr, rwt, rx, rxs, w1,
    w2, rws1, rws2, r_routes, oob): rows rr / rwt[r0 : r0 + rows] (route index, weight) of expert
    e -> route rows wt * GEMM2(act(GEMM1(A[route // TOPK]))) stored at route * H (bf16).
    R8: route rows as MXFP8 (e4m3 + one e8m0 scale per 32 columns): codes [rows][H] at r_routes,
    scales [rows][H / 32] after them (at oob / 2: oob = the route count * H * 2).
    Returns (LDS layout, unit_body)."""
    assert RG % 16 == 0 and RG <= 128 and H % 256 == 0 and I % 128 == 0
    MTE = RG // 16
    KH = H // 2  # A / w1 row bytes
    KCH = H // (128 * KCK)  # GEMM1 chunks (KCK k-steps of 128)
    RBA = 64 * KCK  # A row bytes per chunk
    LPR = RBA // 16  # DMA lanes per A row (16 B each)
    RPJ = 64 // LPR  # A rows per dma16
    MTEW = MTE  # 16-row tiles per wave
    PASSES = I // (NW * 32)  # GEMM1 passes
    NG2 = H // (NW * 64)  # GEMM2 column groups per wave
    KS2 = I // 128  # GEMM2 k-steps
    assert I % (NW * 32) == 0
    CH1 = (H // 32 + 7) // 8
    CH2 = (I // 32 + 7) // 8
    assert KCH % NAB == 0 and KCH % BST == 0 and KS2 % BS2 == 0
    lay = bigm_lds(RG, NAB, I, NW)
    ISS = lay["ISS"]
    LA, LAS, LRIX, LWT = lay["LA"], lay["LAS"], lay["LRIX"], lay["LWT"]
    SIS, LINT, LINTS = lay["SIS"], lay["LINT"], lay["LINTS"]
    AR = RG // NW  # A rows per wave (DMA)
    assert AR in (16, 32) and 64 % (AR * KCK) == 0
    NAJ = AR // RPJ  # dma16 per wave per chunk (RPJ rows x RBA B each)
    NA = NAJ + 1  # A ops per wave per chunk: dma16s + 1 x dma4 (scales)
    NB = (
        4 * KCK + 2
    )  # B ops per wave per chunk: 2 k-steps x 4 tiles + gate / up scale words
    # sched barrier mask after the GEMM2 B issue groups (only VMEM is held)
    SCM = 0x38F
    NB2 = 6  # GEMM2 B ops per k-step: 4 tiles + 2 scale words per 64 columns
    NST2 = 2 * MTEW  # route stores per wave per group
    # steady-state GEMM1 issue order per chunk c: B(c + BST), then A(c + NAB - 1)
    WAIT1 = min(NA + (NA + NB) * (BST - 1), (NA + NB) * (NAB - 2))
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
    def part_g1(L, tid, e, rows, rx, rxs, w1, rws1):
        lane = tid % i32(64)
        w = rocdl.readfirstlane(T.i32, tid // i32(64))
        wc = w % i32(NW)  # column group
        rb = (w // i32(NW)) * i32(RG)  # first row of this wave's tiles
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
            icol0 = p * i32(NW * 32) + wc * i32(32)
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
            su = ((e * i32(2 * I) + i32(I) + icol0) // i32(32)) * i32(
                CH1 * 256
            ) + cw * i32(256)
            vso = lane * i32(4)
            b.append(fx.Int32(bld(rws1, vso, sg, T.i32)))
            b.append(fx.Int32(bld(rws1, vso, su, T.i32)))
            return b

        def g1_chunk(acc, bq, buf, c=0):
            # c: the chunk (compile time)
            sgw, suw = bq[4 * KCK], bq[4 * KCK + 1]
            for k in range_constexpr(KCK):
                kp = (c * KCK + k) % 2  # the global k-step's half of the scale word
                sc = [
                    (x >> i32(8 * (kp * 2 + h))) & i32(0xFF)
                    for x in (sgw, suw)
                    for h in (0, 1)
                ]
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
                    ga, ua = fx.Vector(acc[m * 2 + t]), fx.Vector(
                        acc[2 * MTEW + m * 2 + t]
                    )
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
                    col = p * i32(NW * 32) + wc * i32(32) + i32(t * 16) + q4 * i32(4)
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
                        i32(LINTS) + row * i32(ISS) + p * i32(NW) + wc,
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
        NBV = 4 * KCK + 2  # values per B issue
        assert KCH % BST == 0
        init = [zero] * (4 * MTEW) + [v for b in bpre for v in b]
        phi = i32(PASSES)
        for p_, stt in range(i32(0), phi, i32(1), init=init):
            p = i32(p_)
            acc = [zero] * (4 * MTEW)
            ring = [
                list(stt[4 * MTEW + s * NBV : 4 * MTEW + (s + 1) * NBV])
                for s in range(BST)
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
            res = yield [zero] * (4 * MTEW) + [  # noqa: F841
                v for s in range(BST) for v in ring[s]
            ]
        wait_vm(0)
        wait_lgkm0()  # the intermediate's LDS writes land before other waves read them
        gpu.barrier()

    @flyc.jit
    def part_g2(L, tid, e, rows, w2, rws2, r_routes, oob):
        lane = tid % i32(64)
        w = rocdl.readfirstlane(T.i32, tid // i32(64))
        wc = w % i32(NW)  # column group
        rb = (w // i32(NW)) * i32(RG)  # first row of this wave's tiles
        q4 = lane // i32(16)
        zero = fx.Vector.filled(4, 0.0, fx.Float32)
        # ---- GEMM2 from the LDS intermediate: NG2 groups x KS2 k-steps, B ring BS2 ----
        rw2 = rsrc(fx.Int64(w2) + fx.Int64(e) * fx.Int64(H * (I // 2)))

        def issue_b2(gi, kk):  # one 64-column group
            n0 = (wc + i32(NW) * gi) * i32(64)
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
                # k-steps 2j, 2j + 1 read the same scale word: kept as separate loads (an opaque
                # offset), else CSE merges them and the static counts force a vmcnt(0) drain
                vo = opaque(lane * i32(4))
                b.append(
                    fx.Int32(
                        bld(rws2, vo, (rb * i32(CH2) + kk // i32(2)) * i32(256), T.i32)
                    )
                )
            # the loads stay where the ring issues them (no sinking)
            rocdl.sched_barrier(SCM)
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

        def epi2(acc, gi, valid):
            # route rows of group gi (valid False: stores dropped, counts kept)
            n0 = (wc + i32(NW) * gi) * i32(64)
            ccol = (q4 & i32(1)) * i32(16) + (q4 >> i32(1)) * i32(8)
            for m in range(MTEW) if R8 else ():
                # MXFP8 route rows (the small-M engine's store_route_fp8 layout)
                ok, wt, rix = row_meta[m]
                ok = ok & valid
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
                    ov, r_routes, ok.select(rix * i32(H) + n0 + q4 * i32(16), oob), 0, 0
                )
                soff = oob // i32(2) + rix * i32(H // 32) + n0 // i32(32)
                bst(sc, r_routes, (ok & (q4 == i32(0))).select(soff, oob), 0, 0)
            for m in () if R8 else range(MTEW):
                ok, wt, rix = row_meta[m]
                ok = ok & valid
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
        glo, ghi = i32(0), i32(NG2)
        init2 = [v for b in b2pre for v in b]
        for gi_, st2 in range(glo, ghi, i32(1), init=init2):
            gi = i32(gi_)
            ring2 = [list(st2[s * NB2 : (s + 1) * NB2]) for s in range(BS2)]
            acc = [zero] * (4 * MTEW)
            for kk in range_constexpr(KS2):
                wait_vm(WAIT2[kk])
                bq = ring2[kk % BS2]
                sh0 = (i32(kk) & i32(1)) * i32(16)
                afs, sas = [], []
                for m in range_constexpr(MTEW):
                    row = rb + i32(m * 16) + lane % i32(16)
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
                                i32(LINTS) + row * i32(I // 32) + i32(kk * 4) + q4,
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
                for m in range_constexpr(MTEW):
                    for t in range_constexpr(4):
                        j = m * 4 + t
                        acc[j] = mfma(bq[t], afs[m], acc[j], sb[t], sas[m])
                sn = kk + BS2
                gn = gi + i32(sn // KS2)
                gn = (gn < ghi).select(gn, ghi - i32(1))
                ring2[kk % BS2] = issue_b2(gn, i32(sn % KS2))
            epi2(acc, gi, gi >= glo)
            res2 = yield [v for s in range(BS2) for v in ring2[s]]  # noqa: F841
        wait_vm(0)

    @flyc.jit
    def unit_body(
        L, tid, e, r0, rows, rr, rwt, rx, rxs, w1, w2, rws1, rws2, r_routes, oob
    ):
        part_fill(L, tid, r0, rows, rr, rwt)
        part_g1(L, tid, e, rows, rx, rxs, w1, rws1)
        part_g2(L, tid, e, rows, w2, rws2, r_routes, oob)

    return lay, unit_body


const_expr = fx.const_expr
