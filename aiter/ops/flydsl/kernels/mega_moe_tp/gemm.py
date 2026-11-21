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
    AUX_SYS,
    REL,
    V4F,
    V4I,
    amax,
    bld,
    bst,
    cat8,
    dma4,
    dma16,
    fp4_pack,
    fp8x4_pack,
    fp8x4_unpack,
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
    C_AGALL,
    C_ARDY,
    C_ASEQ,
    C_CNT,
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
    XD, AGS_OV, AGS_RS, ABS = kc.get("XD AGS_OV AGS_RS ABS")
    (RAW,) = kc.get("RAW")
    (ag_rows,) = kc.get("ag_rows")
    # (small-M) a unit's MFMAs over its row tiles only up to 16 rows: 1 tile, else MT
    SMSKIP = kc.SM and MT >= 2
    ABS, ASS, HB = kc.get("ABS ASS HB")
    (TMAX_,) = kc.get("TMAX")
    # (small-M GEMM1) the k-step pair's scale dwords loaded once (with the even step)
    SDD = kc.SM
    # (experiment) GEMM1 B prefetch into L2, BPF k-steps ahead of the ring (0: off)
    BPF = kc.BPF
    BPFO = 1 if BPF > 0 else 0  # (its one VMEM op per k-step)
    (L_BPF,) = kc.get("L_BPF")
    # (HB) half block: 64 columns of GEMM1, wave w: columns 32 (w % 2), K half w // 2
    KH = KS1 // 2
    NCH, NCK, NSC_BLK, RG, SI_STRIDE = kc.get("NCH NCK NSC_BLK RG SI_STRIDE")
    TOPK, XB, a8, a_frag = kc.get("TOPK XB a8 a_frag")
    XSB, APL, SIB, NPC = kc.get("XSB APL SIB NPC")
    LPR = ACB // 16  # A loader lanes per row
    # (the A scale buffers: [plane][half (HB)][k-step])
    HBN = 2 if kc.HB else 1
    act_quant_store, ag_stage_free = kc.get("act_quant_store ag_stage_free")
    ag_wait_chunk, cbar, const_expr = kc.get("ag_wait_chunk cbar const_expr")
    ll_pkt, npp, nsk, route_fp8 = kc.get("ll_pkt npp nsk route_fp8")
    route_region_bytes = kc.route_region_bytes
    (lb,) = kc.get("lb")
    # (SH8) the shared expert's MXFP8 weights: a k-step's B is twice the bytes (2 KB per
    # 16-row tile: [N / 16][K / 128][half][64 lanes][16 B]); its loaders take the ring's
    # slots in pairs (half the ring depth, the same LDS)
    SH8, SH = kc.get("SH8 SH")

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
    def _gemm1_disp(
        L, tid, a, expert, i0, nnb, mte=None, kr=None, out=None, bnt=True, W8=False
    ):
        if const_expr(npp > 1 and kr is None):
            if (nnb % i32(npp)) == i32(0):
                _gemm1(L, tid, a, expert, i0, nnb, npp, mte, bnt=bnt, W8=W8)
            else:
                _gemm1(L, tid, a, expert, i0, nnb, 1, mte, bnt=bnt, W8=W8)
        else:
            _gemm1(L, tid, a, expert, i0, nnb, 1, mte, kr, out, bnt=bnt, W8=W8)

    @flyc.jit
    def _gemm1_any(L, tid, a, expert, i0, nnb, rows, kr, out, half, bnt, W8=False):
        if const_expr(HB and half is not None):
            if half:
                _gemm1h(L, tid, a, expert, i0, bnt=bnt, W8=W8)
            else:
                _gemm1_rows(L, tid, a, expert, i0, nnb, rows, None, None, bnt, W8)
        else:
            _gemm1_rows(L, tid, a, expert, i0, nnb, rows, kr, out, bnt, W8)

    @flyc.jit
    def _gemm1_rows(L, tid, a, expert, i0, nnb, rows, kr, out, bnt, W8=False):
        """(MTSKIP / SMSKIP) the row tiles up to the unit's rows only"""
        if const_expr(MTSKIP and rows is not None):
            if uni(rows) <= i32(3 * 16):
                _gemm1_disp(L, tid, a, expert, i0, nnb, 3, bnt=bnt, W8=W8)
            else:
                _gemm1_disp(L, tid, a, expert, i0, nnb, bnt=bnt, W8=W8)
        elif const_expr(SMSKIP and rows is not None and kr is None):
            if uni(rows) <= i32(16):
                _gemm1_disp(L, tid, a, expert, i0, nnb, 1, bnt=bnt, W8=W8)
            else:
                _gemm1_disp(L, tid, a, expert, i0, nnb, bnt=bnt, W8=W8)
        else:
            _gemm1_disp(L, tid, a, expert, i0, nnb, None, kr, out, bnt=bnt, W8=W8)

    @flyc.jit
    def gemm1(
        L, tid, a, expert, i0, nnb, rows=None, kr=None, out=None, half=None, dup=None
    ):
        """kr: (k0, nk) only k-steps [k0, k0 + nk) of the K dim (nk a multiple of the
        k-step interleave, 4); out: the per-group epilogue (default: act + MXFP4 to
        L_INTER), called as out(L, lane, w, a, accs, grp, P, MTE); half (HB): a 64-column
        half block instead (nnb unused); dup: the expert's weights read by several row
        chunks: cacheable loads (their concurrent reads meet in L2), else non-temporal
        """
        nnb = uni(nnb)
        if const_expr(SH8):
            if uni(expert) == i32(SH):
                _gemm1_w8(L, tid, a, expert, i0, nnb, kr, out, half)
            else:
                _gemm1_dup(L, tid, a, expert, i0, nnb, rows, kr, out, half, dup, False)
        else:
            _gemm1_dup(L, tid, a, expert, i0, nnb, rows, kr, out, half, dup, False)
        wait_vm(0)
        cbar(L, tid)

    @flyc.jit
    def _gemm1_w8(L, tid, a, expert, i0, nnb, kr, out, half):
        """(SH8) the shared expert's MXFP8 GEMM1: one variant per shape (every token's rows:
        all row tiles, cacheable loads, one inter piece per group) -- each variant's
        loop-invariant registers live through the kernel"""
        if const_expr(HB and half is not None):
            if half:
                _gemm1h(L, tid, a, expert, i0, bnt=False, W8=True)
            else:
                _gemm1(L, tid, a, expert, i0, nnb, 1, None, kr, out, bnt=False, W8=True)
        else:
            _gemm1(L, tid, a, expert, i0, nnb, 1, None, kr, out, bnt=False, W8=True)

    @flyc.jit
    def _gemm1_dup(L, tid, a, expert, i0, nnb, rows, kr, out, half, dup, W8):
        if const_expr(dup is None):
            _gemm1_any(L, tid, a, expert, i0, nnb, rows, kr, out, half, True, W8)
        elif dup:
            _gemm1_any(L, tid, a, expert, i0, nnb, rows, kr, out, half, False, W8)
        else:
            _gemm1_any(L, tid, a, expert, i0, nnb, rows, kr, out, half, True, W8)

    def _g1_operands(tid, a, expert, i0, half=False, W8=False):
        lane = tid % i32(64)
        w = uni(tid // i32(64))
        e = uni(expert)
        # (W8: the shared expert's own MXFP8 tensors: row bytes H, scale rows from 0)
        RB = H if W8 else H // 2
        if const_expr(W8):
            rw = rsrc(a["w1h"])
            rws = rsrc(a["w1hs"])
            e = i32(0)
        else:
            rw = rsrc(fx.Int64(a["w1"]) + fx.Int64(e) * fx.Int64(2 * I * (H // 2)))
            rws = rsrc(a["w1s"])
        icol0 = uni(i0) + (w % i32(2) if half else w) * i32(32)
        vg0 = icol0 * i32(RB) + lane * i32(16)
        vg1 = vg0 + i32(16 * RB)
        vu0 = vg0 + i32(I * RB)
        vu1 = vu0 + i32(16 * RB)
        vsl = lane * i32(4)
        sg0 = uni((e * i32(2 * I) + icol0) // i32(32)) * i32(CH1 * 256)
        su0 = uni((e * i32(2 * I) + i32(I) + icol0) // i32(32)) * i32(CH1 * 256)
        return lane, w, rw, rws, (vg0, vg1, vu0, vu1), vsl, sg0, su0

    def _bpf_voff(lane, vs):
        """(BPF) lane's 64 B of the four 1 KB B regions of a k-step: region lane // 16"""
        t = lane // i32(16)
        base = [v - lane * i32(16) for v in vs]
        b = (t == i32(0)).select(
            base[0],
            (t == i32(1)).select(base[1], (t == i32(2)).select(base[2], base[3])),
        )
        return b + (lane % i32(16)) * i32(64)

    def _g1_so(pas, kk, W8=False):
        if W8:
            return pas * i32(128 * H) + kk * i32(2048)
        return pas * i32(128 * (H // 2)) + kk * i32(1024)

    def _b_issue(slot, rw, vs, so, bnt, W8):
        """one k-step's B tiles into a ring slot (W8: each tile's two 1 KB halves)"""
        for t in range_constexpr(4):
            if const_expr(W8):
                for h in range_constexpr(2):
                    dma16(
                        slot + i32(t * 2048 + h * 1024),
                        rw,
                        vs[t],
                        so + i32(h * 1024),
                        nt=bnt,
                    )
            else:
                dma16(slot + i32(t * 1024), rw, vs[t], so, nt=bnt)

    def _b_read(slot, lane, W8):
        """a ring slot's B fragments; W8: readers of each tile's 8 dwords (both halves),
        read right before the tile's MFMAs (four tiles' 32 dwords live at once spill)"""
        if const_expr(W8):

            def rd(t):
                return cat8(
                    lds_ld(slot, i32(t * 2048) + lane * i32(16), V4I, 16),
                    lds_ld(slot, i32(t * 2048 + 1024) + lane * i32(16), V4I, 16),
                )

            return [functools.partial(rd, t) for t in range(4)]
        return [lds_ld(slot, i32(t * 1024) + lane * i32(16), V4I, 16) for t in range(4)]

    def _g1_sso(pas, kk):
        return pas * i32((128 // 32) * CH1 * 256) + (kk // i32(2)) * i32(256)

    def _g1_chunk(L, lane, qb, ks, kh=None):
        """kh (half blocks): this wave pair's K half, the buffer's second chunk for 1"""
        k = ks % KCS
        q = qb + i32(ks // KCS)
        buf = q % i32(NAB)
        rocdl.sched_barrier(0)
        if k == 0:
            wait_a_chunk(L, lane, buf, q)
        abuf = L + i32(L_A) + buf * i32(ABS)
        asbuf = L + i32(L_AS) + buf * i32(ASS) + i32(k * NSC_BLK * 64 * 4)
        if kh is not None:
            abuf = abuf + kh * i32(RG * ACB)
            asbuf = asbuf + kh * i32(KCS * NSC_BLK * 64 * 4)
        return k, buf, abuf, asbuf

    def _g1_mfma(acc, o, b, sgw, suw, kh, abuf, asbuf, lane, k, MTE, W8=False):
        sc = [
            (x >> i32(8 * (kh * 2 + h))) & i32(0xFF) for x in (sgw, suw) for h in (0, 1)
        ]
        # all A fragments / scales of the k-step first, then the MFMAs: the LDS reads
        # overlap instead of one exposed lgkmcnt(0) wait per 16-row block
        for pl in range_constexpr(
            APL
        ):  # (a16: the hi, then the lo term: one live at a time)
            afs, sas = [], []
            for m in range_constexpr(MTE):
                row = i32(m * 16) + lane % i32(16)
                afs.append(a_frag(abuf + i32(pl * 256), row, k, lane // i32(16)))
                sas.append(
                    fx.Int32(
                        lds_ld(
                            asbuf + i32(pl * HBN * KCS * NSC_BLK * 64 * 4),
                            row * i32(4) + lane // i32(16),
                            T.i8,
                            1,
                        )
                    )
                    & i32(0xFF)
                )
            if const_expr(W8):  # (a tile's fragment read right before its MFMAs)
                for t in range_constexpr(4):
                    bt = b[t]()
                    for m in range_constexpr(MTE):
                        j = o + (t // 2) * 2 * MTE + m * 2 + t % 2
                        acc[j] = mfma(bt, afs[m], acc[j], sc[t], sas[m], a8, a8=True)
            else:
                for m in range_constexpr(MTE):
                    for t in range_constexpr(4):
                        j = o + (t // 2) * 2 * MTE + m * 2 + t % 2
                        acc[j] = mfma(b[t], afs[m], acc[j], sc[t], sas[m], a8)

    def _g1_out(L, lane, w, a, acc, grp, P, MTE, i0=None):
        """i0: the unit's first column (XD: the intermediate's xg columns)"""
        if const_expr(not XD):  # (XD: the AG staging has its own LDS)
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
                i0,
                a,
            )

    @flyc.jit
    def _gemm1(
        L, tid, a, expert, i0, nnb, P, MTE=None, kr=None, out=None, bnt=True, W8=False
    ):
        MTE = MT if MTE is None else MTE
        lane, w, rw, rws, vs, vsl, sg0, su0 = _g1_operands(
            tid, a, expert, i0, W8=W8
        )
        ring = uni(L + i32(L_RING) + w * i32(nsk * SLOT))
        # (W8: the ring's slots in pairs: half the depth, twice the B ops per k-step)
        NSKW = nsk // 2 if W8 else nsk
        SLOTW = 2 * SLOT if W8 else SLOT
        OPSW = OPS + 4 if W8 else OPS
        BPFW = 0 if W8 else BPF
        BPFOW = 1 if BPFW > 0 else 0
        BSO = 8192 if W8 else 4096  # the scale words' offset in a slot
        if const_expr(BPFW > 0):
            bpf_dst, bpf_v = L + i32(L_BPF) + w * i32(256), _bpf_voff(lane, vs)
        IL = NSKW * (P * KCS) // math.gcd(NSKW, P * KCS)
        assert KS1 * P % IL == 0
        if const_expr(kr is None):
            k0, nk = i32(0), i32(KS1)
        else:
            assert P == 1
            k0, nk = uni(kr[0]), uni(kr[1])
        total = nnb * nk
        qbase = lds_ld_i32(L, L_CTL + C_QBASE * 4 + w * i32(4))
        GS = nk * i32(P)
        nch = nk // i32(KCS)  # A chunks per group

        # (SDD, one piece: a scale dword serves a k-step pair: loaded with the even one)
        sdd = SDD and P == 1

        def issue_b(g, slot_idx, odd=False):
            gc = fx.min(g, total - i32(1))
            grp = gc // GS
            r = gc - grp * GS
            kk = r // i32(P)
            pas = grp * i32(P) + (r - kk * i32(P))
            kk = kk + k0
            slot = ring + i32(slot_idx * SLOTW)
            _b_issue(slot, rw, vs, _g1_so(pas, kk, W8), bnt, W8)
            if not (sdd and odd):
                sso = _g1_sso(pas, kk)
                dma4(slot + i32(BSO), rws, vsl, sg0 + sso)
                dma4(slot + i32(BSO + 256), rws, vsl, su0 + sso)
            if const_expr(BPFW > 0):
                # (into L2 BPF k-steps ahead of the ring: its loads then hit)
                gp = fx.min(g + i32(BPF), total - i32(1))
                grp = gp // GS
                r = gp - grp * GS
                kp = r // i32(P)
                pp = grp * i32(P) + (r - kp * i32(P))
                dma4(bpf_dst, rw, bpf_v, _g1_so(pp, kp + k0))

        def ops_newer(s_):
            # VMEM ops issued after step s_'s (the NSKW - 1 later slots)
            if not sdd:
                return (NSKW - 1) * (OPSW + BPFOW)
            return sum(OPSW + BPFOW - 2 * ((s_ + j) % 2) for j in range(1, NSKW))

        for kk in range_constexpr(NSKW):
            issue_b(i32(kk), kk, kk % 2 == 1)
        zero = fx.full(4, 0.0, fx.Float32)
        for grp_ in range(i32(0), nnb // i32(P), i32(1)):
            grp = i32(grp_)
            init = [zero] * (4 * MTE * P)
            for g0_, st in range(grp * GS, (grp + i32(1)) * GS, i32(IL), init=init):
                g0 = i32(g0_)
                acc = list(st)
                qb = qbase + grp * nch + (g0 - grp * GS) // i32(P * KCS)
                sgw = suw = None
                for ks in range_constexpr(IL // P):
                    k, buf, abuf, asbuf = _g1_chunk(L, lane, qb, ks)
                    for pp in range_constexpr(P):
                        s_ = ks * P + pp
                        g = g0 + i32(s_)
                        wait_vm(ops_newer(s_))
                        slot = ring + i32((s_ % NSKW) * SLOTW)
                        b = _b_read(slot, lane, W8)
                        if not (sdd and s_ % 2 == 1):
                            sgw = lds_ld_i32(slot, i32(BSO) + lane * i32(4))
                            suw = lds_ld_i32(slot, i32(BSO + 256) + lane * i32(4))
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
                            W8,
                        )
                        wait_lgkm0()
                        rocdl.sched_barrier(0)
                        issue_b(g + i32(NSKW), s_ % NSKW, s_ % 2 == 1)
                        rocdl.sched_barrier(0)
                    if k == KCS - 1:
                        release_a_chunk(L, lane, buf)
                res = yield acc
            (_g1_out if out is None else out)(L, lane, w, a, res, grp, P, MTE, i0)
        qend = qbase + (nnb // i32(P)) * nch
        if lane == i32(0):
            lds_st(L, L_CTL + C_QBASE * 4 + w * i32(4), qend)

    @flyc.jit
    def _gemm1h(L, tid, a, expert, i0, bnt=True, W8=False):
        """(HB) GEMM1 of a 64-column half block: wave w the 32 columns 32 (w % 2) over the
        K half w // 2 (KH k-steps; buffer q holds both halves' A chunks), the second pair's
        accumulators added through LDS (its own B ring), act + MXFP4 by the first pair
        """
        MTE = MT
        lane, w, rw, rws, vs, vsl, sg0, su0 = _g1_operands(
            tid, a, expert, i0, True, W8
        )
        NSKW = nsk // 2 if W8 else nsk
        SLOTW = 2 * SLOT if W8 else SLOT
        OPSW = OPS + 4 if W8 else OPS
        BPFW = 0 if W8 else BPF
        BPFOW = 1 if BPFW > 0 else 0
        BSO = 8192 if W8 else 4096
        if const_expr(BPFW > 0):
            bpf_dst, bpf_v = L + i32(L_BPF) + w * i32(256), _bpf_voff(lane, vs)
        kh = w // i32(2)
        ring = uni(L + i32(L_RING) + w * i32(nsk * SLOT))
        IL = NSKW * KCS // math.gcd(NSKW, KCS)
        assert KH % IL == 0
        k0 = kh * i32(KH)
        qbase = lds_ld_i32(L, L_CTL + C_QBASE * 4 + w * i32(4))

        def issue_b(g, slot_idx, odd=False):
            kk = fx.min(g, i32(KH - 1)) + k0
            slot = ring + i32(slot_idx * SLOTW)
            _b_issue(slot, rw, vs, _g1_so(i32(0), kk, W8), bnt, W8)
            if not (SDD and odd):
                sso = _g1_sso(i32(0), kk)
                dma4(slot + i32(BSO), rws, vsl, sg0 + sso)
                dma4(slot + i32(BSO + 256), rws, vsl, su0 + sso)
            if const_expr(BPFW > 0):
                kp = fx.min(g + i32(BPFW), i32(KH - 1)) + k0
                dma4(bpf_dst, rw, bpf_v, _g1_so(i32(0), kp))

        def ops_newer(s_):
            if not SDD:
                return (NSKW - 1) * (OPSW + BPFOW)
            return sum(OPSW + BPFOW - 2 * ((s_ + j) % 2) for j in range(1, NSKW))

        for kk in range_constexpr(NSKW):
            issue_b(i32(kk), kk, kk % 2 == 1)
        zero = fx.full(4, 0.0, fx.Float32)
        for g0_, st in range(i32(0), i32(KH), i32(IL), init=[zero] * (4 * MTE)):
            g0 = i32(g0_)
            acc = list(st)
            qb = qbase + g0 // i32(KCS)
            sgw = suw = None
            for ks in range_constexpr(IL):
                k, buf, abuf, asbuf = _g1_chunk(L, lane, qb, ks, kh)
                g = g0 + i32(ks)
                wait_vm(ops_newer(ks))
                slot = ring + i32((ks % NSKW) * SLOTW)
                b = _b_read(slot, lane, W8)
                if not (SDD and ks % 2 == 1):
                    sgw = lds_ld_i32(slot, i32(BSO) + lane * i32(4))
                    suw = lds_ld_i32(slot, i32(BSO + 256) + lane * i32(4))
                _g1_mfma(acc, 0, b, sgw, suw, ks & 1, abuf, asbuf, lane, k, MTE, W8)
                wait_lgkm0()
                rocdl.sched_barrier(0)
                issue_b(g + i32(NSKW), ks % NSKW, ks % 2 == 1)
                rocdl.sched_barrier(0)
                if k == KCS - 1:
                    release_a_chunk(L, lane, buf)
            res = yield acc
        acc = list(res)
        if lane == i32(0):
            lds_st(L, L_CTL + C_QBASE * 4 + w * i32(4), qbase + i32(KH // KCS))
        # (the trailing B prefetches into the ring landed before it holds the partials)
        wait_vm(0)
        if kh == i32(1):
            for j in range_constexpr(4 * MTE):
                lds_st(ring, i32(j * 1024) + lane * i32(16), acc[j], 16)
        cbar(L, tid)
        if kh == i32(0):
            pr = uni(L + i32(L_RING) + (w + i32(2)) * i32(nsk * SLOT))
            for j in range_constexpr(4 * MTE):
                acc[j] = fx.Vector(acc[j]) + fx.Vector(
                    lds_ld(pr, i32(j * 1024) + lane * i32(16), V4F, 16)
                )
            _g1_out(L, lane, w, a, acc, i32(0), 1, MTE, i0)

    def unit_fields(L, u):
        f = [
            lds_ld_i32(L, L_CTL + (i32(C_UL) + u * i32(4) + i32(k)) * i32(4))
            for k in range(4)
        ]
        return f[0], f[1], f[2], f[3]

    @flyc.jit
    def a_loader(L, tid, a):
        lane = tid % i32(64)
        # (sized: a chunk's rows past the unit's are out-of-range loads, no memory traffic)
        rx = rsrc(a["ax"], a["ttot"] * i32(XB))
        rxs = rsrc(a["axs"], a["ttot"] * i32(XSB))
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
        icnt = unit_fields(L, u)[2]
        nnb = icnt // i32(128)
        for r0_ in range(i32(0), R, i32(RG)):
            r0 = i32(r0_)
            rows = fx.min(R - r0, i32(RG))
            arow = []
            for j in range_constexpr(NA_ROWOPS):
                # (LPR lanes x 16 B per row: 8 MXFP4, 16 MXFP8, 32 the a16 hi + lo)
                row = i32(j * (64 // LPR)) + lane // i32(LPR)
                col = (lane % i32(LPR)) ^ (row & i32(7))
                rr = (row < rows).select(row, i32(0))
                tk = lds_ld_i32(L, L_RIX + (r0 + rr) * i32(4)) // i32(TOPK)
                arow.append(
                    (row < rows).select(
                        tk * i32(XB) + col * i32(16), a["ttot"] * i32(XB)
                    )
                )
            ascl = []
            for j in range_constexpr(NSC_BLK):
                row = i32(j * 64) + lane
                rr = (row < rows).select(row, i32(0))
                tk = lds_ld_i32(L, L_RIX + (r0 + rr) * i32(4)) // i32(TOPK)
                ascl.append((row < rows).select(tk * i32(XSB), a["ttot"] * i32(XSB)))
            if const_expr(lb):
                nc, c0 = i32(NCH), i32(0)
            else:
                # (small-M split-K units) K piece s of sk: A chunks [s nc, (s + 1) nc)
                kf = unit_fields(L, u)[3]
                nc = i32(NCH) // fx.max(kf.shrui(i32(24)) & i32(0xFF), i32(1))
                c0 = (kf.shrui(i32(16)) & i32(0xFF)) * nc
            nq = (nnb // g1_group(nnb)) * nc
            if lane == i32(0):
                lds_st(L, L_CTL + C_ARDY * 4, i32(0))
            if const_expr(HB):
                if icnt == i32(64):
                    _a_half(L, lane, a, rx, rxs, arow, ascl)
                nq = (icnt == i32(64)).select(i32(0), nq)
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
                if const_expr(XD):  # noqa: SIM102 (the AG staging in the last buffers)
                    if (q < i32(NAB)) & (b >= _ags_b0(a)):
                        ag_stage_free(L, lane, a)
                rocdl.sched_barrier(0)
                cc = c0 + cidx % nc
                if cidx < nc:  # noqa: SIM102
                    if (cc >= uni(lds_ld_i32(L, L_CTL + C_ARDY * 4))) & (
                        uni(lds_ld_i32(L, L_CTL + C_AGALL * 4)) == i32(0)
                    ):
                        ag_wait_chunk(L, lane, a, a["epoch"], cc)
                _a_chunk(L, b, rx, rxs, arow, ascl, cc)
                _loader_advance(L, lane, q)
            wait_vm(0)
            _loader_flush(L, lane)

    def _a_chunk(L, b, rx, rxs, arow, ascl, cc, half=0):
        """A chunk cc (rows + scales) into buffer b (half 1: its second chunk)"""
        abase = L + i32(L_A) + b * i32(ABS) + i32(half * RG * ACB)
        for j in range_constexpr(NA_ROWOPS):
            # (sc1: the peers' rows of this launch, never a stale L2 line of the last one)
            dma16(abase + i32(j * 1024), rx, arow[j], cc * i32(ACB), aux=AUX_SC1)
        for pl in range_constexpr(APL):  # (a16: the hi, then the lo scales)
            for k in range_constexpr(KCS):
                for j in range_constexpr(NSC_BLK):
                    dst = (
                        L
                        + i32(L_AS)
                        + b * i32(ASS)
                        + i32((pl * HBN + half) * KCS * NSC_BLK * 64 * 4)
                        + i32((k * NSC_BLK + j) * 64 * 4)
                    )
                    soff = cc * i32(8 * APL) + i32(pl * 8 + k * 4)
                    dma4(dst, rxs, ascl[j], soff, sys=True)

    def _ags_b0(a):
        """(XD) the first A buffer this CTA's AG staging rows overlap (NAB: none)"""
        if const_expr(not AGS_OV):
            return i32(NAB)
        nr = ag_rows(a)[2]
        return (i32(NAB * ABS) - nr * i32(AGS_RS)) // i32(ABS)

    @flyc.jit
    def _a_half(L, lane, a, rx, rxs, arow, ascl):
        """(HB) a half block's A: KH / KCS buffers, each chunk c and c + NCH / 2 (the wave
        pairs' K halves); every chunk's all-gather landed first"""
        if uni(lds_ld_i32(L, L_CTL + C_AGALL * 4)) == i32(0):
            for c_ in range(i32(0), i32(NCH), i32(1)):
                if i32(c_) >= uni(lds_ld_i32(L, L_CTL + C_ARDY * 4)):
                    ag_wait_chunk(L, lane, a, a["epoch"], i32(c_))
        for cidx_ in range(i32(0), i32(NCH // 2), i32(1)):
            cidx = i32(cidx_)
            q = lds_ld_i32(L, L_CTL + C_LQ * 4)
            b = q % i32(NAB)
            if (lane == i32(0)) & (q >= i32(NAB)):
                spin_lds_ge(
                    L, L_CTL + C_AFREE * 4 + b * i32(4), i32(NW) * (q // i32(NAB))
                )
            if const_expr(XD):  # noqa: SIM102 (the AG staging in the last buffers)
                if (q < i32(NAB)) & (b >= _ags_b0(a)):
                    ag_stage_free(L, lane, a)
            rocdl.sched_barrier(0)
            _a_chunk(L, b, rx, rxs, arow, ascl, cidx)
            _a_chunk(L, b, rx, rxs, arow, ascl, cidx + i32(NCH // 2), 1)
            _loader_advance(L, lane, q, 2)

    @flyc.jit
    def _loader_advance(L, lane, q, nchunk=1):
        pub = lds_ld_i32(L, L_CTL + C_LPUB * 4)
        if q + i32(1) - pub >= i32(ADEPTH):
            wait_vm(nchunk * NA_L * (ADEPTH - 1))
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

    def _route_terms(fs):
        """MXFP8 of the two 8-value halves (scales shared by the 4 lanes of a 32-column
        block): words, scales and the dequantized values"""
        d, e8s, deq = [], [], []
        for f in fs:
            am = amax(f)
            am = am.maximumf(am.shuffle_xor(i32(16), i32(64)))
            am = am.maximumf(am.shuffle_xor(i32(32), i32(64)))
            e8, qs = _e8m0_from_amax(am, max_norm=448.0)
            p0, p1 = fp8x4_pack(f[0:4], qs), fp8x4_pack(f[4:8], qs)
            d += [p0, p1]
            e8s.append(fx.Int32(e8) & i32(0xFF))
            deq.append(fp8x4_unpack(p0, qs) + fp8x4_unpack(p1, qs))
        return d, e8s, deq

    def store_route_fp8(rs, accs, wt, rix, n0, q4, ok, oob, a, lop=None):
        """lop (a16): the region's lo plane offset (the residual's MXFP8 packets)"""
        fs = [
            [
                fx.Float32(fx.Vector(accs[2 * half + h])[i]) * wt
                for h in range(2)
                for i in range(4)
            ]
            for half in range(2)
        ]
        d, e8s, deq = _route_terms(fs)
        if const_expr(lop is not None):
            res = [[fs[h][i] - deq[h][i] for i in range(8)] for h in range(2)]
            dl, e8l, _ = _route_terms(res)
            _route_store(rs, dl, e8l, rix, n0, q4, ok, oob, a, lop)
        _route_store(rs, d, e8s, rix, n0, q4, ok, oob, a)

    def _route_store(rs, d, e8s, rix, n0, q4, ok, oob, a, lop=None):
        d0, d1 = swap16(d[0], d[1])
        d2, d3 = swap16(d[2], d[3])
        d0, d2 = swap32(d0, d2)
        d1, d3 = swap32(d1, d3)
        lo, hi = [d0, d1], [d2, d3]
        col = n0 + q4 * i32(16)
        if const_expr(DLL):
            e8 = ((q4 >> i32(1)) == i32(0)).select(e8s[0], e8s[1])
            off = (rix * i32(H) + col) * i32(2)
            if const_expr(lop is not None):
                off = off + lop
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
        dup = kw.pop("dup", None)
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

        def run(bnt, W8=False):
            if const_expr(MTSKIP):
                if uni(rows) <= i32(3 * 16):
                    g(mte=3, bnt=bnt, W8=W8)
                else:
                    g(bnt=bnt, W8=W8)
            else:
                g(bnt=bnt, W8=W8)

        def run_dup(W8):
            # (dup: as gemm1's)
            if const_expr(dup is None):
                run(True, W8)
            elif dup:
                run(False, W8)
            else:
                run(True, W8)

        if const_expr(SH8):
            if uni(expert) == i32(SH):
                g(bnt=False, W8=True)  # (one variant, as _gemm1_w8)
            else:
                run_dup(False)
        else:
            run_dup(False)

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
        aoff=None,
        sdd=False,
        odd=False,
        bnt=True,
        W8=False,
    ):
        """sdd: (k-steps from 0, even NKS) a k-step pair's scale dwords loaded once;
        odd: [gi_lo, gi_hi) may hold an odd number of groups; W8: the shared expert's MXFP8
        w2 (its own tensors, the ring's slots in pairs)"""
        MTE = MT if mte is None else mte
        assert not (W8 and nxt is not None)
        SLOTW = 2 * SLOT if W8 else SLOT
        OPSW = OPS + 4 if W8 else OPS
        BSO = 8192 if W8 else 4096
        sdd = sdd and NKS % 2 == 0 and nxt is None
        # the intermediate's first k-step in LDS (aoff: it holds the whole inter dim)
        ka = i32(0) if aoff is None else uni(aoff)
        if const_expr(buf is None):
            ib0, is0 = i32(L_INTER), i32(L_INTERS)
            rix0, wt0 = i32(L_RIX), i32(L_WT)
        else:
            b0 = buf == i32(0)
            ib0 = b0.select(i32(L_INTER), i32(L_B1))
            is0 = b0.select(i32(L_INTERS), i32(L_B1S))
            rix0 = i32(L_RIX) + buf * i32(128 * 4)
            wt0 = i32(L_WT) + buf * i32(128 * 4)
        # (small-M: G2 shares of GPC groups: a whole number of ring steps each)
        NSK2 = _nsk2_for(NKS, GPC if kc.SM else G2, nsk // 2 if W8 else nsk)
        GPI = (NSK2 * NKS // math.gcd(NSK2, NKS)) // NKS
        assert G2 % GPI == 0
        WAIT_B2 = (NSK2 - 1) * OPSW
        LAG = max(1, ceildiv(NSK2, GPC * NKS))
        RLAG = GPC * NKS * OPSW
        lane = tid % i32(64)
        w = uni(tid // i32(64))
        e = uni(expert)
        ks0 = uni(ks0)
        if const_expr(W8):
            rw = rsrc(a["w2h"])
            rws = rsrc(a["w2hs"])
            es = i32(0)  # (its scale rows from 0)
        else:
            rw = rsrc(fx.Int64(a["w2"]) + fx.Int64(e) * fx.Int64(H * (I // 2)))
            rws = rsrc(a["w2s"])
            es = e
        routes_bytes = route_region_bytes(a["ttot"])
        pidx = uni(pidx)
        dst = (pidx == i32(0)).select(
            fx.Int64(a["routes"]),
            fx.Int64(a["proutes"]) + fx.Int64(pidx - i32(1)) * fx.Int64(routes_bytes),
        )
        lop = None
        if const_expr(APL > 1 and not RAW):  # (a16: the lo plane after the hi one)
            lop = (pidx == i32(0)).select(routes_bytes, i32(NPC - 1) * routes_bytes)
            r_routes = rsrc(dst, lop + routes_bytes)
            oob = lop + routes_bytes
        else:
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

        def issue(q, slot_idx, odd=False):
            if const_expr(XPF):
                if q >= g_hi * i32(NKS):
                    issue_x(q, slot_idx)
                else:
                    issue_n(q, slot_idx)
            else:
                issue_n(q, slot_idx, odd)

        def wait_b2(k):
            # VMEM ops issued after step k's (sdd: 4 for the odd steps)
            if not sdd:
                return WAIT_B2
            return sum(OPSW - 2 * ((k + d) % 2) for d in range(1, NSK2))

        def issue_n(q, slot_idx, odd=False):
            qc = fx.min(q, g_hi * i32(NKS) - i32(1))
            gi = qc // i32(NKS)
            k = ks0 + qc - gi * i32(NKS)
            n0 = (w + i32(NW) * gi) * i32(64)
            slot = ring + i32(slot_idx * SLOTW)
            for t in range_constexpr(4):
                if const_expr(W8):
                    for h in range_constexpr(2):
                        dma16(
                            slot + i32(t * 2048 + h * 1024),
                            rw,
                            lane * i32(16),
                            (n0 + i32(t * 16)) * i32(I)
                            + k * i32(2048)
                            + i32(h * 1024),
                            nt=bnt,
                        )
                else:
                    dma16(
                        slot + i32(t * 1024),
                        rw,
                        lane * i32(16),
                        _w2_off(n0, t, k),
                        nt=bnt,
                    )
            for p in range_constexpr(0 if (sdd and odd) else 2):
                rb = (es * i32(H) + n0 + i32(p * 32)) // i32(32)
                dma4(
                    slot + i32(BSO + p * 256),
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
                issue(g_lo * i32(NKS) + i32(qq), qq, qq % 2 == 1)
        if const_expr(pre is not None):
            pre()
        if const_expr(rows is None):  # (the routes gathered in pre)
            rows = fx.min(lds_ld_i32(L, L_CTL + C_CNT * 4), i32(TMAX_))

        def body(MTE):
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

            def groups(gi0, nj=GPI):
                for j in range_constexpr(nj):
                    gi = gi0 + i32(j)
                    acc = [zero] * (4 * MTE)
                    s0 = s1 = None
                    for k in range_constexpr(NKS):
                        sidx = (j * NKS + k) % NSK2
                        if const_expr(k < NSK2 and NKS >= NSK2):
                            if const_expr(j > 0):  # noqa: SIM114
                                wait_vm(wait_b2(k) + 2 * MTE)
                            elif gi > g_lo:
                                wait_vm(wait_b2(k) + 2 * MTE)
                            else:
                                wait_vm(wait_b2(k))
                        else:
                            wait_vm(wait_b2(k))
                        slot = ring + i32(sidx * SLOTW)
                        b = _b_read(slot, lane, W8)
                        if not (sdd and k % 2 == 1):
                            s0 = lds_ld_i32(slot, i32(BSO) + lane * i32(4))
                            s1 = lds_ld_i32(slot, i32(BSO + 256) + lane * i32(4))
                        sh0 = ((ks0 + i32(k)) & i32(1)) * i32(16)
                        sb = [
                            (s0 >> sh0) & i32(0xFF),
                            (s0 >> (sh0 + i32(8))) & i32(0xFF),
                            (s1 >> sh0) & i32(0xFF),
                            (s1 >> (sh0 + i32(8))) & i32(0xFF),
                        ]
                        af_l, sa_l = [], []
                        for pm in range_constexpr(APL * MTE):  # (a16: hi, then lo)
                            pl, m = pm // MTE, pm % MTE
                            row = i32(m * 16) + lane % i32(16)
                            ib = ib0 + row * i32(SI_STRIDE) + i32(pl * I)
                            if const_expr(a8):
                                kb = ib + (ka + i32(k)) * i32(128) + q4 * i32(16)
                                af_l.append(
                                    cat8(
                                        lds_ld(L, kb, V4I, 16),
                                        lds_ld(L, kb + i32(64), V4I, 16),
                                    )
                                )
                            else:
                                af_l.append(
                                    lds_ld(
                                        L,
                                        ib + (ka + i32(k)) * i32(64) + q4 * i32(16),
                                        V4I,
                                        16,
                                    )
                                )
                            sa_l.append(
                                fx.Int32(
                                    lds_ld(
                                        L,
                                        is0
                                        + row * i32(SIB)
                                        + i32(pl * (I // 32))
                                        + (ka + i32(k)) * i32(4)
                                        + q4,
                                        T.i8,
                                        1,
                                    )
                                )
                                & i32(0xFF)
                            )
                        rocdl.sched_barrier(0)
                        rocdl.s_setprio(1)
                        if const_expr(W8):  # (as _g1_mfma: one tile's fragment live)
                            for t in range_constexpr(4):
                                bt = b[t]()
                                for pm in range_constexpr(APL * MTE):
                                    m = pm % MTE
                                    acc[m * 4 + t] = mfma(
                                        bt,
                                        af_l[pm],
                                        acc[m * 4 + t],
                                        sb[t],
                                        sa_l[pm],
                                        a8,
                                        a8=True,
                                    )
                        else:
                            for pm in range_constexpr(APL * MTE):
                                m = pm % MTE
                                for t in range_constexpr(4):
                                    acc[m * 4 + t] = mfma(
                                        b[t],
                                        af_l[pm],
                                        acc[m * 4 + t],
                                        sb[t],
                                        sa_l[pm],
                                        a8,
                                    )
                        rocdl.s_setprio(0)
                        wait_lgkm0()
                        rocdl.sched_barrier(0)
                        issue(gi * i32(NKS) + i32(k + NSK2), sidx, (k + NSK2) % 2 == 1)
                        rocdl.sched_barrier(0)
                    n0 = (w + i32(NW) * gi) * i32(64)
                    ccol = (q4 & i32(1)) * i32(16) + (q4 >> i32(1)) * i32(8)
                    for m in range_constexpr(MTE):
                        ok, wt, rix = row_meta[m]
                        if const_expr(route_fp8 and not RAW):
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
                                lop,
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
                                bst(
                                    ov,
                                    r_routes,
                                    ok.select(voff, oob),
                                    0,
                                    AUX_SYS if RAW else AUX_RT,
                                )
                    if const_expr(not DLL):
                        maybe_report(L, lane, w, gi, signal, LAG * RLAG, LAG)

            for gi0_ in range(g_lo, g_hi, i32(GPI)):
                if const_expr(progress is not None):  # noqa: SIM102
                    if i32(gi0_) + i32(GPI) >= g_hi:
                        progress()
                if const_expr(GPI > 1 and odd):
                    # (an odd group range: its last group alone, the ring prefetches past it
                    # clamped to its last k-step)
                    if i32(gi0_) + i32(1) < g_hi:
                        groups(i32(gi0_))
                    else:
                        groups(i32(gi0_), 1)
                else:
                    groups(i32(gi0_))

        if const_expr(SMSKIP and MTE > 1):
            # (small-M: a unit of at most 16 rows (most experts) one row tile; the
            # wide tile for the hot experts' few big chunks)
            if uni(rows) <= i32(16):
                body(1)
            else:
                body(MTE)
        else:
            body(MTE)
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


def _i8x4_fast(f, inv):
    """4 f32 * inv (|.| <= 127) -> int8 x 4 in an i32: + 1.5 * 2^23 rounds to the nearest
    even integer in the low mantissa byte (two's complement), v_perm gathers the bytes"""
    mg = fx.Float32(12582912.0)
    v = [fx.Float32(x * inv + mg).bitcast(fx.Int32) for x in f]
    lo = fx.Int32(rocdl.perm_b32(v[1], v[0], i32(0x0C0C0400)))
    hi = fx.Int32(rocdl.perm_b32(v[3], v[2], i32(0x0C0C0400)))
    return fx.Int32(rocdl.perm_b32(hi, lo, i32(0x05040100)))


def build_bigm_unit(
    H, I, TOPK, RG, NAB, BST, BS2, act, swiglu_limit, R8, situ=(1.0, 1.0), raux=0, mtc=0,
    i8r=False, sh8=False,
):
    """The fused unit device function: unit_body(L, tid, e, r0, rows, rr, rwt, rx, rxs, w1,
    w2, rws1, rws2, r_routes, oob): rows rr / rwt[r0 : r0 + rows] (route index, weight) of expert
    e -> route rows wt * GEMM2(act(GEMM1(A[route // TOPK]))) stored at route * H (bf16).
    R8: route rows as MXFP8 (e4m3 + one e8m0 scale per 32 columns): codes [rows][H] at r_routes,
    scales [rows][H / 32] after them (at oob / 2: oob = the route count * H * 2).
    raux: the route rows' store cache policy (AUX_SYS: write-through).
    sh8: the shared expert's MXFP8 GEMMs too (unit_body's sh = (w1, w2, rws1, rws2, its
    expert index): its own weights, B fragments of 8 dwords, half the B ring depths).
    Returns (LDS layout, unit_body)."""
    assert RG % 16 == 0 and RG <= 128 and H % 256 == 0 and I % 128 == 0
    # 16-row tiles computed (every wave computes every row of the unit): mtc (< RG / 16) of
    # them for a unit of at most 16 mtc rows, its A rows still loaded as RG
    MTE = mtc or RG // 16
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
    # i8r: int8 route rows + one f32 scale per 64 columns (codes [routes][H], the scales
    # [routes][H / 64] after them at oob / 2), quantized in registers, staged in LDS (the
    # codes per group, every row's scales per unit: SST, after the waves' staging tiles in
    # the A ring) and stored as whole 64 B / 224 B row segments
    SST = lay["LA"] + NW * RG * 64
    assert not i8r or SST + RG * (H // 16) <= lay["LA"] + NAB * RG * 64 * KCK
    NST2 = MTE if i8r else 2 * MTE  # route stores per wave per group
    # steady-state GEMM1 issue order per chunk c: B(c + BST), then A(c + NAB - 1). By the
    # count alone A(c) has landed once (NA + NB) * (NAB - 2) newer ops remain, but its LDS
    # writes (direct-to-LDS loads) can still be in flight then: one chunk's ops more are
    # awaited (else a unit now and then reads stale A rows of a chunk, small units mostly)
    WAIT1 = max(min(NA + (NA + NB) * (BST - 1), (NA + NB) * (NAB - 3)), 0)
    assert WAIT1 <= 63

    def _wait2(kk, BS2=BS2, NB2=NB2):
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
        return _activation_mul_batch(
            gs,
            us,
            act=act,
            swiglu_limit=swiglu_limit,
            situ_beta=situ[0],
            situ_linear_beta=situ[1],
        )

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

    def _mk_g1(W8):
        # (W8: B fragments of 8 dwords: a B ring of one chunk, its ops twice the tiles')
        BST_ = 1 if W8 else BST
        NB_ = (8 if W8 else 4) * KCK + 2  # B ops per chunk
        NBE = 4 * KCK + 2  # B ring entries per chunk (W8: 8-dword fragments)
        WAIT1_ = max(min(NA + (NA + NB_) * (BST_ - 1), (NA + NB_) * (NAB - 3)), 0)
        assert WAIT1_ <= 63

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
            if const_expr(W8):  # (the shared expert's own tensors: scale rows from 0)
                rw1 = rsrc(fx.Int64(w1))
                es = i32(0)
            else:
                rw1 = rsrc(fx.Int64(w1) + fx.Int64(e) * fx.Int64(2 * I * KH))
                es = e

            def issue_b(p, c):
                icol0 = p * i32(NW * 32) + w * i32(32)
                KHB = H if W8 else KH  # (W8: row bytes of the MXFP8 w1)
                vg0 = icol0 * i32(KHB) + lane * i32(16)
                vs = (vg0, vg0 + i32(16 * KHB), vg0 + i32(I * KHB), vg0 + i32((I + 16) * KHB))
                b = []
                for k in range_constexpr(KCK):
                    for t in range_constexpr(4):
                        if const_expr(W8):
                            ko = (c * i32(KCK) + i32(k)) * i32(2048)
                            b.append(
                                cat8(
                                    bld(rw1, vs[t], ko, V4I),
                                    bld(rw1, vs[t], ko + i32(1024), V4I),
                                )
                            )
                        else:
                            b.append(
                                fx.Vector(
                                    bld(rw1, vs[t], (c * i32(KCK) + i32(k)) * i32(1024), V4I)
                                )
                            )
                # one scale word = 2 k-steps
                cw = (c * i32(KCK)) // i32(2)
                sg = ((es * i32(2 * I) + icol0) // i32(32)) * i32(CH1 * 256) + cw * i32(256)
                su = ((es * i32(2 * I) + i32(I) + icol0) // i32(32)) * i32(CH1 * 256)
                su = su + cw * i32(256)
                vso = lane * i32(4)
                b.append(fx.Int32(bld(rws1, vso, sg, T.i32)))
                b.append(fx.Int32(bld(rws1, vso, su, T.i32)))
                return b

            def g1_step(acc, bq, c, k, cur, nbuf, nk):
                # k-step k of chunk c (compile time) from the fragments cur = (A tiles, scales)
                # read during the previous step; the next step's (buffer nbuf, k-step nk) read
                # here, one tile's after each tile's MFMAs (the LDS reads under the MFMAs)
                sgw, suw = bq[4 * KCK], bq[4 * KCK + 1]
                kp = (c * KCK + k) % 2  # the global k-step's half of the scale word
                # (the scale byte picked by the MFMA's opsel: no shift / mask)
                sc = [(x, kp * 2 + h) for x in (sgw, suw) for h in (0, 1)]
                afs, sas = cur
                nafs, nsas = [], []
                for m in range_constexpr(MTE):
                    for t in range_constexpr(4):
                        j = (t // 2) * 2 * MTE + m * 2 + t % 2
                        acc[j] = mfma(
                            bq[k * 4 + t],
                            afs[m],
                            acc[j],
                            sc[t][0],
                            sas[m],
                            opa=sc[t][1],
                            a8=W8,
                        )
                    nafs.append(a_frag_m(nbuf, m, nk))
                    nsas.append(a_scale_m(nbuf, m, nk))
                    rocdl.sched_mfma(4)
                    rocdl.sched_dsrd(2)
                return acc, (nafs, nsas)

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
                            acc[j] = mfma(
                                bq[k * 4 + t], afs[m], acc[j], sc[t], sas[m], a8=W8
                            )
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

            # ---- GEMM1: PASSES x KCH chunks, A ring NAB, B ring BST_ (carried over passes) ----
            # prologue = the steady-state issue order of the BST_ virtual steps before c = 0
            # (steps before the first real op of a kind issue dummy loads of the same count)
            bpre = []
            for i in range_constexpr(-max(BST_, NAB - 1), 0):
                if const_expr(i + BST_ >= 0):
                    bpre.append(issue_b(i32(0), i32(i + BST_)))
                else:
                    issue_b(i32(0), i32(0))
                if const_expr(i + NAB - 1 >= 0):
                    issue_a(i32(i + NAB - 1), (i + NAB - 1) % NAB)
                else:
                    for _ in range_constexpr(NA):
                        bld(rx, lane * i32(16), 0, V4I)
            # software pipelined: each k-step's MFMAs interleaved with the next k-step's A tile /
            # scale reads; a chunk's refill and the barrier of the next chunk's A (every wave's DMA
            # landed) before its last k-step, so that step reads the next chunk's first tiles (the
            # next pass's at the end of a pass: the A ring runs on over the passes)
            rocdl.sched_barrier(0)
            wait_vm(WAIT1_)
            gpu.barrier()
            rocdl.sched_barrier(0)
            f0 = (
                [a_frag_m(0, m, 0) for m in range(MTE)],
                [a_scale_m(0, m, 0) for m in range(MTE)],
            )
            NFR = 2 * MTE
            init = [zero] * (4 * MTE) + [v for b in bpre for v in b] + f0[0] + f0[1]
            phi = i32(PASSES)
            for p_, stt in range(i32(0), phi, i32(1), init=init):
                p = i32(p_)
                acc = [zero] * (4 * MTE)
                ring = [
                    list(stt[4 * MTE + s * NBE : 4 * MTE + (s + 1) * NBE])
                    for s in range(BST_)
                ]
                fo = 4 * MTE + BST_ * NBE
                cur = (list(stt[fo : fo + MTE]), list(stt[fo + MTE : fo + NFR]))
                for c in range_constexpr(KCH):
                    bq = ring[c % BST_]
                    for k in range_constexpr(KCK):
                        if const_expr(k == KCK - 1):
                            # refill: B of step c + BST_ (next pass past the end; the last pass
                            # reloads its own chunks: harmless, keeps the counts static)
                            cn = c + BST_
                            pn = p + i32(cn // KCH)
                            pn = (pn < phi).select(pn, phi - i32(1))
                            ring[cn % BST_] = issue_b(pn, i32(cn % KCH))
                            issue_a(i32((c + NAB - 1) % KCH), (c + NAB - 1) % NAB)
                            # the next chunk's A in every wave's LDS (nothing crosses the barrier:
                            # else the scheduler may hoist its reads above s_barrier: a race)
                            rocdl.sched_barrier(0)
                            wait_vm(WAIT1_)
                            gpu.barrier()
                            rocdl.sched_barrier(0)
                            nbuf, nk = (c + 1) % NAB, 0
                        else:
                            nbuf, nk = c % NAB, k + 1
                        acc, cur = g1_step(acc, bq, c, k, cur, nbuf, nk)
                act_store(acc, p)
                # (the yield's value binding is FlyDSL's loop-carried form)
                res = yield (  # noqa: F841
                    [zero] * (4 * MTE)
                    + [v for s in range(BST_) for v in ring[s]]
                    + cur[0]
                    + cur[1]
                )
            wait_vm(0)
            wait_lgkm0()  # the intermediate's LDS writes land before other waves read them
            gpu.barrier()
        return part_g1

    part_g1 = _mk_g1(False)
    part_g1_w8 = _mk_g1(True) if sh8 else None

    def _mk_g2(W8):
        # (W8: B2 fragments of 8 dwords: half the B2 ring depth, twice the tile ops)
        BS2_ = (
            max(d for d in range(1, max(1, BS2 // 2) + 1) if KS2 % d == 0) if W8 else BS2
        )
        NB2_ = 10 if W8 else NB2
        WAIT2_ = [_wait2(kk, BS2_, NB2_) for kk in range(KS2)]

        @flyc.jit
        def part_g2(L, tid, e, rows, w2, rws2, r_routes, oob):
            lane = tid % i32(64)
            w = rocdl.readfirstlane(T.i32, tid // i32(64))  # the wave: its column group
            q4 = lane // i32(16)
            zero = fx.Vector.filled(4, 0.0, fx.Float32)
            # ---- GEMM2 from the LDS intermediate: NG2 groups x KS2 k-steps, B ring BS2_ ----
            if const_expr(W8):
                rw2 = rsrc(fx.Int64(w2))
                es = i32(0)
            else:
                rw2 = rsrc(fx.Int64(w2) + fx.Int64(e) * fx.Int64(H * (I // 2)))
                es = e

            def issue_b2(gi, kk):  # one 64-column group
                n0 = (w + i32(NW) * gi) * i32(64)
                if const_expr(W8):
                    b = [
                        cat8(
                            bld(
                                rw2,
                                lane * i32(16),
                                (n0 + i32(t * 16)) * i32(I) + kk * i32(2048),
                                V4I,
                            ),
                            bld(
                                rw2,
                                lane * i32(16),
                                (n0 + i32(t * 16)) * i32(I) + kk * i32(2048) + i32(1024),
                                V4I,
                            ),
                        )
                        for t in range(4)
                    ]
                else:
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
                    rb = (es * i32(H) + n0 + i32(pp * 32)) // i32(32)
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
            # bf16 route rows through a per-wave LDS staging tile ([RG][64] bf16 in the A ring,
            # idle in GEMM2): each store then writes 8 rows x a whole 128 B row segment (16 rows x
            # 64 B halves straight from the MFMA layout cost ~20% of GEMM2 at Kimi K3.1)
            STG = not R8 and NAB * RBA * RG >= NW * RG * 128
            if const_expr(STG):
                stg = i32(LA) + w * i32(RG * (64 if i8r else 128))
                row_meta2 = []
                for j in range_constexpr(MTE * 2):
                    R = i32(j * 8) + lane // i32(8)
                    ok = R < rows
                    row_meta2.append(
                        (ok, lds_ld_i32(L, i32(LRIX) + ok.select(R, i32(0)) * i32(4)))
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
                            raux,
                        )
                        soff = oob // i32(2) + rix * i32(H // 32) + n0 // i32(32)
                        bst(sc, r_routes, (ok & (q4 == i32(0))).select(soff, oob), 0, raux)
                elif const_expr(STG and i8r):
                    # int8 route rows quantized in registers (a lane: 4 columns of a row in each
                    # of the 4 tiles; the row's 64 columns on the 4 lanes l % 16), staged as
                    # [row][4 x 16 B] (chunk t at t ^ (row & 3)), then 4 lanes per row stored
                    for m in range_constexpr(MTE):
                        ok, wt, rix = row_meta[m]
                        row = i32(m * 16) + lane % i32(16)
                        vals = []
                        for t in range_constexpr(4):
                            av = fx.Vector(acc[m * 4 + t])
                            vals += [fx.Float32(av[i_]) * wt for i_ in range(4)]
                        am = amax(vals)
                        am = am.maximumf(am.shuffle_xor(i32(16), i32(64)))
                        am = am.maximumf(am.shuffle_xor(i32(32), i32(64)))
                        sc = am.maximumf(fx.Float32(1e-30)) / fx.Float32(127.0)
                        inv = fx.Float32(1.0) / sc
                        for t in range_constexpr(4):
                            off = row * i32(64) + (i32(t) ^ (row & i32(3))) * i32(16)
                            pk = _i8x4_fast(vals[4 * t : 4 * t + 4], inv)
                            lds_st(L, stg + off + q4 * i32(4), pk, align=4)
                        # the scale to the unit's LDS scale tile ([RG rows][H / 64]): written out
                        # once per unit, a row's scales contiguous
                        if q4 == i32(0):
                            lds_st(
                                L,
                                i32(SST) + row * i32(H // 16) + (n0 // i32(64)) * i32(4),
                                sc.bitcast(fx.Int32),
                                align=4,
                            )
                    c = lane % i32(4)
                    for it in range_constexpr(MTE):
                        row = i32(it * 16) + lane // i32(4)
                        ok = row < rows
                        rix = lds_ld_i32(L, i32(LRIX) + ok.select(row, i32(0)) * i32(4))
                        v = lds_ld(
                            L, stg + row * i32(64) + (c ^ (row & i32(3))) * i32(16), V4I, 16
                        )
                        bst(
                            v,
                            r_routes,
                            ok.select(rix * i32(H) + n0 + c * i32(16), oob),
                            0,
                            raux,
                        )
                elif const_expr(STG):
                    # [row][8 x 16 B column chunks], chunk c at (c ^ (row & 7)) (no bank conflicts)
                    for m in range_constexpr(MTE):
                        ok, wt, rix = row_meta[m]
                        row = i32(m * 16) + lane % i32(16)
                        for t in range_constexpr(4):
                            av = fx.Vector(acc[m * 4 + t])
                            v2 = fx.Vector.from_elements(
                                [
                                    pack_bf16x2(
                                        fx.Float32(av[2 * h]) * wt,
                                        fx.Float32(av[2 * h + 1]) * wt,
                                    )
                                    for h in range(2)
                                ],
                                fx.Int32,
                            )
                            c = i32(t * 2) + q4 // i32(2)
                            off = row * i32(128) + (c ^ (row & i32(7))) * i32(16)
                            lds_st(L, stg + off + (q4 % i32(2)) * i32(8), v2, align=8)
                    c = lane % i32(8)
                    for j in range_constexpr(MTE * 2):
                        ok, rix = row_meta2[j]
                        row = i32(j * 8) + lane // i32(8)
                        v = lds_ld(
                            L,
                            stg + row * i32(128) + (c ^ (row & i32(7))) * i32(16),
                            V4I,
                            16,
                        )
                        bst(
                            v,
                            r_routes,
                            ok.select((rix * i32(H) + n0 + c * i32(8)) * i32(2), oob),
                            0,
                            raux,
                        )
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
                                raux,
                            )

            # prologue = the steady issue order of the BS2_ steps before (0, 0): their B2 loads,
            # then (the previous group's last step) NST2 route stores, here dropped (OOB)
            b2pre = [issue_b2(i32(s_ // KS2), i32(s_ % KS2)) for s_ in range(BS2_)]
            for _ in range_constexpr(NST2):
                bst(fx.Vector.filled(4, 0, fx.Int32), r_routes, oob, 0)
            ghi = i32(NG2)
            init2 = [v for b in b2pre for v in b]
            for gi_, st2 in range(i32(0), ghi, i32(1), init=init2):
                gi = i32(gi_)
                ring2 = [list(st2[s * NB2 : (s + 1) * NB2]) for s in range(BS2_)]
                acc = [zero] * (4 * MTE)
                for kk in range_constexpr(KS2):
                    wait_vm(WAIT2_[kk])
                    bq = ring2[kk % BS2_]
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
                            acc[j] = mfma(bq[t], afs[m], acc[j], sb[t], sas[m], a8=W8)
                    sn = kk + BS2_
                    gn = gi + i32(sn // KS2)
                    gn = (gn < ghi).select(gn, ghi - i32(1))
                    ring2[kk % BS2_] = issue_b2(gn, i32(sn % KS2))
                epi2(acc, gi)
                res2 = yield [v for s in range(BS2_) for v in ring2[s]]  # noqa: F841
            if const_expr(i8r):
                gpu.barrier()
                W4 = H // 256  # 16 B words of a row's scales
                for it in range_constexpr(ceildiv(RG * W4, NW * 64)):
                    idx = tid + i32(it * NW * 64)
                    r_ = idx // i32(W4)
                    ok = (r_ < rows) & (idx < i32(RG * W4))
                    rix = lds_ld_i32(L, i32(LRIX) + ok.select(r_, i32(0)) * i32(4))
                    v = lds_ld(
                        L, i32(SST) + ok.select(idx, i32(0)) * i32(16), V4I, 16
                    )
                    bst(
                        v,
                        r_routes,
                        ok.select(
                            oob.shrui(i32(1))
                            + rix * i32(H // 16)
                            + (idx - r_ * i32(W4)) * i32(16),
                            oob,
                        ),
                        0,
                        raux,
                    )
            wait_vm(0)
        return part_g2

    part_g2 = _mk_g2(False)
    part_g2_w8 = _mk_g2(True) if sh8 else None

    @flyc.jit
    def unit_body(
        L, tid, e, r0, rows, rr, rwt, rx, rxs, w1, w2, rws1, rws2, r_routes, oob, sh=None
    ):
        part_fill(L, tid, r0, rows, rr, rwt)
        if const_expr(sh is not None):
            w1h, w2h, rws1h, rws2h, she = sh
            if e == she:  # (the MXFP8 shared expert)
                part_g1_w8(L, tid, e, rx, rxs, w1h, rws1h)
                part_g2_w8(L, tid, e, rows, w2h, rws2h, r_routes, oob)
            else:
                part_g1(L, tid, e, rx, rxs, w1, rws1)
                part_g2(L, tid, e, rows, w2, rws2, r_routes, oob)
        else:
            part_g1(L, tid, e, rx, rxs, w1, rws1)
            part_g2(L, tid, e, rows, w2, rws2, r_routes, oob)

    return lay, unit_body
