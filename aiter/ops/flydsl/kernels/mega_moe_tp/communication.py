# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-instance device helpers (ctrl / peer addressing, LL packets, barriers, error
reports, route gather) and the communication: the input all-gather, and the
ReduceScatter of the GEMM2 partials (ar: plus the all-gather of the sums) with its
final sums, LL packets, comm wave roles and the per-launch init / finish."""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ..kernels_common import ceildiv
from ..mxfp4_gemm_common import _activation_mul_batch, _e8m0_from_amax
from .common import (
    ACQ,
    ACQ_REL,
    AUX_SC1,
    AUX_SYS,
    DEADLINE,
    MAX_TP,
    REL,
    V2I,
    V4I,
    _ctpop,
    alive,
    amax,
    before,
    bf16x8_to_f32,
    bld,
    bst,
    cat8,
    e8_scale,
    fp4_pack,
    fp8x4_pack,
    fp8x4_unpack,
    fp8x8_decode,
    g_add_agent,
    g_ld_i32,
    g_ld_rel,
    g_ld_sys,
    g_or_agent,
    g_st_sys,
    i32,
    lds_atomic_add,
    lds_atomic_or,
    lds_cas,
    lds_ld,
    lds_ld_acq,
    lds_ld_i32,
    lds_st,
    lds_st_rel,
    mxfp8x8,
    now,
    pack_bf16x8,
    poll_sys_ge,
    rsrc,
    scales4,
    spin0,
    sum_live,
    traced,
    uni,
    wait_lgkm0,
    wait_vm,
    wave_rank,
    wave_red,
)
from .mega_moe_tp_config import (
    C_AGFREE,
    C_ARDY,
    C_BARCNT,
    C_BARGEN,
    C_CDONE,
    C_CLAIM,
    C_CNT,
    C_DONE,
    C_DYNP,
    C_EPOCH,
    C_FBITS,
    C_FRDY,
    C_GEXP,
    C_LRED,
    C_MBOX,
    C_NSIG,
    C_PBITS,
    C_PRDY,
    C_PULL,
    C_RLAND,
    C_UL,
    C_YAGM,
    CTRL_AGG,
    CTRL_AGX,
    CTRL_CLM,
    CTRL_EPB,
    CTRL_ERR,
    CTRL_LRDY,
    CTRL_SC,
    CTRL_XES,
    ERR_CHUNK,
    ERR_COMM,
    ERR_FLAG,
    ERR_META,
    ERR_YAG,
    FLAG_AGM,
    FLAG_AGQ,
    FLAG_RDY,
    FLAG_YAG,
    LRDY_STRIDE,
    N_XCD,
    NCTA_MAX,
    NT,
    NTT,
    NW,
    POLL_SLEEP,
    SC_ALL,
    SC_LINES,
    SC_PUSH,
    KernelCtx,
)


def build_communication(kc: KernelCtx) -> dict:
    """The communication device functions of one instance (added to kc)."""
    ACB, AG8, ARLL, CW, DLL, E, H, I = kc.get("ACB AG8 ARLL CW DLL E H I")
    L_CTL, L_DCNT, L_DPRE, L_DYN = kc.get("L_CTL L_DCNT L_DPRE L_DYN")
    L_INTER, L_INTERS, L_RIX, L_WT, MLL = kc.get("L_INTER L_INTERS L_RIX L_WT MLL")
    MT, NBW, NCH, NCK, NMETA, NPC, NV = kc.get("MT NBW NCH NCK NMETA NPC NV")
    RG, SCAN_G, SCAN_IT, SI_STRIDE = kc.get("RG SCAN_G SCAN_IT SI_STRIDE")
    TMAX, TN, TOPK, TPC, VPL, XB, XLPR = kc.get("TMAX TN TOPK TPC VPL XB XLPR")
    a8, act, agr, ar, comm_bf16 = kc.get("a8 act agr ar comm_bf16")
    const_expr, lb, ll_rs, route_fp8 = kc.get("const_expr lb ll_rs route_fp8")
    situ_beta, situ_linear_beta = kc.get("situ_beta situ_linear_beta")
    swiglu_limit = kc.swiglu_limit

    def ctrl_at(a, idx):
        return fx.Int64(a["ctrl"]) + fx.Int64(i32(idx) * i32(4))

    def lrdy_at(a, c):
        return ctrl_at(a, i32(CTRL_LRDY) + c * i32(LRDY_STRIDE))

    def peer_sel(a, p):
        v = fx.Int64(a["peer"][0])
        for j in range_constexpr(1, MAX_TP):
            v = (p == i32(j)).select(fx.Int64(a["peer"][j]), v)
        return v

    def peer_rs(a, p, key, nbytes=None):
        return rsrc(fx.Int64(a["peer"][p]) + fx.Int64(a[key]), nbytes)

    def own_rs(a, key, nbytes=None):
        return rsrc(a["mine"] + fx.Int64(a[key]), nbytes)

    def route_region_bytes(ttot):
        return ttot * i32(TOPK * (H + H // 32 if route_fp8 and not DLL else 2 * H))

    def ll_pkt(a, d0, d1, e8):
        tag = ((a["epoch"] & i32(0xFFFFFF)) << i32(8)) | e8
        return fx.Vector.from_elements([d0, a["epoch"], d1, tag], fx.Int32)

    def ll_ok(a, q):
        return (fx.Int32(q[1]) == a["epoch"]) & (
            fx.Int32(q[3]).shrui(i32(8)) == (a["epoch"] & i32(0xFFFFFF))
        )

    def ll_vals(q):
        sc = e8_scale(q[3])
        return fp8x4_unpack(fx.Int32(q[0]), sc) + fp8x4_unpack(fx.Int32(q[2]), sc)

    def ll_pending(a, lane, pkts):
        bad = i32(0)
        for q, live in pkts:
            bad = fx.max(
                bad, (live & (ll_ok(a, q) == fx.Boolean(False))).select(i32(1), i32(0))
            )
        return wave_red(bad, lane, fx.max)

    def route_load(rs, region, ttot, ridx, col, take=None):
        if route_fp8:
            doff = region + ridx * i32(H) + col
            soff = region + ttot * i32(TOPK * H) + ridx * i32(H // 32) + col // i32(32)
            if take is not None:
                end = region + route_region_bytes(ttot) * i32(NPC)
                doff, soff = take.select(doff, end), take.select(soff, end)
            return (bld(rs, doff, 0, V2I, AUX_SC1), bld(rs, soff, 0, T.i8, AUX_SC1))
        off = region + (ridx * i32(H) + col) * i32(2)
        if take is not None:
            off = take.select(off, region + route_region_bytes(ttot) * i32(NPC))
        return bld(rs, off, 0, V4I, AUX_SC1)

    def route_decode(ld):
        return fp8x8_decode(ld) if route_fp8 else bf16x8_to_f32(ld)

    def a_frag(abuf, row, k, q4):
        if const_expr(a8):
            lo = (i32(k * 8) + q4) ^ (row & i32(7))
            hi = (i32(k * 8 + 4) + q4) ^ (row & i32(7))
            return cat8(
                lds_ld(abuf, row * i32(ACB) + lo * i32(16), V4I, 16),
                lds_ld(abuf, row * i32(ACB) + hi * i32(16), V4I, 16),
            )
        col = (i32(k * 4) + q4) ^ (row & i32(7))
        return lds_ld(abuf, row * i32(ACB) + col * i32(16), V4I, 16)

    def act_batch(gs, us):
        return _activation_mul_batch(
            gs,
            us,
            act=act,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            swiglu_limit=swiglu_limit,
        )

    @traced
    def cbar(L, tid):
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

    def report(a, code):
        g_or_agent(ctrl_at(a, CTRL_ERR), code)

    @traced
    def _report_if(a, bad, code):
        if bad:
            report(a, code)

    @traced
    def poll_zero(fn, a=None, lane=None, err=0):
        pend = fn()
        t0 = now()
        while (pend != i32(0)) & alive(t0):
            rocdl.s_sleep(1)
            pend = fn()
        if const_expr(err != 0):
            _report_if(a, (pend != i32(0)) & (lane == i32(0)), err)

    @traced
    def spin_sys_ge(addr, target, a=None):
        cur = poll_sys_ge(addr, target)
        if const_expr(a is not None):
            _report_if(a, before(cur, target), ERR_FLAG)

    def masked(e):
        return fx.Int32(e).bitcast(fx.Uint32) >= fx.Uint32(E)

    @traced
    def gather_routes_chunk(L, tid, a, expert, cc, ce):
        key = expert + i32(1) + (cc << i32(12)) + (ce << i32(20))
        if lds_ld_i32(L, L_CTL + C_GEXP * 4) != key:
            if ce == i32(1):
                _gather_scan(L, tid, a, expert, key)
            else:
                inv = fx.Float32(1.0) / ce.to(fx.Float32)
                _gather_scan(L, tid, a, expert, key, (cc, ce, inv))
        return fx.min(lds_ld_i32(L, L_CTL + C_CNT * 4), i32(TMAX))

    def _in_chunk(idx, chunk):
        if chunk is None:
            return fx.Boolean(True)
        cc, ce, inv = chunk
        t = idx // i32(TOPK)
        q = ((t.to(fx.Float32) + fx.Float32(0.5)) * inv).to(fx.Int32)
        return (t - q * ce) == cc

    @traced
    def _gather_scan(L, tid, a, expert, key, chunk=None):
        ids_addr, tw_addr, ttot = a["ids"], a["tw"], a["ttot"]
        cbar(L, tid)
        if tid == i32(0):
            lds_st(L, L_CTL + C_CNT * 4, i32(0))
        cbar(L, tid)
        n = ttot * i32(TOPK)
        rid = rsrc(ids_addr, n * i32(16 if MLL else 4))
        if const_expr(MLL):
            for i_ in range(tid, n, i32(NT)):
                i = i32(i_)
                pk = fx.Vector(bld(rid, i * i32(16), 0, V4I, 0))
                hit = (fx.Int32(pk[0]) == expert) & _in_chunk(i, chunk)
                _gather_one(L, i, hit, fx.Int32(pk[2]))
        else:
            n4 = n // i32(4)
            rtw = rsrc(tw_addr, n * i32(4))
            for g0 in range_constexpr(0, SCAN_IT, SCAN_G):
                its = list(range(g0, min(g0 + SCAN_G, SCAN_IT)))
                vs, ws = [], []
                for it in its:
                    q = fx.min(tid + i32(it * NT), n4 - i32(1))
                    vs.append(fx.Vector(bld(rid, q * i32(16), 0, V4I, 0)))
                    ws.append(fx.Vector(bld(rtw, q * i32(16), 0, V4I, 0)))
                hits, run = [], i32(0)
                for x, it in enumerate(its):
                    q = tid + i32(it * NT)
                    for j in range_constexpr(4):
                        idx = q * i32(4) + i32(j)
                        hit = (
                            (q < n4)
                            & (fx.Int32(vs[x][j]) == expert)
                            & _in_chunk(idx, chunk)
                        )
                        pos, cnt = wave_rank(hit)
                        hits.append((hit, run + pos, idx, ws[x][j]))
                        run = run + cnt
                base = _wave_claim(L, tid, run)
                for hit, pos, idx, wv in hits:
                    slot = base + pos
                    if hit & (slot < i32(TMAX)):
                        lds_st(L, L_RIX + slot * i32(4), idx)
                        lds_st(L, L_WT + slot * i32(4), fx.Int32(wv))
            for idx_ in range(n4 * i32(4) + tid, n, i32(NT)):
                idx = i32(idx_)
                e = g_ld_i32(fx.Int64(ids_addr) + fx.Int64(idx) * fx.Int64(4))
                wv = g_ld_i32(fx.Int64(tw_addr) + fx.Int64(idx) * fx.Int64(4))
                _gather_one(L, idx, (e == expert) & _in_chunk(idx, chunk), wv)
        cbar(L, tid)
        if tid == i32(0):
            lds_st(L, L_CTL + C_GEXP * 4, key)
        cbar(L, tid)

    @traced
    def zero_masked(L, lane, a, w0, nw):
        n = a["ttot"] * i32(TOPK)
        nblk = i32(gpu.grid_dim.x)
        per = ceildiv(n, nblk)
        lo = i32(gpu.block_id("x")) * per
        hi = fx.min(lo + per, n)
        rid = rsrc(a["ids"], n * i32(16 if MLL else 4))
        if const_expr(MLL):
            spin0(L, lane, L_CTL + C_RLAND * 4, i32(1))
        for b_ in range(lo + i32(w0 * 64), hi, i32(nw * 64)):
            i = i32(b_) + lane
            off = fx.min(i, n - i32(1)) * i32(16 if MLL else 4)
            e = fx.Int32(bld(rid, off, 0, T.i32, 0))
            bal = fx.Int64(rocdl.ballot(T.i64, (i < hi) & masked(e)))
            if bal != fx.Int64(0):
                for j_ in range(i32(0), i32(64), i32(1)):
                    j = i32(j_)
                    if ((bal >> fx.Int64(j)) & fx.Int64(1)) != fx.Int64(0):
                        _zero_route(a, lane, i32(b_) + j)
                wait_vm(0)

    @traced
    def _zero_route(a, lane, ridx):
        rb = route_region_bytes(a["ttot"])
        z = fx.Vector.from_elements([i32(0)] * 4, fx.Int32)
        for p in range_constexpr(NPC):
            rs = rsrc(a["routes"], rb) if p == 0 else rsrc(a["proutes"], i32(p) * rb)
            base = i32(0) if p == 0 else i32(p - 1) * rb
            for q_ in range(lane, i32(H // 8), i32(64)):
                q = i32(q_)
                if const_expr(DLL):
                    v = ll_pkt(a, i32(0), i32(0), i32(0))
                    bst(v, rs, base + (ridx * i32(H) + q * i32(8)) * i32(2), 0, AUX_SC1)
                elif const_expr(route_fp8):
                    if q < i32(H // 16):
                        bst(z, rs, base + ridx * i32(H) + q * i32(16), 0, AUX_SC1)
                    if q < i32(H // 512):
                        soff = a["ttot"] * i32(TOPK * H) + ridx * i32(H // 32)
                        bst(z, rs, base + soff + q * i32(16), 0, AUX_SC1)
                else:
                    bst(z, rs, base + (ridx * i32(H) + q * i32(8)) * i32(2), 0, AUX_SC1)

    @traced
    def _wave_claim(L, tid, n):
        got = i32(0)
        if ((tid % i32(64)) == i32(0)) & (n > i32(0)):
            got = lds_atomic_add(L, L_CTL + C_CNT * 4, n)
        return uni(got)

    @traced
    def _gather_one(L, idx, hit, wv):
        if hit:
            slot = lds_atomic_add(L, L_CTL + C_CNT * 4, 1)
            if slot < i32(TMAX):
                lds_st(L, L_RIX + slot * i32(4), idx)
                lds_st(L, L_WT + slot * i32(4), wv)

    def act_quant_store(L, lane, w, accg, accu, nb, mte=None):
        q4 = lane // i32(16)
        for m in range_constexpr(MT if mte is None else mte):
            row = i32(m * 16) + lane % i32(16)
            gs, us = [], []
            for t in range_constexpr(2):
                ga, ua = fx.Vector(accg[m * 2 + t]), fx.Vector(accu[m * 2 + t])
                for v in range_constexpr(4):
                    gs.append(fx.Float32(ga[v]))
                    us.append(fx.Float32(ua[v]))
            xs = [
                fx.Float32(x).to(fx.BFloat16).to(fx.Float32) for x in act_batch(gs, us)
            ]
            am = amax(xs)
            am = am.maximumf(am.shuffle_xor(i32(16), i32(64)))
            am = am.maximumf(am.shuffle_xor(i32(32), i32(64)))
            e8, qs = _e8m0_from_amax(am, max_norm=448.0 if a8 else 6.0)
            for t in range_constexpr(2):
                col = nb * i32(128) + w * i32(32) + i32(t * 16) + q4 * i32(4)
                if const_expr(a8):
                    pk = fp8x4_pack(xs[t * 4 : t * 4 + 4], qs)
                    lds_st(L, L_INTER + row * i32(SI_STRIDE) + col, pk, align=4)
                else:
                    pk = fp4_pack(xs[t * 4 : t * 4 + 4], qs)
                    lds_st(
                        L,
                        L_INTER + row * i32(SI_STRIDE) + col // i32(2),
                        fx.Int16(pk & i32(0xFFFF)),
                        align=2,
                    )
            _q0_store(L, q4, row, nb, w, e8)

    @traced
    def _q0_store(L, q4, row, nb, w, e8):
        if q4 == i32(0):
            lds_st(
                L, L_INTERS + row * i32(I // 32) + nb * i32(4) + w, fx.Int8(e8), align=1
            )

    def claim_addr(a, bank_off):
        bank = (a["epoch"] + i32(bank_off)) & i32(1)
        return ctrl_at(a, i32(CTRL_CLM) + bank * i32(LRDY_STRIDE))

    @traced
    def col_claim(L, tid, a):
        cbar(L, tid)
        if tid == i32(0):
            v = g_add_agent(claim_addr(a, 0), 1)
            for w in range_constexpr(NW):
                lds_st(L, L_CTL + (C_CLAIM * 4) + i32(w * 4), v)
        cbar(L, tid)

    def push_units(ttot):
        nblk = i32(gpu.grid_dim.x)
        return fx.min(i32(2) * nblk, ttot)

    def push_first_unit(ns, cidx):
        nblk = i32(gpu.grid_dim.x)
        rot = (cidx * ns) % nblk
        return (i32(gpu.block_id("x")) - rot + nblk) % nblk

    def token_dst(a, t):
        owner = t // a["m"]
        r_dst = rsrc(peer_sel(a, owner) + fx.Int64(a["off_part"]))
        return r_dst, a["rank"] * a["mmax"] + (t - owner * a["m"])

    def part_offs(a, prow, col):
        soff = a["tp"] * a["mmax"] * i32(H) + prow * i32(H // 32) + col // i32(32)
        return prow * i32(H) + col, soff

    @traced
    def _push_store(a, r_dst, prow, c0, v, acc, ok):
        col = c0 + v * i32(8)
        if const_expr(comm_bf16):
            if ok:
                bst(pack_bf16x8(acc), r_dst, (prow * i32(H) + col) * i32(2), 0, AUX_SYS)
        else:
            _push_store_fp8(a, r_dst, prow, col, v, acc, ok)

    @traced
    def _push_store_fp8(a, r_dst, prow, col, v, acc, ok):
        d0, d1, e8i = mxfp8x8(acc)
        if const_expr(ll_rs):
            pkt = ll_pkt(a, d0, d1, e8i)
            if ok:
                bst(pkt, r_dst, (prow * i32(H) + col) * i32(2), 0, AUX_SYS)
        else:
            doff, soff = part_offs(a, prow, col)
            sc = scales4(e8i)
            if ok:
                bst(
                    fx.Vector.from_elements([d0, d1], fx.Int32), r_dst, doff, 0, AUX_SYS
                )
                if (v & i32(15)) == i32(0):
                    bst(sc, r_dst, soff, 0, AUX_SYS)

    def sc_addr(a, cidx, slot):
        return ctrl_at(
            a, i32(CTRL_SC) + (cidx * i32(SC_LINES) + slot) * i32(LRDY_STRIDE)
        )

    @traced
    def _push_done_all(a, epoch, cidx, target, n):
        cnt_addr = sc_addr(a, cidx, i32(SC_PUSH))
        if g_add_agent(cnt_addr, n) + n == target:
            g_st_sys(cnt_addr, i32(0))
            fo = fx.Int64((i32(FLAG_RDY) + cidx * i32(MAX_TP) + a["rank"]) * i32(4))
            for p in range_constexpr(MAX_TP):
                if i32(p) < a["tp"]:
                    g_st_sys(
                        fx.Int64(a["peer"][p]) + fx.Int64(a["off_flag"]) + fo, epoch
                    )

    def fin_split(a):
        m = fx.max(a["m"], i32(1))
        nblk = i32(gpu.grid_dim.x)
        return (m < nblk).select(fx.min(i32(NV), nblk // m), i32(1))

    def fin_key(a):
        bid = i32(gpu.block_id("x"))
        return (fin_split(a) > i32(1)).select(bid % fx.max(a["m"], i32(1)), bid)

    def fin_owned(a):
        bid = i32(gpu.block_id("x"))
        m = fx.max(a["m"], i32(1))
        S = fin_split(a)
        sidx = bid // m
        own = i32(0)
        for c in range_constexpr(NV):
            own = own | ((i32(c) % S) == sidx).select(i32(1 << c), i32(0))
        return (bid < m * S).select(own, i32(0))

    def recv_pkts(a, r_recv, row, rstride, c0, vc):
        out = []
        for p in range_constexpr(MAX_TP):
            pc = fx.min(i32(p), a["tp"] - i32(1))
            off = ((pc * rstride + row) * i32(H) + c0 + vc * i32(8)) * i32(2)
            out.append(fx.Vector(bld(r_recv, off, 0, V4I, AUX_SYS)))
        return out

    @traced
    def final_chunk(L, tid, a, cidx):
        lane = tid % i32(64)
        nblk = gpu.grid_dim.x
        c0 = cidx * i32(CW)
        r_recv = own_rs(a, "off_part")
        step = (fin_split(a) > i32(1)).select(a["m"], i32(nblk))
        for row_ in range(fin_key(a), a["m"], step):
            row = i32(row_)
            _final_row(L, tid, a, r_recv, row, c0)
        if const_expr(ar):  # noqa: SIM102
            if lane == i32(0):
                lds_atomic_or(L, L_CTL + C_YAGM * 4, i32(1) << cidx)

    @traced
    def _final_row(L, tid, a, r_recv, row, c0):
        lane = tid % i32(64)
        for j in range_constexpr(VPL):
            v = lane + i32(j * 64)
            vc = fx.min(v, i32(CW // 8 - 1))
            if const_expr(ll_rs):
                _final_ll(tid, a, r_recv, row, c0, v, vc)
                continue
            acc = [fx.Float32(0.0)] * 8
            if const_expr(comm_bf16):
                pk = recv_pkts(a, r_recv, row, a["mmax"], c0, vc)
                for p in range_constexpr(MAX_TP):
                    acc = sum_live(acc, bf16x8_to_f32(pk[p]), i32(p) < a["tp"])
            else:
                lds_ = []
                for p in range_constexpr(MAX_TP):
                    pc = fx.min(i32(p), a["tp"] - i32(1))
                    doff, soff = part_offs(a, pc * a["mmax"] + row, c0 + vc * i32(8))
                    lds_.append(
                        (
                            bld(r_recv, doff, 0, V2I, AUX_SYS),
                            bld(r_recv, soff, 0, T.i8, AUX_SYS),
                        )
                    )
                for p in range_constexpr(MAX_TP):
                    acc = sum_live(acc, fp8x8_decode(lds_[p]), i32(p) < a["tp"])
            _y_store(a, row, c0, v, acc)

    @traced
    def _final_ll(tid, a, r_recv, row, c0, v, vc):
        lane = tid % i32(64)
        pk = recv_pkts(a, r_recv, row, a["mmax"], c0, vc)
        live = [(i32(p) < a["tp"]) & (v < i32(CW // 8)) for p in range(MAX_TP)]
        pend = ll_pending(a, lane, zip(pk, live))
        t0 = now()
        while (pend != i32(0)) & alive(t0):
            rocdl.s_sleep(1)
            pk = recv_pkts(a, r_recv, row, a["mmax"], c0, vc)
            pend = ll_pending(a, lane, zip(pk, live))
        _report_if(a, (pend != i32(0)) & (lane == i32(0)), ERR_COMM)
        acc = [fx.Float32(0.0)] * 8
        for p in range_constexpr(MAX_TP):
            acc = sum_live(acc, ll_vals(pk[p]), i32(p) < a["tp"])
        _y_store(a, row, c0, v, acc)

    @traced
    def _y_store(a, row, c0, v, acc):
        if const_expr(AG8):
            _y_store_ag8(a, row, c0, v, acc)
        else:
            _y_store_bf16(a, row, c0, v, pack_bf16x8(acc))

    @traced
    def _y_store_bf16(a, row, c0, v, o):
        if v < i32(CW // 8):
            if const_expr(ar):
                off = ((a["rank"] * a["m"] + row) * i32(H) + c0 + v * i32(8)) * i32(2)
                for p in range_constexpr(TPC):
                    bst(o, peer_rs(a, p, "off_yall"), off, 0, AUX_SYS)
            else:
                bst(
                    o,
                    rsrc(a["y"]),
                    (row * i32(H) + c0 + v * i32(8)) * i32(2),
                    0,
                    AUX_SYS if TN else 0,
                )

    @traced
    def _y_store_ag8(a, row, c0, v, acc):
        col = c0 + v * i32(8)
        grow = a["rank"] * a["m"] + row
        d0, d1, e8i = mxfp8x8(acc)
        sc = scales4(e8i)
        f = fp8x4_unpack(d0, e8_scale(e8i)) + fp8x4_unpack(d1, e8_scale(e8i))
        if v < i32(CW // 8):
            bst(pack_bf16x8(f), rsrc(a["y"]), (grow * i32(H) + col) * i32(2), 0, 0)
            doff, soff = part_offs(a, grow, col)
            dv = fx.Vector.from_elements([d0, d1], fx.Int32)
            for p in range_constexpr(TPC):
                if i32(p) != a["rank"]:
                    rd = peer_rs(a, p, "off_yall")
                    bst(dv, rd, doff, 0, AUX_SYS)
                    if (v & i32(15)) == i32(0):
                        bst(sc, rd, soff, 0, AUX_SYS)

    @traced
    def ag8_convert(tid, a):
        bid = i32(gpu.block_id("x"))
        if bid < a["m"]:
            step = (fin_split(a) > i32(1)).select(a["m"], i32(gpu.grid_dim.x))
            ry = own_rs(a, "off_yall")
            yo = rsrc(a["y"])
            nq = i32(H // 8)
            for row_ in range(bid, a["m"], step):
                row = i32(row_)
                for q_ in range(tid, i32(TPC * (H // 8)), i32(NTT)):
                    q = i32(q_)
                    p = q // nq
                    col = (q - p * nq) * i32(8)
                    if p != a["rank"]:
                        grow = p * a["m"] + row
                        doff, soff = part_offs(a, grow, col)
                        ld = (
                            bld(ry, doff, 0, V2I, AUX_SYS),
                            bld(ry, soff, 0, T.i8, AUX_SYS),
                        )
                        bst(
                            pack_bf16x8(fp8x8_decode(ld)),
                            yo,
                            (grow * i32(H) + col) * i32(2),
                            0,
                            0,
                        )

    @traced
    def yag_flush(L, tid, a):
        if tid < i32(64):
            mask = lds_ld_i32(L, L_CTL + C_YAGM * 4)
            for it in range_constexpr(ceildiv(NV * TPC, 64)):
                e = i32(it * 64) + tid
                v = e // i32(TPC)
                p = e - v * i32(TPC)
                vc = fx.min(v, i32(NV - 1))
                idx = (
                    i32(FLAG_YAG)
                    + (vc * i32(MAX_TP) + a["rank"]) * i32(NCTA_MAX)
                    + fin_key(a)
                )
                ok = (v < i32(NV)) & (((mask >> vc) & i32(1)) == i32(1))
                _yag_post(a, p, idx, ok)

    @traced
    def _yag_post(a, p, idx, ok):
        if ok:
            fo = fx.Int64(a["off_flag"]) + fx.Int64(idx * i32(4))
            g_st_sys(peer_sel(a, p) + fo, a["epoch"])

    def _yag_pending(a, lane, epoch):
        rf = own_rs(a, "off_flag")
        bid = i32(gpu.block_id("x"))
        pend = i32(0)
        for it in range_constexpr(ceildiv(NV * MAX_TP, 64)):
            e = i32(it * 64) + lane
            cidx = e // i32(MAX_TP)
            src = e - cidx * i32(MAX_TP)
            ok = (cidx < i32(NV)) & (src < a["tp"])
            idx = (
                i32(FLAG_YAG)
                + (fx.min(cidx, i32(NV - 1)) * i32(MAX_TP) + src) * i32(NCTA_MAX)
                + bid
            )
            f = fx.Int32(bld(rf, idx * i32(4), 0, T.i32, AUX_SYS))
            pend = fx.max(pend, (ok & before(f, epoch)).select(i32(1), i32(0)))
        return wave_red(pend, lane, fx.max)

    @traced
    def yag_wait(tid, a, epoch):
        if (tid < i32(64)) & (i32(gpu.block_id("x")) < a["m"]):
            lane = tid % i32(64)
            poll_zero(lambda: _yag_pending(a, lane, epoch), a, lane, ERR_YAG)

    def comm_idle(a):
        bid = i32(gpu.block_id("x"))
        ns = push_units(a["ttot"])
        some = (ns * i32(NV) >= i32(gpu.grid_dim.x)) | (bid < ns * i32(NV))
        return (some.select(i32(0), i32(1)) == i32(1)) & (fin_owned(a) == i32(0))

    VMASK = (1 << NV) - 1

    def _lowest(v):
        return _ctpop((v & (i32(0) - v)) - i32(1))

    @traced
    def _claim_bit(L, rdy_off, bits_off, cnt_off, w):
        res = i32(-1)
        n = lds_ld_acq(L, L_CTL + cnt_off * 4)
        avail = (
            lds_ld_acq(L, L_CTL + rdy_off * 4)
            & (lds_ld_acq(L, L_CTL + bits_off * 4) ^ i32(-1))
            & i32(VMASK)
        )
        if (n < i32(NV)) & (avail != i32(0)):
            c = _lowest(avail)
            old = lds_atomic_or(L, L_CTL + bits_off * 4, i32(1) << c)
            if ((old >> c) & i32(1)) == i32(0):
                lds_atomic_add(L, L_CTL + cnt_off * 4, i32(1))
                res = c
        lds_st(L, L_CTL + (C_MBOX * 4) + w * i32(4), res)

    @traced
    def claim_push(L, lane, w):
        if lane == i32(0):
            _claim_bit(L, C_PRDY, C_PBITS, C_LRED, w)
        rocdl.sched_barrier(0)
        return uni(lds_ld_acq(L, L_CTL + (C_MBOX * 4) + w * i32(4)))

    @traced
    def claim_final(L, lane, w):
        if lane == i32(0):
            _claim_bit(L, C_FRDY, C_FBITS, C_PULL, w)
        rocdl.sched_barrier(0)
        return uni(lds_ld_acq(L, L_CTL + (C_MBOX * 4) + w * i32(4)))

    @traced
    def poll_ready(L, lane, a, epoch):
        c = fx.min(lane, i32(NV - 1))
        live = lane < i32(NV)
        v = g_ld_sys(lrdy_at(a, c))
        pm = i32(
            fx.Int64(
                rocdl.ballot(T.i64, live & (before(v, epoch) == fx.Boolean(False)))
            )
            & fx.Int64(VMASK)
        )
        fm = i32(
            fx.Int64(rocdl.ballot(T.i64, live & _final_ready(a, c, epoch)))
            & fx.Int64(VMASK)
        )
        pm = uni(pm)
        fm = uni(fm)
        old_p = lds_ld_acq(L, L_CTL + C_PRDY * 4)
        old_f = lds_ld_acq(L, L_CTL + C_FRDY * 4)
        if lane == i32(0):
            lds_atomic_or(L, L_CTL + C_PRDY * 4, pm)
            lds_atomic_or(L, L_CTL + C_FRDY * 4, fm)
        return (((pm & (old_p ^ i32(-1))) | (fm & (old_f ^ i32(-1)))) != i32(0)).select(
            i32(1), i32(0)
        )

    def _comm_pending(L):
        full = i32(VMASK)
        push = (lds_ld_acq(L, L_CTL + C_LRED * 4) < i32(NV)) & (
            (lds_ld_acq(L, L_CTL + C_PRDY * 4) & full) != full
        )
        fin = (lds_ld_acq(L, L_CTL + C_PULL * 4) < i32(NV)) & (
            (
                (lds_ld_acq(L, L_CTL + C_FRDY * 4) | lds_ld_acq(L, L_CTL + C_FBITS * 4))
                & full
            )
            != full
        )
        return (lds_ld_acq(L, L_CTL + C_NSIG * 4) < i32(NCK)) | push | fin

    @traced
    def poll_loop(L, tid, a, epoch):
        t0 = now()
        while _comm_pending(L) & alive(t0):
            p1 = comm_signal(L, tid, a, epoch)
            p2 = poll_ready(L, tid % i32(64), a, epoch)
            if (p1 | p2) == i32(0):
                rocdl.s_sleep(POLL_SLEEP)
        _report_if(
            a,
            (now() - t0 >= fx.Int64(DEADLINE)) & ((tid % i32(64)) == i32(0)),
            ERR_COMM,
        )

    def _final_ready(a, c, epoch):
        if const_expr(ll_rs):
            r_recv = own_rs(a, "off_part")
            r1 = fx.Boolean(True)
            for p in range_constexpr(MAX_TP):
                pc = fx.min(i32(p), a["tp"] - i32(1))
                off = (
                    (pc * a["mmax"] + fin_key(a)) * i32(H) + c * i32(CW) + i32(CW - 8)
                ) * i32(2)
                t = fx.Int32(bld(r_recv, off + i32(4), 0, T.i32, AUX_SYS))
                r1 = r1 & ((i32(p) >= a["tp"]) | (t == epoch))
            return r1
        rf = own_rs(a, "off_flag")
        fo = (i32(FLAG_RDY) + c * i32(MAX_TP)) * i32(4)
        fl = fx.Vector(bld(rf, fo, 0, V4I, AUX_SYS))
        fh = fx.Vector(bld(rf, fo + i32(16), 0, V4I, AUX_SYS))
        r1 = fx.Boolean(True)
        for p in range_constexpr(MAX_TP):
            f = fx.Int32((fl if p < 4 else fh)[p % 4])
            r1 = r1 & ((i32(p) >= a["tp"]) | (before(f, epoch) == fx.Boolean(False)))
        return r1

    @traced
    def push_chunk_dyn(L, tid, a, epoch, cidx):
        lane = tid % i32(64)
        ttot = a["ttot"]
        nblk = gpu.grid_dim.x
        ns = push_units(ttot)
        u0 = push_first_unit(ns, cidx)
        c0 = cidx * i32(CW)
        routes_bytes = route_region_bytes(ttot)
        r_routes = rsrc(a["routes"], routes_bytes)
        r_pr = rsrc(a["proutes"], i32(NPC - 1) * routes_bytes)
        P = uni(lds_ld_i32(L, L_CTL + C_DYNP * 4))
        n = fx.max(ceildiv(ns - u0, i32(nblk)), i32(0))
        for u_ in range(u0, ns, i32(nblk)):
            u = i32(u_)
            for t_ in range(u, ttot, ns):
                t = i32(t_)
                for j in range_constexpr(VPL):
                    v = lane + i32(j * 64)
                    vc = fx.min(v, i32(CW // 8 - 1))
                    lds_ = []
                    for k in range_constexpr(TOPK):
                        ridx = t * i32(TOPK) + i32(k)
                        lds_.append(
                            route_load(r_routes, i32(0), ttot, ridx, c0 + vc * i32(8))
                        )
                        for p in range_constexpr(1, NPC):
                            lds_.append(
                                route_load(
                                    r_pr,
                                    i32(p - 1) * routes_bytes,
                                    ttot,
                                    ridx,
                                    c0 + vc * i32(8),
                                    i32(p) < P,
                                )
                            )
                    acc = [fx.Float32(0.0)] * 8
                    for ld in lds_:
                        acc = [x + y for x, y in zip(acc, route_decode(ld))]
                    r_dst, prow = token_dst(a, t)
                    _push_store(a, r_dst, prow, c0, v, acc, v < i32(CW // 8))
        if const_expr(not ll_rs):
            wait_vm(0)
            if (lane == i32(0)) & (n > i32(0)):
                _push_done_all(a, epoch, cidx, ns, n)

    @traced
    def comm_work(L, tid, a, epoch):
        lane = tid % i32(64)
        w = tid // i32(64)
        cp = claim_final(L, lane, w)
        if cp >= i32(0):
            final_chunk(L, tid, a, cp)
        cr = claim_push(L, lane, w)
        if cr >= i32(0):
            push_chunk_dyn(L, tid, a, epoch, cr)
        return ((cr >= i32(0)) | (cp >= i32(0))).select(i32(1), i32(0))

    @traced
    def comm_signal(L, tid, a, epoch):
        lane = tid % i32(64)
        w = tid // i32(64)
        mn = i32(NCK)
        for wv in range_constexpr(NW):
            mn = fx.min(mn, lds_ld_acq(L, L_CTL + C_DONE * 4 + wv * 4))
        start = uni(lds_ld_acq(L, L_CTL + C_NSIG * 4))
        end = uni(fx.min(mn + i32(1), i32(NCK)))
        _signal_claim(L, lane, w, start, end)
        rocdl.sched_barrier(0)
        won = uni(lds_ld_acq(L, L_CTL + (C_MBOX * 4) + w * i32(4)))
        if won == i32(1):  # noqa: SIM102
            if start + lane < end:
                _signal_one(a, epoch, start + lane)
        rocdl.sched_barrier(0)
        return (end > start).select(i32(1), i32(0))

    @traced
    def _signal_claim(L, lane, w, start, end):
        if lane == i32(0):
            won = i32(0)
            if end > start:
                got = lds_cas(L, L_CTL + C_NSIG * 4, start, end)
                won = (got == start).select(i32(1), i32(0))
            lds_st(L, L_CTL + (C_MBOX * 4) + w * i32(4), won)

    def _signal_one(a, epoch, cidx):
        nblk = i32(gpu.grid_dim.x)
        x = i32(gpu.block_id("x")) % i32(N_XCD)
        _signal_part(a, epoch, cidx, x, ceildiv(nblk - x, i32(N_XCD)))

    @traced
    def _signal_part(a, epoch, cidx, slot, target):
        xa = sc_addr(a, cidx, slot)
        if g_add_agent(xa, 1) + i32(1) == target:
            g_st_sys(xa, i32(0))
            _signal_count(a, epoch, cidx)

    @traced
    def _signal_count(a, epoch, cidx):
        cnt_addr = sc_addr(a, cidx, i32(SC_ALL))
        parts = fx.min(i32(gpu.grid_dim.x), i32(N_XCD))
        if g_add_agent(cnt_addr, 1) + i32(1) == parts:
            g_st_sys(cnt_addr, i32(0))
            g_st_sys(lrdy_at(a, cidx), epoch)

    @traced
    def signal_loop(L, tid, a, epoch):
        t0 = now()
        while (lds_ld_acq(L, L_CTL + C_NSIG * 4) < i32(NCK)) & alive(t0):
            if comm_signal(L, tid, a, epoch) == i32(0):
                rocdl.s_sleep(1)

    @traced
    def comm_wave(L, tid, a, epoch):
        t0 = now()
        while (
            (lds_ld_acq(L, L_CTL + C_NSIG * 4) < i32(NCK))
            | (lds_ld_acq(L, L_CTL + C_LRED * 4) < i32(NV))
            | (lds_ld_acq(L, L_CTL + C_PULL * 4) < i32(NV))
        ) & alive(t0):
            p1 = comm_signal(L, tid, a, epoch)
            if const_expr(not ARLL):
                p1 = p1 | poll_ready(L, tid % i32(64), a, epoch)
            p2 = comm_work(L, tid, a, epoch)
            if (p1 | p2) == i32(0):
                rocdl.s_sleep(POLL_SLEEP)
        _report_if(
            a,
            (now() - t0 >= fx.Int64(DEADLINE)) & ((tid % i32(64)) == i32(0)),
            ERR_COMM,
        )

    @traced
    def comm_help(L, tid, a, epoch):
        t0 = now()
        while (
            (lds_ld_acq(L, L_CTL + C_LRED * 4) < i32(NV))
            | (lds_ld_acq(L, L_CTL + C_PULL * 4) < i32(NV))
        ) & alive(t0):
            if comm_work(L, tid, a, epoch) == i32(0):
                rocdl.s_sleep(POLL_SLEEP)

    AG_SROW = XB
    AG_SCB = agr * AG_SROW
    assert AG_SCB + agr * (H // 32) <= L_INTERS - L_INTER + RG * (I // 32)
    NCHA = H // 256
    AG_WIN = max(1, 64 // TPC)

    def ag_split(a):
        m = fx.max(a["m"], i32(1))
        return fx.min(fx.max(i32(gpu.grid_dim.x) // m, i32(1)), i32(NCHA))

    def ag_rows(a):
        bid = i32(gpu.block_id("x"))
        nblk = i32(gpu.grid_dim.x)
        S = ag_split(a)
        nr = fx.min(fx.max(ceildiv(a["m"] - bid, nblk), i32(0)), i32(agr))
        m = fx.max(a["m"], i32(1))
        s = bid // m
        split = S > i32(1)
        live = s < S
        row0 = split.select(bid - s * m, bid)
        nrows = split.select(live.select(i32(1), i32(0)), nr)
        qb = split.select(s, i32(0))
        qs = split.select(S, i32(1))
        cnt = split.select(live.select(ceildiv(i32(NCHA) - s, S), i32(0)), i32(NCHA))
        return row0, nblk, nrows, qb, qs, cnt

    def _row32(rs, row, g, aux):
        f = []
        for v in range_constexpr(4):
            f += bf16x8_to_f32(
                bld(
                    rs,
                    ((row * i32(H)) + g * i32(32) + i32(v * 8)) * i32(2),
                    0,
                    V4I,
                    aux,
                )
            )
        return f

    def ag_row_vals(a, rxl, i, g):
        return _row32(rxl, (a["rank"] * a["m"] if ar else i32(0)) + i, g, 0)

    @traced
    def quant_chunks(L, t0, stride, a, k_lo, k_hi):
        row0, stride_r, nrows, qb, qstep, cnt = ag_rows(a)
        k_hi = fx.min(k_hi, cnt)
        nk = fx.max(k_hi - k_lo, i32(0))
        rxl = rsrc(a["x"], (a["ttot"] if ar else a["m"]) * i32(H * 2))
        for q_ in range(t0, nrows * nk * i32(8), i32(stride)):
            q = i32(q_)
            r = q // (nk * i32(8))
            e = q - r * nk * i32(8)
            g = (qb + (k_lo + e // i32(8)) * qstep) * i32(8) + e % i32(8)
            f = ag_row_vals(a, rxl, row0 + r * stride_r, g)
            e8, qs = _e8m0_from_amax(amax(f), max_norm=448.0 if a8 else 6.0)
            if const_expr(a8):
                words = [fp8x4_pack(f[dw * 4 : dw * 4 + 4], qs) for dw in range(8)]
            else:
                words = [fp4_pack(f[dw * 8 : dw * 8 + 8], qs) for dw in range(4)]
            lds_st(
                L,
                L_INTER + r * i32(AG_SROW) + g * i32(4 * len(words)),
                fx.Vector.from_elements(words, fx.Int32),
                16,
            )
            lds_st(
                L, L_INTER + i32(AG_SCB) + r * i32(H // 32) + g, fx.Int8(e8), align=1
            )

    @traced
    def ag_send(L, lane, a):
        row0, stride, nrows, qb, qs, cnt = ag_rows(a)
        rank = a["rank"]
        if const_expr(not ar and not MLL):
            mp = (ag_split(a) > i32(1)).select(
                i32(0), _meta_pending(a, lane, a["epoch"], own=True)
            )
            t0 = now()
            while (mp != i32(0)) & alive(t0):
                rocdl.s_sleep(1)
                mp = _meta_pending(a, lane, a["epoch"], own=True)
        r = lane // i32(XLPR)
        j = lane - r * i32(XLPR)
        rok = r < nrows
        i = row0 + fx.min(r, fx.max(nrows - i32(1), i32(0))) * stride
        gslot = rank * a["m"] + i
        x_bytes = a["ttot"] * i32(XB)
        xs_bytes = a["ttot"] * i32(H // 32)
        rx = [peer_rs(a, p, "off_x", x_bytes) for p in range(TPC)]
        rs = [peer_rs(a, p, "off_xs", xs_bytes) for p in range(TPC)]
        for k in range_constexpr(NCHA):
            kl = i32(k) < cnt
            q = fx.min(qb + i32(k) * qs, i32(NCHA - 1))
            if kl:
                ag_chunk(L, lane, a, q, r, j, rok, gslot, rx, rs, x_bytes, xs_bytes)
            if const_expr(k == NCHA - 1):
                wait_lgkm0()
                if lane == i32(0):
                    lds_atomic_add(L, L_CTL + C_AGFREE * 4, 1, REL)
            if const_expr(k >= 1):  # noqa: SIM102
                if i32(k - 1) < cnt:
                    if kl:
                        wait_vm(2 * TPC)
                    else:
                        wait_vm(0)
                    _ag_bump(a, lane, qb + i32(k - 1) * qs)
            rocdl.sched_barrier(0)
        wait_vm(0)
        if i32(NCHA - 1) < cnt:
            _ag_bump(a, lane, qb + i32(NCHA - 1) * qs)

    @traced
    def ag_chunk(L, lane, a, q, r, j, rok, gslot, rx, rs, x_bytes, xs_bytes):
        rr = fx.min(r, i32(agr - 1))
        cb = q * i32(XLPR * 16) + j * i32(16)
        dv = lds_ld(L, L_INTER + rr * i32(AG_SROW) + cb, V4I, 16)
        sv = lds_ld(L, L_INTER + i32(AG_SCB) + rr * i32(H // 32) + q * i32(8), V2I, 8)
        doff = rok.select(gslot * i32(XB) + cb, x_bytes)
        soff = (rok & (j == i32(0))).select(gslot * i32(H // 32) + q * i32(8), xs_bytes)
        for p in range_constexpr(TPC):
            bst(dv, rx[p], doff, 0, AUX_SYS)
            bst(sv, rs[p], soff, 0, AUX_SYS)

    @traced
    def ag_stage_free(L, lane, a):
        spin0(L, lane, L_CTL + C_AGFREE * 4, i32(1))

    def _ag_senders(a, q, x):
        m = fx.max(a["m"], i32(1))
        S = ag_split(a)
        lo = (S > i32(1)).select((q % S) * m, i32(0))
        hi = (S > i32(1)).select(lo + m, i32(gpu.grid_dim.x))
        return ceildiv(hi - x, i32(N_XCD)) - ceildiv(lo - x, i32(N_XCD))

    @traced
    def _ag_bump(a, lane, q):
        if lane == i32(0):
            x = i32(gpu.block_id("x")) % i32(N_XCD)
            xa = ctrl_at(a, i32(CTRL_AGX) + (q * i32(N_XCD) + x) * i32(LRDY_STRIDE))
            if g_add_agent(xa, 1) + i32(1) == _ag_senders(a, q, x):
                g_st_sys(xa, i32(0))
                _ag_bump_rank(a, q)

    @traced
    def _ag_bump_rank(a, q):
        ga = ctrl_at(a, i32(CTRL_AGG) + q * i32(LRDY_STRIDE))
        m = fx.max(a["m"], i32(1))
        nx = fx.min((ag_split(a) > i32(1)).select(m, i32(gpu.grid_dim.x)), i32(N_XCD))
        if g_add_agent(ga, 1) + i32(1) == nx:
            g_st_sys(ga, i32(0))
            idx = i32(FLAG_AGQ) + q * i32(MAX_TP) + a["rank"]
            for p in range_constexpr(TPC):
                bst(a["epoch"], peer_rs(a, p, "off_flag"), idx * i32(4), 0, AUX_SYS)

    def _ag_nmeta(a):
        return fx.max(ceildiv(a["m"] * i32(TOPK) // i32(4), i32(64)), i32(1))

    @traced
    def _ag_send_meta(lane, a):
        bid = i32(gpu.block_id("x"))
        if bid < _ag_nmeta(a):
            n = a["m"] * i32(TOPK)
            n4 = n // i32(4)
            rid = rsrc(a["ids_in"], n * i32(4))
            rtw = rsrc(a["tw_in"], n * i32(4))
            dst0 = a["rank"] * n * i32(4)
            v = bid * i32(64) + lane
            idv = bld(rid, v * i32(16), 0, V4I, 0)
            wvv = bld(rtw, v * i32(16), 0, V4I, 0)
            e = n4 * i32(4) + lane
            ide = bld(rid, e * i32(4), 0, T.i32, 0)
            wte = bld(rtw, e * i32(4), 0, T.i32, 0)
            big = i32(1 << 30)
            vo = (v < n4).select(dst0 + v * i32(16), big)
            eo = ((bid == i32(0)) & (e < n)).select(dst0 + e * i32(4), big)
            for p in range_constexpr(TPC):
                ri = peer_rs(a, p, "off_ids", big)
                rw = peer_rs(a, p, "off_w", big)
                bst(idv, ri, vo, 0, AUX_SYS)
                bst(wvv, rw, vo, 0, AUX_SYS)
                bst(ide, ri, eo, 0, AUX_SYS)
                bst(wte, rw, eo, 0, AUX_SYS)
            wait_vm(0)
            if lane == i32(0):
                fo = (i32(FLAG_AGM) + a["rank"] * i32(NMETA) + bid) * i32(4)
                bst(a["epoch"], own_rs(a, "off_flag"), fo, 0, AUX_SYS)
                for p in range_constexpr(TPC):
                    if i32(p) != a["rank"]:
                        bst(a["epoch"], peer_rs(a, p, "off_flag"), fo, 0, AUX_SYS)

    @traced
    def _ag_send_meta_ll(lane, a):
        n = a["m"] * i32(TOPK)
        rid = rsrc(a["ids_in"], n * i32(4))
        rtw = rsrc(a["tw_in"], n * i32(4))
        big = i32(1 << 30)
        for i_ in range(
            i32(gpu.block_id("x")) * i32(64) + lane, n, i32(gpu.grid_dim.x) * i32(64)
        ):
            i = i32(i_)
            e = fx.Int32(bld(rid, i * i32(4), 0, T.i32, 0))
            wv = fx.Int32(bld(rtw, i * i32(4), 0, T.i32, 0))
            pkt = fx.Vector.from_elements([e, a["epoch"], wv, a["epoch"]], fx.Int32)
            off = (a["rank"] * n + i) * i32(16)
            for p in range_constexpr(TPC):
                bst(pkt, peer_rs(a, p, "off_ids", big), off, 0, AUX_SYS)

    @traced
    def meta_ll_wait(tid, a):
        n = a["ttot"] * i32(TOPK)
        rid = rsrc(a["ids"])
        for i_ in range(tid, n, i32(NT)):
            i = i32(i_)
            t = fx.Int32(fx.Vector(bld(rid, i * i32(16), 0, V4I, AUX_SYS))[1])
            t0 = now()
            while (t != a["epoch"]) & alive(t0):
                rocdl.s_sleep(1)
                t = fx.Int32(fx.Vector(bld(rid, i * i32(16), 0, V4I, AUX_SYS))[1])
            _report_if(a, t != a["epoch"], ERR_META)

    def _meta_pending(a, lane, epoch, own=False):
        rf = own_rs(a, "off_flag")
        nmeta = _ag_nmeta(a)
        pend = i32(0)
        for it in range_constexpr(MAX_TP * NMETA // 64):
            e = i32(it * 64) + lane
            p = e // i32(NMETA)
            j = e - p * i32(NMETA)
            ok = ((p == a["rank"]) if own else (p < a["tp"])) & (j < nmeta)
            f = fx.Int32(
                bld(
                    rf, (i32(FLAG_AGM) + p * i32(NMETA) + j) * i32(4), 0, T.i32, AUX_SYS
                )
            )
            pend = fx.max(pend, (ok & before(f, epoch)).select(i32(1), i32(0)))
        return wave_red(pend, lane, fx.max)

    @traced
    def ag_wait_meta(a, tid, epoch):
        if tid < i32(64):
            lane = tid % i32(64)
            poll_zero(lambda: _meta_pending(a, lane, epoch), a, lane, ERR_META)

    def _ag_first_pending(lane, a, epoch, c_lo, nwin):
        rf = own_rs(a, "off_flag")
        end = fx.min(c_lo + i32(nwin), i32(NCH))
        first = end
        for it in range_constexpr(ceildiv(TPC * nwin, 64)):
            e = i32(it * 64) + lane
            cc = c_lo + e // i32(TPC)
            src = e % i32(TPC)
            idx = i32(FLAG_AGQ) + fx.min(cc, i32(NCH - 1)) * i32(MAX_TP) + src
            f = fx.Int32(bld(rf, idx * i32(4), 0, T.i32, AUX_SYS))
            first = fx.min(first, ((cc < end) & before(f, epoch)).select(cc, end))
        return wave_red(first, lane, fx.min)

    @traced
    def ag_wait_chunk(L, lane, a, epoch, cc):
        rdy = _ag_first_pending(lane, a, epoch, cc, AG_WIN)
        if cc == i32(0):
            rdy = _ag_first_pending(lane, a, epoch, i32(0), NCH)
        t0 = now()
        while (rdy <= cc) & alive(t0):
            rocdl.s_sleep(1)
            rdy = _ag_first_pending(lane, a, epoch, cc, AG_WIN)
        _report_if(a, (rdy <= cc) & (lane == i32(0)), ERR_CHUNK)
        if lane == i32(0):
            lds_st(L, L_CTL + C_ARDY * 4, rdy)
        rocdl.sched_barrier(0)

    @traced
    def finish(tid, a, epoch):
        if tid == i32(0):
            bid = i32(gpu.block_id("x"))
            g_st_sys(ctrl_at(a, i32(CTRL_EPB) + bid), epoch)
            x = bid % i32(N_XCD)
            nx = ceildiv(i32(gpu.grid_dim.x) - x, i32(N_XCD))
            xc = ctrl_at(a, i32(CTRL_XES) + x * i32(LRDY_STRIDE))
            if g_add_agent(xc, 1) == nx - i32(1):
                g_st_sys(xc, i32(0))
                fx.memory_fence(syncscope="one-as", ordering=ACQ)

    @traced
    def _dyn_zero(L, tid):
        if (tid >= i32(NT + 64)) & (tid < i32(NT + 64 + NBW)):
            lds_st(L, L_DYN + (tid - i32(NT + 64)) * i32(4), i32(0))
        for e_ in range(tid, i32(E), i32(NTT)):
            lds_st(L, L_DCNT + i32(e_) * i32(4), i32(0))
        if const_expr(lb):
            for e_ in range(tid, i32(E), i32(NTT)):
                lds_st(L, L_DPRE + i32(e_) * i32(4), i32(0))

    @traced
    def init_lds(L, tid, a):
        bid = i32(gpu.block_id("x"))
        if tid < i32(C_UL):
            is_done = (tid >= i32(C_DONE)) & (tid < i32(C_DONE + NW))
            ep = g_ld_rel(ctrl_at(a, i32(CTRL_EPB) + bid), "agent") + i32(1)
            v = is_done.select(i32(-1), (tid == i32(C_EPOCH)).select(ep, i32(0)))
            stage = (tid == i32(C_LRED)) | (tid == i32(C_PULL))
            v = (stage & comm_idle(a)).select(i32(NV), v)
            if const_expr(DLL):
                v = ((tid == i32(C_LRED)) | (tid == i32(C_NSIG))).select(i32(NCK), v)
            if const_expr(lb):
                v = (tid == i32(C_NSIG)).select(i32(NCK), v)
            own = fin_owned(a)
            v = (tid == i32(C_PULL)).select(i32(NV) - _ctpop(own), v)
            v = (tid == i32(C_FBITS)).select(i32((1 << NV) - 1) & (own ^ i32(-1)), v)
            if const_expr(ARLL):
                v = (tid == i32(C_PULL)).select(i32(NCK), v)
            lds_st(L, L_CTL + tid * i32(4), v)

    def _dll_rows(a, t, c0, vc, P):
        rb = route_region_bytes(a["ttot"])
        r_routes = rsrc(a["routes"], rb)
        r_pr = rsrc(a["proutes"], i32(NPC - 1) * rb)
        out = []
        for k in range_constexpr(TOPK):
            off = ((t * i32(TOPK) + i32(k)) * i32(H) + c0 + vc * i32(8)) * i32(2)
            out.append(fx.Vector(bld(r_routes, off, 0, V4I, AUX_SC1)))
            for p in range_constexpr(1, NPC):
                o2 = (i32(p) < P).select(i32(p - 1) * rb + off, i32(NPC) * rb)
                out.append(fx.Vector(bld(r_pr, o2, 0, V4I, AUX_SC1)))
        return out

    def _dll_items(ws):
        nblk = i32(gpu.grid_dim.x)
        return i32(gpu.block_id("x")) + i32(ws) * nblk, nblk * i32(4)

    @traced
    def push_dll(L, tid, a, ws=0):
        lane = tid % i32(64)
        spin0(L, lane, L_CTL + C_CDONE * 4, i32(1))
        P = uni(lds_ld_acq(L, L_CTL + C_DYNP * 4))
        ttot = a["ttot"]
        it0, istep = _dll_items(ws)
        for it_ in range(it0, ttot * i32(NCK), istep):
            it = i32(it_)
            c = it // ttot
            t = it - c * ttot
            c0 = c * i32(CW)
            r_dst, prow = token_dst(a, t)
            if const_expr(ARLL):
                prow = a["rank"] * a["mmax"] * a["tp"] + t
            for j in range_constexpr(VPL):
                v = lane + i32(j * 64)
                vc = fx.min(v, i32(CW // 8 - 1))
                live = [(v < i32(CW // 8)) & (i32(p) < P) for p in range(NPC)] * TOPK
                if const_expr(TOPK * NPC <= 48):
                    pk = _dll_rows(a, t, c0, vc, P)
                    pend = ll_pending(a, lane, zip(pk, live))
                    t0 = now()
                    while (pend != i32(0)) & alive(t0):
                        rocdl.s_sleep(8)
                        pk = _dll_rows(a, t, c0, vc, P)
                        pend = ll_pending(a, lane, zip(pk, live))
                else:
                    pend = ll_pending(a, lane, zip(_dll_rows(a, t, c0, vc, P), live))
                    t0 = now()
                    while (pend != i32(0)) & alive(t0):
                        rocdl.s_sleep(8)
                        pend = ll_pending(
                            a, lane, zip(_dll_rows(a, t, c0, vc, P), live)
                        )
                    pk = _dll_rows(a, t, c0, vc, P)
                _report_if(a, (pend != i32(0)) & (lane == i32(0)), ERR_COMM)
                acc = [fx.Float32(0.0)] * 8
                for k in range_constexpr(TOPK):
                    for p in range_constexpr(NPC):
                        acc = sum_live(acc, ll_vals(pk[k * NPC + p]), i32(p) < P)
                if const_expr(ARLL):
                    for pr in range_constexpr(TPC):
                        _push_store(
                            a,
                            peer_rs(a, pr, "off_part"),
                            prow,
                            c0,
                            v,
                            acc,
                            v < i32(CW // 8),
                        )
                else:
                    _push_store(a, r_dst, prow, c0, v, acc, v < i32(CW // 8))
        if const_expr(ARLL):
            final_all_ll(tid, a, ws)

    @traced
    def final_all_ll(tid, a, ws=0):
        lane = tid % i32(64)
        ttot = a["ttot"]
        trows = a["mmax"] * a["tp"]
        r_recv = own_rs(a, "off_part")
        ry = rsrc(a["y"], ttot * i32(H * 2))
        it0, istep = _dll_items(ws)
        for it_ in range(it0, ttot * i32(NCK), istep):
            it = i32(it_)
            c = it // ttot
            t = it - c * ttot
            c0 = c * i32(CW)
            for j in range_constexpr(VPL):
                v = lane + i32(j * 64)
                vc = fx.min(v, i32(CW // 8 - 1))
                live = [(i32(p) < a["tp"]) & (v < i32(CW // 8)) for p in range(MAX_TP)]
                pk = recv_pkts(a, r_recv, t, trows, c0, vc)
                pend = ll_pending(a, lane, zip(pk, live))
                t0 = now()
                while (pend != i32(0)) & alive(t0):
                    rocdl.s_sleep(1)
                    pk = recv_pkts(a, r_recv, t, trows, c0, vc)
                    pend = ll_pending(a, lane, zip(pk, live))
                _report_if(a, (pend != i32(0)) & (lane == i32(0)), ERR_COMM)
                acc = [fx.Float32(0.0)] * 8
                for p in range_constexpr(MAX_TP):
                    acc = sum_live(acc, ll_vals(pk[p]), i32(p) < a["tp"])
                if v < i32(CW // 8):
                    bst(
                        pack_bf16x8(acc),
                        ry,
                        (t * i32(H) + c0 + v * i32(8)) * i32(2),
                        0,
                        0,
                    )

    return locals()
