# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Sequence-parallel int8 reduce-scatter + residual add + RMSNorm (+ MoE router), one kernel."""

from __future__ import annotations

import functools
import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.distributed as dist
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from ..kernels_common import LOG2E, ceildiv
from ..symmetric_arena import SymmetricArena
from ..tensor_shim import _preload_compiled, _run_compiled
from .common import (
    AUX_SYS,
    MAX_TP,
    amax,
    before,
    bf16x8_to_f32,
    bld,
    bst,
    g_or_agent,
    g_st_sys,
    gptr,
    i32,
    lds_ld_i32,
    lds_st,
    pack_bf16x8,
    poll_sys_ge,
    rsrc,
    sum_live,
    traced,
    uni,
    wave_red,
)

__all__ = ["SpRsNorm"]

NTH = 256
KS_MAX = 16
const_expr = fx.const_expr


ZT = 16
NRT_MAX = 128
ROWF = NRT_MAX * 16


@functools.cache
def compile_sp_rs_norm(
    H: int,
    tp: int,
    eps: float,
    E: int = 0,
    topk: int = 0,
    scale: float = 1.0,
    shared_w: float = 0.0,
    gemma: bool = True,
    logit_bf16: bool = True,
    nb: int = 256,
):
    NB = nb  # CTAs: one per CU
    ERR = NB + ROWF + NRT_MAX * KS_MAX  # ctrl: sticky watchdog bit
    RT = E > 0
    EPL = E // 64
    assert not RT or (E % 64 == 0 and EPL in (2, 4) and topk <= 64)
    assert H % 256 == 0
    NP = H // 8
    KS_DIVS = [d for d in range(KS_MAX, 1, -1) if (H // 128) % d == 0]
    NPF = min(3 if E <= 128 else 1, H // KS_MAX // 4 // 32)
    PIT = ceildiv(NP, NTH)
    assert NP % NTH == 0

    def fbits(v):
        return f"{struct.unpack('<I', struct.pack('<f', v))[0]:x}"

    name = (
        f"sp_rs_norm_h{H}_tp{tp}_e{fbits(eps)}_i8"
        + (f"_rt{E}k{topk}s{fbits(scale)}w{fbits(shared_w)}" if RT else "")
        + ("" if gemma else "_rms")
        + ("_lf32" if RT and not logit_bf16 else "")
        + (f"_nb{NB}" if NB != 256 else "")
    )

    def i8x4_pack(f, inv):
        w = i32(0)
        for k in range_constexpr(4):
            q = fx.Int32(fmath.roundeven(f[k] * inv))
            q = fx.max(fx.min(q, i32(127)), i32(-127))
            w = w | ((q & i32(0xFF)) << i32(8 * k))
        return w

    def i8x4_unpack(d, sc):
        out = []
        for k in range_constexpr(4):
            b = (fx.Int32(d) << i32(24 - 8 * k)) >> i32(24)
            out.append(b.to(fx.Float32) * sc)
        return out

    def fadd(x, y):
        return (x.bitcast(fx.Float32) + y.bitcast(fx.Float32)).bitcast(fx.Int32)

    LDSB = 128 + (NTH // 64) * 64 * (E // 16 if E else 1) * 16
    Shared = fx.struct(
        type("Shared", (), {"__annotations__": {"buf": fx.Array[fx.Int8, LDSB, 16]}})
    )

    def lds_f(L, off):
        return lds_ld_i32(L, off).bitcast(fx.Float32)

    def lds_stf(L, off, v):
        lds_st(L, off, fx.Float32(v).bitcast(fx.Int32))

    @traced
    def wait_epoch(a, addr):
        cur = poll_sys_ge(addr, a["epoch"], 1)
        if before(cur, a["epoch"]):
            g_or_agent(fx.Int64(a["ctrl"]) + fx.Int64(i32(ERR * 4)), i32(1))

    def zflag(a, rt, ks):
        return fx.Int64(a["ctrl"]) + fx.Int64(
            (i32(NB + ROWF) + rt * i32(KS_MAX) + ks) * i32(4)
        )

    def rt_tiles(m):
        rt = m // i32(ZT)
        cap = i32(NB) // fx.max(rt, i32(1))
        ks = i32(1)
        for d in KS_DIVS:
            ks = ((ks == i32(1)) & (cap >= i32(d))).select(i32(d), ks)
        return rt, ks

    @traced
    def router_tasks(L, a, tid):
        m = a["m"]
        bid = i32(gpu.block_id("x"))
        lane = tid % i32(64)
        w = tid // i32(64)
        RT, KS = rt_tiles(m)
        kw = i32(H) // KS
        ro = rsrc(a["out"])
        rg = rsrc(a["wg"])
        rzp = rsrc(a["zp"])
        for t_ in range(i32(NB - 1) - bid, RT * KS, i32(NB)):
            t = i32(t_)
            rt = t // KS
            ks = t - rt * KS
            row = a["rank"] * m + rt * i32(ZT) + lane % i32(16)
            k0 = ks * kw + w * (kw // i32(4))
            pf = []
            for j in range_constexpr(NPF):
                kc = k0 + i32(j * 32) + (lane // i32(16)) * i32(8)
                pf.append(
                    [
                        bld(
                            rg,
                            ((i32(n * 16) + lane % i32(16)) * i32(H) + kc) * i32(2),
                            0,
                            T.vec(4, T.i32),
                        )
                        for n in range(E // 16)
                    ]
                )
            if tid < i32(ZT):
                r = rt * i32(ZT) + tid
                addr = fx.Int64(a["ctrl"]) + fx.Int64((i32(NB) + r) * i32(4))
                wait_epoch(a, addr)
            gpu.barrier()

            def step(acc, kc, bvs, row):
                av = fx.Vector(
                    bld(ro, (row * i32(H) + kc) * i32(2), 0, T.vec(4, T.i32), AUX_SYS)
                ).bitcast(fx.BFloat16)
                out_ = []
                for n in range_constexpr(E // 16):
                    bv = fx.Vector(bvs[n]).bitcast(fx.BFloat16)
                    out_.append(
                        fx.Vector(
                            rocdl.mfma_f32_16x16x32_bf16(
                                T.vec(4, T.f32), [av, bv, acc[n], 0, 0, 0]
                            )
                        )
                    )
                return out_

            acc0 = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(E // 16)]
            for j in range_constexpr(NPF):
                acc0 = step(
                    acc0, k0 + i32(j * 32) + (lane // i32(16)) * i32(8), pf[j], row
                )
            for kk_, st in range(i32(NPF * 32), kw // i32(4), i32(32), init=acc0):
                kc = k0 + i32(kk_) + (lane // i32(16)) * i32(8)
                bvs = [
                    bld(
                        rg,
                        ((i32(n * 16) + lane % i32(16)) * i32(H) + kc) * i32(2),
                        0,
                        T.vec(4, T.i32),
                    )
                    for n in range(E // 16)
                ]
                res = yield step(list(st), kc, bvs, row)
            for n in range_constexpr(E // 16):
                for i in range_constexpr(4):
                    lds_stf(
                        L,
                        i32(128)
                        + (
                            ((w * i32(E // 16) + i32(n)) * i32(64) + lane) * i32(4)
                            + i32(i)
                        )
                        * i32(4),
                        fx.Vector(res[n])[i],
                    )
            gpu.barrier()
            if w == i32(0):
                for n in range_constexpr(E // 16):
                    for i in range_constexpr(4):
                        v = fx.Float32(0.0)
                        for ww in range_constexpr(NTH // 64):
                            v = v + lds_f(
                                L,
                                i32(128)
                                + (
                                    (i32((ww * (E // 16) + n) * 64) + lane) * i32(4)
                                    + i32(i)
                                )
                                * i32(4),
                            )
                        rl = (lane // i32(16)) * i32(4) + i32(i)
                        ex = i32(n * 16) + lane % i32(16)
                        zo = (
                            ((rt * i32(KS_MAX) + ks) * i32(ZT) + rl) * i32(E) + ex
                        ) * i32(4)
                        bst(v.bitcast(fx.Int32), rzp, zo, 0, AUX_SYS)
                rocdl.s_waitcnt(vmcnt=0)
                if lane == i32(0):
                    g_st_sys(zflag(a, rt, ks), a["epoch"])
                if lane < KS:
                    wait_epoch(a, zflag(a, rt, lane))
            gpu.barrier()
            for rl_ in range(ks + w * KS, i32(ZT), KS * i32(NTH // 64)):
                router_row(a, lane, rt, i32(rl_), KS)
            gpu.barrier()

    @traced
    def router_row(a, lane, rt, rl, KS):
        rzp = rsrc(a["zp"])
        rb = rsrc(a["bias"])
        r = rt * i32(ZT) + rl
        vals, orig, idxs = [], [], []
        for i in range_constexpr(EPL):
            e = lane + i32(i * 64)
            vs = []
            for k in range_constexpr(KS_MAX):
                kc = fx.min(i32(k), KS - i32(1))
                zo = (((rt * i32(KS_MAX) + kc) * i32(ZT) + rl) * i32(E) + e) * i32(4)
                vs.append(
                    (
                        i32(k) < KS,
                        fx.Int32(bld(rzp, zo, 0, T.i32, AUX_SYS)).bitcast(fx.Float32),
                    )
                )
            x = fx.Float32(0.0)
            for live, v in vs:
                x = x + live.select(v, fx.Float32(0.0))
            if const_expr(logit_bf16):
                x = fx.Float32(x).to(fx.BFloat16).to(fx.Float32)
            sc = fx.Float32(1.0) / (
                fx.Float32(1.0) + fx.Float32(fmath.exp2(x * fx.Float32(-LOG2E)))
            )
            orig.append(sc)
            b = fx.Int32(bld(rb, e * i32(4), 0, T.i32)).bitcast(fx.Float32)
            vals.append(sc + b)
            idxs.append(e)
        if const_expr(EPL == 2):
            sw = vals[1] > vals[0]
            v0, v1 = sw.select(vals[1], vals[0]), sw.select(vals[0], vals[1])
            o0, o1 = sw.select(orig[1], orig[0]), sw.select(orig[0], orig[1])
            i0, i1 = sw.select(idxs[1], idxs[0]), sw.select(idxs[0], idxs[1])
            cur = i32(0)
            tot = fx.Float32(0.0)
            my_id = i32(0)
            my_w = fx.Float32(0.0)
            ninf = fx.Float32(float("-inf"))
            for k in range_constexpr(topk):
                mv = (cur == i32(0)).select(v0, (cur == i32(1)).select(v1, ninf))
                mi = (cur == i32(0)).select(i0, i1)
                mo = (cur == i32(0)).select(o0, o1)
                mx = wave_red(
                    mv.bitcast(fx.Int32),
                    lane,
                    lambda p_, q_: p_.bitcast(fx.Float32)
                    .maximumf(q_.bitcast(fx.Float32))
                    .bitcast(fx.Int32),
                ).bitcast(fx.Float32)
                bal = fx.Int64(rocdl.ballot(T.i64, mv == mx))
                win = i32(fx.ctpop(fx.Int64((bal & (fx.Int64(0) - bal)) - fx.Int64(1))))
                win = (bal == fx.Int64(0)).select(i32(0), win)
                wid = i32(rocdl.readlane(T.i32, mi, win))
                wgt = fx.Int32(
                    rocdl.readlane(T.i32, mo.bitcast(fx.Int32), win)
                ).bitcast(fx.Float32)
                cur = cur + ((lane == win) & (cur < i32(2))).select(i32(1), i32(0))
                tot = tot + wgt
                my_id = (lane == i32(k)).select(wid, my_id)
                my_w = (lane == i32(k)).select(wgt, my_w)
        else:
            tot = fx.Float32(0.0)
            my_id = i32(0)
            my_w = fx.Float32(0.0)
            ninf = fx.Float32(float("-inf"))
            for k in range_constexpr(topk):
                bv, bo, bi, bj = vals[0], orig[0], idxs[0], i32(0)
                for j in range_constexpr(1, EPL):
                    t = vals[j] > bv
                    bv = t.select(vals[j], bv)
                    bo = t.select(orig[j], bo)
                    bi = t.select(idxs[j], bi)
                    bj = t.select(i32(j), bj)
                mx = wave_red(
                    bv.bitcast(fx.Int32),
                    lane,
                    lambda p_, q_: p_.bitcast(fx.Float32)
                    .maximumf(q_.bitcast(fx.Float32))
                    .bitcast(fx.Int32),
                ).bitcast(fx.Float32)
                bal = fx.Int64(rocdl.ballot(T.i64, bv == mx))
                win = i32(fx.ctpop(fx.Int64((bal & (fx.Int64(0) - bal)) - fx.Int64(1))))
                win = (bal == fx.Int64(0)).select(i32(0), win)
                wid = i32(rocdl.readlane(T.i32, bi, win))
                wgt = fx.Int32(
                    rocdl.readlane(T.i32, bo.bitcast(fx.Int32), win)
                ).bitcast(fx.Float32)
                for j in range_constexpr(EPL):
                    vals[j] = ((lane == win) & (bj == i32(j))).select(ninf, vals[j])
                tot = tot + wgt
                my_id = (lane == i32(k)).select(wid, my_id)
                my_w = (lane == i32(k)).select(wgt, my_w)
        f = fx.Float32(float(scale)) / tot.maximumf(fx.Float32(1e-20))
        ri = rsrc(a["ids"])
        rw = rsrc(a["tw"])
        if lane < i32(topk):
            bst(my_id, ri, (r * i32(topk + 1) + lane) * i32(4), 0)
            bst(
                (my_w * f).bitcast(fx.Int32), rw, (r * i32(topk + 1) + lane) * i32(4), 0
            )
        if lane == i32(topk):
            bst(i32(E), ri, (r * i32(topk + 1) + lane) * i32(4), 0)
            bst(
                fx.Float32(float(shared_w)).bitcast(fx.Int32),
                rw,
                (r * i32(topk + 1) + lane) * i32(4),
                0,
            )

    @traced
    def blk_sum(L, tid, v):
        r = wave_red(v.bitcast(fx.Int32), tid % i32(64), fadd)
        if ((tid % i32(64)) == i32(0)) & (tid < i32(NTH)):
            lds_st(L, i32(16) + (tid // i32(64)) * i32(4), r)
        gpu.barrier()
        t = fx.Float32(0.0)
        for k in range_constexpr(NTH // 64):
            t = t + lds_ld_i32(L, 16 + k * 4).bitcast(fx.Float32)
        gpu.barrier()
        return t

    @traced
    def send_rows(a, tid):
        m = a["m"]
        bid = i32(gpu.block_id("x"))
        lane = tid % i32(64)
        rp = rsrc(a["part"])
        for it_ in range(bid, m * i32(tp - 1), i32(NB)):
            it = i32(it_)
            r = it // i32(tp - 1)
            k = it - r * i32(tp - 1)
            d = (a["rank"] + i32(1) + k) % i32(tp)
            i = d * m + r
            peer = a["peer"][0]
            for j in range_constexpr(1, tp):
                peer = (d == i32(j)).select(a["peer"][j], peer)
            rx = rsrc(fx.Int64(peer) + fx.Int64(a["off_x"]))
            rsx = rsrc(fx.Int64(peer) + fx.Int64(a["off_s"]))
            slot = a["rank"] * m + r
            for k in range_constexpr(PIT):
                p = tid + i32(k * NTH)
                f = bf16x8_to_f32(
                    bld(rp, (i * i32(H) + p * i32(8)) * i32(2), 0, T.vec(4, T.i32))
                )
                am = amax(f)
                am = am.maximumf(am.shuffle_xor(i32(1), i32(64)))
                am = am.maximumf(am.shuffle_xor(i32(2), i32(64)))
                sc = am.maximumf(fx.Float32(1e-30)) / fx.Float32(127.0)
                inv = fx.Float32(1.0) / sc
                dv = fx.Vector.from_elements(
                    [i8x4_pack(f[0:4], inv), i8x4_pack(f[4:8], inv)], fx.Int32
                )
                bst(dv, rx, slot * i32(H) + p * i32(8), 0, AUX_SYS)
                if (lane & i32(3)) == i32(0):
                    bst(
                        sc.bitcast(fx.Int32),
                        rsx,
                        (slot * i32(H // 32) + p // i32(4)) * i32(4),
                        0,
                        AUX_SYS,
                    )

    @traced
    def post_flags(L, a, tid):
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        if tid < i32(tp):
            bid = i32(gpu.block_id("x"))
            peer = a["peer"][0]
            for j in range_constexpr(1, tp):
                peer = (tid == i32(j)).select(a["peer"][j], peer)
            fo = fx.Int64(a["off_f"]) + fx.Int64((a["rank"] * i32(NB) + bid) * i32(4))
            g_st_sys(fx.Int64(peer) + fo, a["epoch"])

    @traced
    def wait_row(a, tid, r):
        if tid < i32(tp):
            k = (a["rank"] - tid - i32(1) + i32(tp)) % i32(tp)
            src_cta = (r * i32(tp - 1) + k) % i32(NB)
            addr = (
                a["mine"]
                + fx.Int64(a["off_f"])
                + fx.Int64((tid * i32(NB) + src_cta) * i32(4))
            )
            if tid != a["rank"]:
                wait_epoch(a, addr)
        gpu.barrier()

    @traced
    def reduce_rows(L, a, tid):
        m = a["m"]
        bid = i32(gpu.block_id("x"))
        rp = rsrc(a["part"])
        rr = rsrc(a["res"])
        ro = rsrc(a["res_out"])
        rout = rsrc(a["out"])
        rw = rsrc(a["w"])
        rx = rsrc(a["mine"] + fx.Int64(a["off_x"]))
        rsx = rsrc(a["mine"] + fx.Int64(a["off_s"]))
        for r_ in range(bid, m, i32(NB)):
            r = i32(r_)
            wait_row(a, tid, r)
            g = a["rank"] * m + r
            fs = []
            ss = fx.Float32(0.0)
            for k in range_constexpr(PIT):
                p = tid + i32(k * NTH)
                off = (g * i32(H) + p * i32(8)) * i32(2)
                f = bf16x8_to_f32(bld(rp, off, 0, T.vec(4, T.i32)))
                rv = bf16x8_to_f32(bld(rr, off, 0, T.vec(4, T.i32)))
                f = [x + y for x, y in zip(f, rv)]
                for s in range_constexpr(tp):
                    src = i32(s)
                    live = src != a["rank"]
                    slot = src * m + r
                    d = fx.Vector(
                        bld(rx, slot * i32(H) + p * i32(8), 0, T.vec(2, T.i32), AUX_SYS)
                    )
                    sc = fx.Int32(
                        bld(
                            rsx,
                            (slot * i32(H // 32) + p // i32(4)) * i32(4),
                            0,
                            T.i32,
                            AUX_SYS,
                        )
                    ).bitcast(fx.Float32)
                    v = i8x4_unpack(d[0], sc) + i8x4_unpack(d[1], sc)
                    f = sum_live(f, v, live)
                bst(pack_bf16x8(f), ro, off, 0)
                for x in f:
                    ss = ss + x * x
                fs.append(f)
            tot = blk_sum(L, tid, ss)
            rcp = fx.Float32(
                fmath.rsqrt(tot / fx.Float32(float(H)) + fx.Float32(float(eps)))
            )
            for k in range_constexpr(PIT):
                p = tid + i32(k * NTH)
                off = (g * i32(H) + p * i32(8)) * i32(2)
                wv = bf16x8_to_f32(bld(rw, p * i32(16), 0, T.vec(4, T.i32)))
                bst(
                    pack_bf16x8(
                        [
                            x * rcp * ((w + fx.Float32(1.0)) if gemma else w)
                            for x, w in zip(fs[k], wv)
                        ]
                    ),
                    rout,
                    off,
                    0,
                    AUX_SYS if RT else 0,
                )
            if const_expr(RT):
                rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                if tid == i32(0):
                    g_st_sys(
                        fx.Int64(a["ctrl"]) + fx.Int64((i32(NB) + r) * i32(4)),
                        a["epoch"],
                    )

    @flyc.kernel(name=name, known_block_size=[NTH, 1, 1])
    def sp_rs_norm_kernel(
        part: fx.Int64,
        res: fx.Int64,
        res_out: fx.Int64,
        out: fx.Int64,
        w: fx.Int64,
        ctrl: fx.Int64,
        p0: fx.Int64,
        p1: fx.Int64,
        p2: fx.Int64,
        p3: fx.Int64,
        p4: fx.Int64,
        p5: fx.Int64,
        p6: fx.Int64,
        p7: fx.Int64,
        off_x: fx.Int64,
        off_s: fx.Int64,
        off_f: fx.Int64,
        xbank: fx.Int64,
        rank: fx.Int32,
        m: fx.Int32,
        wg: fx.Int64,
        bias: fx.Int64,
        ids: fx.Int64,
        tw: fx.Int64,
        zp: fx.Int64,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        lds = fx.SharedAllocator().allocate(Shared).peek()
        L = uni(fx.Int32(fx.ptrtoint(lds.buf.ptr)))
        peers = [p0, p1, p2, p3, p4, p5, p6, p7]
        mine = peers[0]
        for j in range_constexpr(1, MAX_TP):
            mine = (rank == i32(j)).select(peers[j], mine)
        bid = i32(gpu.block_id("x"))
        ea = fx.Int64(ctrl) + fx.Int64(bid * i32(4))
        epoch = i32(fx.generic_load(gptr(ea), dtype=fx.Int32)) + i32(1)
        a = {
            "part": part,
            "res": res,
            "res_out": res_out,
            "out": out,
            "w": w,
            "peer": peers,
            "mine": fx.Int64(mine),
            # receive buffers double-buffered by launch parity: a peer's next launch
            # never overwrites the rows this one still reads
            "off_x": off_x + fx.Int64(epoch & i32(1)) * xbank,
            "off_s": off_s + fx.Int64(epoch & i32(1)) * (xbank // fx.Int64(8)),
            "off_f": off_f,
            "rank": rank,
            "m": m,
            "epoch": epoch,
            "ctrl": ctrl,
            "wg": wg,
            "bias": bias,
            "ids": ids,
            "tw": tw,
            "zp": zp,
        }
        if tid < i32(4):
            lds_st(L, tid * i32(4), i32(0))
        gpu.barrier()
        send_rows(a, tid)
        post_flags(L, a, tid)
        reduce_rows(L, a, tid)
        if const_expr(RT):
            router_tasks(L, a, tid)
        if tid == i32(0):
            fx.generic_store(gptr(ea), epoch)

    @flyc.jit
    def launch(
        part: fx.Int64,
        res: fx.Int64,
        res_out: fx.Int64,
        out: fx.Int64,
        w: fx.Int64,
        ctrl: fx.Int64,
        p0: fx.Int64,
        p1: fx.Int64,
        p2: fx.Int64,
        p3: fx.Int64,
        p4: fx.Int64,
        p5: fx.Int64,
        p6: fx.Int64,
        p7: fx.Int64,
        off_x: fx.Int64,
        off_s: fx.Int64,
        off_f: fx.Int64,
        xbank: fx.Int64,
        rank: fx.Int32,
        m: fx.Int32,
        wg: fx.Int64,
        bias: fx.Int64,
        ids: fx.Int64,
        tw: fx.Int64,
        zp: fx.Int64,
        stream: fx.Stream,
    ):
        sp_rs_norm_kernel(
            part,
            res,
            res_out,
            out,
            w,
            ctrl,
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            off_x,
            off_s,
            off_f,
            xbank,
            rank,
            m,
            wg,
            bias,
            ids,
            tw,
            zp,
        ).launch(grid=(NB, 1, 1), block=(NTH, 1, 1), stream=stream)

    return launch


class SpRsNorm:
    """forward(part, res, w[, router]) -> (out, res_out) for this rank's rows.

    part / res: bf16 [T, H] (T = tp * m; this rank's partial of every token and the
    residual), w: bf16 [H]; out / res_out: bf16 [T, H], rows rank*m:(rank+1)*m written.
    router = (E, topk, scale, shared_w) at construction; forward(router=(wg, bias, ids,
    tw)): wg bf16 [E, H], bias fp32 [E], ids int32 / tw fp32 [>= m, topk + 1] (the last
    column is the shared expert E). A collective like MegaMoeTP (see its contract): a wait
    that times out sets error_flag(); reset() (collective) restarts the state.
    """

    def __init__(
        self,
        hidden: int,
        max_tokens: int,
        eps: float,
        group=None,
        device=None,
        router=None,
        gemma: bool = True,
        logit_bf16: bool = True,
    ):
        device = torch.device(device if device is not None else "cuda")
        if device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        self.device = device
        self.H = int(hidden)
        self.eps = float(eps)
        self.group = group
        self.tp = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        mmax = ceildiv(int(max_tokens), self.tp)
        self.mmax = mmax
        self.router = router
        E = int(router[0]) if router else 0
        topk = int(router[1]) if router else 0
        self.nb = int(torch.cuda.get_device_properties(device).multi_processor_count)
        bad = []
        if self.tp > MAX_TP:
            bad.append(f"tp <= {MAX_TP}")
        if self.H % (8 * NTH):
            bad.append(f"hidden % {8 * NTH} == 0")
        if router and (E not in (128, 256) or not 0 < topk < 64):
            bad.append("router experts in (128, 256) and 0 < topk < 64")
        if router and mmax // ZT > NRT_MAX:
            bad.append(f"router: max_tokens / tp <= {NRT_MAX * ZT}")
        if bad:
            raise ValueError(f"SpRsNorm h{self.H} tp{self.tp}: need " + ", ".join(bad))
        self._xbank = self.tp * mmax * self.H
        arena = SymmetricArena(group=group, device=device)
        self._x = arena.reserve("x", (2 * self._xbank,), torch.uint8)
        self._s = arena.reserve("s", (2 * self._xbank // 8,), torch.uint8)
        self._f = arena.reserve("f", (MAX_TP * self.nb,), torch.int32)
        arena.commit()
        self.arena = arena
        self._err = self.nb + ROWF + NRT_MAX * KS_MAX
        self.ctrl = torch.zeros(self._err + 1, dtype=torch.int32, device=device)
        self._zp = torch.empty(
            (NRT_MAX * KS_MAX * ZT * max(E, 1),) if router else (1,),
            dtype=torch.float32,
            device=device,
        )
        self._fn = compile_sp_rs_norm(
            self.H, self.tp, self.eps, gemma=bool(gemma), nb=self.nb
        )
        self._fn_rt = (
            compile_sp_rs_norm(
                self.H,
                self.tp,
                self.eps,
                E,
                topk,
                float(router[2]),
                float(router[3]),
                gemma=bool(gemma),
                logit_bf16=bool(logit_bf16),
                nb=self.nb,
            )
            if router
            else None
        )
        self._armed = set()

    def routes(self, tokens: int) -> bool:
        """forward(router=...) runs for this batch."""
        m = tokens // self.tp
        return (
            self._fn_rt is not None
            and tokens % self.tp == 0
            and m % ZT == 0
            and 0 < m <= self.mmax
        )

    def _check_args(self, part, res, w, out, res_out, router):
        T, H = part.shape[0], self.H
        if T % self.tp or not 0 < T // self.tp <= self.mmax:
            raise ValueError(
                f"SpRsNorm: {T} tokens: need tp | T and T / tp <= {self.mmax}"
            )
        for name, t, shp in (
            ("part", part, (T, H)),
            ("res", res, (T, H)),
            ("out", out, (T, H)),
            ("res_out", res_out, (T, H)),
            ("w", w, (H,)),
        ):
            if (
                tuple(t.shape) != shp
                or t.dtype != torch.bfloat16
                or not t.is_contiguous()
            ):
                raise ValueError(f"SpRsNorm: {name} needs contiguous bf16 {list(shp)}")
        if router is not None:
            if not self.routes(T):
                raise ValueError(f"SpRsNorm: {T} tokens cannot route (tp * 16 | T)")
            E, k1 = int(self.router[0]), int(self.router[1]) + 1
            wg, bias, ids, tw = router
            for name, t, dt, ok in (
                ("wg", wg, torch.bfloat16, tuple(wg.shape) == (E, H)),
                ("bias", bias, torch.float32, tuple(bias.shape) == (E,)),
                ("ids", ids, torch.int32, ids.dim() == 2 and ids.shape[1] == k1),
                ("tw", tw, torch.float32, tw.dim() == 2 and tw.shape[1] == k1),
            ):
                if not ok or t.dtype != dt or not t.is_contiguous():
                    raise ValueError(f"SpRsNorm: router {name}: wrong shape / dtype")
            if ids.shape[0] < T // self.tp or tw.shape[0] < T // self.tp:
                raise ValueError("SpRsNorm: router ids / tw need >= tokens / tp rows")

    def forward(self, part, res, w, out=None, res_out=None, router=None):
        m = part.shape[0] // self.tp
        out = torch.empty_like(part) if out is None else out
        res_out = torch.empty_like(res) if res_out is None else res_out
        self._check_args(part, res, w, out, res_out, router)
        peers = [int(b) for b in self.arena.base_ptrs] + [0] * (MAX_TP - self.tp)
        if router is not None:
            wg, bias, ids, tw = router
            rt = (wg.data_ptr(), bias.data_ptr(), ids.data_ptr(), tw.data_ptr())
            fn = self._fn_rt
        else:
            rt = (0, 0, 0, 0)
            fn = self._fn
        args = (
            part.data_ptr(),
            res.data_ptr(),
            res_out.data_ptr(),
            out.data_ptr(),
            w.data_ptr(),
            self.ctrl.data_ptr(),
            *peers,
            self._x.offset,
            self._s.offset,
            self._f.offset,
            self._xbank,
            self.rank,
            m,
            *rt,
            self._zp.data_ptr(),
        )
        if fn not in self._armed:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("SpRsNorm: first use inside graph capture")
            _preload_compiled(fn, *args, torch.cuda.current_stream())
            torch.cuda.synchronize(self.device)
            self.arena.barrier()
            self._armed.add(fn)
        _run_compiled(fn, *args, torch.cuda.current_stream())
        return out, res_out

    def warm(self) -> None:
        """Collective: compile and run once (eagerly; graph capture cannot), then
        check every rank's watchdog (reset + retry once on a timeout)."""
        tp = self.tp
        rows = tp * ZT if self._fn_rt is not None else tp
        x = torch.zeros((rows, self.H), dtype=torch.bfloat16, device=self.device)
        w = torch.zeros((self.H,), dtype=torch.bfloat16, device=self.device)
        for _ in range(2):
            self.forward(x, x.clone(), w)
            if self._fn_rt is not None:
                E, k = int(self.router[0]), int(self.router[1])
                dev = self.device
                wg = torch.zeros((E, self.H), dtype=torch.bfloat16, device=dev)
                bias = torch.zeros((E,), dtype=torch.float32, device=dev)
                ids = torch.empty((ZT, k + 1), dtype=torch.int32, device=dev)
                tw = torch.empty((ZT, k + 1), dtype=torch.float32, device=dev)
                self.forward(x, x.clone(), w, router=(wg, bias, ids, tw))
            torch.cuda.synchronize(self.device)
            errs = [None] * tp
            dist.all_gather_object(errs, self.poll_errors(), group=self.group)
            if not any(errs):
                return
            self.reset()
        raise RuntimeError(f"SpRsNorm.warm: launches timed out ({errs})")

    def error_flag(self) -> torch.Tensor:
        """Sticky watchdog bit (nonzero: a launch timed out) as a device tensor."""
        return self.ctrl[self._err : self._err + 1]

    def poll_errors(self) -> int:
        return int(self.ctrl[self._err].item())

    def check_errors(self) -> None:
        if self.poll_errors():
            self.ctrl[self._err] = 0
            raise RuntimeError(
                f"SpRsNorm rank {self.rank}: a wait timed out; the output is invalid "
                "(ranks out of step need reset())"
            )

    def reset(self) -> None:
        """Collective: drop the cross-rank state and restart the launch epochs."""
        torch.cuda.synchronize(self.device)
        self.arena.barrier()
        self.arena.storage.zero_()
        self.ctrl.zero_()
        torch.cuda.synchronize(self.device)
        self.arena.barrier()

    __call__ = forward
