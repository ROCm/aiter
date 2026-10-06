# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Sequence-parallel reduce-scatter + residual add + GemmaRMSNorm, one kernel.

Every rank holds its bf16 partial ``part`` [T, H] of every token (e.g. a
row-parallel output projection); rank r owns rows ``r*m : (r+1)*m``
(``T = tp * m``). Each rank sends the other ranks' rows of its partial as int8
with an fp32 scale per 32 columns (or MXFP8; about half the bf16 bytes over the
links) into their symmetric arena, then every rank sums its own rows -- its own
partial unquantized, the peers' dequantized -- adds ``res`` and writes
``res_out = sum`` and ``out = GemmaRMSNorm(sum; w, eps)`` (bf16, its own rows
only); optionally also the MoE router of its own rows (see compile_sp_rs_norm).
"""

from __future__ import annotations

import functools
import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.distributed as dist
from flydsl._mlir import ir
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl

const_expr = fx.const_expr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from .. import buffer_ops
from ..mxfp4_gemm_common import _e8m0_from_amax, _fabs_f32
from ..symmetric_arena import SymmetricArena
from ..tensor_shim import _preload_compiled, _run_compiled

__all__ = ["SpRsNorm"]

MAX_TP = 8
NB = 256  # CTAs
NTH = 256  # threads per CTA
NALL = NTH
KS_MAX = 16  # router: K slices per 16-row tile
AUX_SYS = 1 | 16
DEADLINE = 200_000_000

traced = ASTRewriter.transform


def _attr(v):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), int(v))


def _u(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


ZT = 16  # router: rows per tile
NRT_MAX = 128  # router row tiles (m / 16)
ROWF = NRT_MAX * 16  # router row flags (m)


@functools.cache
def compile_sp_rs_norm(
    H: int,
    tp: int,
    eps: float,
    i8: bool = True,
    E: int = 0,
    topk: int = 0,
    scale: float = 1.0,
    shared_w: float = 0.0,
):
    """i8: int8 + fp32 scale per 32 columns (~0.5% rel error); else MXFP8
    (E4M3 + E8M0, ~2%: too coarse for an attention output).

    E > 0: also the router of the own rows: once a 16-row tile of the normed
    rows is out, K slices of its logits (bf16 MFMA against Wg [E, H], all E
    experts) on whichever CTAs take them; the last slice of a tile sums them
    and runs the sigmoid + bias top-k of aiter's topk_gating (renormalized,
    times ``scale``) plus one shared expert (id E, weight shared_w)."""
    RT = E > 0
    assert not RT or (E % 64 == 0 and E // 64 == 2 and topk <= 64)
    assert H % 256 == 0
    NP = H // 8  # 8-column pieces of a row
    KS_DIVS = [d for d in range(KS_MAX, 1, -1) if (H // 128) % d == 0]
    # router: K steps (32 columns per wave) of a slice prefetched; a slice
    # has at least H / KS_MAX / 4 / 32 of them
    NPF = min(3, H // KS_MAX // 4 // 32)
    PIT = (NP + NTH - 1) // NTH
    assert NP % NTH == 0
    def fbits(v):
        return f"{struct.unpack('<I', struct.pack('<f', v))[0]:x}"

    name = f"sp_rs_norm_h{H}_tp{tp}_e{fbits(eps)}" + ("_i8" if i8 else "") + (
        f"_rt{E}k{topk}s{fbits(scale)}w{fbits(shared_w)}" if RT else ""
    )

    def i32(v):
        return fx.Int32(v)

    def uni(v):
        return i32(rocdl.readfirstlane(T.i32, _u(i32(v))))

    def uni64(v):
        v = fx.Int64(v)
        lo = uni(fx.Int32(v & fx.Int64(0xFFFFFFFF)))
        hi = uni(fx.Int32(v >> fx.Int64(32)))
        return (fx.Int64(hi) << fx.Int64(32)) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))

    def rsrc(addr):
        return buffer_ops.create_buffer_resource_from_addr(_u(uni64(addr)))

    def bld(rs, voff, ty, aux=0):
        return rocdl.raw_ptr_buffer_load(ty, rs, _u(i32(voff)), _u(i32(0)), aux=_attr(aux))

    def bst(val, rs, voff, aux=0):
        rocdl.raw_ptr_buffer_store(_u(val), rs, _u(i32(voff)), _u(i32(0)), aux=_attr(aux))

    def gptr(addr):
        return fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), fx.Int64(addr))

    def g_ld_sys(addr):
        return i32(
            fx.generic_load(gptr(addr), dtype=fx.Int32, memory_order=fx.AtomicOrdering.Monotonic,
                            syncscope="one-as", volatile=True)
        )

    def g_st_sys(addr, v):
        fx.generic_store(gptr(addr), i32(v), memory_order=fx.AtomicOrdering.Monotonic, syncscope="one-as")

    def asm(text):
        from flydsl._mlir.dialects import llvm as _llvm

        _llvm.inline_asm(None, [], text, "", has_side_effects=True)

    def now():
        from flydsl._mlir.dialects import llvm as _llvm

        return fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))

    def bf16x8(d):
        dv = fx.Vector(d)
        out = []
        for q in range_constexpr(4):
            w = fx.Int32(dv[q])
            out.append((w << i32(16)).bitcast(fx.Float32))
            out.append((w & i32(-65536)).bitcast(fx.Float32))
        return out

    def bf16_bits(f):
        return fx.Int32(
            fx.Vector.from_elements([fx.Float32(f).to(fx.BFloat16)], fx.BFloat16).bitcast(fx.Int16)[0]
        ) & i32(0xFFFF)

    def pack_bf16x8(acc):
        return fx.Vector.from_elements(
            [bf16_bits(acc[2 * q]) | (bf16_bits(acc[2 * q + 1]) << i32(16)) for q in range(4)], fx.Int32
        )

    def fp8x4_pack(f, qs):
        zero = fx.Vector.filled(2, 0, fx.Int16)
        w = rocdl.cvt_scalef32_pk_fp8_f32(T.vec(2, T.i16), _u(zero), _u(f[0]), _u(f[1]), _u(qs), False)
        w = rocdl.cvt_scalef32_pk_fp8_f32(T.vec(2, T.i16), w, _u(f[2]), _u(f[3]), _u(qs), True)
        return fx.Int32(fx.Vector(w).bitcast(fx.Int32)[0])

    def fp8x4_unpack(d, sc):
        out = []
        for sel in (False, True):
            v = fx.Vector(rocdl.cvt_scalef32_pk_f32_fp8(T.vec(2, T.f32), _u(i32(d)), _u(sc), sel))
            out += [fx.Float32(v[0]), fx.Float32(v[1])]
        return out

    def i8x4_pack(f, inv):
        w = i32(0)
        for k in range_constexpr(4):
            q = fx.Int32(fmath.roundeven(_u(f[k] * inv)))
            q = fx.max(fx.min(q, i32(127)), i32(-127))
            w = w | ((q & i32(0xFF)) << i32(8 * k))
        return w

    def i8x4_unpack(d, sc):
        out = []
        for k in range_constexpr(4):
            b = (fx.Int32(d) << i32(24 - 8 * k)) >> i32(24)  # sign-extended byte k
            out.append(b.to(fx.Float32) * sc)
        return out

    def e8_scale(e8):
        return ((fx.Int32(e8) & i32(0xFF)) << i32(23)).bitcast(fx.Float32)

    def wave_red(v, lane, op):
        for k in (1, 2, 4, 8, 16, 32):
            v = op(v, i32(rocdl.ds_bpermute(T.i32, _u((lane ^ i32(k)) * i32(4)), _u(v))))
        return uni(v)

    def fadd(x, y):
        return (x.bitcast(fx.Float32) + y.bitcast(fx.Float32)).bitcast(fx.Int32)

    LDSB = 128 + (NTH // 64) * 64 * (E // 16 if E else 1) * 16  # (router: the waves' partials)
    Shared = fx.struct(type("Shared", (), {"__annotations__": {"buf": fx.Array[fx.Int8, LDSB, 16]}}))

    def lds_f(L, off):
        return fx.Int32(
            fx.ptr_load(fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Shared, 4), L + off),
                        result_type=T.i32)
        ).bitcast(fx.Float32)

    def lds_stf(L, off, v):
        fx.ptr_store(_u(fx.Float32(v).bitcast(fx.Int32)),
                     fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Shared, 4), L + off))

    def zflag(a, rt, ks):
        return fx.Int64(a["ctrl"]) + fx.Int64((i32(5 * NB + ROWF) + rt * i32(KS_MAX) + ks) * i32(4))

    def rt_tiles(m):
        # the most K slices (a divisor of H / 128) with a CTA per task
        rt = m // i32(ZT)
        cap = i32(NB) // fx.max(rt, i32(1))
        ks = i32(1)
        for d in KS_DIVS:
            ks = ((ks == i32(1)) & (cap >= i32(d))).select(i32(d), ks)
        return rt, ks

    @traced
    def router_tasks(L, a, tid):
        # task t = (row tile rt, K slice k) on CTA NB - 1 - t % NB: the slice's partial
        # logits of the 16 rows for every expert (bf16 MFMA, the 4 waves each
        # a quarter of the slice, summed in LDS) into the scratch; the last
        # slice of a tile to land sums them all and routes the tile's rows
        m = a["m"]
        bid = i32(gpu.block_id("x"))
        lane = tid % i32(64)
        w = tid // i32(64)
        RT, KS = rt_tiles(m)
        kw = i32(H) // KS  # columns per slice (a multiple of 128)
        ro = rsrc(a["out"])
        rg = rsrc(a["wg"])
        rzp = rsrc(a["zp"])
        # (tasks from the last CTA down: at small m those only send, if that)
        for t_ in range(i32(NB - 1) - bid, RT * KS, i32(NB)):
            t = i32(t_)
            rt = t // KS
            ks = t - rt * KS
            row = a["rank"] * m + rt * i32(ZT) + lane % i32(16)
            k0 = ks * kw + w * (kw // i32(4))
            # the slice's first NPF K steps of the gate weight, in flight while
            # the tile's rows are reduced
            pf = []
            for j in range_constexpr(NPF):
                kc = k0 + i32(j * 32) + (lane // i32(16)) * i32(8)
                pf.append([
                    bld(rg, ((i32(n * 16) + lane % i32(16)) * i32(H) + kc) * i32(2), T.vec(4, T.i32))
                    for n in range(E // 16)
                ])
            # the tile's rows are out (row r: flag of its reducing CTA)
            if tid < i32(ZT):
                r = rt * i32(ZT) + tid
                addr = fx.Int64(a["ctrl"]) + fx.Int64((i32(5 * NB) + r) * i32(4))
                t0 = now()
                cur = g_ld_sys(addr)
                while (cur < a["epoch"]) & ((now() - t0) < fx.Int64(DEADLINE)):
                    rocdl.s_sleep(1)
                    cur = g_ld_sys(addr)
            gpu.barrier()

            def step(acc, kc, bvs):
                av = fx.Vector(bld(ro, (row * i32(H) + kc) * i32(2), T.vec(4, T.i32), AUX_SYS)).bitcast(fx.BFloat16)
                out_ = []
                for n in range_constexpr(E // 16):
                    bv = fx.Vector(bvs[n]).bitcast(fx.BFloat16)
                    out_.append(
                        fx.Vector(
                            rocdl.mfma_f32_16x16x32_bf16(
                                T.vec(4, T.f32), [_u(av), _u(bv), _u(acc[n]), 0, 0, 0]
                            )
                        )
                    )
                return out_

            acc0 = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(E // 16)]
            for j in range_constexpr(NPF):
                acc0 = step(acc0, k0 + i32(j * 32) + (lane // i32(16)) * i32(8), pf[j])
            for kk_, st in range(i32(NPF * 32), kw // i32(4), i32(32), init=acc0):
                kc = k0 + i32(kk_) + (lane // i32(16)) * i32(8)
                bvs = [
                    bld(rg, ((i32(n * 16) + lane % i32(16)) * i32(H) + kc) * i32(2), T.vec(4, T.i32))
                    for n in range(E // 16)
                ]
                res = yield step(list(st), kc, bvs)
            for n in range_constexpr(E // 16):
                for i in range_constexpr(4):
                    lds_stf(L, i32(128) + (((w * i32(E // 16) + i32(n)) * i32(64) + lane) * i32(4) + i32(i)) * i32(4), fx.Vector(res[n])[i])
            gpu.barrier()
            # wave 0 sums the waves; lane l, element i: row 4 (l // 16) + i,
            # expert 16 n + l % 16
            if w == i32(0):
                for n in range_constexpr(E // 16):
                    for i in range_constexpr(4):
                        v = fx.Float32(0.0)
                        for ww in range_constexpr(NTH // 64):
                            v = v + lds_f(L, i32(128) + ((i32((ww * (E // 16) + n) * 64) + lane) * i32(4) + i32(i)) * i32(4))
                        rl = (lane // i32(16)) * i32(4) + i32(i)
                        ex = i32(n * 16) + lane % i32(16)
                        zo = (((rt * i32(KS_MAX) + ks) * i32(ZT) + rl) * i32(E) + ex) * i32(4)
                        bst(v.bitcast(fx.Int32), rzp, zo, AUX_SYS)
                asm("s_waitcnt vmcnt(0)")
                if lane == i32(0):
                    g_st_sys(zflag(a, rt, ks), a["epoch"])
                # every slice of the tile landed: this slice's CTA routes rows
                # ks, ks + KS, ... of it
                if lane < KS:
                    t0 = now()
                    cur = g_ld_sys(zflag(a, rt, lane))
                    while (cur < a["epoch"]) & ((now() - t0) < fx.Int64(DEADLINE)):
                        rocdl.s_sleep(1)
                        cur = g_ld_sys(zflag(a, rt, lane))
            gpu.barrier()
            for rl_ in range(ks + w * KS, i32(ZT), KS * i32(NTH // 64)):
                router_row(a, lane, rt, i32(rl_), KS)
            gpu.barrier()

    @traced
    def router_row(a, lane, rt, rl, KS):
        # (one wave) row rl of tile rt: its logits (the slices' sum), top-k
        rzp = rsrc(a["zp"])
        rb = rsrc(a["bias"])
        r = rt * i32(ZT) + rl
        vals, orig, idxs = [], [], []
        for i in range_constexpr(2):
            e = lane + i32(i * 64)
            # every slice's partial in flight at once (slices >= KS: none)
            vs = []
            for k in range_constexpr(KS_MAX):
                kc = fx.min(i32(k), KS - i32(1))
                zo = (((rt * i32(KS_MAX) + kc) * i32(ZT) + rl) * i32(E) + e) * i32(4)
                vs.append((i32(k) < KS, fx.Int32(bld(rzp, zo, T.i32, AUX_SYS)).bitcast(fx.Float32)))
            x = fx.Float32(0.0)
            for live, v in vs:
                x = x + live.select(v, fx.Float32(0.0))
            # (the reference rounds the gate GEMM output to bf16)
            x = fx.Float32(x).to(fx.BFloat16).to(fx.Float32)
            sc = fx.Float32(1.0) / (fx.Float32(1.0) + fx.Float32(fmath.exp2(_u(x * fx.Float32(-1.4426950408889634)))))
            orig.append(sc)
            b = fx.Int32(bld(rb, e * i32(4), T.i32)).bitcast(fx.Float32)
            vals.append(sc + b)
            idxs.append(e)
        # thread-local sort, descending (stable: the lower expert first on ties)
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
            mx = wave_red(mv.bitcast(fx.Int32), lane, lambda p_, q_: p_.bitcast(fx.Float32).maximumf(q_.bitcast(fx.Float32)).bitcast(fx.Int32)).bitcast(fx.Float32)
            bal = fx.Int64(rocdl.ballot(T.i64, _u(mv == mx)))
            win = i32(fx.ctpop(fx.Int64((bal & (fx.Int64(0) - bal)) - fx.Int64(1))))
            win = (bal == fx.Int64(0)).select(i32(0), win)
            wid = i32(rocdl.readlane(T.i32, _u(mi), _u(win)))
            wgt = fx.Int32(rocdl.readlane(T.i32, _u(mo.bitcast(fx.Int32)), _u(win))).bitcast(fx.Float32)
            cur = cur + ((lane == win) & (cur < i32(2))).select(i32(1), i32(0))
            tot = tot + wgt
            my_id = (lane == i32(k)).select(wid, my_id)
            my_w = (lane == i32(k)).select(wgt, my_w)
        f = fx.Float32(float(scale)) / tot.maximumf(fx.Float32(1e-20))
        ri = rsrc(a["ids"])
        rw = rsrc(a["tw"])
        if lane < i32(topk):
            bst(my_id, ri, (r * i32(topk + 1) + lane) * i32(4))
            bst((my_w * f).bitcast(fx.Int32), rw, (r * i32(topk + 1) + lane) * i32(4))
        if lane == i32(topk):
            bst(i32(E), ri, (r * i32(topk + 1) + lane) * i32(4))
            bst(fx.Float32(float(shared_w)).bitcast(fx.Int32), rw, (r * i32(topk + 1) + lane) * i32(4))

    @traced
    def blk_sum(L, tid, v):
        r = wave_red(v.bitcast(fx.Int32), tid % i32(64), fadd)
        if ((tid % i32(64)) == i32(0)) & (tid < i32(NTH)):
            fx.ptr_store(_u(r), fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Shared, 4),
                                             L + i32(16) + (tid // i32(64)) * i32(4)))
        gpu.barrier()
        t = fx.Float32(0.0)
        for k in range_constexpr(NTH // 64):
            t = t + fx.Int32(
                fx.ptr_load(fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Shared, 4), L + i32(16 + k * 4)),
                            result_type=T.i32)
            ).bitcast(fx.Float32)
        gpu.barrier()
        return t

    @traced
    def send_rows(a, tid):
        # items (dest d != rank, row r) = d * m + r, CTA i % NB; a thread takes
        # pieces tid, tid + NTH, ... (8 columns each); 4 consecutive lanes hold
        # a 32-column block
        m = a["m"]
        bid = i32(gpu.block_id("x"))
        lane = tid % i32(64)
        rp = rsrc(a["part"])
        # items it = r * (tp - 1) + k: row r for peer rank + 1 + k; CTA it % NB
        # (consecutive items go to different peers: every link busy, every
        # CTA with work)
        for it_ in range(bid, m * i32(tp - 1), i32(NB)):
            it = i32(it_)
            r = it // i32(tp - 1)
            k = it - r * i32(tp - 1)
            d = (a["rank"] + i32(1) + k) % i32(tp)
            i = d * m + r
            if const_expr(True):
                peer = a["peer"][0]
                for j in range_constexpr(1, tp):
                    peer = (d == i32(j)).select(a["peer"][j], peer)
                rx = rsrc(fx.Int64(peer) + fx.Int64(a["off_x"]))
                rsx = rsrc(fx.Int64(peer) + fx.Int64(a["off_s"]))
                slot = a["rank"] * m + r
                for k in range_constexpr(PIT):
                    p = tid + i32(k * NTH)
                    f = bf16x8(bld(rp, (i * i32(H) + p * i32(8)) * i32(2), T.vec(4, T.i32)))
                    am = _fabs_f32(f[0])
                    for x in f[1:]:
                        am = am.maximumf(_fabs_f32(x))
                    am = am.maximumf(am.shuffle_xor(i32(1), i32(64)))
                    am = am.maximumf(am.shuffle_xor(i32(2), i32(64)))
                    if const_expr(i8):
                        sc = am.maximumf(fx.Float32(1e-30)) / fx.Float32(127.0)
                        inv = fx.Float32(1.0) / sc
                        dv = fx.Vector.from_elements([i8x4_pack(f[0:4], inv), i8x4_pack(f[4:8], inv)], fx.Int32)
                        bst(dv, rx, slot * i32(H) + p * i32(8), AUX_SYS)
                        if (lane & i32(3)) == i32(0):
                            bst(sc.bitcast(fx.Int32), rsx, (slot * i32(H // 32) + p // i32(4)) * i32(4), AUX_SYS)
                    else:
                        e8, qs = _e8m0_from_amax(am, max_norm=448.0)
                        dv = fx.Vector.from_elements([fp8x4_pack(f[0:4], qs), fp8x4_pack(f[4:8], qs)], fx.Int32)
                        bst(dv, rx, slot * i32(H) + p * i32(8), AUX_SYS)
                        if (lane & i32(3)) == i32(0):
                            bst(fx.Int8(fx.Int32(e8) & i32(0xFF)), rsx, slot * i32(H // 32) + p // i32(4), AUX_SYS)

    @traced
    def post_flags(L, a, tid):
        asm("s_waitcnt vmcnt(0)")
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
        # the CTA that sent row r to this rank, on every other rank
        if tid < i32(tp):
            # on rank tid, this rank is peer k = rank - tid - 1 (mod tp)
            k = (a["rank"] - tid - i32(1) + i32(tp)) % i32(tp)
            src_cta = (r * i32(tp - 1) + k) % i32(NB)
            addr = a["mine"] + fx.Int64(a["off_f"]) + fx.Int64((tid * i32(NB) + src_cta) * i32(4))
            if tid != a["rank"]:
                t0 = now()
                cur = g_ld_sys(addr)
                while (cur < a["epoch"]) & ((now() - t0) < fx.Int64(DEADLINE)):
                    rocdl.s_sleep(1)
                    cur = g_ld_sys(addr)
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
            act = tid < i32(NTH)  # (the router waves only join the barriers)
            for k in range_constexpr(PIT):
                p = fx.min(tid, i32(NTH - 1)) + i32(k * NTH)
                off = (g * i32(H) + p * i32(8)) * i32(2)
                f = bf16x8(bld(rp, off, T.vec(4, T.i32)))
                rv = bf16x8(bld(rr, off, T.vec(4, T.i32)))
                f = [x + y for x, y in zip(f, rv)]
                for s in range_constexpr(tp):
                    if s != 0 or True:
                        src = i32(s)
                        live = src != a["rank"]
                        slot = src * m + r
                        d = fx.Vector(bld(rx, slot * i32(H) + p * i32(8), T.vec(2, T.i32), AUX_SYS))
                        if const_expr(i8):
                            sc = fx.Int32(
                                bld(rsx, (slot * i32(H // 32) + p // i32(4)) * i32(4), T.i32, AUX_SYS)
                            ).bitcast(fx.Float32)
                            v = i8x4_unpack(d[0], sc) + i8x4_unpack(d[1], sc)
                        else:
                            sc = e8_scale(bld(rsx, slot * i32(H // 32) + p // i32(4), T.i8, AUX_SYS))
                            v = fp8x4_unpack(fx.Int32(d[0]), sc) + fp8x4_unpack(fx.Int32(d[1]), sc)
                        f = [x + live.select(y, fx.Float32(0.0)) for x, y in zip(f, v)]
                if act:
                    bst(pack_bf16x8(f), ro, off)
                for x in f:
                    ss = ss + act.select(x * x, fx.Float32(0.0))
                fs.append(f)
            tot = blk_sum(L, tid, ss)
            rcp = fx.Float32(fmath.rsqrt(_u(tot / fx.Float32(float(H)) + fx.Float32(float(eps)))))
            for k in range_constexpr(PIT):
                p = fx.min(tid, i32(NTH - 1)) + i32(k * NTH)
                off = (g * i32(H) + p * i32(8)) * i32(2)
                wv = bf16x8(bld(rw, p * i32(16), T.vec(4, T.i32)))
                if act:
                    bst(
                        pack_bf16x8([x * rcp * (w + fx.Float32(1.0)) for x, w in zip(fs[k], wv)]),
                        rout,
                        off,
                        AUX_SYS if RT else 0,
                    )
            if const_expr(RT):
                # the row is out (written through): the router may take it
                asm("s_waitcnt vmcnt(0)")
                gpu.barrier()
                if tid == i32(0):
                    g_st_sys(fx.Int64(a["ctrl"]) + fx.Int64((i32(5 * NB) + r) * i32(4)), a["epoch"])

    @flyc.kernel(name=name, known_block_size=[NALL, 1, 1])
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
        rank: fx.Int32,
        m: fx.Int32,
        mmax: fx.Int32,
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
            "part": part, "res": res, "res_out": res_out, "out": out, "w": w,
            "peer": peers, "mine": fx.Int64(mine), "off_x": off_x, "off_s": off_s,
            "off_f": off_f, "rank": rank, "m": m, "epoch": epoch, "ctrl": ctrl,
            "mmax": mmax, "wg": wg, "bias": bias, "ids": ids, "tw": tw,
            "zp": zp,
        }
        st = fx.Int64(ctrl) + fx.Int64((i32(NB) + bid * i32(4)) * i32(4))
        if tid < i32(4):
            # the subset barriers' counters
            fx.ptr_store(_u(i32(0)), fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Shared, 4), L + tid * i32(4)))
        gpu.barrier()
        if tid == i32(0):
            fx.generic_store(gptr(st), fx.Int32(now()))
        send_rows(a, tid)
        post_flags(L, a, tid)
        if tid == i32(0):
            fx.generic_store(gptr(st + fx.Int64(4)), fx.Int32(now()))
        reduce_rows(L, a, tid)
        if tid == i32(0):
            fx.generic_store(gptr(st + fx.Int64(8)), fx.Int32(now()))
        if const_expr(RT):
            router_tasks(L, a, tid)
        if tid == i32(0):
            fx.generic_store(gptr(ea), epoch)
            fx.generic_store(gptr(st + fx.Int64(12)), fx.Int32(now()))

    @flyc.jit
    def launch(
        part: fx.Int64, res: fx.Int64, res_out: fx.Int64, out: fx.Int64, w: fx.Int64,
        ctrl: fx.Int64, p0: fx.Int64, p1: fx.Int64, p2: fx.Int64, p3: fx.Int64,
        p4: fx.Int64, p5: fx.Int64, p6: fx.Int64, p7: fx.Int64,
        off_x: fx.Int64, off_s: fx.Int64, off_f: fx.Int64, rank: fx.Int32, m: fx.Int32,
        mmax: fx.Int32, wg: fx.Int64, bias: fx.Int64, ids: fx.Int64, tw: fx.Int64,
        zp: fx.Int64,
        stream: fx.Stream,
    ):
        sp_rs_norm_kernel(
            part, res, res_out, out, w, ctrl, p0, p1, p2, p3, p4, p5, p6, p7,
            off_x, off_s, off_f, rank, m, mmax, wg, bias, ids, tw, zp,
        ).launch(grid=(NB, 1, 1), block=(NALL, 1, 1), stream=stream)

    return launch


class SpRsNorm:
    """Per TP group and hidden size: ``forward(part, res, w)`` -> (out, res_out)
    for this rank's rows (bf16 [T, H] each, rows ``rank*m:(rank+1)*m`` valid).

    router = (E, topk, scale, shared_w): ``forward(..., router=(wg, bias, ids,
    tw))`` also routes the own rows (see compile_sp_rs_norm): wg = the gate
    weight bf16 [E, H], bias fp32 [E]; ids int32 / tw fp32 [m, topk + 1] (the
    last column the shared expert). Needs m % 16 == 0."""

    def __init__(
        self,
        hidden: int,
        max_tokens: int,
        eps: float,
        group=None,
        device=None,
        i8: bool = True,
        router=None,
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
        assert self.tp <= MAX_TP
        mmax = -(-int(max_tokens) // self.tp)
        self.mmax = mmax
        self.router = router
        E = int(router[0]) if router else 0
        assert not router or mmax // ZT <= NRT_MAX
        arena = SymmetricArena(group=group, device=device)
        self._x = arena.reserve("x", (self.tp * mmax * self.H,), torch.uint8)
        self._s = arena.reserve("s", (self.tp * mmax * self.H // 32 * 4,), torch.uint8)
        self._f = arena.reserve("f", (MAX_TP * NB,), torch.int32)
        arena.commit()
        self.arena = arena
        # ctrl: per-CTA epochs, [tl] stamps, router row flags, tile counters
        self.ctrl = torch.zeros(NB * 5 + ROWF + NRT_MAX * KS_MAX, dtype=torch.int32, device=device)
        self._zp = torch.empty(
            (NRT_MAX * KS_MAX * ZT * max(E, 1),) if router else (1,),
            dtype=torch.float32,
            device=device,
        )
        self._fn = compile_sp_rs_norm(self.H, self.tp, self.eps, bool(i8))
        self._fn_rt = (
            compile_sp_rs_norm(self.H, self.tp, self.eps, bool(i8), *[
                int(router[0]), int(router[1]), float(router[2]), float(router[3])
            ])
            if router
            else None
        )
        self._armed = set()

    def routes(self, tokens: int) -> bool:
        """forward(router=...) runs for this batch."""
        m = tokens // self.tp
        return self._fn_rt is not None and tokens % self.tp == 0 and m % ZT == 0 and 0 < m <= self.mmax

    def forward(self, part, res, w, out=None, res_out=None, router=None):
        Tt = part.shape[0]
        assert Tt % self.tp == 0 and Tt // self.tp <= self.mmax
        assert part.dtype == torch.bfloat16 and part.is_contiguous() and res.is_contiguous()
        m = Tt // self.tp
        out = torch.empty_like(part) if out is None else out
        res_out = torch.empty_like(res) if res_out is None else res_out
        peers = [int(b) for b in self.arena.base_ptrs] + [0] * (MAX_TP - self.tp)
        if router is not None:
            assert self.routes(Tt)
            wg, bias, ids, tw = router
            rt = (wg.data_ptr(), bias.data_ptr(), ids.data_ptr(), tw.data_ptr())
            fn = self._fn_rt
        else:
            rt = (0, 0, 0, 0)
            fn = self._fn
        args = (
            part.data_ptr(), res.data_ptr(), res_out.data_ptr(), out.data_ptr(), w.data_ptr(),
            self.ctrl.data_ptr(), *peers, self._x.offset, self._s.offset, self._f.offset,
            self.rank, m, self.mmax, *rt, self._zp.data_ptr(),
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
        """Collective: compile and run once (eagerly; graph capture cannot)."""
        tp = self.tp
        rows = tp * ZT if self._fn_rt is not None else tp
        x = torch.zeros((rows, self.H), dtype=torch.bfloat16, device=self.device)
        w = torch.zeros((self.H,), dtype=torch.bfloat16, device=self.device)
        self.forward(x, x.clone(), w)
        if self._fn_rt is not None:
            E, k = int(self.router[0]), int(self.router[1])
            wg = torch.zeros((E, self.H), dtype=torch.bfloat16, device=self.device)
            bias = torch.zeros((E,), dtype=torch.float32, device=self.device)
            ids = torch.empty((ZT, k + 1), dtype=torch.int32, device=self.device)
            tw = torch.empty((ZT, k + 1), dtype=torch.float32, device=self.device)
            self.forward(x, x.clone(), w, router=(wg, bias, ids, tw))
        torch.cuda.synchronize(self.device)

    __call__ = forward
