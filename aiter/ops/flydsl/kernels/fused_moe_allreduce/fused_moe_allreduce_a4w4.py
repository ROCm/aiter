# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .. import buffer_ops

__all__ = [
    "DN_SC_BLK",
    "DN_SC_EXPERT",
    "ERR_AR",
    "ERR_MIDS",
    "ERR_SCORES",
    "GRID",
    "HIDDEN",
    "INTER",
    "MAX_PES",
    "MX_BLOCK",
    "NUM_EXPERTS",
    "RT_CH",
    "RT_WAVES",
    "SLOTS",
    "SUPPORTED_PROTOS",
    "SUPPORTED_SAMPLES",
    "SYM_BYTES",
    "TIMELINE_SLOTS",
    "TOP_K",
    "UG_SC_EXPERT",
    "UG_SC_TILE",
    "compile_fused_moe_allreduce_a4w4",
]

HIDDEN = 6144
NUM_EXPERTS = 256
TOP_K = 8
SLOTS = TOP_K + 1
INTER = 256
GRID = 256
NT = 512
NWAVE = NT // 64
MAX_PES = 8
SUPPORTED_SAMPLES = (1, 2, 4)
SUPPORTED_PROTOS = (0, 1)
MX_BLOCK = 32

KBLK = HIDDEN // 128
KB_BYTES = 64 * 16
RT_WAVES = NWAVE - 1
RT_CH = (KBLK + RT_WAVES - 1) // RT_WAVES
UG_TILE_BYTES = KBLK * KB_BYTES
UG_EXPERT_BYTES = (2 * INTER // 16) * UG_TILE_BYTES
UG_SC_TILE = RT_WAVES * 64 * 8
UG_SC_EXPERT = (2 * INTER // 16) * UG_SC_TILE
DN_BLK_BYTES = 3 * KB_BYTES
DN_EXPERT_BYTES = (HIDDEN // 24) * DN_BLK_BYTES
DN_SC_BLK = 64 * 4
DN_SC_EXPERT = (HIDDEN // 24) * DN_SC_BLK
ASC_G_BYTES = RT_WAVES * 8
ROUTER_BLK_BYTES = 8 * HIDDEN * 2
ROUTER_WAVE_KC = HIDDEN // 64 // NWAVE
SCORE_SAMPLE_BYTES = 32 * 32 * 4
SCORE_LINE_BYTES = 64
MIDP_SAMPLE_I32 = SLOTS * INTER
WIRE_BYTES = {0: 96, 1: 64}
SYM_BYTES = (MAX_PES - 1) * 2 * GRID * WIRE_BYTES[0] * max(SUPPORTED_SAMPLES)

TIMELINE_SLOTS = 16
POLL_LIMIT = 1 << 22
ERR_SCORES, ERR_MIDS, ERR_AR = 1, 2, 4
LDS_MAX = 160 * 1024

ROUTE_SCALE = 2.5
LOG2E = 1.4426950408889634
RMS_EPS = 1e-5
INV_FP4_MAX_BITS = 0x3E2AAAAB
E8M0_ONE = 127

AUX_NT = 2
AUX_SC1 = 16
AUX_SYS = 1 | 16

ROW_SHR = (0x111, 0x112, 0x114, 0x118)
ROW_BCAST15, ROW_BCAST31 = 0x142, 0x143
ROW_ROR = {1: 0x121, 2: 0x122, 4: 0x124, 8: 0x128}
QUAD_XOR1, QUAD_XOR2, ROW_HALF_MIRROR = 0xB1, 0x4E, 0x141

traced = ASTRewriter.transform


def _u(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _attr(v):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), int(v))


@functools.cache
def fused_moe_allreduce_a4w4_consts(samples: int, proto: int) -> dict:
    S = samples
    c = {"S": S, "BPS": GRID // S, "G": GRID // S // 32, "ROWS": 24 * S}
    c["N_ITEMS"] = SLOTS * S * 3
    c["ITEMS_PER_WAVE"] = (c["N_ITEMS"] + NWAVE - 1) // NWAVE
    c["WIRE"] = WIRE_BYTES[proto] * S
    off = 0

    def take(n, align=16):
        nonlocal off
        off = (off + align - 1) // align * align
        start = off
        off += n
        return start

    c["L_NORM"] = take(HIDDEN * 2)
    c["L_ACT"] = take(HIDDEN // 2 + 64)
    c["L_ASC"] = take(4 * ASC_G_BYTES)
    c["L_SSQ"] = take(NWAVE * 4)
    c["L_REDR"] = take(NWAVE * 16 * 4)
    c["L_REDT"] = take((1 + S) * NWAVE * 16 * 4)
    c["L_MIDT"] = take((1 + S) * 16)
    c["L_TIDX"] = take(TOP_K * 4)
    c["L_TPRB"] = take(TOP_K * 4)
    c["L_SIG"] = take(NUM_EXPERTS * 4)
    c["L_MID4"] = take(SLOTS * INTER // 2)
    c["L_MSC"] = take(SLOTS * INTER // MX_BLOCK)
    c["L_P"] = take(SLOTS * S * 2 * 24 * 4)
    c["L_PART"] = take(c["ROWS"] * 2)
    c["L_RECV"] = take((MAX_PES - 1) * c["ROWS"] * 2)
    c["LDS_BYTES"] = (off + 127) // 128 * 128
    assert GRID % (32 * S) == 0 and 8 % c["G"] == 0
    assert c["LDS_BYTES"] <= LDS_MAX
    return c


@functools.cache
def compile_fused_moe_allreduce_a4w4(
    samples: int, proto: int = 0, timeline: bool = False, device: int = 0
):
    if samples not in SUPPORTED_SAMPLES:
        raise ValueError(f"samples must be one of {SUPPORTED_SAMPLES}, got {samples}")
    if proto not in SUPPORTED_PROTOS:
        raise ValueError(f"proto must be one of {SUPPORTED_PROTOS}, got {proto}")
    c = fused_moe_allreduce_a4w4_consts(samples, proto)
    S, BPS, G, ROWS = c["S"], c["BPS"], c["G"], c["ROWS"]
    N_ITEMS, IPW, WIRE = c["N_ITEMS"], c["ITEMS_PER_WAVE"], c["WIRE"]
    L_NORM, L_ACT, L_ASC, L_SSQ = c["L_NORM"], c["L_ACT"], c["L_ASC"], c["L_SSQ"]
    L_REDR, L_REDT, L_MIDT = c["L_REDR"], c["L_REDT"], c["L_MIDT"]
    L_TIDX, L_TPRB, L_MID4, L_MSC = c["L_TIDX"], c["L_TPRB"], c["L_MID4"], c["L_MSC"]
    L_SIG = c["L_SIG"]
    L_P, L_PART, L_RECV, LDS_BYTES = c["L_P"], c["L_PART"], c["L_RECV"], c["LDS_BYTES"]
    PKTS = 6 * S
    LINE_LANES = 8 * S
    name = f"fused_moe_allreduce_a4w4_s{S}_p{proto}" + ("_tl" if timeline else "")
    const_expr = fx.const_expr

    def i32(v):
        return fx.Int32(v)

    def i64(v):
        return fx.Int64(v)

    def f32(v):
        return fx.Float32(v)

    def uni(v):
        return i32(rocdl.readfirstlane(T.i32, _u(i32(v))))

    def uni64(v):
        v = i64(v)
        lo = uni(i32(v & i64(0xFFFFFFFF)))
        hi = uni(i32(v >> i64(32)))
        return (i64(hi) << i64(32)) | (i64(lo) & i64(0xFFFFFFFF))

    def rsrc(addr, nbytes=None):
        if nbytes is None:
            return buffer_ops.create_buffer_resource_from_addr(_u(uni64(addr)))
        return buffer_ops.create_buffer_resource_from_addr(
            _u(uni64(addr)), num_records_bytes=_u(i64(uni(nbytes)))
        )

    def bld(rs, voff, ty, aux=0, soff=0):
        return rocdl.raw_ptr_buffer_load(
            ty, rs, _u(i32(voff)), _u(i32(soff)), aux=_attr(aux)
        )

    def bst(val, rs, voff, aux=0, soff=0):
        rocdl.raw_ptr_buffer_store(
            _u(val), rs, _u(i32(voff)), _u(i32(soff)), aux=_attr(aux)
        )

    def ld_v4i(rs, voff, aux=0):
        return fx.Vector(bld(rs, voff, T.vec(4, T.i32), aux))

    def ld_f(rs, voff):
        return f32(bld(rs, voff, T.f32))

    def lds_ptr(base, off, ty, align):
        if isinstance(ty, ir.VectorType):
            ty = ty.element_type
        return fx.inttoptr(
            fx.PointerType.get(ty, fx.AddressSpace.Shared, align), base + i32(off)
        )

    def lds_ld(base, off, ty, align=4):
        return fx.ptr_load(lds_ptr(base, off, ty, align), result_type=ty)

    def lds_st(base, off, val, align=4):
        v = _u(val)
        fx.ptr_store(v, lds_ptr(base, off, v.type, align))

    def lds_i_vol(base, off):
        p3 = _llvm.IntToPtrOp(
            _llvm.PointerType.get(address_space=3), _u(i32(base) + i32(off))
        ).result
        return i32(
            _llvm.LoadOp(
                T.i32,
                p3,
                alignment=4,
                ordering=_llvm.AtomicOrdering.monotonic,
                syncscope="workgroup",
            ).res
        )

    def lds_f(base, off):
        return f32(lds_ld(base, off, T.f32))

    def lds_i(base, off):
        return i32(lds_ld(base, off, T.i32))

    def lds_u8(base, off):
        return i32(lds_ld(base, off, T.i8, 1)) & i32(0xFF)

    def gptr(addr):
        return _llvm.IntToPtrOp(
            _llvm.PointerType.get(address_space=1), _u(i64(addr))
        ).result

    def ld_agent_i64(addr):
        return i64(
            _llvm.LoadOp(
                T.i64,
                gptr(addr),
                alignment=8,
                ordering=_llvm.AtomicOrdering.monotonic,
                syncscope="agent",
            ).res
        )

    def ld_agent_i32(addr):
        return i32(
            _llvm.LoadOp(
                T.i32,
                gptr(addr),
                alignment=4,
                ordering=_llvm.AtomicOrdering.monotonic,
                syncscope="agent",
            ).res
        )

    def ld_sys_i64(addr):
        return i64(
            _llvm.LoadOp(
                T.i64,
                gptr(addr),
                alignment=8,
                ordering=_llvm.AtomicOrdering.monotonic,
                syncscope="one-as",
            ).res
        )

    def ld_sys_v4i(addr):
        return fx.Vector(
            _llvm.LoadOp(
                T.vec(4, T.i32), gptr(addr), alignment=16, volatile_=True
            ).res
        )

    def st_agent_i32(addr, v):
        _llvm.StoreOp(
            _u(i32(v)),
            gptr(addr),
            alignment=4,
            ordering=_llvm.AtomicOrdering.monotonic,
            syncscope="agent",
        )

    def atomic_or_agent(addr, v):
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp._or,
            gptr(addr),
            _u(i32(v)),
            _llvm.AtomicOrdering.monotonic,
            syncscope="agent",
        )

    def ballot(pred):
        return i64(rocdl.ballot(T.i64, _u(pred)))

    def readlane_i(v, lane):
        return i32(rocdl.readlane(T.i32, _u(i32(v)), _u(i32(lane))))

    def dpp(src, ctrl):
        return i32(
            _llvm.call_intrinsic(
                T.i32,
                "llvm.amdgcn.update.dpp.i32",
                [
                    _u(i32(0)),
                    _u(i32(src)),
                    _u(i32(ctrl)),
                    _u(i32(0xF)),
                    _u(i32(0xF)),
                    _u(fx.Boolean(True)),
                ],
                [],
                [],
            )
        )

    def dpp_f(x, ctrl):
        return dpp(f32(x).bitcast(fx.Int32), ctrl).bitcast(fx.Float32)

    def wave_sum_f32(x):
        x = f32(x)
        for ctl in ROW_SHR + (ROW_BCAST15, ROW_BCAST31):
            x = x + dpp_f(x, ctl)
        return f32(rocdl.readlane(T.f32, _u(x), _u(i32(63))))

    def wave_max_i32(x):
        x = i32(x)
        for ctl in ROW_SHR + (ROW_BCAST15, ROW_BCAST31):
            x = fx.max(x, dpp(x, ctl))
        return readlane_i(x, 63)

    def wave_min_i32(x):
        x = i32(x)
        for ctl in ROW_SHR + (ROW_BCAST15, ROW_BCAST31):
            x = fx.min(x, dpp(x, ctl))
        return readlane_i(x, 63)

    def oct_max_pos(x):
        v = f32(x).bitcast(fx.Int32)
        for ctl in (QUAD_XOR1, QUAD_XOR2, ROW_HALF_MIRROR):
            v = fx.max(v, dpp(v, ctl))
        return v.bitcast(fx.Float32)

    def exp2_raw(x):
        return f32(
            _llvm.call_intrinsic(T.f32, "llvm.amdgcn.exp2.f32", [_u(f32(x))], [], [])
        )

    def sigmoid(x):
        return f32(1.0) / (f32(1.0) + exp2_raw(f32(x) * f32(-LOG2E)))

    def bf16_lo(w):
        return (i32(w) << i32(16)).bitcast(fx.Float32)

    def bf16_hi(w):
        return (i32(w) & i32(-65536)).bitcast(fx.Float32)

    def bf16_bits(f):
        return i32(
            fx.Vector.from_elements([f32(f).to(fx.BFloat16)], fx.BFloat16).bitcast(
                fx.Int16
            )[0]
        ) & i32(0xFFFF)

    def bf16_round(f):
        return bf16_lo(bf16_bits(f))

    def _absf(x):
        return (f32(x).bitcast(fx.Int32) & i32(0x7FFFFFFF)).bitcast(fx.Float32)

    def mx_scale(amax):
        w = (f32(amax) * i32(INV_FP4_MAX_BITS).bitcast(fx.Float32)).bitcast(fx.Int32)
        e = (w.shrui(i32(23)) & i32(0xFF)) + ((w & i32(0x7FFFFF)) != i32(0)).select(
            i32(1), i32(0)
        )
        e = fx.min(e, i32(254))
        inv = ((i32(254) - e) << i32(23)).bitcast(fx.Float32)
        return e, inv

    def fp4x4(v, inv):
        one = _u(f32(1.0))
        w = rocdl.cvt_scalef32_pk_fp4_f32(
            T.i32, _u(i32(0)), _u(v[0] * inv), _u(v[1] * inv), one, 0
        )
        w = rocdl.cvt_scalef32_pk_fp4_f32(
            T.i32, w, _u(v[2] * inv), _u(v[3] * inv), one, 1
        )
        return i32(w) & i32(0xFFFF)

    def mfma_fp4(a4, b4, acc, sa, opa, sb, opb):
        return fx.Vector(
            rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                T.vec(4, T.f32),
                [_u(a4), _u(b4), _u(acc), 4, 4, opa, _u(i32(sa)), opb, _u(i32(sb))],
            )
        )

    def fdot2(a, b, acc):
        av = fx.Vector.from_elements([i32(a)], fx.Int32).bitcast(fx.BFloat16)
        bv = fx.Vector.from_elements([i32(b)], fx.Int32).bitcast(fx.BFloat16)
        return f32(rocdl.fdot2_f32_bf16_(T.f32, _u(av), _u(bv), _u(f32(acc)), clamp=False))

    def norm(L, wave, lane, xv, gv):
        xs = []
        for j in range_constexpr(3):
            for q in range_constexpr(2):
                w = i32(xv[j][q])
                xs += [bf16_lo(w), bf16_hi(w)]
        ss = xs[0] * xs[0]
        for k in range_constexpr(1, 12):
            ss = ss + xs[k] * xs[k]
        wsum = wave_sum_f32(ss)
        _store_if_lane0(L, lane, L_SSQ + wave * i32(4), wsum)
        _topk_clear(L, lane, wave)
        gpu.barrier()
        tot = f32(0.0)
        for w in range_constexpr(NWAVE):
            tot = tot + lds_f(L, L_SSQ + w * 4)
        rms = fx.math.rsqrt(tot / f32(float(HIDDEN)) + f32(RMS_EPS))
        e0 = wave * i32(768) + lane * i32(4)
        nvs = []
        for j in range_constexpr(3):
            nv = []
            for q in range_constexpr(4):
                g = f32(fx.Vector(gv[j])[q])
                nv.append(bf16_round((g * xs[j * 4 + q]) * rms))
            e = e0 + i32(j * 256)
            lds_st(
                L,
                L_NORM + e * i32(2),
                fx.Vector.from_elements(
                    [
                        (nv[0].bitcast(fx.Int32).shrui(i32(16)))
                        | (nv[1].bitcast(fx.Int32) & i32(-65536)),
                        (nv[2].bitcast(fx.Int32).shrui(i32(16)))
                        | (nv[3].bitcast(fx.Int32) & i32(-65536)),
                    ],
                    fx.Int32,
                ),
                align=8,
            )
            nvs.append(nv)
        gpu.barrier()
        return nvs

    def act_quant(L, wave, lane, nvs):
        for j in range_constexpr(3):
            nv = nvs[j]
            am = _absf(nv[0]).maximumf(_absf(nv[1]))
            am = am.maximumf(_absf(nv[2])).maximumf(_absf(nv[3]))
            am = oct_max_pos(am)
            e, inv = mx_scale(am)
            lds_st(
                L,
                L_ACT + wave * i32(384) + i32(j * 128) + lane * i32(2),
                fx.Int16(fp4x4(nv, inv)),
                align=2,
            )
            kgrp = wave * i32(24) + i32(j * 8) + (lane >> i32(3))
            kb = kgrp >> i32(2)
            off = (kgrp & i32(3)) * i32(ASC_G_BYTES) + (kb // i32(RT_CH)) * i32(8) + kb % i32(RT_CH)
            _store_scale_byte(L, (lane & i32(7)) == i32(0), L_ASC + off, e)
        _asc_pad(L, wave, lane)

    @traced
    def _store_scale_byte(L, pred, off, e):
        if pred:
            lds_st(L, off, fx.Int8(e), align=1)

    @traced
    def _asc_pad(L, wave, lane):
        if (wave == i32(0)) & (lane < i32(4)):
            lds_st(
                L,
                L_ASC + lane * i32(ASC_G_BYTES) + i32((RT_WAVES - 1) * 8 + RT_CH - 1),
                fx.Int8(i32(E8M0_ONE)),
                align=1,
            )

    @traced
    def _topk_clear(L, lane, wave):
        if (wave == i32(0)) & (lane < i32(TOP_K)):
            lds_st(L, L_TIDX + lane * i32(4), i32(-1))

    @traced
    def _store_if_lane0(L, lane, off, v):
        if lane == i32(0):
            lds_st(L, off, v)

    def router_dot(L, wave, lane, wr):
        kgo = ((lane >> i32(2)) & i32(7)) * i32(4)
        acc = f32(0.0)
        for kc in range_constexpr(ROUTER_WAVE_KC):
            kbase = (wave * i32(ROUTER_WAVE_KC) + i32(kc)) * i32(64) + kgo
            for sh in range_constexpr(2):
                nv = fx.Vector(
                    lds_ld(L, L_NORM + (kbase + i32(sh * 32)) * i32(2), T.vec(2, T.i32), 8)
                )
                acc = fdot2(wr[kc][2 * sh], nv[0], acc)
                acc = fdot2(wr[kc][2 * sh + 1], nv[1], acc)
        acc = acc + dpp_f(acc, ROW_ROR[4])
        acc = acc + dpp_f(acc, ROW_ROR[8])
        return acc

    @traced
    def router_publish(L, a, tid, wave, lane, acc, s, lb, tag):
        if (lane & i32(12)) == i32(0):
            idx = wave * i32(16) + (lane >> i32(4)) * i32(4) + (lane & i32(3))
            lds_st(L, L_REDR + idx * i32(4), acc)
        gpu.barrier()
        if (lb < i32(32)) & (tid < i32(8)):
            r = tid
            logit = f32(0.0)
            for w in range_constexpr(NWAVE):
                for h in range_constexpr(2):
                    idx = i32(w * 16) + ((r >> i32(2)) * i32(2) + i32(h)) * i32(4) + (r & i32(3))
                    logit = logit + lds_f(L, L_REDR + idx * i32(4))
            e = lb * i32(8) + r
            bst(logit, rsrc(a["scores"]), (s * i32(NUM_EXPERTS) + e) * i32(4))
            sig = sigmoid(logit)
            pair = fx.Vector.from_elements([sig.bitcast(fx.Int32), tag], fx.Int32)
            bst(
                pair,
                rsrc(a["score_lines"]),
                s * i32(SCORE_SAMPLE_BYTES) + lb * i32(SCORE_LINE_BYTES) + r * i32(8),
                AUX_SC1,
            )
        if (lb < i32(32)) & (tid < i32(24)):
            v = lds_ld(L, L_NORM + (lb * i32(192) + tid * i32(8)) * i32(2), T.vec(4, T.i32), 16)
            bst(
                v,
                rsrc(a["norm_out"]),
                (s * i32(HIDDEN) + lb * i32(192) + tid * i32(8)) * i32(2),
            )

    def ug_issue(a, e, t, wave, lane, nbytes, sbytes):
        base = i64(a["ug_w"]) + i64(e) * i64(UG_EXPERT_BYTES) + i64(t) * i64(UG_TILE_BYTES)
        rs = rsrc(base, nbytes)
        cb = (wave - i32(1)) * i32(RT_CH)
        ws = [ld_v4i(rs, (cb + i32(ch)) * i32(KB_BYTES) + lane * i32(16), AUX_NT) for ch in range(RT_CH)]
        sbase = i64(a["ug_scales"]) + i64(e) * i64(UG_SC_EXPERT) + i64(t) * i64(UG_SC_TILE)
        sv = fx.Vector(
            bld(rsrc(sbase, sbytes), ((wave - i32(1)) * i32(64) + lane) * i32(8), T.vec(2, T.i32))
        )
        return ws, sv

    def ug_mfma(L, wave, lane, ws, sv, red):
        cb = (wave - i32(1)) * i32(RT_CH)
        g = lane >> i32(4)
        asc = fx.Vector(
            lds_ld(L, L_ASC + g * i32(ASC_G_BYTES) + (wave - i32(1)) * i32(8), T.vec(2, T.i32), 8)
        )
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for ch in range_constexpr(RT_CH):
            kb = fx.min(cb + i32(ch), i32(KBLK - 1))
            b4 = lds_ld(L, L_ACT + kb * i32(64) + g * i32(16), T.vec(4, T.i32), 16)
            acc = mfma_fp4(ws[ch], b4, acc, sv[ch // 4], ch % 4, asc[ch // 4], ch % 4)
        _store_tile_red(L, lane, red + (wave * i32(16) + (lane >> i32(4)) * i32(4)) * i32(4), acc)

    @traced
    def shared_mfma(L, wave, lane, ws, sv):
        if wave != i32(0):
            ug_mfma(L, wave, lane, ws, sv, L_REDT)

    @traced
    def _store_tile_red(L, lane, off, acc):
        if (lane & i32(15)) == i32(0):
            lds_st(L, off, acc, align=16)

    @traced
    def tile_finish(L, a, tid, red, midt, s, slot, t, tag, active):
        if tid < i32(8):
            g = f32(0.0)
            u = f32(0.0)
            for w in range_constexpr(1, NWAVE):
                g = g + lds_f(L, red + (i32(w * 16) + tid) * i32(4))
                u = u + lds_f(L, red + (i32(w * 16 + 8) + tid) * i32(4))
            sg = sigmoid(g)
            m = (g * sg) * u
            lds_st(L, midt + tid * i32(2), fx.Int16(bf16_bits(m)), align=2)
        if active & (tid < i32(4)):
            d = lds_i(L, midt + tid * i32(4))
            pair = fx.Vector.from_elements([d, tag], fx.Int32)
            base_i32 = s * i32(MIDP_SAMPLE_I32) + slot * i32(INTER) + t * i32(8) + tid * i32(2)
            bst(pair, rsrc(a["mid_pairs"]), base_i32 * i32(4), AUX_SC1)
            if i64(a["hidden_mid"]) != i64(0):
                bst(
                    d,
                    rsrc(a["hidden_mid"]),
                    (s * i32(MIDP_SAMPLE_I32) + slot * i32(INTER) + t * i32(8) + tid * i32(2)) * i32(2),
                )

    def lane_sort(keys):
        k = list(keys)
        j = [i32(q) for q in range(4)]
        for x, y in ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2)):
            sw = (k[y] > k[x]) | ((k[y] == k[x]) & (j[y] < j[x]))
            k[x], k[y] = sw.select(k[y], k[x]), sw.select(k[x], k[y])
            j[x], j[y] = sw.select(j[y], j[x]), sw.select(j[x], j[y])
        return k, j[0] | (j[1] << i32(2)) | (j[2] << i32(4)) | (j[3] << i32(6))

    @traced
    def rank_round(k, jpack, lane):
        m = wave_max_i32(k[0])
        b = ballot(k[0] == m)
        win = uni(i32(fx.math.cttz(b)))
        if i32(fx.math.ctpop(b)) > i32(1):
            c = (k[0] == m).select(((jpack & i32(3)) << i32(6)) | lane, i32(1 << 20))
            win = wave_min_i32(c) & i32(63)
        idx = ((readlane_i(jpack, win) & i32(3)) << i32(6)) | win
        hit = lane == win
        k = [
            hit.select(k[1], k[0]),
            hit.select(k[2], k[1]),
            hit.select(k[3], k[2]),
            hit.select(i32(-(2**31)), k[3]),
        ]
        jpack = hit.select(jpack.shrui(i32(2)), jpack)
        return k, jpack, idx

    @traced
    def _lane0_st(L, lane, off, v):
        if lane == i32(0):
            lds_st(L, off, v)

    @traced
    def topk_rank(L, a, lane, wave, s, lb, tag, tl, tid, bid):
        if wave == i32(0):
            mark(tl, tid, bid, 10)
            brs = rsrc(a["bias"])
            bias = [ld_f(brs, (lane + i32(64 * j)) * i32(4)) for j in range(4)]
            base = i64(a["score_lines"]) + i64(s * i32(SCORE_SAMPLE_BYTES)) + i64(lane * i32(8))
            v0 = ld_agent_i64(base)
            v1 = ld_agent_i64(base + i64(512))
            v2 = ld_agent_i64(base + i64(1024))
            v3 = ld_agent_i64(base + i64(1536))
            bad = ballot(
                _tag_bad(v0, tag) | _tag_bad(v1, tag) | _tag_bad(v2, tag) | _tag_bad(v3, tag)
            )
            n = i32(0)
            while (bad != i64(0)) & (n < i32(POLL_LIMIT)):
                v0 = ld_agent_i64(base)
                v1 = ld_agent_i64(base + i64(512))
                v2 = ld_agent_i64(base + i64(1024))
                v3 = ld_agent_i64(base + i64(1536))
                bad = ballot(
                    _tag_bad(v0, tag) | _tag_bad(v1, tag) | _tag_bad(v2, tag) | _tag_bad(v3, tag)
                )
                n = n + i32(1)
            _poll_report(a, lane, bid, bad != i64(0), ERR_SCORES)
            mark(tl, tid, bid, 11)
            sig = [i32(v & i64(0xFFFFFFFF)).bitcast(fx.Float32) for v in (v0, v1, v2, v3)]
            keys, jpack = lane_sort([_order_key(sig[j] + bias[j]) for j in range(4)])
            for j in range_constexpr(4):
                lds_st(L, L_SIG + (lane + i32(64 * j)) * i32(4), sig[j])
            idxs = []
            for it in range_constexpr(TOP_K):
                keys, jpack, idx = rank_round(keys, jpack, lane)
                _lane0_st(L, lane, L_TIDX + i32(it * 4), idx)
                idxs.append(idx)
            mark(tl, tid, bid, 12)
            scs = [lds_f(L, L_SIG + idxs[k] * i32(4)) for k in range(TOP_K)]
            tot = f32(0.0)
            for k in range_constexpr(TOP_K):
                tot = tot + scs[k]
            fac = f32(ROUTE_SCALE) / tot
            my_prb = scs[0] * fac
            for k in range_constexpr(1, TOP_K):
                my_prb = (lane == i32(k)).select(scs[k] * fac, my_prb)
            li = fx.min(lane, i32(TOP_K - 1))
            _topk_store(L, a, lane, s, lb, lds_i(L, L_TIDX + li * i32(4)), my_prb)

    @traced
    def routed_tiles(L, a, wave, lane, grp, tile, tl, tid, bid):
        if wave != i32(0):
            pend = None
            for m in range_constexpr(S):
                if m == 0:
                    pend = routed_wait_issue(L, a, wave, lane, grp, tile, 0)
                cur = pend
                if m + 1 < S:
                    pend = routed_wait_issue(L, a, wave, lane, grp, tile, m + 1)
                if m == 0:
                    mark(tl, tid, bid, 13, 64)
                ws, sv = cur
                ug_mfma(L, wave, lane, ws, sv, L_REDT + (1 + m) * NWAVE * 64)
            mark(tl, tid, bid, 14, 64)

    def routed_wait_issue(L, a, wave, lane, grp, tile, m):
        r = (grp ^ i32(1)) + i32(G * m)
        e = _spin_rank(L, r)
        t = ug_issue(a, e + i32(1), tile, wave, lane, i32(UG_TILE_BYTES), i32(UG_SC_TILE))
        rocdl.sched_barrier(0)
        return t

    @traced
    def _spin_rank(L, r):
        e = lds_i_vol(L, L_TIDX + r * i32(4))
        while e < i32(0):
            rocdl.s_sleep(1)
            e = lds_i_vol(L, L_TIDX + r * i32(4))
        return e

    @traced
    def _poll_report(a, lane, bid, timed_out, code):
        if timed_out & (lane == i32(0)):
            atomic_or_agent(i64(a["flags"]) + i64((i32(GRID) + bid) * i32(4)), code)

    def _tag_bad(v, tag):
        return i32(v >> i64(32)) != tag

    def _order_key(x):
        b = f32(x).bitcast(fx.Int32)
        return (b < i32(0)).select(b ^ i32(0x7FFFFFFF), b)

    @traced
    def _topk_store(L, a, lane, s, lb, my_idx, my_prb):
        if lane < i32(TOP_K):
            lds_st(L, L_TPRB + lane * i32(4), my_prb)
            if lb == i32(0):
                o = (s * i32(TOP_K) + lane) * i32(4)
                bst(my_prb, rsrc(a["probs_out"]), o)
                bst(my_idx, rsrc(a["indices_out"]), o)

    def slot_expert(L, j):
        jm = fx.max(j - i32(1), i32(0))
        return (j == i32(0)).select(i32(0), lds_i(L, L_TIDX + jm * i32(4)) + i32(1))

    def down_issue(L, a, wave, lane, lb):
        items = []
        for k in range_constexpr(IPW):
            it = wave + i32(NWAVE * k)
            valid = it < i32(N_ITEMS)
            itc = fx.min(it, i32(N_ITEMS - 1))
            part = itc % i32(3)
            jq = itc // i32(3)
            j = jq // i32(S)
            q = jq % i32(S)
            e = slot_expert(L, j)
            blk = lb * i32(S) + q
            rs = rsrc(i64(a["down_w"]) + i64(e) * i64(DN_EXPERT_BYTES) + i64(blk) * i64(DN_BLK_BYTES))
            ws = ld_v4i(rs, part * i32(KB_BYTES) + lane * i32(16), AUX_NT)
            srs = rsrc(i64(a["down_scales"]) + i64(e) * i64(DN_SC_EXPERT) + i64(blk) * i64(DN_SC_BLK))
            sd = i32(bld(srs, lane * i32(4), T.i32))
            items.append((valid, part, j, q, ws, sd))
        return items

    @traced
    def mid_slot(L, a, lane, s, j, tag):
        addr = i64(a["mid_pairs"]) + i64(((s * i32(SLOTS) + j) * i32(INTER) + lane * i32(4)) * i32(4))
        v0 = ld_agent_i64(addr)
        v1 = ld_agent_i64(addr + i64(8))
        bad = ballot(_tag_bad(v0, tag) | _tag_bad(v1, tag))
        n = i32(0)
        while (bad != i64(0)) & (n < i32(POLL_LIMIT)):
            v0 = ld_agent_i64(addr)
            v1 = ld_agent_i64(addr + i64(8))
            bad = ballot(_tag_bad(v0, tag) | _tag_bad(v1, tag))
            n = n + i32(1)
        _poll_report(a, lane, i32(gpu.block_id("x")), bad != i64(0), ERR_MIDS)
        w0 = i32(v0 & i64(0xFFFFFFFF))
        w1 = i32(v1 & i64(0xFFFFFFFF))
        xv = [bf16_lo(w0), bf16_hi(w0), bf16_lo(w1), bf16_hi(w1)]
        am = _absf(xv[0]).maximumf(_absf(xv[1]))
        am = am.maximumf(_absf(xv[2])).maximumf(_absf(xv[3]))
        am = oct_max_pos(am)
        e, inv = mx_scale(am)
        lds_st(L, L_MID4 + j * i32(INTER // 2) + lane * i32(2), fx.Int16(fp4x4(xv, inv)), align=2)
        if (lane & i32(7)) == i32(0):
            lds_st(L, L_MSC + j * i32(INTER // MX_BLOCK) + (lane >> i32(3)), fx.Int8(e), align=1)

    @traced
    def mid_all(L, a, lane, wave, s, tag):
        mid_slot(L, a, lane, s, wave, tag)
        if wave == i32(0):
            mid_slot(L, a, lane, s, i32(SLOTS - 1), tag)
        gpu.barrier()

    def down_mfma(L, lane, items):
        kg = lane >> i32(4)
        cc = (lane >> i32(3)) & i32(1)
        zero = fx.Vector.filled(4, 0.0, fx.Float32)
        for valid, part, j, q, ws, sd in items:
            tail = part == i32(2)
            bchunk = tail.select(cc, part)
            b4 = lds_ld(L, L_MID4 + j * i32(INTER // 2) + bchunk * i32(64) + kg * i32(16), T.vec(4, T.i32), 16)
            sb = lds_u8(L, L_MSC + j * i32(INTER // MX_BLOCK) + bchunk * i32(4) + kg)
            sa = sd.shrui(part * i32(8))
            p = mfma_fp4(ws, b4, zero, sa, 0, sb, 0)
            writer = tail.select(
                (lane == i32(0)) | (lane == i32(16)) | (lane == i32(40)) | (lane == i32(56)),
                (lane & i32(15)) == i32(0),
            )
            cw = tail.select(lane >> i32(5), part)
            roww = tail.select(i32(16) + ((lane >> i32(4)) & i32(1)) * i32(4), kg * i32(4))
            off = L_P + (((j * i32(S) + q) * i32(2) + cw) * i32(24) + roww) * i32(4)
            _store_p(L, valid & writer, off, p)

    @traced
    def _store_p(L, pred, off, p):
        if pred:
            lds_st(L, off, p, align=16)

    @traced
    def down_finish(L, tid):
        if tid < i32(ROWS):
            q = tid // i32(24)
            rr = tid % i32(24)
            acc = f32(0.0)
            for j in range_constexpr(SLOTS):
                wj = f32(1.0) if j == 0 else lds_f(L, L_TPRB + i32((j - 1) * 4))
                pj = f32(0.0)
                for cc in range_constexpr(2):
                    pj = pj + lds_f(L, L_P + (((i32(j * S) + q) * i32(2) + i32(cc)) * i32(24) + rr) * i32(4))
                acc = acc + pj * wj
            lds_st(L, L_PART + tid * i32(2), fx.Int16(bf16_bits(acc)), align=2)
        gpu.barrier()

    def peer_base(a, p):
        return i64(bld(rsrc(a["sym"]), p * i32(8), T.i64))

    @traced
    def ar_proto0(L, a, lane, wave, bid, mype, npes, arflag):
        par = arflag & i32(1)
        if wave < npes - i32(1):
            p = wave + (wave >= mype).select(i32(1), i32(0))
            ms = mype - (mype > p).select(i32(1), i32(0))
            dst = peer_base(a, p) + i64(((ms * i32(2) + par) * i32(GRID) + bid) * i32(WIRE))
            if lane < i32(PKTS):
                d = fx.Vector(lds_ld(L, L_PART + lane * i32(8), T.vec(2, T.i32), 8))
                pkt = fx.Vector.from_elements([i32(d[0]), arflag, i32(d[1]), arflag], fx.Int32)
                bst(pkt, rsrc(dst), lane * i32(16), AUX_SYS)
            src = (
                peer_base(a, mype)
                + i64(((wave * i32(2) + par) * i32(GRID) + bid) * i32(WIRE))
                + i64(fx.min(lane, i32(PKTS - 1)) * i32(16))
            )
            active = lane < i32(PKTS)
            v = ld_sys_v4i(src)
            bad = ballot(active & ((i32(v[1]) != arflag) | (i32(v[3]) != arflag)))
            n = i32(0)
            while (bad != i64(0)) & (n < i32(POLL_LIMIT)):
                v = ld_sys_v4i(src)
                bad = ballot(active & ((i32(v[1]) != arflag) | (i32(v[3]) != arflag)))
                n = n + i32(1)
            _poll_report(a, lane, bid, bad != i64(0), ERR_AR)
            if active:
                lds_st(
                    L,
                    L_RECV + (wave * i32(ROWS) + lane * i32(4)) * i32(2),
                    fx.Vector.from_elements([i32(v[0]), i32(v[2])], fx.Int32),
                    align=8,
                )
        gpu.barrier()

    @traced
    def ar_proto1(L, a, lane, wave, bid, mype, npes, arflag):
        par = arflag & i32(1)
        if wave < npes - i32(1):
            p = wave + (wave >= mype).select(i32(1), i32(0))
            ms = mype - (mype > p).select(i32(1), i32(0))
            dst = peer_base(a, p) + i64(((ms * i32(2) + par) * i32(GRID) + bid) * i32(WIRE))
            line = lane >> i32(3)
            pp = lane & i32(7)
            if lane < i32(LINE_LANES):
                d = fx.Vector(
                    lds_ld(
                        L,
                        L_PART + (line * i32(24) + fx.min(pp, i32(5)) * i32(4)) * i32(2),
                        T.vec(2, T.i32),
                        8,
                    )
                )
                is_pay = pp < i32(6)
                w = fx.Vector.from_elements(
                    [is_pay.select(i32(d[0]), arflag), is_pay.select(i32(d[1]), arflag)],
                    fx.Int32,
                )
                bst(w, rsrc(dst), lane * i32(8), AUX_SYS)
            src = peer_base(a, mype) + i64(((wave * i32(2) + par) * i32(GRID) + bid) * i32(WIRE))
            lc = fx.min(lane, i32(LINE_LANES - 1))
            fl_addr = src + i64((lc >> i32(3)) * i32(64) + i32(56))
            want = (i64(arflag) << i64(32)) | (i64(arflag) & i64(0xFFFFFFFF))
            active = lane < i32(LINE_LANES)
            fl = ld_sys_i64(fl_addr)
            bad = ballot(active & (fl != want))
            n = i32(0)
            while (bad != i64(0)) & (n < i32(POLL_LIMIT)):
                fl = ld_sys_i64(fl_addr)
                bad = ballot(active & (fl != want))
                n = n + i32(1)
            _poll_report(a, lane, bid, bad != i64(0), ERR_AR)
            if active & ((lane & i32(7)) < i32(6)):
                v = ld_sys_i64(src + i64(lc * i32(8)))
                lds_st(
                    L,
                    L_RECV + (wave * i32(ROWS) + (lc >> i32(3)) * i32(24) + (lc & i32(7)) * i32(4)) * i32(2),
                    v,
                    align=8,
                )
        gpu.barrier()

    @traced
    def ar_finish(L, a, tid, s, lb, mype, npes):
        if tid < i32(ROWS):
            acc = f32(0.0)
            own = bf16_lo(i32(lds_ld(L, L_PART + tid * i32(2), T.i16, 2)) & i32(0xFFFF))
            for p in range_constexpr(MAX_PES):
                slot = i32(p) - (i32(p) > mype).select(i32(1), i32(0))
                sl = fx.min(fx.max(slot, i32(0)), i32(MAX_PES - 2))
                rv = bf16_lo(
                    i32(lds_ld(L, L_RECV + (sl * i32(ROWS) + tid) * i32(2), T.i16, 2)) & i32(0xFFFF)
                )
                v = (i32(p) == mype).select(own, rv)
                acc = (i32(p) < npes).select(acc + v, acc)
            row = s * i32(HIDDEN) + lb * i32(ROWS) + tid
            if i64(a["residual"]) != i64(0):
                r = i32(bld(rsrc(a["residual"]), row * i32(2), T.i16)) & i32(0xFFFF)
                acc = acc + bf16_lo(r)
            bst(fx.Int16(bf16_bits(acc)), rsrc(a["out"]), row * i32(2))

    def routed_publish(L, a, tid, wave, lane, s, lb, grp, tile, tag):
        for m in range(S):
            r = (grp ^ i32(1)) + i32(G * m)
            tile_finish(
                L, a, tid, L_REDT + (1 + m) * NWAVE * 64, L_MIDT + (1 + m) * 16, s,
                r + i32(1), tile, tag, fx.Boolean(True),
            )
        items = down_issue(L, a, wave, lane, lb)
        rocdl.sched_barrier(0)
        return items

    @traced
    def _mark(tl, tid, bid, k):
        if tid == i32(0):
            t = i64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))
            bst(t, rsrc(tl), (bid * i32(TIMELINE_SLOTS) + i32(k)) * i32(8))

    def mark(tl, tid, bid, k, who=0):
        if const_expr(timeline):
            _mark(tl, tid - i32(who), bid, k)

    @traced
    def epoch_finish(a, tid, bid, use_epoch, ep):
        if use_epoch & (tid == i32(0)):
            st_agent_i32(i64(a["flags"]) + i64(bid * i32(4)), ep + i32(1))

    Shared = fx.struct(
        type(
            "Shared", (), {"__annotations__": {"buf": fx.Array[fx.Int8, LDS_BYTES, 16]}}
        )
    )

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def fused_moe_allreduce_a4w4_kernel(
        hidden: fx.Int64,
        gamma: fx.Int64,
        router_w: fx.Int64,
        ug_w: fx.Int64,
        ug_scales: fx.Int64,
        bias: fx.Int64,
        down_w: fx.Int64,
        down_scales: fx.Int64,
        residual: fx.Int64,
        sym: fx.Int64,
        mype: fx.Int32,
        npes: fx.Int32,
        flag: fx.Int32,
        norm_out: fx.Int64,
        scores: fx.Int64,
        score_lines: fx.Int64,
        flags: fx.Int64,
        probs_out: fx.Int64,
        indices_out: fx.Int64,
        hidden_mid: fx.Int64,
        out: fx.Int64,
        mid_pairs: fx.Int64,
        sen_tag: fx.Int32,
        timeline_ptr: fx.Int64,
    ):
        a = {
            "gamma": gamma,
            "ug_w": ug_w,
            "ug_scales": ug_scales,
            "bias": bias,
            "down_w": down_w,
            "down_scales": down_scales,
            "residual": residual,
            "sym": sym,
            "norm_out": norm_out,
            "scores": scores,
            "score_lines": score_lines,
            "flags": flags,
            "probs_out": probs_out,
            "indices_out": indices_out,
            "hidden_mid": hidden_mid,
            "out": out,
            "mid_pairs": mid_pairs,
        }
        tid = i32(gpu.thread_id("x"))
        bid = i32(gpu.block_id("x"))
        lane = tid % i32(64)
        wave = uni(tid // i32(64))
        s = bid // i32(BPS)
        lb = bid % i32(BPS)
        grp = lb // i32(32)
        tile = lb % i32(32)
        lds = fx.SharedAllocator().allocate(Shared).peek()
        L = uni(i32(fx.ptrtoint(lds.buf.ptr)))

        use_epoch = sen_tag == i32(0)
        ep = ld_agent_i32(i64(flags) + i64(bid * i32(4)))
        tag = use_epoch.select(ep + i32(1), sen_tag)
        arflag = use_epoch.select(ep + i32(1), flag)

        e0 = wave * i32(768) + lane * i32(4)
        xrs = rsrc(i64(hidden) + i64(s * i32(HIDDEN * 2)))
        grs = rsrc(gamma)
        xv = [fx.Vector(bld(xrs, (e0 + i32(j * 256)) * i32(2), T.vec(2, T.i32))) for j in range(3)]
        gv = [fx.Vector(bld(grs, (e0 + i32(j * 256)) * i32(4), T.vec(4, T.f32))) for j in range(3)]
        rrs = rsrc(
            i64(router_w) + i64(fx.min(lb, i32(31)) * i32(ROUTER_BLK_BYTES)),
            (lb < i32(32)).select(i32(ROUTER_BLK_BYTES), i32(0)),
        )
        wr = []
        for kc in range_constexpr(ROUTER_WAVE_KC):
            v = ld_v4i(rrs, (wave * i32(ROUTER_WAVE_KC) + i32(kc)) * i32(1024) + lane * i32(16), AUX_NT)
            wr.append([i32(v[q]) for q in range(4)])
        shared_on = (grp == i32(1)) & (wave != i32(0))
        ws0, sv0 = ug_issue(
            a,
            i32(0),
            tile,
            wave,
            lane,
            shared_on.select(i32(UG_TILE_BYTES), i32(0)),
            shared_on.select(i32(UG_SC_TILE), i32(0)),
        )
        rocdl.sched_barrier(0)

        tl = timeline_ptr
        mark(tl, tid, bid, 0)
        nvs = norm(L, wave, lane, xv, gv)
        mark(tl, tid, bid, 1)

        acc_r = router_dot(L, wave, lane, wr)
        router_publish(L, a, tid, wave, lane, acc_r, s, lb, tag)
        mark(tl, tid, bid, 2)
        act_quant(L, wave, lane, nvs)
        gpu.barrier()

        shared_mfma(L, wave, lane, ws0, sv0)
        gpu.barrier()
        tile_finish(L, a, tid, L_REDT, L_MIDT, s, i32(0), tile, tag, grp == i32(1))
        mark(tl, tid, bid, 3)

        topk_rank(L, a, lane, wave, s, lb, tag, tl, tid, bid)
        routed_tiles(L, a, wave, lane, grp, tile, tl, tid, bid)
        gpu.barrier()
        mark(tl, tid, bid, 4)

        dn_items = routed_publish(L, a, tid, wave, lane, s, lb, grp, tile, tag)
        mark(tl, tid, bid, 5)

        mid_all(L, a, lane, wave, s, tag)
        mark(tl, tid, bid, 6)
        down_mfma(L, lane, dn_items)
        gpu.barrier()
        down_finish(L, tid)
        mark(tl, tid, bid, 7)

        if const_expr(proto == 0):
            ar_proto0(L, a, lane, wave, bid, mype, npes, arflag)
        else:
            ar_proto1(L, a, lane, wave, bid, mype, npes, arflag)
        mark(tl, tid, bid, 8)
        ar_finish(L, a, tid, s, lb, mype, npes)
        epoch_finish(a, tid, bid, use_epoch, ep)
        mark(tl, tid, bid, 9)

    @flyc.jit
    def launch(
        hidden: fx.Int64,
        gamma: fx.Int64,
        router_w: fx.Int64,
        ug_w: fx.Int64,
        ug_scales: fx.Int64,
        bias: fx.Int64,
        down_w: fx.Int64,
        down_scales: fx.Int64,
        residual: fx.Int64,
        sym: fx.Int64,
        mype: fx.Int32,
        npes: fx.Int32,
        flag: fx.Int32,
        norm_out: fx.Int64,
        scores: fx.Int64,
        score_lines: fx.Int64,
        flags: fx.Int64,
        probs_out: fx.Int64,
        indices_out: fx.Int64,
        hidden_mid: fx.Int64,
        out: fx.Int64,
        mid_pairs: fx.Int64,
        sen_tag: fx.Int32,
        timeline_ptr: fx.Int64,
        stream: fx.Stream,
    ):
        fused_moe_allreduce_a4w4_kernel(
            hidden,
            gamma,
            router_w,
            ug_w,
            ug_scales,
            bias,
            down_w,
            down_scales,
            residual,
            sym,
            mype,
            npes,
            flag,
            norm_out,
            scores,
            score_lines,
            flags,
            probs_out,
            indices_out,
            hidden_mid,
            out,
            mid_pairs,
            sen_tag,
            timeline_ptr,
        ).launch(grid=(GRID, 1, 1), block=(NT, 1, 1), stream=stream)

    launch.consts = c
    return launch
