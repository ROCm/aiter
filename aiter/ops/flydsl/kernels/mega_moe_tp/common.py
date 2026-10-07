# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device helpers shared by the MegaMoE TP and SpRsNorm kernels."""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T

from .. import buffer_ops
from ..mxfp4_gemm_common import _e8m0_from_amax, _fabs_f32

traced = ASTRewriter.transform

MAX_TP = 8
DEADLINE = 200_000_000
AUX_SC1 = 16
AUX_SYS = 1 | 16

MONO, ACQ, REL, ACQ_REL = (
    fx.AtomicOrdering.Monotonic,
    fx.AtomicOrdering.Acquire,
    fx.AtomicOrdering.Release,
    fx.AtomicOrdering.AcqRel,
)


def _attr(v):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), int(v))


def _u(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def i32(v):
    raw = v if isinstance(v, ir.Value) else getattr(v, "ir_value", lambda: None)()
    if raw is not None and isinstance(raw.type, ir.IndexType):
        return fx.Int32(fx.index_cast(T.i32, raw))
    return fx.Int32(v)


class _LazyTy:
    def __init__(self, fn):
        self.fn = fn


V4I = _LazyTy(lambda: T.vec(4, T.i32))
V4F = _LazyTy(lambda: T.vec(4, T.f32))
V2I = _LazyTy(lambda: T.vec(2, T.i32))


def _ty(t):
    return t.fn() if isinstance(t, _LazyTy) else t


# LDS is one byte arena carved by computed offsets (no typed views): pointers
# are built from the integer address.
def lds_ptr(base, off, ty, align=4):
    ty = _ty(ty)
    if isinstance(ty, ir.VectorType):
        ty = ty.element_type
    return fx.inttoptr(
        fx.PointerType.get(ty, fx.AddressSpace.Shared, align), base + i32(off)
    )


def lds_ld(base, off, ty=None, align=4):
    ty = _ty(ty) if ty is not None else T.i32
    return fx.ptr_load(lds_ptr(base, off, ty, align), result_type=ty)


def lds_ld_i32(base, off):
    return i32(lds_ld(base, off))


def lds_st(base, off, val, align=4):
    v = _u(val)
    fx.ptr_store(v, lds_ptr(base, off, v.type, align))


def lds_p(base, off):
    return fx.inttoptr(
        fx.PointerType.get(T.i32, fx.AddressSpace.Shared, 4),
        fx.Int32(base + i32(off)),
    )


def gptr(addr):
    return fx.inttoptr(
        fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), fx.Int64(addr)
    )


def lds_atomic_add(base, off, v, ordering=MONO):
    return i32(
        fx.atomic_add(
            lds_p(base, off), i32(v), syncscope="workgroup", ordering=ordering
        )
    )


def lds_atomic_or(base, off, v):
    return i32(fx.atomic_or(lds_p(base, off), i32(v), syncscope="workgroup"))


def lds_ld_acq(base, off):
    return i32(
        fx.generic_load(
            lds_p(base, off),
            dtype=fx.Int32,
            memory_order=MONO,
            syncscope="workgroup",
            volatile=True,
        )
    )


def lds_st_rel(base, off, v):
    fx.generic_store(lds_p(base, off), i32(v), memory_order=REL, syncscope="workgroup")


def lds_cas(base, off, cmp, new):
    return i32(
        fx.atomic_cas(lds_p(base, off), i32(cmp), i32(new), syncscope="workgroup")[0]
    )


def _ctpop(v):
    return i32(fx.ctpop(i32(v)))


def g_ld_rel(addr, scope):
    return i32(
        fx.generic_load(
            gptr(addr),
            dtype=fx.Int32,
            memory_order=MONO,
            syncscope=scope,
            volatile=True,
        )
    )


def g_ld_sys(addr):
    return g_ld_rel(addr, "one-as")


def g_st_sys(addr, v):
    fx.generic_store(gptr(addr), i32(v), memory_order=MONO, syncscope="one-as")


def g_add_agent(addr, v):
    return i32(fx.atomic_add(gptr(addr), i32(v), syncscope="agent"))


def g_or_agent(addr, v):
    fx.atomic_or(gptr(addr), i32(v), syncscope="agent")


def g_ld_i32(addr):
    return i32(fx.generic_load(gptr(addr), dtype=fx.Int32))


def uni(v):
    return i32(rocdl.readfirstlane(T.i32, i32(v)))


def uni64(v):
    v = fx.Int64(v)
    lo = uni(fx.Int32(v & fx.Int64(0xFFFFFFFF)))
    hi = uni(fx.Int32(v >> fx.Int64(32)))
    return (fx.Int64(hi) << fx.Int64(32)) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))


# Buffer resources from raw (peer / arena) addresses plus per-lane byte offsets
# and explicit cache-policy bits (sc0 / sc1 / nt): no layout-tensor form.
def rsrc(addr, nbytes=None):
    if nbytes is None:
        return buffer_ops.create_buffer_resource_from_addr(uni64(addr))
    return buffer_ops.create_buffer_resource_from_addr(
        uni64(addr), num_records_bytes=fx.Int64(uni(nbytes))
    )


def bld(rs, voff, soff, ty, aux=0):
    return rocdl.raw_ptr_buffer_load(_ty(ty), rs, i32(voff), i32(soff), aux=aux)


def bst(val, rs, voff, soff, aux=0):
    rocdl.raw_ptr_buffer_store(
        _u(val), rs, _u(i32(voff)), _u(i32(soff)), aux=_attr(aux)
    )


def asm(text, cons="", args=()):
    _llvm.inline_asm(None, [_u(a) for a in args], text, cons, has_side_effects=True)


# LDS-DMA has no wrapper that keeps the compiler out of its counters: inline
# asm, so it neither sees nor waits on these loads; waits are explicit.
def dma16(lds_addr, rs, voff, soff, nt=False):
    asm(
        "s_mov_b32 m0, $0\n\tbuffer_load_dwordx4 $1, $2, $3 offen"
        + (" nt" if nt else "")
        + " lds",
        "s,v,s,s",
        (uni(lds_addr), i32(voff), rs, uni(soff)),
    )


def dma4(lds_addr, rs, voff, soff, sys=False):
    asm(
        "s_mov_b32 m0, $0\n\tbuffer_load_dword $1, $2, $3 offen"
        + (" sc0 sc1" if sys else "")
        + " lds",
        "s,v,s,s",
        (uni(lds_addr), i32(voff), rs, uni(soff)),
    )


# Waits stay inline asm: they count the asm LDS-DMA loads, which the compiler's
# waitcnt pass cannot see (it would merge or drop an s_waitcnt intrinsic).
def wait_vm(n):
    asm(f"s_waitcnt vmcnt({min(int(n), 63)})")


def wait_lgkm0():
    asm("s_waitcnt lgkmcnt(0)")


def amax(vals):
    am = _fabs_f32(vals[0])
    for v in vals[1:]:
        am = am.maximumf(_fabs_f32(v))
    return am


def fp4_pack(vals, qs):
    pk = i32(0)
    for k in range(len(vals) // 2):
        pk = rocdl.cvt_scalef32_pk_fp4_f32(
            T.i32, pk, vals[2 * k], vals[2 * k + 1], qs, k
        )
    return fx.Int32(pk)


def fp8x4_pack(f, qs):
    zero = fx.Vector.filled(2, 0, fx.Int16)
    w = rocdl.cvt_scalef32_pk_fp8_f32(
        T.vec(2, T.i16), _u(zero), _u(f[0]), _u(f[1]), _u(qs), False
    )
    w = rocdl.cvt_scalef32_pk_fp8_f32(
        T.vec(2, T.i16), w, _u(f[2]), _u(f[3]), _u(qs), True
    )
    return fx.Int32(fx.Vector(w).bitcast(fx.Int32)[0])


def fp8x4_unpack(d, sc):
    out = []
    for sel in (False, True):
        v = fx.Vector(
            rocdl.cvt_scalef32_pk_f32_fp8(T.vec(2, T.f32), _u(i32(d)), _u(sc), sel)
        )
        out += [fx.Float32(v[0]), fx.Float32(v[1])]
    return out


def e8_scale(e8):
    return ((fx.Int32(e8) & i32(0xFF)) << i32(23)).bitcast(fx.Float32)


def fp8x8_decode(ld):
    dv = fx.Vector(ld[0])
    sc = e8_scale(ld[1])
    return fp8x4_unpack(fx.Int32(dv[0]), sc) + fp8x4_unpack(fx.Int32(dv[1]), sc)


def mxfp8x8(acc):
    am = amax(acc)
    am = am.maximumf(am.shuffle_xor(i32(1), i32(64)))
    am = am.maximumf(am.shuffle_xor(i32(2), i32(64)))
    e8, qs = _e8m0_from_amax(am, max_norm=448.0)
    return fp8x4_pack(acc[0:4], qs), fp8x4_pack(acc[4:8], qs), fx.Int32(e8) & i32(0xFF)


def scales4(e8i):
    e = e8i.bitcast(fx.Float32)
    sc = e8i
    for k in range_constexpr(1, 4):
        sc = sc | (e.shuffle_xor(i32(4 * k), i32(64)).bitcast(fx.Int32) << i32(8 * k))
    return sc


def bf16_bits(f):
    return fx.Int32(
        fx.Vector.from_elements([fx.Float32(f).to(fx.BFloat16)], fx.BFloat16).bitcast(
            fx.Int16
        )[0]
    ) & i32(0xFFFF)


def pack_bf16x2(a, b):
    return bf16_bits(a) | (bf16_bits(b) << i32(16))


def bf16x8_to_f32(d):
    dv = fx.Vector(d)
    out = []
    for q in range_constexpr(4):
        w = fx.Int32(dv[q])
        out.append((w << i32(16)).bitcast(fx.Float32))
        out.append((w & i32(-65536)).bitcast(fx.Float32))
    return out


def pack_bf16x8(acc):
    return fx.Vector.from_elements(
        [pack_bf16x2(acc[2 * q], acc[2 * q + 1]) for q in range(4)], fx.Int32
    )


def sum_live(acc, vals, live):
    return [x + live.select(y, fx.Float32(0.0)) for x, y in zip(acc, vals)]


def wave_red(v, lane, op):
    for k in (1, 2, 4, 8, 16, 32):
        v = op(v, i32(rocdl.ds_bpermute(T.i32, (lane ^ i32(k)) * i32(4), v)))
    return uni(v)


def wave_rank(hit):
    b = fx.Int64(rocdl.ballot(T.i64, hit))
    lo = i32(b & fx.Int64(0xFFFFFFFF))
    hi = i32(b >> fx.Int64(32))
    below = rocdl.mbcnt_lo(T.i32, _u(lo), _u(i32(0)))
    below = i32(rocdl.mbcnt_hi(T.i32, _u(hi), below))
    return below, _ctpop(lo) + _ctpop(hi)


def _swap(fn, x, y):
    st = _llvm.StructType.get_literal([T.i32, T.i32])
    r = fn(st, _u(i32(x)), _u(i32(y)), False, False)
    return (
        i32(_llvm.ExtractValueOp(T.i32, r, [0]).res),
        i32(_llvm.ExtractValueOp(T.i32, r, [1]).res),
    )


def swap16(x, y):
    return _swap(rocdl.permlane16_swap, x, y)


def swap32(x, y):
    return _swap(rocdl.permlane32_swap, x, y)


def widen(v4):
    z = fx.Vector.filled(4, 0, fx.Int32)
    return fx.Vector(v4).shuffle(z, list(range(8)))


# Scaled MFMA with per-lane E8M0 scales read from LDS: no MMA-atom form.
def mfma(a4, b4, cacc, sa, sb, b8=False):
    bx = b4 if b8 else widen(b4)
    return fx.Vector(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            _ty(V4F),
            [widen(a4), bx, cacc, 4, 0 if b8 else 4, 0, i32(sa), 0, i32(sb)],
        )
    )


def cat8(lo, hi):
    return _u(fx.Vector(lo).shuffle(fx.Vector(hi), list(range(8))))


# s_memrealtime (100 MHz) has no wrapper.
def now():
    return fx.Int64(
        _llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], [])
    )


def alive(t0):
    return (now() - t0) < fx.Int64(DEADLINE)


@traced
def spin_lds_ge(L, off, target):
    cur = lds_ld_acq(L, off)
    while cur < target:
        rocdl.s_sleep(0)
        cur = lds_ld_acq(L, off)


@traced
def spin0(L, lane, off, target):
    if lane == i32(0):
        spin_lds_ge(L, off, target)
    rocdl.sched_barrier(0)


@traced
def poll_sys_ge(addr, target, nap=2):
    cur = g_ld_sys(addr)
    t0 = now()
    while (cur < target) & alive(t0):
        rocdl.s_sleep(nap)
        cur = g_ld_sys(addr)
    return cur
