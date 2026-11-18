# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""Low-level helpers of the step kernel: raw 64-bit global addresses and LDS byte offsets, scoped loads / stores /
atomics, gfx1250 intrinsics (WMMA, FP8 conversions, clocks), and the bounded polls (wait_count, handoff) with their
timeout counter.

All functions are traced inside a @flyc.kernel body.
"""
from __future__ import annotations

import os
import re as _re

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, llvm
from flydsl.compiler.ast_rewriter import ASTRewriter as _ASTR
from flydsl.expr import gpu, rocdl
from flydsl.expr.typing import T, as_ir_value

# ---------------------------------------------------------------- types
def i32():
    return ir.IntegerType.get_signless(32)


def i64():
    return ir.IntegerType.get_signless(64)


def vty(n, elem):
    return ir.VectorType.get([n], elem)


def gptr_ty():
    return ir.Type.parse("!llvm.ptr<1>")


def lptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def _v(x):
    return as_ir_value(x) if not isinstance(x, ir.Value) else x


# ---------------------------------------------------------------- addresses
def gptr(addr):
    """Global pointer from a 64-bit byte address (fx.Int64)."""
    return llvm.IntToPtrOp(gptr_ty(), _v(fx.Int64(addr))).result


def lds_base():
    """Byte offset of the kernel's dynamic LDS (fx.Int32)."""
    p = fx.get_dyn_shared()
    return fx.Int32(fx.Int64(fx.ptrtoint(p)))


def lptr(off):
    """LDS pointer from a byte offset (fx.Int32)."""
    return llvm.IntToPtrOp(lptr_ty(), _v(fx.Int32(off))).result


# ---------------------------------------------------------------- loads / stores
_ORD = {None: None, "relaxed": llvm.AtomicOrdering.monotonic}


def _wrap(res, kind, n):
    if n == 1:
        return kind(res)
    return fx.Vector(res, (n,), kind)


def gload(addr, kind=fx.Int32, n=1, nt=False, scope=None, align=None):
    """Load n elements of `kind` at global byte address `addr`. scope='agent' -> relaxed device-scope atomic load."""
    ty = kind.ir_type if n == 1 else vty(n, kind.ir_type)
    kw = dict(alignment=align or (kind.width // 8) * n)
    if nt:
        kw["nontemporal"] = True
    if scope:
        kw["ordering"] = llvm.AtomicOrdering.monotonic
        kw["syncscope"] = scope
    return _wrap(llvm.LoadOp(ty, gptr(addr), **kw).result, kind, n)


_EB = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "f16": 2, "bf16": 2, "f32": 4, "f64": 8}


def tbytes(t):
    """Size in bytes of a scalar or vector<N x elem> MLIR type."""
    s = str(t)
    m = _re.fullmatch(r"vector<(\d+)x(\w+)>", s)
    return int(m[1]) * _EB[m[2]] if m else _EB[s]


def gstore(addr, val, nt=False, scope=None, align=None):
    v = _v(val)
    kw = {"alignment": align or tbytes(v.type)}
    if nt:
        kw["nontemporal"] = True
    if scope:
        kw["ordering"] = llvm.AtomicOrdering.monotonic
        kw["syncscope"] = scope
    llvm.StoreOp(v, gptr(addr), **kw)


def lload(off, kind=fx.Int32, n=1, align=None):
    ty = kind.ir_type if n == 1 else vty(n, kind.ir_type)
    return _wrap(llvm.LoadOp(ty, lptr(off), alignment=align or (kind.width // 8) * n).result, kind, n)


def lstore(off, val, align=None):
    v = _v(val)
    llvm.StoreOp(v, lptr(off), alignment=align or tbytes(v.type))


# ---------------------------------------------------------------- atomics / fences / sync
def atomic_add(addr, val, scope="agent"):
    return fx.Int32(llvm.AtomicRMWOp(llvm.AtomicBinOp.add, gptr(addr), _v(fx.Int32(val)),
                                     llvm.AtomicOrdering.monotonic, syncscope=scope, alignment=4).result)


def atomic_fadd(addr, val, scope="agent"):
    llvm.AtomicRMWOp(llvm.AtomicBinOp.fadd, gptr(addr), _v(fx.Float32(val)), llvm.AtomicOrdering.monotonic,
                     syncscope=scope, alignment=4)


def latomic_add_f32(off, val):
    llvm.AtomicRMWOp(llvm.AtomicBinOp.fadd, lptr(off), _v(fx.Float32(val)), llvm.AtomicOrdering.monotonic,
                     syncscope="workgroup", alignment=4)


def latomic_max_u32(off, val):
    llvm.AtomicRMWOp(llvm.AtomicBinOp.umax, lptr(off), _v(fx.Int32(val)), llvm.AtomicOrdering.monotonic,
                     syncscope="workgroup", alignment=4)


def fence_release(scope):
    llvm.fence(llvm.AtomicOrdering.release, syncscope=scope)


def fence_acquire(scope):
    llvm.fence(llvm.AtomicOrdering.acquire, syncscope=scope)


def intrin(res_ty, name, args):
    return llvm.call_intrinsic(res_ty, name, [_v(a) for a in args], [], [])


def s_sleep(n=1):
    rocdl.s_sleep(n)


def wait_storecnt0():
    llvm.inline_asm(None, [], "s_wait_storecnt 0x0", "", has_side_effects=True)


def cluster_barrier():
    """__builtin_amdgcn_s_cluster_barrier()."""
    intrin(None, "llvm.amdgcn.s.cluster.barrier", [])


def clock():
    """__builtin_readsteadycounter() (100 MHz)."""
    return fx.Int64(intrin(i64(), "llvm.readsteadycounter", []))


def prefetch(addr):
    """__builtin_amdgcn_global_prefetch(p, 0)."""
    intrin(None, "llvm.amdgcn.global.prefetch", [gptr(addr), fx.Int32(0)])


def async_to_lds_b128(gaddr, loff, cpol=0):
    """__builtin_amdgcn_global_load_async_to_lds_b128(g, l, 0, cpol)."""
    intrin(None, "llvm.amdgcn.global.load.async.to.lds.b128", [gptr(gaddr), lptr(loff), fx.Int32(0), fx.Int32(cpol)])


TDM_ELEM = int(os.environ.get('TDM_ELEM', 4))


def tdm_load(gaddr, loff, row_bytes, rows, cpol=0, elem=TDM_ELEM):
    """TDM: rows x row_bytes contiguous bytes at the 64-bit address gaddr -> LDS byte offset loff, one instruction of
    this wave (tensor_load_to_lds, counted by s_wait_tensorcnt). gaddr / loff must be wave-uniform."""
    from flydsl.expr.rocdl import tdm_ops as _t
    from flydsl._mlir.dialects import vector as _vec, rocdl as _rd
    from flydsl.expr.typing import T as _T
    ga = fx.Int64(gaddr)
    lo = fx.Int32(ga)
    hi = fx.Int32(ga >> 32) | fx.Int32(1 << 31)
    c = lambda v: _v(fx.Int32(v))
    dg0 = _vec.from_elements(_T.vec(4, _T.i32), [c(1), _v(fx.Int32(loff)), _v(lo), _v(hi)])
    d0, d1 = row_bytes // elem, rows
    dsz = {1: 0, 2: 1, 4: 2, 8: 3}[elem]
    dg1 = _vec.from_elements(_T.vec(8, _T.i32), [c(dsz << 16), c((d0 & 0xFFFF) << 16), c(((d0 >> 16) & 0xFFFF) | ((d1 & 0xFFFF) << 16)),
                                                 c(((d1 >> 16) & 0xFFFF) | (d0 << 16)), c(d1 & 0xFFFF), c(d0), c(0), c(0)])
    z4, z8 = _t._raw(_t._zero_dgroup_v4i32()), _t._raw(_t._zero_dgroup_v8i32())
    _rd.tensor_load_to_lds(dg0, dg1, z4, _t._raw(_t._zero_dgroup_v4i32()), z8, cache_policy=_t._cache_policy_attr(cpol))


def wait_tensorcnt(n=0):
    rocdl.s_wait_tensorcnt(n)


def wait_asynccnt0():
    rocdl.s_wait_asynccnt(0)


def opaque_v(x):
    """asm volatile("" : "+v"(x)) on an i32."""
    return fx.Int32(llvm.inline_asm(i32(), [_v(fx.Int32(x))], "", "=v,0", has_side_effects=True))


def opaque_s(x):
    """asm volatile("" : "+s"(x)) on an i32."""
    return fx.Int32(llvm.inline_asm(i32(), [_v(fx.Int32(x))], "", "=s,0", has_side_effects=True))


# ---------------------------------------------------------------- matrix
def wmma_mx(a, sa, b, sb, c):
    """__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(0, a, 0, b, 0, c, 0, 0, sa, 0, 0, sb, false, false):
    a, b vector<16xi32> (fp8 e4m3), c vector<8xf32>, sa/sb i32 E8M0 x4."""
    r = rocdl.wmma_scale_f32_16x16x128_f8f6f4(vty(8, T.f32), _v(a), _v(b), _v(c), _v(fx.Int32(sa)), _v(fx.Int32(sb)),
                                              fmtA=0, fmtB=0)
    return fx.Vector(r, (8,), fx.Float32)


def wmma_bf16(a, b, c):
    """__builtin_amdgcn_wmma_f32_16x16x32_bf16(false, a, false, b, 0, c, false, false): a, b vector<16xbf16>."""
    r = rocdl.wmma_f32_16x16x32_bf16_(vty(8, T.f32), _v(a), _v(b), _v(c))
    return fx.Vector(r, (8,), fx.Float32)


def ds_load_tr16(loff):
    """__builtin_amdgcn_ds_load_tr16_b128_v8bf16(lds ptr) -> vector<8xbf16>."""
    r = intrin(vty(8, T.bf16), "llvm.amdgcn.ds.load.tr16.b128.v8bf16", [lptr(loff)])
    return fx.Vector(r, (8,), fx.BFloat16)


def cvt_pk_fp8(a, b=0.0):
    """__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false) & 0xff -> e4m3 byte in an i32."""
    r = rocdl.cvt_pk_fp8_f32(i32(), _v(fx.Float32(a)), _v(fx.Float32(b)), _v(fx.Int32(0)), False)
    return fx.Int32(r)


def stx_dev(addr, val):
    """Exchange store at device scope (scope:SCOPE_DEV): relaxed agent-scope atomic stores of <= 8 bytes each, so a
    fence-free handoff (every wave s_wait_storecnt 0 + relaxed counter) publishes them (HDEV)."""
    nb = tbytes(_v(val).type)
    if nb == 16:
        v = val.bitcast(fx.Int64)
        gstore(addr, v[0], scope="agent", align=8)
        gstore(addr + fx.Int64(8), v[1], scope="agent", align=8)
    elif nb == 8 and isinstance(val, fx.Vector):
        gstore(addr, val.bitcast(fx.Int64)[0], scope="agent", align=8)
    elif nb == 2 and not isinstance(val, fx.Vector):
        gstore(addr, val.bitcast(fx.Int16), scope="agent", align=2)
    else:
        gstore(addr, val, scope="agent", align=nb)


SLOAD_INV = __import__("os").environ.get("SLOAD_INV", "0") == "1"


def sload64(addr):
    """i64 load through the constant address space (addrspace 4) at a wave-uniform address -> s_load_b64 (scalar cache;
    for read-only tables such as pointer tables)."""
    p = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<4>"), _v(fx.Int64(addr))).result
    return fx.Int64(llvm.LoadOp(fx.Int64.ir_type, p, alignment=8, invariant=SLOAD_INV).result)


def llvm_wait_loads():
    """s_wait_loadcnt 0 (diagnostic)."""
    llvm.inline_asm(None, [], "s_wait_loadcnt 0x0", "", has_side_effects=True)


def cvt_pk_fp8_w(old, a, b, hi):
    """__builtin_amdgcn_cvt_pk_fp8_f32(a, b, old, hi): e4m3(a), e4m3(b) into the low (hi=False) / high 16 bits of old."""
    r = rocdl.cvt_pk_fp8_f32(i32(), _v(fx.Float32(a)), _v(fx.Float32(b)), _v(fx.Int32(old)), hi)
    return fx.Int32(r)


def vbitcast_i32x4(v):
    """8 x bf16 (16 bytes) -> 4 x i32 vector (llvm.bitcast)."""
    return _wrap(llvm.BitcastOp(vty(4, i32()), _v(v)).result, fx.Int32, 4)


def latomic_add_i32(off, val):
    llvm.AtomicRMWOp(llvm.AtomicBinOp.add, lptr(off), _v(fx.Int32(val)), llvm.AtomicOrdering.monotonic,
                     syncscope="workgroup", alignment=4)


def lane_gather(v, src_lane):
    """Lane src_lane's value of v (ds_bpermute)."""
    return fx.Int32(rocdl.ds_bpermute(T.i32, _v(fx.Int32(src_lane) * 4), _v(fx.Int32(v))))


def fresh(v):
    """v (per-lane Int32) through an opaque move: computed at its use, not hoisted / CSE'd across stages."""
    return fx.Int32(llvm.inline_asm(i32(), [_v(fx.Int32(v))], "v_mov_b32 $0, $1", "=v,v", has_side_effects=True))


def fp8x2_to_f32(word, hi):
    """e4m3 bytes 0,1 (hi=False) or 2,3 (hi=True) of an i32 -> two exact f32 (integer decode: the packed cvt's
    encoding is rejected on gfx1250)."""
    w = fx.Int32(word)
    out = []
    for k in (2, 3) if hi else (0, 1):
        b = (w >> (8 * k)) & 255
        e = (b >> 3) & 15
        m = b & 7
        sg = (b & 128) << 24
        normal = (sg | ((e + 120) << 23) | (m << 20)).bitcast(fx.Float32)
        sub = (m.to(fx.Float32) * fx.Float32(2.0 ** -9))
        sub = ((b & 128) != 0).select(fx.Float32(0.0) - sub, sub)
        out.append((e == 0).select(sub, normal))
    return out[0], out[1]


def unlikely(b):
    """llvm.expect(b, false): block placement moves the guarded code out of the hot path (I-cache)."""
    i1 = ir.IntegerType.get_signless(1)
    f = arith.ConstantOp(i1, ir.IntegerAttr.get(i1, 0)).result
    return fx.Boolean(llvm.call_intrinsic(i1, "llvm.expect.i1", [_v(b), f], [], []))


# Poll timeouts made visible. Every bounded poll (cross-rank tag waits, device handoffs) gives up after 2 s and the
# kernel goes on with whatever it read; a timeout therefore means wrong results. TOFLAG[0] (set by the step kernel to its
# 'cyc' scratch) counts them: int32 [0] += 1, int32 [1] = the last site id. The runner reads it after a step.
TOFLAG = [None]
WSLEEP = tuple(int(v) for v in os.environ.get('WSLEEP', '1,1').split(','))   # a poll sleeps [0] x s_sleep([1]) between loads
POLL_TICKS = 200000000   # a bounded poll gives up after this many clock ticks (~2 s) and records a timeout


@_ASTR.transform
def note_timeout(t0, site):
    if fx.const_expr(TOFLAG[0] is not None):
        if clock() - t0 >= fx.Int64(POLL_TICKS):
            atomic_add(TOFLAG[0], 1)
            gstore(TOFLAG[0] + fx.Int64(4), fx.Int32(site))



@_ASTR.transform
def wait_count(addr, n, site, ws=None):
    """Spin until the agent-scope int32 counter at addr reaches n (bounded: POLL_TICKS, a timeout counted at site).
    addr: an address, or a function giving it (evaluated in the loop, as the polls were written before this helper)."""
    t0 = clock()
    while (gload(addr() if callable(addr) else addr, fx.Int32, scope="agent") < n) & (clock() - t0 < fx.Int64(POLL_TICKS)):
        for _ in range((ws or WSLEEP)[0]):
            s_sleep((ws or WSLEEP)[1])
    note_timeout(t0, site)


@_ASTR.transform
def handoff(cnt, t, hdev, n=256, site=None):
    """Device-wide handoff of n CTAs on counter cnt (thread t). hdev: the stages' exchange stores are device-scope ->
    every wave waits its stores, no fences; else release / acquire fences around the count."""
    if fx.const_expr(hdev):
        wait_storecnt0()
        gpu.barrier()
        if t == 0:
            atomic_add(cnt, 1)
            wait_count(cnt, n, 4 if site is None else site)
        gpu.barrier()
    else:
        gpu.barrier()
        if t == 0:
            fence_release("agent")
            atomic_add(cnt, 1)
            wait_count(cnt, n, 5 if site is None else site)
            fence_acquire("agent")
        gpu.barrier()


def latomic_max64(off, val):
    llvm.AtomicRMWOp(llvm.AtomicBinOp.max, lptr(off), _v(fx.Int64(val)), llvm.AtomicOrdering.monotonic,
                     syncscope="workgroup", alignment=8)


def gatomic_max64(addr, val):
    llvm.AtomicRMWOp(llvm.AtomicBinOp.max, gptr(addr), _v(fx.Int64(val)), llvm.AtomicOrdering.monotonic,
                     syncscope="agent", alignment=8)
