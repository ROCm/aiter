# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""FlyDSL device primitives for the cco-LSA dispatch/combine kernels: system-scope
atomics, ordered stores, fences, uncached/non-temporal loads and volatile spin-waits,
on top of cco peer pointers (cco.Window(h).lsa_ptr(pe, off)).

The atomic / ordered-store / fence / volatile-load ops stay on
flydsl._mlir.dialects.llvm: the high-level FlyDSL API exposes no memory ordering,
sync-scope, volatile, or non-temporal control, which these primitives require.
"""
from os import environ as _os_environ

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl._mlir.dialects import scf
from flydsl.expr import arith
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T
import flydsl.expr as fx


def _gptr(addr_i64):
    return _llvm_d.IntToPtrOp(
        _llvm_d.PointerType.get(address_space=1), arith.unwrap(addr_i64)
    ).result


def _ptr_plus(base_i64, offset, elem_bytes):
    """Global pointer for base + offset*elem_bytes (offset may be i32 or i64)."""
    addr = fx.Int64(arith.unwrap(base_i64)) + fx.Int64(arith.unwrap(offset)) * elem_bytes
    return _gptr(addr)


def atomic_add_global(addr_i64, val):
    """Monotonic remote global fetch-and-add at addr_i64; returns old value."""
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add,
        _gptr(addr_i64),
        arith.unwrap(val),
        _llvm_d.AtomicOrdering.monotonic,
    ).res


def atomic_add_release(addr_i64, val):
    """System-release fetch-and-add: every prior memory op of this thread is
    ordered before the increment becomes visible to an acquiring reader."""
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add,
        _gptr(addr_i64),
        arith.unwrap(val),
        _llvm_d.AtomicOrdering.release,
        syncscope="one-as",
    ).res


def atomic_add_acqrel(addr_i64, val):
    """System acq_rel fetch-and-add; returns the old value."""
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add,
        _gptr(addr_i64),
        arith.unwrap(val),
        _llvm_d.AtomicOrdering.acq_rel,
        syncscope="one-as",
    ).res


def atomic_cas_acquire(addr_i64, cmp_val, new_val):
    """System compare-and-swap; returns the value found at addr (== cmp_val when
    the swap happened). Drives the device-local READY -> CLAIMED task handoff."""
    res = _llvm_d.AtomicCmpXchgOp(
        _gptr(addr_i64),
        arith.unwrap(cmp_val),
        arith.unwrap(new_val),
        _llvm_d.AtomicOrdering.acquire,
        _llvm_d.AtomicOrdering.monotonic,
        syncscope="one-as",
    ).res
    return _llvm_d.ExtractValueOp(T.i32, res, [0]).res


def store_i32_system(addr_i64, offset, val):
    """System-release i32 store at addr + offset*4."""
    _llvm_d.StoreOp(
        arith.unwrap(val),
        _ptr_plus(addr_i64, offset, 4),
        alignment=4,
        ordering=_llvm_d.AtomicOrdering.release,
        syncscope="one-as",
    )


def store_i32_relaxed(addr_i64, offset, val):
    """Volatile, unordered i32 store at addr + offset*4. Carries no release, so it
    is only for the microbench's negative control."""
    _llvm_d.StoreOp(
        arith.unwrap(val),
        _ptr_plus(addr_i64, offset, 4),
        alignment=4,
        volatile_=True,
    )


def store_i64_system(addr_i64, offset, val):
    """System-release i64 store at addr + offset*8."""
    _llvm_d.StoreOp(
        arith.unwrap(val),
        _ptr_plus(addr_i64, offset, 8),
        alignment=8,
        ordering=_llvm_d.AtomicOrdering.release,
        syncscope="one-as",
    )


def fence_system_acquire():
    _llvm_d.FenceOp(_llvm_d.AtomicOrdering.acquire, syncscope="one-as")


def fence_system_release():
    _llvm_d.FenceOp(_llvm_d.AtomicOrdering.release, syncscope="one-as")


def _unwrap(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def load_i32_acquire(addr_i64):
    """Volatile monotonic i32 load: volatile+ordering keeps the spin re-read from
    being hoisted/CSE'd out of the wait loop (LICM would otherwise spin on a stale
    value)."""
    return _llvm_d.LoadOp(
        T.i32,
        _gptr(addr_i64),
        alignment=4,
        volatile_=True,
        ordering=_llvm_d.AtomicOrdering.monotonic,
        syncscope="one-as",
    ).res


def load_i64_acquire(addr_i64):
    """Volatile monotonic i64 load."""
    return _llvm_d.LoadOp(
        T.i64,
        _gptr(addr_i64),
        alignment=8,
        volatile_=True,
        ordering=_llvm_d.AtomicOrdering.monotonic,
        syncscope="one-as",
    ).res


def load_i32_nt(base_i64, offset):
    """Non-temporal global i32 load at base + offset*4. A raw global load (VGPR
    address) avoids the per-expert buffer-descriptor waterfall; caller ensures the
    address is in-bounds."""
    return _llvm_d.LoadOp(
        T.i32, _ptr_plus(base_i64, offset, 4), alignment=4, nontemporal=True
    ).res


def load_v4i32_nt(base_i64, offset):
    """Non-temporal global vector<4xi32> load at base + offset*4 (global_load_dwordx4).
    offset is in i32 units; alignment=4 since the per-warp offset is only 4B-aligned."""
    return _llvm_d.LoadOp(
        T.i32x4, _ptr_plus(base_i64, offset, 4), alignment=4, nontemporal=True
    ).res


def load_v4i32(base_i64, offset):
    """Cached global vector<4xi32> load at base + offset*4. Unlike the ``_nt``
    variant this can be served from L1, so it only observes a peer's payload once
    an acquire fence has invalidated the stale lines."""
    return _llvm_d.LoadOp(T.i32x4, _ptr_plus(base_i64, offset, 4), alignment=16).res


def store_v4i32(base_i64, offset, val):
    """Plain global vector<4xi32> store at base + offset*4 (global_store_dwordx4).
    Single-producer payload write; ordering comes from a later release fence."""
    _llvm_d.StoreOp(
        arith.unwrap(val), _ptr_plus(base_i64, offset, 4), alignment=16
    )


# Backoff inserted between poll iterations, in s_sleep units (~64 clocks each);
# 0 disables it. The polling load is volatile and system-scoped, so it misses L1
# and reaches L2/fabric every iteration. That is affordable for a handful of
# waiters, but the fused gemm2 parks ~1k waves on the same loop at once, and the
# resulting request storm competes with the very P2P writes they are waiting for.
_SPIN_SLEEP = int(_os_environ.get("AITER_EP_SPIN_SLEEP", "1"))


def _spin(addr_i64, keep_waiting, *, width=32):
    """Spin on a volatile/atomic load at addr_i64 until keep_waiting(cur) is false;
    returns the awaited value. Self-contained (mori's wait_until_* need ShmemStates
    and cannot run on a cco-only stack)."""
    if width == 64:
        ty, load, wrap = T.i64, load_i64_acquire, fx.Int64
    else:
        ty, load, wrap = T.i32, load_i32_acquire, fx.Int32
    loop = scf.WhileOp([ty], [_unwrap(load(addr_i64))])
    cond = ir.Block.create_at_start(loop.before, [ty])
    body = ir.Block.create_at_start(loop.after, [ty])
    with ir.InsertionPoint(cond):
        scf.ConditionOp(
            _unwrap(keep_waiting(wrap(cond.arguments[0]))), [cond.arguments[0]]
        )
    with ir.InsertionPoint(body):
        if _SPIN_SLEEP:
            _rocdl.s_sleep(_SPIN_SLEEP)
        scf.YieldOp([_unwrap(load(addr_i64))])
    return wrap(loop.results[0])


def spin_until_eq_i64(addr_i64, val):
    """Spin until *addr (i64) == val."""
    return _spin(addr_i64, lambda cur: cur != fx.Int64(val), width=64)


def spin_until_eq_i32(addr_i64, val):
    """Spin until *addr == val."""
    return _spin(addr_i64, lambda cur: cur != fx.Int32(val))


def spin_until_gt_i32(addr_i64, val):
    """Spin until *addr > val (signed); returns the value seen."""
    return _spin(addr_i64, lambda cur: cur <= fx.Int32(val))


def spin_until_ge_i32(addr_i64, val):
    """Spin until *addr >= val (signed); returns the value seen. Epoch counters are
    monotonic, so a consumer that arrives late still passes immediately."""
    return _spin(addr_i64, lambda cur: cur < fx.Int32(val))


def spin_until_ge_i32_bounded(addr_i64, val, max_polls):
    """Bounded :func:`spin_until_ge_i32`; returns the last value this lane saw.

    The unbounded form only terminates if whoever publishes the flag is resident
    and progressing, which a kernel cannot assume on a GPU it does not have to
    itself. The poll budget is carried per wave, so every lane leaves the loop
    together and the caller decides with a ballot whether the wait succeeded --
    a lane that timed out simply reports a value below ``val``."""
    ty = T.i32
    loop = scf.WhileOp(
        [ty, ty],
        [_unwrap(load_i32_acquire(addr_i64)), _unwrap(arith.constant(0, type=ty))],
    )
    cond = ir.Block.create_at_start(loop.before, [ty, ty])
    body = ir.Block.create_at_start(loop.after, [ty, ty])
    with ir.InsertionPoint(cond):
        keep = arith.andi(
            fx.Int32(cond.arguments[0]) < fx.Int32(val),
            fx.Int32(cond.arguments[1]) < fx.Int32(max_polls),
        )
        scf.ConditionOp(_unwrap(keep), [cond.arguments[0], cond.arguments[1]])
    with ir.InsertionPoint(body):
        scf.YieldOp(
            [
                _unwrap(load_i32_acquire(addr_i64)),
                _unwrap(fx.Int32(body.arguments[1]) + fx.Int32(1)),
            ]
        )
    return fx.Int32(loop.results[0])
