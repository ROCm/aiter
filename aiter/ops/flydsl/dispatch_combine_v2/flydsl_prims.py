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
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl._mlir.dialects import scf
from flydsl.expr import arith
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


def store_i32_system(addr_i64, offset, val):
    """System-release i32 store at addr + offset*4."""
    _llvm_d.StoreOp(
        arith.unwrap(val),
        _ptr_plus(addr_i64, offset, 4),
        alignment=4,
        ordering=_llvm_d.AtomicOrdering.release,
        syncscope="one-as",
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
