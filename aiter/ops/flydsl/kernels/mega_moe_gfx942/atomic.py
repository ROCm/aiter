# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Atomics between workgroups, and the coherent loads and stores that poll and publish
what they guard.

Global atomics are monotonic at agent scope (coherent across the XCD L2s) or at system
scope for another rank's memory, LDS atomics at workgroup scope. The "one-as" scopes
order the accessed address space only.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T

from .mem import _pointer


def _ptr_i32(addr):
    """i32 global pointer at the i64 device address ``addr``."""
    return _pointer(addr, fx.Int32, 4)


def _latomic_add(lds_i32, idx, val):
    """old = LDS i32 [idx] += val, workgroup scope."""
    ptr = fx.inttoptr(
        fx.PointerType.get(T.i32, fx.AddressSpace.Shared, 4),
        fx.Int64(lds_i32 + idx * fx.Int32(4)),
    )
    return fx.atomic_add(ptr, fx.Int32(val), syncscope="workgroup")


def _atomic_add_at(addr, value, scope):
    """Monotonic i32 fetch-add at an i64 address ("agent-one-as": device, "one-as":
    system, e.g. another rank's memory)."""
    return fx.atomic_add(_ptr_i32(addr), fx.Int32(value), syncscope=scope)


def _atomic_load_at(addr, scope):
    """Monotonic i32 load at an i64 address, coherent at ``scope`` (polling)."""
    return fx.generic_load(
        _ptr_i32(addr), memory_order=fx.AtomicOrdering.Monotonic, syncscope=scope
    )


def _sys_store_at(addr, value):
    """System-scope relaxed atomic store (sc0 sc1: written through) of an i32 at an i64
    address."""
    fx.generic_store(
        _ptr_i32(addr), fx.Int32(value), memory_order=fx.AtomicOrdering.Monotonic
    )


def _sys_load_at(addr):
    """System-scope relaxed atomic load (sc0 sc1) of an i32 at an i64 address."""
    return fx.generic_load(_ptr_i32(addr), memory_order=fx.AtomicOrdering.Monotonic)


def _inv_l1():
    """Invalidate the CU's L1 (data written through, or into the uncached MORI heap,
    before the flag a wave waited for)."""
    # Raw asm: FlyDSL 0.5.0 has no stable L1-only invalidate. fx.memory_fence(Acquire)
    # emits buffer_inv sc1 at agent scope (the L2 too) and nothing at workgroup scope
    # (gfx942 compile-only probe).
    llvm.inline_asm(None, [], "buffer_inv sc0", "", has_side_effects=True)


def _xcc_id():
    """The XCD (0..7) this wave runs on."""
    # Raw asm: FlyDSL 0.5.0 has no stable hardware-register read (s_getreg).
    return fx.Int32(
        llvm.inline_asm(
            T.i32,
            [],
            "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)",
            "=s",
            has_side_effects=True,
        )
    ) & fx.Int32(7)
