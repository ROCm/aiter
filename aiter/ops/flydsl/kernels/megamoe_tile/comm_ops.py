# SPDX-License-Identifier: MIT
"""MegaMoE Tile communication primitives.

The generic helpers remain imported from AITER's shared communication module.
The acquire polling and last-arriver operations required by this operator live
here so MegaMoE Tile does not alter shared kernels merely for its own protocol.
"""

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl import expr as fx
from flydsl.expr import arith

from aiter.ops.flydsl.kernels.communication_ops_utils import (
    GeometryTuningTable,
    atomic_add_agent,
    atomic_add_global_at,
    atomic_add_system,
    fence_acquire,
    fence_agent_acquire,
    fence_agent_release,
    fence_release,
    fence_system_acquire,
    fence_system_release,
    load_i64_global,
    store_i32_system,
    store_i64_global_system,
)

__all__ = [
    "GeometryTuningTable",
    "atomic_add_agent",
    "atomic_add_agent_acq_rel",
    "atomic_add_global_at",
    "atomic_add_system",
    "atomic_add_system_acq_rel",
    "fence_acquire",
    "fence_agent_acquire",
    "fence_agent_release",
    "fence_release",
    "fence_system_acquire",
    "fence_system_release",
    "load_i32_global_system",
    "load_i64_global",
    "load_i64_global_system",
    "read_wall_clock",
    "store_i32_system",
    "store_i64_global_relaxed",
    "store_i64_global_system",
]


def _to_ptr_global(value):
    return _llvm_d.IntToPtrOp(
        _llvm_d.PointerType.get(address_space=1), arith.unwrap(value)
    ).result


def load_i32_global_system(addr_i64):
    """System-scope acquire i32 load for NIC/peer-written readiness words."""
    return _llvm_d.LoadOp(
        ir.IntegerType.get_signless(32),
        _to_ptr_global(addr_i64),
        alignment=4,
        ordering=_llvm_d.AtomicOrdering.acquire,
        syncscope="one-as",
    ).result


def load_i64_global_system(addr_i64):
    """System-scope acquire i64 load for NIC/peer-written generations."""
    return _llvm_d.LoadOp(
        ir.IntegerType.get_signless(64),
        _to_ptr_global(addr_i64),
        alignment=8,
        ordering=_llvm_d.AtomicOrdering.acquire,
        syncscope="one-as",
    ).result


def read_wall_clock():
    """Read gfx950's constant-frequency 64-bit wall clock."""

    return _llvm_d.inline_asm(
        ir.IntegerType.get_signless(64),
        [],
        "s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)",
        "=s",
        has_side_effects=True,
    )


def store_i64_global_relaxed(addr_i64, value):
    """Store a diagnostic timestamp without adding a system fence."""

    _llvm_d.StoreOp(
        arith.unwrap(value),
        _to_ptr_global(addr_i64),
        alignment=8,
    )


def atomic_add_system_acq_rel(addr_i64, value):
    """System-scope acquire-release fetch-add used by last-arriver logic."""
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add,
        _to_ptr_global(addr_i64),
        arith.unwrap(value),
        _llvm_d.AtomicOrdering.acq_rel,
        syncscope="one-as",
    ).res
def atomic_add_agent_acq_rel(addr_i64, value):
    """Agent-scope acquire-release fetch-add for local CTA handoff."""

    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add,
        _to_ptr_global(addr_i64),
        arith.unwrap(value),
        _llvm_d.AtomicOrdering.acq_rel,
        syncscope=fx.rocdl.SyncScope.Agent,
    ).res
