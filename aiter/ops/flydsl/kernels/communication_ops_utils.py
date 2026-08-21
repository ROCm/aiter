# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Low-level cross-card (P2P) communication primitives for communication kernels.

These wrap LLVM-dialect global memory ops with explicit memory ordering and
syncscope -- which the high-level FlyDSL APIs (buffer_ops / Pointer) do not
expose -- so dispatch/combine can publish and observe data across cards.

Also hosts :class:`GeometryTuningTable`, the per-shape launch-geometry lookup
shared by the dispatch/combine ops.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl._mlir.dialects import rocdl as _rocdl_d
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith
from flydsl.expr.typing import T

__all__ = [
    "GeometryTuningTable",
    "atomic_add_agent",
    "atomic_add_global_at",
    "atomic_add_system",
    "fence_acquire",
    "fence_agent_acquire",
    "fence_agent_release",
    "fence_release",
    "fence_system_acquire",
    "fence_system_release",
    "load_i32_acquire",
    "load_i32_nt",
    "load_i64_acquire",
    "load_i64_global",
    "load_v4i32_nt",
    "spin_until_eq_i32",
    "spin_until_ge_i64",
    "spin_until_gt_i32",
    "store_i32_system",
    "store_i64_global_system",
    "traced",
    "waitcnt_all",
]


def _to_ptr_global(v):
    """Cast an i64 address to ``!llvm.ptr<1>`` (global address space)."""
    return _llvm_d.IntToPtrOp(
        _llvm_d.PointerType.get(address_space=1), arith.unwrap(v)
    ).result


def _ptr_plus(base_i64, offset, elem_bytes):
    """Global pointer for base + offset*elem_bytes."""
    addr = (
        fx.Int64(arith.unwrap(base_i64)) + fx.Int64(arith.unwrap(offset)) * elem_bytes
    )
    return _to_ptr_global(addr)


def traced(fn):
    """Run FlyDSL's AST rewriting over a helper that kernel bodies call.

    ``@flyc.kernel`` and ``@flyc.jit`` apply this same transform to their own
    source but do not recurse into callees, so a helper that wants ``if`` /
    ``while`` over traced values (rather than a host-side truthiness test) has
    to opt in.
    """
    return ASTRewriter.transform(fn)


def waitcnt_all():
    """Drain outstanding gfx12 load/store counters before a grid barrier."""
    _rocdl_d.s_wait_storecnt(0)
    _rocdl_d.s_wait_loadcnt(0)


def load_i32_acquire(addr_i64):
    """Volatile monotonic i32 load suitable for a spin-wait."""
    return _llvm_d.LoadOp(
        T.i32,
        _to_ptr_global(addr_i64),
        alignment=4,
        volatile_=True,
        ordering=_llvm_d.AtomicOrdering.monotonic,
        syncscope="one-as",
    ).res


def load_i64_acquire(addr_i64):
    """Volatile monotonic i64 load suitable for a spin-wait."""
    return _llvm_d.LoadOp(
        T.i64,
        _to_ptr_global(addr_i64),
        alignment=8,
        volatile_=True,
        ordering=_llvm_d.AtomicOrdering.monotonic,
        syncscope="one-as",
    ).res


def load_i32_nt(base_i64, offset):
    """Non-temporal global i32 load at base + offset*4."""
    return _llvm_d.LoadOp(
        T.i32, _ptr_plus(base_i64, offset, 4), alignment=4, nontemporal=True
    ).res


def load_v4i32_nt(base_i64, offset):
    """Non-temporal global vector<4xi32> load at base + offset*4."""
    return _llvm_d.LoadOp(
        T.i32x4, _ptr_plus(base_i64, offset, 4), alignment=4, nontemporal=True
    ).res


@traced
def spin_until_ge_i64(addr_i64, val):
    """Spin until a monotonic cross-device i64 flag is at least ``val``."""
    cur = fx.Int64(load_i64_acquire(addr_i64))
    while cur < fx.Int64(val):
        cur = fx.Int64(load_i64_acquire(addr_i64))
    return cur


@traced
def spin_until_eq_i32(addr_i64, val):
    """Spin until an i32 flag equals ``val``."""
    cur = fx.Int32(load_i32_acquire(addr_i64))
    while cur != fx.Int32(val):
        cur = fx.Int32(load_i32_acquire(addr_i64))
    return cur


@traced
def spin_until_gt_i32(addr_i64, val):
    """Spin until an i32 flag exceeds ``val`` and return the observed value."""
    cur = fx.Int32(load_i32_acquire(addr_i64))
    while cur <= fx.Int32(val):
        cur = fx.Int32(load_i32_acquire(addr_i64))
    return cur


def store_i32_system(addr_i64, offset, val):
    """System-scope release i32 store at ``addr_i64 + offset*4``."""
    base = arith.unwrap(addr_i64)
    off = arith.unwrap(offset)
    val_ = arith.unwrap(val)
    _i64 = ir.IntegerType.get_signless(64)
    _i32 = ir.IntegerType.get_signless(32)
    _nuw = ir.Attribute.parse("#llvm.overflow<none>")
    off64 = _llvm_d.ZExtOp(_i64, off).res if off.type == _i32 else off
    byte_off = _llvm_d.MulOp(
        off64, _llvm_d.ConstantOp(_i64, ir.IntegerAttr.get(_i64, 4)).result, _nuw
    ).result
    addr = _llvm_d.AddOp(base, byte_off, _nuw).result
    gptr = _llvm_d.IntToPtrOp(_llvm_d.PointerType.get(address_space=1), addr).result
    _llvm_d.StoreOp(
        val_,
        gptr,
        alignment=4,
        ordering=_llvm_d.AtomicOrdering.release,
        syncscope="one-as",
    )


def store_i64_global_system(addr_i64, val):
    """System-scope release i64 store to ``addr_i64``."""
    gptr = _to_ptr_global(addr_i64)
    _llvm_d.StoreOp(
        arith.unwrap(val),
        gptr,
        alignment=8,
        ordering=_llvm_d.AtomicOrdering.release,
        syncscope="one-as",
    )


def fence_acquire(syncscope):
    """Emit an acquire fence for the selected AMDGPU memory scope."""
    _llvm_d.FenceOp(_llvm_d.AtomicOrdering.acquire, syncscope=syncscope)


def fence_release(syncscope):
    """Emit a release fence for the selected AMDGPU memory scope."""
    _llvm_d.FenceOp(_llvm_d.AtomicOrdering.release, syncscope=syncscope)


def fence_system_acquire():
    """System-scope acquire fence."""
    fence_acquire(fx.rocdl.SyncScope.OneAs)


def fence_system_release():
    """System-scope release fence."""
    fence_release(fx.rocdl.SyncScope.OneAs)


def fence_agent_acquire():
    """Agent-scope acquire fence."""
    fence_acquire(fx.rocdl.SyncScope.AgentOneAs)


def fence_agent_release():
    """Agent-scope release fence."""
    fence_release(fx.rocdl.SyncScope.AgentOneAs)


def load_i64_global(addr_i64):
    """Relaxed global i64 load from ``addr_i64``."""
    ptr = _to_ptr_global(addr_i64)
    _i64 = ir.IntegerType.get_signless(64)
    return _llvm_d.LoadOp(_i64, ptr, alignment=8).result


def atomic_add_global_at(addr_i64, val, syncscope="one-as"):
    """Monotonic global fetch-add with configurable agent/system visibility."""
    ptr = _to_ptr_global(addr_i64)
    kwargs = {} if syncscope is None else {"syncscope": syncscope}
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add,
        ptr,
        arith.unwrap(val),
        _llvm_d.AtomicOrdering.monotonic,
        **kwargs,
    ).res


def atomic_add_agent(addr_i64, val):
    """Agent-scope monotonic global fetch-and-add."""
    return atomic_add_global_at(addr_i64, val, syncscope=fx.rocdl.SyncScope.Agent)


def atomic_add_system(addr_i64, val):
    """System-scope monotonic global fetch-and-add."""
    return atomic_add_global_at(addr_i64, val)


@dataclass
class GeometryTuningTable:
    """Per-shape token-count -> (block_num, warp_num_per_block) lookup; rounds up
    to the smallest bucket >= count (largest on overflow, mori parity)."""

    dispatch: dict[int, tuple[int, int]] = field(default_factory=dict)
    combine: dict[int, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self):
        for phase, tbl in (("dispatch", self.dispatch), ("combine", self.combine)):
            for n_tok, (bn, wpb) in tbl.items():
                if bn <= 0 or wpb <= 0:
                    raise ValueError(
                        f"GeometryTuningTable.{phase}[{n_tok}] must be positive, "
                        f"got block_num={bn}, warp_num_per_block={wpb}"
                    )

    @classmethod
    def from_tuning_file(
        cls,
        path,
        *,
        dtype,
        hidden_dim,
        zero_copy,
        topk=None,
        local_expert_num=None,
        combine_dtype="bf16",
    ):
        """Build a per-op table from a multi-shape tuning JSON, filtered to this
        op's shape; empty table => cfg defaults."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        def _match(r, want_dtype, need_zc):
            if (
                r.get("dtype") != want_dtype
                or int(r.get("hidden_dim", -1)) != hidden_dim
            ):
                return False
            if topk is not None and "topk" in r and int(r["topk"]) != topk:
                return False
            if (
                local_expert_num is not None
                and "local_expert_num" in r
                and int(r["local_expert_num"]) != local_expert_num
            ):
                return False
            return not (need_zc and bool(r.get("zero_copy", False)) != bool(zero_copy))

        def _build(rules, want_dtype, need_zc):
            return {
                int(r["num_tokens"]): (
                    int(r["block_num"]),
                    int(r["warp_num_per_block"]),
                )
                for r in rules
                if _match(r, want_dtype, need_zc)
            }

        return cls(
            dispatch=_build(raw.get("dispatch", []), dtype, need_zc=False),
            combine=_build(raw.get("combine", []), combine_dtype, need_zc=True),
        )

    def lookup(self, phase, num_tokens):
        """Smallest bucket >= num_tokens (largest on overflow); None if empty."""
        tbl = self.dispatch if phase == "dispatch" else self.combine
        if not tbl:
            return None
        if num_tokens in tbl:
            return tbl[num_tokens]
        candidates = [k for k in tbl if k >= num_tokens]
        return tbl[min(candidates)] if candidates else tbl[max(tbl)]
