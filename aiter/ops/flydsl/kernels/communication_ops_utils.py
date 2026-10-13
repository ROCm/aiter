# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Cross-card (P2P) communication primitives for communication kernels.

FlyDSL's generic memory and atomic APIs carry the required memory ordering,
syncscope, address space, and alignment for dispatch/combine synchronization.

Also hosts :class:`GeometryTuningTable`, the per-shape launch-geometry lookup
shared by the dispatch/combine ops.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr.typing import T

__all__ = [
    "GeometryTuningTable",
    "atomic_add_agent",
    "atomic_add_agent_one_as",
    "atomic_add_global_at",
    "atomic_add_lds",
    "atomic_add_system",
    "atomic_add_workgroup",
    "fence_acquire",
    "fence_agent_acquire",
    "fence_agent_release",
    "fence_release",
    "fence_system_acquire",
    "fence_system_release",
    "load_i32_acquire",
    "load_i32_global_agent",
    "load_i32_global_system",
    "load_i32_lds",
    "load_i32_nt",
    "load_i32_system",
    "load_i64_acquire",
    "load_i64_global",
    "load_v4i32_nt",
    "spin_until_eq_i32",
    "spin_until_ge_i32_agent",
    "spin_until_ge_i32_system",
    "spin_until_ge_i64",
    "spin_until_gt_i32",
    "store_i32_global_agent_release",
    "store_i32_global_system_monotonic",
    "store_i32_global_system_release",
    "store_i32_lds",
    "store_i32_system",
    "store_i64_global_system",
    "wait_i32_until_equals",
    "wait_i32_until_greater_than",
    "wait_i64_until_equals",
    "waitcnt_all",
    "waitcnt_stores",
]


def _to_ptr_global(v, dtype=fx.Int32, alignment=4):
    """Cast an integer address to an aligned global FlyDSL pointer."""
    ptr_type = fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Global, alignment)
    return fx.inttoptr(ptr_type, fx.Int64(v))


def _to_ptr_shared(v):
    """Cast an integer address to an aligned shared i32 pointer."""
    ptr_type = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 4)
    return fx.inttoptr(ptr_type, fx.Int64(v))


_to_ptr_lds = _to_ptr_shared


def _ptr_plus(base_i64, offset, elem_bytes):
    """Global pointer for base + offset*elem_bytes."""
    addr = fx.Int64(base_i64) + fx.Int64(offset) * elem_bytes
    return _to_ptr_global(addr)


def waitcnt_all():
    """Drain outstanding gfx12 load/store counters before a grid barrier."""
    fx.rocdl.s_wait_storecnt(0)
    fx.rocdl.s_wait_loadcnt(0)


def waitcnt_stores():
    """Drain outstanding stores only -- the narrow half of :func:`waitcnt_all`.

    Store completion alone is what makes those writes visible to a peer released
    by a grid barrier; in-flight loads need no wait, because their results are
    already ordered by the register dependencies that consume them.
    """
    fx.rocdl.s_wait_storecnt(0)


def atomic_add_lds(addr_i64, val):
    """Workgroup-scope LDS fetch-and-add at ``addr_i64``; returns the old value.

    ``addr_i64`` is a raw LDS byte address (``ptrtoint`` of a SharedAllocator
    pointer), not a global one: the block-local counters this serves are hit by
    every lane of every warp in the block, which in global memory would cost a
    device-scope atomic per route.
    """
    return fx.atomic_add(_to_ptr_lds(addr_i64), fx.Int32(val), syncscope="workgroup")


def load_i32_lds(addr_i64):
    """Plain i32 load from a raw LDS byte address."""
    return fx.ptr_load(_to_ptr_lds(addr_i64))


def store_i32_lds(addr_i64, val):
    """Plain i32 store to a raw LDS byte address."""
    fx.ptr_store(fx.Int32(val), _to_ptr_lds(addr_i64))


def load_i32_acquire(addr_i64):
    """Volatile monotonic i32 load suitable for a spin-wait."""
    return fx.generic_load(
        _to_ptr_global(addr_i64),
        volatile=True,
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope="one-as",
    )


def load_i32_global_system(addr_i64, *, acquire=False):
    """Volatile i32 load with system visibility for a spin-wait."""
    return fx.generic_load(
        _to_ptr_global(addr_i64),
        volatile=True,
        memory_order=(
            fx.AtomicOrdering.Acquire if acquire else fx.AtomicOrdering.Monotonic
        ),
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


def load_i32_global_agent(addr_i64):
    """Volatile monotonic i32 load with agent-one-as visibility."""
    return fx.generic_load(
        _to_ptr_global(addr_i64),
        volatile=True,
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    )


def load_i64_acquire(addr_i64):
    """Volatile monotonic i64 load suitable for a spin-wait."""
    return fx.generic_load(
        _to_ptr_global(addr_i64, fx.Int64, 8),
        volatile=True,
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope="one-as",
    )


def load_i32_system(addr_i64, index):
    """Compatibility system-scope i32 load at ``addr_i64 + index * 4``."""
    item_addr = fx.Int64(addr_i64) + fx.Int64(index) * fx.Int64(4)
    return fx.Int32(load_i32_acquire(item_addr))


def load_i32_nt(base_i64, offset):
    """Non-temporal global i32 load at base + offset*4."""
    return fx.generic_load(_ptr_plus(base_i64, offset, 4), nontemporal=True)


def load_v4i32_nt(base_i64, offset):
    """Non-temporal global vector<4xi32> load at base + offset*4."""
    return fx.generic_load(_ptr_plus(base_i64, offset, 4), count=4, nontemporal=True)


@flyc.jit
def spin_until_ge_i64(addr_i64, val):
    """Spin until a monotonic cross-device i64 flag is at least ``val``."""
    cur = fx.Int64(load_i64_acquire(addr_i64))
    while cur < fx.Int64(val):
        cur = fx.Int64(load_i64_acquire(addr_i64))
    return cur


@flyc.jit
def spin_until_ge_i32_system(addr_i64, val, *, acquire=False, sleep=True):
    """Spin on a system-visible i32 flag until it reaches ``val``."""
    cur = fx.Int32(load_i32_global_system(addr_i64, acquire=acquire))
    while cur < fx.Int32(val):
        if fx.const_expr(sleep):
            fx.rocdl.s_sleep(1)
        cur = fx.Int32(load_i32_global_system(addr_i64, acquire=acquire))
    return cur


@flyc.jit
def spin_until_ge_i32_agent(addr_i64, val, *, sleep=True):
    """Spin on an agent-visible i32 flag until it reaches ``val``."""
    cur = fx.Int32(load_i32_global_agent(addr_i64))
    while cur < fx.Int32(val):
        if fx.const_expr(sleep):
            fx.rocdl.s_sleep(1)
        cur = fx.Int32(load_i32_global_agent(addr_i64))
    return cur


@flyc.jit
def spin_until_eq_i32(addr_i64, val):
    """Spin until an i32 flag equals ``val``."""
    cur = fx.Int32(load_i32_acquire(addr_i64))
    while cur != fx.Int32(val):
        cur = fx.Int32(load_i32_acquire(addr_i64))
    return cur


@flyc.jit
def spin_until_gt_i32(addr_i64, val):
    """Spin until an i32 flag exceeds ``val`` and return the observed value."""
    cur = fx.Int32(load_i32_acquire(addr_i64))
    while cur <= fx.Int32(val):
        cur = fx.Int32(load_i32_acquire(addr_i64))
    return cur


def wait_i32_until_equals(addr_i64, expected):
    """Compatibility wrapper for the historical MegaMoE wait helper."""
    spin_until_eq_i32(addr_i64, expected)


def wait_i32_until_greater_than(addr_i64, expected):
    """Compatibility wrapper returning the first i32 value above ``expected``."""
    return spin_until_gt_i32(addr_i64, expected)


@flyc.jit
def wait_i64_until_equals(addr_i64, expected):
    """Spin until a system-visible i64 flag equals ``expected``."""
    cur = fx.Int64(load_i64_acquire(addr_i64))
    while cur != fx.Int64(expected):
        cur = fx.Int64(load_i64_acquire(addr_i64))


def store_i32_system(addr_i64, offset, val):
    """System-scope release i32 store at ``addr_i64 + offset*4``."""
    off = arith.unwrap(offset)
    off64 = fx.Uint64(fx.Uint32(off)) if off.type == T.i32 else fx.Int64(off)
    addr = fx.Int64(addr_i64) + fx.Int64(off64) * 4
    fx.generic_store(
        _to_ptr_global(addr),
        fx.Int32(val),
        memory_order=fx.AtomicOrdering.Release,
        syncscope="one-as",
    )


def store_i64_global_system(addr_i64, val):
    """System-scope release i64 store to ``addr_i64``."""
    fx.generic_store(
        _to_ptr_global(addr_i64, fx.Int64, 8),
        fx.Int64(val),
        memory_order=fx.AtomicOrdering.Release,
        syncscope="one-as",
    )


def store_i32_global_agent_release(addr_i64, val):
    """Agent-one-as release store to a global i32 address."""
    fx.generic_store(
        _to_ptr_global(addr_i64),
        fx.Int32(val),
        memory_order=fx.AtomicOrdering.Release,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    )


def store_i32_global_system_monotonic(addr_i64, val):
    """System-visible monotonic store to a global i32 address."""
    fx.generic_store(
        _to_ptr_global(addr_i64),
        fx.Int32(val),
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


def store_i32_global_system_release(addr_i64, val):
    """System-visible release store to a global i32 address."""
    fx.generic_store(
        _to_ptr_global(addr_i64),
        fx.Int32(val),
        memory_order=fx.AtomicOrdering.Release,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


def fence_acquire(syncscope):
    """Emit an acquire fence for the selected AMDGPU memory scope."""
    fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope=syncscope)


def fence_release(syncscope):
    """Emit a release fence for the selected AMDGPU memory scope."""
    fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope=syncscope)


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
    return fx.ptr_load(_to_ptr_global(addr_i64, fx.Int64, 8))


def atomic_add_global_at(addr_i64, val, syncscope="one-as"):
    """Monotonic global fetch-add with configurable agent/system visibility."""
    dtype = (
        fx.Int32
        if isinstance(val, int)
        else fx.Int64 if fx.as_ir_value(val).type == T.i64 else fx.Int32
    )
    ptr = _to_ptr_global(addr_i64, dtype, dtype.width // 8)
    return fx.atomic_add(ptr, dtype(val), syncscope=syncscope)


def atomic_add_agent(addr_i64, val):
    """Agent-scope monotonic global fetch-and-add."""
    return atomic_add_global_at(addr_i64, val, syncscope=fx.rocdl.SyncScope.Agent)


def atomic_add_agent_one_as(addr_i64, val):
    """Agent-one-as monotonic global fetch-and-add."""
    return atomic_add_global_at(addr_i64, val, syncscope=fx.rocdl.SyncScope.AgentOneAs)


def atomic_add_system(addr_i64, val):
    """System-scope monotonic global fetch-and-add."""
    return atomic_add_global_at(addr_i64, val)


def atomic_add_workgroup(addr_i64, val):
    """Workgroup-scope monotonic shared-memory fetch-and-add."""
    return fx.atomic_add(_to_ptr_shared(addr_i64), fx.Int32(val), syncscope="workgroup")


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
        ep_size,
        gfx=None,
        gpu_model=None,
        dtype,
        hidden_dim,
        zero_copy,
        topk=None,
        local_expert_num=None,
        combine_dtype="bf16",
    ):
        """Build a per-op table from ``tuned_dispatch_combine_intranode.csv``."""
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(
                csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
            )

        def _row_match(r, want_dtype, need_zc):
            if (
                r.get("dtype") != want_dtype
                or int(r.get("hidden_dim", -1)) != hidden_dim
            ):
                return False
            if int(r.get("ep_size", ep_size)) != int(ep_size):
                return False
            row_gfx = (r.get("gfx") or "").strip()
            if gfx and row_gfx and row_gfx != gfx:
                return False
            row_model = (r.get("gpu_model") or "").strip()
            if gpu_model and row_model and row_model != gpu_model:
                return False
            if topk is not None and r.get("topk") and int(r["topk"]) != topk:
                return False
            if (
                local_expert_num is not None
                and r.get("local_expert_num")
                and int(r["local_expert_num"]) != local_expert_num
            ):
                return False
            if need_zc:
                zc_raw = (r.get("zero_copy") or "").strip()
                if zc_raw:
                    row_zc = zc_raw.lower() in ("1", "true", "yes")
                    if row_zc != bool(zero_copy):
                        return False
            return True

        def _build(phase, want_dtype, need_zc):
            out = {}
            for r in rows:
                if r.get("phase") != phase or not _row_match(r, want_dtype, need_zc):
                    continue
                out[int(r["num_tokens"])] = (
                    int(r["block_num"]),
                    int(r["warp_num_per_block"]),
                )
            return out

        return cls(
            dispatch=_build("dispatch", dtype, need_zc=False),
            combine=_build("combine", combine_dtype, need_zc=True),
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
