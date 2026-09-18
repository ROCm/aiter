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
    traced,
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
    "atomic_add_system_release",
    "fence_acquire",
    "fence_agent_acquire",
    "fence_agent_release",
    "fence_release",
    "fence_system_acquire",
    "fence_system_release",
    "load_i32_global_system",
    "load_i64_global_system",
    "spin_until_ge_i64_system",
    "load_i32_global_system_relaxed",
    "load_i32_global_agent_relaxed",
    "load_i64_global_agent_relaxed",
    "load_i64_global",
    "load_i64_global_system",
    "load_i64_global_system_relaxed",
    "read_hw_id",
    "read_wall_clock",
    "store_i32_system",
    "store_i64_global_relaxed",
    "store_i64_global_system",
    "store_i64_global_system_relaxed",
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
    """System-scope acquire i64 load for NIC/peer-written generation words.

    The rail credit/ready words are written by a remote node's RDMA, so the
    load must be system-scoped AND acquire: observing the generation has to
    publish the payload the peer wrote before it. ``one-as`` is system scope
    restricted to one address space, which is what every other system load in
    this module uses. Note that the shared
    ``communication_ops_utils.load_i64_acquire`` is, despite its name, only
    *monotonic* -- swapping it in here would silently weaken the ordering, and
    that shows up as slowly-wrong numbers rather than a crash.
    """
    return _llvm_d.LoadOp(
        ir.IntegerType.get_signless(64),
        _to_ptr_global(addr_i64),
        alignment=8,
        ordering=_llvm_d.AtomicOrdering.acquire,
        syncscope="one-as",
    ).result


def load_i32_global_system_relaxed(addr_i64):
    """Poll a peer-written atomic counter, acquiring its payload only once.

    The system scope is necessary even when the counter resides on this GPU:
    its producers are peer GPUs. A monotonic atomic load still observes their
    RMW updates. After the expected count is observed, the consuming wave must
    issue a system acquire fence before reading inbox payloads. This pairs with
    each producer's release fence and following relaxed RMW, while avoiding an
    L2 invalidation on every unsuccessful poll. Never use a plain cached load
    or an agent-scoped load for this inter-GPU handoff.
    """
    return _llvm_d.LoadOp(
        ir.IntegerType.get_signless(32),
        _to_ptr_global(addr_i64),
        alignment=4,
        ordering=_llvm_d.AtomicOrdering.monotonic,
        syncscope="one-as",
    ).result


def load_i32_global_agent_relaxed(addr_i64):
    """Poll a same-GPU atomic counter; acquire once after readiness is met.

    Monotonic atomic loads remain observable on every loop iteration. The
    caller must issue an agent acquire fence before consuming producer data.
    This avoids system L2 invalidation on every unsuccessful local poll.
    """
    return _llvm_d.LoadOp(
        ir.IntegerType.get_signless(32),
        _to_ptr_global(addr_i64),
        alignment=4,
        ordering=_llvm_d.AtomicOrdering.monotonic,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    ).result


def load_i64_global_agent_relaxed(addr_i64):
    """Read a local grid ticket without adding another contending RMW.

    Each CTA releases once through the ticket increment. After observing the
    full grid count, the caller acquires those releases before continuing.
    """
    return _llvm_d.LoadOp(
        ir.IntegerType.get_signless(64),
        _to_ptr_global(addr_i64),
        alignment=8,
        ordering=_llvm_d.AtomicOrdering.monotonic,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
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


def load_i64_global_system_relaxed(addr_i64):
    """Poll a peer-written u64 ready word without invalidating L2 each time.

    cco's wait_ready polls with system-*acquire* semantics, and every acquire
    load carries a buffer_inv.  That is right for a single handoff and ruinous
    for a bulk wait: kernel2 checks 512 arrival flags per chunk from each of
    ~10 CTAs, so the acquire form spends 82k L2 invalidations to observe words
    that are almost always already set -- measured at +39us.  This is the
    mirror image of the release-store writeback that cost kernel1 +6.7ms.

    Monotonic keeps the one-as scope, so a peer's write is still observed on
    every iteration; only the invalidation goes away.  The caller owns the
    ordering and MUST issue one fence_system_acquire() after the wait completes
    and before reading any payload the flag guards.
    """
    return _llvm_d.LoadOp(
        ir.IntegerType.get_signless(64),
        _to_ptr_global(addr_i64),
        alignment=8,
        ordering=_llvm_d.AtomicOrdering.monotonic,
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


def read_hw_id():
    """Read gfx9 HW_ID: WAVE[3:0] SIMD[5:4] PIPE[7:6] CU[11:8] SH[12] SE[15:13]
    TG[19:16] VM[23:20] QUEUE[26:24] STATE[29:27] ME[31:30].

    XCC_ID is deliberately not folded in, so these ids repeat across the part's
    8 XCDs: enough to spot a CU-level tail, not to name a physical CU. ME_ID in
    the top bits makes the i32 look negative -- mask with 0xFFFFFFFF host-side.
    """

    return _llvm_d.inline_asm(
        ir.IntegerType.get_signless(32),
        [],
        "s_getreg_b32 $0, hwreg(HW_REG_HW_ID)",
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


def store_i64_global_system_relaxed(addr_i64, value):
    """System-scope store with no release ordering.

    store_i64_global_system is a *release* store, and LLVM gives every one of
    those a buffer_wbl2 -- a full L2 writeback.  That is right when the store
    is what publishes the payload, and ruinous when the payload was already
    flushed by a fence the caller issued once for a whole batch: measured at
    +6.7ms on kernel1 for ~18k of them per generation.  Monotonic keeps the
    one-as scope, so the store still lands where a peer can see it, and drops
    the writeback.  The caller owns the ordering.
    """
    _llvm_d.StoreOp(
        arith.unwrap(value),
        _to_ptr_global(addr_i64),
        alignment=8,
        ordering=_llvm_d.AtomicOrdering.monotonic,
        syncscope="one-as",
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


def atomic_add_system_release(addr_i64, value):
    """Publish peer payload completion without acquiring consumer data.

    Callers join the payload-writing lanes before publication. The release RMW
    supplies the system writeback itself; a preceding system release fence
    would duplicate that expensive writeback on gfx950. Consumer acquire reads
    synchronize with the contributors' RMW release sequences on this counter.
    Use acq_rel instead when the returned value determines a last-arriver action.
    """
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add,
        _to_ptr_global(addr_i64),
        arith.unwrap(value),
        _llvm_d.AtomicOrdering.release,
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


@traced
def spin_until_ge_i64_system(addr_i64, expected):
    """Poll a monotonic u64 ready/credit word written by a peer's NIC.

    Acquire + system scope on every load: observing the generation must also
    publish the payload the remote node wrote before it.
    """
    cur = fx.Int64(load_i64_global_system(addr_i64))
    while cur < fx.Int64(expected):
        cur = fx.Int64(load_i64_global_system(addr_i64))
    return cur
