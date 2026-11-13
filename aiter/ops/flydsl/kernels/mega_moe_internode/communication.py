# SPDX-License-Identifier: MIT
"""Device-side communication of MegaMoEInternode.

Primitives: the generic helpers are imported from AITER's shared communication
module; the acquire polling and last-arriver operations this operator's
protocol needs live here so it does not alter shared kernels.

Rail GDA (``rail``): MORI's public FlyDSL binding (``mori.cco.device.flydsl``,
ROCm/mori PR 895 and later) with the rail team (``Team.GDA``: ``peer`` is a node
index) and deferred doorbells (post WQEs without ringing; the next flush rings
once per QP).  Every rail post here is caller-ordered: the flusher knows all
posts on its QP have returned before it flushes, so posts skip the per-post
ready mark and its system fence.  MORI with ``GdaDoorbell`` (PR 895 final)
takes that as the handle mode ``CALLER_ORDERED``; older PR 895 builds take the
per-call ``defer_doorbell`` (+ ``posted_before_flush`` when present).  See
``_gda``.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import mori.cco.device.flydsl as cco
from flydsl import expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl._mlir.dialects import rocdl as _mlir_rocdl

from aiter.ops.flydsl.kernels.communication_ops_utils import (
    atomic_add_agent,
    atomic_add_system,
    fence_system_acquire,
    fence_system_release,
    load_i64_global,
    store_i32_system,
    store_i64_global_system,
)

__all__ = [
    "atomic_add_agent",
    "atomic_add_agent_acq_rel",
    "atomic_add_system",
    "atomic_add_system_acq_rel",
    "fence_system_acquire",
    "fence_system_release",
    "load_i32_global_system",
    "load_i32_global_system_relaxed",
    "load_i64_global",
    "load_i64_global_agent_relaxed",
    "load_i64_global_system",
    "load_i64_global_system_relaxed",
    "spin_until_ge_i64_sleep_rx",
    "spin_until_ge_i64_system_rx",
    "store_i32_global_system_relaxed",
    "store_i32_system",
    "store_i64_global_relaxed",
    "store_i64_global_system",
    "store_i64_global_system_relaxed",
]


def _to_ptr_global(value):
    ptr_ty = fx.PointerType.get(fx.Int8.ir_type, fx.AddressSpace.Global)
    return fx.to_llvm_ptr(fx.inttoptr(ptr_ty, fx.Int64(value)))


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


def store_i64_global_relaxed(addr_i64, value):
    """Plain (non-atomic) i64 global store, no fence and no ordering.

    For words whose visibility is published by a later release in the same
    thread: stage1's ``expert_tile_map_ready`` generations and the stage2
    peer-address table filled before launch.
    """

    _llvm_d.StoreOp(
        value.ir_value(),
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
        value.ir_value(),
        _to_ptr_global(addr_i64),
        alignment=8,
        ordering=_llvm_d.AtomicOrdering.monotonic,
        syncscope="one-as",
    )


def store_i32_global_system_relaxed(addr_i64, value):
    """i32 form of :func:`store_i64_global_system_relaxed` (no writeback)."""
    _llvm_d.StoreOp(
        fx.Int32(value).ir_value(),
        _to_ptr_global(addr_i64),
        alignment=4,
        ordering=_llvm_d.AtomicOrdering.monotonic,
        syncscope="one-as",
    )


def atomic_add_system_acq_rel(addr_i64, value):
    """System-scope acquire-release fetch-add used by last-arriver logic."""
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add,
        _to_ptr_global(addr_i64),
        value.ir_value(),
        _llvm_d.AtomicOrdering.acq_rel,
        syncscope="one-as",
    ).res


def atomic_add_agent_acq_rel(addr_i64, value):
    """Agent-scope acquire-release fetch-add for local CTA handoff."""

    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add,
        _to_ptr_global(addr_i64),
        value.ir_value(),
        _llvm_d.AtomicOrdering.acq_rel,
        syncscope=fx.rocdl.SyncScope.Agent,
    ).res


@flyc.jit
def spin_until_ge_i64_system_rx(addr_i64, expected):
    """Poll as in spin_until_ge_i64_system, using relaxed system loads without buffer_inv.
    Perform one acquire load after observing the target value, before reading
    any protected data. This preserves ordering and invalidates L2 once per wait
    instead of once per polling iteration.
    """
    cur = fx.Int64(load_i64_global_system_relaxed(addr_i64))
    while cur < fx.Int64(expected):
        cur = fx.Int64(load_i64_global_system_relaxed(addr_i64))
    return fx.Int64(load_i64_global_system(addr_i64))


@flyc.jit
def spin_until_ge_i64_sleep_rx(addr_i64, expected, sleep):
    """Poll as in spin_until_ge_i64_sleep, with one acquire load on exit (see _system_rx)."""
    cur = fx.Int64(load_i64_global_system_relaxed(addr_i64))
    while cur < fx.Int64(expected):
        _mlir_rocdl.s_sleep(sleep)
        cur = fx.Int64(load_i64_global_system_relaxed(addr_i64))
    return fx.Int64(load_i64_global_system(addr_i64))


def ccqe_enabled() -> bool:
    """Whether MORI creates this host's ionic CQs in CCQE (collapsed) mode.

    MORI keys its own GDA bitcode on the CQ mode, but a FlyDSL kernel that
    inlined the CQ poll is cached by its name and closure, not by MORI's mode.
    A kernel compiled under the other mode polls the wrong CQ layout: it keeps
    working for a while and then hangs on a completion it misreads. So kernels
    that flush the rail carry the mode in their name.
    """
    try:
        from mori.jit.core import is_ccqe_enabled
    except ImportError:
        return False
    return bool(is_ccqe_enabled())


_WARP = cco.CoopScope.WARP
_GDA = cco.Team.GDA


def _posted_before_flush_supported() -> bool:
    try:
        from mori.cco.device.ops import GdaFlags
    except ImportError:
        return False
    return hasattr(GdaFlags, "POSTED_BEFORE_FLUSH")


# Caller-ordered posting.  Every rail QP here meets its contract (all posts on
# the QP have returned when a flush starts; only these handles touch the rail
# contexts; fewer WQEs than the SQ between flushes): stage2 k2's flushing wave
# waits for the WQE_POSTED count before it flushes, and stage1 posts a credit on
# a QP only after that QP's dispatch request has completed.  Without it a
# deferred post publishes a ready mark behind a system fence and the flush
# sweeps the marks.
_PBF = {"posted_before_flush": True} if _posted_before_flush_supported() else {}


def _doorbell_modes():
    try:
        from mori.cco.device.ops import GdaDoorbell
    except ImportError:
        return None
    return GdaDoorbell


_DOORBELL = _doorbell_modes()


def _gda(dev_comm, *, deferred=True):
    """(handle, extra per-call kwargs) for a rail post or flush.

    New MORI: the doorbell mode is the handle's (CALLER_ORDERED for the
    deferred posts and their flushes).  Older PR 895: one default handle and the
    mode as per-call flags."""
    if _DOORBELL is not None:
        mode = _DOORBELL.CALLER_ORDERED if deferred else _DOORBELL.IMMEDIATE
        return cco.DevComm(dev_comm).gda(0, doorbell=mode), {}
    if not deferred:
        return cco.DevComm(dev_comm).gda(0), {}
    return cco.DevComm(dev_comm).gda(0), {"defer_doorbell": True, **_PBF}


def _gda_flush(dev_comm):
    """Handle + kwargs for a flush that releases the caller-ordered posts."""
    if _DOORBELL is not None:
        return cco.DevComm(dev_comm).gda(0, doorbell=_DOORBELL.CALLER_ORDERED), {}
    return cco.DevComm(dev_comm).gda(0), dict(_PBF)


@flyc.jit
def _virtual_put(
    dev_comm: fx.Int64,
    ctx: fx.Int32,
    peer: fx.Int32,
    dst_win: fx.Int64,
    dst_off: fx.Int64,
    src_win: fx.Int64,
    src_off: fx.Int64,
    nbytes: fx.Int64,
    rank: fx.Constexpr[int],
    physical_width: fx.Constexpr[int],
    aggregate: fx.Constexpr[bool],
):
    if peer // fx.Int32(physical_width) == fx.Int32(rank // physical_width):
        dst = fx.Int64(
            cco.Window(dst_win).lsa_ptr(peer % fx.Int32(physical_width), dst_off)
        )
        src = fx.Int64(
            cco.Window(src_win).lsa_ptr(fx.Int32(rank % physical_width), src_off)
        )
        lane = fx.Int32(fx.thread_idx.x) % fx.Int32(64)
        # Protocol payloads are multiples of eight bytes. Only the posting
        # wave enters; a workgroup barrier here would strand the other waves.
        # C1 bounds each registered rail window to 1 GiB. FlyDSL range uses
        # i32 counters; widen its byte offset before doing address arithmetic.
        for offset in range(lane * fx.Int32(8), fx.Int32(nbytes), fx.Int32(64 * 8)):
            value = fx.Int64(load_i64_global_agent_relaxed(src + fx.Int64(offset)))
            store_i64_global_system_relaxed(dst + fx.Int64(offset), value)
        fx.rocdl.s_waitcnt(vmcnt=0)
        fence_system_release()
    else:
        gda, kw = _gda(dev_comm, deferred=aggregate)
        gda.put(
            peer,
            dst_win,
            dst_off,
            src_win,
            src_off,
            nbytes,
            coop=_WARP,
            team=cco.Team.WORLD,
            ctx=ctx,
            **kw,
        )


@flyc.jit
def _virtual_put_value(
    dev_comm: fx.Int64,
    ctx: fx.Int32,
    peer: fx.Int32,
    dst_win: fx.Int64,
    dst_off: fx.Int64,
    value: fx.Int64,
    rank: fx.Constexpr[int],
    physical_width: fx.Constexpr[int],
    aggregate: fx.Constexpr[bool],
):
    if peer // fx.Int32(physical_width) == fx.Int32(rank // physical_width):
        if fx.Int32(fx.thread_idx.x) % fx.Int32(64) == fx.Int32(0):
            dst = fx.Int64(
                cco.Window(dst_win).lsa_ptr(peer % fx.Int32(physical_width), dst_off)
            )
            store_i64_global_system(dst, value)
    else:
        gda, kw = _gda(dev_comm, deferred=aggregate)
        gda.put_value(
            peer,
            dst_win,
            dst_off,
            value,
            coop=_WARP,
            team=cco.Team.WORLD,
            ctx=ctx,
            **kw,
        )


@flyc.jit
def _virtual_flush_async(
    dev_comm: fx.Int64,
    ctx: fx.Int32,
    peer: fx.Int32,
    rank: fx.Constexpr[int],
    physical_width: fx.Constexpr[int],
):
    request = fx.Int64(-1)
    if peer // fx.Int32(physical_width) == fx.Int32(rank // physical_width):
        fx.rocdl.s_waitcnt(vmcnt=0)
        fence_system_release()
    else:
        gda, kw = _gda_flush(dev_comm)
        request = gda.flush_async(peer, coop=_WARP, team=cco.Team.WORLD, ctx=ctx, **kw)
    return request


@flyc.jit
def _virtual_wait(dev_comm: fx.Int64, request: fx.Int64):
    # LSA completion is ordered before its local request sentinel is returned.
    if request != fx.Int64(-1):
        gda, _ = _gda_flush(dev_comm)
        gda.wait(request, coop=_WARP)


class RailGda:
    """The (rail team, warp coop) GDA ops the kernels use.

    ``peer`` is a logical node index and ``ctx`` the QP context of this call.
    Physical nodes use ``Team.GDA`` on a ``GDA_CONNECTION_RAIL`` DevComm.
    Virtual nodes use CROSSNODE connectivity with WORLD peers; same-host
    virtual peers use LSA copies. Each logical peer maps to its aligned global
    rank with ``peer * peer_stride + peer_lane``.
    ``context_base`` reserves a distinct context range for concurrently
    executing stages; logical QP indices and scratch counters stay unchanged.
    """

    def __init__(
        self, context_base=0, *, peer_stride=1, peer_lane=0, rank=None, physical_width=0
    ):
        self.context_base = int(context_base)
        self.peer_stride = int(peer_stride)
        self.peer_lane = int(peer_lane)
        self.rank = None if rank is None else int(rank)
        self.physical_width = int(physical_width)
        if self.context_base < 0:
            raise ValueError("context_base must be non-negative")
        if self.peer_stride < 1 or not 0 <= self.peer_lane < self.peer_stride:
            raise ValueError("peer_lane must belong to the positive peer stride")
        if self.peer_stride != 1 and (
            self.rank is None
            or self.rank < 0
            or self.physical_width < self.peer_stride
            or self.physical_width % self.peer_stride
        ):
            raise ValueError("virtual GDA needs a rank and matching physical LSA width")

    def _context(self, ctx):
        return ctx if self.context_base == 0 else ctx + fx.Int32(self.context_base)

    def _peer(self, peer):
        # Keep the physical rail expression unchanged.
        return (
            peer
            if self.peer_stride == 1
            else peer * fx.Int32(self.peer_stride) + fx.Int32(self.peer_lane)
        )

    def put(
        self,
        dev_comm,
        ctx,
        peer,
        dst_win,
        dst_off,
        src_win,
        src_off,
        nbytes,
        *,
        aggregate=True,
    ):
        """Post a window-to-window RDMA write to ``peer``.

        ``aggregate=True`` leaves the doorbell unrung: the transfer is only
        released by a later flush on the same ctx.
        """
        if self.peer_stride != 1:
            _virtual_put(
                dev_comm,
                self._context(ctx),
                self._peer(peer),
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                self.rank,
                self.physical_width,
                bool(aggregate),
            )
            return
        gda, kw = _gda(dev_comm, deferred=bool(aggregate))
        gda.put(
            self._peer(peer),
            dst_win,
            dst_off,
            src_win,
            src_off,
            nbytes,
            coop=_WARP,
            team=_GDA,
            ctx=self._context(ctx),
            **kw,
        )

    def put_value(
        self, dev_comm, ctx, peer, dst_win, dst_off, value, *, aggregate=True
    ):
        """Post an inline 8-byte write; same doorbell rule as :meth:`put`."""
        if self.peer_stride != 1:
            _virtual_put_value(
                dev_comm,
                self._context(ctx),
                self._peer(peer),
                dst_win,
                dst_off,
                value,
                self.rank,
                self.physical_width,
                bool(aggregate),
            )
            return
        gda, kw = _gda(dev_comm, deferred=bool(aggregate))
        gda.put_value(
            self._peer(peer),
            dst_win,
            dst_off,
            value,
            coop=_WARP,
            team=_GDA,
            ctx=self._context(ctx),
            **kw,
        )

    def flush_peer(self, dev_comm, ctx, peer):
        """Ring the doorbell for ctx/peer and poll its CQ to completion."""
        if self.peer_stride != 1:
            _virtual_wait(dev_comm, self.flush_async(dev_comm, ctx, peer))
            return
        gda, kw = _gda_flush(dev_comm)
        gda.flush_peer(
            self._peer(peer), coop=_WARP, team=_GDA, ctx=self._context(ctx), **kw
        )

    def flush_async(self, dev_comm, ctx, peer):
        """Ring the doorbell and return a request handle for :meth:`wait`.

        Only worth the split when real work sits between the two; stage1 does
        interleave a credit request with the original one.
        """
        if self.peer_stride != 1:
            return _virtual_flush_async(
                dev_comm,
                self._context(ctx),
                self._peer(peer),
                self.rank,
                self.physical_width,
            )
        gda, kw = _gda_flush(dev_comm)
        return gda.flush_async(
            self._peer(peer), coop=_WARP, team=_GDA, ctx=self._context(ctx), **kw
        )

    def wait(self, dev_comm, ctx, request):
        """Block until the batch named by a :meth:`flush_async` handle lands.

        The request names its own QP; ``ctx`` is accepted for symmetry.
        """
        if self.peer_stride != 1:
            _virtual_wait(dev_comm, request)
            return
        gda, _ = _gda_flush(dev_comm)
        gda.wait(request, coop=_WARP)


rail = RailGda()
