# SPDX-License-Identifier: MIT
"""Device-side communication of MegaMoEInternode.

Primitives: the generic helpers are imported from AITER's shared communication
module; the acquire polling and last-arriver operations this operator's
protocol needs live here so it does not alter shared kernels.

Rail GDA externs (``RailGda`` / ``checked`` / ``release``): the one piece of CCO
this operator cannot get from MORI's public binding.
``mori.cco.device.flydsl`` covers everything else the kernels need
-- ``Window.lsa_ptr`` for node-local addressing, the signal/wait ops, and the
whole host side (``Communicator``/``alloc_mem``/``register_window``/
``create_dev_comm``). Two gaps remain on the cross-node return path, both in
MORI's *wrapper*, not in its C++ API:

* every exported GDA symbol takes the ``ccoTeamMode`` default ``CCO_TEAM_WORLD``
  (``cco_scale_out.hpp:212``), so ``peer`` is a world rank -- the rail team,
  whose ``peer`` is a node index, is unreachable; and
* ``ccoGdaOptFlagsAggregateRequests`` (post the WQE, do not ring the doorbell)
  is passed for SDMA but never for GDA. This is a different axis from the
  ``at`` symbol's ``ccoGdaThreadAggregate``, which coalesces a warp's lanes
  and defers nothing.

``gda_rail.cpp`` instantiates exactly the (rail, warp) combination the kernels
use. Delete it and ``RailGda`` once MORI's wrapper takes a team and optFlags.
"""

from __future__ import annotations

import tempfile
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d, rocdl as _mlir_rocdl
from flydsl import expr as fx
from flydsl.expr import arith
from aiter.ops.flydsl.kernels.communication_ops_utils import (
    atomic_add_agent,
    atomic_add_system,
    fence_system_acquire,
    fence_system_release,
    load_i64_global,
    store_i32_system,
    store_i64_global_system,
)
from flydsl.compiler.ast_rewriter import ASTRewriter
from pathlib import Path


def traced(fn):
    """Run FlyDSL's AST rewriting over a helper that kernel bodies call
    (upstream dropped this from communication_ops_utils)."""
    return ASTRewriter.transform(fn)

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
def spin_until_ge_i64_system_rx(addr_i64, expected):
    """spin_until_ge_i64_system,但轮询用 relaxed system load(不带 buffer_inv),
    看到目标值之后再做一次 acquire load。语义不变:acquire 仍然发生在观察到值之后、
    读任何被它保护的数据之前;只是把「每次轮询一次 L2 失效」变成「每次等待一次」。"""
    cur = fx.Int64(load_i64_global_system_relaxed(addr_i64))
    while cur < fx.Int64(expected):
        cur = fx.Int64(load_i64_global_system_relaxed(addr_i64))
    return fx.Int64(load_i64_global_system(addr_i64))


@traced
def spin_until_ge_i64_sleep_rx(addr_i64, expected, sleep):
    """spin_until_ge_i64_sleep 的 relaxed 轮询版,退出时一次 acquire load(见 _system_rx)。"""
    cur = fx.Int64(load_i64_global_system_relaxed(addr_i64))
    while cur < fx.Int64(expected):
        _mlir_rocdl.s_sleep(sleep)
        cur = fx.Int64(load_i64_global_system_relaxed(addr_i64))
    return fx.Int64(load_i64_global_system(addr_i64))


_U64, _I32 = "uint64", "int32"
# (devComm, ctx, peer, dstWin, dstOff, srcWin, srcOff, nbytes, aggregate)
_PUT_ARGS = [_U64, _I32, _I32, _U64, _U64, _U64, _U64, _U64, _I32]
# (devComm, ctx, peer, dstWin, dstOff, value, aggregate)
_PUT_VALUE_ARGS = [_U64, _I32, _I32, _U64, _U64, _U64, _I32]
# (devComm, ctx, peer)
_FLUSH_ARGS = [_U64, _I32, _I32]
# (devComm, ctx, packedRequest)
_WAIT_ARGS = [_U64, _I32, _U64]

_bitcode: dict[tuple[int, bool], str] = {}


def _sdma_enabled() -> bool:
    """Mirror the installed MORI host build flag in this private device TU.

    This TU only instantiates GDA and ``ccoDevComm``'s layout is stable either
    way; matching the host flag keeps a stale cache entry from masking a
    packaging mismatch.
    """
    import os

    env = os.environ.get("BUILD_CCO_SDMA")
    if env is not None:
        return env.strip().upper() in {"1", "ON", "TRUE", "YES"}
    try:
        from mori.cco.device._build_flags import BUILD_CCO_SDMA

        return bool(BUILD_CCO_SDMA)
    except (ImportError, AttributeError):
        return False


def ccqe_enabled() -> bool:
    """Whether MORI creates this host's ionic CQs in CCQE (collapsed) mode.

    The shim's CQ poll is compiled two ways (``-DIONIC_CCQE``) and MORI picks
    the CQ mode at runtime from the NIC firmware (>= 1.117.5-a-58). A bitcode or
    kernel cached under the other mode polls the wrong CQ layout: it keeps
    working for a while and then hangs on a completion it misreads. So the mode
    has to be part of every cache key that ends up containing the poll.
    """
    try:
        from mori.jit.core import is_ccqe_enabled
    except ImportError:
        return False
    return bool(is_ccqe_enabled())


def get_bitcode_path(cov: int = 6, *, ndebug: bool = False) -> str:
    """JIT ``gda_rail.cpp`` for the active arch/NIC, once per process."""
    key = (int(cov), bool(ndebug))
    if key in _bitcode:
        return _bitcode[key]

    from mori.jit.config import (
        detect_build_config,
        detect_nic_type,
        get_mori_source_root,
    )
    from mori.jit.cache import get_cache_dir
    from mori.jit.core import (
        FileBaton,
        _collect_include_dirs,
        _hipcc_device_bc,
        _strip_lifetime_intrinsics,
    )

    source = Path(__file__).with_name("gda_rail.cpp")
    mori_root = get_mori_source_root()
    if mori_root is None:
        raise RuntimeError("MORI JIT sources are required for the rail GDA shim")

    cfg = detect_build_config()
    sdma = _sdma_enabled()
    cache = get_cache_dir(
        cfg.arch,
        [
            source,
            mori_root / "include" / "mori" / "cco",
            mori_root / "include" / "mori" / "core",
        ],
        detect_nic_type(),
        cov=cov,
        ccqe=ccqe_enabled(),
    )
    cache = cache / f"megamoe_gda_rail_{'ndebug_' if ndebug else ''}sdma{int(sdma)}"
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / "libmegamoe_gda_rail.bc"
    lock = cache / ".libmegamoe_gda_rail.bc.lock"

    with FileBaton(lock, wait_for=str(output)) as baton:
        if not baton.skipped and not output.is_file():
            with tempfile.TemporaryDirectory() as td:
                raw = Path(td) / "gda_rail.bc"
                defines = [f"-DBUILD_CCO_SDMA={int(sdma)}"]
                if ndebug:
                    defines.append("-DNDEBUG")
                _hipcc_device_bc(
                    cfg, source, _collect_include_dirs(mori_root), raw,
                    cov=cov, extra_defines=defines,
                )
                _strip_lifetime_intrinsics(cfg, raw, output)

    _bitcode[key] = str(output)
    return _bitcode[key]


class _LazyExtern:
    """Link one shim symbol on first use, not at import time."""

    def __init__(self, symbol, args, ndebug, ret="void"):
        self._ret = ret
        self._symbol, self._args, self._ndebug = symbol, args, bool(ndebug)
        self._fn = None

    def __call__(self, *args):
        if self._fn is None:
            from flydsl.compiler.extern_link import link_extern
            from flydsl.expr.extern import ffi

            self._fn = link_extern(
                ffi(self._symbol, self._args, self._ret),
                bitcode_path=get_bitcode_path(ndebug=self._ndebug),
                module_init_fn=None,
            )
        return self._fn(*args)


class RailGda:
    """The (rail team, warp coop) GDA ops, bound to one bitcode build."""

    def __init__(self, *, ndebug: bool = False):
        self._put = _LazyExtern("megamoe_gda_put_rail_warp", _PUT_ARGS, ndebug)
        self._put_value = _LazyExtern(
            "megamoe_gda_put_value_rail_warp", _PUT_VALUE_ARGS, ndebug
        )
        self._flush_peer = _LazyExtern(
            "megamoe_gda_flush_peer_rail_warp", _FLUSH_ARGS, ndebug
        )
        self._flush_async = _LazyExtern(
            "megamoe_gda_flush_async_rail_warp", _FLUSH_ARGS, ndebug, "uint64"
        )
        self._wait_request = _LazyExtern(
            "megamoe_gda_wait_request_warp", _WAIT_ARGS, ndebug
        )

    def put(self, dev_comm, ctx, peer, dst_win, dst_off, src_win, src_off,
            nbytes, *, aggregate=True):
        """Post a window-to-window RDMA write to ``peer`` (a NODE index).

        ``aggregate=True`` leaves the doorbell unrung: the transfer is only
        released by a later :meth:`flush_peer` on the same ctx.
        """
        return self._put(dev_comm, ctx, peer, dst_win, dst_off, src_win,
                         src_off, nbytes, int(aggregate))

    def put_value(self, dev_comm, ctx, peer, dst_win, dst_off, value,
                  *, aggregate=True):
        """Post an inline 8-byte write; same doorbell rule as :meth:`put`."""
        return self._put_value(dev_comm, ctx, peer, dst_win, dst_off, value,
                               int(aggregate))

    def flush_peer(self, dev_comm, ctx, peer):
        """Ring the doorbell for ctx/peer and poll its CQ to completion."""
        return self._flush_peer(dev_comm, ctx, peer)

    def flush_async(self, dev_comm, ctx, peer):
        """Ring the doorbell and return a request handle for :meth:`wait`.

        Only worth the split when real work sits between the two; stage1 does
        interleave a credit request with the original one.
        """
        return self._flush_async(dev_comm, ctx, peer)

    def wait(self, dev_comm, ctx, request):
        """Block until the batch named by a :meth:`flush_async` handle lands."""
        return self._wait_request(dev_comm, ctx, request)


# Two builds, matching what the kernels used before: the rail-critical kernel2
# takes the NDEBUG one (MORI's device asserts compiled out), stage1/stage2 keep
# the checked build.
checked = RailGda(ndebug=False)
release = RailGda(ndebug=True)
