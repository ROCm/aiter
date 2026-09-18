# SPDX-License-Identifier: MIT
"""The one piece of CCO this operator cannot get from MORI's public binding.

``mori.cco.device.flydsl`` covers everything else the MegaMoE Tile kernels need
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
use. Delete both files once MORI's wrapper takes a team and optFlags.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

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
