# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Minimal reference FusionResources (ATOM may wrap or replace it).

Owns one MORI CCO communicator per process and the windows requested by plans through WindowReq.
Phase-1 layout is preserved: a plan's Rx window and control window are separate windows; the control block is zeroed
(epoch, flags, credits, poison all 0) before first use and after any poison.

Deliberate deviations from the Phase-1 drivers (recorded in aiter-integration-notes.md):
  * the comm is shared by all plans; each plan INSTANCE (keyed by its build key) owns its own Rx + control windows,
    because the flag protocol counts pushers per (src,dst) (K_sd differs between PUSH8/24-push/HALVES) and the epoch is
    device-owned per kernel pair. Windows are equal-sized within a plan and every size is rounded up to a 4 MiB multiple.
  * all windows are registered before the single devcomm is created (collective order: every rank must call
    ensure() with identical requests in identical order; ensure() after finalize() raises).

Lifecycle: ensure(owner, reqs) ... finalize() ... [quiesce_and_reset()]* ... release().
quiesce_and_reset = device sync -> barrier -> zero all windows -> device sync -> barrier  (all ranks).
"""

from __future__ import annotations

import ctypes as C
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .plan import WindowReq

log = logging.getLogger("aiter.gemm_fusion")

GRANULE = 4 << 20  # 4 MiB (Phase-1 r8g finding)
PER_RANK_VMM_BYTES = 4294967296  # 4 GiB, as in the Phase-1 drivers


class ResourceError(RuntimeError):
    pass


def round_up(n: int, g: int = GRANULE) -> int:
    return ((int(n) + g - 1) // g) * g


@dataclass
class Window:
    owner: str
    name: str
    kind: str
    bytes: int  # rounded
    handle: object
    local: int  # local device pointer (int)


class CcoBackend:
    """ctypes wrapper over the JIT-built gf_comm.so (csrc/gemm_fusion/sp_qkv/host/gf_comm.cpp)."""

    def __init__(self, so_path: Optional[str] = None):
        if so_path is None:
            from aiter.ops.gemm_fusion.sp_qkv.jit import build_comm

            so_path = build_comm()
        self.lib = lib = C.CDLL(so_path)
        vp, u32, u64, i32 = C.c_void_p, C.c_uint32, C.c_uint64, C.c_int32
        for n, a in [
            ("gf_uid_v1", [u32, vp]), ("gf_comm_create_v1", [u32, u32, vp, u64, C.POINTER(vp)]),
            ("gf_win_register_v1", [vp, u64, C.POINTER(vp), C.POINTER(vp)]), ("gf_peer_ptr_v1", [vp, vp, i32, C.POINTER(vp)]),
            ("gf_devcomm_create_v1", [vp, u32, C.POINTER(vp)]), ("gf_zero_v1", [i32, vp, u64]), ("gf_device_sync_v1", [i32]),
            ("gf_devcomm_destroy_v1", [vp, vp]), ("gf_win_deregister_v1", [vp, vp]), ("gf_comm_destroy_v1", [vp]),
        ]:
            f = getattr(lib, n)
            f.restype = i32
            f.argtypes = a

    @staticmethod
    def _chk(rc, what):
        if rc != 0:
            raise ResourceError(f"{what} rc={rc}")

    def uid(self, rank):
        b = (C.c_uint8 * 128)()
        self._chk(self.lib.gf_uid_v1(rank, b), "gf_uid")
        return bytes(b)

    def comm_create(self, rank, world, uid, vmm_bytes):
        h = C.c_void_p()
        self._chk(self.lib.gf_comm_create_v1(rank, world, uid, vmm_bytes, C.byref(h)), "gf_comm_create")
        return h.value

    def win_register(self, comm, nbytes):
        w, l = C.c_void_p(), C.c_void_p()
        self._chk(self.lib.gf_win_register_v1(comm, nbytes, C.byref(w), C.byref(l)), "gf_win_register")
        return w.value, l.value

    def peer_ptr(self, comm, local, peer):
        o = C.c_void_p()
        self._chk(self.lib.gf_peer_ptr_v1(comm, local, peer, C.byref(o)), "gf_peer_ptr")
        return o.value

    def devcomm_create(self, comm, sdma_queues):
        o = C.c_void_p()
        self._chk(self.lib.gf_devcomm_create_v1(comm, sdma_queues, C.byref(o)), "gf_devcomm_create")
        return o.value

    def zero(self, device, local, nbytes):
        self._chk(self.lib.gf_zero_v1(device, local, nbytes), "gf_zero")

    def device_sync(self, device):
        self._chk(self.lib.gf_device_sync_v1(device), "gf_device_sync")

    def devcomm_destroy(self, comm, dev):
        self._chk(self.lib.gf_devcomm_destroy_v1(comm, dev), "gf_devcomm_destroy")

    def win_deregister(self, comm, win):
        self._chk(self.lib.gf_win_deregister_v1(comm, win), "gf_win_deregister")

    def comm_destroy(self, comm):
        self._chk(self.lib.gf_comm_destroy_v1(comm), "gf_comm_destroy")


class FusionResources:
    """rank/world/group come from the caller (ATOM's SP group). ``dist`` is torch.distributed (or a fake)."""

    def __init__(self, rank: int, world: int, group=None, dist=None, backend=None, device: Optional[int] = None,
                 caps: frozenset = frozenset(), vmm_bytes: int = PER_RANK_VMM_BYTES):
        if dist is None:
            import torch.distributed as dist  # type: ignore
        self.rank, self.world, self.group, self.dist = rank, world, group, dist
        self.device = rank if device is None else device
        self.caps = frozenset(caps)
        self.backend = backend or CcoBackend()
        self._vmm = vmm_bytes
        self.comm = None
        self.devcomm = None
        self.windows: Dict[str, Dict[str, Window]] = {}  # owner -> name -> Window
        self._init_hooks: Dict[str, Callable[[], None]] = {}  # owner -> control-block initializer (re-run after every zeroing)
        self.finalized = False
        self._released = False
        self._init_comm()

    # -- helpers -----------------------------------------------------------
    def _barrier(self):
        self.dist.barrier(group=self.group)

    def _init_comm(self):
        uid = [self.backend.uid(self.rank) if self.rank == 0 else None]
        self.dist.broadcast_object_list(uid, src=self._src0(), group=self.group)
        self.comm = self.backend.comm_create(self.rank, self.world, uid[0], self._vmm)
        log.info("event=resources comm rank=%d world=%d", self.rank, self.world)

    def _src0(self):
        # broadcast_object_list takes a GLOBAL rank; map group rank 0 when a group is given
        if self.group is not None and hasattr(self.dist, "get_global_rank"):
            return self.dist.get_global_rank(self.group, 0)
        return 0

    # -- window management ---------------------------------------------------
    def ensure(self, owner: str, reqs: List[WindowReq]) -> Dict[str, Window]:
        """Register this owner's windows once (idempotent per owner; collective on first call)."""
        if owner in self.windows:
            return self.windows[owner]
        if self.finalized:
            raise ResourceError(
                f"ensure({owner!r}) after finalize(): resolve every plan on every rank before finalize (collective ordering)"
            )
        wins: Dict[str, Window] = {}
        wreqs = [r for r in reqs if r.kind in ("p2p_window", "control")]
        # equal sizes within a plan: all windows of an owner get the max, rounded to 4 MiB
        size = round_up(max((r.bytes for r in wreqs), default=0)) if wreqs else 0
        for r in wreqs:
            h, loc = self.backend.win_register(self.comm, size)
            wins[r.name] = Window(owner, r.name, r.kind, size, h, loc)
        self._sdma = getattr(self, "_sdma", 0) + sum(1 for r in reqs if r.kind == "sdma_queue")
        self.windows[owner] = wins
        self._zero_windows(wins.values())  # local only; ranks barrier before first launch (finalize)
        return wins

    def set_init(self, owner: str, fn: Callable[[], None]):
        """Register + run the owner's control-block initializer (e.g. the large-M epoch=1/magic image); it is re-run after
        every zeroing in quiesce_and_reset. Local only: ranks barrier (finalize()/quiesce_and_reset()) before the first launch."""
        self._init_hooks[owner] = fn
        fn()

    def peers(self, win: Window) -> List[int]:
        return [self.backend.peer_ptr(self.comm, win.local, p) for p in range(self.world)]

    def finalize(self):
        """Create the single devcomm (after all windows are registered), then barrier. Collective."""
        if self.finalized:
            return
        self.devcomm = self.backend.devcomm_create(self.comm, max(1, getattr(self, "_sdma", 0)))
        self.finalized = True
        self.backend.device_sync(self.device)
        self._barrier()

    def _zero_windows(self, wins):
        for w in wins:
            self.backend.zero(self.device, w.local, w.bytes)

    def quiesce_and_reset(self):
        """A2-H: device sync -> barrier -> reset control/Rx windows -> device sync -> barrier. All ranks must call it."""
        self.backend.device_sync(self.device)
        self._barrier()
        for owner, wins in self.windows.items():
            self._zero_windows(wins.values())
            hook = self._init_hooks.get(owner)
            if hook is not None:
                hook()
        self.backend.device_sync(self.device)
        self._barrier()
        log.info("event=resources quiesce_and_reset rank=%d windows=%d", self.rank, sum(len(w) for w in self.windows.values()))

    def local_m(self, tokens: int) -> int:
        return tokens // self.world

    def release(self):
        if self._released:
            return
        self._released = True
        self.backend.device_sync(self.device)
        self._barrier()
        if self.devcomm is not None:
            self.backend.devcomm_destroy(self.comm, self.devcomm)
        for wins in self.windows.values():
            for w in wins.values():
                self.backend.win_deregister(self.comm, w.handle)
        self.backend.comm_destroy(self.comm)
        self.windows.clear()
