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
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .plan import WindowReq

log = logging.getLogger("aiter.gemm_fusion")

GRANULE = 4 << 20  # 4 MiB (Phase-1 r8g finding)
PER_RANK_VMM_BYTES = 4294967296  # 4 GiB, as in the Phase-1 drivers (ccoCommCreate rounds the slot to a 4 GiB multiple)
HEAP_SLACK_BYTES = 256 << 20  # ccoDevCommCreate allocates its own resource window from the same heap
ENV_POLICY = "AITER_GEMM_FUSION_WINDOW_POLICY"  # "uniform" (default) | "per_owner"
ENV_WINDOW_MIB = "AITER_GEMM_FUSION_WINDOW_MIB"  # override the uniform window size (MiB)
ENV_ALIGN_CAP_MIB = "AITER_GEMM_FUSION_ALIGN_CAP_MIB"  # aligned policy: cap of the start alignment (default 256 MiB)
POLICIES = ("uniform", "per_owner", "aligned")


class ResourceError(RuntimeError):
    pass


def round_up(n: int, g: int = GRANULE) -> int:
    return ((int(n) + g - 1) // g) * g


def next_pow2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


def start_alignment(size: int, cap: int = 256 << 20) -> int:
    """Alignment of a window's slot offset under the 'aligned' policy: next_pow2(size), at least the 4 MiB granule, at most cap."""
    return max(GRANULE, min(next_pow2(size), cap))


def aligned_plan(sizes: List[int], cap: int = 256 << 20, start: int = 0) -> List[Tuple[str, int, int]]:
    """Pure: the registration sequence the 'aligned' policy performs for `sizes` (in order), assuming a monotonic first-fit allocator
    that places each registration at the current top. Returns [(kind, size, offset)] with kind in {"window", "spacer"}."""
    out, top = [], start
    for s in sizes:
        a = start_alignment(s, cap)
        if top % a:
            pad = a - top % a
            out.append(("spacer", pad, top))
            top += pad
        out.append(("window", s, top))
        top += s
    return out


def aligned_total_bytes(sizes: List[int], cap: int = 256 << 20) -> int:
    """Worst-case heap use of `sizes` registered in ANY order under the aligned policy (every window may need a full-alignment pad)."""
    return sum(s + max(0, start_alignment(s, cap) - GRANULE) for s in sizes)


def heap_bytes_for(window_bytes: int, max_windows: int, floor: int = PER_RANK_VMM_BYTES) -> int:
    """Per-rank VMM heap for max_windows equal windows (+devcomm slack), rounded up to a 4 GiB multiple, never below the Phase-1 4 GiB."""
    g = 4 << 30
    need = window_bytes * max_windows + HEAP_SLACK_BYTES
    return max(floor, ((need + g - 1) // g) * g)


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
                 caps: frozenset = frozenset(), vmm_bytes: Optional[int] = None, window_bytes: Optional[int] = None,
                 max_windows: Optional[int] = None, policy: Optional[str] = None):
        if dist is None:
            import torch.distributed as dist  # type: ignore
        self.rank, self.world, self.group, self.dist = rank, world, group, dist
        self.device = rank if device is None else device
        self.caps = frozenset(caps)
        self.backend = backend or CcoBackend()
        self.policy = (policy or os.environ.get(ENV_POLICY) or "uniform").lower()
        if self.policy not in POLICIES:
            raise ResourceError(f"{ENV_POLICY} must be one of {POLICIES}, got {self.policy!r}")
        self.align_cap = int(os.environ.get(ENV_ALIGN_CAP_MIB, "256")) << 20
        self._window_sizes: List[int] = []
        self.window_bytes, self.max_windows = self._budget(window_bytes, max_windows)
        if vmm_bytes is None:
            if self.policy == "uniform" and self.window_bytes:
                vmm_bytes = heap_bytes_for(self.window_bytes, self.max_windows)
            elif self.policy == "aligned" and self._window_sizes:
                g = 4 << 30
                need = aligned_total_bytes(self._window_sizes, self.align_cap) + HEAP_SLACK_BYTES
                vmm_bytes = max(PER_RANK_VMM_BYTES, (need + g - 1) // g * g)
            else:
                vmm_bytes = PER_RANK_VMM_BYTES
        self._vmm = vmm_bytes
        self._reg_log: List[Tuple[str, str, int, int]] = []  # (owner, name, bytes, cumulative) in registration order
        self._slot_base: Optional[int] = None  # local pointer of the first registration (slot offset 0 of this rank's heap)
        self._top = 0  # measured end offset of the last registration (monotonic first-fit assumption, verified per registration)
        self._spacers: List[Tuple[object, int, int]] = []  # (handle, local, bytes) of alignment pads, never used, freed at release
        self.comm = None
        self.devcomm = None
        self.windows: Dict[str, Dict[str, Window]] = {}  # owner -> name -> Window
        self._init_hooks: Dict[str, Callable[[], None]] = {}  # owner -> control-block initializer (re-run after every zeroing)
        self._drain_hooks: Dict[str, Callable[[], None]] = {}  # owner -> pre-reset SDMA drain
        self._close_hooks: Dict[str, Callable[[], None]] = {}  # owner -> free host/driver objects (L-5), run at release()
        self._poison_hooks: Dict[str, Callable[[], int]] = {}  # owner -> cheap poison word read (host, outside capture)
        self.finalized = False
        self._released = False
        self._init_comm()

    # -- helpers -----------------------------------------------------------
    def _budget(self, window_bytes, max_windows):
        """Uniform window size / count: explicit args, then env MiB override, then the loaded profile\x27s window_budget. (None, N) if unknown."""
        if window_bytes is None and os.environ.get(ENV_WINDOW_MIB):
            window_bytes = int(os.environ[ENV_WINDOW_MIB]) << 20
        if window_bytes is None or max_windows is None:
            try:
                from .profile import load_profile

                wb = dict(load_profile().meta.get("window_budget") or {})
                window_bytes = window_bytes if window_bytes is not None else wb.get("window_bytes")
                max_windows = max_windows if max_windows is not None else wb.get("max_windows")
                self._window_sizes = [int(x) for x in (wb.get("window_sizes") or [])]
            except Exception:  # no profile configured here: first-request sizing (see ensure)
                pass
        if window_bytes is not None:
            window_bytes = round_up(window_bytes)
        return window_bytes, int(max_windows or 0)

    def _barrier(self):
        self.dist.barrier(group=self.group)

    def _init_comm(self):
        uid = [self.backend.uid(self.rank) if self.rank == 0 else None]
        self.dist.broadcast_object_list(uid, src=self._src0(), group=self.group)
        self.comm = self.backend.comm_create(self.rank, self.world, uid[0], self._vmm)
        log.info("event=resources comm rank=%d world=%d heap_bytes=%d policy=%s window_bytes=%s max_windows=%s",
                 self.rank, self.world, self._vmm, self.policy, self.window_bytes, self.max_windows)

    def _src0(self):
        # broadcast_object_list takes a GLOBAL rank; map group rank 0 when a group is given
        if self.group is not None and hasattr(self.dist, "get_global_rank"):
            return self.dist.get_global_rank(self.group, 0)
        return 0

    # -- window management ---------------------------------------------------
    def _offset_of(self, loc: int) -> Optional[int]:
        if self._slot_base is None:
            return None
        return int(loc) - int(self._slot_base)

    def _register(self, owner: str, name: str, size: int, spacer: bool = False):
        """One collective window registration with measured-offset logging and a rich failure message."""
        cum = (self._reg_log[-1][3] if self._reg_log else 0) + size
        expect = self._top
        try:
            h, loc = self.backend.win_register(self.comm, size)
        except ResourceError as e:
            raise ResourceError(
                f"{e} while registering {owner}/{name}{' (alignment spacer)' if spacer else ''}: size={size} B, "
                f"window #{len(self._reg_log) + 1}, expected offset {expect} B, {cum - size} B already registered on this rank, "
                f"heap={self._vmm} B, policy={self.policy}; layout so far: {self._reg_log[-8:]}") from e
        if self._slot_base is None:
            self._slot_base = int(loc)
        off = self._offset_of(loc)
        if off != expect:
            log.warning("event=resources offset_mismatch rank=%d owner=%s name=%s expected_offset=%d measured_offset=%s "
                        "(allocator is not the monotonic first-fit the aligned policy assumes)", self.rank, owner, name, expect, off)
        self._top = (off if off is not None else expect) + size
        self._reg_log.append((owner, name + ("[spacer]" if spacer else ""), size, cum))
        log.info("event=resources window rank=%d owner=%s name=%s bytes=%d n=%d cumulative=%d offset=%s heap=%d%s",
                 self.rank, owner, name, size, len(self._reg_log), cum, off, self._vmm, " spacer=1" if spacer else "")
        return h, loc

    def _pad_to_alignment(self, owner: str, name: str, align: int):
        """Aligned policy: ccoMemAlloc has no alignment argument (HeapVAManager::Allocate(size, granule)), so pad the heap top
        with an unused spacer window until the NEXT registration starts at a multiple of `align`."""
        rem = self._top % align
        if rem:
            pad = align - rem
            h, loc = self._register(owner, name + ".pad", pad, spacer=True)
            self._spacers.append((h, loc, pad))

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
        need = round_up(max((r.bytes for r in wreqs), default=0)) if wreqs else 0
        if self.policy == "uniform" and wreqs:
            # Phase-1 layout: EVERY window of the shared comm has the same size (r8g: equal, 4 MiB multiple), so slot offsets are
            # multiples of that size exactly as in the accepted drivers. Size = profile budget / env, else the first request.
            if self.window_bytes is None:
                self.window_bytes = need
            if need > self.window_bytes:
                raise ResourceError(
                    f"ensure({owner!r}): window need {need} B exceeds the uniform window size {self.window_bytes} B; "
                    f"pre-size via the profile window_budget or {ENV_WINDOW_MIB}")
            size = self.window_bytes
        else:
            size = need  # per_owner and aligned: per-plan sizes (4 MiB multiples)
        for r in wreqs:
            if self.policy == "aligned":
                self._pad_to_alignment(owner, r.name, start_alignment(size, self.align_cap))
            h, loc = self._register(owner, r.name, size)
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

    def set_drain(self, owner: str, fn: Callable[[], None]):
        """Register the owner's pre-reset drain (large-M: lg_comm_drain_all = quietQueue on every SDMA peer queue; it must
        complete, with a device sync, on EVERY rank before any window is zeroed). Small-M plans have no SDMA queues: no hook."""
        self._drain_hooks[owner] = fn

    def set_close(self, owner: str, fn: Callable[[], None]):
        """Register the owner's teardown (free binding blocks, P, modules, driver ctx). Runs in release() after a device sync
        and BEFORE devcomm/window/comm teardown (L-5)."""
        self._close_hooks[owner] = fn

    def set_poison(self, owner: str, fn: Callable[[], int]):
        self._poison_hooks[owner] = fn

    def poison_local(self) -> int:
        """OR of every plan's device poison word on THIS rank (cheap: a 4-byte D2H + stream sync per plan; NOT capture-safe,
        call outside graph capture, e.g. once per model step). The caller must all-reduce(MAX) across ranks before deciding on
        the collective quiesce_and_reset (a poison on any rank requires recovery on all)."""
        p = 0
        for fn in self._poison_hooks.values():
            p |= int(fn())
        return p

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
        """A2-H: device sync -> per-plan drain (large-M: lg_comm_drain_all) + device sync -> barrier -> reset/init windows
        -> device sync -> barrier. All ranks must call it."""
        self.backend.device_sync(self.device)
        for hook in self._drain_hooks.values():
            hook()
        if self._drain_hooks:
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
        for fn in self._close_hooks.values():  # L-5: free Blk / P / modules / driver ctx before any comm teardown
            fn()
        self._close_hooks.clear()
        self._poison_hooks.clear()
        self._drain_hooks.clear()
        self._init_hooks.clear()
        self._barrier()
        if self.devcomm is not None:
            self.backend.devcomm_destroy(self.comm, self.devcomm)
        for wins in self.windows.values():
            for w in wins.values():
                self.backend.win_deregister(self.comm, w.handle)
        for h, _loc, _n in self._spacers:
            self.backend.win_deregister(self.comm, h)
        self._spacers.clear()
        self.backend.comm_destroy(self.comm)
        self.windows.clear()
