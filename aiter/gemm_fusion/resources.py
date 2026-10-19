# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FusionResources: one MORI CCO communicator per process and the windows requested by plans through WindowReq.

Window POLICIES (AITER_GEMM_FUSION_WINDOW_POLICY or FusionResources(policy=)):
  arena   (DEFAULT) ONE registered CCO window per comm (sized from the explicit profile's window_budget / AITER_GEMM_FUSION_ARENA_MIB), sub-allocated by aiter with 4 MiB
          alignment; every plan window is a device-memory VIEW struct (a copy of the arena's ccoWindowDevice with winBase shifted: the shipped kernels read only
          winBase/stride4G/lsaRank through ccoGetLocalPtr/ccoGetLsaPeerPtr). All large-M plans share ONE S, ONE R and ONE control window (380 MiB per GPU in total).
  uniform the Phase-1 layout: every window of the comm has the SAME size (the profile's window_budget: 39 x 164 MiB = 6.25 GiB per GPU). Kept selectable as the
          fallback when the arena is refused (no arena size, MORI ABI pin mismatch, failed view self-test) and as a hardware/MORI fallback.

Registered windows are zeroed ONCE in finalize() (MORI's own method, chunked hipMemcpy H2D: no hipMemset on ccoMemAlloc windows) after every registration and the devcomm;
a launch before that is refused (pending_zero). All windows are registered before the single devcomm is created: every rank must call ensure() with identical requests in
identical order, and ensure() after finalize() raises.

Arena CONTRACT (reviewer C-S3): all large plans share ONE control block and ONE S/R pair, so EVERY rank must launch the same SEQUENCE of large plans (not merely the same
number of launches per plan). A rank that runs a different large plan (or the same plans in a different order) is NOT detected by the device: the epoch/credit/arrival values
still match and layouts would be mixed silently for that step. Divergence is excluded by the host-side rank consensus and detected after the fact by the host-side per-step
digest (ATOM check_step: site|tokens|vid|n_fused), not by the device. recv_view/reuse_view of any large plan alias the shared R and are valid only until the next large launch
on this rank; nothing returned from the op or stored on a long-lived object may point into a window.

Lifecycle: ensure(owner, reqs) ... finalize() ... [quiesce_and_reset()]* ... release().
quiesce_and_reset = device sync -> poison log -> drain -> barrier -> zero all windows -> init hooks -> device sync -> barrier  (all ranks).
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
DEFAULT_POLICY = "arena"  # A-3 (arena review): AITER_GEMM_FUSION_WINDOW_POLICY=uniform restores the Phase-1 equal-size layout (6.4 GiB)
POLICIES = ("arena", "uniform")
PER_RANK_VMM_BYTES = 4294967296  # 4 GiB, as in the Phase-1 drivers (ccoCommCreate rounds the slot to a 4 GiB multiple)
HEAP_SLACK_BYTES = 256 << 20  # ccoDevCommCreate allocates its own resource window from the same heap
ARENA_ALIGN = 4 << 20  # sub-window offsets/sizes inside the arena are multiples of 4 MiB
ENV_POLICY = "AITER_GEMM_FUSION_WINDOW_POLICY"  # "arena" (default) | "uniform"
ENV_ARENA_MIB = "AITER_GEMM_FUSION_ARENA_MIB"  # arena size override (MiB)
ENV_WAIT_BUDGET_US = "AITER_GEMM_FUSION_WAIT_BUDGET_US"  # TUNING ONLY: overrides the profile's wait_budget_us (large-M per-wait poll budget, microseconds)
DEFAULT_WAIT_BUDGET_US = 5_000_000  # 5 s when neither the env nor the explicit profile carries one (the r11 kernel default of 50 ms is too tight for a serving engine)
# r11 poison reasons (lg_comm_abi_v2.h LGC_POISON_*): which wait failed
LG_POISON_NAMES = {1: "W1_TIMEOUT(credit wait: a peer never began this epoch)", 2: "W2_TIMEOUT(arrival wait: a peer's puts did not complete)",
                   3: "EPOCH_MISMATCH", 4: "DUP_CHUNK", 5: "CONFIG(host/device build or plan mismatch)", 6: "POSTER_WAIT(local GEMM group wait timed out)"}
SHARED_KEY = "<shared>"  # hook key of the shared window set


class ResourceError(RuntimeError):
    pass


class _ArenaRefused(Exception):
    """Internal: the arena failed its read-back self-test (on some rank): every rank falls back to the uniform policy."""


def round_up(n: int, g: int = GRANULE) -> int:
    return ((int(n) + g - 1) // g) * g


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
    shared: bool = False  # one window used by several owners (the arena's shared large S/R/C set)
    arena_offset: Optional[int] = None  # arena policies: this window is a VIEW at this byte offset inside the single registered arena window (handle = the view struct)


class CcoBackend:
    """ctypes wrapper over the JIT-built gf_comm.so (csrc/gemm_fusion/sp_qkv/host/gf_comm.cpp)."""

    def __init__(self, so_path: Optional[str] = None):
        if so_path is None:
            from aiter.ops.gemm_fusion.sp_qkv.jit import build_comm

            so_path = build_comm()
        self.lib = lib = C.CDLL(so_path)
        vp, u32, u64, i32 = C.c_void_p, C.c_uint32, C.c_uint64, C.c_int32
        for n, a in [
            ("gf_uid_v1", [u32, vp]), ("gf_comm_create_v1", [u32, u32, i32, vp, u64, C.POINTER(vp)]),
            ("gf_win_register_v1", [vp, u64, C.POINTER(vp), C.POINTER(vp)]),
            ("gf_devcomm_create_v1", [vp, u32, C.POINTER(vp)]), ("gf_devcomm_info_v1", [vp, vp, C.POINTER(u64)]), ("gf_zero_v1", [i32, vp, u64]), ("gf_mem_info_v1", [i32, C.POINTER(u64)]), ("gf_comm_va_info_v1", [vp, C.POINTER(u64)]), ("gf_win_free_v1", [vp, vp]), ("gf_abi_info_v1", [C.POINTER(u64)]),
                     ("gf_win_view_v1", [vp, vp, u64, C.POINTER(vp)]), ("gf_win_view_free_v1", [vp, vp]), ("gf_win_info_v1", [vp, vp, C.POINTER(u64)]), ("gf_device_sync_v1", [i32]),
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

    def comm_create(self, rank, world, uid, vmm_bytes, device=None):
        """`device` = the HIP device this process drives (torch.cuda.current_device()); NOT derived from the SP rank."""
        if device is None:
            raise ResourceError("comm_create needs an explicit HIP device index (the SP rank is not a device index)")
        h = C.c_void_p()
        self._chk(self.lib.gf_comm_create_v1(rank, world, int(device), uid, vmm_bytes, C.byref(h)), "gf_comm_create")
        return h.value

    def win_register(self, comm, nbytes):
        w, l = C.c_void_p(), C.c_void_p()
        self._chk(self.lib.gf_win_register_v1(comm, nbytes, C.byref(w), C.byref(l)), "gf_win_register")
        return w.value, l.value

    def devcomm_create(self, comm, sdma_queues):
        o = C.c_void_p()
        self._chk(self.lib.gf_devcomm_create_v1(comm, sdma_queues, C.byref(o)), "gf_devcomm_create")
        return o.value

    def devcomm_info(self, comm, dev):
        b = (C.c_uint64 * 7)()
        self._chk(self.lib.gf_devcomm_info_v1(comm, dev, b), "gf_devcomm_info")
        s = lambda v: v - (1 << 64) if v >= (1 << 63) else v  # noqa: E731
        k = ("lsaSize", "lsaRank", "sdmaNumQueue", "worldSize", "commRank", "myNodeStart", "sizeofDevComm")
        return {n: s(int(v)) for n, v in zip(k, list(b))}

    def zero(self, device, local, nbytes):
        self._chk(self.lib.gf_zero_v1(device, local, nbytes), "gf_zero")

    def abi_check(self):
        """(ok, reason): MORI library + header hashes are the pinned ones AND this binary's compiled ccoWindowDevice layout is the pinned one."""
        from . import capabilities as caps

        ok, why = caps.check_mori_abi_pin()
        if not ok:
            return False, why
        b = (C.c_uint64 * 5)()
        if self.lib.gf_abi_info_v1(b) != 0 or tuple(int(x) for x in b) != caps.PINNED_WINDOW_DEVICE_ABI:
            return False, f"compiled ccoWindowDevice layout {tuple(int(x) for x in b)} != pinned {caps.PINNED_WINDOW_DEVICE_ABI}"
        return True, "ok"

    def win_free(self, comm, local):
        self._chk(self.lib.gf_win_free_v1(comm, local), "gf_win_free")

    def win_view(self, comm, arena_win, offset):
        out = C.c_void_p()
        self._chk(self.lib.gf_win_view_v1(comm, arena_win, offset, C.byref(out)), "gf_win_view")
        return out.value

    def win_view_free(self, comm, view):
        self._chk(self.lib.gf_win_view_free_v1(comm, view), "gf_win_view_free")

    def win_info(self, comm, win):
        b = (C.c_uint64 * 3)()
        self._chk(self.lib.gf_win_info_v1(comm, win, b), "gf_win_info")
        return int(b[0]), int(b[1]), int(b[2])

    def mem_info(self, device):
        b = (C.c_uint64 * 2)()
        self._chk(self.lib.gf_mem_info_v1(device, b), "gf_mem_info")
        return int(b[0]), int(b[1])

    def device_sync(self, device):
        self._chk(self.lib.gf_device_sync_v1(device), "gf_device_sync")

    def devcomm_destroy(self, comm, dev):
        self._chk(self.lib.gf_devcomm_destroy_v1(comm, dev), "gf_devcomm_destroy")

    def win_deregister(self, comm, win):
        self._chk(self.lib.gf_win_deregister_v1(comm, win), "gf_win_deregister")

    def comm_destroy(self, comm):
        self._chk(self.lib.gf_comm_destroy_v1(comm), "gf_comm_destroy")


def resolve_wait_budget_us(profile=None) -> Tuple[int, str]:
    """Per-wait poll budget (microseconds) of the large-M kernels and where it came from: ('env'|'profile'|'default').
    Precedence: AITER_GEMM_FUSION_WAIT_BUDGET_US (tuning only) > the EXPLICIT profile's top-level wait_budget_us > DEFAULT_WAIT_BUDGET_US (5 s).
    The value is a per-launch KERNEL ARGUMENT that is BAKED into a captured graph's argument block at capture time: it must be resolved before capture;
    changing the env/profile afterwards does not affect graphs that were already captured.

    WHY 5 s (A-5): a clean first large call poisoned on all 4 ranks when the ranks entered the kernel ~213 ms apart (cold host skew, Evidence's skew
    characterization 05055ac9: 0.2/20/40 ms skew passed, 60/100 ms and the natural 213 ms failed the previous 50 ms budget), and a serving engine can see
    skew of hundreds of ms (GC pauses, cold module/first-SDMA costs, scheduler jitter). 5 s is ~20x the observed natural skew; it is a judgement, not derived.
    Cost: only failure-DETECTION latency: a dead peer holds the stream (and the poster block's CU) for up to ~2-3 budgets (per sequential wait) before the poison.
    The torch process-group / NCCL timeout MUST stay longer than that detection latency PLUS the recovery (poison all-reduce + quiesce_and_reset), i.e. PG timeout >
    ~3 x budget + recover; the default 10 min watchdog satisfies this. Record any override in the run bindings (FusionResources.run_bindings() reports
    wait_budget_us and wait_budget_source = env|profile|default)."""
    env = os.environ.get(ENV_WAIT_BUDGET_US)
    if env not in (None, ""):
        try:
            v = int(env)
        except ValueError:
            raise ResourceError(f"{ENV_WAIT_BUDGET_US} must be a positive integer (microseconds), got {env!r}") from None
        if v <= 0:
            raise ResourceError(f"{ENV_WAIT_BUDGET_US} must be > 0 microseconds, got {v}")
        return v, "env"
    if profile is not None:
        v = getattr(profile, "meta", {}).get("wait_budget_us")
        if v is not None:
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ResourceError(f"profile wait_budget_us must be a positive integer (microseconds), got {v!r}")
            return int(v), "profile"
    return DEFAULT_WAIT_BUDGET_US, "default"


class FusionResources:
    """rank/world/group come from the caller (ATOM's SP group). ``dist`` is torch.distributed (or a fake)."""

    def __init__(self, rank: int, world: int, group=None, dist=None, backend=None, device: Optional[int] = None,
                 caps: frozenset = frozenset(), policy: Optional[str] = None, profile=None):
        if dist is None:
            import torch.distributed as dist  # type: ignore
        self.rank, self.world, self.group, self.dist = rank, world, group, dist
        self.device = self._default_device(rank) if device is None else int(device)
        self.caps = frozenset(caps)
        self.backend = backend or CcoBackend()
        req = (policy or os.environ.get(ENV_POLICY) or "").lower()
        self.policy_source = "explicit" if req else "default"
        self.policy = req or DEFAULT_POLICY  # A-3: arena is the default; uniform stays selectable (explicit arg or the env var)
        self.policy_fallback_reason: Optional[str] = None
        if self.policy not in POLICIES:
            raise ResourceError(f"{ENV_POLICY} must be one of {POLICIES}, got {self.policy!r}")
        self._budget_doc: dict = {}
        self.profile = self._resolve_profile(profile)  # explicit only: a Profile object or a name/path; NEVER the AITER_GEMM_FUSION_PROFILE env
        self.wait_budget_us, self.wait_budget_source = resolve_wait_budget_us(self.profile)
        self.window_bytes, self.max_windows = self._budget()
        self._select_policy()
        self._vmm = self._default_vmm()
        self._reg_log: List[Tuple[str, str, int, int]] = []  # (owner, name, bytes, cumulative) in registration order
        self._regs: List[Window] = []  # every registered window exactly ONCE (the arena is the only one under the arena policy)
        self._shared: Dict[str, Window] = {}  # share tag -> the one shared view (arena policy)
        self._reg_seq = 0  # number of ensure() calls (collective agreement check)
        self._arena: Optional[Window] = None  # the ONE registered window of the comm (arena policy)
        self._arena_bytes = 0
        self._arena_top = 0  # bump pointer inside the arena (4 MiB aligned)
        self._views: List[Window] = []  # every view struct handed out (freed before the arena is deregistered)
        self._unzeroed: List[Window] = []  # registered windows not yet zeroed (zeroed in finalize(), after ALL registrations + the devcomm)
        self._mem_before_comm = None  # (free, total) before ccoCommCreate, for the HBM delta log (C-S6)
        self.comm = None
        self.devcomm = None
        self.windows: Dict[str, Dict[str, Window]] = {}  # owner -> name -> Window
        self._init_hooks: Dict[str, Callable[[], None]] = {}  # owner -> control-block initializer (re-run after every zeroing)
        self._drain_hooks: Dict[str, Callable[[], None]] = {}  # owner -> pre-reset SDMA drain
        self._close_hooks: Dict[str, Callable[[], None]] = {}  # owner -> free host/driver objects (L-5), run at release()
        self._poison_info: Dict[str, tuple] = {}  # owner -> (report reader, poison decoder), logging only
        self._poison_hooks: Dict[str, Callable[[], int]] = {}  # owner -> cheap poison word read (host, outside capture)
        self.sdma_queues = 0  # set by finalize(): sdmaQueueCount given to ccoDevCommCreate
        self.devcomm_info = None  # set by finalize(): {lsaSize, lsaRank, sdmaNumQueue, worldSize, ...} read back from the created devcomm
        self.finalized = False
        self._released = False
        self._init_comm()

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _default_device(rank: int) -> int:
        """The HIP device of THIS process: torch's current device when torch is available (it owns the thread's device), else the rank.
        The SP-group rank equals the local device index only in the simplest deployments (TP x SP, several groups and HIP_VISIBLE_DEVICES differ)."""
        try:
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.current_device())
        except Exception:  # noqa: BLE001
            pass
        return rank

    @staticmethod
    def _resolve_profile(profile):
        if profile is None:
            return None
        from .profile import resolve_profile

        return resolve_profile(profile)

    def _budget(self):
        """Uniform window size / count from the EXPLICIT ``profile=`` window_budget (no implicit profile lookup: the env var that selects a profile is not read
        here). (None, 0) when unknown: the uniform policy then sizes its window from the FIRST request (a warning is logged; see ensure)."""
        window_bytes = max_windows = None
        if self.profile is not None:
            from .profile import window_budget_of

            wb = window_budget_of(self.profile)
            window_bytes, max_windows = wb.get("window_bytes"), wb.get("max_windows")
            self._budget_doc = wb
        if window_bytes is None:
            log.warning("event=resources no_budget rank=%d profile=%s: no window budget available (%s); the uniform window size will be taken "
                        "from the FIRST ensure() request and a later larger plan will be rejected. Pass profile=<name|path|Profile> whose "
                        "window_budget is set.", self.rank, getattr(self.profile, "name", None),
                        "profile has no window_budget" if self.profile is not None else "FusionResources built without profile=")
        if window_bytes is not None:
            window_bytes = round_up(window_bytes)
        return window_bytes, int(max_windows or 0)

    def _barrier(self):
        self.dist.barrier(group=self.group)

    def _init_comm(self):
        uid = [self.backend.uid(self.rank) if self.rank == 0 else None]
        self.dist.broadcast_object_list(uid, src=self._src0(), group=self.group)
        self._mem_before_comm = self._mem_info()
        self.comm = self.backend.comm_create(self.rank, self.world, uid[0], self._vmm, self.device)
        log.info("event=resources comm rank=%d world=%d heap_bytes=%d policy=%s window_bytes=%s max_windows=%s zero=h2d_deferred wait_budget_us=%d (%s)",
                 self.rank, self.world, self._vmm, self.policy, self.window_bytes, self.max_windows, self.wait_budget_us, self.wait_budget_source)
        if self._budget_doc:
            tbl = {}
            if self.window_bytes and self.max_windows:
                tbl["uniform"] = self.window_bytes * self.max_windows
            if self._arena_size_from_budget():
                tbl["arena"] = self._arena_size_from_budget()
            log.info("event=resources policy_memory rank=%d selected=%s %s", self.rank, self.policy,
                     " ".join(f"{k}={v / (1 << 20):.0f}MiB" for k, v in tbl.items()))

    def _src0(self):
        # broadcast_object_list takes a GLOBAL rank; map group rank 0 when a group is given
        if self.group is not None and hasattr(self.dist, "get_global_rank"):
            return self.dist.get_global_rank(self.group, 0)
        return 0

    def _register(self, owner: str, name: str, size: int):
        """One collective window registration with a rich failure message."""
        cum = (self._reg_log[-1][3] if self._reg_log else 0) + size
        try:
            h, loc = self.backend.win_register(self.comm, size)
        except ResourceError as e:
            raise ResourceError(
                f"{e} while registering {owner}/{name}: size={size} B, window #{len(self._reg_log) + 1}, {cum - size} B already registered on this rank, "
                f"heap={self._vmm} B, policy={self.policy}; layout so far: {self._reg_log[-8:]}") from e
        self._reg_log.append((owner, name, size, cum))
        log.info("event=resources window rank=%d owner=%s name=%s bytes=%d n=%d cumulative=%d heap=%d local=%#x", self.rank, owner, name, size, len(self._reg_log), cum,
                 self._vmm, int(loc))
        return h, loc

    def _select_policy(self):
        """A-3: the DEFAULT policy is arena. The arena is refused (-> uniform, reason logged, kept in policy_fallback_reason) when the DEFAULT was not overridden and no
        arena size can be derived (no explicit profile window_budget / AITER_GEMM_FUSION_ARENA_MIB), or when the MORI ABI pin does not hold (A-1: library + header hashes,
        compiled ccoWindowDevice layout). An EXPLICITLY requested arena keeps raising at ensure() for a missing budget (a configuration error) but is also refused on an
        ABI-pin mismatch, because that is a safety property, not a configuration choice."""
        if self.policy != "arena":
            return
        reason = None
        if not self._arena_size_from_budget() and self.policy_source == "default":
            reason = "no arena size (FusionResources built without a profile window_budget and no AITER_GEMM_FUSION_ARENA_MIB)"
        else:
            chk = getattr(self.backend, "abi_check", None)
            if chk is not None:
                try:
                    ok, why = chk()
                except Exception as e:  # noqa: BLE001
                    ok, why = False, f"abi_check error {e!r}"
                if not ok:
                    reason = f"MORI ABI pin mismatch: {why}"
        if reason:
            self._fall_back_to_uniform(reason)

    def _fall_back_to_uniform(self, reason: str):
        log.warning("event=resources arena_refused rank=%d policy=%s -> uniform reason=%s", self.rank, self.policy, reason)
        self.policy, self.policy_fallback_reason = "uniform", reason

    def _default_vmm(self) -> int:
        """Per-rank VMM heap for the CURRENT policy (recomputed on an arena -> uniform fallback), a 4 GiB multiple, never below the Phase-1 4 GiB."""
        g = 4 << 30
        if self.policy == "arena":
            need = self._arena_size_from_budget() + HEAP_SLACK_BYTES
            return max(PER_RANK_VMM_BYTES, (need + g - 1) // g * g)
        if self.policy == "uniform" and self.window_bytes:
            return heap_bytes_for(self.window_bytes, self.max_windows)
        return PER_RANK_VMM_BYTES

    def _arena_size_from_budget(self) -> int:
        """Arena bytes: AITER_GEMM_FUSION_ARENA_MIB, else the explicit profile budget (the shared S/R/C set + every small plan's windows)."""
        env = os.environ.get(ENV_ARENA_MIB)
        if env:
            return round_up(int(env) << 20)
        b = self._budget_doc or {}
        sh, small = b.get("shared") or {}, b.get("small_window_sizes") or []
        if sh and small:
            return sum(round_up(int(v)) for v in sh.values()) + sum(round_up(int(s)) for s in small)
        return 0

    def _ensure_arena(self) -> Window:
        if self._arena is not None:
            return self._arena
        size = self._arena_size_from_budget()
        if not size:
            raise ResourceError(f"policy {self.policy!r} needs the arena size: construct FusionResources(profile=<profile with window_budget>) or set {ENV_ARENA_MIB}")
        if self.world > 1:
            self._agree_registration("<arena>", [WindowReq("arena", size, "p2p_window")])
        h, loc = self._register("<arena>", "arena", size)  # THE registration: a single window, first, never followed by another size
        w = Window("<arena>", "arena", "p2p_window", size, h, loc, shared=True)
        self._arena, self._arena_bytes, self._arena_top = w, size, 0
        self._regs.append(w)
        ok, why = self._view_selftest(w)
        got = [None] * self.world
        self.dist.all_gather_object(got, (ok, why), group=self.group)  # every rank must agree to keep the arena: peers address it at identical offsets
        if not all(g and g[0] for g in got):
            raise _ArenaRefused("; ".join(f"rank{i}: {g[1]}" for i, g in enumerate(got) if not (g and g[0])) or "a rank failed the view self-test")
        self._unzeroed.append(w)  # zeroed once in finalize()
        log.info("event=resources arena rank=%d bytes=%d (%.1f MiB) local=%#x", self.rank, size, size / (1 << 20), int(loc))
        return w

    def _view_selftest(self, arena: Window):
        """A-1: PROVE the view property on this MORI build before anything uses it: create a view at a non-zero 4 MiB-aligned offset, read it back and require
        winBase == arena.winBase + offset with equal stride4G / lsaRank. Backends that cannot read windows back (win_info) skip the check (fakes)."""
        info = getattr(self.backend, "win_info", None)
        if info is None:
            return True, "win_info unavailable: self-test skipped"
        off = ARENA_ALIGN if self._arena_bytes >= 2 * ARENA_ALIGN else 0
        try:
            v = self.backend.win_view(self.comm, arena.handle, off)
            try:
                good, why = self._check_view(arena, v, off)
            finally:
                self.backend.win_view_free(self.comm, v)
            return good, why
        except Exception as e:  # noqa: BLE001
            return False, f"view self-test error {e!r}"

    def _check_view(self, arena_win: Window, view_handle, off: int):
        info = getattr(self.backend, "win_info", None)
        if info is None:
            return True, "skipped"
        ab, astride, alsa = info(self.comm, arena_win.handle)
        vb, vstride, vlsa = info(self.comm, view_handle)
        if vb != ab + off or (vstride, vlsa) != (astride, alsa):
            return False, (f"view read-back mismatch at offset {off}: winBase {vb:#x} (expected {ab + off:#x}), stride4G {vstride} (arena {astride}), "
                           f"lsaRank {vlsa} (arena {alsa})")
        return True, "ok"

    def _arena_fallback(self, reason: str):
        """The arena was refused after its single registration: tear the comm down and rebuild it for the uniform policy (registering a window of a
        different size on the same comm is exactly what must not be done), identically on every rank (the refusal was agreed collectively)."""
        log.warning("event=resources arena_refused rank=%d policy=%s -> uniform reason=%s", self.rank, self.policy, reason)
        for v in self._views:
            self.backend.win_view_free(self.comm, v.handle)
        self._views.clear()
        for w in self._regs:
            self.backend.win_deregister(self.comm, w.handle)
            self.backend.win_free(self.comm, w.local)
        self._regs.clear()
        self._shared.clear()
        self._unzeroed.clear()
        self.backend.comm_destroy(self.comm)
        self.policy, self.policy_fallback_reason = "uniform", reason
        self._arena, self._arena_bytes, self._arena_top = None, 0, 0
        self._reg_log.clear()
        self._vmm = self._default_vmm()
        self._init_comm()

    def _arena_view(self, owner: str, name: str, kind: str, nbytes: int, share: Optional[str] = None) -> Window:
        """Bump-allocate `nbytes` (4 MiB multiple) in the arena and return a Window whose handle is a VIEW struct and whose local is arena.local + offset."""
        a = self._arena
        size = -(-int(nbytes) // ARENA_ALIGN) * ARENA_ALIGN
        off = self._arena_top
        if off + size > self._arena_bytes:
            raise ResourceError(f"arena overflow at {owner}/{name}: need {size} B at offset {off}, arena is {self._arena_bytes} B (policy {self.policy}; "
                                f"size it via the profile window_budget or {ENV_ARENA_MIB})")
        self._arena_top = off + size
        view = self.backend.win_view(self.comm, a.handle, off)
        good, why = self._check_view(a, view, off)
        if not good:
            self.backend.win_view_free(self.comm, view)
            raise ResourceError(f"arena view for {owner}/{name}: {why}; fusion must be aborted")
        w = Window(SHARED_KEY if share else owner, share or name, kind, size, view, int(a.local) + off, shared=bool(share), arena_offset=off)
        self._views.append(w)
        return w

    def _arena_ensure(self, owner: str, wreqs, wins: Dict[str, Window]):
        self._ensure_arena()
        # deterministic offsets: the shared set is laid out at its FIRST use (sized from the profile's shared budget), then owners bump in ensure() order; peers address
        # this rank's arena at the SAME offsets, so the layout is verified collectively below.
        for r in wreqs:
            if r.share:
                w = self._shared.get(r.share)
                need = int(r.min_bytes if r.min_bytes is not None else r.bytes)
                if w is None:
                    # the shared set is sized from the profile's shared budget (the LARGEST large plan), never from whichever plan is ensured first
                    budget = int(((self._budget_doc or {}).get("shared") or {}).get(r.share) or 0)
                    if budget and need > budget:
                        raise ResourceError(f"ensure({owner!r}): shared arena window {r.share!r} need {need} B exceeds the budgeted {budget} B")
                    w = self._arena_view(owner, r.name, r.kind, budget or need, share=r.share)
                    self._shared[r.share] = w
                elif need > w.bytes:
                    raise ResourceError(f"ensure({owner!r}): shared arena window {r.share!r} need exceeds its size {w.bytes} B")
                wins[r.name] = w
            else:
                wins[r.name] = self._arena_view(owner, r.name, r.kind, r.bytes)
        offs = tuple((n, w.arena_offset, w.bytes) for n, w in wins.items())
        got = [None] * self.world
        self.dist.all_gather_object(got, (self._reg_seq, owner, offs), group=self.group)
        if any(g != got[0] for g in got):
            raise ResourceError(f"arena layout disagreement across ranks at ensure({owner!r}): {got}; peers address this arena at identical offsets, fusion must be aborted")

    def ensure(self, owner: str, reqs: List[WindowReq]) -> Dict[str, Window]:
        """Create this owner's windows once (idempotent per owner; collective on the first call, identical requests in identical order on every rank).
        After finalize() a NEW owner is refused."""
        if owner in self.windows:
            return self.windows[owner]
        if self.finalized:
            raise ResourceError(f"ensure({owner!r}) after finalize(): resolve every plan on every rank before finalize (collective ordering)")
        wins: Dict[str, Window] = {}
        wreqs = [r for r in reqs if r.kind in ("p2p_window", "control")]
        self._reg_seq += 1
        if self.policy == "arena":
            if wreqs:
                try:
                    self._arena_ensure(owner, wreqs, wins)
                except _ArenaRefused as e:
                    self._arena_fallback(str(e))
                    return self.ensure(owner, reqs)  # re-dispatch on the uniform policy
        elif wreqs:
            # Phase-1 layout: EVERY window of the comm has the same size (r8g: equal, 4 MiB multiple), so slot offsets are multiples of that size exactly as in the
            # accepted drivers. Size = the profile budget, else the first request.
            need = round_up(max(r.bytes for r in wreqs))
            if self.window_bytes is None:
                self.window_bytes = need
            if need > self.window_bytes:
                raise ResourceError(
                    f"ensure({owner!r}): window need {need} B exceeds the uniform window size {self.window_bytes} B; "
                    f"pre-size by constructing FusionResources(profile=<the active profile>) (its window_budget)")
            for r in wreqs:
                h, loc = self._register(owner, r.name, self.window_bytes)
                w = Window(owner, r.name, r.kind, self.window_bytes, h, loc)
                wins[r.name] = w
                self._regs.append(w)
                self._unzeroed.append(w)  # zeroed in finalize(), after every registration and the devcomm
        # sdma_queue request: bytes = queues PER PEER wanted (0 = default 1). The shared devcomm gets the MAX over plans, never the
        # number of plans (9 large plans must not ask CCO for 9 queues: the r11 kernels require exactly 1, as in Phase-1).
        self._sdma = max([getattr(self, "_sdma", 0)] + [max(1, int(r.bytes)) for r in reqs if r.kind == "sdma_queue"])
        self.windows[owner] = wins
        return wins

    def _agree_registration(self, owner: str, wreqs: List[WindowReq]):
        mine = (self._reg_seq, owner, tuple((r.name, r.kind, int(r.bytes), r.share) for r in wreqs))
        got = [None] * self.world
        self.dist.all_gather_object(got, mine, group=self.group)
        if any(g != mine for g in got):
            raise ResourceError(f"registration disagreement across ranks at ensure({owner!r}) (seq {self._reg_seq}): {got}; fusion must be aborted on every rank")

    # -- hooks shared by several owners (shared window set): registered ONCE, by the first owner that attaches --------
    def has_hook(self, kind: str, key: str) -> bool:
        return key in {"init": self._init_hooks, "drain": self._drain_hooks, "poison": self._poison_hooks, "close": self._close_hooks}[kind]

    def set_shared_hooks(self, key: str = SHARED_KEY, *, init=None, drain=None, poison=None, poison_report=None, poison_decode=None) -> bool:
        """Register the control-block init / SDMA drain / poison read of a SHARED window set exactly once. Returns True for the first caller.
        Later owners attaching to the same live set must NOT re-run init (that would reset the epoch on this rank only)."""
        if self.has_hook("init", key) or self.has_hook("drain", key) or self.has_hook("poison", key):
            return False
        if drain is not None:
            self.set_drain(key, drain)
        if poison is not None:
            self.set_poison(key, poison, poison_report, poison_decode)
        if init is not None:
            self.set_init(key, init)  # runs once now (epoch=1 + magic), then after every zeroing
        return True

    def budget(self) -> dict:
        """Per-rank accounting of what THIS process registered: windows, shared bytes, VA heap, peer-mapped VA."""
        total = sum(w.bytes for w in self._regs)
        shared = sum(w.bytes for w in self._regs if w.shared)
        return {"policy": self.policy, "windows": len(self._regs), "registered_bytes": total, "shared_bytes": shared,
                "heap_bytes": self._vmm, "peer_mapped_va_bytes": (self.world - 1) * total, "owners": len(self.windows),
                "by_owner": {o: sum(w.bytes for w in ws.values() if not w.shared) for o, ws in self.windows.items()},
                "devcomm_sdma_queues": self.sdma_queues}

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

    def set_poison(self, owner: str, fn: Callable[[], int], report: Optional[Callable[[], dict]] = None, decode: Optional[Callable[[int], str]] = None):
        """`fn` returns the device poison word. Optional `report` (host read of the owner's report words, only called when poisoned) and `decode`
        (poison word -> which wait failed) are used ONLY to log the reason before quiesce_and_reset clears it."""
        self._poison_hooks[owner] = fn
        self._poison_info[owner] = (report, decode)

    def run_bindings(self) -> dict:
        """What ATOM/Evidence should record in the run bindings: the effective policy (and why it is not the requested/default one), the wait budget and where it
        came from (A-5: record any override), the heap and the arena size."""
        return {"window_policy": self.policy, "window_policy_source": self.policy_source, "window_policy_fallback_reason": self.policy_fallback_reason,
                "wait_budget_us": self.wait_budget_us, "wait_budget_source": self.wait_budget_source, "zero_mode": "h2d_deferred",
                "heap_bytes": self._vmm, "arena_bytes": self._arena_bytes or None}

    def log_poison(self, where: str) -> int:
        """Log every poisoned owner's reason (and the report words when available) BEFORE the reset zeroes them; returns the OR of the poison words.
        Reads the 4-byte poison word per owner (and the report only if poisoned); call outside capture. Logged at WARNING: event=resources poison."""
        total = 0
        for owner, fn in list(self._poison_hooks.items()):
            try:
                p = int(fn())
            except Exception as e:  # noqa: BLE001  (diagnostic only: never mask the recovery that follows)
                log.warning("event=resources poison rank=%d where=%s owner=%s read_failed=%s", self.rank, where, owner, e)
                continue
            total |= p
            if not p:
                continue
            report, decode = self._poison_info.get(owner, (None, None))
            rep = ""
            if report is not None:
                try:
                    rep = " report=" + ",".join(f"{k}={v}" for k, v in report().items())
                except Exception as e:  # noqa: BLE001
                    rep = f" report_failed={e}"
            what = decode(p) if decode is not None else f"word={p:#x}"
            log.warning("event=resources poison rank=%d where=%s owner=%s reason=%s%s wait_budget_us=%d (%s)", self.rank, where, owner, what, rep,
                        self.wait_budget_us, self.wait_budget_source)
        return total

    def poison_local(self) -> int:
        """OR of every plan's device poison word on THIS rank (cheap: a 4-byte D2H + stream sync per plan; NOT capture-safe,
        call outside graph capture, e.g. once per model step). The caller must all-reduce(MAX) across ranks before deciding on
        the collective quiesce_and_reset (a poison on any rank requires recovery on all)."""
        p = 0
        for fn in self._poison_hooks.values():
            p |= int(fn())
        return p

    def finalize(self):
        """Create the single devcomm (after all windows are registered), then barrier. Collective."""
        if self.finalized:
            return
        self.sdma_queues = max(1, getattr(self, "_sdma", 0))
        self.devcomm = self.backend.devcomm_create(self.comm, self.sdma_queues)
        self.devcomm_info = None
        info_fn = getattr(self.backend, "devcomm_info", None)
        if info_fn is not None:
            try:
                self.devcomm_info = info_fn(self.comm, self.devcomm)
            except Exception as e:  # noqa: BLE001  (diagnostic only)
                log.warning("event=resources devcomm_info_failed rank=%d err=%s", self.rank, e)
        log.info("event=resources devcomm rank=%d sdma_queues=%d windows=%d info=%s", self.rank, self.sdma_queues, len(self._reg_log), self.devcomm_info)
        self.finalized = True
        if self._unzeroed:  # all registrations and the devcomm are done; now zero every window once, then (re)write any control images
            self._zero_windows(self._unzeroed)
            self._unzeroed.clear()
            for hook in list(self._init_hooks.values()):
                hook()
        b = self.budget()
        log.info("event=resources budget rank=%d policy=%s windows=%d registered_bytes=%d (%.1f MiB) shared_bytes=%d heap_bytes=%d "
                 "peer_mapped_va_bytes=%d owners=%d sdma_queues=%d (registered_bytes counts the fusion windows only: MORI's own comm/devcomm allocations are "
                 "NOT included and are unattributed; see the separate hbm delta event)", self.rank, b["policy"], b["windows"], b["registered_bytes"], b["registered_bytes"] / (1 << 20),
                 b["shared_bytes"], b["heap_bytes"], b["peer_mapped_va_bytes"], b["owners"], b["devcomm_sdma_queues"])
        self.backend.device_sync(self.device)
        mem1 = self._mem_info()
        if self._mem_before_comm and mem1:
            delta = self._mem_before_comm[0] - mem1[0]
            reg = b["registered_bytes"]
            log.info("event=resources hbm rank=%d free_before_comm=%d free_after_finalize=%d delta_bytes=%d (%.1f MiB) registered_bytes=%d delta_over_registered=%.3f "
                     "(delta also contains the comm/devcomm internal allocations and any torch allocation in between)", self.rank, self._mem_before_comm[0], mem1[0],
                     delta, delta / (1 << 20), reg, (delta / reg) if reg else 0.0)
        self._barrier()

    def _mem_info(self):
        fn = getattr(self.backend, "mem_info", None)
        if fn is None:
            return None
        try:
            return fn(self.device)
        except Exception:  # noqa: BLE001  (accounting only)
            return None

    @property
    def pending_zero(self) -> int:
        """Windows registered but not yet zeroed (h2d_deferred before finalize()). A launch while this is non-zero would run on garbage control state."""
        return len(self._unzeroed)

    def _zero_windows(self, wins):
        for w in wins:
            self.backend.zero(self.device, w.local, w.bytes)

    def quiesce_and_reset(self):
        """A2-H: device sync -> per-plan drain (large-M: lg_comm_drain_all) + device sync -> barrier -> reset/init windows
        -> device sync -> barrier. All ranks must call it."""
        self.backend.device_sync(self.device)
        self.log_poison("quiesce_and_reset")  # the reason must be logged BEFORE the drain/zero/init below clears it
        for hook in self._drain_hooks.values():
            hook()
        if self._drain_hooks:
            self.backend.device_sync(self.device)
        self._barrier()
        self._zero_windows(self._regs)  # every registered window exactly once (a shared window is not zeroed once per owner)
        self._unzeroed.clear()
        for hook in list(self._init_hooks.values()):  # control-block images; one per owner, one for a shared set
            hook()
        self.backend.device_sync(self.device)
        self._barrier()
        log.info("event=resources quiesce_and_reset rank=%d windows=%d", self.rank, len(self._regs))

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
        for v in self._views:  # arena views are plain device allocations: freed BEFORE (and never deregistered like) the arena window
            self.backend.win_view_free(self.comm, v.handle)
        self._views.clear()
        for w in self._regs:  # each registered window once (the arena is the only registered window under the arena policy)
            self.backend.win_deregister(self.comm, w.handle)
            self.backend.win_free(self.comm, w.local)  # MORI teardown contract: WindowDeregister -> MemFree(localPtr) (before ccoCommDestroy)
        self._regs.clear()
        self._shared.clear()
        self.backend.comm_destroy(self.comm)
        self.windows.clear()
