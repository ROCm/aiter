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

Zeroing (D14 addendum r1a, H-Z): MORI avoids hipMemset on ccoMemAlloc windows (cco_init.cpp CcoZeroWindowMem: chunked hipMemcpy H2D). gf_zero_v1 does the same by
default (AITER_GEMM_FUSION_ZERO_MODE=h2d_deferred): windows are registered WITHOUT any write or device sync in between, and zeroed once in finalize() after every
registration and the devcomm; control images are rewritten right after. h2d_interleaved (MORI's method, right after each owner's registration) and
memset_interleaved (the original hipMemset, right after registration) exist only for A/B. A launch before the zeroing is refused (pending_zero).

Policy 'shared_large' CONTRACT (reviewer C-S3): all large plans share ONE control block and ONE S/R pair, so EVERY rank must launch the same SEQUENCE of large plans
(not merely the same number of launches per plan). A rank that runs a different large plan (or the same plans in a different order) is NOT detected by the device:
the epoch/credit/arrival values still match and layouts would be mixed silently for that step. Divergence is excluded by the host-side rank consensus and detected after
the fact by the host-side per-step digest (ATOM check_step: site|tokens|vid|n_fused), not by the device. recv_view/reuse_view of any large plan alias the shared R
and are valid only until the next large launch on this rank; assert_no_window_alias() (AITER_GEMM_FUSION_DEBUG_ALIAS=1) checks that nothing returned/stored points into a window.
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
POLICIES = ("uniform", "per_owner", "aligned", "shared_large", "abs_aligned", "arena", "arena_per_owner")
ARENA_ALIGN = 4 << 20  # sub-window offsets/sizes inside the arena are multiples of 4 MiB (the size every window used in the passing layouts was a multiple of)
ENV_FREE_BEFORE_DESTROY = "AITER_GEMM_FUSION_FREE_BEFORE_DESTROY"  # default ON: ccoMemFree(local) after each ccoWindowDeregister (MORI's documented teardown); =0 = old behaviour (A/B)
ENV_WAIT_BUDGET_US = "AITER_GEMM_FUSION_WAIT_BUDGET_US"  # TUNING ONLY: overrides the profile's wait_budget_us (large-M per-wait poll budget, microseconds)
DEFAULT_WAIT_BUDGET_US = 5_000_000  # 5 s when neither the env nor the explicit profile carries one (the r11 kernel default of 50 ms is too tight for a serving engine)
# r11 poison reasons (lg_comm_abi_v2.h LGC_POISON_*): which wait failed
LG_POISON_NAMES = {1: "W1_TIMEOUT(credit wait: a peer never began this epoch)", 2: "W2_TIMEOUT(arrival wait: a peer's puts did not complete)",
                   3: "EPOCH_MISMATCH", 4: "DUP_CHUNK", 5: "CONFIG(host/device build or plan mismatch)", 6: "POSTER_WAIT(local GEMM group wait timed out)"}
ENV_ARENA_MIB = "AITER_GEMM_FUSION_ARENA_MIB"  # arena policies: arena size override (MiB)
ENV_ABS_ALIGN = "AITER_GEMM_FUSION_ABS_ALIGN"  # shared_large only: =0 disables the absolute-VA alignment / ordered pre-registration (A/B); default on
ENV_SPACER_GRAN_MIB = "AITER_GEMM_FUSION_SPACER_GRAN_MIB"  # round spacer sizes up to this many MiB (default: the CCO granule G, 4 KiB)
ABS_MIN_ALIGN = 2 << 20  # A(s) floor: 2 MiB
ENV_REG_BARRIER = "AITER_GEMM_FUSION_REG_BARRIER"  # =1: collective barrier after EVERY window registration (E2 seam; default off)
ENV_REG_SYNC = "AITER_GEMM_FUSION_REG_SYNC"  # =1: device sync BEFORE every registration (optional companion of REG_BARRIER)
ENV_LATE_REG = "AITER_GEMM_FUSION_LATE_REG"  # =1: allow ensure() after finalize() (G-late seam; default off)
ENV_ZERO_MODE = "AITER_GEMM_FUSION_ZERO_MODE"  # h2d_deferred (default) | h2d_interleaved | memset_interleaved (the original behaviour, for A/B)
ENV_DEBUG_ALIAS = "AITER_GEMM_FUSION_DEBUG_ALIAS"  # =1: assert_no_window_alias() actually checks (C-S4); default off
ZERO_MODES = ("h2d_deferred", "h2d_interleaved", "memset_interleaved")
SHARED_KEY = "<shared>"  # hook key of a shared window set


class ResourceError(RuntimeError):
    pass


class WindowAliasError(ResourceError):
    """A tensor handed out/stored by the integration points into a CCO window (S/R/C/...): it would be overwritten by a peer or by the next launch."""


def tensor_extent(t):
    """(first_byte_address, one_past_last_byte_address) of a tensor-like (data_ptr/shape/stride/element_size), honouring strides; None if empty."""
    shape = tuple(int(x) for x in t.shape)
    if not shape or any(s == 0 for s in shape):
        return None
    strides = tuple(int(x) for x in t.stride())
    last = sum((s - 1) * st for s, st in zip(shape, strides) if st > 0)
    first = sum((s - 1) * st for s, st in zip(shape, strides) if st < 0)  # negative strides (rare) extend below the base pointer
    base = int(t.data_ptr())
    es = int(t.element_size())
    return base + first * es, base + (last + 1) * es


def iter_tensors(obj, path="out"):
    """Yield (path, tensor-like) from nested dict/list/tuple structures (tensors are recognised by data_ptr/shape/stride)."""
    if hasattr(obj, "data_ptr") and hasattr(obj, "shape") and hasattr(obj, "stride"):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_tensors(v, f"{path}[{k!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from iter_tensors(v, f"{path}[{i}]")
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        for k, v in vars(obj).items():
            yield from iter_tensors(v, f"{path}.{k}")


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


def _flag(arg, env_name: str) -> bool:
    """Tuning flag: explicit arg wins, else the env var ('1'/'true'/'yes'), default off."""
    if arg is not None:
        return bool(arg)
    return os.environ.get(env_name, "").strip().lower() in ("1", "true", "yes", "on")


def _flag_default(arg, env_name: str, default: bool) -> bool:
    if arg is not None:
        return bool(arg)
    v = os.environ.get(env_name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def policy_memory_table(budget: dict, align_cap: int = 256 << 20) -> Dict[str, int]:
    """Predicted REGISTERED window bytes per rank for every policy, from a profile window_budget (log/accounting; excludes devcomm + slack).
    uniform: every window the max size; per_owner: each owner's own size; aligned: per_owner + worst-case spacer padding;
    shared_large: small owners' windows + ONE shared S, R and C set."""
    sizes = [int(x) for x in (budget.get("window_sizes") or [])]
    out: Dict[str, int] = {}
    if budget.get("window_bytes") and budget.get("max_windows"):
        out["uniform"] = int(budget["window_bytes"]) * int(budget["max_windows"])
    if sizes:
        out["per_owner"] = sum(sizes)
        out["aligned"] = aligned_total_bytes(sizes, align_cap)
    sh = budget.get("shared") or {}
    small = [int(x) for x in (budget.get("small_window_sizes") or [])]
    if sh and small:
        out["shared_large"] = sum(small) + sum(int(v) for v in sh.values())
    if sh and small:
        out["arena"] = sum(-(-int(v) // ARENA_ALIGN) * ARENA_ALIGN for v in sh.values()) + sum(-(-s // ARENA_ALIGN) * ARENA_ALIGN for s in small)
    if sizes:
        out["arena_per_owner"] = sum(-(-s // ARENA_ALIGN) * ARENA_ALIGN for s in sizes)
    for pol, key, fill in (("abs_aligned", "abs_aligned_ub", False), ("shared_large", "shared_large_abs_ub", True)):
        items = abs_items_from_budget(budget, pol)
        if items:
            out[key] = abs_layout_total_bytes(items, fill_gaps=fill, cap=align_cap)  # windows + spacers, worst over the sampled base misalignments
    return out


def abs_align_of(size: int, cap: int = 256 << 20, amin: int = ABS_MIN_ALIGN) -> int:
    """A(s) = max(2 MiB, min(next_pow2(s), 256 MiB)): the ABSOLUTE start-VA alignment the abs_aligned / shared_large policies give a window of `size`."""
    return max(int(amin), min(next_pow2(size), int(cap)))


class AbsLayout:
    """Decides, one registration at a time, what to register next so that every window starts at an ABSOLUTE VA aligned to A(size) (D14 r1c, H-BASE).
    Windows are placed LARGEST FIRST. Before a window of size s a spacer of (-va_top) mod A(s) bytes (rounded up to `gran`) is registered, computed from the
    measured absolute VA where the next slot will start. With fill_gaps=True (shared_large) pending windows that fit entirely inside that gap and are themselves
    aligned there are registered first instead of wasting it: S(164) -> 52 MiB of small windows + control -> 40 MiB spacer -> R at +256 MiB (the Architect's sequence).
    Pure: the policy drives it with the real VA, tests/probes with a simulated monotonic allocator. `items` are (size, role) pairs (role is opaque)."""

    def __init__(self, items, cap: int = 256 << 20, amin: int = ABS_MIN_ALIGN, gran: int = 4096, fill_gaps: bool = False, spacer_gran: int = 0):
        self.pending = sorted(((int(s), r) for s, r in items), key=lambda x: -x[0])  # stable: equal sizes keep the given role order (S before R)
        self.cap, self.amin, self.gran, self.fill_gaps = int(cap), int(amin), max(1, int(gran)), fill_gaps
        self.spacer_gran = max(self.gran, int(spacer_gran or 0))
        self._spaced_for = None

    def align_of(self, s: int) -> int:
        return abs_align_of(s, self.cap, self.amin)

    def done(self) -> bool:
        return not self.pending

    def next(self, va_top: int):
        """-> None | ("window", size, role) | ("spacer", pad)."""
        if not self.pending:
            return None
        s, role = self.pending[0]
        a = self.align_of(s)
        pad = (-int(va_top)) % a
        if pad:
            pad = -(-pad // self.spacer_gran) * self.spacer_gran
        # alignment unreachable after we already spaced for this window (granularity larger than the residue): place it, never loop
        if pad == 0 or self._spaced_for == id(self.pending[0]):
            self._spaced_for = None
            self.pending.pop(0)
            return ("window", s, role)
        if self.fill_gaps:
            for i in range(1, len(self.pending)):
                w, wr = self.pending[i]
                if w <= pad and int(va_top) % self.align_of(w) == 0:
                    self.pending.pop(i)
                    return ("window", w, wr)
        self._spaced_for = id(self.pending[0])
        return ("spacer", pad)


def abs_layout_simulate(items, base_va: int, fill_gaps: bool = False, cap: int = 256 << 20, gran: int = 4096, spacer_gran: int = 0):
    """Run AbsLayout against a monotonic first-fit allocator starting at absolute `base_va`. Returns [(kind, size, va, role)] (kind window|spacer)."""
    lay, va, out = AbsLayout(items, cap=cap, gran=gran, fill_gaps=fill_gaps, spacer_gran=spacer_gran), int(base_va), []
    while True:
        act = lay.next(va)
        if act is None:
            return out
        if act[0] == "spacer":
            out.append(("spacer", act[1], va, None))
            va += act[1]
        else:
            out.append(("window", act[1], va, act[2]))
            va += -(-act[1] // gran) * gran


# base VAs (mod 256 MiB) that bracket the process-dependent flatBase misalignment: aligned, 4 KiB, 2 MiB +/- 4 KiB, 128 MiB + 4 KiB, 256 MiB - 4 KiB
_SAMPLE_BASES = (0, 4096, (2 << 20) - 4096, (2 << 20) + 4096, (128 << 20) + 4096, (256 << 20) - 4096)


def abs_layout_total_bytes(items, fill_gaps: bool = False, cap: int = 256 << 20, spacer_gran: int = 0) -> int:
    """Upper bound (over the sample base misalignments) of windows + spacers registered by the abs layout: used for heap sizing / the memory table."""
    base = 0x7F0000000000  # 4 GiB aligned: only its low 28 bits matter
    best = 0
    for m in _SAMPLE_BASES:
        lay = abs_layout_simulate(items, base + m, fill_gaps=fill_gaps, cap=cap, spacer_gran=spacer_gran)
        best = max(best, sum(sz for _k, sz, _v, _r in lay))
    return best


def abs_items_from_budget(budget: dict, policy: str):
    """(size, role) items of the pre-registration for a policy, from a profile window_budget; None if the budget cannot describe it.
    abs_aligned: every window of window_sizes (role None). shared_large: shared S, R (and C) tags first, then the small windows."""
    if policy == "abs_aligned":
        sizes = [int(x) for x in (budget.get("window_sizes") or [])]
        return [(s, None) for s in sizes] or None
    if policy == "shared_large":
        sh = budget.get("shared") or {}
        small = [int(x) for x in (budget.get("small_window_sizes") or [])]
        if not sh or not small:
            return None
        def _rank(tag):  # S before R before C (the Architect's sequence); unknown tags last, by name
            for i, suffix in enumerate(("/S", "/R", "/C")):
                if tag.endswith(suffix):
                    return (i, tag)
            return (3, tag)

        tags = sorted(sh.items(), key=lambda kv: (-int(kv[1]), _rank(kv[0])))  # big windows first (S then R), the 4 MiB control tag after them
        return [(int(v), ("shared", t)) for t, v in tags] + [(s, None) for s in small]
    return None


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
    shared: bool = False  # one registered window used by several owners (shared_large policy)
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
            ("gf_win_register_v1", [vp, u64, C.POINTER(vp), C.POINTER(vp)]), ("gf_peer_ptr_v1", [vp, vp, i32, C.POINTER(vp)]),
            ("gf_devcomm_create_v1", [vp, u32, C.POINTER(vp)]), ("gf_devcomm_info_v1", [vp, vp, C.POINTER(u64)]), ("gf_zero_v1", [i32, vp, u64, i32]), ("gf_mem_info_v1", [i32, C.POINTER(u64)]), ("gf_comm_va_info_v1", [vp, C.POINTER(u64)]), ("gf_win_free_v1", [vp, vp]),
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

    def peer_ptr(self, comm, local, peer):
        o = C.c_void_p()
        self._chk(self.lib.gf_peer_ptr_v1(comm, local, peer, C.byref(o)), "gf_peer_ptr")
        return o.value

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

    _zero_method = 0  # 0 = MORI's chunked H2D copy (default), 1 = hipMemset (A/B only)

    def set_zero_mode(self, mode: str):
        self._zero_method = 1 if mode == "memset_interleaved" else 0

    def zero(self, device, local, nbytes):
        self._chk(self.lib.gf_zero_v1(device, local, nbytes, self._zero_method), "gf_zero")

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

    def va_info(self, comm):
        """(slot_base_va, per_rank_size, granule) of this rank's slot: where the first-fit allocator starts (absolute VA)."""
        b = (C.c_uint64 * 3)()
        self._chk(self.lib.gf_comm_va_info_v1(comm, b), "gf_comm_va_info")
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
    changing the env/profile afterwards does not affect graphs that were already captured."""
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
                 caps: frozenset = frozenset(), vmm_bytes: Optional[int] = None, window_bytes: Optional[int] = None,
                 max_windows: Optional[int] = None, policy: Optional[str] = None, profile=None,
                 reg_barrier: Optional[bool] = None, reg_sync: Optional[bool] = None, late_registration: Optional[bool] = None,
                 zero_mode: Optional[str] = None):
        if dist is None:
            import torch.distributed as dist  # type: ignore
        self.rank, self.world, self.group, self.dist = rank, world, group, dist
        self.device = self._default_device(rank) if device is None else int(device)
        self.caps = frozenset(caps)
        self.backend = backend or CcoBackend()
        self.policy = (policy or os.environ.get(ENV_POLICY) or "uniform").lower()
        if self.policy not in POLICIES:
            raise ResourceError(f"{ENV_POLICY} must be one of {POLICIES}, got {self.policy!r}")
        self.align_cap = int(os.environ.get(ENV_ALIGN_CAP_MIB, "256")) << 20
        self.abs_align = self.policy == "abs_aligned" or (self.policy == "shared_large" and _flag_default(None, ENV_ABS_ALIGN, True))
        self.spacer_gran = int(os.environ.get(ENV_SPACER_GRAN_MIB, "0") or 0) << 20
        self._window_sizes: List[int] = []
        self._budget_doc: dict = {}
        self.zero_mode = (zero_mode or os.environ.get(ENV_ZERO_MODE) or "h2d_deferred").lower()
        if self.zero_mode not in ZERO_MODES:
            raise ResourceError(f"{ENV_ZERO_MODE} must be one of {ZERO_MODES}, got {self.zero_mode!r}")
        if hasattr(self.backend, "set_zero_mode"):
            self.backend.set_zero_mode(self.zero_mode)
        self.debug_alias = _flag(None, ENV_DEBUG_ALIAS)
        self.reg_barrier = _flag(reg_barrier, ENV_REG_BARRIER)
        self.reg_sync = _flag(reg_sync, ENV_REG_SYNC)
        self.allow_late = _flag(late_registration, ENV_LATE_REG)
        self.profile = self._resolve_profile(profile)  # explicit only: a Profile object or a name/path; NEVER the AITER_GEMM_FUSION_PROFILE env
        self.wait_budget_us, self.wait_budget_source = resolve_wait_budget_us(self.profile)
        self.window_bytes, self.max_windows = self._budget(window_bytes, max_windows)
        if vmm_bytes is None:
            if self.policy == "uniform" and self.window_bytes:
                vmm_bytes = heap_bytes_for(self.window_bytes, self.max_windows)
            elif self.policy in ("arena", "arena_per_owner"):
                g = 4 << 30
                need = self._arena_size_from_budget() + HEAP_SLACK_BYTES
                vmm_bytes = max(PER_RANK_VMM_BYTES, (need + g - 1) // g * g)
            elif self.policy in ("abs_aligned", "shared_large") and self.abs_align and abs_items_from_budget(self._budget_doc, self.policy):
                g = 4 << 30
                need = abs_layout_total_bytes(abs_items_from_budget(self._budget_doc, self.policy), fill_gaps=self.policy == "shared_large",
                                              cap=self.align_cap, spacer_gran=self.spacer_gran) + HEAP_SLACK_BYTES
                vmm_bytes = max(PER_RANK_VMM_BYTES, (need + g - 1) // g * g)
            elif self.policy == "shared_large" and self._budget_doc.get("shared_large_total_bytes"):
                g = 4 << 30
                need = int(self._budget_doc["shared_large_total_bytes"]) + HEAP_SLACK_BYTES
                vmm_bytes = max(PER_RANK_VMM_BYTES, (need + g - 1) // g * g)
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
        self._regs: List[Window] = []  # every registered window exactly ONCE (shared windows appear once): zeroing / deregistration / accounting
        self._shared: Dict[str, Window] = {}  # share tag -> the one registered window
        self._reg_seq = 0  # number of registering ensure() calls (agreement check)
        self._arena: Optional[Window] = None  # the ONE registered window of the comm (arena policies)
        self._arena_bytes = 0
        self._arena_top = 0  # bump pointer inside the arena (4 MiB aligned)
        self._views: List[Window] = []  # every view struct handed out (freed before the arena is deregistered)
        self._va_top: Optional[int] = None  # ABSOLUTE local VA where the next slot will start (measured: previous local + aligned size)
        self._gran = 4096  # CCO granule G (hipMemGetAllocationGranularity RECOMMENDED; reported 4096)
        self._va_log: List[Tuple[str, str, int, int]] = []  # (owner, name, size, absolute local VA) of every registration, in order
        self._pool: List[Window] = []  # abs policies: pre-registered per-owner windows not yet claimed by an ensure()
        self._prereg_done = False
        self._unzeroed: List[Window] = []  # h2d_deferred: windows registered but not yet zeroed (zeroed in finalize(), after ALL registrations + the devcomm)
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

    def _budget(self, window_bytes, max_windows):
        """Uniform window size / count. Precedence: explicit args, then the tuning env AITER_GEMM_FUSION_WINDOW_MIB (size only), then the EXPLICIT
        ``profile=`` window_budget. There is no implicit profile lookup (the process env var that selects a profile is not read here).
        (None, 0) when unknown: the first request then sizes the uniform window (a warning is logged; see ensure)."""
        if window_bytes is None and os.environ.get(ENV_WINDOW_MIB):
            window_bytes = int(os.environ[ENV_WINDOW_MIB]) << 20
        if self.profile is not None:
            from .profile import window_budget_of

            wb = window_budget_of(self.profile)
            window_bytes = window_bytes if window_bytes is not None else wb.get("window_bytes")
            max_windows = max_windows if max_windows is not None else wb.get("max_windows")
            self._window_sizes = [int(x) for x in (wb.get("window_sizes") or [])]
            self._budget_doc = wb
        if window_bytes is None:
            log.warning("event=resources no_budget rank=%d profile=%s: no window budget available (%s); the uniform window size will be taken "
                        "from the FIRST ensure() request and a later larger plan will be rejected. Pass profile=<name|path|Profile> whose "
                        "window_budget is set, or window_bytes=/max_windows=.", self.rank, getattr(self.profile, "name", None),
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
        log.info("event=resources comm rank=%d world=%d heap_bytes=%d policy=%s window_bytes=%s max_windows=%s reg_barrier=%d reg_sync=%d late=%d zero_mode=%s wait_budget_us=%d (%s)",
                 self.rank, self.world, self._vmm, self.policy, self.window_bytes, self.max_windows, self.reg_barrier, self.reg_sync, self.allow_late, self.zero_mode,
                 self.wait_budget_us, self.wait_budget_source)
        if self._budget_doc:
            tbl = policy_memory_table(self._budget_doc, self.align_cap)
            log.info("event=resources policy_memory rank=%d selected=%s %s", self.rank, self.policy,
                     " ".join(f"{k}={v / (1 << 20):.0f}MiB" for k, v in tbl.items()))

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
        if self.reg_sync:  # E2 seam (optional): no outstanding device work while this rank allocates/maps
            self.backend.device_sync(self.device)
        try:
            h, loc = self.backend.win_register(self.comm, size)
        except ResourceError as e:
            raise ResourceError(
                f"{e} while registering {owner}/{name}{' (alignment spacer)' if spacer else ''}: size={size} B, "
                f"window #{len(self._reg_log) + 1}, expected offset {expect} B, {cum - size} B already registered on this rank, "
                f"heap={self._vmm} B, policy={self.policy}; layout so far: {self._reg_log[-8:]}") from e
        if self._slot_base is None:
            self._slot_base = int(loc)
        va = int(loc)
        if self._va_top is not None and va != self._va_top:
            log.warning("event=resources va_mismatch rank=%d owner=%s name=%s expected_va=%#x measured_va=%#x (allocator is not the monotonic first-fit the abs layout assumes)",
                        self.rank, owner, name, self._va_top, va)
        self._va_top = va + (-(-int(size) // self._gran) * self._gran)
        self._va_log.append((owner, name, int(size), va))
        off = self._offset_of(loc)
        if off != expect:
            log.warning("event=resources offset_mismatch rank=%d owner=%s name=%s expected_offset=%d measured_offset=%s "
                        "(allocator is not the monotonic first-fit the aligned policy assumes)", self.rank, owner, name, expect, off)
        self._top = (off if off is not None else expect) + size
        self._reg_log.append((owner, name + ("[spacer]" if spacer else ""), size, cum))
        MiB = 1 << 20
        log.info("event=resources window rank=%d owner=%s name=%s bytes=%d n=%d cumulative=%d offset=%s heap=%d%s va=%#x va_mod_2MiB=%d va_mod_16MiB=%d "
                 "va_mod_64MiB=%d va_mod_256MiB=%d", self.rank, owner, name, size, len(self._reg_log), cum, off, self._vmm, " spacer=1" if spacer else "", va,
                 va % (2 * MiB), va % (16 * MiB), va % (64 * MiB), va % (256 * MiB))
        if self.reg_barrier or self.finalized:  # E2 seam: no rank starts allocating window k+1 while a peer may still be importing/mapping window k
            self._barrier()
        return h, loc

    def _pad_to_alignment(self, owner: str, name: str, align: int):
        """Aligned policy: ccoMemAlloc has no alignment argument (HeapVAManager::Allocate(size, granule)), so pad the heap top
        with an unused spacer window until the NEXT registration starts at a multiple of `align`."""
        rem = self._top % align
        if rem:
            pad = align - rem
            h, loc = self._register(owner, name + ".pad", pad, spacer=True)
            self._spacers.append((h, loc, pad))

    # -- absolute VA alignment (policies abs_aligned, shared_large) -------------------------------------------------
    # -- ARENA (D14 r1e/r1f): ONE registered window per comm, sub-allocated by aiter; every plan window is a VIEW struct --------------
    def _arena_size_from_budget(self) -> int:
        """Arena bytes: AITER_GEMM_FUSION_ARENA_MIB, else the explicit profile budget (arena: shared S/R/C + small windows; arena_per_owner: every per-owner window)."""
        env = os.environ.get(ENV_ARENA_MIB)
        if env:
            return round_up(int(env) << 20)
        b = self._budget_doc or {}
        if self.policy == "arena":
            sh, small = b.get("shared") or {}, b.get("small_window_sizes") or []
            if sh and small:
                return sum(round_up(int(v)) for v in sh.values()) + sum(round_up(int(s)) for s in small)
        else:
            sizes = b.get("window_sizes") or []
            if sizes:
                return sum(round_up(int(s)) for s in sizes)
        return 0

    def _ensure_arena(self, late: bool) -> Window:
        if self._arena is not None:
            return self._arena
        size = self._arena_size_from_budget()
        if not size:
            raise ResourceError(f"policy {self.policy!r} needs the arena size: construct FusionResources(profile=<profile with window_budget>) or set {ENV_ARENA_MIB}")
        if late or self.reg_barrier or self.world > 1:
            self._agree_registration("<arena>", [WindowReq("arena", size, "p2p_window")])
        h, loc = self._register("<arena>", "arena", size)  # THE registration: a single window, first, never followed by another size
        w = Window("<arena>", "arena", "p2p_window", size, h, loc, shared=True)
        self._arena, self._arena_bytes, self._arena_top = w, size, 0
        self._regs.append(w)
        if self.zero_mode == "h2d_deferred" and not late:
            self._unzeroed.append(w)
        else:
            self._zero_windows([w])
        log.info("event=resources arena rank=%d policy=%s bytes=%d (%.1f MiB) local=%#x", self.rank, self.policy, size, size / (1 << 20), int(loc))
        return w

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
        w = Window(SHARED_KEY if share else owner, share or name, kind, size, view, int(a.local) + off, shared=bool(share), arena_offset=off)
        self._views.append(w)
        return w

    def _arena_ensure(self, owner: str, wreqs, late: bool, wins: Dict[str, Window]):
        self._ensure_arena(late)
        if late:
            self._refuse_inside_capture(owner)
        # deterministic offsets: the shared set is laid out in the budget order (S, R, C) at its FIRST use, then owners bump in ensure() order; peers address
        # this rank's arena at the SAME offsets, so the layout is verified collectively below.
        for r in wreqs:
            if self.policy == "arena" and r.share:
                w = self._shared.get(r.share)
                need = int(r.min_bytes if r.min_bytes is not None else r.bytes)
                if w is None:
                    # the shared set is sized from the profile's shared budget (the LARGEST large plan), never from whichever plan is ensured first
                    budget = int(((self._budget_doc or {}).get("shared") or {}).get(r.share) or 0)
                    if budget and need > budget:
                        raise ResourceError(f"ensure({owner!r}): shared arena window {r.share!r} need {need} B exceeds the budgeted {budget} B")
                    w = self._arena_view(owner, r.name, r.kind, budget or need, share=r.share)
                    self._shared[r.share] = w
                elif int(r.min_bytes if r.min_bytes is not None else r.bytes) > w.bytes:
                    raise ResourceError(f"ensure({owner!r}): shared arena window {r.share!r} need exceeds its size {w.bytes} B")
                wins[r.name] = w
            else:
                wins[r.name] = self._arena_view(owner, r.name, r.kind, round_up(max(x.bytes for x in wreqs)) if self.policy == "arena_per_owner" else r.bytes)
        offs = tuple((n, w.arena_offset, w.bytes) for n, w in wins.items())
        got = [None] * self.world
        self.dist.all_gather_object(got, (self._reg_seq, owner, offs), group=self.group)
        if any(g != got[0] for g in got):
            raise ResourceError(f"arena layout disagreement across ranks at ensure({owner!r}): {got}; peers address this arena at identical offsets, fusion must be aborted")

    def _init_va(self):
        """Absolute VA where the first slot will start, from the comm (no registration needed). Backends without va_info: None (the first
        window is then registered unaligned and the alignment starts from its measured VA)."""
        if self._va_top is not None:
            return
        fn = getattr(self.backend, "va_info", None)
        if fn is None:
            return
        try:
            base, _per_rank, gran = fn(self.comm)
        except Exception as e:  # noqa: BLE001
            log.warning("event=resources va_info_failed rank=%d err=%s", self.rank, e)
            return
        self._va_top = int(base)
        self._gran = int(gran) or 4096
        log.info("event=resources va_base rank=%d slot_base=%#x granule=%d base_mod_2MiB=%d base_mod_256MiB=%d", self.rank, self._va_top, self._gran,
                 self._va_top % (2 << 20), self._va_top % (256 << 20))

    def _abs_pad(self, owner: str, name: str, size: int):
        """On-demand absolute alignment (no pre-registration plan): one spacer so the window starts at a multiple of A(size)."""
        if self._va_top is None:
            return
        lay = AbsLayout([(size, None)], cap=self.align_cap, gran=self._gran, spacer_gran=self.spacer_gran)
        act = lay.next(self._va_top)
        if act and act[0] == "spacer":
            h, loc = self._register(owner, name + ".abspad", act[1], spacer=True)
            self._spacers.append((h, loc, act[1]))

    def _preregister(self, late: bool):
        """Register the WHOLE budgeted window set once, LARGEST FIRST, each at an absolute VA aligned to A(size) (D14 r1b/r1c). Deterministic from the explicit
        profile budget, so every rank issues the same sequence. abs_aligned: every window goes to a pool that ensure() claims best-fit. shared_large: the shared
        S/R/C windows are registered at their place of the Architect's sequence (S, 52 MiB of small windows + control, spacer, R at +256 MiB), small windows pooled."""
        if self._prereg_done:
            return
        self._prereg_done = True
        items = abs_items_from_budget(self._budget_doc, self.policy) if self._budget_doc else None
        if not items:
            log.warning("event=resources abs_no_budget rank=%d policy=%s: the profile window_budget cannot describe the pre-registration; windows are registered "
                        "on demand (each absolutely aligned, but NOT largest-first)", self.rank, self.policy)
            return
        self._init_va()
        if late or self.reg_barrier:
            self._agree_registration("<prereg>", [WindowReq(str(r), int(s), "p2p_window") for s, r in items])
        lay = AbsLayout(items, cap=self.align_cap, gran=self._gran, fill_gaps=self.policy == "shared_large", spacer_gran=self.spacer_gran)
        fresh: List[Window] = []
        while True:
            if self._va_top is None:  # no va_info: register the first (largest) window unaligned, align from its measured VA on
                act = ("window",) + (lay.pending[0] if lay.pending else (0, None))
                if lay.pending:
                    lay.pending.pop(0)
            else:
                act = lay.next(self._va_top)
            if act is None or (act[0] == "window" and not act[1]):
                break
            if act[0] == "spacer":
                h, loc = self._register("<prereg>", "abspad", act[1], spacer=True)
                self._spacers.append((h, loc, act[1]))
                continue
            _k, size, role = act
            if role and role[0] == "shared":
                tag = role[1]
                h, loc = self._register(SHARED_KEY, tag, size)
                kind = "control" if tag.endswith("/C") else "p2p_window"
                w = Window(SHARED_KEY, tag, kind, size, h, loc, shared=True)
                self._shared[tag] = w
            else:
                h, loc = self._register("<pool>", f"pool{size >> 20}M", size)
                w = Window("<pool>", f"pool{size >> 20}M", "p2p_window", size, h, loc)
                self._pool.append(w)
            self._regs.append(w)
            fresh.append(w)
            if self.zero_mode != "h2d_deferred" or late:
                self._zero_windows([w])  # interleaved A/B modes / late: right after the registration (deferred mode zeroes in finalize())
                fresh.pop()
        if self.zero_mode == "h2d_deferred" and not late:
            self._unzeroed.extend(fresh)
        log.info("event=resources prereg_done rank=%d policy=%s windows=%d spacers=%d spacer_bytes=%d", self.rank, self.policy, len(self._regs), len(self._spacers),
                 sum(n for _h, _l, n in self._spacers))

    def _claim_pooled(self, owner: str, r: WindowReq, size: int) -> Optional[Window]:
        """Best-fit (smallest pooled window >= size). The window object is re-labelled to its owner; it is already in _regs and already queued for zeroing."""
        best = None
        for w in self._pool:
            if w.bytes >= size and (best is None or w.bytes < best.bytes):
                best = w
        if best is None:
            return None
        self._pool.remove(best)
        best.owner, best.name, best.kind = owner, r.name, r.kind
        return best

    def ensure(self, owner: str, reqs: List[WindowReq]) -> Dict[str, Window]:
        """Register this owner's windows once (idempotent per owner; collective on first call)."""
        if owner in self.windows:
            return self.windows[owner]
        late = self.finalized
        if self.policy in ("arena", "arena_per_owner"):
            # no collective REGISTRATION after the first one: a late owner only needs a device-memory view struct, so G-late is not required (and capture is refused)
            wreqs0 = [r for r in reqs if r.kind in ("p2p_window", "control")]
            wins0: Dict[str, Window] = {}
            self._reg_seq += 1
            if wreqs0:
                self._arena_ensure(owner, wreqs0, late, wins0)
            self._sdma = max([getattr(self, "_sdma", 0)] + [max(1, int(r.bytes)) for r in reqs if r.kind == "sdma_queue"])
            self.windows[owner] = wins0
            return wins0
        if late and not self.allow_late:
            raise ResourceError(
                f"ensure({owner!r}) after finalize(): resolve every plan on every rank before finalize (collective ordering); "
                f"set {ENV_LATE_REG}=1 (late_registration=True) to allow registering after the devcomm exists"
            )
        wins: Dict[str, Window] = {}
        fresh: List[Window] = []  # windows created by THIS call (only those are zeroed: a shared window may already be live)
        wreqs = [r for r in reqs if r.kind in ("p2p_window", "control")]
        if late:
            self._refuse_inside_capture(owner)
        if wreqs and (late or self.reg_barrier):
            self._agree_registration(owner, wreqs)  # D14 rule 2: same (seq, owner, sizes) on every rank, else abort fusion
        self._reg_seq += 1
        if wreqs and self.abs_align and self.policy in ("abs_aligned", "shared_large"):
            self._preregister(late)
        need = round_up(max((r.bytes for r in wreqs), default=0)) if wreqs else 0
        if self.policy == "uniform" and wreqs:
            # Phase-1 layout: EVERY window of the shared comm has the same size (r8g: equal, 4 MiB multiple), so slot offsets are
            # multiples of that size exactly as in the accepted drivers. Size = profile budget / env, else the first request.
            if self.window_bytes is None:
                self.window_bytes = need
            if need > self.window_bytes:
                raise ResourceError(
                    f"ensure({owner!r}): window need {need} B exceeds the uniform window size {self.window_bytes} B; "
                    f"pre-size by constructing FusionResources(profile=<the active profile>) (its window_budget), window_bytes=, or {ENV_WINDOW_MIB}")
            size = self.window_bytes
        else:
            size = need  # per_owner, aligned, shared_large (unshared requests): per-plan sizes (4 MiB multiples)
        for r in wreqs:
            if self.policy == "shared_large" and r.share:
                wins[r.name] = self._shared_window(owner, r, fresh)
                continue
            if self.policy == "aligned":
                self._pad_to_alignment(owner, r.name, start_alignment(size, self.align_cap))
            if self.abs_align and self.policy in ("abs_aligned", "shared_large"):
                pooled = self._claim_pooled(owner, r, size)
                if pooled is not None:
                    wins[r.name] = pooled  # pre-registered (largest first, absolutely aligned) in _preregister()
                    continue
                if self._prereg_done and self._budget_doc and abs_items_from_budget(self._budget_doc, self.policy):
                    log.warning("event=resources pool_miss rank=%d owner=%s name=%s need=%d: no pre-registered window fits; registering on demand", self.rank, owner, r.name, size)
                self._init_va()
                self._abs_pad(owner, r.name, size)
            h, loc = self._register(owner, r.name, size)
            w = Window(owner, r.name, r.kind, size, h, loc)
            wins[r.name] = w
            self._regs.append(w)
            fresh.append(w)
        # sdma_queue request: bytes = queues PER PEER wanted (0 = default 1). The shared devcomm gets the MAX over plans, never the
        # number of plans (9 large plans must not ask CCO for 9 queues: the r11 kernels require exactly 1, as in Phase-1).
        self._sdma = max([getattr(self, "_sdma", 0)] + [max(1, int(r.bytes)) for r in reqs if r.kind == "sdma_queue"])
        self.windows[owner] = wins
        if self.zero_mode == "h2d_deferred" and not late:
            self._unzeroed.extend(fresh)  # NOT between registrations: zeroed in finalize(), after every registration and the devcomm (D14 addendum r1a)
        else:
            self._zero_windows(fresh)  # interleaved modes, and any late registration (the devcomm already exists). NEVER a pre-existing shared window (it may be live)
        if late:
            log.info("event=resources late_registration rank=%d owner=%s windows=%d (after ccoDevCommCreate; barrier done)", self.rank, owner, len(fresh))
        return wins

    # -- shared windows (policy shared_large) --------------------------------------
    def _shared_window(self, owner: str, r: WindowReq, fresh: List[Window]) -> Window:
        """One registered window per share tag. Size = the profile budget for the tag, else the first request (a later larger request is rejected)."""
        need = round_up(int(r.min_bytes if r.min_bytes is not None else r.bytes))
        w = self._shared.get(r.share)
        if w is not None:
            if need > w.bytes:
                raise ResourceError(f"ensure({owner!r}): shared window {r.share!r} need {need} B exceeds its size {w.bytes} B; pre-size via "
                                    f"FusionResources(profile=...) window_budget['shared'][{r.share!r}]")
            return w
        budget = (self._budget_doc.get("shared") or {}).get(r.share)
        size = round_up(int(budget)) if budget else need
        if need > size:
            raise ResourceError(f"ensure({owner!r}): shared window {r.share!r} need {need} B exceeds the budgeted {size} B (profile window_budget['shared'])")
        if self.abs_align:
            self._init_va()
            self._abs_pad(SHARED_KEY, r.share, size)
        h, loc = self._register(SHARED_KEY, r.share, size)
        w = Window(SHARED_KEY, r.share, r.kind, size, h, loc, shared=True)
        self._shared[r.share] = w
        self._regs.append(w)
        fresh.append(w)
        return w

    # -- collective safety of (lazy / late) registration ---------------------------
    def _refuse_inside_capture(self, owner: str):
        try:
            import torch

            cap = getattr(getattr(torch, "cuda", None), "is_current_stream_capturing", None)
            if cap is not None and cap():
                raise ResourceError(f"ensure({owner!r}): window registration is a collective HIP/VMM operation and is refused inside graph capture")
        except ResourceError:
            raise
        except Exception:  # noqa: BLE001  (no torch/cuda: nothing to refuse)
            pass

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

    # -- memory accounting ----------------------------------------------------------
    def budget(self) -> dict:
        """Per-rank accounting of what THIS process registered: bytes per window, shared bytes, spacers, VA heap, peer-mapped VA."""
        total = sum(w.bytes for w in self._regs)
        spacer = sum(n for _h, _l, n in self._spacers)
        shared = sum(w.bytes for w in self._regs if w.shared)
        return {"policy": self.policy, "windows": len(self._regs), "registered_bytes": total, "shared_bytes": shared, "spacer_bytes": spacer,
                "heap_bytes": self._vmm, "peer_mapped_va_bytes": (self.world - 1) * (total + spacer), "owners": len(self.windows),
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

    def peers(self, win: Window) -> List[int]:
        return [self.backend.peer_ptr(self.comm, win.local, p) for p in range(self.world)]

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
        if self._unzeroed:  # h2d_deferred: all registrations and the devcomm are done; now zero every window once, then (re)write any control images
            self._zero_windows(self._unzeroed)
            self._unzeroed.clear()
            for hook in list(self._init_hooks.values()):
                hook()
        b = self.budget()
        log.info("event=resources budget rank=%d policy=%s windows=%d registered_bytes=%d (%.1f MiB) shared_bytes=%d spacer_bytes=%d heap_bytes=%d "
                 "peer_mapped_va_bytes=%d owners=%d sdma_queues=%d", self.rank, b["policy"], b["windows"], b["registered_bytes"], b["registered_bytes"] / (1 << 20),
                 b["shared_bytes"], b["spacer_bytes"], b["heap_bytes"], b["peer_mapped_va_bytes"], b["owners"], b["devcomm_sdma_queues"])
        self.backend.device_sync(self.device)
        mem1 = self._mem_info()
        if self._mem_before_comm and mem1:
            delta = self._mem_before_comm[0] - mem1[0]
            reg = b["registered_bytes"] + b["spacer_bytes"]
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

    # -- C-S4: nothing returned or stored may point into a CCO window ----------------------------------------------
    def window_ranges(self, kinds=("p2p_window", "control")) -> List[Tuple[str, int, int]]:
        """[(label, start, end)] of every registered window (+ alignment spacers) in THIS process's address space."""
        out = [(f"{w.owner}/{w.name}" + (" [shared]" if w.shared else ""), int(w.local), int(w.local) + int(w.bytes)) for w in self._regs if w.kind in kinds]
        out += [(f"spacer#{i}", int(loc), int(loc) + int(n)) for i, (_h, loc, n) in enumerate(self._spacers)]
        return out

    def find_window_alias(self, obj, kinds=("p2p_window", "control")) -> List[str]:
        """Names of the CCO windows that any tensor reachable from `obj` (nested dict/list/tuple/object attributes) overlaps. Empty = clean."""
        wins = self.window_ranges(kinds)
        hits = []
        for path, t in iter_tensors(obj):
            ext = tensor_extent(t)
            if ext is None:
                continue
            for label, a, b in wins:
                if ext[0] < b and a < ext[1]:
                    hits.append(f"{path} [{ext[0]:#x},{ext[1]:#x}) overlaps window {label} [{a:#x},{b:#x})")
        return hits

    def assert_no_window_alias(self, obj, what: str = "gemm_fusion output", force: bool = False):
        """The recv_view / reuse_view of a large plan aliases the R window (shared_large: the SAME R for every large plan), valid only until the next large
        launch on this rank. A consumer must finish within the op: nothing RETURNED from the custom op or STORED on a long-lived object (e.g. attn.impl._index_q)
        may point into a window. Call this on the op's results/attributes (cheap but O(windows x tensors)): it is a no-op unless
        AITER_GEMM_FUSION_DEBUG_ALIAS=1 (or force=True / the resources' debug_alias flag). Raises WindowAliasError."""
        if not (force or self.debug_alias):
            return
        hits = self.find_window_alias(obj)
        if hits:
            raise WindowAliasError(f"{what} points into a CCO window (valid only until the next launch; copy it within the op): " + "; ".join(hits))

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
        for v in self._views:  # arena views are plain device allocations: freed BEFORE (and never deregistered like) the arena window
            self.backend.win_view_free(self.comm, v.handle)
        self._views.clear()
        free = getattr(self.backend, "win_free", None) if _flag_default(None, ENV_FREE_BEFORE_DESTROY, True) else None
        for w in self._regs:  # each registered window once (a shared window is referenced by several owners; the arena is the only registered window)
            self.backend.win_deregister(self.comm, w.handle)
            if free is not None:
                free(self.comm, w.local)  # MORI teardown contract: WindowDeregister -> MemFree(localPtr) (before ccoCommDestroy)
        self._regs.clear()
        self._shared.clear()
        for h, loc, _n in self._spacers:
            self.backend.win_deregister(self.comm, h)
            if free is not None:
                free(self.comm, loc)
        self._spacers.clear()
        self.backend.comm_destroy(self.comm)
        self.windows.clear()
