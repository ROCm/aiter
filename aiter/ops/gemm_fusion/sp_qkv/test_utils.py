# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Everything that exists only for tests, evidence runs and bare-metal probes of aiter.gemm_fusion.

NOT imported by any runtime module (aiter.gemm_fusion.*, aiter.ops.gemm_fusion.sp_qkv.{variants,variants_large,small_rt,large_rt,jit,specs}): the end-to-end path
(ATOM --gemm-fusion-profile -> match_prefix -> plan.prepare/run/recv_view/reuse_view/poison -> FusionResources) never needs it. Contents:

  * test doubles: FakeDist, FakeBackend, FirstFitBackend, RealisticFirstFit, ArenaBackend (a CCO backend with first-fit slots and window VIEWs)
  * profile helpers: make_profile, make_res, clear_profile_cache
  * the C-S4 aliasing assertion as FREE functions over a FusionResources: window_ranges, find_window_alias, assert_no_window_alias (+ tensor_extent, iter_tensors,
    WindowAliasError)
  * log_capture: collect aiter.gemm_fusion log records regardless of the real aiter logger's propagate=False
  * a bare-metal window probe (run / run_arena / sequence / main): torchrun --nproc_per_node=4 -m aiter.ops.gemm_fusion.sp_qkv.test_utils --sequence arena
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Callable, List, Optional, Tuple

MiB = 1 << 20


# ------------------------------------------------------------------------------------------------------------- test doubles
class FakeDist:
    """torch.distributed stand-in (single process). ``diverge_rank``: all_gather_object pretends that rank saw a different object."""

    def __init__(self, diverge_rank=None):
        self.calls = []
        self.diverge_rank = diverge_rank

    def broadcast_object_list(self, obj, src=0, group=None):
        self.calls.append(("bcast", src))

    def barrier(self, group=None):
        self.calls.append(("barrier",))

    def all_gather_object(self, out, obj, group=None):
        self.calls.append(("allgather",))
        for i in range(len(out)):
            out[i] = obj if i != self.diverge_rank else ("DIVERGED",) + tuple(obj[1:])


class FakeBackend:
    """The native CCO layer, recorded. Implements the whole backend surface FusionResources calls (uid, comm/window/devcomm lifecycle, zero, views)."""

    def __init__(self):
        self.log = []
        self.n = 0

    def uid(self, rank):
        return b"u" * 128

    def comm_create(self, rank, world, uid, vmm, device=None):
        self.log.append(("comm", vmm))
        return 1

    def win_register(self, comm, nbytes):
        self.n += 1
        self.log.append(("win", nbytes))
        return 100 + self.n, 4096 * self.n

    def peer_ptr(self, comm, local, peer):
        return local + peer

    def devcomm_create(self, comm, q):
        self.log.append(("dev", q))
        return 7

    def zero(self, dev, local, n):
        self.log.append(("zero", local, n))

    def device_sync(self, dev):
        self.log.append(("sync",))

    def devcomm_destroy(self, comm, dev):
        self.log.append(("devdestroy",))

    def win_deregister(self, comm, w):
        self.log.append(("dereg", w))

    def win_free(self, comm, local):
        self.log.append(("free", local))

    def comm_destroy(self, comm):
        self.log.append(("commdestroy",))


class FirstFitBackend(FakeBackend):
    """FakeBackend that mimics MORI's per-rank first-fit slot allocator: records offsets, fails when the heap is exhausted."""

    def __init__(self, heap):
        super().__init__()
        self.heap, self.next_off, self.offsets, self.sizes = heap, 0, [], []

    def win_register(self, comm, nbytes):
        from aiter.gemm_fusion.resources import ResourceError

        if self.next_off + nbytes > self.heap:
            raise ResourceError("gf_win_register rc=-3 (slot exhausted)")
        self.offsets.append(self.next_off)
        self.sizes.append(nbytes)
        self.next_off += nbytes
        return super().win_register(comm, nbytes)

    def comm_create(self, rank, world, uid, vmm, device=None):
        self.heap = vmm
        return super().comm_create(rank, world, uid, vmm, device)


class RealisticFirstFit(FakeBackend):
    """First-fit like MORI's per-rank slot allocator, with REAL-looking pointers: local = slot_base + offset (slot_base 4 GiB-aligned)."""

    def __init__(self, base=0x7F0000000000):
        super().__init__()
        self.base, self.top, self.heap, self.regs = base, 0, 1 << 62, []

    def comm_create(self, rank, world, uid, vmm, device=None):
        self.heap = vmm
        return super().comm_create(rank, world, uid, vmm, device)

    def win_register(self, comm, nbytes):
        from aiter.gemm_fusion.resources import ResourceError

        if self.top + nbytes > self.heap:
            raise ResourceError("slot exhausted")
        off, self.top = self.top, self.top + nbytes
        self.regs.append((off, nbytes))
        return 1000 + len(self.regs), self.base + off


class ArenaBackend(RealisticFirstFit):
    """RealisticFirstFit with window VIEW support: a view is (arena handle, offset); win_info returns what a device copy of ccoWindowDevice would hold."""

    def __init__(self):
        super().__init__(0x7F0000000000)
        self.views, self.freed = {}, []

    def win_register(self, comm, n):
        h, loc = super().win_register(comm, n)
        self.arena_base = {h: loc}
        return h, loc

    def win_view(self, comm, arena_win, offset):
        v = 0xA000 + len(self.views)
        self.views[v] = (arena_win, offset)
        return v

    def win_view_free(self, comm, v):
        self.freed.append(v)

    def win_info(self, comm, w):
        if w in self.views:
            a, off = self.views[w]
            return (self.arena_base[a] + off, 1, 0)
        return (self.arena_base[w], 1, 0)


# ------------------------------------------------------------------------------------------------------------- profile helpers
def clear_profile_cache() -> None:
    """Forget every profile load_profile() cached (tests that rewrite a profile file)."""
    from aiter.gemm_fusion import profile as profmod

    profmod._CACHE.clear()


def make_profile(window_bytes=None, max_windows=None, window_budget=None, wait_budget_us=None, name="test"):
    """A rows-less Profile carrying only a window_budget (and optionally wait_budget_us): enough for FusionResources."""
    from aiter.gemm_fusion.profile import parse_profile

    doc = {"profile": name, "rows": []}
    wb = dict(window_budget or {})
    if window_bytes is not None:
        wb["window_bytes"] = window_bytes
    if max_windows is not None:
        wb["max_windows"] = max_windows
    if wb:
        doc["window_budget"] = wb
    if wait_budget_us is not None:
        doc["wait_budget_us"] = wait_budget_us
    return parse_profile(doc)


def make_res(*, rank=0, world=4, dist=None, backend=None, policy=None, profile=None, window_bytes=None, max_windows=None, **kw):
    """FusionResources(...) for tests: ``window_bytes`` / ``max_windows`` become a synthetic window_budget (merged over ``profile`` when one is given)."""
    from aiter.gemm_fusion.profile import parse_profile, resolve_profile, window_budget_of
    from aiter.gemm_fusion.resources import FusionResources

    if window_bytes is not None or max_windows is not None:
        if profile is not None:
            prof = resolve_profile(profile)
            wb = dict(window_budget_of(prof))
            if window_bytes is not None:
                wb["window_bytes"] = window_bytes
            if max_windows is not None:
                wb["max_windows"] = max_windows
            profile = parse_profile({**prof.meta, "profile": prof.name, "window_budget": wb, "rows": []})
        else:
            profile = make_profile(window_bytes, max_windows)
    return FusionResources(rank, world, dist=dist if dist is not None else FakeDist(), backend=backend if backend is not None else RealisticFirstFit(),
                           policy=policy, profile=profile, **kw)


def log_capture(level=logging.INFO):
    """(messages, undo): collect aiter.gemm_fusion records (the real aiter logger has propagate=False, so caplog would see nothing)."""
    msgs = []

    class H(logging.Handler):
        def emit(self, rec):
            msgs.append(rec.getMessage())

    lg = logging.getLogger("aiter.gemm_fusion")
    h, old = H(level=level), lg.level
    lg.addHandler(h)
    lg.setLevel(level)
    return msgs, lambda: (lg.removeHandler(h), lg.setLevel(old))


# ------------------------------------------------------------------------------------------------------------- C-S4: aliasing assertion
ENV_DEBUG_ALIAS = "AITER_GEMM_FUSION_DEBUG_ALIAS"  # =1: assert_no_window_alias() actually checks (default off)


class WindowAliasError(RuntimeError):
    """A tensor handed out/stored by the integration points into a CCO window (S/R/C/arena...): it would be overwritten by a peer or by the next launch."""


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
    """Yield (path, tensor-like) from nested dict/list/tuple/object-attribute structures (tensors are recognised by data_ptr/shape/stride)."""
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


def window_ranges(res, kinds=("p2p_window", "control")) -> List[Tuple[str, int, int]]:
    """[(label, start, end)] of every window REGISTERED by ``res`` (under the arena policy: the one arena) in this process's address space."""
    return [(f"{w.owner}/{w.name}" + (" [shared]" if w.shared else ""), int(w.local), int(w.local) + int(w.bytes)) for w in res._regs if w.kind in kinds]


def find_window_alias(res, obj, kinds=("p2p_window", "control")) -> List[str]:
    """Descriptions of every overlap between a tensor reachable from ``obj`` and a window of ``res``. Empty = clean."""
    wins = window_ranges(res, kinds)
    hits = []
    for path, t in iter_tensors(obj):
        ext = tensor_extent(t)
        if ext is None:
            continue
        for label, a, b in wins:
            if ext[0] < b and a < ext[1]:
                hits.append(f"{path} [{ext[0]:#x},{ext[1]:#x}) overlaps window {label} [{a:#x},{b:#x})")
    return hits


def assert_no_window_alias(res, obj, what: str = "gemm_fusion output", force: bool = False) -> None:
    """The recv_view / reuse_view of a large plan aliases the shared R window (the SAME R for every large plan), valid only until the next large launch on this rank.
    Nothing RETURNED from the op or STORED on a long-lived object (e.g. attn.impl._index_q) may point into a window. No-op unless AITER_GEMM_FUSION_DEBUG_ALIAS=1 or
    force=True. Raises WindowAliasError."""
    if not (force or os.environ.get(ENV_DEBUG_ALIAS, "").strip().lower() in ("1", "true", "yes", "on")):
        return
    hits = find_window_alias(res, obj)
    if hits:
        raise WindowAliasError(f"{what} points into a CCO window (valid only until the next launch; copy it within the op): " + "; ".join(hits))


# ------------------------------------------------------------------------------------------------------------- bare-metal window probe
def sequence(name: str, mib: str = "", n: int = 39, window_mib: int = 164) -> List[int]:
    """Window sizes (bytes) of a named registration sequence for ``run`` (plain sequences; the arena has run_arena)."""
    if name == "phase1_equal":
        return [40 * MiB] * 3
    if name == "small_then_big":
        return [4 * MiB] * 12 + [40 * MiB]
    if name == "uniform":
        return [window_mib * MiB] * n
    if name == "gate03_prefix":
        return [4 * MiB] * 4 + [40 * MiB]
    if name == "custom":
        return [int(x) * MiB for x in mib.split(",") if x]
    raise SystemExit(f"unknown sequence {name!r} (phase1_equal, small_then_big, uniform, gate03_prefix, custom, arena)")


def _va_fields(va: int) -> str:
    return f"va={va:#x} va_mod_2MiB={va % (2 * MiB)} va_mod_256MiB={va % (256 * MiB)}"


def run(seq: List[int], heap_bytes: int, rank: int, world: int, dist, backend, tag: str, stagger_ms: int = 0, sleep=None, device=None) -> bool:
    """Register `seq` on a fresh comm (HIP device = rank unless ``device`` is given); returns True if all succeeded. Pure function of the injected dist/backend."""
    uid = [backend.uid(rank) if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    comm = backend.comm_create(rank, world, uid[0], heap_bytes, rank if device is None else device)
    cum, ok, wins = 0, True, []
    if sleep is None:
        import time

        sleep = time.sleep
    for i, size in enumerate(seq):
        if stagger_ms:
            sleep(rank * stagger_ms / 1000.0)
        try:
            w = backend.win_register(comm, size)
            wins.append(w)
            cum += size
            print(f"PROBE rank={rank} seq={tag} idx={i} size={size} cumulative={cum} offset_mib={(cum - size) // MiB} {_va_fields(int(w[1]))} result=OK", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"PROBE rank={rank} seq={tag} idx={i} size={size} cumulative={cum} offset_mib={cum // MiB} result=FAIL err={e}", flush=True)
            ok = False
            break
    for w in reversed(wins):
        try:
            backend.win_deregister(comm, w[0])
            backend.win_free(comm, w[1])
        except Exception:  # noqa: BLE001
            pass
    try:
        backend.comm_destroy(comm)
    except Exception:  # noqa: BLE001
        pass
    dist.barrier()
    return ok


def _arena_layout(wb: dict):
    """(arena bytes, [(size, role)]) of the arena policy's budget: the shared S/R/C set then each small owner's windows (4 MiB multiples)."""
    from aiter.gemm_fusion.resources import ARENA_ALIGN

    def up(x):
        return -(-int(x) // ARENA_ALIGN) * ARENA_ALIGN

    items = [(up(v), "shared") for v in sorted(wb["shared"].values(), reverse=True)] + [(up(s), "small") for s in wb["small_window_sizes"]]
    return sum(s for s, _ in items), items


def run_arena(heap_bytes: int, rank: int, world: int, dist, backend, tag: str, device=None, profile="minimax_m3_sp4_gfx950_v1") -> bool:
    """ONE registration (the arena: 380 MiB for the shipped profile) followed by the view structs the arena policy would create (4 MiB aligned). Checks that every view
    struct reads back winBase = arena winBase + offset with unchanged stride4G/lsaRank, then frees the views and deregisters the arena. A second registration is never issued. Not tested: kernels dereferencing a view on the device."""
    from aiter.gemm_fusion.profile import load_profile, window_budget_of

    size, items = _arena_layout(window_budget_of(load_profile(profile)))
    uid = [backend.uid(rank) if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    comm = backend.comm_create(rank, world, uid[0], heap_bytes, rank if device is None else device)  # HIP device = rank (the torchrun mapping), NOT 0
    ok, views, h, loc = True, [], None, 0
    try:
        h, loc = backend.win_register(comm, size)
        print(f"PROBE rank={rank} seq={tag} idx=0 size={size} cumulative={size} kind=arena {_va_fields(int(loc))} result=OK", flush=True)
        off = 0
        for i, (sz, role) in enumerate(items, 1):
            v = backend.win_view(comm, h, off)
            views.append(v)
            wbase, stride, lsa = backend.win_info(comm, v)
            abase, astride, alsa = backend.win_info(comm, h)
            good = wbase == abase + off and (stride, lsa) == (astride, alsa)
            print(f"PROBE rank={rank} seq={tag} idx={i} size={sz} offset={off} role={role} view_winBase_ok={int(good)} "
                  f"result={'OK' if good else 'FAIL'}", flush=True)
            ok &= good
            off += sz
        print(f"PROBE rank={rank} seq={tag} registrations=1 views={len(views)} arena_used={off} arena={size} result={'OK' if off <= size and ok else 'FAIL'}", flush=True)
        ok &= off <= size
    except Exception as e:  # noqa: BLE001
        print(f"PROBE rank={rank} seq={tag} result=FAIL err={e}", flush=True)
        ok = False
    for v in views:
        try:
            backend.win_view_free(comm, v)
        except Exception:  # noqa: BLE001
            pass
    try:
        if h is not None:
            backend.win_deregister(comm, h)
            backend.win_free(comm, loc)
        backend.comm_destroy(comm)
    except Exception:  # noqa: BLE001
        pass
    dist.barrier()
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description="bare-metal CCO window probe (torchrun --nproc_per_node=4 -m aiter.ops.gemm_fusion.sp_qkv.test_utils ...)")
    ap.add_argument("--sequence", required=True, help="arena | phase1_equal | small_then_big | uniform | gate03_prefix | custom")
    ap.add_argument("--heap-gib", type=int, default=4)
    ap.add_argument("--mib", default="")
    ap.add_argument("--n", type=int, default=39)
    ap.add_argument("--window-mib", type=int, default=164)
    ap.add_argument("--torch-gib", type=float, default=0.0)
    ap.add_argument("--init-dist-nccl", action="store_true")
    ap.add_argument("--stagger-ms", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args(argv)
    import torch
    import torch.distributed as dist

    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    from aiter.gemm_fusion.resources import CcoBackend

    hold = None
    if a.torch_gib > 0:  # real runs hold model weights / KV cache in the torch caching allocator before the comm exists
        hold = torch.empty(int(a.torch_gib * (1 << 30)), dtype=torch.uint8, device=f"cuda:{rank}")
        hold.fill_(1)
        torch.cuda.synchronize()
    if a.init_dist_nccl:
        dist.new_group(backend="nccl")
    ok = True
    for rep in range(a.repeat):
        tag = f"{a.sequence}#{rep}"
        if a.sequence == "arena":
            ok &= run_arena(a.heap_gib << 30, rank, world, dist, CcoBackend(), tag, device=rank)
        else:
            ok &= run(sequence(a.sequence, a.mib, a.n, a.window_mib), a.heap_gib << 30, rank, world, dist, CcoBackend(), tag, a.stagger_ms)
    del hold
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
