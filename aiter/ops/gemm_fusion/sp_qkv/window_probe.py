# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Window-registration probe for the shared CCO comm (read-only triage tool; no kernels, no ATOM, no MORI edits).

Run it with the SAME launcher/wrapper env as the failing run (MORI_ENABLE_SDMA=1 CCO_GDR_CAPABLE=0 ...), one process per rank:
  python -m aiter.ops.gemm_fusion.sp_qkv.window_probe --sequence NAME [--heap-gib N] [--mib 4,40]   (torchrun env:// style)
Sequences (each creates a FRESH comm, registers windows in order, prints per-rank PASS / first failure with index, size, offset):
  phase1_equal   : 3 x 40 MiB                     (the accepted Phase-1 large-M layout; must PASS)
  small_then_big : 12 x 4 MiB then 40 MiB         (the gate02 failing layout)
  uniform        : N x <window MiB>               (the uniform policy; default 39 x 164 MiB)
  descending     : big windows first, then 4 MiB  (alignment hypothesis)
  custom         : --mib a,b,c,... explicit list
  gate03_prefix  : 4,4,4,4,40 MiB (the real per_owner failing prefix: window #5 = 40 MiB at offset 16 MiB; same as --sequence custom --mib 4,4,4,4,40)
  aligned_gate03 : what the 'aligned' policy registers for that prefix: 4,4,4,4,[48 MiB spacer],40 -> the 40 MiB window starts at 64 MiB
  aligned_all    : the 'aligned' layout of the real 15-plan sequence (per_owner sizes, all windows, spacers included; heap needs ~ --heap-gib 16)
  largest_first  : the same windows sorted by descending size, no padding (the other layout hypothesis)
Options that reproduce the real process state: --torch-gib N (allocate N GiB with torch on this rank BEFORE the comm is created and keep it
alive), --init-dist-nccl (also create an NCCL group), --stagger-ms K (rank r sleeps r*K ms before each registration: tests the
concurrent-registration hypothesis; the default 0 = lockstep like the real run), --repeat R (R fresh comms in a row).
ATOM-like mode (reproduces the E2 'HIP operation not supported' on torch.zeros_like right after FusionResources init, outside ATOM):
  python -m aiter.ops.gemm_fusion.sp_qkv.window_probe --atom-like [--policy uniform|per_owner|aligned] [--torch-first-gib N] [--large-empty-gib G]
  Order: torch + process group first (a torch tensor is allocated BEFORE the comm), then FusionResources built EXACTLY like ATOM's
  FusionResourcesWrapper (rank = group rank, device = torch.cuda.current_device(), caps = detect_capabilities), then ensure() of the real 15-plan
  windows, finalize(), and AFTER each stage the E2 trigger sequence: a pinned-host -> device copy, zeros_like on a device tensor, a large torch.empty,
  hipGetLastError via torch.cuda.synchronize(). Lines: ATOMLIKE rank= stage= device_before= device_after= result=OK|FAIL err=...
  Compare --policy runs and `AITER_GEMM_FUSION_WINDOW_POLICY` to separate (1) current-device change, (2) heap/window mappings, (3) a sticky error.
Absolute-VA sequences (D14 r1c, H-BASE): MORI reserves flatBase only G(4 KiB)-aligned, so a window's ABSOLUTE alignment is process dependent; these run the SAME
layout engine the abs_aligned / shared_large policies use (aiter.gemm_fusion.resources.AbsLayout), driven by the measured absolute VA of each registration:
  abs_aligned_gate03: 4,4,4,4,40 MiB with every window start aligned to A(size)=max(2 MiB, min(next_pow2(size), 256 MiB)) (the 40 MiB window at a 64 MiB boundary)
  abs_aligned_all   : the 39 per-owner windows of the real 15-plan sequence, LARGEST FIRST, plain spacers (policy abs_aligned)
  shared_large_abs  : the Architect's sequence: S(164) at a 256 MiB boundary, control + 12 small windows, a 40 MiB spacer, R(164) at the next 256 MiB boundary
  gate03_prefix     : the plain failing 4,4,4,4,40 (kept; every PROBE line now carries the absolute VA)
  arena / arena_per_owner : D14 r1e/r1f: ONE registration (380 MiB / 2508 MiB) + the view structs the arena policy creates; checks each view's winBase and peer pointers
Output lines: PROBE rank=R seq=S idx=I size=B cumulative=C [kind=spacer] va=0x.. va_mod_2MiB= va_mod_16MiB= va_mod_64MiB= va_mod_256MiB= result=OK|FAIL
(abs sequences also print: PROBE_BASE slot_base=0x.. base_mod_2MiB= base_mod_256MiB=)
Comparing sequences from ONE run isolates the failing condition without guessing (size vs. offset alignment vs. count vs. heap).
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from aiter.gemm_fusion.resources import AbsLayout, aligned_plan

MiB = 1 << 20


def real_sequence_sizes() -> List[int]:
    """The per_owner window sizes of the 15-plan gate sequence in registration order: small owners (2 x 4 MiB) first, then large owners
    (S, R, C of equal size per owner). Taken from the generated profile's window_budget when available (sorted desc there), else rebuilt."""
    from aiter.ops.gemm_fusion.sp_qkv import specs, variants, variants_large
    from aiter.gemm_fusion.resources import round_up

    out = []
    for (mode, T), (_v, params) in specs.SMALL_CELLS.items():
        b = variants.small_window_bytes(mode)
        out += [round_up(max(b["rx"], b["ctl"], 8192))] * 2
    for (mode, T), _ in specs.LARGE_CELLS.items():
        out += [variants_large.large_window_bytes(mode, T // 4)] * 3
    return out


ABS_SEQUENCES = ("abs_aligned_gate03", "abs_aligned_all", "shared_large_abs")
ARENA_SEQUENCES = ("arena", "arena_per_owner")  # D14 r1e/r1f: ONE registered window, then device-memory VIEW structs (no second registration of any size)


def run_arena(kind: str, heap_bytes: int, rank: int, world: int, dist, backend, tag: str, device=None) -> bool:
    """ONE registration (the arena, sized like the policy: 380 MiB shared design / 2508 MiB per-owner variant) followed by the view structs the policy would create
    (shared S/R/C + the small windows / every per-owner window, 4 MiB aligned). Checks that every view struct reads back winBase = arena winBase + offset with unchanged
    stride/lsaRank, that peer pointers of a view equal the arena's peer pointers + offset, then frees the views and deregisters the arena. A second registration is never issued."""
    from aiter.gemm_fusion.resources import ARENA_ALIGN, policy_memory_table
    import json, os as _os
    from aiter.gemm_fusion.profile import load_profile, window_budget_of

    wb = window_budget_of(load_profile(_os.environ.get("PROBE_PROFILE", "minimax_m3_sp4_gfx950_v1")))
    size = policy_memory_table(wb)[kind]
    if kind == "arena":
        offs = [(s, "shared") for s in sorted((int(v) for v in wb["shared"].values()), reverse=True)] + [(int(s), "small") for s in wb["small_window_sizes"]]
    else:
        offs = [(int(s), "owner") for s in wb["window_sizes"]]
    uid = [backend.uid(rank) if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    comm = backend.comm_create(rank, world, uid[0], heap_bytes, rank if device is None else device)  # HIP device = rank (the torchrun mapping), as run() does; NOT 0
    ok, views = True, []
    try:
        h, loc = backend.win_register(comm, size)
        print(f"PROBE rank={rank} seq={tag} idx=0 size={size} cumulative={size} kind=arena {_va_fields(int(loc))} result=OK", flush=True)
        off = 0
        for i, (sz, role) in enumerate(offs, 1):
            sz = -(-sz // ARENA_ALIGN) * ARENA_ALIGN
            v = backend.win_view(comm, h, off)
            views.append(v)
            wbase, stride, lsa = backend.win_info(comm, v)
            abase, astride, alsa = backend.win_info(comm, h)
            good = wbase == abase + off and (stride, lsa) == (astride, alsa)
            peers_ok = all(backend.peer_ptr(comm, int(loc) + off, p) == backend.peer_ptr(comm, int(loc), p) + off for p in range(world))
            print(f"PROBE rank={rank} seq={tag} idx={i} size={sz} offset={off} role={role} view_winBase_ok={int(good)} peer_ptrs_ok={int(peers_ok)} result={'OK' if good and peers_ok else 'FAIL'}", flush=True)
            ok &= good and peers_ok
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
        backend.win_deregister(comm, h)
        _free_local(backend, comm, loc)  # MORI teardown contract: WindowDeregister -> MemFree(local) before ccoCommDestroy (PROBE_FREE=0 = old behaviour)
        backend.comm_destroy(comm)
    except Exception:  # noqa: BLE001
        pass
    dist.barrier()
    return ok


def abs_items(name: str):
    """(size, role) items + fill_gaps for an absolute-VA sequence; the order of registration is decided by AbsLayout from the measured VA."""
    if name == "abs_aligned_gate03":
        return [(4 * MiB, None)] * 4 + [(40 * MiB, None)], False
    if name == "abs_aligned_all":
        return [(s, None) for s in real_sequence_sizes()], False
    if name == "shared_large_abs":
        from aiter.ops.gemm_fusion.sp_qkv import specs, variants, variants_large
        from aiter.gemm_fusion.resources import round_up

        small = [s for s in real_sequence_sizes() if s == 4 * MiB]
        big = max(variants_large.large_window_bytes(m, T // 4) for (m, T) in specs.LARGE_CELLS)
        return [(big, ("shared", "S")), (big, ("shared", "R")), (round_up(8192), ("shared", "C"))] + [(s, None) for s in small], True
    raise SystemExit(f"{name!r} is not an absolute-VA sequence {ABS_SEQUENCES}")


def sequence(name: str, mib: str = "", n: int = 39, window_mib: int = 164) -> List[int]:
    if name in ARENA_SEQUENCES:
        from aiter.gemm_fusion.resources import policy_memory_table
        from aiter.gemm_fusion.profile import load_profile, window_budget_of

        return [policy_memory_table(window_budget_of(load_profile("minimax_m3_sp4_gfx950_v1")))[name]]
    if name in ABS_SEQUENCES:  # the SIZES (in the order of the plan); the actual registration order/spacers come from AbsLayout at run time
        return [s for s, _r in abs_items(name)[0]]
    if name == "phase1_equal":
        return [40 * MiB] * 3
    if name == "small_then_big":
        return [4 * MiB] * 12 + [40 * MiB]
    if name == "uniform":
        return [window_mib * MiB] * n
    if name == "descending":
        return [164 * MiB, 160 * MiB, 80 * MiB, 40 * MiB] + [4 * MiB] * 12
    if name == "aligned_gate03":
        return [s for _k, s, _o in aligned_plan([4 * MiB] * 4 + [40 * MiB])]
    if name in ("aligned_all", "largest_first"):
        sizes = real_sequence_sizes()
        if name == "largest_first":
            return sorted(sizes, reverse=True)
        return [s for _k, s, _o in aligned_plan(sizes)]
    if name == "gate03_prefix":
        return [4 * MiB] * 4 + [40 * MiB]
    if name == "custom":
        return [int(x) * MiB for x in mib.split(",") if x]
    raise SystemExit(f"unknown sequence {name!r}")


def _free_local(backend, comm, local) -> None:
    """ccoMemFree of a window's local allocation after its Deregister (the documented teardown). Backends without win_free, or PROBE_FREE=0, skip it."""
    import os as _o

    fn = getattr(backend, "win_free", None)
    if fn is not None and _o.environ.get("PROBE_FREE", "1") != "0":
        fn(comm, local)


def _va_fields(va: int) -> str:
    return (f"va={va:#x} va_mod_2MiB={va % (2 * MiB)} va_mod_16MiB={va % (16 * MiB)} va_mod_64MiB={va % (64 * MiB)} va_mod_256MiB={va % (256 * MiB)}")


def run(seq: List[int], heap_bytes: int, rank: int, world: int, dist, backend, tag: str, stagger_ms: int = 0, sleep=None, device=None,
        abs_layout=None) -> bool:
    """Register `seq` on a fresh comm; returns True if all succeeded. Pure function of the injected dist/backend (unit-tested)."""
    uid = [backend.uid(rank) if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    comm = backend.comm_create(rank, world, uid[0], heap_bytes, rank if device is None else device)
    cum, ok = 0, True
    wins = []
    if sleep is None:
        import time

        sleep = time.sleep
    va_top = None
    if abs_layout is not None:  # absolute-VA sequence: the layout engine decides each registration (window or spacer) from the measured VA
        fn = getattr(backend, "va_info", None)
        if fn is not None:
            base, _pr, gran = fn(comm)
            va_top = int(base)
            abs_layout.gran = int(gran) or 4096
            print(f"PROBE_BASE rank={rank} seq={tag} slot_base={va_top:#x} granule={abs_layout.gran} base_mod_2MiB={va_top % (2 * MiB)} base_mod_256MiB={va_top % (256 * MiB)}", flush=True)

    def steps():
        if abs_layout is None:
            for size in seq:
                yield "window", size
            return
        while True:
            act = abs_layout.next(va_top_box[0]) if va_top_box[0] is not None else None
            if act is None:
                if va_top_box[0] is None and abs_layout.pending:  # no va_info (fake/old backend): place the largest first, align from its measured VA on
                    s, _r = abs_layout.pending.pop(0)
                    yield "window", s
                    continue
                return
            yield ("spacer", act[1]) if act[0] == "spacer" else ("window", act[1])

    va_top_box = [va_top]
    for i, (kind, size) in enumerate(steps()):
        if stagger_ms:
            sleep(rank * stagger_ms / 1000.0)
        try:
            w = backend.win_register(comm, size)
            wins.append(w)
            cum += size
            va = int(w[1])
            if abs_layout is not None:
                va_top_box[0] = va + (-(-size // abs_layout.gran) * abs_layout.gran)
            print(f"PROBE rank={rank} seq={tag} idx={i} size={size} cumulative={cum} offset_mib={(cum - size) // MiB}"
                  + (" kind=spacer" if kind == "spacer" else "") + f" {_va_fields(va)} result=OK", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"PROBE rank={rank} seq={tag} idx={i} size={size} cumulative={cum} offset_mib={cum // MiB}"
                  + (" kind=spacer" if kind == "spacer" else "") + f" result=FAIL err={e}", flush=True)
            ok = False
            break
    for w in reversed(wins):
        try:
            backend.win_deregister(comm, w[0])
            _free_local(backend, comm, w[1])
        except Exception:  # noqa: BLE001
            pass
    try:
        backend.comm_destroy(comm)
    except Exception:  # noqa: BLE001
        pass
    dist.barrier()
    return ok


def e2_trigger(torch, device_index: int, large_gib: float = 0.5):
    """The sequence that failed in ATOM after FusionResources init: pinned->device copy, zeros_like, a large empty, then a sync that surfaces
    any pending/sticky HIP error. Returns None on success or the exception text."""
    try:
        host = torch.empty(1 << 20, dtype=torch.float32, pin_memory=True)
        host.fill_(1.0)
        dev = host.to(f"cuda:{device_index}", non_blocking=True)
        z = torch.zeros_like(dev)
        big = torch.empty(int(large_gib * (1 << 30)), dtype=torch.uint8, device=f"cuda:{device_index}")
        big.fill_(0)
        torch.cuda.synchronize()
        assert float(z.sum().item()) == 0.0 and float(dev[:4].sum().item()) == 4.0
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def atom_like(a, torch, dist, rank: int, world: int) -> bool:
    """Build FusionResources exactly as ATOM does (golden path: torch first, then configure) and run the E2 trigger after every stage."""
    from aiter.gemm_fusion import capabilities
    from aiter.gemm_fusion.resources import FusionResources

    dev0 = torch.cuda.current_device()
    held = torch.empty(int(a.torch_first_gib * (1 << 30)), dtype=torch.uint8, device=f"cuda:{dev0}") if a.torch_first_gib > 0 else None
    ok = True

    def stage(name, fn=None):
        nonlocal ok
        before = torch.cuda.current_device()
        err = None
        try:
            if fn:
                fn()
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        after = torch.cuda.current_device()
        trig = e2_trigger(torch, after, a.large_empty_gib) if err is None else None
        bad = err or trig or (after != before and f"current device changed {before}->{after}")
        ok &= not bad
        print(f"ATOMLIKE rank={rank} stage={name} device_before={before} device_after={after} result={'FAIL' if bad else 'OK'}"
              + (f" err={bad}" if bad else ""), flush=True)

    stage("torch_only")
    box = {}
    caps = capabilities.detect_capabilities(None)
    print(f"ATOMLIKE rank={rank} caps={sorted(caps)} device={dev0} policy={a.policy or 'default'}", flush=True)

    def build():
        # profile= is passed EXPLICITLY (as ATOM's wrapper now does from config.gemm_fusion.profile): the window budget never comes from the env
        box["res"] = FusionResources(rank=rank, world=world, group=None, device=dev0, caps=caps, policy=a.policy or None, profile=a.profile)

    stage("FusionResources_init", build)
    res = box.get("res")
    if res is None:
        return False
    from aiter.ops.gemm_fusion.sp_qkv import specs, variants, variants_large

    def ensure_all():
        for (mode, T), (_vid, params) in specs.SMALL_CELLS.items():
            plan = variants.SpQkvSmall(specs.sp_qkv_spec(mode), params, _ctx(rank, world, T // 4, caps))
            res.ensure(plan.build_key, plan.resources())
        for (mode, T), (_vid, params) in specs.LARGE_CELLS.items():
            plan = variants_large.SpQkvLarge(specs.sp_qkv_spec(mode).prefix(1), params, _ctx(rank, world, T // 4, caps))
            res.ensure(plan.build_key, plan.resources())

    stage("ensure_all_windows", ensure_all)
    stage("finalize", res.finalize)
    stage("quiesce_and_reset", res.quiesce_and_reset)
    stage("release", res.release)
    del held
    return ok


def _ctx(rank, world, m, caps):
    from aiter.gemm_fusion.match import MatchCtx

    return MatchCtx("gfx950", 256, world, m, frozenset(caps) | {"gfx950_sc_bypass"})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default="")
    ap.add_argument("--atom-like", action="store_true")
    ap.add_argument("--policy", default="")
    ap.add_argument("--profile", default="minimax_m3_sp4_gfx950_v1", help="profile name/path given to FusionResources(profile=...) in --atom-like mode")
    ap.add_argument("--torch-first-gib", type=float, default=1.0)
    ap.add_argument("--large-empty-gib", type=float, default=0.5)
    ap.add_argument("--heap-gib", type=int, default=4)
    ap.add_argument("--mib", default="")
    ap.add_argument("--n", type=int, default=39)
    ap.add_argument("--window-mib", type=int, default=164)
    ap.add_argument("--torch-gib", type=float, default=0.0)
    ap.add_argument("--init-dist-nccl", action="store_true")
    ap.add_argument("--stagger-ms", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args(argv)
    import os

    import torch
    import torch.distributed as dist

    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    from aiter.gemm_fusion.resources import CcoBackend

    if a.atom_like:
        sys.exit(0 if atom_like(a, torch, dist, rank, world) else 3)
    if not a.sequence:
        raise SystemExit("--sequence or --atom-like is required")
    hold = None
    if a.torch_gib > 0:  # real runs hold model weights / KV cache in the torch caching allocator before the comm exists
        hold = torch.empty(int(a.torch_gib * (1 << 30)), dtype=torch.uint8, device=f"cuda:{rank}")
        hold.fill_(1)
        torch.cuda.synchronize()
    if a.init_dist_nccl:
        dist.new_group(backend="nccl")
    ok = True
    for rep in range(a.repeat):
        if a.sequence in ARENA_SEQUENCES:
            ok &= run_arena(a.sequence, a.heap_gib << 30, rank, world, dist, CcoBackend(), f"{a.sequence}#{rep}", device=rank)
            continue
        lay = None
        if a.sequence in ABS_SEQUENCES:
            items, fill = abs_items(a.sequence)
            lay = AbsLayout(items, fill_gaps=fill)
        ok &= run(sequence(a.sequence, a.mib, a.n, a.window_mib), a.heap_gib << 30, rank, world, dist, CcoBackend(), f"{a.sequence}#{rep}", a.stagger_ms,
                   abs_layout=lay)
    del hold
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()