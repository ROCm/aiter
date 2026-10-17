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
Output lines: PROBE rank=R seq=S idx=I size=B cumulative=C result=OK|FAIL
Comparing sequences from ONE run isolates the failing condition without guessing (size vs. offset alignment vs. count vs. heap).
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from aiter.gemm_fusion.resources import aligned_plan

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


def sequence(name: str, mib: str = "", n: int = 39, window_mib: int = 164) -> List[int]:
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


def run(seq: List[int], heap_bytes: int, rank: int, world: int, dist, backend, tag: str, stagger_ms: int = 0, sleep=None, device=None) -> bool:
    """Register `seq` on a fresh comm; returns True if all succeeded. Pure function of the injected dist/backend (unit-tested)."""
    uid = [backend.uid(rank) if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    comm = backend.comm_create(rank, world, uid[0], heap_bytes, rank if device is None else device)
    cum, ok = 0, True
    wins = []
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
            print(f"PROBE rank={rank} seq={tag} idx={i} size={size} cumulative={cum} offset_mib={(cum - size) // MiB} result=OK", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"PROBE rank={rank} seq={tag} idx={i} size={size} cumulative={cum} offset_mib={cum // MiB} result=FAIL err={e}", flush=True)
            ok = False
            break
    for w in reversed(wins):
        try:
            backend.win_deregister(comm, w[0])
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
        ok &= run(sequence(a.sequence, a.mib, a.n, a.window_mib), a.heap_gib << 30, rank, world, dist, CcoBackend(), f"{a.sequence}#{rep}", a.stagger_ms)
    del hold
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()