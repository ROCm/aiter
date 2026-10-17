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
Output lines: PROBE rank=R seq=S idx=I size=B cumulative=C result=OK|FAIL
Comparing sequences from ONE run isolates the failing condition without guessing (size vs. offset alignment vs. count vs. heap).
"""

from __future__ import annotations

import argparse
import sys
from typing import List

MiB = 1 << 20


def sequence(name: str, mib: str = "", n: int = 39, window_mib: int = 164) -> List[int]:
    if name == "phase1_equal":
        return [40 * MiB] * 3
    if name == "small_then_big":
        return [4 * MiB] * 12 + [40 * MiB]
    if name == "uniform":
        return [window_mib * MiB] * n
    if name == "descending":
        return [164 * MiB, 160 * MiB, 80 * MiB, 40 * MiB] + [4 * MiB] * 12
    if name == "custom":
        return [int(x) * MiB for x in mib.split(",") if x]
    raise SystemExit(f"unknown sequence {name!r}")


def run(seq: List[int], heap_bytes: int, rank: int, world: int, dist, backend, tag: str) -> bool:
    """Register `seq` on a fresh comm; returns True if all succeeded. Pure function of the injected dist/backend (unit-tested)."""
    uid = [backend.uid(rank) if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    comm = backend.comm_create(rank, world, uid[0], heap_bytes)
    cum, ok = 0, True
    wins = []
    for i, size in enumerate(seq):
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--heap-gib", type=int, default=4)
    ap.add_argument("--mib", default="")
    ap.add_argument("--n", type=int, default=39)
    ap.add_argument("--window-mib", type=int, default=164)
    a = ap.parse_args(argv)
    import os

    import torch
    import torch.distributed as dist

    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    from aiter.gemm_fusion.resources import CcoBackend

    ok = run(sequence(a.sequence, a.mib, a.n, a.window_mib), a.heap_gib << 30, rank, world, dist, CcoBackend(), a.sequence)
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()