# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Measure this machine's peer read/write ceiling before tuning combine.

Combine reads peers at 232 GB/s; dispatch writes peers at 359 GB/s; upstream
MoonEP's combine reaches 650 GB/s on B300.  Those three numbers do not settle
whether our combine is slow or our link is, so this script asks the machine
directly with the simplest possible streaming access: contiguous, fully
coalesced, no scatter, no unpack, peers rotated by block so every link runs at
once.

What each row answers:

``read/remote``   the ceiling combine is chasing.  If combine's 232 GB/s is
                  already close to this, a deeper pipeline cannot help and the
                  only remaining levers are fewer bytes or a different
                  direction.
``write/remote``  the same for dispatch, and the control for whether remote
                  writes really do beat remote reads on xGMI (they are posted;
                  reads are round trips).  A large gap makes a push-based
                  combine -- destination ranks write results home, source ranks
                  reduce locally -- worth building.
``read/local``    harness sanity: the identical kernel against local HBM should
                  land in the TB/s range.
depth sweep       loads issued back to back before any is consumed.  Flat means
                  in-flight depth is not the limit, which is the whole question
                  behind a 16-stage pipeline.

Run under torchrun with one process per GPU::

    torchrun --standalone --nproc-per-node=8 \
        op_tests/benchmark_moonep_link_probe.py
"""

from __future__ import annotations

import json
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms
import torch
import torch.distributed as dist
from mori.shmem import mori_shmem_create_tensor, mori_shmem_free_tensor

from aiter.ops.flydsl.kernels.moonep_link_probe import (
    make_link_probe_jit,
    probe_stride_dwords,
)
from op_tests.test_moonep_gfx950_real_shape import _share_shmem_unique_id

WARMUP = int(os.environ.get("MOONEP_BENCH_WARMUP", "3"))
ITERS = int(os.environ.get("MOONEP_BENCH_ITERS", "10"))
# 128 MiB keeps every (block_num, depth) below divisible without a tail; see
# probe_stride_dwords.  Per iteration a rank moves SLAB * (world - 1) bytes,
# which at world=8 is ~940 MB, i.e. the same order as combine's 822 MB.
SLAB_MIB = int(os.environ.get("MOONEP_PROBE_SLAB_MIB", "128"))
DEPTHS = [int(x) for x in os.environ.get("MOONEP_PROBE_DEPTHS", "1,4,16").split(",")]
BLOCK_NUMS = [
    int(x) for x in os.environ.get("MOONEP_PROBE_BLOCKS", "256,1024,2048").split(",")
]
BLOCK_THREADS = int(os.environ.get("MOONEP_PROBE_THREADS", "256"))
DIRECTIONS = os.environ.get("MOONEP_PROBE_DIRS", "read,write").split(",")
# gfx940+ buffer aux bits: bit0=sc0, bit1=nt, bit4=sc1.  Peer traffic is
# streaming with no reuse, so the default policy may be paying for cache
# lookups and coherence it cannot use.  0 is what production uses today.
CACHE_MODS = [
    int(x) for x in os.environ.get("MOONEP_PROBE_CACHE_MODS", "0").split(",")
]
# The local-HBM sanity point costs a compile; skip it once it has been seen.
WITH_LOCAL = os.environ.get("MOONEP_PROBE_LOCAL", "1") == "1"


def time_gpu_op(launch_fn, group):
    """Same methodology as benchmark_moonep_dispatch_aligned: cross-rank mean.

    Eager only -- torch.cuda.graph does not capture the FlyDSL launch path.
    """

    for _ in range(WARMUP):
        launch_fn()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        launch_fn()
    end.record()
    end.synchronize()
    local_us = start.elapsed_time(end) / ITERS * 1e3

    world = dist.get_world_size(group=group)
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    t = torch.tensor([local_us], dtype=torch.float64, device=dev)
    outs = [torch.empty(1, dtype=torch.float64, device=dev) for _ in range(world)]
    dist.all_gather(outs, t, group=group)
    allr = torch.cat(outs)
    return allr.mean().item(), allr.max().item()


def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    status = ms.shmem_init_attr(
        ms.MORI_SHMEM_INIT_WITH_UNIQUEID,
        rank,
        world_size,
        _share_shmem_unique_id(rank),
    )
    assert status == 0

    slab_dwords = SLAB_MIB * 1024 * 1024 // 4
    slab = mori_shmem_create_tensor((slab_dwords,), torch.int32)
    # Non-trivial contents: a constant pattern would be a gift to any
    # compression on the path.
    gen = torch.Generator(device="cpu").manual_seed(20260810 + rank)
    slab.copy_(
        torch.randint(-(2**31), 2**31 - 1, (slab_dwords,), generator=gen, dtype=torch.int32)
    )
    torch.cuda.synchronize(device)
    ms.shmem_barrier_all()

    remote = [p for p in range(world_size) if p != rank]
    remote_ptrs = torch.tensor(
        [ms.shmem_ptr_p2p(slab.data_ptr(), rank, p) for p in remote],
        dtype=torch.int64,
        device=device,
    )
    # Peer-count sweep.  The headline 235 GB/s is the *aggregate* over all 7
    # peers; an op that only touches one peer (a single prefetched expert, say)
    # is bounded by one link, not by the aggregate.  Rotating the start by rank
    # keeps the pairing symmetric so no link is doubly loaded.
    peer_subsets = {}
    for n in [int(x) for x in os.environ.get("MOONEP_PROBE_NPEERS", "").split(",") if x]:
        sub = [remote[(rank + i) % len(remote)] for i in range(n)]
        peer_subsets[n] = torch.tensor(
            [ms.shmem_ptr_p2p(slab.data_ptr(), rank, p) for p in sub],
            dtype=torch.int64,
            device=device,
        )
    local_ptrs = torch.tensor(
        [ms.shmem_ptr_p2p(slab.data_ptr(), rank, rank)],
        dtype=torch.int64,
        device=device,
    )
    sink = torch.zeros(max(BLOCK_NUMS), dtype=torch.int32, device=device)
    stream = torch.cuda.current_stream(device)

    def run_case(*, direction, ptrs, num_peers, depth, block_num, cache_mod):
        stride = probe_stride_dwords(
            block_num=block_num, block_threads=BLOCK_THREADS, depth=depth
        )
        if slab_dwords % stride != 0:
            return None
        jit = make_link_probe_jit(
            covered_dwords=slab_dwords,
            num_peers=num_peers,
            depth=depth,
            block_num=block_num,
            block_threads=BLOCK_THREADS,
            direction=direction,
            cache_modifier=cache_mod,
        )
        args = (fx.Int64(ptrs.data_ptr()), fx.Int64(sink.data_ptr()), stream)
        compiled = flyc.compile(jit, *args)
        raw = (ptrs.data_ptr(), sink.data_ptr(), stream)

        def run():
            compiled(*raw)

        mean_us, max_us = time_gpu_op(run, dist.group.WORLD)
        torch.cuda.synchronize(device)
        dist.barrier()
        return mean_us, max_us

    slab_bytes = slab_dwords * 4
    results = {}
    cases = []
    for direction in DIRECTIONS:
        for block_num in BLOCK_NUMS:
            for depth in DEPTHS:
                for cache_mod in CACHE_MODS:
                    cases.append(
                        (
                            f"{direction}/remote",
                            direction,
                            remote_ptrs,
                            len(remote),
                            depth,
                            block_num,
                            cache_mod,
                        )
                    )
    for n, ptrs in sorted(peer_subsets.items()):
        for direction in DIRECTIONS:
            cases.append(
                (f"{direction}/peers{n}", direction, ptrs, n, 4, 1024, 0)
            )
    # Harness sanity: identical kernel, local HBM.  One point is enough.
    if WITH_LOCAL:
        cases.append(("read/local", "read", local_ptrs, 1, 4, 1024, 0))

    for label, direction, ptrs, num_peers, depth, block_num, cache_mod in cases:
        out = run_case(
            direction=direction,
            ptrs=ptrs,
            num_peers=num_peers,
            depth=depth,
            block_num=block_num,
            cache_mod=cache_mod,
        )
        if out is None:
            if rank == 0:
                print(f"[skip] {label} d{depth} bn{block_num}: slab not divisible")
            continue
        mean_us, max_us = out
        moved = slab_bytes * num_peers
        results[f"{label}/d{depth}/bn{block_num}/cm{cache_mod}"] = {
            "mean_us": mean_us,
            "max_us": max_us,
            "bytes": moved,
            "gbps": moved / mean_us / 1e3,
        }
        if rank == 0:
            print(
                f"[done] {label:<14} depth={depth:<3} bn={block_num:<5} "
                f"cm={cache_mod:<3} {mean_us:>9.1f} us  "
                f"{moved / mean_us / 1e3:>7.1f} GB/s",
                flush=True,
            )

    if rank == 0:
        print()
        print(f"slab={SLAB_MIB} MiB/peer  world={world_size}  threads={BLOCK_THREADS}")
        print(f"warmup={WARMUP} iters={ITERS} (eager, cross-rank mean)")
        print()
        print(f"{'case':<34}{'mean us':>10}{'max us':>10}{'GB/s':>10}")
        for key in sorted(results):
            r = results[key]
            print(
                f"{key:<34}{r['mean_us']:>10.1f}{r['max_us']:>10.1f}{r['gbps']:>10.1f}"
            )
        print()
        print("for reference, same machine, same methodology:")
        print("  dispatch remote writes (real plan)      359.1 GB/s")
        print("  dispatch remote writes (ideal addr)     383.7 GB/s")
        print("  combine  remote reads  (all K rows)     232.8 GB/s")
        print("  combine  remote reads  (dedup'd)        233.5 GB/s")
        print()
        print("MOONEP_LINK_PROBE_JSON " + json.dumps(results))

    dist.barrier()
    ms.shmem_barrier_all()
    mori_shmem_free_tensor(slab)
    ms.shmem_finalize()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
