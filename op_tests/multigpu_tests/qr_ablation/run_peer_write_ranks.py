# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multi-rank FlyDSL peer-write primitive: the four-writer control for section 14.

WHY
===
Section 14.3 of ``op_tests/dump_data/docs/mi350p_peer_write_primitive_2026-09-07.md``
produced the investigation's first positive discriminator. Both kernels on
Agent 3, same cadence, same geometry, same request count:

    qr_int4 nowait        529,701 req/disp   11,517 cy in flight   230M credit stalls
    peer_write primitive  545,295 req/disp      242 cy in flight     0 credit stalls

Section 14.4 named the confound that keeps that from being conclusive: **qr_int4
has all four GPUs writing to each other at once; the primitive has one GPU
writing to three.** Four-way incast is the obvious candidate for exhausting the
IO credit pool, and it was not excluded, because the C++ primitive's four-writer
mode reaches the GPUs through ``fork()`` and rocprofv3 does not follow a fork --
that run produced no counter rows at all.

This is the fix section 14.4 asked for: the same FlyDSL kernel, driven from
``multiprocessing`` with the spawn start method, as ``run_ablation.py`` does.
rocprofv3 follows spawned children and writes one CSV per agent, so the four-
writer counters can finally be read on Agent 3 and put next to the row above.

    if 4 writers show ~11,000 cy and credit exhaustion -> the cause is incast,
                                                          and qr_int4 is innocent
    if 4 writers stay near 242 cy                      -> the cause is in the
                                                          kernel, and 14.3 localises it

THE CONTROL IS IN THE SAME HARNESS
==================================
``--writers 1`` makes rank 0 the only writer and the other ranks pure
destinations, which is byte-for-byte the single-writer primitive section 14.3
measured. ``--writers 4`` turns the same allocation, the same kernel and the
same geometry into four-way incast. One code path, one process layout, one
allocation -- the writer count is the only thing that moves. That is a stronger
A/B than comparing this file against ``test_flydsl_peer_write_bw.py --one``,
because it cannot pick up an incidental difference between two drivers.

THE WIRE LAYOUT
===============
Every rank allocates ``world_size`` slots and shares the whole buffer over
``hipIpcGetMemHandle``, so writer ``r`` owns slot ``r`` in every destination and
no two writers touch the same bytes. The offset is folded into the peer-pointer
table rather than passed to the kernel, so the kernel compiled here is the same
object as the one the single-writer runs use -- ``npeers`` and ``publish_bytes``
are the only things baked in, exactly as before.

This reaches the destinations through ``hipIpcOpenMemHandle`` across processes,
which is what ``QRInt4`` does; ``test_flydsl_peer_write_bw.py`` uses same-process
``hipDeviceEnablePeerAccess``. The C++ measured the two mappings as identical to
0.2%, so this is not expected to move the number -- but it does mean the
comparison against ``qr_int4`` now matches on the mapping too.

USAGE
=====
    # the A/B, wall clock only
    HIP_VISIBLE_DEVICES=0,1,2,3 python3 run_peer_write_ranks.py --writers 1
    HIP_VISIBLE_DEVICES=0,1,2,3 python3 run_peer_write_ranks.py --writers 4
    HIP_VISIBLE_DEVICES=0,1,2,3 python3 run_peer_write_ranks.py --sweep

    # under counters, via pmc_compare.py's prim1/prim4 workloads. Note the
    # device order: GPU 0's TCC EA and GRBM counters under-report on this host
    # (section 14.2), so the writer of interest has to land on Agent 3.
    HIP_VISIBLE_DEVICES=1,2,3,0 python3 run_peer_write_ranks.py --writers 4 \
        --iters 3 --warmup 2

Geometry defaults to section 14's: 224 blocks, 512 B runs, 55296 B publish,
fine-grained inbox, 32 MiB per writer per measured iteration.
"""

import argparse
import ctypes
import os
import sys
from multiprocessing import Pool, set_start_method

REPO = "/home/vpietila/git/aiter"

# Section 14 geometry. 224 blocks is qr_int4's grid at M=4096 with ST=8, and
# 55296 B is that kernel's publish cadence; matching both is what makes the
# counter rows comparable.
DEFAULT_BLOCKS = 224
DEFAULT_CHUNK = 512
DEFAULT_PUBLISH = 55296
DEFAULT_ITERS = 30
DEFAULT_WARMUP = 5


def _worker(world, rank, writers, init_method, cfg):
    """One rank.

    Returns ``(rank, gbps, total_bytes)``; a non-writer returns ``gbps`` None.
    ``total_bytes`` is carried back rather than read from the module in the
    parent, so the parent never imports flydsl or touches HIP.
    """
    import torch
    import torch.distributed as dist

    # Imported inside the worker so the parent process stays HIP-free until the
    # children have set their devices -- same reason run_ablation.py does it.
    sys.path.insert(0, REPO)
    sys.path.insert(0, f"{REPO}/op_tests/flydsl_tests")
    import test_flydsl_peer_write_bw as pw

    # `torch.cuda.set_device` alone is not enough. It is lazy: with no CUDA work
    # yet done in this fresh spawned process it records the ordinal without
    # creating a context, and the raw `hipExtMallocWithFlags` below -- which
    # goes through ctypes, not torch -- then lands on device 0 for *every* rank.
    # Nothing errors. Every rank gets a distinct pointer, IPC export and import
    # both succeed, and the kernel runs at ~273 GB/s because it is writing to
    # local HBM instead of over PCIe. Forcing the context here, and checking
    # arrival at the end, is what makes that failure loud.
    torch.cuda.set_device(rank)
    _ctx_probe = torch.empty(1, device=f"cuda:{rank}")  # noqa: F841 -- keeps the context
    assert torch.cuda.current_device() == rank, (
        f"rank {rank} bound to device {torch.cuda.current_device()}"
    )
    hip = pw.UncachedIpcHeap._load_hip()
    hip.hipSetDevice.restype = ctypes.c_int
    hip.hipSetDevice.argtypes = [ctypes.c_int]
    pw.UncachedIpcHeap._hip_check(hip.hipSetDevice(rank), what=f"hipSetDevice({rank})")

    dist.init_process_group(
        backend="gloo", init_method=init_method, world_size=world, rank=rank
    )

    npeers = world - 1
    slot = pw.TOTAL_BYTES + pw.FLAG_TAIL_BYTES
    inbox = pw.UncachedIpcHeap.alloc(world * slot, pw.ALLOC_FLAGS[cfg["alloc"]])

    handle = pw.UncachedIpcHeap.get_mem_handle_bytes(inbox)
    handles = [None] * world
    dist.all_gather_object(handles, handle)

    # A rank cannot open its own handle, and does not need to: npeers = world-1
    # excludes the self-write, matching the single-writer primitive.
    opened = {}
    for p in range(world):
        if p != rank:
            opened[p] = pw.UncachedIpcHeap.open_mem_handle(handles[p])

    # Round-robin starting one past self, so every rank has a different first
    # destination and the incast is symmetric rather than all-onto-rank-0.
    order = [(rank + 1 + i) % world for i in range(npeers)]
    ptrs = [opened[p] + rank * slot for p in order]
    table = pw.UncachedIpcHeap.alloc_uncached(npeers * 8)
    pw.UncachedIpcHeap.copy_host_to_device(
        table, (ctypes.c_int64 * npeers)(*ptrs), npeers * 8
    )

    gbps = None
    is_writer = rank < writers
    if is_writer:
        launch = pw._get_kernel(npeers, cfg["publish"], True)
        args = (
            pw.Int64(int(table)),
            pw.Int32(pw.TOTAL_BYTES // pw.V4_BYTES),
            pw.Int32(pw._shift_of(cfg["chunk"])),
            # Flags sit just past this writer's payload, inside its own slot.
            pw.Int32(pw.TOTAL_BYTES // pw.V4_BYTES),
            pw.Int32(cfg["blocks"]),
            pw.Int32(cfg["blocks"]),
            pw.Stream(None),
        )
        compiled = pw.flyc.compile(launch, *args)  # compiles AND launches once
        for _ in range(cfg["warmup"]):
            compiled(*args)
        torch.cuda.synchronize()

    # Barrier *after* warmup and compilation, so the timed windows overlap
    # rather than one rank still JIT-compiling while another is being measured.
    # Without this the four-writer run is not actually four concurrent writers.
    dist.barrier()

    if is_writer:
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(cfg["iters"]):
            compiled(*args)
        stop.record()
        stop.synchronize()
        us = start.elapsed_time(stop) * 1e3 / cfg["iters"]
        gbps = pw.TOTAL_BYTES / (us * 1e3)

    # Nobody frees while anybody may still be writing into them.
    dist.barrier()

    # Did the bytes actually arrive over the wire? Every writer w != rank owns
    # slot w of this rank's inbox and writes offset 0 of it first, so a zeroed
    # slot means that writer's stores went somewhere else. This is the assertion
    # that catches a rank silently bound to the wrong device -- the failure mode
    # that produces a fast, plausible, entirely wrong number.
    probe = (ctypes.c_ubyte * 64)()
    for w in range(min(writers, world)):
        if w == rank:
            continue
        pw.UncachedIpcHeap._hip_check(
            hip.hipMemcpy(
                ctypes.byref(probe),
                ctypes.c_void_p(inbox + w * slot),
                ctypes.c_size_t(64),
                ctypes.c_int(2),  # hipMemcpyDeviceToHost
            ),
            what="hipMemcpy(D2H) arrival probe",
        )
        assert any(probe), (
            f"rank {rank}: slot {w} is still zero -- writer {w} never reached "
            f"this GPU, so the measured bandwidth is not peer traffic"
        )
    dist.barrier()
    for p, ptr in opened.items():
        pw.UncachedIpcHeap.close_mem_handle(ptr)
    pw.UncachedIpcHeap.free_device_mem(table)
    pw.UncachedIpcHeap.free_device_mem(inbox)
    dist.destroy_process_group()
    return rank, gbps, pw.TOTAL_BYTES


def run_one(world, writers, cfg):
    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=world) as pool:
        rets = [
            pool.apply_async(_worker, args=(world, r, writers, init_method, cfg))
            for r in range(world)
        ]
        pool.close()
        pool.join()
    return [r.get() for r in rets]


def _report(rows, world, writers, cfg):
    got = sorted((r, g) for r, g, _ in rows if g is not None)
    per = [g for _, g in got]
    mib = (rows[0][2] >> 20) if rows else 0
    print()
    print(
        f"world={world} writers={writers} peers={world - 1} "
        f"alloc={cfg['alloc']} blocks={cfg['blocks']} chunk={cfg['chunk']}B "
        f"publish={cfg['publish']}B  {mib} MiB/writer/iter"
    )
    for r, g in got:
        print(f"  rank {r}: {g:7.2f} GB/s")
    if per:
        print(
            f"  per-writer min/mean/max: {min(per):.2f} / "
            f"{sum(per) / len(per):.2f} / {max(per):.2f} GB/s"
        )
        print(f"  aggregate: {sum(per):.2f} GB/s")
    return per


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--world", type=int, default=4, help="ranks, one GPU each")
    ap.add_argument("--writers", type=int, default=4,
                    help="how many ranks issue. 1 reproduces the single-writer "
                         "primitive of section 14.3; --world reproduces "
                         "qr_int4's all-to-all incast.")
    ap.add_argument("--sweep", action="store_true",
                    help="run writers=1..world in one go and print the ladder")
    ap.add_argument("--alloc", default="fine", choices=("fine", "coarse", "uncached"))
    ap.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    ap.add_argument("--publish", type=int, default=DEFAULT_PUBLISH)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS,
                    help="a counter run wants this small so few dispatch "
                         "folders are left, but warmup non-zero so the "
                         "measured dispatch is a warm one")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    args = ap.parse_args()

    cfg = {
        "alloc": args.alloc, "blocks": args.blocks, "chunk": args.chunk,
        "publish": args.publish, "iters": args.iters, "warmup": args.warmup,
    }

    if args.sweep:
        ladder = []
        for w in range(1, args.world + 1):
            rows = run_one(args.world, w, cfg)
            per = _report(rows, args.world, w, cfg)
            if per:
                ladder.append((w, min(per), sum(per) / len(per), sum(per)))
        print()
        print("incast ladder, same allocation and geometry throughout")
        print(f"{'writers':>8}{'slowest GB/s':>15}{'mean GB/s':>12}"
              f"{'aggregate GB/s':>17}{'vs 1 writer':>13}")
        base = ladder[0][2] if ladder else None
        for w, lo, mean, agg in ladder:
            rel = f"{mean / base:.2f}x" if base else "-"
            print(f"{w:>8}{lo:>15.2f}{mean:>12.2f}{agg:>17.2f}{rel:>13}")
        return 0

    rows = run_one(args.world, args.writers, cfg)
    per = _report(rows, args.world, args.writers, cfg)
    for r, g, _ in sorted(rows):
        if g is not None:
            print(f"RESULT writers={args.writers} rank={r} {g:.3f}")
    return 0 if per else 1


if __name__ == "__main__":
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    set_start_method("spawn", force=True)
    sys.exit(main())
