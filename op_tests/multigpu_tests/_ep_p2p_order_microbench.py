# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Validate the P2P publish/observe order the fused GEMM2 combine relies on.

Every rank writes a payload into the NEXT peer's symmetric slot with plain 16 B
vector stores, issues one system-scope release fence per slot, then stores a ready
epoch. Concurrently (separate stream) every rank spins on its OWN slots' epochs,
takes an acquire fence, and re-reads the payload it was just told about.

A mismatch means the peer observed "ready epoch" without the payload that preceded
it, i.e. `payload stores -> release fence -> ready store` is not an ordering the
hardware honours across xGMI, and the tile-ready protocol must fall back to real
communication primitives instead of a plain ready flag.

`--fence 0` drops the release fence: that run is expected to be able to fail, and
is what proves the checker is actually sensitive to reordering.

    torchrun --standalone --nproc_per_node=4 _ep_p2p_order_microbench.py
"""
import argparse
import os

import torch
import torch.distributed as dist

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, T, vector
from flydsl.expr.typing import Int32, Int64

import mori.cco.device.flydsl as cco

from aiter.ops.flydsl.dispatch_combine_v2 import SymmArena
from aiter.ops.flydsl.dispatch_combine_v2 import flydsl_prims as P
from aiter.ops.flydsl.dispatch_combine_v2.intranode_kernels import WAVE, LANE_MASK, LOG2_WAVE

BLOCKS = 64
WARPS = 4
THREADS = WARPS * WAVE

# Payload pattern: value(epoch, slot, i) = epoch*EPOCH_MIX + slot*SLOT_MIX + i.
EPOCH_MIX = 1000003
SLOT_MIX = 7919


def _pattern_base(epoch, slot):
    return epoch * EPOCH_MIX + slot * SLOT_MIX


def make_producer(*, n_slots, slot_i32, off_data, off_flag, order):
    """One warp per slot: vec4 payload stores -> ordering -> ready epoch.

    order="fence"    explicit release fence, then a release-store of the epoch
    order="release"  release-store of the epoch only (no separate fence)
    order="relaxed"  ONE release fence, then plain volatile epoch stores. This is
                     what a producer wants when it publishes many flags at once:
                     a release store per flag would drain the write path each time.
    order="none"     plain volatile epoch store, no ordering at all (control)
    """

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def p2p_producer(arena: Int64, dst_pe: Int32, epoch: Int32):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & LANE_MASK
        warp = tid >> LOG2_WAVE
        gwarp = bid * WARPS + warp
        gwarps = BLOCKS * WARPS

        window = cco.Window(arena)
        data_base = fx.Int64(window.lsa_ptr(dst_pe, off_data))
        flag_base = fx.Int64(window.lsa_ptr(dst_pe, off_flag))

        for slot in range(gwarp, n_slots, gwarps):
            slot_addr = data_base + fx.Int64(slot) * fx.Int64(slot_i32 * 4)
            base_val = epoch * EPOCH_MIX + slot * SLOT_MIX
            for off in range(lane * 4, slot_i32, WAVE * 4):
                vals = [base_val + off + j for j in range_constexpr(4)]
                vec = vector.from_elements(
                    T.VectorType.get([4], T.i32()), [arith.unwrap(v) for v in vals]
                )
                P.store_v4i32(slot_addr, off, vec)
            if const_expr(order in ("fence", "relaxed")):
                P.fence_system_release()
            if lane == 0:
                if const_expr(order in ("none", "relaxed")):
                    P.store_i32_relaxed(flag_base, slot, epoch)
                else:
                    P.store_i32_system(flag_base, slot, epoch)

    @flyc.jit
    def run(arena: Int64, dst_pe: Int32, epoch: Int32, stream=fx.Stream(None)):
        p2p_producer(arena, dst_pe, epoch).launch(
            grid=(BLOCKS, 1, 1), block=[THREADS, 1, 1], stream=stream
        )

    return run


def make_consumer(*, n_slots, slot_i32, off_data, off_flag, cached_load, order):
    """One warp per slot: spin on the ready epoch -> acquire -> verify payload."""

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def p2p_consumer(arena: Int64, my_lsa_rank: Int32, epoch: Int32, addr_err: Int64):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & LANE_MASK
        warp = tid >> LOG2_WAVE
        gwarp = bid * WARPS + warp
        gwarps = BLOCKS * WARPS

        window = cco.Window(arena)
        data_base = fx.Int64(window.lsa_ptr(my_lsa_rank, off_data))
        flag_base = fx.Int64(window.lsa_ptr(my_lsa_rank, off_flag))

        for slot in range(gwarp, n_slots, gwarps):
            # Uniform across the wave, so every lane resumes only after the epoch
            # for this slot has landed.
            P.spin_until_ge_i32(flag_base + fx.Int64(slot) * fx.Int64(4), epoch)
            if const_expr(order != "none"):
                P.fence_system_acquire()

            slot_addr = data_base + fx.Int64(slot) * fx.Int64(slot_i32 * 4)
            base_val = epoch * EPOCH_MIX + slot * SLOT_MIX
            for off in range(lane * 4, slot_i32, WAVE * 4):
                if const_expr(cached_load):
                    vec = P.load_v4i32(slot_addr, off)
                else:
                    vec = P.load_v4i32_nt(slot_addr, off)
                for j in range_constexpr(4):
                    got = vector.extract(vec, static_position=[j])
                    want = base_val + off + j
                    if got != want:
                        P.atomic_add_global(fx.Int64(addr_err), arith.constant(1))
                        P.store_i32_system(fx.Int64(addr_err), 1, slot)
                        P.store_i32_system(fx.Int64(addr_err), 2, off + j)

    @flyc.jit
    def run(
        arena: Int64,
        my_lsa_rank: Int32,
        epoch: Int32,
        addr_err: Int64,
        stream=fx.Stream(None),
    ):
        p2p_consumer(arena, my_lsa_rank, epoch, addr_err).launch(
            grid=(BLOCKS, 1, 1), block=[THREADS, 1, 1], stream=stream
        )

    return run


class Dist:
    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
        torch.cuda.set_device(self.local_rank)

    def bcast_uid(self, uid):
        objs = [uid if self.rank == 0 else None]
        dist.broadcast_object_list(objs, src=0)
        return objs[0]

    def sum_int(self, value):
        t = torch.tensor([int(value)], dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return int(t.item())


def main():
    ap = argparse.ArgumentParser(description="P2P release/acquire ordering probe")
    ap.add_argument("--slots", type=int, default=1024, help="ready flags per rank")
    ap.add_argument("--slot_bytes", type=int, default=4096, help="payload per slot")
    ap.add_argument("--epochs", type=int, default=32, help="publish rounds")
    ap.add_argument(
        "--order",
        choices=["fence", "release", "relaxed", "none"],
        default="fence",
        help="publish ordering; 'none' is the unordered negative control",
    )
    ap.add_argument("--cached", type=int, default=1, help="1 = cached consumer loads")
    args = ap.parse_args()

    d = Dist()
    from mori.cco import Communicator

    uid = Communicator.get_unique_id() if d.rank == 0 else None
    uid = d.bcast_uid(uid)
    comm = Communicator.init(d.world, d.rank, uid)

    slot_i32 = args.slot_bytes // 4
    assert slot_i32 % (WAVE * 4) == 0, "slot must be a whole number of wave vec4 rows"

    arena = SymmArena(
        comm, [("data", args.slots * args.slot_bytes), ("flag", args.slots * 4)]
    )
    arena.zero()
    comm.barrier()

    producer = make_producer(
        n_slots=args.slots,
        slot_i32=slot_i32,
        off_data=arena.offset("data"),
        off_flag=arena.offset("flag"),
        order=args.order,
    )
    consumer = make_consumer(
        n_slots=args.slots,
        slot_i32=slot_i32,
        off_data=arena.offset("data"),
        off_flag=arena.offset("flag"),
        cached_load=bool(args.cached),
        order=args.order,
    )

    dev = torch.device("cuda", d.local_rank)
    err = torch.zeros(4, dtype=torch.int32, device=dev)
    dst_pe = (d.rank + 1) % d.world

    prod_stream = torch.cuda.Stream()
    cons_stream = torch.cuda.Stream()
    comm.barrier()

    for ep in range(1, args.epochs + 1):
        prod_stream.wait_stream(torch.cuda.current_stream())
        cons_stream.wait_stream(torch.cuda.current_stream())
        # Consumer spins on the previous peer while this rank publishes to the next
        # one, so the two directions really are in flight at the same time.
        with torch.cuda.stream(cons_stream):
            consumer(arena.handle, d.rank, ep, err.data_ptr(), fx.Stream(cons_stream))
        with torch.cuda.stream(prod_stream):
            producer(arena.handle, dst_pe, ep, fx.Stream(prod_stream))
        torch.cuda.current_stream().wait_stream(prod_stream)
        torch.cuda.current_stream().wait_stream(cons_stream)
        torch.cuda.synchronize()
        comm.barrier()

    e = err.cpu().tolist()
    total = d.sum_int(e[0])
    if d.rank == 0:
        cfg = (
            f"slots={args.slots} slot_bytes={args.slot_bytes} epochs={args.epochs} "
            f"order={args.order} cached_load={args.cached} world={d.world}"
        )
        if total == 0:
            print(f"PASS: P2P release/acquire ordering held ({cfg})", flush=True)
        else:
            print(
                f"FAIL: {total} payload mismatches ({cfg}); "
                f"rank0 first bad slot={e[1]} idx={e[2]}",
                flush=True,
            )
    arena.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
