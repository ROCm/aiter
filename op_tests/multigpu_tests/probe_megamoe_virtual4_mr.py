# SPDX-License-Identifier: MIT
"""Check large multi-MR CCO windows on the two-host, 16-GPU virtual4 setup.

This transport probe deliberately has no AITER dependency. Launch eight ranks
on each physical host with the same MORI/RDMA environment. Each GPU allocates
one to three independent windows, then its aligned peer on the other host writes an
i64 marker at the highest eight-byte address of every window.
When an allocation is larger than its MR, also check every same-host LSA peer
at the allocation tail beyond the registered range.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
import torch
import torch.distributed as dist
from flydsl._mlir.dialects import llvm as llvm_d
from mori.cco import GDA_CONNECTION_RAIL, CCODevCommRequirements, Communicator, UniqueId

WORLD = 16
GPUS_PER_HOST = 8
_DEFAULT_STREAM = fx.Stream(None)
logger = logging.getLogger(__name__)


class _CudaI64:
    def __init__(self, ptr: int):
        self.__cuda_array_interface__ = {
            "data": (int(ptr), False),
            "shape": (1,),
            "typestr": "<i8",
            "version": 3,
        }


@flyc.kernel(known_block_size=[64, 1, 1])
def lsa_flag_kernel(
    win_handle: fx.Int64,
    peer_lsa: fx.Int32,
    dst_off: fx.Int64,
    generation: fx.Int64,
):
    if fx.Int32(fx.thread_idx.x) == fx.Int32(0):
        addr = cco.Window(win_handle).lsa_ptr(peer_lsa, dst_off)
        ptr_type = fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8)
        ptr = fx.to_llvm_ptr(fx.inttoptr(ptr_type, fx.Int64(addr)))
        llvm_d.StoreOp(
            generation.ir_value(),
            ptr,
            alignment=8,
            ordering=llvm_d.AtomicOrdering.release,
            syncscope="one-as",
        )


@flyc.jit
def run_lsa_flag(
    win_handle: fx.Int64,
    peer_lsa: fx.Int32,
    dst_off: fx.Int64,
    generation: fx.Int64,
    stream=_DEFAULT_STREAM,
):
    lsa_flag_kernel(win_handle, peer_lsa, dst_off, generation).launch(
        grid=(1, 1, 1), block=(64, 1, 1), stream=stream
    )


@flyc.kernel(known_block_size=[64, 1, 1])
def rail_flag_kernel(
    dev_comm: fx.Int64,
    win_handle: fx.Int64,
    peer_rank: fx.Int32,
    dst_off: fx.Int64,
    generation: fx.Int64,
):
    gda = cco.DevComm(dev_comm).gda(0)
    gda.put_value(
        peer_rank,
        win_handle,
        dst_off,
        generation,
        coop=cco.CoopScope.WARP,
    )
    gda.flush_peer(peer_rank, coop=cco.CoopScope.WARP)


@flyc.jit
def run_rail_flag(
    dev_comm: fx.Int64,
    win_handle: fx.Int64,
    peer_rank: fx.Int32,
    dst_off: fx.Int64,
    generation: fx.Int64,
    stream=_DEFAULT_STREAM,
):
    rail_flag_kernel(dev_comm, win_handle, peer_rank, dst_off, generation).launch(
        grid=(1, 1, 1), block=(64, 1, 1), stream=stream
    )


def _parse_sizes(raw: str) -> tuple[int, ...]:
    try:
        mib = tuple(int(part) for part in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must be comma-separated MiB") from exc
    if not 1 <= len(mib) <= 3 or min(mib) < 1:
        raise argparse.ArgumentTypeError("one to three positive MiB sizes are required")
    return tuple(value << 20 for value in mib)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mr-sizes-mib", type=_parse_sizes, default=_parse_sizes("960,960,704")
    )
    parser.add_argument(
        "--allocation-sizes-mib",
        type=_parse_sizes,
        help="physical allocation sizes; defaults to the registered MR sizes",
    )
    parser.add_argument("--read-timeout-s", type=float, default=20.0)
    args = parser.parse_args()
    if args.read_timeout_s <= 0:
        parser.error("--read-timeout-s must be positive")
    allocation_sizes = args.allocation_sizes_mib or args.mr_sizes_mib
    if len(allocation_sizes) != len(args.mr_sizes_mib):
        parser.error("allocation and MR sizes must have the same number of entries")
    if any(alloc < mr for alloc, mr in zip(allocation_sizes, args.mr_sizes_mib)):
        parser.error("each allocation must cover its registered MR")

    os.environ["MORI_CCO_FABRIC_DISABLE"] = "1"
    if os.getenv("MORI_VPOD_ID") is not None:
        raise RuntimeError("MORI_VPOD_ID must be unset for physical 2x8 CCO teams")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    if dist.get_world_size() != WORLD or local_rank != rank % GPUS_PER_HOST:
        raise RuntimeError(
            "expected two hosts with eight ranks each in host-major order"
        )
    peer_node = 1 - rank // GPUS_PER_HOST
    peer_rank = peer_node * GPUS_PER_HOST + local_rank

    uid_bytes = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(uid_bytes, src=0)
    uid = UniqueId.from_bytes(uid_bytes[0])
    vmm_bytes = max(
        1 << 30,
        ((sum(allocation_sizes) + (512 << 20) + (1 << 30) - 1) // (1 << 30))
        * (1 << 30),
    )
    print(
        f"[rank {rank}] CCO init; sizes_MiB={[v >> 20 for v in args.mr_sizes_mib]} "
        f"allocation_MiB={[v >> 20 for v in allocation_sizes]} "
        f"GDR={os.getenv('CCO_GDR_CAPABLE', 'default')}",
        flush=True,
    )
    with Communicator.init(WORLD, rank, uid, per_rank_vmm=vmm_bytes) as comm:
        memories = []
        for index, size in enumerate(allocation_sizes):
            memory = None
            error = ""
            # Serializing physical GPU lanes reproduces the allocation order
            # used when the original third-MR hipMemSetAccess failure appeared.
            for lane in range(GPUS_PER_HOST):
                if local_rank == lane:
                    try:
                        print(
                            f"[rank {rank}] alloc MR {index} bytes={size}", flush=True
                        )
                        memory = comm.alloc_mem(size)
                    except Exception as exc:  # allow every rank to leave its barrier
                        logger.exception("CCO window allocation failed")
                        error = f"{type(exc).__name__}: {exc}"
                dist.barrier()
            errors = [None] * WORLD
            dist.all_gather_object(errors, error)
            if any(errors):
                raise RuntimeError(f"MR {index} allocation failed by rank: {errors}")
            memories.append(memory)

        windows = []
        for index, (memory, size) in enumerate(zip(memories, args.mr_sizes_mib)):
            print(f"[rank {rank}] register MR {index}", flush=True)
            windows.append(comm.register_window(memory.ptr, size))
            dist.barrier()

        req = CCODevCommRequirements()
        req.gda_connection_type = GDA_CONNECTION_RAIL
        req.gda_context_count = 1
        req.gda_signal_count = 0
        req.gda_counter_count = 0
        req.lsa_barrier_count = 0
        req.rail_gda_barrier_count = 0
        req.barrier_count = 0
        print(f"[rank {rank}] create device communicator", flush=True)
        dev_comm = comm.create_dev_comm(req)
        print(f"[rank {rank}] device communicator ready", flush=True)
        if dev_comm.lsa_size != GPUS_PER_HOST:
            raise RuntimeError(f"CCO LSA size {dev_comm.lsa_size}, expected 8")

        tail_views = [
            torch.as_tensor(_CudaI64(win.local_ptr + size - 8), device="cuda")
            for win, size in zip(windows, args.mr_sizes_mib)
        ]
        for tail in tail_views:
            tail.zero_()
        torch.cuda.synchronize()
        print(f"[rank {rank}] tail initialization complete", flush=True)
        dist.barrier()
        for index, (win, size) in enumerate(zip(windows, args.mr_sizes_mib)):
            run_rail_flag(
                dev_comm.ptr,
                win.handle,
                # This installed MORI FlyDSL API uses world-rank peer IDs.
                # GDA_CONNECTION_RAIL restricts endpoints to the aligned PE.
                peer_rank,
                size - 8,
                1000 + rank * 64 + index,
            )
        torch.cuda.synchronize()
        print(f"[rank {rank}] rail sends complete", flush=True)
        dist.barrier()
        failures = 0
        for index, tail in enumerate(tail_views):
            expected = 1000 + peer_rank * 64 + index
            deadline = time.monotonic() + args.read_timeout_s
            actual = int(tail.item())
            while actual != expected and time.monotonic() < deadline:
                time.sleep(0.01)
                actual = int(tail.item())
            if actual != expected:
                failures += 1
                print(
                    f"[rank {rank}] MR {index} tail actual={actual} expected={expected}",
                    flush=True,
                )
        total = torch.tensor([failures], dtype=torch.int64)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        if rank == 0:
            print(
                f"virtual4 MR tail writes: {len(windows) * WORLD - int(total[0])}/"
                f"{len(windows) * WORLD} "
                f"passed; GDR={os.getenv('CCO_GDR_CAPABLE', 'default')}",
                flush=True,
            )
        if int(total[0]):
            raise AssertionError(f"high-address rail writes failed: {int(total[0])}")

        padded = [
            (index, win, alloc)
            for index, (win, alloc, mr) in enumerate(
                zip(windows, allocation_sizes, args.mr_sizes_mib)
            )
            if alloc > mr
        ]
        lsa_tails = [
            torch.as_tensor(_CudaI64(win.local_ptr + alloc - 8), device="cuda")
            for _, win, alloc in padded
        ]
        lsa_failures = 0
        for shift in range(1, GPUS_PER_HOST) if padded else ():
            for tail in lsa_tails:
                tail.zero_()
            torch.cuda.synchronize()
            dist.barrier()
            for index, win, alloc in padded:
                run_lsa_flag(
                    win.handle,
                    (local_rank + shift) % GPUS_PER_HOST,
                    alloc - 8,
                    2000 + shift * WORLD * 64 + rank * 64 + index,
                )
            torch.cuda.synchronize()
            dist.barrier()
            source_rank = rank - local_rank + (local_rank - shift) % GPUS_PER_HOST
            for (index, _, _), tail in zip(padded, lsa_tails):
                expected = 2000 + shift * WORLD * 64 + source_rank * 64 + index
                actual = int(tail.item())
                if actual != expected:
                    lsa_failures += 1
                    print(
                        f"[rank {rank}] LSA shift {shift} allocation {index} "
                        f"tail actual={actual} expected={expected}",
                        flush=True,
                    )
        if padded:
            total = torch.tensor([lsa_failures], dtype=torch.int64)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            checks = len(padded) * WORLD * (GPUS_PER_HOST - 1)
            if rank == 0:
                print(
                    f"unregistered allocation tail LSA writes: "
                    f"{checks - int(total[0])}/{checks} passed",
                    flush=True,
                )
            if int(total[0]):
                raise AssertionError(
                    f"unregistered LSA tail writes failed: {int(total[0])}"
                )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
