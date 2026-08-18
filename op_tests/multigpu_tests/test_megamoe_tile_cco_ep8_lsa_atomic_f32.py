# SPDX-License-Identifier: MIT
"""EP8 CCO-LSA system-scope FP32 atomic-add consistency microbenchmark.

This is deliberately independent of MegaMoE Stage-1/Stage-2.  Producers
resolve one aligned target rank through ``cco.lsa_ptr`` and concurrently issue
system-scope LLVM ``atomicrmw fadd`` operations into the same accumulator
array.  Every addend is exactly ``1.0`` and the expected total is below 2**24,
so a non-zero error is a lost/stale update rather than floating-point order
noise.  It covers three producer sets (local-only, one remote, all eight) and
two completion paths:

* ``device_ready``: the target immediately launches a reader that polls the
  producer-published system-scope ready counter; there is no host/device sync
  or Gloo barrier between producer launch and read.
* ``host_sync``: every producer synchronizes its stream and all ranks enter a
  Gloo barrier before all eight ranks read the target.

This separates atomic-update correctness from publication/visibility bugs.

Launch on one 8-GPU node::

    GLOO_SOCKET_IFNAME=enp193s0f1np1 \
    MORI_SOCKET_IFNAME=enp193s0f1np1 \
    torchrun --standalone --nproc-per-node=8 \
      op_tests/multigpu_tests/test_megamoe_tile_cco_ep8_lsa_atomic_f32.py \
      --epochs 20
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import os
import sys

import torch
import torch.distributed as dist

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_d
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from mori.cco import (
    Communicator,
    CCODevCommRequirements,
    GDA_CONNECTION_NONE,
    UniqueId,
)

from aiter.ops.flydsl.kernels.megamoe_tile import cco
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.megamoe_tile import comm_ops


WORLD = 8
WINDOW_BYTES = 4096
ELEMENTS = 256
BLOCKS = 64
THREADS = 256
REPEATS = 8
ACCUMULATOR_OFFSET = 0
BLOCK_DONE_OFFSET = ELEMENTS * 4
READY_OFFSET = BLOCK_DONE_OFFSET + WORLD * 4


def _atomic_add_f32_system(address, value):
    """Match the Stage-2 peer-LSA FP32 atomic instruction exactly."""

    ptr = llvm_d.IntToPtrOp(
        llvm_d.PointerType.get(address_space=1), arith.unwrap(address)
    ).result
    return llvm_d.AtomicRMWOp(
        llvm_d.AtomicBinOp.fadd,
        ptr,
        arith.unwrap(value),
        llvm_d.AtomicOrdering.monotonic,
        syncscope="one-as",
    ).res


@flyc.kernel(
    name="megamoe_tile_cco_ep8_lsa_atomic_f32",
    known_block_size=[THREADS, 1, 1],
)
def atomic_add_kernel(
    window: fx.Int64,
    target_rank: fx.Int32,
    source_rank: fx.Int32,
):
    tid = (
        fx.Int32(gpu.block_id("x")) * fx.Int32(THREADS)
        + fx.Int32(gpu.thread_id("x"))
    )
    element = tid % fx.Int32(ELEMENTS)
    target = cco.lsa_ptr(
        window,
        target_rank,
        fx.Int64(ACCUMULATOR_OFFSET) + fx.Int64(element) * fx.Int64(4),
    )
    for _ in range_constexpr(REPEATS):
        _atomic_add_f32_system(target, fx.Float32(1.0))
    rocdl.s_waitcnt(0)
    comm_ops.fence_system_release()
    gpu.barrier()

    # The final block from a producer publishes one ready count.  acq_rel on
    # the per-rank block counter joins every block's release sequence before
    # the final block releases the producer-ready update.
    if fx.Int32(gpu.thread_id("x")) == fx.Int32(0):
        block_done = cco.lsa_ptr(
            window,
            target_rank,
            fx.Int64(BLOCK_DONE_OFFSET)
            + fx.Int64(source_rank) * fx.Int64(4),
        )
        old = fx.Int32(
            comm_ops.atomic_add_system_acq_rel(block_done, fx.Int32(1))
        )
        if old == fx.Int32(BLOCKS - 1):
            ready = cco.lsa_ptr(
                window, target_rank, fx.Int64(READY_OFFSET)
            )
            comm_ops.atomic_add_system_acq_rel(ready, fx.Int32(1))


@flyc.kernel(
    name="megamoe_tile_cco_ep8_lsa_read_f32",
    known_block_size=[THREADS, 1, 1],
)
def read_target_kernel(
    window: fx.Int64,
    target_rank: fx.Int32,
    output: fx.Int64,
    expected_ready: fx.Int32,
):
    element = fx.Int32(gpu.thread_id("x"))
    if element == fx.Int32(0):
        ready_address = cco.lsa_ptr(
            window, target_rank, fx.Int64(READY_OFFSET)
        )
        ready = fx.Int32(comm_ops.load_i32_global_system(ready_address))
        while ready < expected_ready:
            ready = fx.Int32(
                comm_ops.load_i32_global_system(ready_address)
            )
    gpu.barrier()
    comm_ops.fence_system_acquire()
    target = cco.lsa_ptr(
        window,
        target_rank,
        fx.Int64(ACCUMULATOR_OFFSET) + fx.Int64(element) * fx.Int64(4),
    )
    target_rsrc = buffer_ops.create_buffer_resource_from_addr(target)
    output_rsrc = buffer_ops.create_buffer_resource_from_addr(output)
    value = buffer_ops.buffer_load(
        target_rsrc, fx.Int32(0), vec_width=1, dtype=T.f32
    )
    buffer_ops.buffer_store(value, output_rsrc, element)


@flyc.jit
def launch_atomic_add(
    window: fx.Int64,
    target_rank: fx.Int32,
    source_rank: fx.Int32,
    stream=fx.Stream(None),
):
    atomic_add_kernel(window, target_rank, source_rank).launch(
        grid=(BLOCKS, 1, 1),
        block=(THREADS, 1, 1),
        stream=stream,
    )


@flyc.jit
def launch_read_target(
    window: fx.Int64,
    target_rank: fx.Int32,
    output: fx.Int64,
    expected_ready: fx.Int32,
    stream=fx.Stream(None),
):
    read_target_kernel(window, target_rank, output, expected_ready).launch(
        grid=(1, 1, 1),
        block=(THREADS, 1, 1),
        stream=stream,
    )


def _broadcast_cco_uid(rank: int) -> UniqueId:
    obj = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    payload = obj[0]
    if not isinstance(payload, bytes) or len(payload) != 128:
        raise RuntimeError("invalid CCO unique id broadcast")
    return UniqueId.from_bytes(payload)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--target-rank", type=int, default=0)
    return parser.parse_args()


def _local_stats(output: torch.Tensor, expected: float):
    finite = torch.isfinite(output)
    mismatches = int(torch.count_nonzero(output != expected).item())
    if bool(torch.all(finite).item()):
        actual_min = float(output.min().item())
        actual_max = float(output.max().item())
        max_error = float((output - expected).abs().max().item())
    else:
        actual_min = float("-inf")
        actual_max = float("inf")
        max_error = float("inf")
    return actual_min, actual_max, max_error, mismatches


def _reduce_and_print(
    *,
    rank: int,
    case: str,
    observation: str,
    epoch: int,
    epochs: int,
    expected: float,
    stats,
):
    actual_min, actual_max, max_error, mismatches = stats
    minimum = torch.tensor([actual_min], dtype=torch.float64)
    maximum = torch.tensor([actual_max], dtype=torch.float64)
    error = torch.tensor([max_error], dtype=torch.float64)
    mismatch = torch.tensor([mismatches], dtype=torch.int64)
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    dist.all_reduce(mismatch, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(
            "CCO_LSA_ATOMIC_F32 "
            f"case={case} observation={observation} "
            f"epoch={epoch}/{epochs} expected={expected:.1f} "
            f"actual_min={float(minimum.item()):.1f} "
            f"actual_max={float(maximum.item()):.1f} "
            f"max_error={float(error.item()):.1f} "
            f"mismatches={int(mismatch.item())}",
            flush=True,
        )
    return int(mismatch.item())


def main() -> int:
    args = _parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not 0 <= args.target_rank < WORLD:
        raise ValueError(f"--target-rank must be in [0, {WORLD})")

    dist.init_process_group("gloo", timeout=timedelta(minutes=10))
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != WORLD:
        raise ValueError(f"requires one-node world_size={WORLD}, got {world}")
    if rank != local_rank:
        raise ValueError("one-node EP8 test requires rank == LOCAL_RANK")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    uid = _broadcast_cco_uid(rank)
    output = torch.empty(ELEMENTS, dtype=torch.float32, device=device)
    failures = 0

    with Communicator.init(
        world, rank, uid, per_rank_vmm=64 * 1024 * 1024
    ) as comm:
        memory = comm.alloc_mem(WINDOW_BYTES)
        window = comm.register_window(memory.ptr, memory.size)

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_context_count = 1
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.lsa_barrier_count = 0
        reqs.rail_gda_barrier_count = 0
        reqs.barrier_count = 0
        dc = comm.create_dev_comm(reqs)
        if dc.lsa_size != WORLD or dc.lsa_rank != local_rank:
            raise RuntimeError(
                f"unexpected LSA mapping size={dc.lsa_size} rank={dc.lsa_rank}"
            )

        # Compile both kernels before entering the synchronized epoch loop.
        if rank == args.target_rank:
            cco.zero_window(window.local_ptr, WINDOW_BYTES)
        launch_atomic_add(
            window.handle,
            args.target_rank,
            rank,
            stream=torch.cuda.current_stream(device),
        )
        torch.cuda.synchronize()
        dist.barrier()
        output.fill_(float("nan"))
        launch_read_target(
            window.handle,
            args.target_rank,
            output.data_ptr(),
            0,
            stream=torch.cuda.current_stream(device),
        )
        torch.cuda.synchronize()
        dist.barrier()

        remote_rank = (args.target_rank + 1) % WORLD
        cases = (
            ("local_only", (args.target_rank,)),
            ("single_remote", (remote_rank,)),
            ("all_ranks", tuple(range(WORLD))),
        )
        for case, producers in cases:
            active = rank in producers
            expected = float(
                len(producers)
                * (BLOCKS * THREADS // ELEMENTS)
                * REPEATS
            )
            for observation in ("device_ready", "host_sync"):
                for epoch in range(1, args.epochs + 1):
                    if rank == args.target_rank:
                        cco.zero_window(window.local_ptr, WINDOW_BYTES)
                    dist.barrier()

                    if active:
                        launch_atomic_add(
                            window.handle,
                            args.target_rank,
                            rank,
                            stream=torch.cuda.current_stream(device),
                        )

                    if observation == "device_ready":
                        # The owner queues this directly behind its producer
                        # (if any); remote producers run independently.  There
                        # is intentionally no host sync/Gloo barrier here.
                        if rank == args.target_rank:
                            output.fill_(float("nan"))
                            launch_read_target(
                                window.handle,
                                args.target_rank,
                                output.data_ptr(),
                                len(producers),
                                stream=torch.cuda.current_stream(device),
                            )
                        torch.cuda.synchronize()
                        dist.barrier()
                        if rank == args.target_rank:
                            stats = _local_stats(output, expected)
                        else:
                            stats = (float("inf"), float("-inf"), 0.0, 0)
                    else:
                        torch.cuda.synchronize()
                        dist.barrier()
                        output.fill_(float("nan"))
                        launch_read_target(
                            window.handle,
                            args.target_rank,
                            output.data_ptr(),
                            0,
                            stream=torch.cuda.current_stream(device),
                        )
                        torch.cuda.synchronize()
                        stats = _local_stats(output, expected)

                    failures += _reduce_and_print(
                        rank=rank,
                        case=case,
                        observation=observation,
                        epoch=epoch,
                        epochs=args.epochs,
                        expected=expected,
                        stats=stats,
                    )
                    dist.barrier()

    global_failures = torch.tensor([failures], dtype=torch.int64)
    dist.all_reduce(global_failures, op=dist.ReduceOp.MAX)
    failed = int(global_failures.item())
    if rank == 0:
        print(
            f"MEGAMOE_CCO_EP8_LSA_ATOMIC_F32_"
            f"{'PASS' if failed == 0 else 'FAIL'} "
            f"epochs={args.epochs} target_rank={args.target_rank} "
            f"cases=3 observations=2 total_mismatches={failed}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
