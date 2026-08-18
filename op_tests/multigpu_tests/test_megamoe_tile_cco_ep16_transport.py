# SPDX-License-Identifier: MIT
"""EP16 CCO RAIL transport smoke: torchrun 2 nodes x 8 GPUs.

Example launch (run once on each node with node-rank 0/1)::

    MORI_SOCKET_IFNAME=enp193s0f1np1 \
    GLOO_SOCKET_IFNAME=enp193s0f1np1 \
    torchrun --nnodes=2 --nproc-per-node=8 --node-rank=${NODE_RANK} \
      --master-addr=10.2.80.17 --master-port=29616 \
      op_tests/multigpu_tests/test_megamoe_tile_cco_ep16_transport.py

Every rank talks to the same-local-rank proxy on the other node. The RAIL peer
argument is the remote node index, not a world rank.
"""

from __future__ import annotations

from datetime import timedelta
import os
import sys

import torch
import torch.distributed as dist

from mori.cco import Communicator, UniqueId

from aiter.ops.flydsl.kernels.megamoe_tile import cco


LOCAL_WORLD_SIZE = 8
EXPECTED_WORLD_SIZE = 16
NUM_QP = int(os.environ.get("MEGAMOE_CCO_QP", "4"))
BATCH = int(os.environ.get("MEGAMOE_CCO_BATCH", "2"))
CHUNK_BYTES = int(os.environ.get("MEGAMOE_CCO_CHUNK", "1024"))
EPOCHS = int(os.environ.get("MEGAMOE_CCO_EPOCHS", "2"))


def _broadcast_cco_uid(rank: int) -> UniqueId:
    obj = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    payload = obj[0]
    if not isinstance(payload, bytes) or len(payload) != 128:
        raise RuntimeError("invalid CCO unique id broadcast")
    return UniqueId.from_bytes(payload)


def _pattern(rank: int, epoch: int, words: int):
    base = rank * 1_000_000 + epoch * 100_000
    return tuple(base + index for index in range(1, words + 1))


def main() -> int:
    dist.init_process_group("gloo", timeout=timedelta(minutes=10))
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    if world != EXPECTED_WORLD_SIZE:
        raise ValueError(f"requires world_size=16, got {world}")
    if local_rank != rank % LOCAL_WORLD_SIZE:
        raise ValueError(
            f"requires node-major torchrun ranks: rank={rank}, local_rank={local_rank}"
        )
    if EPOCHS <= 0:
        raise ValueError("the smoke must run at least one non-zero epoch")

    torch.cuda.set_device(local_rank)
    uid = _broadcast_cco_uid(rank)

    smoke = cco.build_transport_smoke(
        num_qp=NUM_QP,
        batch=BATCH,
        chunk_bytes=CHUNK_BYTES,
        team=cco.TEAM_RAIL,
    )
    arena_bytes = max(64 * 1024, (smoke.required_bytes + 4095) // 4096 * 4096)

    node = rank // LOCAL_WORLD_SIZE
    remote_node = 1 - node
    remote_rank = remote_node * LOCAL_WORLD_SIZE + local_rank

    local_failures = 0
    with Communicator.init(world, rank, uid, per_rank_vmm=64 * 1024 * 1024) as comm:
        resources = cco.create_transport_resources(
            comm,
            arena_bytes,
            num_qp=NUM_QP,
            team=cco.TEAM_RAIL,
        )
        arena_win = resources.window
        dc = resources.dev_comm
        cco.zero_window(arena_win.local_ptr, arena_bytes)
        payload_words = smoke.payload_bytes // 8
        send_ptr = arena_win.local_ptr + smoke.send_base
        recv_ptr = arena_win.local_ptr + smoke.recv_base

        for epoch in range(1, EPOCHS + 1):
            cco.write_window_u64(
                send_ptr, _pattern(rank, epoch, payload_words)
            )
            cco.zero_window(recv_ptr, smoke.payload_bytes)
            torch.cuda.synchronize()
            comm.barrier()

            # For CCO_TEAM_GDA + GDA_CONNECTION_RAIL the peer is node index.
            smoke.launch_send(dc.ptr, arena_win.handle, remote_node, epoch)
            smoke.launch_wait_words(arena_win.local_ptr, smoke.ready_base, epoch)
            torch.cuda.synchronize()

            expected = _pattern(remote_rank, epoch, payload_words)
            got = cco.read_window_u64(recv_ptr, payload_words)
            local_failures += sum(
                int(lhs != rhs) for lhs, rhs in zip(got, expected)
            )

            smoke.launch_credit(dc.ptr, arena_win.handle, remote_node, epoch)
            smoke.launch_wait_words(arena_win.local_ptr, smoke.credit_base, epoch)
            torch.cuda.synchronize()
            comm.barrier()

        ready = cco.read_window_u64(
            arena_win.local_ptr + smoke.ready_base, NUM_QP
        )
        credit = cco.read_window_u64(
            arena_win.local_ptr + smoke.credit_base, NUM_QP
        )
        local_failures += sum(int(value != EPOCHS) for value in ready)
        local_failures += sum(int(value != EPOCHS) for value in credit)

    status = torch.tensor([local_failures], dtype=torch.int64)
    dist.all_reduce(status, op=dist.ReduceOp.SUM)
    global_failures = int(status.item())
    dist.barrier()

    print(
        f"MEGAMOE_CCO_EP16_{'PASS' if global_failures == 0 else 'FAIL'} "
        f"rank={rank} local_rank={local_rank} remote_rank={remote_rank} "
        f"local_failures={local_failures} global_failures={global_failures}",
        flush=True,
    )
    dist.destroy_process_group()
    return 0 if global_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
