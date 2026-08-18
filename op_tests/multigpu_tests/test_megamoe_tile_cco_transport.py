# SPDX-License-Identifier: MIT
"""Two-rank WORLD/RAIL smoke for the private CCO transport protocol."""

from __future__ import annotations

import os
import sys
import time

import torch

from mori.cco import Communicator, UniqueId

from aiter.ops.flydsl.kernels.megamoe_tile import cco


NUM_QP = int(os.environ.get("MEGAMOE_CCO_QP", "4"))
BATCH = int(os.environ.get("MEGAMOE_CCO_BATCH", "2"))
CHUNK_BYTES = int(os.environ.get("MEGAMOE_CCO_CHUNK", "1024"))
EPOCHS = 2
TEAM_NAME = os.environ.get("MEGAMOE_CCO_TEAM", cco.TEAM_WORLD).lower()


def bootstrap():
    rank = int(os.environ["CCO_RANK"])
    world = int(os.environ["CCO_WORLD"])
    path = os.environ["CCO_UID_FILE"]
    while not os.path.exists(path) or os.path.getsize(path) != 128:
        time.sleep(0.05)
    with open(path, "rb") as f:
        return rank, world, UniqueId.from_bytes(f.read())


def _pattern(rank: int, epoch: int, words: int):
    base = rank * 1_000_000 + epoch * 100_000
    return tuple(base + index for index in range(1, words + 1))


def main() -> int:
    rank, world, uid = bootstrap()
    if world != 2:
        raise ValueError("requires exactly two ranks")
    if EPOCHS <= 0:
        raise ValueError("the smoke must run at least one non-zero epoch")

    torch.cuda.set_device(int(os.environ.get("CCO_GPU", "0")))
    smoke = cco.build_transport_smoke(
        num_qp=NUM_QP,
        batch=BATCH,
        chunk_bytes=CHUNK_BYTES,
        team=TEAM_NAME,
    )
    arena_bytes = max(64 * 1024, (smoke.required_bytes + 4095) // 4096 * 4096)
    peer = 1 - rank

    failures = 0
    with Communicator.init(world, rank, uid, per_rank_vmm=64 * 1024 * 1024) as comm:
        resources = cco.create_transport_resources(
            comm, arena_bytes, num_qp=NUM_QP, team=TEAM_NAME
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

            smoke.launch_send(dc.ptr, arena_win.handle, peer, epoch)
            smoke.launch_wait_words(arena_win.local_ptr, smoke.ready_base, epoch)
            torch.cuda.synchronize()

            expected = _pattern(1 - rank, epoch, payload_words)
            got = cco.read_window_u64(recv_ptr, payload_words)
            failures += sum(int(lhs != rhs) for lhs, rhs in zip(got, expected))

            smoke.launch_credit(dc.ptr, arena_win.handle, peer, epoch)
            smoke.launch_wait_words(arena_win.local_ptr, smoke.credit_base, epoch)
            torch.cuda.synchronize()
            comm.barrier()

        ready = cco.read_window_u64(
            arena_win.local_ptr + smoke.ready_base, NUM_QP
        )
        credit = cco.read_window_u64(
            arena_win.local_ptr + smoke.credit_base, NUM_QP
        )
        failures += sum(int(value != EPOCHS) for value in ready)
        failures += sum(int(value != EPOCHS) for value in credit)
        print(
            f"MEGAMOE_CCO_TRANSPORT_{'PASS' if failures == 0 else 'FAIL'} "
            f"rank={rank} team={TEAM_NAME} qps={NUM_QP} batch={BATCH} "
            f"chunk={CHUNK_BYTES} payload={smoke.payload_bytes} epochs={EPOCHS}",
            flush=True,
        )

    return failures


if __name__ == "__main__":
    sys.exit(main())
