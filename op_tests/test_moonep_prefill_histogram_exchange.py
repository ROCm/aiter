# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Distributed check for PrefillPolicy's symmetric histogram exchange.

Run with::

    torchrun --standalone --nproc-per-node=8 \
        op_tests/test_moonep_prefill_histogram_exchange.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from aiter.ops.flydsl.moonep import MoonEPSymmetricHistogramExchange

E = int(os.environ.get("T_E", "384"))


def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    import mori
    import mori.shmem as ms

    cpu_group = dist.new_group(backend="gloo")
    torch._C._distributed_c10d._register_process_group("mori", cpu_group)
    mori.shmem.shmem_torch_process_group_init("mori")

    exchange = MoonEPSymmetricHistogramExchange(
        rank=rank, world_size=world, num_experts=E, device=device
    )
    failures: list[str] = []

    for epoch in (1, 2):
        exchange.buffer.fill_(-1)
        torch.cuda.synchronize(device)
        ms.shmem_barrier_all()

        local = (
            torch.arange(E, dtype=torch.int32, device=device) + epoch * 1000 + rank * E
        )
        got = exchange.publish(local).clone()
        expected_rows = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(expected_rows, local)
        expected = torch.stack(expected_rows)
        if not torch.equal(got, expected):
            failures.append(f"epoch {epoch}: symmetric exchange != all_gather")

    exchange.close()
    dist.barrier()
    if rank == 0:
        print(
            "MOONEP_PREFILL_HISTOGRAM "
            + ("PASS" if not failures else "FAIL " + "; ".join(failures)),
            flush=True,
        )
    dist.destroy_process_group()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
