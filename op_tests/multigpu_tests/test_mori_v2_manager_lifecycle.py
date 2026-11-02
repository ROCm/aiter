# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multi-rank lifecycle coverage for the MORI v2 all-to-all manager.

Run directly, as the multi-GPU CI harness does::

    python3 op_tests/multigpu_tests/test_mori_v2_manager_lifecycle.py
"""

import importlib.util
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aiter.dist.device_communicators.all2all import MoriV2All2AllManager
from aiter.dist.device_communicators.base_device_communicator import (
    All2AllManagerBase,
)

WORLD_SIZE = 2
PER_RANK_VMM = 1 << 29


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _has_module(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _run_rank(rank: int, world_size: int, port: int):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    original_base_init = All2AllManagerBase.__init__

    def init_base(self, cpu_group):
        self.cpu_group = cpu_group
        self.rank = rank
        self.world_size = world_size
        self.internode = False

    All2AllManagerBase.__init__ = init_base
    try:
        manager = MoriV2All2AllManager(dist.group.WORLD)

        # A rank-local capacity must not reach CCO initialization until every
        # participant has agreed on the same symmetric VMM reservation.
        try:
            manager.get_or_create_cco_communicator(PER_RANK_VMM + rank * 4096)
        except RuntimeError as exc:
            assert "per_rank_vmm must match" in str(exc)
        else:
            raise AssertionError("mismatched per_rank_vmm was accepted")
        assert manager._cco_communicator is None

        # Matching capacities create one real communicator per rank, and smaller
        # subsequent requests reuse it without another collective initialization.
        comm = manager.get_or_create_cco_communicator(PER_RANK_VMM)
        assert comm.is_valid
        assert manager.get_or_create_cco_communicator(PER_RANK_VMM // 2) is comm
        dist.barrier()

        manager.destroy()
        assert not comm.is_valid
        assert manager._destroyed
        assert manager._cco_communicator is None
        assert manager._cco_per_rank_vmm == 0

        try:
            manager.get_or_create_cco_communicator(PER_RANK_VMM)
        except RuntimeError as exc:
            assert "manager has been destroyed" in str(exc)
        else:
            raise AssertionError("destroyed manager recreated its communicator")

        dist.barrier()
    finally:
        All2AllManagerBase.__init__ = original_base_init
        dist.destroy_process_group()


def main():
    missing = [
        module
        for module in ("mori.cco", "mori.ops.dispatch_combine_v2")
        if not _has_module(module)
    ]
    if missing:
        print(f"Skipping MORI v2 lifecycle test: missing {', '.join(missing)}")
        return
    if torch.cuda.device_count() < WORLD_SIZE:
        print(
            "Skipping MORI v2 lifecycle test: "
            f"requires {WORLD_SIZE} GPUs, found {torch.cuda.device_count()}"
        )
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    mp.spawn(
        _run_rank,
        args=(WORLD_SIZE, _free_port()),
        nprocs=WORLD_SIZE,
        join=True,
    )
    print("MORI v2 manager multi-rank lifecycle: PASS")


if __name__ == "__main__":
    main()
