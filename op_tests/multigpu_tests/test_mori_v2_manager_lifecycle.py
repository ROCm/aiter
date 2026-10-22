# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multi-rank lifecycle coverage for the MORI v2 all-to-all manager.

Run directly, as the multi-GPU CI harness does::

    python3 op_tests/multigpu_tests/test_mori_v2_manager_lifecycle.py
"""

import importlib.util
import os
import socket
import threading

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aiter.dist.device_communicators.all2all import MoriV2All2AllManager
from aiter.dist.device_communicators.base_device_communicator import (
    All2AllManagerBase,
)

WORLD_SIZE = 2
PER_RANK_VMM = 1 << 29


class FakeHandle:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


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

        # Inject a one-rank handle-construction failure after a real CCO
        # communicator has been initialized. Every rank must observe the
        # failure and release its local handle/communicator before retrying.
        partial_handles = []

        def partial_failure(comm, **kwargs):
            if rank == 0:
                raise RuntimeError("injected rank-0 handle failure")
            handle = FakeHandle()
            partial_handles.append(handle)
            return handle

        manager._make_handle = partial_failure
        try:
            manager.get_handle(
                {
                    "per_rank_vmm": PER_RANK_VMM,
                    "hidden_dim": 1024,
                }
            )
        except RuntimeError as exc:
            assert "handle creation failed across the EP group" in str(exc)
            assert "injected rank-0 handle failure" in str(exc)
        else:
            raise AssertionError("partial handle failure was not propagated")

        assert manager._cco_communicator is None
        assert manager._cco_per_rank_vmm == 0
        assert all(handle.closed for handle in partial_handles)
        assert not manager._destroyed
        dist.barrier()

        # Hold handle creation inside the lifecycle lock while destroy races on
        # another thread. Destroy must wait, then close the registered handle
        # before invalidating the real CCO communicator.
        handle_started = threading.Event()
        release_handle = threading.Event()
        destroy_finished = threading.Event()
        created_handles = []
        create_errors = []

        def blocking_handle(comm, **kwargs):
            handle_started.set()
            if not release_handle.wait(timeout=120):
                raise RuntimeError("timed out waiting to finish handle creation")
            handle = FakeHandle()
            created_handles.append(handle)
            return handle

        manager._make_handle = blocking_handle

        def create_handle():
            try:
                manager.get_handle(
                    {
                        "per_rank_vmm": PER_RANK_VMM,
                        "hidden_dim": 1024,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                create_errors.append(exc)

        def destroy_manager():
            manager.destroy()
            destroy_finished.set()

        create_thread = threading.Thread(target=create_handle)
        create_thread.start()
        assert handle_started.wait(timeout=120)
        dist.barrier()

        comm = manager._cco_communicator
        assert comm is not None and comm.is_valid
        destroy_thread = threading.Thread(target=destroy_manager)
        destroy_thread.start()
        assert not destroy_finished.wait(timeout=0.2)
        dist.barrier()

        release_handle.set()
        create_thread.join(timeout=120)
        destroy_thread.join(timeout=120)
        assert not create_thread.is_alive()
        assert not destroy_thread.is_alive()
        assert not create_errors
        assert len(created_handles) == 1
        assert created_handles[0].closed
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
