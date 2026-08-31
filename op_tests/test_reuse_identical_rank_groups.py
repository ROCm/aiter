# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""2-rank test for comm-group reuse (AITER_REUSE_IDENTICAL_COMM_GROUPS).

When identical-rank parallel groups span the same ranks, they should share one set
of process groups and communicators while staying distinct GroupCoordinator
objects with correct unique_names -- so an EP group keeps an "ep"-named
device_communicator and all2all still initializes, but no second communicator
set is allocated.

Run: torchrun --nproc_per_node=2 op_tests/test_reuse_identical_rank_groups.py
 or: python op_tests/test_reuse_identical_rank_groups.py   (uses mp.spawn)
Needs 2 GPUs (device communicators require CUDA/NCCL).
"""

import os

import torch
import torch.multiprocessing as mp

from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_dcp_group,
    get_ep_group,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)


def _init(rank, world_size, port, reuse):
    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        local_rank=rank,
        backend="nccl",
    )
    # TP, DCP and EP all span the same ranks here, so DCP and EP reuse TP when
    # reuse=True. Reuse is gated by the env var (read inside
    # initialize_model_parallel); set it explicitly so the reuse=False leg does
    # not inherit a "1" left over from the reuse=True leg in the same process.
    os.environ["AITER_REUSE_IDENTICAL_COMM_GROUPS"] = "1" if reuse else "0"
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        decode_context_model_parallel_size=world_size,
    )
    # Wire the custom-allreduce signal buffer exactly as init_dist_env does, so
    # the (shared) ca_comm is functional -- EP reuses this same ca object.
    ca_comm = get_tp_group().device_communicator.ca_comm
    if ca_comm is not None and not getattr(ca_comm, "_is_gfx1250", False):
        signal = torch.zeros(world_size * 64, dtype=torch.int64, device=rank)
        ca_comm.signal = signal
        ca_comm.register_input_buffer(signal)
        ca_comm.buffer = ca_comm._pool["input"].tensor


def _teardown():
    destroy_model_parallel()
    destroy_distributed_environment()
    torch.cuda.empty_cache()


def _check_reuse_true():
    tp = get_tp_group()
    ep = get_ep_group()
    dcp = get_dcp_group()

    # EP must be a distinct object with its own EP-named comm so
    # is_ep_communicator/use_all2all are True and all2all can initialize.
    assert ep is not tp, "EP must not alias TP"
    assert (
        "ep" in ep.unique_name
    ), f"EP unique_name should contain 'ep': {ep.unique_name}"
    assert (
        ep.device_communicator is not tp.device_communicator
    ), "EP must own its device_communicator (EP-named), not share TP's"
    assert ep.device_communicator.is_ep_communicator is True
    assert ep.device_communicator.use_all2all is True
    # ...but it reuses TP's allreduce handles (the point: no second NCCL comm).
    assert ep.device_communicator.pynccl_comm is tp.device_communicator.pynccl_comm
    assert ep.device_communicator.ca_comm is tp.device_communicator.ca_comm
    assert ep.device_communicator.qr_comm is tp.device_communicator.qr_comm
    # EP shares TP's underlying process groups (identity => no duplicate group).
    assert ep.device_group is tp.device_group
    assert ep.cpu_group is tp.cpu_group

    # DCP (non-EP) shares TP's device_communicator wholesale -- allreduce/gather
    # ignore unique_name, so this maximizes the saving.
    assert dcp is not tp, "DCP must be a distinct GroupCoordinator"
    assert dcp.device_group is tp.device_group, "DCP must share TP's device_group"
    assert dcp.device_communicator is tp.device_communicator

    # A functional collective over the reused comms must still work. Use pynccl
    # directly (reuse shares TP's pynccl handle) -- this validates the shared
    # communicator through EP's distinct coordinator without depending on the
    # custom-allreduce signal-buffer wiring.
    dev = f"cuda:{tp.local_rank}"
    pynccl = ep.device_communicator.pynccl_comm
    t = torch.ones(8, device=dev)
    out = pynccl.all_reduce(t)
    torch.cuda.synchronize()
    assert torch.allclose(
        out, torch.full_like(out, float(tp.world_size))
    ), f"reused pynccl all_reduce gave {out[0].item()}, expected {tp.world_size}"


def _check_reuse_false():
    tp = get_tp_group()
    ep = get_ep_group()
    dcp = get_dcp_group()
    # Baseline: every identical-rank group gets its own communicators.
    assert ep is not tp
    assert ep.device_communicator is not tp.device_communicator
    assert ep.device_communicator.pynccl_comm is not tp.device_communicator.pynccl_comm
    assert (
        dcp.device_group is not tp.device_group
    ), "with reuse off, DCP must allocate its own process group"
    # EP still carries an EP-named communicator regardless of reuse.
    assert ep.device_communicator.is_ep_communicator is True


def _worker(rank, world_size, port):
    try:
        _init(rank, world_size, port, reuse=True)
        _check_reuse_true()
        _teardown()

        _init(rank, world_size, port + 1, reuse=False)
        _check_reuse_false()
        _teardown()
        if rank == 0:
            print("test_reuse_identical_rank_groups: PASSED")
    except Exception:
        _teardown()
        raise


def main():
    world_size = 2
    if torch.cuda.device_count() < world_size:
        print(f"SKIP: need {world_size} GPUs, have {torch.cuda.device_count()}")
        return
    port = int(os.environ.get("MASTER_PORT", "29513"))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # launched under torchrun
        _worker(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), port)
    else:
        mp.spawn(_worker, args=(world_size, port), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
