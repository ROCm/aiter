# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""init_dist_env(skip_mori_shmem_init=...) must reach the EP group's
MoriAll2AllManager, and only it.

With the flag set, the manager must not call
mori.shmem.shmem_torch_process_group_init: on an EP group that spans nodes with
no RDMA NIC, that init asserts "no transport available for peer". The default
(False) must keep initializing the heap, which mori's get_handle ops and
MegaMoEV2 rely on.

This runs on one node, so it checks the plumbing, not the cross-node failure
itself: shmem init is replaced by a recorder, and the test asserts whether it
was called.
"""

import argparse
from multiprocessing import Pool, freeze_support, set_start_method

import torch

set_start_method("spawn", force=True)


def _worker(world_size, rankID, skip, port):
    import mori

    from aiter.dist.parallel_state import get_ep_group
    from aiter.dist.utils import get_distributed_init_method
    from aiter.ops.communication import destroy_dist_env, init_dist_env

    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)

    calls = []
    real_init = mori.shmem.shmem_torch_process_group_init

    def recording_init(group_name):
        calls.append(group_name)
        return real_init(group_name)

    mori.shmem.shmem_torch_process_group_init = recording_init
    try:
        # DP x TP1: the EP group spans all ranks, as under DP-attention.
        init_dist_env(
            tensor_model_parallel_size=1,
            rankID=0,
            backend="nccl",
            distributed_init_method=get_distributed_init_method("127.0.0.1", port),
            local_rank=rankID,
            data_parallel_size=world_size,
            data_parallel_rank=rankID,
            skip_mori_shmem_init=skip,
        )
        ep_comm = get_ep_group().device_communicator
        # all2all_manager is built lazily, on first access.
        manager = ep_comm.all2all_manager
        ret = {
            "manager": type(manager).__name__,
            "ep_flag": ep_comm.skip_mori_shmem_init,
            "manager_flag": manager.skip_shmem_init,
            "shmem_init_calls": len(calls),
        }
    finally:
        mori.shmem.shmem_torch_process_group_init = real_init
        destroy_dist_env()
    return ret


def test_mori_skip_shmem_init(world_size, skip, port):
    pool = Pool(processes=world_size)
    rets = [
        pool.apply_async(_worker, args=(world_size, i, skip, port))
        for i in range(world_size)
    ]
    pool.close()
    pool.join()
    rets = [r.get() for r in rets]

    for rank, ret in enumerate(rets):
        assert ret["manager"] == "MoriAll2AllManager", f"rank {rank}: {ret}"
        assert ret["ep_flag"] is skip, f"rank {rank}: {ret}"
        assert ret["manager_flag"] is skip, f"rank {rank}: {ret}"
        expected_calls = 0 if skip else 1
        assert ret["shmem_init_calls"] == expected_calls, (
            f"rank {rank}: skip_mori_shmem_init={skip} but shmem init ran "
            f"{ret['shmem_init_calls']}x: {ret}"
        )
    return rets[0]


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="config input of test")
    parser.add_argument("-w", "--world_size", type=int, default=2)
    args = parser.parse_args()

    for i, skip in enumerate((False, True)):
        ret = test_mori_skip_shmem_init(args.world_size, skip, 49390 + i)
        print(f"skip_mori_shmem_init={skip}: {ret}")
