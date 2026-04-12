# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method
from typing_extensions import Optional

import torch
import torch.distributed as dist

from aiter.dist.communication_op import custom_all_reduce
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_custom_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.test_common import benchmark, checkAllclose, perftest

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


def allreduce_custom_group(
    tp_size,
    dp_size,
    rankID,
    deviceID,
    x,
    custom_tp_group_ranks,
    custom_dp_group_ranks,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
):
    device = torch.device(f"cuda:{deviceID}")
    torch.cuda.set_device(device)
    world_size = tp_size * dp_size
    logger.info(
        f"RANK: {rankID} device: {deviceID} "
        f"custom group allreduce init_process_group..."
    )
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
        local_rank=deviceID,
    )
    ensure_model_parallel_initialized(
        tp_size,
        1,
        data_parallel_size=dp_size,
        custom_tp_group_ranks=custom_tp_group_ranks,
        custom_dp_group_ranks=custom_dp_group_ranks,
    )
    x = x.to(device)

    # warmup and align all gpu
    custom_group = get_custom_group()
    dist.all_reduce(torch.zeros(1, device=device), group=custom_group.device_group)
    torch.cuda.synchronize()

    if withGraph:
        graph = torch.cuda.CUDAGraph()
        with custom_group.graph_capture() as gc:
            with torch.cuda.graph(graph, stream=gc.stream):
                out = custom_all_reduce(x)
        out.fill_(0)

        @perftest()
        def run():
            graph.replay()

        _, us = run()
        out = (out, us)
    else:

        @perftest()
        def run(x):
            return custom_all_reduce(x)

        out = run(x)

    # destroy
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


# ============================================================
# Test 1: custom TP group on GPUs [0,2,4,6]
# ============================================================
@benchmark()
def test_custom_tp(
    shape,
    dtype,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    device_ids = [0, 2, 4, 6]
    world_size = len(device_ids)  # 4
    tp_size = world_size
    dp_size = 1
    # 1D list: all ranks in one custom TP group
    custom_tp = list(range(world_size))  # [0,1,2,3]

    pool = Pool(processes=world_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    for i in range(world_size):
        x = torch.randn(shape, dtype=dtype)
        ref += x
        rets.append(
            pool.apply_async(
                allreduce_custom_group,
                args=(
                    tp_size,
                    dp_size,
                    i,
                    device_ids[i],
                    x,
                    custom_tp,
                    None,
                    withGraph,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]
    for out, us in rets:
        msg = (
            f"test_custom_tp: GPUs={device_ids} {shape=} {dtype=} "
            f"{withGraph=} {us:>8.2f}"
        )
        checkAllclose(ref, out.to(ref), msg=msg)


# ============================================================
# Test 2: custom DP group on GPUs [1,3,5,7]
# ============================================================
@benchmark()
def test_custom_dp(
    shape,
    dtype,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    device_ids = [1, 3, 5, 7]
    world_size = len(device_ids)  # 4
    tp_size = 1
    dp_size = world_size

    custom_dp = list(range(world_size))  # [0,1,2,3]

    pool = Pool(processes=world_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    for i in range(world_size):
        x = torch.randn(shape, dtype=dtype)
        ref += x
        rets.append(
            pool.apply_async(
                allreduce_custom_group,
                args=(
                    tp_size,
                    dp_size,
                    i,
                    device_ids[i],
                    x,
                    None,
                    custom_dp,
                    withGraph,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]
    for out, us in rets:
        msg = (
            f"test_custom_dp: GPUs={device_ids} {shape=} {dtype=} "
            f"{withGraph=} {us:>8.2f}"
        )
        checkAllclose(ref, out.to(ref), msg=msg)


# ============================================================
# Test 3: custom EP (derived from TP + DP)
#   tp: [[0,1,2,3],[4,5,6,7]]  dp: [[0,4],[1,5],[2,6],[3,7]]
#   => ep: [[0,1,2,3,4,5,6,7]]
# ============================================================
@benchmark()
def test_custom_ep(
    shape,
    dtype,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    world_size = 8
    tp_size = 4
    dp_size = 2
    custom_tp = [[0, 1, 2, 3], [4, 5, 6, 7]]
    custom_dp = [[0, 4], [1, 5], [2, 6], [3, 7]]
    # derived EP group: [[0,1,2,3,4,5,6,7]] — all ranks in one group

    pool = Pool(processes=world_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    for i in range(world_size):
        x = torch.randn(shape, dtype=dtype)
        ref += x
        rets.append(
            pool.apply_async(
                allreduce_custom_group,
                args=(
                    tp_size,
                    dp_size,
                    i,
                    i,
                    x,
                    custom_tp,
                    custom_dp,
                    withGraph,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]
    for out, us in rets:
        msg = (
            f"test_custom_ep: tp={custom_tp} dp={custom_dp} {shape=} "
            f"{dtype=} {withGraph=} {us:>8.2f}"
        )
        checkAllclose(ref, out.to(ref), msg=msg)


if __name__ == "__main__":
    freeze_support()
    shape = (128, 8192)
    dtype = torch.bfloat16

    for withGraph in [True, False]:
        for test_fn in [test_custom_tp, test_custom_dp, test_custom_ep]:
            test_fn(
                shape,
                dtype,
                withGraph=withGraph,
                distributed_init_method=get_distributed_init_method(
                    get_ip(), get_open_port()
                ),
            )
