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
    CustomGroupConfig,
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
    custom_group_config,
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
        custom_group_config=custom_group_config,
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
    config = {"default": {"tp_group": custom_tp}}

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
                    config,
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
    config = {"default": {"dp_group": custom_dp}}

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
                    config,
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
    config = {"default": {"tp_group": custom_tp, "dp_group": custom_dp}}

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
                    config,
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


# ============================================================
# Test 4: two-phase named custom groups (attention tp4dp2 + comm tp8)
#   Both groups initialized upfront via CustomGroupConfig, selected
#   by name at runtime — no destroy/reinit between phases.
#   "attn" group: tp4dp2 → derived EP = all 8 ranks
#   "comm" group: tp8 = all 8 ranks
# ============================================================
def allreduce_two_phase(
    rankID,
    deviceID,
    x_phase1,
    x_phase2,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
):
    device = torch.device(f"cuda:{deviceID}")
    torch.cuda.set_device(device)
    world_size = 8

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
        local_rank=deviceID,
    )

    # Initialize both custom groups at once using CustomGroupConfig
    config = CustomGroupConfig()
    config.add_group(
        "attn",
        tp_group=[[0, 1, 2, 3], [4, 5, 6, 7]],
        dp_group=[[0, 4], [1, 5], [2, 6], [3, 7]],
    )
    config.add_group(
        "comm",
        tp_group=list(range(world_size)),
    )
    ensure_model_parallel_initialized(
        world_size,
        1,
        custom_group_config=config.data(),
    )

    # warmup all custom groups
    attn_group = get_custom_group("attn")
    comm_group = get_custom_group("comm")
    dist.all_reduce(torch.zeros(1, device=device), group=attn_group.device_group)
    dist.all_reduce(torch.zeros(1, device=device), group=comm_group.device_group)
    torch.cuda.synchronize()

    inp1 = x_phase1.to(device)
    inp2 = x_phase2.to(device)

    if withGraph:
        # Phase 1: attention allreduce
        graph1 = torch.cuda.CUDAGraph()
        with attn_group.graph_capture() as gc:
            with torch.cuda.graph(graph1, stream=gc.stream):
                out1 = custom_all_reduce(inp1, group="attn")
        out1.fill_(0)

        @perftest()
        def run1():
            graph1.replay()

        _, us1 = run1()
        result1 = (out1, us1)

        # Phase 2: communication allreduce
        graph2 = torch.cuda.CUDAGraph()
        with comm_group.graph_capture() as gc:
            with torch.cuda.graph(graph2, stream=gc.stream):
                out2 = custom_all_reduce(inp2, group="comm")
        out2.fill_(0)

        @perftest()
        def run2():
            graph2.replay()

        _, us2 = run2()
        result2 = (out2, us2)
    else:

        @perftest()
        def run1(x):
            return custom_all_reduce(x, group="attn")

        result1 = run1(inp1)

        @perftest()
        def run2(x):
            return custom_all_reduce(x, group="comm")

        result2 = run2(inp2)

    # destroy once
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    return result1, result2


@benchmark()
def test_two_phase_tp4dp2_then_tp8(
    shape,
    dtype,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    world_size = 8

    pool = Pool(processes=world_size)
    xs_phase1 = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
    xs_phase2 = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]

    # Phase 1 ref: attn group (derived EP = all 8 ranks)
    ref_phase1 = torch.zeros(shape, dtype=dtype)
    for x in xs_phase1:
        ref_phase1 += x

    # Phase 2 ref: comm group (tp8 = all 8 ranks)
    ref_phase2 = torch.zeros(shape, dtype=dtype)
    for x in xs_phase2:
        ref_phase2 += x

    rets = []
    for i in range(world_size):
        rets.append(
            pool.apply_async(
                allreduce_two_phase,
                args=(
                    i,
                    i,
                    xs_phase1[i],
                    xs_phase2[i],
                    withGraph,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]

    for result1, result2 in rets:
        out1, us1 = result1
        out2, us2 = result2
        msg1 = (
            f"test_two_phase (attn tp4dp2->ep8): {shape=} {dtype=} "
            f"{withGraph=} {us1:>8.2f}"
        )
        checkAllclose(ref_phase1, out1.to(ref_phase1), msg=msg1)
        msg2 = (
            f"test_two_phase (comm tp8): {shape=} {dtype=} " f"{withGraph=} {us2:>8.2f}"
        )
        checkAllclose(ref_phase2, out2.to(ref_phase2), msg=msg2)


if __name__ == "__main__":
    freeze_support()
    shape = (128, 8192)
    dtype = torch.bfloat16

    for withGraph in [True, False]:
        for test_fn in [
            test_custom_tp,
            test_custom_dp,
            test_custom_ep,
            test_two_phase_tp4dp2_then_tp8,
        ]:
            test_fn(
                shape,
                dtype,
                withGraph=withGraph,
                distributed_init_method=get_distributed_init_method(
                    get_ip(), get_open_port()
                ),
            )
