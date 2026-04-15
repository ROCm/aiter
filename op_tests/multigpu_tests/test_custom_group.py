# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method
from typing_extensions import Optional

import pandas as pd
import torch
import torch.distributed as dist

from aiter.dist.communication_op import (
    custom_all_gather,
    custom_all_reduce,
    custom_reduce_scatter,
)
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


# ============================================================
# Worker: single custom group — runs allreduce, allgather, reduce_scatter
# ============================================================
def custom_group_worker(
    tp_size,
    dp_size,
    rankID,
    deviceID,
    x_ar,
    x_ag,
    x_rs,
    custom_group_config,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
):
    device = torch.device(f"cuda:{deviceID}")
    torch.cuda.set_device(device)
    world_size = tp_size * dp_size
    logger.info(
        f"RANK: {rankID} device: {deviceID} "
        f"custom group worker init_process_group..."
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
    x_ar = x_ar.to(device)
    x_ag = x_ag.to(device)
    x_rs = x_rs.to(device)

    # warmup and align all gpu
    custom_group = get_custom_group()
    dist.all_reduce(torch.zeros(1, device=device), group=custom_group.device_group)
    torch.cuda.synchronize()

    if withGraph:
        graph = torch.cuda.CUDAGraph()
        with custom_group.graph_capture() as gc:
            with torch.cuda.graph(graph, stream=gc.stream):
                out_ar = custom_all_reduce(x_ar)
                out_ag = custom_all_gather(x_ag)
                out_rs = custom_reduce_scatter(x_rs)
        out_ar.fill_(0)
        out_ag.fill_(0)
        out_rs.fill_(0)

        @perftest()
        def run():
            graph.replay()

        _, us = run()
        results = (out_ar, out_ag, out_rs, us)
    else:

        @perftest()
        def run(x_ar, x_ag, x_rs):
            o_ar = custom_all_reduce(x_ar)
            o_ag = custom_all_gather(x_ag)
            o_rs = custom_reduce_scatter(x_rs)
            return o_ar, o_ag, o_rs

        (out_ar, out_ag, out_rs), us = run(x_ar, x_ag, x_rs)
        results = (out_ar, out_ag, out_rs, us)

    # destroy
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return results


# ============================================================
# Worker: two-phase named groups — runs all 3 ops on each group
# ============================================================
def two_phase_worker(
    rankID,
    deviceID,
    x_attn_ar,
    x_attn_ag,
    x_attn_rs,
    x_comm_ar,
    x_comm_ag,
    x_comm_rs,
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

    inp_attn_ar = x_attn_ar.to(device)
    inp_attn_ag = x_attn_ag.to(device)
    inp_attn_rs = x_attn_rs.to(device)
    inp_comm_ar = x_comm_ar.to(device)
    inp_comm_ag = x_comm_ag.to(device)
    inp_comm_rs = x_comm_rs.to(device)

    if withGraph:
        # Phase 1: attn group
        graph1 = torch.cuda.CUDAGraph()
        with attn_group.graph_capture() as gc:
            with torch.cuda.graph(graph1, stream=gc.stream):
                out_attn_ar = custom_all_reduce(inp_attn_ar, group="attn")
                out_attn_ag = custom_all_gather(inp_attn_ag, group="attn")
                out_attn_rs = custom_reduce_scatter(inp_attn_rs, group="attn")
        out_attn_ar.fill_(0)
        out_attn_ag.fill_(0)
        out_attn_rs.fill_(0)

        @perftest()
        def run1():
            graph1.replay()

        _, us1 = run1()

        # Phase 2: comm group
        graph2 = torch.cuda.CUDAGraph()
        with comm_group.graph_capture() as gc:
            with torch.cuda.graph(graph2, stream=gc.stream):
                out_comm_ar = custom_all_reduce(inp_comm_ar, group="comm")
                out_comm_ag = custom_all_gather(inp_comm_ag, group="comm")
                out_comm_rs = custom_reduce_scatter(inp_comm_rs, group="comm")
        out_comm_ar.fill_(0)
        out_comm_ag.fill_(0)
        out_comm_rs.fill_(0)

        @perftest()
        def run2():
            graph2.replay()

        _, us2 = run2()
    else:

        @perftest()
        def run1(ar, ag, rs):
            o_ar = custom_all_reduce(ar, group="attn")
            o_ag = custom_all_gather(ag, group="attn")
            o_rs = custom_reduce_scatter(rs, group="attn")
            return o_ar, o_ag, o_rs

        (out_attn_ar, out_attn_ag, out_attn_rs), us1 = run1(
            inp_attn_ar, inp_attn_ag, inp_attn_rs
        )

        @perftest()
        def run2(ar, ag, rs):
            o_ar = custom_all_reduce(ar, group="comm")
            o_ag = custom_all_gather(ag, group="comm")
            o_rs = custom_reduce_scatter(rs, group="comm")
            return o_ar, o_ag, o_rs

        (out_comm_ar, out_comm_ag, out_comm_rs), us2 = run2(
            inp_comm_ar, inp_comm_ag, inp_comm_rs
        )

    # destroy once
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    return (
        out_attn_ar,
        out_attn_ag,
        out_attn_rs,
        us1,
        out_comm_ar,
        out_comm_ag,
        out_comm_rs,
        us2,
    )


# ============================================================
# Helper: compute references for allreduce, allgather, reduce_scatter
# ============================================================
def compute_refs(xs_ar, xs_ag, xs_rs, world_size):
    """Compute reference outputs for all 3 ops.

    Returns:
        ref_ar: allreduce reference (sum of all inputs)
        ref_ag: allgather reference (concat along dim 0)
        chunks_rs: list of reduce_scatter references (one per rank)
    """
    ref_ar = torch.zeros_like(xs_ar[0])
    for x in xs_ar:
        ref_ar += x

    ref_ag = torch.cat(xs_ag, dim=0)

    ref_rs_sum = torch.zeros_like(xs_rs[0])
    for x in xs_rs:
        ref_rs_sum += x
    chunks_rs = list(ref_rs_sum.chunk(world_size, dim=0))

    return ref_ar, ref_ag, chunks_rs


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
    world_size = len(device_ids)
    tp_size = world_size
    dp_size = 1
    custom_tp = list(range(world_size))
    config = {"default": {"tp_group": custom_tp}}

    pool = Pool(processes=world_size)
    xs_ar = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
    xs_ag = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
    xs_rs = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
    ref_ar, ref_ag, chunks_rs = compute_refs(xs_ar, xs_ag, xs_rs, world_size)

    rets = []
    for i in range(world_size):
        rets.append(
            pool.apply_async(
                custom_group_worker,
                args=(
                    tp_size,
                    dp_size,
                    i,
                    device_ids[i],
                    xs_ar[i],
                    xs_ag[i],
                    xs_rs[i],
                    config,
                    withGraph,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]
    all_us = [r[3] for r in rets]
    max_err = 0.0
    for i, (out_ar, out_ag, out_rs, us) in enumerate(rets):
        tag = f"test_custom_tp: GPUs={device_ids} {shape=} {dtype=} {withGraph=} {us:>8.2f}"
        max_err = max(
            max_err, checkAllclose(ref_ar, out_ar.to(ref_ar), msg=f"{tag} allreduce")
        )
        max_err = max(
            max_err, checkAllclose(ref_ag, out_ag.to(ref_ag), msg=f"{tag} allgather")
        )
        max_err = max(
            max_err,
            checkAllclose(
                chunks_rs[i], out_rs.to(chunks_rs[i]), msg=f"{tag} reduce_scatter"
            ),
        )
    return {
        "test": "custom_tp",
        "min_us": min(all_us),
        "max_us": max(all_us),
        "err": max_err,
    }


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
    world_size = len(device_ids)
    tp_size = 1
    dp_size = world_size
    custom_dp = list(range(world_size))
    config = {"default": {"dp_group": custom_dp}}

    pool = Pool(processes=world_size)
    xs_ar = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
    xs_ag = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
    xs_rs = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
    ref_ar, ref_ag, chunks_rs = compute_refs(xs_ar, xs_ag, xs_rs, world_size)

    rets = []
    for i in range(world_size):
        rets.append(
            pool.apply_async(
                custom_group_worker,
                args=(
                    tp_size,
                    dp_size,
                    i,
                    device_ids[i],
                    xs_ar[i],
                    xs_ag[i],
                    xs_rs[i],
                    config,
                    withGraph,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]
    all_us = [r[3] for r in rets]
    max_err = 0.0
    for i, (out_ar, out_ag, out_rs, us) in enumerate(rets):
        tag = f"test_custom_dp: GPUs={device_ids} {shape=} {dtype=} {withGraph=} {us:>8.2f}"
        max_err = max(
            max_err, checkAllclose(ref_ar, out_ar.to(ref_ar), msg=f"{tag} allreduce")
        )
        max_err = max(
            max_err, checkAllclose(ref_ag, out_ag.to(ref_ag), msg=f"{tag} allgather")
        )
        max_err = max(
            max_err,
            checkAllclose(
                chunks_rs[i], out_rs.to(chunks_rs[i]), msg=f"{tag} reduce_scatter"
            ),
        )
    return {
        "test": "custom_dp",
        "min_us": min(all_us),
        "max_us": max(all_us),
        "err": max_err,
    }


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
    config = {"default": {"tp_group": custom_tp, "dp_group": custom_dp}}

    pool = Pool(processes=world_size)
    xs_ar = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
    xs_ag = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
    xs_rs = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
    ref_ar, ref_ag, chunks_rs = compute_refs(xs_ar, xs_ag, xs_rs, world_size)

    rets = []
    for i in range(world_size):
        rets.append(
            pool.apply_async(
                custom_group_worker,
                args=(
                    tp_size,
                    dp_size,
                    i,
                    i,
                    xs_ar[i],
                    xs_ag[i],
                    xs_rs[i],
                    config,
                    withGraph,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]
    all_us = [r[3] for r in rets]
    max_err = 0.0
    for i, (out_ar, out_ag, out_rs, us) in enumerate(rets):
        tag = (
            f"test_custom_ep: tp={custom_tp} dp={custom_dp} "
            f"{shape=} {dtype=} {withGraph=} {us:>8.2f}"
        )
        max_err = max(
            max_err, checkAllclose(ref_ar, out_ar.to(ref_ar), msg=f"{tag} allreduce")
        )
        max_err = max(
            max_err, checkAllclose(ref_ag, out_ag.to(ref_ag), msg=f"{tag} allgather")
        )
        max_err = max(
            max_err,
            checkAllclose(
                chunks_rs[i], out_rs.to(chunks_rs[i]), msg=f"{tag} reduce_scatter"
            ),
        )
    return {
        "test": "custom_ep",
        "min_us": min(all_us),
        "max_us": max(all_us),
        "err": max_err,
    }


# ============================================================
# Test 4: two-phase named custom groups (attn tp4dp2 + comm tp8)
#   Both groups initialized upfront via CustomGroupConfig, selected
#   by name at runtime — no destroy/reinit between phases.
#   Each group runs allreduce, allgather, reduce_scatter.
# ============================================================
@benchmark()
def test_two_phase(
    shape,
    dtype,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    world_size = 8

    pool = Pool(processes=world_size)
    # attn group inputs
    xs_attn_ar = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
    xs_attn_ag = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
    xs_attn_rs = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
    # comm group inputs
    xs_comm_ar = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
    xs_comm_ag = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
    xs_comm_rs = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]

    # references
    ref_attn_ar, ref_attn_ag, chunks_attn_rs = compute_refs(
        xs_attn_ar, xs_attn_ag, xs_attn_rs, world_size
    )
    ref_comm_ar, ref_comm_ag, chunks_comm_rs = compute_refs(
        xs_comm_ar, xs_comm_ag, xs_comm_rs, world_size
    )

    rets = []
    for i in range(world_size):
        rets.append(
            pool.apply_async(
                two_phase_worker,
                args=(
                    i,
                    i,
                    xs_attn_ar[i],
                    xs_attn_ag[i],
                    xs_attn_rs[i],
                    xs_comm_ar[i],
                    xs_comm_ag[i],
                    xs_comm_rs[i],
                    withGraph,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]

    all_us1 = []
    all_us2 = []
    max_err = 0.0
    for i, (
        out_attn_ar,
        out_attn_ag,
        out_attn_rs,
        us1,
        out_comm_ar,
        out_comm_ag,
        out_comm_rs,
        us2,
    ) in enumerate(rets):
        all_us1.append(us1)
        all_us2.append(us2)
        tag1 = (
            f"test_two_phase (attn tp4dp2->ep8): "
            f"{shape=} {dtype=} {withGraph=} {us1:>8.2f}"
        )
        max_err = max(
            max_err,
            checkAllclose(
                ref_attn_ar, out_attn_ar.to(ref_attn_ar), msg=f"{tag1} allreduce"
            ),
        )
        max_err = max(
            max_err,
            checkAllclose(
                ref_attn_ag, out_attn_ag.to(ref_attn_ag), msg=f"{tag1} allgather"
            ),
        )
        max_err = max(
            max_err,
            checkAllclose(
                chunks_attn_rs[i],
                out_attn_rs.to(chunks_attn_rs[i]),
                msg=f"{tag1} reduce_scatter",
            ),
        )
        tag2 = (
            f"test_two_phase (comm tp8): " f"{shape=} {dtype=} {withGraph=} {us2:>8.2f}"
        )
        max_err = max(
            max_err,
            checkAllclose(
                ref_comm_ar, out_comm_ar.to(ref_comm_ar), msg=f"{tag2} allreduce"
            ),
        )
        max_err = max(
            max_err,
            checkAllclose(
                ref_comm_ag, out_comm_ag.to(ref_comm_ag), msg=f"{tag2} allgather"
            ),
        )
        max_err = max(
            max_err,
            checkAllclose(
                chunks_comm_rs[i],
                out_comm_rs.to(chunks_comm_rs[i]),
                msg=f"{tag2} reduce_scatter",
            ),
        )
    return {
        "test": "two_phase",
        "attn_min_us": min(all_us1),
        "attn_max_us": max(all_us1),
        "comm_min_us": min(all_us2),
        "comm_max_us": max(all_us2),
        "err": max_err,
    }


if __name__ == "__main__":
    freeze_support()
    shape = (128, 8192)
    dtype = torch.bfloat16

    df = []
    for withGraph in [True, False]:
        for test_fn in [
            test_custom_tp,
            test_custom_dp,
            test_custom_ep,
            test_two_phase,
        ]:
            ret = test_fn(
                shape,
                dtype,
                withGraph=withGraph,
                distributed_init_method=get_distributed_init_method(
                    get_ip(), get_open_port()
                ),
            )
            df.append(ret)
    df = pd.DataFrame(df)
    show_cols = [
        "test",
        "shape",
        "dtype",
        "withGraph",
        "min_us",
        "max_us",
        "attn_min_us",
        "attn_max_us",
        "comm_min_us",
        "comm_max_us",
        "err",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    logger.info(
        "custom group comm ops summary (markdown):\n%s",
        df[show_cols].to_markdown(index=False),
    )
