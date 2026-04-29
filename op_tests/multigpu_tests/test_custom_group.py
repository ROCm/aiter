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
# Returns per-op timing: (out_ar, us_ar, out_ag, us_ag, out_rs, us_rs)
# ============================================================
def custom_group_worker(
    world_size,
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
        # capture and time each op separately
        graph_ar = torch.cuda.CUDAGraph()
        with custom_group.graph_capture() as gc:
            with torch.cuda.graph(graph_ar, stream=gc.stream):
                out_ar = custom_all_reduce(x_ar)
        out_ar.fill_(0)

        graph_ag = torch.cuda.CUDAGraph()
        with custom_group.graph_capture() as gc:
            with torch.cuda.graph(graph_ag, stream=gc.stream):
                out_ag = custom_all_gather(x_ag)
        out_ag.fill_(0)

        graph_rs = torch.cuda.CUDAGraph()
        with custom_group.graph_capture() as gc:
            with torch.cuda.graph(graph_rs, stream=gc.stream):
                out_rs = custom_reduce_scatter(x_rs)
        out_rs.fill_(0)

        @perftest()
        def replay_ar():
            graph_ar.replay()

        @perftest()
        def replay_ag():
            graph_ag.replay()

        @perftest()
        def replay_rs():
            graph_rs.replay()

        _, us_ar = replay_ar()
        _, us_ag = replay_ag()
        _, us_rs = replay_rs()
    else:

        @perftest()
        def run_ar(x):
            return custom_all_reduce(x)

        @perftest()
        def run_ag(x):
            return custom_all_gather(x)

        @perftest()
        def run_rs(x):
            return custom_reduce_scatter(x)

        out_ar, us_ar = run_ar(x_ar)
        out_ag, us_ag = run_ag(x_ag)
        out_rs, us_rs = run_rs(x_rs)

    # move tensors to CPU before destroying distributed env,
    # otherwise CUDA IPC serialization fails with invalid argument
    out_ar = out_ar.cpu()
    out_ag = out_ag.cpu()
    out_rs = out_rs.cpu()
    results = (out_ar, us_ar, out_ag, us_ag, out_rs, us_rs)

    # destroy
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return results


# ============================================================
# Multi-group config:
#   oe:  [[0,4],[1,5],[2,6],[3,7]]  — 4 independent DP2 groups
#   att: [[0,1,2,3],[4,5,6,7]]     — 2 independent TP4 groups
#   ffn: [0,1,2,3,4,5,6,7]         — 1 TP8 group
# ============================================================
MULTI_GROUP_CONFIG = {
    "oe": [[0, 4], [1, 5], [2, 6], [3, 7]],
    "att": [[0, 1, 2, 3], [4, 5, 6, 7]],
    "ffn": [0, 1, 2, 3, 4, 5, 6, 7],
}
MULTI_GROUP_NAMES = list(MULTI_GROUP_CONFIG.keys())


# ============================================================
# Worker: multi-group — runs each op separately per group
# Returns dict: {gname: (out_ar, us_ar, out_ag, us_ag, out_rs, us_rs)}
# ============================================================
def multi_group_worker(
    rankID,
    deviceID,
    inputs,
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
    for gname, ranks in MULTI_GROUP_CONFIG.items():
        config.add_group(gname, ranks)
    ensure_model_parallel_initialized(
        1,
        1,
        custom_group_config=config.data(),
    )

    # warmup all groups
    for gname in MULTI_GROUP_NAMES:
        group = get_custom_group(gname)
        dist.all_reduce(torch.zeros(1, device=device), group=group.device_group)
    torch.cuda.synchronize()

    results = {}
    for gname in MULTI_GROUP_NAMES:
        group = get_custom_group(gname)
        x_ar, x_ag, x_rs = [t.to(device) for t in inputs[gname]]

        if withGraph:
            graph_ar = torch.cuda.CUDAGraph()
            with group.graph_capture() as gc:
                with torch.cuda.graph(graph_ar, stream=gc.stream):
                    out_ar = custom_all_reduce(x_ar, group=gname)
            out_ar.fill_(0)

            graph_ag = torch.cuda.CUDAGraph()
            with group.graph_capture() as gc:
                with torch.cuda.graph(graph_ag, stream=gc.stream):
                    out_ag = custom_all_gather(x_ag, group=gname)
            out_ag.fill_(0)

            graph_rs = torch.cuda.CUDAGraph()
            with group.graph_capture() as gc:
                with torch.cuda.graph(graph_rs, stream=gc.stream):
                    out_rs = custom_reduce_scatter(x_rs, group=gname)
            out_rs.fill_(0)

            @perftest()
            def replay_ar(g=graph_ar):
                g.replay()

            @perftest()
            def replay_ag(g=graph_ag):
                g.replay()

            @perftest()
            def replay_rs(g=graph_rs):
                g.replay()

            _, us_ar = replay_ar()
            _, us_ag = replay_ag()
            _, us_rs = replay_rs()
        else:

            @perftest()
            def run_ar(x, g=gname):
                return custom_all_reduce(x, group=g)

            @perftest()
            def run_ag(x, g=gname):
                return custom_all_gather(x, group=g)

            @perftest()
            def run_rs(x, g=gname):
                return custom_reduce_scatter(x, group=g)

            out_ar, us_ar = run_ar(x_ar)
            out_ag, us_ag = run_ag(x_ag)
            out_rs, us_rs = run_rs(x_rs)

        results[gname] = (out_ar.cpu(), us_ar, out_ag.cpu(), us_ag, out_rs.cpu(), us_rs)

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    return results


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
    config = {"default": custom_tp}

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
                    world_size,
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

    all_us_ar, all_us_ag, all_us_rs = [], [], []
    err_ar, err_ag, err_rs = 0.0, 0.0, 0.0
    for i, (out_ar, us_ar, out_ag, us_ag, out_rs, us_rs) in enumerate(rets):
        all_us_ar.append(us_ar)
        all_us_ag.append(us_ag)
        all_us_rs.append(us_rs)
        tag = f"test_custom_tp: GPUs={device_ids} {shape=} {dtype=} {withGraph=}"
        err_ar = max(
            err_ar, checkAllclose(ref_ar, out_ar.to(ref_ar), msg=f"{tag} allreduce")
        )
        err_ag = max(
            err_ag, checkAllclose(ref_ag, out_ag.to(ref_ag), msg=f"{tag} allgather")
        )
        err_rs = max(
            err_rs,
            checkAllclose(
                chunks_rs[i], out_rs.to(chunks_rs[i]), msg=f"{tag} reduce_scatter"
            ),
        )
    return {
        "test": "custom_tp",
        "ar_min_us": min(all_us_ar),
        "ar_max_us": max(all_us_ar),
        "ar_err": err_ar,
        "ag_min_us": min(all_us_ag),
        "ag_max_us": max(all_us_ag),
        "ag_err": err_ag,
        "rs_min_us": min(all_us_rs),
        "rs_max_us": max(all_us_rs),
        "rs_err": err_rs,
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
    config = {"default": custom_dp}

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
                    world_size,
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

    all_us_ar, all_us_ag, all_us_rs = [], [], []
    err_ar, err_ag, err_rs = 0.0, 0.0, 0.0
    for i, (out_ar, us_ar, out_ag, us_ag, out_rs, us_rs) in enumerate(rets):
        all_us_ar.append(us_ar)
        all_us_ag.append(us_ag)
        all_us_rs.append(us_rs)
        tag = f"test_custom_dp: GPUs={device_ids} {shape=} {dtype=} {withGraph=}"
        err_ar = max(
            err_ar, checkAllclose(ref_ar, out_ar.to(ref_ar), msg=f"{tag} allreduce")
        )
        err_ag = max(
            err_ag, checkAllclose(ref_ag, out_ag.to(ref_ag), msg=f"{tag} allgather")
        )
        err_rs = max(
            err_rs,
            checkAllclose(
                chunks_rs[i], out_rs.to(chunks_rs[i]), msg=f"{tag} reduce_scatter"
            ),
        )
    return {
        "test": "custom_dp",
        "ar_min_us": min(all_us_ar),
        "ar_max_us": max(all_us_ar),
        "ar_err": err_ar,
        "ag_min_us": min(all_us_ag),
        "ag_max_us": max(all_us_ag),
        "ag_err": err_ag,
        "rs_min_us": min(all_us_rs),
        "rs_max_us": max(all_us_rs),
        "rs_err": err_rs,
    }


# ============================================================
# Test 3: custom 2D subgroups (two independent TP4 groups)
#   [[0,1,2,3],[4,5,6,7]] → devices 0-3 form one TP4, devices 4-7 form another
# ============================================================
@benchmark()
def test_custom_2d(
    shape,
    dtype,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    world_size = 8
    tp_size = 1
    dp_size = 1
    subgroups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    config = {"default": subgroups}
    subgroup_size = len(subgroups[0])

    pool = Pool(processes=world_size)
    xs_ar = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
    xs_ag = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
    xs_rs = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]

    # compute references per subgroup
    refs = {}
    for sg in subgroups:
        sg_ar = [xs_ar[r] for r in sg]
        sg_ag = [xs_ag[r] for r in sg]
        sg_rs = [xs_rs[r] for r in sg]
        ref_ar, ref_ag, chunks_rs = compute_refs(sg_ar, sg_ag, sg_rs, subgroup_size)
        for local_idx, global_rank in enumerate(sg):
            refs[global_rank] = (ref_ar, ref_ag, chunks_rs[local_idx])

    rets = []
    for i in range(world_size):
        rets.append(
            pool.apply_async(
                custom_group_worker,
                args=(
                    world_size,
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

    all_us_ar, all_us_ag, all_us_rs = [], [], []
    err_ar, err_ag, err_rs = 0.0, 0.0, 0.0
    for i, (out_ar, us_ar, out_ag, us_ag, out_rs, us_rs) in enumerate(rets):
        all_us_ar.append(us_ar)
        all_us_ag.append(us_ag)
        all_us_rs.append(us_rs)
        ref_ar, ref_ag, ref_rs = refs[i]
        tag = f"test_custom_2d: subgroups={subgroups} {shape=} {dtype=} {withGraph=}"
        err_ar = max(
            err_ar, checkAllclose(ref_ar, out_ar.to(ref_ar), msg=f"{tag} allreduce")
        )
        err_ag = max(
            err_ag, checkAllclose(ref_ag, out_ag.to(ref_ag), msg=f"{tag} allgather")
        )
        err_rs = max(
            err_rs,
            checkAllclose(ref_rs, out_rs.to(ref_rs), msg=f"{tag} reduce_scatter"),
        )
    return {
        "test": "custom_2d",
        "ar_min_us": min(all_us_ar),
        "ar_max_us": max(all_us_ar),
        "ar_err": err_ar,
        "ag_min_us": min(all_us_ag),
        "ag_max_us": max(all_us_ag),
        "ag_err": err_ag,
        "rs_min_us": min(all_us_rs),
        "rs_max_us": max(all_us_rs),
        "rs_err": err_rs,
    }


# ============================================================
# Helper: normalize 1D group config to 2D for reference computation
# ============================================================
def normalize_group_config(cfg):
    """[0,1,2,3] → [[0,1,2,3]];  [[0,1],[2,3]] stays as-is."""
    if all(isinstance(r, int) for r in cfg):
        return [cfg]
    return cfg


# ============================================================
# Test 4: multi-group (oe dp2x4 + att tp4x2 + ffn tp8)
#   All groups initialized upfront via CustomGroupConfig, selected
#   by name at runtime — no destroy/reinit between phases.
#   oe:  [[0,4],[1,5],[2,6],[3,7]] → 4 independent DP2 groups
#   att: [[0,1,2,3],[4,5,6,7]]    → 2 independent TP4 groups
#   ffn: [0,1,2,3,4,5,6,7]        → 1 TP8 group
#   Each group runs allreduce, allgather, reduce_scatter.
# ============================================================
@benchmark()
def test_multi_group(
    shape,
    dtype,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    world_size = 8

    pool = Pool(processes=world_size)

    # generate inputs and compute references per group
    group_xs = {}  # {gname: (xs_ar[], xs_ag[], xs_rs[])}
    group_refs = {}  # {gname: {rank: (ref_ar, ref_ag, ref_rs)}}
    for gname, cfg in MULTI_GROUP_CONFIG.items():
        xs_ar = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
        xs_ag = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
        xs_rs = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]
        group_xs[gname] = (xs_ar, xs_ag, xs_rs)

        refs = {}
        for sg in normalize_group_config(cfg):
            sg_ar = [xs_ar[r] for r in sg]
            sg_ag = [xs_ag[r] for r in sg]
            sg_rs = [xs_rs[r] for r in sg]
            ref_ar, ref_ag, chunks_rs = compute_refs(sg_ar, sg_ag, sg_rs, len(sg))
            for local_idx, global_rank in enumerate(sg):
                refs[global_rank] = (ref_ar, ref_ag, chunks_rs[local_idx])
        group_refs[gname] = refs

    # launch workers
    rets = []
    for i in range(world_size):
        worker_inputs = {
            gname: (xs_ar[i], xs_ag[i], xs_rs[i])
            for gname, (xs_ar, xs_ag, xs_rs) in group_xs.items()
        }
        rets.append(
            pool.apply_async(
                multi_group_worker,
                args=(i, i, worker_inputs, withGraph, distributed_init_method),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]

    # collect per-group per-op timing and errors
    us_data = {gname: {"ar": [], "ag": [], "rs": []} for gname in MULTI_GROUP_NAMES}
    err_data = {gname: {"ar": 0.0, "ag": 0.0, "rs": 0.0} for gname in MULTI_GROUP_NAMES}

    for i, rank_results in enumerate(rets):
        tag = f"test_multi_group {shape=} {dtype=} {withGraph=}"
        for gname in MULTI_GROUP_NAMES:
            out_ar, u_ar, out_ag, u_ag, out_rs, u_rs = rank_results[gname]
            us_data[gname]["ar"].append(u_ar)
            us_data[gname]["ag"].append(u_ag)
            us_data[gname]["rs"].append(u_rs)

            ref_ar, ref_ag, ref_rs = group_refs[gname][i]
            err_data[gname]["ar"] = max(
                err_data[gname]["ar"],
                checkAllclose(
                    ref_ar,
                    out_ar.to(ref_ar),
                    msg=f"{tag} {gname} allreduce",
                ),
            )
            err_data[gname]["ag"] = max(
                err_data[gname]["ag"],
                checkAllclose(
                    ref_ag,
                    out_ag.to(ref_ag),
                    msg=f"{tag} {gname} allgather",
                ),
            )
            err_data[gname]["rs"] = max(
                err_data[gname]["rs"],
                checkAllclose(
                    ref_rs,
                    out_rs.to(ref_rs),
                    msg=f"{tag} {gname} reduce_scatter",
                ),
            )

    # build return dict: per-group per-op min/max/err
    ret = {"test": "multi_group"}
    for gname in MULTI_GROUP_NAMES:
        for op in ("ar", "ag", "rs"):
            ret[f"{gname}_{op}_min_us"] = min(us_data[gname][op])
            ret[f"{gname}_{op}_max_us"] = max(us_data[gname][op])
            ret[f"{gname}_{op}_err"] = err_data[gname][op]
    return ret


# ============================================================
# Helper: expand multi_group result into one row per group
# ============================================================
def expand_groups(ret):
    """Transform multi_group result dict into one row per group,
    each with columns: test, ar_min_us, ar_max_us, ar_err, ag_..., rs_..."""
    common = {k: v for k, v in ret.items() if k in ("shape", "dtype", "withGraph")}
    rows = []
    for gname in MULTI_GROUP_NAMES:
        row = dict(common)
        row["test"] = f"multi_group:{gname}"
        for op in ("ar", "ag", "rs"):
            row[f"{op}_min_us"] = ret[f"{gname}_{op}_min_us"]
            row[f"{op}_max_us"] = ret[f"{gname}_{op}_max_us"]
            row[f"{op}_err"] = ret[f"{gname}_{op}_err"]
        rows.append(row)
    return rows


if __name__ == "__main__":
    freeze_support()
    shape = (128, 8192)
    dtype = torch.bfloat16

    df = []
    for withGraph in [True, False]:
        for test_fn in [
            test_custom_tp,
            test_custom_dp,
            test_custom_2d,
            test_multi_group,
        ]:
            ret = test_fn(
                shape,
                dtype,
                withGraph=withGraph,
                distributed_init_method=get_distributed_init_method(
                    get_ip(), get_open_port()
                ),
            )
            if test_fn is test_multi_group:
                df.extend(expand_groups(ret))
            else:
                df.append(ret)
    df = pd.DataFrame(df)
    show_cols = [
        "test",
        "withGraph",
        "ar_min_us",
        "ar_max_us",
        "ar_err",
        "ag_min_us",
        "ag_max_us",
        "ag_err",
        "rs_min_us",
        "rs_max_us",
        "rs_err",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    logger.info(
        "custom group comm ops summary (markdown):\n%s",
        df[show_cols].to_markdown(index=False),
    )
