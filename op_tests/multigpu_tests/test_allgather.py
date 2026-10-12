# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method

import pandas as pd
import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.dist.communication_op import tensor_model_parallel_all_gather
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.test_common import (
    checkAllclose,
    perftest,
)

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


def barrier_before_teardown():
    """Align all ranks before tearing down the distributed groups.

    Drain this rank's GPU work, then join a barrier so no rank starts freeing
    IPC buffers / destroying process groups while a peer is still inside a
    NCCL / custom-all-reduce collective -- that race intermittently hangs when
    these comm UTs run back-to-back in CI. No-op if dist is uninitialized.
    """
    if not dist.is_initialized():
        return
    torch.cuda.synchronize()
    get_tp_group().barrier()
    torch.cuda.synchronize()


def _run_allgather_case(
    rankID, tp_size, case_idx, shape, dtype, withGraph, use_custom, dim, graphs
):
    """All-gather one case on an initialized rank and check it against the
    concatenation of every rank's input along ``dim``.

    Each rank regenerates all ``tp_size`` inputs from the same ``case_idx``
    seed and keeps its own, so the reference is available locally and only
    scalars go back to the parent. In graph mode the captured graph and its
    buffers are appended to ``graphs`` and must outlive the whole sweep:
    custom collectives cache the peer IPC address of every captured buffer by
    its local pointer, so freeing one and capturing a later case at the same
    address would replay with stale peer pointers.
    """
    if rankID == 0:
        print(f"run perf test, use custom allgather {use_custom}")
    gen = torch.Generator(device="cuda").manual_seed(case_idx)
    inputs = [
        torch.randn(shape, dtype=dtype, device="cuda", generator=gen)
        for _ in range(tp_size)
    ]
    x = inputs[rankID]
    ref = torch.cat(inputs, dim)

    if withGraph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
            out = tensor_model_parallel_all_gather(x, use_custom=use_custom, dim=dim)
        out.fill_(0)
        graphs.append((graph, x, out))

        @perftest()
        def run_ca():
            graph.replay()

        _, us = run_ca()
    else:

        @perftest()
        def run_ca(x):
            return tensor_model_parallel_all_gather(x, use_custom=use_custom, dim=dim)

        out, us = run_ca(x)

    msg = (
        f"allgather (use custom {use_custom}): rank={rankID} "
        f"{shape=} {dtype=} {withGraph=} {us:>8.2f}"
    )
    return {"us": us, "err": checkAllclose(ref, out, msg=msg)}


def allgather_sweep(
    tp_size,
    pp_size,
    rankID,
    cases,
    dtype,
    withGraph=False,
    distributed_init_method: str | None = None,
):
    """Run every case on one rank inside a single distributed init.

    Setting up the TP group dominates a single case (seconds vs. microseconds
    of kernel time), so the group is created and torn down once. Custom and
    RCCL all-gather share it: ``use_custom`` is a per-call switch.
    """
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
    logger.info(f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)

    # warmup and align all gpu
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    graphs = []
    results = [
        _run_allgather_case(
            rankID,
            tp_size,
            case_idx,
            case["shape"],
            dtype,
            withGraph,
            case["use_custom"],
            case["dim"],
            graphs,
        )
        for case_idx, case in enumerate(cases)
    ]

    # destroy
    if dist.is_initialized():
        barrier_before_teardown()
        destroy_model_parallel()
        destroy_distributed_environment()
        graphs.clear()
        torch.cuda.empty_cache()
    return results


def call_ccl_allgather_naive(
    tp_size,
    pp_size,
    rankID,
    x,
    use_custom=True,
    loop_time=1,
    distributed_init_method: str | None = None,
):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
    logger.info(f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    x = x.to(device)

    # warmup and align all gpu. device_group is a plain attribute assigned in
    # GroupCoordinator.__init__, so the access itself does nothing -- the point is
    # get_tp_group(), which raises if the TP group was never initialised.
    _ = get_tp_group().device_group
    torch.cuda.synchronize()

    for i in range(loop_time):
        out = tensor_model_parallel_all_gather(x, use_custom=use_custom)

    # destroy
    if dist.is_initialized():
        barrier_before_teardown()
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


def allgather_acctest(
    tp_size,
    pp_size,
    shape,
    dtype,
    use_custom=False,
    distributed_init_method: str | None = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    rets = []
    input_list = []
    for i in range(tp_size):
        input = torch.randn(shape, dtype=dtype, device="cuda")
        input_list.append(input)
        # print(input)
        rets.append(
            pool.apply_async(
                call_ccl_allgather_naive,
                args=(
                    tp_size,
                    pp_size,
                    i,
                    input,
                    use_custom,
                    1,
                    distributed_init_method,
                ),
            )
            # pool.apply_async(call_aiter_allgather_naive, args=(tp_size, pp_size, i, input, 1))
        )
    pool.close()
    pool.join()
    ref = input_list[0]
    for i in range(tp_size - 1):
        ref = torch.concat((ref, input_list[i + 1]), -1)

    ar_rslt = []
    for i, ret in enumerate(rets):
        rslt = ret.get()
        ar_rslt.append(rslt)
    for i in ar_rslt:
        checkAllclose(ref, i.to(ref))


def allgather_perftest(tp_size, pp_size, cases, dtype, withGraph=False):
    """Sweep ``cases`` on one TP group and return one summary row per case."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=tp_size) as pool:
        rets = [
            pool.apply_async(
                allgather_sweep,
                args=(tp_size, pp_size, rank, cases, dtype, withGraph, init_method),
            )
            for rank in range(tp_size)
        ]
        per_rank = [el.get() for el in rets]
    rows = []
    for i, case in enumerate(cases):
        all_us = [results[i]["us"] for results in per_rank]
        rows.append(
            {
                "tp_size": tp_size,
                "shape": case["shape"],
                "dtype": dtype,
                "withGraph": withGraph,
                "use_custom": case["use_custom"],
                "dim": case["dim"],
                "min_us": min(all_us),
                "max_us": max(all_us),
                "err": max(results[i]["err"] for results in per_rank),
            }
        )
    return rows


l_dtype = ["bf16"]
l_shape = [
    (1345,),
    (128, 7168),
    # exceeds max_size/world_size but satisfies all other custom ag
    # conditions (contiguous, 16-byte aligned) — should fallback to RCCL
    # threshold: 64 MB (2 GPU) / 32 MB (4 GPU) / 16 MB (8 GPU)
    # this shape = 4097*8192*2 bytes ≈ 64.015 MB, exceeds even the 2-GPU threshold
    (4097, 8192),
    # --- gfx1250 unrolled-allgather tail-coverage repro ---
    # The gfx1250 ag_gfx1250_lastdim / ag_gfx1250_naive_unroll4 kernels loop
    # with guard `idx + blockDim.x*(unroll-1) < size`, so any tail shorter than
    # blockDim.x*unroll packed elements is never written (output is
    # torch.empty -> garbage). Triggered only when the packed element count is
    # NOT a multiple of that stride. Existing shapes all divide evenly so they
    # never hit the tail; these do.
    #
    # dim=-1 (LM head geometry): DeepSeek-V4 vocab=129280, tp=4 -> per-rank
    # shard 32320. packed last_dim = 32320/8 = 4040; size = 65*4040 = 262600,
    # which is NOT a multiple of 512*4 = 2048 -> lastdim kernel drops the tail.
    (65, 32320),
    # dim=0 path: size = 65*7168/8 = 58240, NOT a multiple of 256*4 = 1024 ->
    # naive_unroll4 kernel drops the tail.
    (65, 7168),
]

parser = argparse.ArgumentParser(description="config input of test")
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="data type",
)
parser.add_argument(
    "-s",
    "--shape",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="shape. e.g. -s 128,8192",
)
parser.add_argument(
    "-t",
    "--tp_size",
    type=int,
    choices=[2, 4, 8],
    default=4,
    help="tensor-parallel world size (default: 4)",
)


if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    if args.dtype is None:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    if args.shape is not None:
        l_shape = [args.shape]
    tp_size = args.tp_size
    l_dim = [0, -1]
    # One TP group per dtype; shapes, dims and custom/RCCL are swept inside it.
    cases = [
        {"shape": shape, "dim": dim, "use_custom": use_custom}
        for shape in l_shape
        for dim in l_dim
        for use_custom in [False, True]
    ]
    df = []
    for dtype in l_dtype:
        df.extend(allgather_perftest(tp_size, 1, cases, dtype, withGraph=False))
    df = pd.DataFrame(df)
    show_cols = [
        "tp_size",
        "shape",
        "dtype",
        "withGraph",
        "use_custom",
        "dim",
        "min_us",
        "max_us",
        "err",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    logger.info(
        "allgather summary (markdown):\n%s",
        df[show_cols].to_markdown(index=False),
    )
