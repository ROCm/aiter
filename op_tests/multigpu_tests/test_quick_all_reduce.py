# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import logging
import multiprocessing
import os
from multiprocessing import Pool, freeze_support, set_start_method

import pandas as pd
import torch
import torch.distributed as dist

import aiter as ops
from aiter import dtypes
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.device_communicators.quick_all_reduce import qr_exchange_handles
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
from aiter.test_common import checkAllclose, perftest

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


def _run_allreduce_quick_case(
    rankID, tp_size, case_idx, shape, dtype, withGraph, graphs
):
    """All-reduce one case on an initialized rank and check it against the
    sum of every rank's input.

    Each rank regenerates all ``tp_size`` inputs from the same ``case_idx``
    seed and keeps its own, so the reference is available locally and only
    scalars go back to the parent. In graph mode the captured graph and its
    buffers are appended to ``graphs`` and must outlive the whole sweep: the
    custom / quick all-reduce communicators cache the peer IPC address of
    every captured buffer by its local pointer, so freeing one and capturing
    a later case at the same address would replay with stale peer pointers.
    """
    gen = torch.Generator(device="cuda").manual_seed(case_idx)
    ref = torch.zeros(shape, dtype=dtype, device="cuda")
    for rank in range(tp_size):
        xr = torch.randn(shape, dtype=dtype, device="cuda", generator=gen)
        ref += xr
        if rank == rankID:
            x = xr

    if withGraph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
            out = tensor_model_parallel_all_reduce(x)
        out.fill_(0)
        graphs.append((graph, x, out))

        @perftest()
        def run_ca():
            graph.replay()

        _, us = run_ca()
    else:

        @perftest()
        def run_ca(x):
            return tensor_model_parallel_all_reduce(x)

        out, us = run_ca(x)

    atol = 1.25 * tp_size
    rtol = 0.5 * tp_size
    msg = f"test_allreduce_quick: {shape=} {dtype=} {withGraph=} {us:>8.2f}"
    return {"us": us, "err": checkAllclose(out, ref, msg=msg, atol=atol, rtol=rtol)}


def allreduce_quick_sweep(
    tp_size,
    pp_size,
    rankID,
    cases,
    distributed_init_method: str | None = None,
):
    """Run every case on one rank inside a single distributed init.

    Setting up the TP group dominates a single case (seconds vs. microseconds
    of kernel time), so the group is created and torn down once. Results are
    returned rather than asserted here so every rank walks the full case list
    and the collectives stay aligned.
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
        _run_allreduce_quick_case(
            rankID,
            tp_size,
            case_idx,
            case["shape"],
            case["dtype"],
            case["withGraph"],
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


def test_allreduce_quick(tp_size, pp_size, cases, quantization: str = "INT4"):
    """Run ``cases`` on one TP group and return one summary row per case.

    The quantization regime is read when the quick-all-reduce communicator is
    created, so it is fixed for the whole launch.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    # Quantization regime: FP / FP8 / INT6 / INT4 / INT3 / NONE.
    # INT3 is only supported on TP2 (world_size == 2).
    os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = quantization
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=tp_size) as pool:
        rets = [
            pool.apply_async(
                allreduce_quick_sweep,
                args=(tp_size, pp_size, rank, cases, init_method),
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
                "quantization": quantization,
                "shape": case["shape"],
                "dtype": case["dtype"],
                "withGraph": case["withGraph"],
                "min_us": min(all_us),
                "max_us": max(all_us),
                "err": max(results[i]["err"] for results in per_rank),
            }
        )
    return rows


def qr_variable_input(rank, world_size):
    """
    When the tensor parallelism is set to 4 or 8, frequent changes
    in the input shape can cause QuickReduce to hang (this issue
    has been observed with the gpt_oss model).
    """
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    qr_max_size = None  # MB
    _ptr = ops.init_custom_qr(rank, world_size, qr_max_size)
    ranks = list(range(world_size))
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29500",
        rank=rank,
        world_size=world_size,
    )
    cpu_group = torch.distributed.new_group(ranks, backend="nccl")

    world_size = dist.get_world_size(group=cpu_group)
    qr_exchange_handles(_ptr, world_size, cpu_group)

    num = 1
    s1 = 1024
    while num < 50000:  # 50000 is sufficient to identify issues.
        dtype = torch.float16
        if num % 2 == 0:
            s2 = 1024
            inp1 = torch.zeros(
                (s1, s2), dtype=dtype, device=torch.cuda.current_device()
            )
        else:
            s2 = 2048
            inp1 = torch.ones((s1, s2), dtype=dtype, device=torch.cuda.current_device())
        result = torch.empty_like(inp1)
        # FP = 0 FP8 = 1 INT6 = 2 INT4 = 3 INT3 = 4 NONE = 5
        ops.qr_all_reduce(_ptr, inp1, result, 3, cast_bf2half=True)
        try:
            if inp1[0, 0] == 0:
                assert torch.all(result == 0)
            else:
                assert torch.all(result == world_size)
        except AssertionError:
            print("Assertion failed! Allreduce results are incorrect.")
            raise
        num += 1


def test_custom_quick_allreduce_variable_input(tp_size, pipeline_parallel_size=1):
    multiprocessing.set_start_method("spawn", force=True)
    # 60s is enough
    timeout = 60
    processes = []
    for rank in range(tp_size):
        p = multiprocessing.Process(target=qr_variable_input, args=(rank, tp_size))
        p.start()
        processes.append((rank, p))
    for rank, p in processes:
        p.join(timeout=timeout)
        if p.is_alive():
            for r, proc in processes:
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
            raise RuntimeError(f"QuickReduce hang detected after {timeout} seconds!")


l_dtype = ["fp16", "bf16"]
l_shape = [(1024, 8192)]

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


if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    if args.dtype is None:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    if args.shape is not None:
        l_shape = [args.shape]
    rows = test_allreduce_quick(
        8,
        1,
        [
            {"shape": shape, "dtype": dtype, "withGraph": withGraph}
            for dtype in l_dtype
            for shape in l_shape
            for withGraph in (True, False)
        ],
    )

    # INT3 quantization is only supported on TP2 (world_size == 2).
    rows += test_allreduce_quick(
        2,
        1,
        [
            {"shape": shape, "dtype": dtype, "withGraph": False}
            for dtype in l_dtype
            for shape in l_shape
        ],
        quantization="INT3",
    )
    df = pd.DataFrame(rows)
    logger.info("quick allreduce summary (markdown):\n%s", df.to_markdown(index=False))

    # check variable input for qr
    test_custom_quick_allreduce_variable_input(tp_size=4)
    print("done")
