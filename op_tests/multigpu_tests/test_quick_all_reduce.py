# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import multiprocessing
import os
from typing import Optional

import torch
import torch.distributed as dist
import argparse
import aiter as ops
from aiter import dtypes

from aiter.dist.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
    get_tp_group,
    graph_capture,
    destroy_model_parallel,
    destroy_distributed_environment,
)
from aiter.dist.utils import get_open_port, get_distributed_init_method, get_ip
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.test_common import (
    checkAllclose,
    perftest,
    benchmark,
)
from multiprocessing import set_start_method, Pool, freeze_support
import logging

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


def allreduce_quick(
    tp_size,
    pp_size,
    rankID,
    x,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
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
    # dist.barrier(device_ids=[i for i in range(tp_size)])

    # warmup and align all gpu
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    if withGraph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc:
            with torch.cuda.graph(graph, stream=gc.stream):
                out = tensor_model_parallel_all_reduce(x)
        out.fill_(0)

        @perftest()
        def run_ca():
            graph.replay()

        _, us = run_ca()
        out = (out, us)
    else:

        @perftest()
        def run_ca(x):
            return tensor_model_parallel_all_reduce(x)

        out = run_ca(x)

    # destroy
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


@benchmark()
def test_allreduce_quick(
    tp_size,
    pp_size,
    shape,
    dtype,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = "INT4"
    pool = Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        ref += x
        rets.append(
            pool.apply_async(
                allreduce_quick,
                args=(tp_size, pp_size, i, x, withGraph, distributed_init_method),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]
    atol = 1.25 * tp_size
    rtol = 0.5 * tp_size
    for out, us in rets:
        msg = f"test_allreduce_quick: {shape=} {dtype=} {withGraph=} {us:>8.2f}"
        checkAllclose(out.cpu(), ref.cpu(), msg=msg, atol=atol, rtol=rtol)
        # ref = ref.to(out.device)
        # torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


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
    ranks = []
    for i in range(world_size):
        ranks.append(i)
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29500",
        rank=rank,
        world_size=world_size,
    )
    cpu_group = torch.distributed.new_group(ranks, backend="nccl")

    handle = ops.qr_get_handle(_ptr)
    world_size = dist.get_world_size(group=cpu_group)
    handles = [None] * world_size
    dist.all_gather_object(handles, handle, group=cpu_group)
    ops.qr_open_handles(_ptr, handles)

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
        # FP = 0 INT8 = 1 INT6 = 2 INT4 = 3 NONE = 4
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
    for dtype in l_dtype:
        for shape in l_shape:
            test_allreduce_quick(
                8,
                1,
                shape,
                dtype,
                withGraph=True,
                distributed_init_method=get_distributed_init_method(
                    get_ip(), get_open_port()
                ),
            )
            test_allreduce_quick(
                8,
                1,
                shape,
                dtype,
                withGraph=False,
                distributed_init_method=get_distributed_init_method(
                    get_ip(), get_open_port()
                ),
            )

    # check variable input for qr
    test_custom_quick_allreduce_variable_input(tp_size=4)
    print("done")
