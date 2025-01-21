# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import os

import torch
import torch.distributed as dist

from aiter.dist.parallel_state import (ensure_model_parallel_initialized,
                                      init_distributed_environment,
                                      set_custom_all_reduce,
                                      get_tp_group,
                                      graph_capture,
                                      destroy_model_parallel,
                                      destroy_distributed_environment)
from aiter.dist.utils import (get_open_port,
                             get_distributed_init_method,
                             get_ip)
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.test_common import checkAllclose, perftest, tensor_dump, tensor_load
from multiprocessing import set_start_method, Pool, freeze_support
import logging
logger = logging.getLogger("aiter")

set_start_method('spawn', force=True)


def allreduce_custom(tp_size, pp_size, rankID, x, withGraph=False):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
    logger.info(
        f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()))
    ensure_model_parallel_initialized(tp_size, pp_size)
    x = x.to(device)
    # dist.barrier(device_ids=[i for i in range(tp_size)])

    # warmup and align all gpu
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    if withGraph:
        @perftest()
        def run_ca(graph):
            graph.replay()

        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc:
            with torch.cuda.graph(graph, stream=gc.stream):
                out = tensor_model_parallel_all_reduce(x)
        out.fill_(0)

        _, us = run_ca(graph)
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


def test_allreduce_custom(tp_size, pp_size, shape, dtype, withGraph=False):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        ref += x
        rets.append(pool.apply_async(allreduce_custom,
                                     args=(tp_size, pp_size, i, x, withGraph)))
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]
    for out, us in rets:
        msg = f'test_allreduce_custom: {shape=} {dtype=} {withGraph=} {us:.2f}'
        checkAllclose(ref, out.to(ref), msg=msg)


if __name__ == '__main__':
    freeze_support()
    for dtype in [torch.bfloat16, torch.float16]:
        for shape in [(128, 8192)]:
            test_allreduce_custom(8, 1, shape, dtype, withGraph=True)
            # test_allreduce_custom(8, 1, shape, dtype, withGraph=False)
