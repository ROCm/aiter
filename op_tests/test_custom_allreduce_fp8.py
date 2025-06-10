# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import os

import torch
import torch.distributed as dist
from aiter import get_hip_quant, QuantType

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


def allreduce_custom(tp_size, pp_size, rankID, x, withGraph=False):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
    logger.info(f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()),
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
                out = tensor_model_parallel_all_reduce(x, open_fp8_quant=True)
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
def test_allreduce_custom(tp_size, pp_size, shape, dtype, withGraph=False):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        max_val = x.max().item()
        min_val = x.min().item()
        if max_val > min_val:
            mm = x.max()
        else:
            mm = abs(x.min())
        x = x / mm
        ref += x
        rets.append(
            pool.apply_async(allreduce_custom, args=(tp_size, pp_size, i, x, withGraph))
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]

    a = ref.clone().cuda()
    a.to(float)
    qtype = QuantType.per_1x128
    quant_function = get_hip_quant(qtype)
    a = a.reshape(8192, 128)
    fp8_output, scale = quant_function(a, quant_dtype=torch.float8_e4m3fnuz)
    fp32_output = fp8_output.to(torch.float) * scale
    fp16_quanted_ref = fp32_output.to(torch.float16).reshape(128, 8192)
    for out, us in rets:
        gpu_id = out.device.index
        ori_ref = ref.clone()
        ori_tensor = ori_ref[gpu_id * 16 : (gpu_id + 1) * 16][:]
        c = fp16_quanted_ref.clone()
        c[gpu_id * 16 : (gpu_id + 1) * 16][:] = ori_tensor
        msg = f"test_allreduce_custom: {shape=} {dtype=} {withGraph=} {us:>8.2f}"
        checkAllclose(c.cpu(), out.cpu(), msg=msg)


if __name__ == "__main__":
    freeze_support()
    for dtype in [torch.float16]:
        for shape in [(128, 8192)]:
            test_allreduce_custom(8, 1, shape, dtype, withGraph=True)
