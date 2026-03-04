# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import os
import torch
import torch.distributed as dist
from typing import Optional
import argparse
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
from aiter.dist.communication_op import tensor_model_parallel_reduce_scatter
from aiter.test_common import (
    checkAllclose,
    perftest,
    benchmark,
)
from multiprocessing import set_start_method, Pool, freeze_support
import logging

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


def reduce_scatter(
    tp_size,
    pp_size,
    rankID,
    x,
    withGraph=False,
    use_custom=False,
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
                out = tensor_model_parallel_reduce_scatter(x, use_custom=use_custom)
        out.fill_(0)

        @perftest()
        def run_ca():
            graph.replay()

        _, us = run_ca()
        out = (out, us)
    else:

        @perftest()
        def run_ca(x):
            return tensor_model_parallel_reduce_scatter(x, use_custom=use_custom)

        out = run_ca(x)

    # destroy
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


def get_reduce_scatter_output(
    tp_size,
    pp_size,
    shape,
    dtype,
    rand_seed,
    use_custom,
    distributed_init_method: Optional[str] = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    rets = []
    for i in range(tp_size):
        # input = torch.randn(shape, dtype=dtype, device="cuda")
        input_ = torch.ones(shape, dtype=dtype, device="cuda")
        n = input_.numel()
        chunk_size = n // 8
        input = rand_seed.repeat_interleave(chunk_size)
        rets.append(
            pool.apply_async(
                reduce_scatter,
                args=(
                    tp_size,
                    pp_size,
                    i,
                    input,
                    False,
                    use_custom,
                    # 1,
                    distributed_init_method,
                ),
            )
            # pool.apply_async(call_aiter_allgather_naive, args=(tp_size, pp_size, i, input, 1))
        )
    pool.close()
    pool.join()

    ar_rslt = []
    rets = [el.get() for el in rets]
    for out, us in rets:
        ar_rslt.append(out)
    return ar_rslt


def reduce_scatter_acctest(
    tp_size, pp_size, shape, dtype, distributed_init_method: Optional[str] = None
):
    rand_seed = torch.randint(1, 16, (tp_size,), dtype=dtype, device="cuda")
    dist_rslt = get_reduce_scatter_output(
        tp_size, pp_size, shape, dtype, rand_seed, False, distributed_init_method
    )
    aiter_rslt = get_reduce_scatter_output(
        tp_size, pp_size, shape, dtype, rand_seed, True, distributed_init_method
    )
    error = 0.0
    for i in range(len(dist_rslt)):
        error += checkAllclose(dist_rslt[i], aiter_rslt[i])
    if error == 0:
        print("accuracy pass")
    else:
        print("accuracy failed")


@benchmark()
def reduce_scatter_perftest(
    tp_size,
    pp_size,
    shape,
    dtype,
    withGraph=False,
    use_custom=False,
    distributed_init_method: Optional[str] = None,
):
    print(f"run perf test, use custom allgather {use_custom}")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    input_list = []
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        input_list.append(x)
        rets.append(
            pool.apply_async(
                reduce_scatter,
                args=(
                    tp_size,
                    pp_size,
                    i,
                    x,
                    withGraph,
                    use_custom,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()
    ref = input_list[0]
    for i in range(tp_size - 1):
        ref = torch.concat((ref, input_list[i + 1]), -1)

    rets = [el.get() for el in rets]
    for out, us in rets:
        msg = f"reduce_scatter (use custom {use_custom}): {shape=} {dtype=} {withGraph=} {us:>8.2f}"
        print(msg)
        # print(cpu_rslt[out.device.index])
        # checkAllclose(ref, out.to(ref), msg=msg)
        # checkAllclose(ref, out.to(ref), msg=msg)


l_dtype = ["bf16"]
l_shape = [
    # (4096, 2048)
    (128, 8192)
    # (16, 512)
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
            print(f"accuracy test of dtype:{dtype}, shape:{shape}")
            reduce_scatter_acctest(
                8,
                1,
                shape,
                dtype,
                distributed_init_method=get_distributed_init_method(
                    get_ip(), get_open_port()
                ),
            )
            print(f"perf test of dtype:{dtype}, shape:{shape}")
            reduce_scatter_perftest(
                8,
                1,
                shape,
                dtype,
                withGraph=False,
                use_custom=True,
                distributed_init_method=get_distributed_init_method(
                    get_ip(), get_open_port()
                ),
            )
            reduce_scatter_perftest(
                8,
                1,
                shape,
                dtype,
                withGraph=False,
                use_custom=False,
                distributed_init_method=get_distributed_init_method(
                    get_ip(), get_open_port()
                ),
            )
