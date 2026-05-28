# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import torch
import torch.distributed as dist
from typing import Optional
import argparse
import pandas as pd
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
    dim=0,
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
                out = tensor_model_parallel_reduce_scatter(
                    x, use_custom=use_custom, dim=dim
                )
        out.fill_(0)

        @perftest()
        def run_ca():
            graph.replay()

        _, us = run_ca()
        out = (rankID, out.cpu(), us)
    else:

        @perftest()
        def run_ca(x):
            return tensor_model_parallel_reduce_scatter(
                x, use_custom=use_custom, dim=dim
            )

        result, us = run_ca(x)
        out = (rankID, result.cpu(), us)

    # destroy
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


def get_reduce_scatter_output(
    tp_size,
    pp_size,
    input_list,
    use_custom,
    dim=0,
    distributed_init_method: Optional[str] = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    rets = []
    for i in range(tp_size):
        rets.append(
            pool.apply_async(
                reduce_scatter,
                args=(
                    tp_size,
                    pp_size,
                    i,
                    input_list[i],
                    False,
                    use_custom,
                    dim,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()

    results = {}
    rets = [el.get() for el in rets]
    for rankID, out, us in rets:
        results[rankID] = out
    return results


def reduce_scatter_acctest(
    tp_size,
    pp_size,
    shape,
    dtype,
    dim=0,
    distributed_init_method: Optional[str] = None,
):
    input_list = [torch.randn(shape, dtype=dtype) for _ in range(tp_size)]
    dist_rslt = get_reduce_scatter_output(
        tp_size, pp_size, input_list, False, dim, distributed_init_method
    )
    aiter_rslt = get_reduce_scatter_output(
        tp_size, pp_size, input_list, True, dim, distributed_init_method
    )

    ref = input_list[0].clone()
    for i in range(1, tp_size):
        ref = ref + input_list[i]
    ref_chunks = ref.chunk(tp_size, dim=dim)

    error = 0.0
    for rankID in range(tp_size):
        expected = ref_chunks[rankID]
        msg = f"rank {rankID} dist vs ref (dim={dim})"
        error += checkAllclose(
            expected, dist_rslt[rankID], rtol=0.05, atol=0.2, msg=msg
        )
        msg = f"rank {rankID} aiter vs ref (dim={dim})"
        error += checkAllclose(
            expected, aiter_rslt[rankID], rtol=0.05, atol=0.2, msg=msg
        )
        msg = f"rank {rankID} dist vs aiter (dim={dim})"
        error += checkAllclose(
            dist_rslt[rankID], aiter_rslt[rankID], rtol=0.05, atol=0.2, msg=msg
        )
    if error == 0:
        print(f"accuracy pass (dim={dim})")
    else:
        print(f"accuracy failed (dim={dim})")


@benchmark()
def reduce_scatter_perftest(
    tp_size,
    pp_size,
    shape,
    dtype,
    withGraph=False,
    use_custom=False,
    dim=0,
    distributed_init_method: Optional[str] = None,
):
    print(f"run perf test, use_custom={use_custom}, dim={dim}")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
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
                    dim,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()

    ref = input_list[0].clone()
    for i in range(1, tp_size):
        ref = ref + input_list[i]
    ref_chunks = ref.chunk(tp_size, dim=dim)

    rets = [el.get() for el in rets]
    all_us = [us for _, _, us in rets]
    max_err = 0.0
    for rankID, out, us in rets:
        expected = ref_chunks[rankID]
        msg = (
            f"reduce_scatter (use_custom={use_custom}, dim={dim}): "
            f"{shape=} {dtype=} {withGraph=} {us:>8.2f}"
        )
        err = checkAllclose(expected, out, rtol=0.05, atol=0.2, msg=msg)
        max_err = max(max_err, err)
    return {
        "min_us": min(all_us),
        "max_us": max(all_us),
        "err": max_err,
    }


l_dtype = ["bf16"]
l_shape = [
    (128, 8192),
    (4, 8, 8192),
]
l_large_shape = [
    (32768, 8192),
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
    l_dim = [0, 1, -1]
    tp_size = 8
    df = []
    test_cases = []
    for shape in l_shape:
        for dim in l_dim:
            test_cases.append((shape, dim))
    for shape in l_large_shape:
        test_cases.append((shape, 0))

    for dtype in l_dtype:
        for shape, dim in test_cases:
            actual_dim = dim if dim >= 0 else dim + len(shape)
            if shape[actual_dim] % tp_size != 0:
                print(
                    f"skip shape={shape} dim={dim}: "
                    f"shape[{actual_dim}]={shape[actual_dim]} "
                    f"not divisible by tp_size={tp_size}"
                )
                continue
            print(f"accuracy test of dtype:{dtype}, shape:{shape}, dim:{dim}")
            reduce_scatter_acctest(
                tp_size,
                1,
                shape,
                dtype,
                dim=dim,
                distributed_init_method=get_distributed_init_method(
                    get_ip(), get_open_port()
                ),
            )
            print(f"perf test of dtype:{dtype}, shape:{shape}, dim:{dim}")
            for use_custom in [True, False]:
                ret = reduce_scatter_perftest(
                    tp_size,
                    1,
                    shape,
                    dtype,
                    withGraph=False,
                    use_custom=use_custom,
                    dim=dim,
                    distributed_init_method=get_distributed_init_method(
                        get_ip(), get_open_port()
                    ),
                )
                df.append(ret)
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
        "reduce scatter summary (markdown):\n%s",
        df[show_cols].to_markdown(index=False),
    )
