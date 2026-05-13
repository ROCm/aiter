# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method
from typing_extensions import Optional

import torch
import torch.distributed as dist
import pandas as pd

from aiter import dtypes
from aiter.dist.communication_op import tensor_model_parallel_fused_qknorm_allreduce
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
from aiter.test_common import benchmark, checkAllclose, perftest

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


def qknorm_allreduce(
    tp_size,
    pp_size,
    rankID,
    qkv_in,
    q_w,
    k_w,
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
    qkv_in = qkv_in.to(device)
    q_w = q_w.to(device)
    k_w = k_w.to(device)
    # dist.barrier(device_ids=[i for i in range(tp_size)])

    # warmup and align all gpu
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    if withGraph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc:
            with torch.cuda.graph(graph, stream=gc.stream):
                q_out, k_out, v_out = tensor_model_parallel_fused_qknorm_allreduce(
                    qkv_in, q_w, k_w, 1e-6
                )
        q_out.fill_(0)
        k_out.fill_(0)
        v_out.fill_(0)

        @perftest()
        def run_ca():
            graph.replay()

        _, us = run_ca()
        out = ((q_out, k_out, v_out), us)
    else:

        @perftest()
        def run_ca(qkv_in, q_w, k_w):
            return tensor_model_parallel_fused_qknorm_allreduce(qkv_in, q_w, k_w, 1e-6)

        out = run_ca(qkv_in, q_w, k_w)

    # destroy
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


def qknorm_allreduce_host(qkv_ins, q_ws, k_ws, eps=1e-6):
    tp_size = len(qkv_ins)
    token_num = qkv_ins[0].shape[0]
    hidden_dim_q = q_ws[0].shape[0]
    hidden_dim_k = k_ws[0].shape[0]
    hidden_dim_v = qkv_ins[0].shape[1] - hidden_dim_q - hidden_dim_k
    qs = []
    ks = []
    vs = []
    q_vars = []
    k_vars = []
    for i in range(tp_size):
        qkv_in = qkv_ins[i]
        q, k, v = qkv_in.split([hidden_dim_q, hidden_dim_k, hidden_dim_v], dim=1)
        vs.append(v)
        orig_dtype = q.dtype
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        qs.append(q)
        ks.append(k)
        q_var = q.pow(2).mean(dim=-1, keepdim=True)
        k_var = k.pow(2).mean(dim=-1, keepdim=True)
        q_vars.append(q_var)
        k_vars.append(k_var)
    q_var_all = torch.zeros((token_num, 1), dtype=torch.float32)
    k_var_all = torch.zeros((token_num, 1), dtype=torch.float32)
    for i in range(tp_size):
        q_var_all += q_vars[i]
        k_var_all += k_vars[i]
    q_var_all = q_var_all / tp_size
    k_var_all = k_var_all / tp_size
    q_outs = []
    k_outs = []
    for i in range(tp_size):
        q = qs[i]
        k = ks[i]
        qw = q_ws[i]
        kw = k_ws[i]
        q = (q * torch.rsqrt(q_var_all + eps) * qw).to(orig_dtype)
        k = (k * torch.rsqrt(k_var_all + eps) * kw).to(orig_dtype)
        q_outs.append(q)
        k_outs.append(k)
    return q_outs, k_outs, vs


@benchmark()
def test_qknorm_allreduce(
    tp_size,
    pp_size,
    shape,
    dtype,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    qkv_ins = []
    q_ws = []
    k_ws = []
    rets = []
    for i in range(tp_size):
        token_num = shape[0]
        hidden_dim_q = shape[1]
        hidden_dim_k = shape[2]
        hidden_dim_v = shape[3]
        hidden_dim = hidden_dim_q + hidden_dim_k + hidden_dim_v
        qkv_in = torch.randn((token_num, hidden_dim), dtype=dtype)
        q_w = torch.randn((hidden_dim_q,), dtype=dtype)
        k_w = torch.randn((hidden_dim_k,), dtype=dtype)
        qkv_ins.append(qkv_in)
        q_ws.append(q_w)
        k_ws.append(k_w)
        rets.append(
            pool.apply_async(
                qknorm_allreduce,
                args=(
                    tp_size,
                    pp_size,
                    i,
                    qkv_in,
                    q_w,
                    k_w,
                    withGraph,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]
    all_us = [us for _, us in rets]
    q_outs, k_outs, v_outs = qknorm_allreduce_host(qkv_ins, q_ws, k_ws)
    max_err = 0.0
    ii = 0
    for outs, us in rets:
        msg = f"test_qknorm_allreduce: {shape=} {dtype=} {withGraph=} {us:>8.2f}"
        err_q = checkAllclose(q_outs[ii], outs[0].to(q_outs[ii]), msg=msg)
        err_k = checkAllclose(k_outs[ii], outs[1].to(k_outs[ii]), msg=msg)
        err_v = checkAllclose(v_outs[ii], outs[2].to(v_outs[ii]), msg=msg)
        max_err = max(max_err, err_q)
        max_err = max(max_err, err_k)
        max_err = max(max_err, err_v)
        ii += 1
    return {
        "min_us": min(all_us),
        "max_us": max(all_us),
        "err": max_err,
    }


l_dtype = ["fp16", "bf16"]
l_shape = [(1, 3072, 512, 1024), (2, 3072, 512, 1024), (16, 3072, 512, 1024)]


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
    help="shape. e.g. -s 1,3072,512,1024",
)
parser.add_argument(
    "-g",
    "--with-graph",
    type=lambda x: str(x).lower() in ["true", "1", "yes"],
    default=True,
    help="use CUDA graph (default: True). e.g. -g true or -g false",
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
    df = []
    for dtype in l_dtype:
        for shape in l_shape:
            ret = test_qknorm_allreduce(
                8,
                1,
                shape,
                dtype,
                withGraph=args.with_graph,
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
        "min_us",
        "max_us",
        "err",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    logger.info(
        "fused qknorm allreduce summary (markdown):\n%s",
        df[show_cols].to_markdown(index=False),
    )
