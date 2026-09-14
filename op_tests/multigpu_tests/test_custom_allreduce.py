# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method

import pandas as pd
import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
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


def allreduce_custom(
    tp_size,
    pp_size,
    rankID,
    x,
    withGraph=False,
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
    # dist.barrier(device_ids=[i for i in range(tp_size)])

    # warmup and align all gpu
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    if withGraph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
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
def test_allreduce_custom(
    tp_size,
    pp_size,
    shape,
    dtype,
    withGraph=False,
    distributed_init_method: str | None = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        ref += x
        rets.append(
            pool.apply_async(
                allreduce_custom,
                args=(tp_size, pp_size, i, x, withGraph, distributed_init_method),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]
    all_us = [us for _, us in rets]
    max_err = 0.0
    for out, us in rets:
        msg = f"test_allreduce_custom: {shape=} {dtype=} {withGraph=} {us:>8.2f}"
        err = checkAllclose(ref, out.to(ref), msg=msg)
        max_err = max(max_err, err)
    return {
        "min_us": min(all_us),
        "max_us": max(all_us),
        "err": max_err,
    }


l_dtype = ["fp16", "bf16"]
l_shape = [(2, 7168), (128, 8192)]

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
    "-g",
    "--with-graph",
    type=lambda x: str(x).lower() in ["true", "1", "yes"],
    default=True,
    help="use CUDA graph (default: True). e.g. -g true or -g false",
)


def _run_torchrun_worker(l_dtype, l_shape):
    """Single-rank worker for a torchrun launch (single- or multi-node).

    Each process is one rank. rank/world_size/local_rank come from the env that
    torchrun sets. Used for the cross-node ws=8 (2x4) run, which the Pool path
    above cannot reach (it assumes all GPUs live on one node).

    Correctness is checked against an independent RCCL all_reduce of the same
    input — so a silent fallback to RCCL cannot mask a broken custom-AR result:
    both sides would then just be RCCL and agree. Watch the aiter log line
    ("using torch.symm_mem transport ...") to confirm the custom path is live.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    # init_distributed_environment() force-sets HIP_VISIBLE_DEVICES to
    # range(world_size) when it is unset — a single-node assumption that maps
    # rank 4..7 to devices that do not exist on a 4-GPU node. Pin visibility to
    # the local GPUs first so that block is skipped and cuda:{local_rank} is a
    # valid per-node device index.
    os.environ.setdefault(
        "HIP_VISIBLE_DEVICES",
        ",".join(map(str, range(torch.cuda.device_count()))),
    )
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    ensure_model_parallel_initialized(world_size, 1)

    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    max_err = 0.0
    for dtype in l_dtype:
        for shape in l_shape:
            torch.manual_seed(1234 + rank)
            x = torch.randn(shape, dtype=dtype, device=device)
            ref = x.clone()
            dist.all_reduce(ref, group=group)  # RCCL ground truth

            out = tensor_model_parallel_all_reduce(x)
            torch.cuda.synchronize()

            err = (out.float() - ref.float()).abs().max().item()
            denom = ref.float().abs().max().item() + 1e-6
            rel = err / denom
            max_err = max(max_err, rel)
            logger.info(
                "[rank %d/%d] shape=%s dtype=%s rel_err=%.3e",
                rank,
                world_size,
                tuple(shape),
                dtype,
                rel,
            )
            # bf16/fp16 accumulation over up to 8 addends: ~1e-2 is expected.
            assert rel < 3e-2, (
                f"[rank {rank}] custom AR mismatch vs RCCL: shape={tuple(shape)} "
                f"dtype={dtype} rel_err={rel:.3e}"
            )

    logger.info("[rank %d/%d] PASS (max rel_err=%.3e)", rank, world_size, max_err)
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    if args.dtype is None:
        l_dtype_resolved = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype_resolved = [dtypes.d_dtypes[args.dtype]]
    l_shape_resolved = [args.shape] if args.shape is not None else l_shape

    # torchrun launch (single- or multi-node): each process is one rank. This is
    # the only path that reaches ws=8 on gfx1250, where 8 GPUs span 2 nodes.
    #   torchrun --nnodes=2 --nproc_per_node=4 --node_rank=<0|1> \
    #     --master_addr=<node0-ip> --master_port=<port> \
    #     op_tests/multigpu_tests/test_custom_allreduce.py
    # Set AITER_CUSTOM_AR_USE_SYMM_MEM=1 for the mori-backed cross-node transport.
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        _run_torchrun_worker(l_dtype_resolved, l_shape_resolved)
        raise SystemExit(0)

    l_dtype = l_dtype_resolved
    l_shape = l_shape_resolved
    df = []
    for dtype in l_dtype:
        for shape in l_shape:
            ret = test_allreduce_custom(
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
        "custom allreduce summary (markdown):\n%s",
        df[show_cols].to_markdown(index=False),
    )
