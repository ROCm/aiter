# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import os

import torch
import torch.distributed as dist
import argparse
import aiter
from aiter import dtypes
from aiter.fused_moe import torch_moe, fused_topk
from aiter.dist.parallel_state import get_world_group
import mori
import multiprocessing as mp
from aiter.test_common import (
    checkAllclose,
    perftest,
    benchmark,
)

def run_mori(rankID, world_size, E, tokens, topk_weights, topk_ids):
    hdim = tokens.shape[-1]
    topk = topk_weights.shape[-1]
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(world_size, rankID)
    tokens.to(device)
    topk_weights.to(device)
    topk_ids.to(device)

    # init dist
    world_group = torch.distributed.group.WORLD
    assert world_group is not None
    torch._C._distributed_c10d._register_process_group("default", world_group)
    mori.shmem.shmem_torch_process_group_init("default")
    mori_config = mori.ops.EpDispatchCombineConfig(
        data_type=dtype,
        rank=rankID,
        world_size=world_size,
        hidden_dim=hdim,
        max_num_inp_token_per_rank=128,
        num_experts_per_rank=E // world_size,
        num_experts_per_token=topk,
    )
    mori_op = mori.ops.EpDispatchCombineOp(mori_config)

    dispatch_output, dispatch_weights, dispatch_ids = mori_op.dispatch(
        tokens, topk_weights, topk_ids
    )
    torch.cuda.synchronize()
    src_token_pos = mori_op.get_dispatch_src_token_pos()

    combine_output = mori_op.combine(dispatch_output, topk_weights, topk_ids)
    # destroy dist
    aiter.destroy_dist_env()
    # return dispatch_ids[:10].cpu(), 1
    return dispatch_ids[: src_token_pos.shape[0]+1].cpu(), 1


def test_dispatch_combine(world_size, shape, dtype, E, topk):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = mp.Pool(processes=world_size)

    tokenNum, hdim = shape
    tokens = torch.randn(shape, dtype=dtype).cuda()
    score = torch.randn((tokenNum, E), device="cuda", dtype=dtype)
    topk_weights, topk_ids = fused_topk(tokens, score, topk, True)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    tokens_dp = tokens.split(world_size)
    topk_weights_dp = topk_weights.split(world_size)
    topk_ids_dp = topk_ids.to(torch.uint32).split(world_size)
    for i in range(world_size):
        x = torch.randn(shape, dtype=dtype)
        ref += x
        rets.append(
            pool.apply_async(
                run_mori,
                args=(
                    i,
                    world_size,
                    E,
                    tokens_dp[i],
                    topk_weights_dp[i],
                    topk_ids_dp[i],
                ),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]
    print(f"{topk_ids=}")
    for i, (out, us) in enumerate(rets):
        mask = (topk_ids >= i * E // world_size) & (
            topk_ids < (i + 1) * E // world_size
        )
        ref = topk_ids[mask.any(1)]
        print(f"{mask=}")
        print(f"{ref=}")
        print(f"{out=}")
        print(f"{ref.shape=}, {out.shape=}")
        checkAllclose(topk_ids[mask.any(1)], out.to(topk_ids))


l_dtype = ["bf16"]
l_shape = [(128, 8192)]

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
    choices=l_shape,
    nargs="?",
    const=None,
    default=None,
    help="shape",
)


if __name__ == "__main__":
    mp.freeze_support()
    args = parser.parse_args()
    if args.dtype is None:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    if args.shape is not None:
        l_shape = [args.shape]

    for dtype in l_dtype:
        for shape in l_shape:
            test_dispatch_combine(8, shape, dtype, 128, 4)
