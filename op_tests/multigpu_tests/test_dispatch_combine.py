# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import os

import torch
import torch.distributed as dist
import argparse
import aiter
from aiter import dtypes
from aiter.fused_moe import torch_moe, fused_topk, fused_moe
from aiter.dist.parallel_state import get_world_group
from aiter.ops.shuffle import shuffle_weight
import mori
import multiprocessing as mp
from aiter.test_common import (
    checkAllclose,
    perftest,
    benchmark,
)


def run_ref(
    world_size,
    E,
    tokens,
    topk_weights,
    topk_ids,
    w1_ep,
    w2_ep,
    w1_scale_ep,
    w2_scale_ep,
    quant_type,
):
    ref = torch.zeros(tokens.shape, dtype=dtypes.bf16, device=tokens.device)
    out_list = []
    for i in range(world_size):
        mask = (topk_ids >= i * E // world_size) & (
            topk_ids < (i + 1) * E // world_size
        )
        if not mask.any():
            continue
        topk_ids_ep = topk_ids[mask.any(1)]
        topk_weights_ep = topk_weights[mask.any(1)]
        tokens_ep = tokens[mask.any(1)]
        expert_mask = torch.zeros((E,), dtype=dtypes.i32, device=w1_ep[i].device)
        expert_mask[E // world_size * i : E // world_size * (i + 1)] = 1
        num_local_tokens = torch.tensor(
            [tokens_ep.shape[0]], dtype=dtypes.i32, device=tokens_ep.device
        )
        out = fused_moe(
            tokens_ep,
            w1_ep[i],
            w2_ep[i],
            topk_weights_ep,
            topk_ids_ep,
            expert_mask,
            num_local_tokens=num_local_tokens,
            w1_scale=w1_scale_ep[i],
            w2_scale=w2_scale_ep[i],
            quant_type=quant_type,
        )
        # out_list.append(out)
        # return out_list
        ref[mask.any(1)] += out
        # ref[mask.any(1)] += out.to(dtypes.fp32)
    return ref.to(tokens)


def run_mori(
    rankID,
    world_size,
    E,
    tokens,
    topk_weights,
    topk_ids,
    w1,
    w2,
    w1_scale,
    w2_scale,
    quant_type,
):
    token_num = tokens.shape[0]
    hdim = tokens.shape[-1]
    topk = topk_weights.shape[-1]
    dtype = tokens.dtype
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(world_size, rankID)
    tokens = tokens.to(device)
    topk_weights = topk_weights.to(device)
    topk_ids = topk_ids.to(device)
    w1 = w1.to(device)
    w2 = w2.to(device)
    w1_scale = w1_scale.to(device) if w1_scale is not None else None
    w2_scale = w2_scale.to(device) if w2_scale is not None else None

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

    dispatch_output, dispatch_weights, dispatch_ids, dispatch_recv_token_num = (
        mori_op.dispatch(tokens, topk_weights, topk_ids)
    )
    torch.cuda.synchronize()
    # src_token_pos = mori_op.get_dispatch_src_token_pos().cpu()
    # src_token_num = src_token_pos.shape[0]
    # src_token_order = torch.sort(src_token_pos)[1].cpu()
    # print(
    #     f"{rankID=} {src_token_pos=} {src_token_order=}"
    # )
    # dispatch_ids = dispatch_ids[: src_token_num].to(dtypes.i32)[src_token_order]
    # dispatch_output = dispatch_output[: src_token_num][src_token_order]
    # dispatch_weights = dispatch_weights[: src_token_num][src_token_order]

    expert_mask = torch.zeros((E,), dtype=dtypes.i32, device=device)
    expert_mask[E // world_size * rankID : E // world_size * (rankID + 1)] = 1
    out = fused_moe(
        dispatch_output,
        w1,
        w2,
        dispatch_weights,
        dispatch_ids.to(dtypes.i32),
        expert_mask,
        num_local_tokens=dispatch_recv_token_num.to(dtypes.i32),
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        quant_type=quant_type,
    )

    # aiter.destroy_dist_env()
    # return out[:src_token_num].cpu()

    combine_output = mori_op.combine(out, topk_weights, topk_ids)
    aiter.destroy_dist_env()
    return combine_output[:token_num].cpu()


def weight_per_128x128_quant(weight, quant_dtype):
    E, dim1, dim2 = weight.shape
    weight_blocks = weight.view(
        E, dim1 // 128, 128, dim2 // 128, 128
    )  # [E, num_blocks_dim1, 128, num_blocks_dim2, 128]
    weight_blocks = weight_blocks.permute(
        0, 1, 3, 2, 4
    ).contiguous()  # [E, num_blocks_dim1, num_blocks_dim2, 128, 128]
    weight_blocks = weight_blocks.view(E, -1, 128 * 128)  # [E, num_blocks, 128*128]
    weight_qt, weight_scale = aiter.pertoken_quant(
        weight_blocks, quant_dtype=quant_dtype
    )
    weight_qt = weight_qt.view(
        E, dim1 // 128, dim2 // 128, 128, 128
    )  # [E, num_blocks_dim1, num_blocks_dim2, 128, 128]
    weight_qt = weight_qt.permute(
        0, 1, 3, 2, 4
    ).contiguous()  # [E, num_blocks_dim1, 128, num_blocks_dim2, 128]
    weight_qt = weight_qt.view(E, dim1, dim2)  # [E, dim1, dim2]
    weight_scale = weight_scale.view(
        E, dim1 // 128, dim2 // 128
    )  # [E, num_blocks_dim1, num_blocks_dim2]
    return weight_qt, weight_scale


def test_dispatch_combine(
    world_size, shape, dtype, E, topk, quant_type=aiter.QuantType.No
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(processes=world_size)

    quant_func = (
        aiter.get_torch_quant(quant_type)
        if quant_type != aiter.QuantType.per_128x128
        else weight_per_128x128_quant
    )

    tokenNum, hdim, idim = shape
    tokens = torch.randn((tokenNum, hdim), dtype=dtype, device="cuda")
    score = torch.randn((tokenNum, E), device="cuda", dtype=dtype)
    topk_weights, topk_ids = fused_topk(tokens, score, topk, True)

    w1 = torch.randn((E, 2 * idim, hdim), dtype=dtype, device="cuda")
    w2 = torch.randn((E, hdim, idim), dtype=dtype, device="cuda")
    if quant_type == aiter.QuantType.per_128x128:
        weight_per_128x128_quant(w1, quant_dtype=dtypes.fp8)

    w1_qt, w1_scale = quant_func(w1, quant_dtype=dtypes.fp8)
    w2_qt, w2_scale = quant_func(w2, quant_dtype=dtypes.fp8)
    w1_qt = shuffle_weight(w1_qt)
    w2_qt = shuffle_weight(w2_qt)

    tokens_dp = tokens.chunk(world_size)
    topk_weights_dp = topk_weights.chunk(world_size)
    topk_ids_dp = topk_ids.to(torch.uint32).chunk(world_size)
    w1_ep = w1_qt.chunk(world_size)
    w2_ep = w2_qt.chunk(world_size)
    w1_scale_ep = (
        w1_scale.chunk(world_size) if w1_scale is not None else [None] * world_size
    )
    w2_scale_ep = (
        w2_scale.chunk(world_size) if w2_scale is not None else [None] * world_size
    )
    ref_noep = fused_moe(
        tokens,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        quant_type=quant_type,
    )
    ref = run_ref(
        world_size,
        E,
        tokens,
        topk_weights,
        topk_ids,
        w1_ep,
        w2_ep,
        w1_scale_ep,
        w2_scale_ep,
        quant_type,
    )
    checkAllclose(ref, ref_noep.to(ref), msg=f"EP ref vs no EP ref")
    ref_dp = ref.chunk(world_size)

    rets = []
    for i in range(world_size):
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
                    w1_ep[i],
                    w2_ep[i],
                    w1_scale_ep[i],
                    w2_scale_ep[i],
                    quant_type,
                ),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]

    ret_out = torch.cat(rets, dim=0)
    checkAllclose(ref, ret_out.to(ref), msg=f"total tokens:")

    for i in range(world_size):
        checkAllclose(ref_dp[i], rets[i].to(ref_dp[i]), msg=f"rank:{i}")


l_dtype = ["bf16"]
l_shape = [(128, 6144, 1024)]
quant_types = [
    aiter.QuantType.No,
    aiter.QuantType.per_Token,
    aiter.QuantType.per_128x128,
][-1:]

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

    for quant_type in quant_types:
        for dtype in l_dtype:
            for shape in l_shape:
                test_dispatch_combine(8, shape, dtype, 16, 2, quant_type)
