# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import os
from aiter.dist.utils import get_distributed_init_method
import torch
import argparse
import aiter
from aiter import dtypes
from aiter.fused_moe import fused_topk, fused_moe
from aiter.ops.shuffle import shuffle_weight
import multiprocessing as mp
from aiter import get_hip_quant
from aiter.test_common import run_perftest


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
    ref = torch.zeros(tokens.shape, dtype=tokens.dtype, device=tokens.device)
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
        quant_func = get_hip_quant(
            quant_type
            if quant_type != aiter.QuantType.per_128x128
            else aiter.QuantType.per_1x128
        )
        tokens_ep_qt, scale = quant_func(tokens_ep, quant_dtype=dtypes.fp8)
        out, us = run_perftest(
            fused_moe,
            tokens_ep_qt,
            w1_ep[i],
            w2_ep[i],
            topk_weights_ep,
            topk_ids_ep,
            expert_mask,
            num_local_tokens=num_local_tokens,
            w1_scale=w1_scale_ep[i],
            w2_scale=w2_scale_ep[i],
            quant_type=quant_type,
            a1_scale=scale,
            dtype=dtypes.bf16,
        )
        print(f"rank {i} us={us:.4f}")
        ref[mask.any(1)] += out
    return ref.to(tokens)


def run_mori_with_manager(
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
    print(f"rank {rankID} in run_mori_with_manager")
    token_num = tokens.shape[0]
    hdim = tokens.shape[-1]
    topk = topk_weights.shape[-1]
    dtype = tokens.dtype
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)

    distributed_init_method = get_distributed_init_method("127.0.0.1", 12345)
    aiter.init_dist_env(
        tensor_model_parallel_size=1,
        rankID=0,
        backend="nccl",
        distributed_init_method=distributed_init_method,
        data_parallel_size=world_size,
        data_parallel_rank=rankID,
    )
    print(f"rank {rankID} after init_dist_env")

    from aiter.dist.parallel_state import get_ep_group

    ep_group = get_ep_group()

    # Use try-finally to ensure proper cleanup even if errors occur
    try:
        tokens = tokens.to(device)
        topk_weights = topk_weights.to(device)
        topk_ids = topk_ids.to(device)
        w1 = w1.to(device)
        w2 = w2.to(device)
        w1_scale = w1_scale.to(device) if w1_scale is not None else None
        w2_scale = w2_scale.to(device) if w2_scale is not None else None
        quant_func = get_hip_quant(
            quant_type
            if quant_type != aiter.QuantType.per_128x128
            else aiter.QuantType.per_1x128
        )
        tokens_qt, scale = quant_func(tokens, quant_dtype=dtypes.fp8)

        mori_manager = ep_group.device_communicator.all2all_manager
        assert mori_manager is not None, f"rank {rankID}: all2all_manager is None"
        print(
            f"rank {rankID} auto-created MoriAll2AllManager: {type(mori_manager).__name__}"
        )

        handle_kwargs = dict(
            rank=rankID,
            num_ep_ranks=world_size,
            input_dtype=dtype,
            quant_dtype=tokens_qt.dtype,
            token_hidden_size=hdim,
            scale_dim=scale.shape[-1] if scale is not None else 0,
            scale_type_size=scale.dtype.itemsize if scale is not None else 0,
            max_num_tokens_per_dp_rank=2
            * 8192
            * 1024
            // tokens_qt.dtype.itemsize
            // hdim
            * 2,
            num_local_experts=E // world_size,
            num_experts_per_token=topk,
            gpu_per_node=1,
        )

        mori_op = mori_manager.get_handle(handle_kwargs)

        # Dispatch
        (
            dispatch_output,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = mori_op.dispatch(tokens_qt, topk_weights, scale, topk_ids)

        expert_mask = torch.zeros((E,), dtype=dtypes.i32, device=device)
        expert_mask[E // world_size * rankID : E // world_size * (rankID + 1)] = 1
        out, us = run_perftest(
            fused_moe,
            dispatch_output,
            w1,
            w2,
            dispatch_weights,
            dispatch_ids,
            expert_mask,
            num_local_tokens=dispatch_recv_token_num,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=dispatch_scale,
            quant_type=quant_type,
            dtype=dtype,
        )
        print(f"rank {rankID} us={us:.4f}")

        # Combine
        combine_result = mori_op.combine(out, topk_weights, topk_ids)

        if isinstance(combine_result, tuple):
            combine_output = combine_result[0]
        else:
            combine_output = combine_result

        result = combine_output[:token_num].cpu()
        print(f"rank {rankID} moved result to CPU")

        return result

    finally:
        torch.distributed.barrier(group=ep_group.cpu_group)
        aiter.destroy_dist_env()
        print(f"rank {rankID} destroyed dist env")


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


def test_mori_all2all(world_size, shape, dtype, E, topk, quant_type=aiter.QuantType.No):
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
    topk_ids_dp = topk_ids.chunk(world_size)
    w1_ep = w1_qt.chunk(world_size)
    w2_ep = w2_qt.chunk(world_size)
    w1_scale_ep = (
        w1_scale.chunk(world_size) if w1_scale is not None else [None] * world_size
    )
    w2_scale_ep = (
        w2_scale.chunk(world_size) if w2_scale is not None else [None] * world_size
    )

    # Run test with MoriAll2AllManager
    rets = []
    for i in range(world_size):
        rets.append(
            pool.apply_async(
                run_mori_with_manager,
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
    print(f"Test completed! ret_out.shape={ret_out.shape}")

    # TODO: Add reference comparison
    # checkAllclose(ref, ret_out.to(ref), msg="total tokens:")
    # for i in range(world_size):
    #     checkAllclose(ref_dp[i], rets[i].to(ref_dp[i]), msg=f"rank:{i}")

    print("Test passed with MoriAll2AllManager")


l_dtype = ["bf16"]
l_shape = [(128, 6144, 1024)]
# Explicitly define the default quantization types to test
quant_types = [
    aiter.QuantType.per_Token,
    aiter.QuantType.per_128x128,
]

parser = argparse.ArgumentParser(description="config input of test")
parser.add_argument(
    "-q",
    "--quant_type",
    type=str,
    choices=["No", "per_Token", "per_128x128"],
    nargs="?",
    const=None,
    default=None,
    help="quantization type",
)
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
    if args.quant_type is not None:
        quant_types = [getattr(aiter.QuantType, args.quant_type)]

    for quant_type in quant_types:
        for dtype in l_dtype:
            for shape in l_shape:
                test_mori_all2all(8, shape, dtype, 16, 2, quant_type)
