# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import torch
from einops import rearrange

import aiter
from aiter import dtypes, pertoken_quant
from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    torch_moe,
    torch_moe_blockscale,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose, run_perftest

BLOCK_SIZE_M = 32


def torch_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert, inter_dim, 1]
    fc2_scale=None,  # [expert, model_dim, 1]
    fc1_smooth_scale=None,  # [expert, 1, model_dim]
    fc2_smooth_scale=None,  # [expert, 1, inter_dim]
):
    return torch_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        fc1_scale,
        fc2_scale,
        fc1_smooth_scale,
        fc2_smooth_scale,
    )


def asm_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,
    fc2_scale=None,
    a1_scale=None,
    scale_blk=(128, 128),
):

    model_dim = hidden_states.shape[-1]
    topk = topk_ids.shape[-1]
    E = w1.shape[0]
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, out_asm = (
        moe_sorting(topk_ids, topk_weights, E, model_dim, dtype)
    )
    scale_blk_n, scale_blk_k = scale_blk
    aiter.fmoe_fp8_blockscale_g1u1(
        out_asm,
        hidden_states,
        w1,
        w2,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        a1_scale,
        fc1_scale,
        fc2_scale,
        "",
        scale_blk_n,
        scale_blk_k,
        None,
    )
    return out_asm


torch.set_default_device("cuda")


def test_fmoe(
    dtype,
    token,
    model_dim,
    inter_dim,
    scale_blks,
    E,
    topk,
    quant="No",
    use_g1u1=False,
    shared_E=0,
):
    input = torch.randn((token, model_dim), dtype=dtype)
    if use_g1u1:
        w1 = torch.randn((E + shared_E, inter_dim * 2, model_dim), dtype=dtype) / 10
    else:
        w1 = torch.randn((E + shared_E, inter_dim, model_dim), dtype=dtype)
    w2 = torch.randn((E + shared_E, model_dim, inter_dim), dtype=dtype) / 10
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    scale_blk_n, scale_blk_k = scale_blks
    quant_dtype = dtypes.fp8

    # block quant w1
    tmp = rearrange(
        w1.view(
            -1,
            w1.shape[1] // scale_blk_n,
            scale_blk_n,
            w1.shape[2] // scale_blk_k,
            scale_blk_k,
        ),
        "e num_blk_n blk_n num_blk_k blk_k -> e num_blk_n num_blk_k (blk_n blk_k)",
    ).contiguous()
    w1_q, w1_scale = pertoken_quant(tmp, quant_dtype=quant_dtype)
    w1_q = rearrange(
        w1_q.view(
            -1,
            w1.shape[1] // scale_blk_n,
            w1.shape[2] // scale_blk_k,
            scale_blk_n,
            scale_blk_k,
        ),
        "e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)",
    ).contiguous()
    w1_scale = w1_scale.view(E, -1)

    # block quant w2
    tmp = rearrange(
        w2.view(
            -1,
            model_dim // scale_blk_n,
            scale_blk_n,
            inter_dim // scale_blk_k,
            scale_blk_k,
        ),
        "e num_blk_n blk_n num_blk_k blk_k -> e num_blk_n num_blk_k (blk_n blk_k)",
    ).contiguous()
    w2_q, w2_scale = pertoken_quant(tmp, quant_dtype=quant_dtype)
    w2_q = rearrange(
        w2_q.view(
            -1,
            w2.shape[1] // scale_blk_n,
            w2.shape[2] // scale_blk_k,
            scale_blk_n,
            scale_blk_k,
        ),
        "e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)",
    ).contiguous()
    w2_scale = w2_scale.view(E, -1)

    # block quant input
    a1_q, a1_scale = pertoken_quant(
        input.view(-1, model_dim // scale_blk_k, scale_blk_k), quant_dtype=quant_dtype
    )
    a1_q = a1_q.view(-1, model_dim)
    a1_scale = a1_scale.squeeze(-1)

    # w2, fc2_scale = pertoken_quant(w2, quant_dtype=quant_dtype)

    out_ref, us_ref = run_perftest(
        torch_moe_blockscale,
        a1_q,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        scale_blks=scale_blks,
        fc1_scale=w1_scale,
        fc2_scale=w2_scale,
        a_scale=a1_scale,
        num_warmup=1,
        num_iters=2,
    )
    # out_ref2, us_ref = run_perftest(torch_moe,
    #                                 input,
    #                                 w1,
    #                                 w2,
    #                                 topk_weights,
    #                                 topk_ids,
    #                                 num_warmup=1, num_iters=2)
    # msg = '111'
    # checkAllclose(out_ref, out_ref2, rtol=0.01, atol=100, msg=msg)

    out_asm, us_ref = run_perftest(
        asm_moe_test,
        a1_q,
        shuffle_weight(w1_q, (16, 16)),
        shuffle_weight(w2_q, (16, 16)),
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        a1_scale.t().contiguous(),
        (scale_blk_n, scale_blk_k),
    )
    msg = f"[perf] a8w8 asm: {us_ref:>8.2f} us ...... {m=}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, dtype: {dtype}"
    checkAllclose(out_ref, out_asm, rtol=0.05, atol=0.05, msg=msg)


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    nargs="*",
    default=[dtypes.d_dtypes["bf16"]],
    help="""Data type.
    e.g.: -d bf16""",
)

parser.add_argument(
    "-m",
    type=int,
    nargs="*",
    default=[1, 2, 5, 16, 32, 163840],
    help="""M of mnk.
    e.g.: -m 32""",
)

parser.add_argument(
    "-dim",
    type=int,
    default=7168,
    help="""Hidden dimension. Default is 7168.
    e.g.: -dim 7168""",
)

parser.add_argument(
    "-idim",
    type=int,
    default=256,
    help="""Intermediate dimension. Default is 256.
    e.g.: -idim 256""",
)
parser.add_argument(
    "-e",
    "--expert",
    type=int,
    default=256,
    help="""Number of experts. Default is 256.
    e.g.: -e 256""",
)
parser.add_argument(
    "-k",
    "--topk",
    type=int,
    default=8,
    help="""Top-k value. Default is 8.
    e.g.: -k 8""",
)

args = parser.parse_args()

for dtype in args.dtype:
    for m in args.m:
        for dim in [args.dim]:
            for idim in [args.idim]:
                scale_blks = (128, 128)
                test_fmoe(
                    dtype,
                    m,
                    dim,
                    idim,
                    scale_blks,
                    args.expert,
                    args.topk,
                    quant="No",
                    use_g1u1=True,
                )
