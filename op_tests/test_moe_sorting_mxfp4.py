# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import torch
import aiter
from aiter.test_common import checkAllclose, benchmark, run_perftest, perftest
from aiter.fused_moe import moe_sorting, fused_topk
from aiter import get_torch_quant, dtypes
from aiter.utility import fp4_utils
import pandas as pd
import itertools
import argparse

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
torch.set_printoptions(threshold=float("inf"))


@perftest()
def run_torch(scale, sorted_ids, num_valid_ids, token_num):
    topk = 1
    if len(scale.shape) == 3:
        topk = scale.shape[1]
        scale = scale.view(-1, scale.shape[-1])
    sorted_ids[num_valid_ids:] = token_num
    topk_ids = sorted_ids >> 24
    sorted_ids = sorted_ids & 0xFFFFFF
    mask = sorted_ids == token_num
    if topk > 1:
        sorted_ids = sorted_ids * topk + topk_ids
    sorted_ids[mask] = 0  # set to 0 to avoid overflow
    scale = scale[sorted_ids]
    scale[mask] = 0
    sm, sn = scale.shape
    tmp = torch.zeros(
        ((sm + 31) // 32 * 32, sn), dtype=scale.dtype, device=scale.device
    )
    tmp[:sm, :sn] = scale
    scale = tmp
    sm, sn = scale.shape
    scale = scale.view(sm // 32, 2, 16, sn // 8, 2, 4)
    scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
    ref = scale.view(-1, sn)
    return ref


@benchmark()
def test_moe_mxfp4_sort(dtype, token_num, model_dim, E, topk, block_size, stage):
    input = torch.randn((token_num, model_dim), dtype=dtype)
    score = torch.randn((token_num, E), dtype=dtype)

    topk_weights, topk_ids = fused_topk(input, score, topk, True)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids,
        topk_weights,
        E,
        model_dim,
        dtype,
    )
    if stage == "stage1":
        scale = torch.arange(token_num * model_dim // 32, dtype=torch.uint8)
        scale = scale.view(token_num, model_dim // 32)
    else:
        scale = torch.arange(token_num * topk * model_dim // 32, dtype=torch.uint8)
        scale = scale.view(token_num, topk, model_dim // 32)
    ref, us_ref = run_torch(scale.clone(), sorted_ids.clone(), num_valid_ids, token_num)
    sorted_mxfp4_scale, us = run_perftest(
        fp4_utils.moe_mxfp4_sort,
        scale,
        sorted_ids,
        num_valid_ids,
        token_num,
        block_size,
    )

    num_valid_ids = num_valid_ids.item()
    num_valid_ids = (num_valid_ids + block_size - 1) // block_size * block_size

    err = checkAllclose(
        ref[:num_valid_ids],
        sorted_mxfp4_scale[:num_valid_ids].view(torch.uint8),
        msg="sorted_mxfp4_scale",
    )
    return {"us_ref": us_ref, "us": us, "err": err}


l_dtype = ["bf16"]
list_dim = [4096, 6144, 8192]
list_Expert = [32, 256, 257, 512]
list_topk = [5, 8]
list_m = [1, 31, 64, 128, 256, 10000, 163840]
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
    "--dim",
    choices=list_dim,
    type=int,
    default=None,
)
parser.add_argument(
    "-e",
    "--expert",
    choices=list_Expert,
    type=int,
    default=None,
)
parser.add_argument(
    "-t",
    "--topk",
    choices=list_topk,
    type=int,
    default=None,
)
parser.add_argument(
    "-m",
    type=int,
    default=None,
)

args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]

if args.dim is not None:
    list_dim = [args.dim]
if args.expert is not None:
    list_Expert = [args.expert]
if args.topk is not None:
    list_topk = [args.topk]
if args.m is not None:
    list_m = [args.m]

df = []
for dtype in l_dtype:
    for (
        dim,
        E,
        topk,
        m,
    ) in itertools.product(list_dim, list_Expert, list_topk, list_m):
        ret = test_moe_mxfp4_sort(dtype, m, dim, E, topk, 32, "stage1")
        df.append(ret)
df = pd.DataFrame(df)
aiter.logger.info(f"summary:\n{df}")

df = []
for dtype in l_dtype:
    for (
        dim,
        E,
        topk,
        m,
    ) in itertools.product(list_dim, list_Expert, list_topk, list_m):
        ret = test_moe_mxfp4_sort(dtype, m, dim, E, topk, 32, "stage2")
        df.append(ret)
df = pd.DataFrame(df)
aiter.logger.info(f"summary:\n{df}")
