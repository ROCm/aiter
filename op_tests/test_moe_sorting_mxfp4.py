# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import torch
import aiter
from aiter.test_common import checkAllclose, benchmark, run_perftest
from aiter.fused_moe import moe_sorting, fused_topk
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort
from aiter.jit.utils.chip_info import get_gfx
from aiter import get_torch_quant, dtypes
from aiter.utility import fp4_utils
import pandas as pd
import itertools
import argparse

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
torch.set_printoptions(threshold=float("inf"))


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
    scale.view(torch.uint8)[mask] = 0
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


def run_split_quant_sort(scale, input, sorted_ids, num_valid_ids, topk, block_size):
    out, scale_ = aiter.per_1x32_f4_quant_hip(input, None, dtypes.fp4x2)
    token_num = input.numel() // (input.shape[-1] * topk)
    aiter.mxfp4_moe_sort_hip(
        scale,
        scale_,
        sorted_ids,
        num_valid_ids,
        token_num,
        input.shape[-1],
        topk,
        block_size,
    )
    return out, scale


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
    num_valid_ids = num_valid_ids[0]
    if stage == "stage1":
        scale = torch.arange(token_num * model_dim // 32, dtype=torch.uint8)
        scale = scale.view(token_num, model_dim // 32)
        topk = 1
    else:
        scale = torch.arange(token_num * topk * model_dim // 32, dtype=torch.uint8)
        scale = scale.view(token_num, topk, model_dim // 32)
    ref = run_torch(scale.clone(), sorted_ids.clone(), num_valid_ids, token_num)
    triton_scale, triton_us = run_perftest(
        fp4_utils.moe_mxfp4_sort,
        scale,
        sorted_ids,
        num_valid_ids,
        token_num,
        block_size,
    )

    hip_scale = torch.zeros(
        ((sorted_ids.shape[0] + 31) // 32 * 32, model_dim // 32),
        dtype=torch.uint8,
        device=input.device,
    )
    _, hip_us = run_perftest(
        aiter.mxfp4_moe_sort_hip,
        hip_scale,
        scale,
        sorted_ids,
        num_valid_ids,
        token_num,
        model_dim,
        topk,
        block_size,
    )

    num_valid_ids = num_valid_ids.item()
    num_valid_ids = (num_valid_ids + block_size - 1) // block_size * block_size

    triton_err = checkAllclose(
        ref[:num_valid_ids],
        triton_scale[:num_valid_ids].view(torch.uint8),
        msg="sorted_mxfp4_scale",
    )

    hip_err = checkAllclose(
        ref[:num_valid_ids].view(torch.uint8),
        hip_scale[:num_valid_ids].view(torch.uint8),
        msg="hip sorted_mxfp4_scale",
    )
    return {
        "triton_us": triton_us,
        "triton_err": triton_err,
        "hip_us": hip_us,
        "hip_err": hip_err,
    }


@benchmark()
def test_moe_mxfp4_quant_sort(dtype, token_num, model_dim, E, topk, block_size, stage):
    if get_gfx().startswith("gfx94"):
        return {}
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
    num_valid_ids = num_valid_ids[0]
    if stage != "stage1":
        input = torch.randn((token_num * topk, model_dim), dtype=dtype)
    else:
        topk = 1
    ref_out, scale = get_torch_quant(aiter.QuantType.per_1x32)(
        input, quant_dtype=dtypes.fp4x2
    )
    ref_scale = run_torch(scale.clone(), sorted_ids.clone(), num_valid_ids, token_num)

    split_scale = torch.zeros(
        ((sorted_ids.shape[0] + 31) // 32 * 32, model_dim // 32),
        dtype=torch.uint8,
        device=input.device,
    )
    (split_out, split_scale), split_us = run_perftest(
        run_split_quant_sort,
        split_scale,
        input,
        sorted_ids,
        num_valid_ids,
        topk,
        block_size,
    )

    hip_scale = torch.zeros(
        ((sorted_ids.shape[0] + 31) // 32 * 32, model_dim // 32),
        dtype=torch.uint8,
        device=input.device,
    )
    hip_out = torch.empty(
        (token_num * topk, model_dim // 2),
        dtype=dtypes.fp4x2,
        device=input.device,
    )
    _, hip_us = run_perftest(
        aiter.fused_dynamic_mxfp4_quant_moe_sort_hip,
        hip_out,
        hip_scale,
        input,
        sorted_ids,
        num_valid_ids,
        topk,
        block_size,
    )

    (triton_out, triton_scale), triton_us = run_perftest(
        fused_dynamic_mxfp4_quant_moe_sort,
        input,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token_num,
        topk=topk,
        block_size=block_size,
    )

    mask = ref_scale == 0
    triton_scale = triton_scale[: ref_scale.shape[0]]
    triton_scale.view(torch.uint8)[mask] = 0
    num_valid_ids = num_valid_ids.item()
    num_valid_ids = (num_valid_ids + block_size - 1) // block_size * block_size

    checkAllclose(ref_out.view(torch.uint8), hip_out.view(torch.uint8), msg="hip out")
    hip_err = checkAllclose(
        ref_scale[:num_valid_ids].view(torch.uint8),
        hip_scale[:num_valid_ids].view(torch.uint8),
        msg="hip sorted_mxfp4_scale",
    )

    # checkAllclose(ref_out.view(torch.uint8), triton_out.view(torch.uint8), msg="triton out")
    triton_err = checkAllclose(
        ref_scale[:num_valid_ids].view(torch.uint8),
        triton_scale[:num_valid_ids].view(torch.uint8),
        msg="triton sorted_mxfp4_scale",
    )

    split_err = checkAllclose(
        ref_scale[:num_valid_ids].view(torch.uint8),
        split_scale[:num_valid_ids].view(torch.uint8),
        msg="split sorted_mxfp4_scale",
    )

    return {
        "triton_us": triton_us,
        "triton_err": triton_err,
        "hip_us": hip_us,
        "hip_err": hip_err,
        "split_us": split_us,
        "split_err": split_err,
    }


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["bf16"]],
    nargs="*",
    default=[dtypes.d_dtypes["bf16"]],
    metavar="{bf16}",
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-dim1",
    type=int,
    nargs="*",
    default=[4096, 7168],
    help="""Model dimension for stage1.
    e.g.: -dim1 4096""",
)
parser.add_argument(
    "-dim2",
    type=int,
    nargs="*",
    default=[256, 2048],
    help="""Inter dimension for stage2.
    e.g.: -dim2 256""",
)
parser.add_argument(
    "-ek",
    "--expert_topk",
    type=dtypes.str2tuple,
    nargs="*",
    default=[[32, 5], [256, 8], [512, 8]],
    help="""Number of experts.
    e.g.: -e 32,5""",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="*",
    default=[1, 64, 128, 256, 1024, 2050, 4200, 10000, 163840],
    help="""M of mnk.
    e.g.: -m 64""",
)
parser.add_argument(
    "-bm",
    "--block_m",
    type=int,
    default=32,
    choices=[16, 32, 64, 80, 128],
    help="""Block M.
    e.g.: -bm 64""",
)

args = parser.parse_args()

df = []
for dtype in args.dtype:
    for (
        dim,
        (E, topk),
        m,
    ) in itertools.product(args.dim1, args.expert_topk, args.m):
        ret = test_moe_mxfp4_sort(dtype, m, dim, E, topk, args.block_m, "stage1")
        df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("moe_sorting_mxfp4_stage1 summary (markdown):\n%s", df_md)

df = []
for dtype in args.dtype:
    for (
        dim,
        (E, topk),
        m,
    ) in itertools.product(args.dim2, args.expert_topk, args.m):
        ret = test_moe_mxfp4_sort(dtype, m, dim, E, topk, args.block_m, "stage2")
        df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("moe_sorting_mxfp4_stage2 summary (markdown):\n%s", df_md)

df = []
for dtype in args.dtype:
    for (
        dim,
        (E, topk),
        m,
    ) in itertools.product(args.dim1, args.expert_topk, args.m):
        ret = test_moe_mxfp4_quant_sort(dtype, m, dim, E, topk, args.block_m, "stage1")
        df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("moe_mxfp4_quant_sort_stage1 summary (markdown):\n%s", df_md)

df = []
for dtype in args.dtype:
    for (
        dim,
        (E, topk),
        m,
    ) in itertools.product(args.dim2, args.expert_topk, args.m):
        ret = test_moe_mxfp4_quant_sort(dtype, m, dim, E, topk, args.block_m, "stage2")
        df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("moe_mxfp4_quant_sort_stage2 summary (markdown):\n%s", df_md)
