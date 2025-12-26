# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import sys
import os

# Add parent directory to path to ensure we use local aiter module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from einops import repeat as eirp
from typing_extensions import List

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, perftest

block_shape = (128, 128)


@perftest(num_iters=5)
def run_torch(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x = x.to(x_scale.dtype).view(
        m, k // block_shape[1], block_shape[1]
    ) * x_scale.unsqueeze(-1)
    x = x.view(m, k)

    w_scale = rearrange(
        w_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k),
        "num_blk_n num_blk_k blk_n blk_k -> (num_blk_n blk_n) (num_blk_k blk_k)",
    )
    w_scale = w_scale[:n, :k]
    weight = weight.to(w_scale.dtype) * w_scale

    out = F.linear(x.to(dtypes.fp32), weight.to(dtypes.fp32))
    return out.to(dtype)


@perftest()
def run_gemm_ck(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_blockscale(x, weight, x_scale, w_scale, dtype)


@perftest()
def run_gemm_bpreshuffle_ck(x, weightshuffle, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_blockscale_bpreshuffle(
        x, weightshuffle, x_scale, w_scale, dtype
    )


@benchmark()
def test_gemm(dtype, m, n, k, ck_preshuffle=True):
    ret = {}
    dim = (m, n, k)
    block_shape_n, block_shape_k = block_shape
    scale_m = m
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x = (torch.rand((m, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    x_scale = torch.rand([scale_m, scale_k], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")

    a, avg_a = run_torch(x, weight, x_scale, w_scale, dtype)

    x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    gemm_x_scale = x_scale_t if ck_preshuffle else x_scale
    gemm_weight = shuffle_weight(weight, layout=(16, 16)) if ck_preshuffle else weight
    run_func = run_gemm_bpreshuffle_ck if ck_preshuffle else run_gemm_ck
    b, avg_b = run_func(x, gemm_weight, gemm_x_scale, w_scale, dtype)

    err_ck = checkAllclose(a, b, msg="ck")
    ret["ck us"] = avg_b
    ret["ck TFLOPS"] = m * n * k * 2 / avg_b / 1e6
    ret["ck TB/s"] = (x.nbytes + weight.nbytes) / avg_b / 1e6
    ret["ck err"] = err_ck

    tag = "asm"
    weight_asm = shuffle_weight(weight, layout=(32, 16))
    # kernel_name = "_ZN5aiter43fp8gemm_bf16_blockscale_BpreShuffle_128x128E"
    # c, avg_c = run_asm(x, weight_asm, x_scale, w_scale, dtype, kernel_name=kernel_name)
    c, avg_c = run_asm(x, weight_asm, x_scale, w_scale, dtype)

    err_asm = checkAllclose(a, c, msg=f"{tag}")
    ret[f"{tag} us"] = avg_c
    ret[f"{tag} TFLOPS"] = m * n * k * 2 / avg_c / 1e6
    ret[f"{tag} TB/s"] = (x.nbytes + weight.nbytes) / avg_c / 1e6
    ret[f"{tag} err"] = err_asm
    ret["asm/ck"] = avg_c / avg_b

    return ret


@perftest(num_iters=5)
def run_torch2(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]

    x_scale_ = eirp(x_scale, "m k -> m (k repeat)", repeat=block_shape_k)
    x_scale_ = x_scale_[:m, :k]

    w_scale_ = eirp(w_scale, "n k -> (n repeat) k", repeat=block_shape_n)
    w_scale_ = eirp(w_scale_, "n k -> n (k repeat)", repeat=block_shape_k)
    w_scale_ = w_scale_[:n, :k]

    x_ = x.to(x_scale.dtype) * x_scale_
    weight_ = weight.to(w_scale.dtype) * w_scale_

    out = F.linear(x_.to(dtypes.fp32), weight_.to(dtypes.fp32))
    return out.to(dtype)


@perftest()
def run_asm(x, weight, x_scale, w_scale, dtype=dtypes.bf16, kernel_name=None):
    m, k = x.shape
    n, _ = weight.shape
    out = torch.empty((m, n), dtype=dtype, device=x.device)
    return aiter.gemm_a8w8_blockscale_bpreshuffle_asm(x, weight, out, x_scale, w_scale)


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=["bf16"],
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="""M of mnk.
    e.g.: -m 32""",
)
parser.add_argument(
    "-nk",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""N&K of mnk.
    e.g.: -nk 4096,512""",
)
parser.add_argument(
    "--ck_preshuffle",
    nargs="?",
    default=[True, False],
    help="weight ck_preshuffle or not",
)

args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in ["bf16"]]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]
if args.m is not None:
    l_m = [args.m]
if args.nk is not None:
    l_nk = [args.nk]
l_preshuffle: List[bool] = args.ck_preshuffle

df = []
for dtype in [dtypes.bf16]:
    # deepseek-r1
    for m in [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        96,
        128,
        160,
        192,
        224,
        256,
        288,
        320,
        352,
        384,
        416,
        448,
        480,
        512,
        1024,
        2048,
        4096,
        6144,
        8192,
        10240,
    ]:
        for n, k in [
            (24576, 1536),
            # (32768, 512),
            # (7168, 16384),
            # (36864, 7168),
        ]:
            ret = test_gemm(dtype, m, n, k)
            df.append(ret)
df = pd.DataFrame(df)

# Configure pandas to show all columns without truncation
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.expand_frame_repr", False)

print("\n" + "=" * 150)
print("COMPLETE PERFORMANCE SUMMARY (All Columns)")
print("=" * 150)
print(df.to_string(index=False))
print("=" * 150)

aiter.logger.info(f"summary:\n{df}")
