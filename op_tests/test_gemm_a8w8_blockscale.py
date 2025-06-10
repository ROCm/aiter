# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose, perftest, benchmark
from einops import rearrange
from einops import repeat as eirp
import pandas as pd
import argparse
from aiter import dtypes

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
    return aiter.gemm_a8w8_blockscale_CK(x, weight, x_scale, w_scale, dtype)


@benchmark()
def test_gemm(dtype, m, n, k):
    dim = (m, n, k)
    block_shape_n, block_shape_k = block_shape
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x = (torch.rand((m, k), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
    x_scale = torch.rand([m, scale_k], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")

    a, avg_a = run_torch(x, weight, x_scale, w_scale, dtype)
    b, avg_b = run_gemm_ck(x, weight, x_scale, w_scale, dtype)

    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b -1:<5.1%}"
    checkAllclose(a, b, msg="a,b: " + msg, rtol=1e-2, atol=0.01)

    return {"us": avg_b}


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
def run_asm(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.flatmm_a8w8_blockscale_ASM(x, weight, x_scale, w_scale, dtype)


@benchmark()
def test_gemm_asm(dtype, m, n, k):
    dim = (m, n, k)
    block_shape_n, block_shape_k = block_shape
    scale_m = m
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k

    x = (torch.rand((m, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)

    x_scale = torch.rand([scale_k, scale_m], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_k, scale_n], dtype=dtypes.fp32, device="cuda")

    x_scale_trans = torch.transpose(x_scale, 0, 1)
    w_scale_trans = torch.transpose(w_scale, 0, 1)

    flat_weight = weight.view(n // 16, 16, k // 64, 4, 16)
    flat_weight = flat_weight.permute(0, 2, 3, 1, 4).contiguous()
    flat_weight = flat_weight.view(n, -1)

    a, avg_a = run_torch2(x, weight, x_scale_trans, w_scale_trans, dtype)
    b, avg_b = run_asm(x, flat_weight, x_scale, w_scale, dtype)

    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, asm avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b -1:<5.1%}"
    checkAllclose(a, b, msg="a,b: " + msg, rtol=1e-2, atol=0.01)


l_dtype = ['bf16']
l_m =  [16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 4096, 8192, 16384, 20480]
l_nk =  [
            (1536, 7168),
            (3072, 1536),
            (576, 7168),
            (7168, 256),
            (7168, 2048),
            (4608, 7168),
            (7168, 2304),
            (512, 7168),
            (4096, 512),
        ]

parser = argparse.ArgumentParser(description='config input of test')
parser.add_argument('-d', '--dtype',
                    type=str,
                    choices=l_dtype,
                    nargs='?',
                    const=None,
                    default=None,
                    help='data type')
parser.add_argument('-m',
                    type=int,
                    choices=l_m,
                    nargs='?',
                    const=None,
                    default=None,
                    help='shape')
parser.add_argument('-nk',
                    type=dtypes.str2tuple,
                    choices=l_nk,
                    nargs='?',
                    const=None,
                    default=None,
                    help='shape')

args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]
if args.m is not None:
    l_m = [args.m]
if args.nk is not None:
    l_nk = [args.nk]

df = []
for dtype in l_dtype:
    # deepseek-r1
    for m in l_m:
        for n, k in l_nk:
            ret = test_gemm(dtype, m, n, k)
            df.append(ret)
df = pd.DataFrame(df)
aiter.logger.info(f"summary:\n{df}")
# for dtype in [dtypes.fp16]:
#     # deepseek-r1
#     for m in [16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 4096, 8192, 16384, 20480]:
#         for (n, k) in [(1536,7168), (3072,1536), (7168, 256), (7168, 2048), (4608, 7168), (7168, 2304), (512, 7168), (4096, 512)][1:2]:
#             test_gemm_asm(dtype, m, n, k)
#             break
