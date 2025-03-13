# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from aiter.test_common import checkAllclose, perftest, tensor_dump
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import aiter
from einops import rearrange

block_shape = (128, 128)

@perftest()
def run_torch(x, weight, x_scale, w_scale, dtype=torch.bfloat16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    scale_n =  (n + block_shape_n - 1) // block_shape_n
    scale_k =  (k + block_shape_k - 1) // block_shape_k
    x = x.to(x_scale.dtype).view(m, k//block_shape[1], block_shape[1]) * x_scale.unsqueeze(-1)
    x = x.view(m, k)

    w_scale = rearrange(w_scale.view(-1, 1).repeat(1, block_shape_n*block_shape_k).view(scale_n, scale_k, block_shape_n, block_shape_k),
                              'num_blk_n num_blk_k blk_n blk_k -> (num_blk_n blk_n) (num_blk_k blk_k)')
    w_scale = w_scale[:n, :k]
    weight = weight.to(w_scale.dtype) * w_scale

    out = F.linear(x.to(torch.float32), weight.to(torch.float32))
    return out.to(dtype)

@perftest()
def run_gemm_ck(x, weight, x_scale, w_scale, dtype=torch.bfloat16):
    return aiter.gemm_a8w8_blockscale_CK(x, weight, x_scale, w_scale, dtype)

def test_gemm(dtype, m, n, k):
    dim = (m, n, k)
    block_shape_n, block_shape_k = block_shape
    scale_n =  (n + block_shape_n - 1) // block_shape_n
    scale_k =  (k + block_shape_k - 1) // block_shape_k
    x = (torch.rand((m, k), dtype=torch.float16, device="cuda")/10).to(torch.float8_e4m3fnuz)
    weight = (torch.rand( (n, k), dtype=torch.float16, device="cuda")/10).to(torch.float8_e4m3fnuz)
    x_scale = torch.rand([m, scale_k], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device="cuda")
    
    a, avg_a = run_torch(x, weight, x_scale, w_scale, dtype)
    b, avg_b = run_gemm_ck(x, weight, x_scale, w_scale, dtype)

    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b -1:<5.1%}"
    checkAllclose(a, b, msg="a,b: "+msg, rtol=1e-2, atol=0.01)


for dtype in [torch.bfloat16]:
    # deepseek-r1 
    for m in [16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 4096, 8192, 16384, 20480]:
        for (n, k) in [(1536,7168), (3072,1536), (576,7168), (7168, 256), (7168, 2048), (4608, 7168), (7168, 2304), (512, 7168), (4096, 512)]:
            test_gemm(dtype, m, n, k)
