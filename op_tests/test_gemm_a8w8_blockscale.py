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
def run_torch(x, weight, x_scale, w_scale, block_size, dtype=torch.bfloat16):
    m, k = x.shape
    n = weight.shape[0]
    x = x.to(x_scale.dtype).view(m, k//block_size, block_size) * x_scale[..., None]    
    x = x.reshape(m, k)
    weight = weight.to(w_scale.dtype).view(n//block_size, block_size, k//block_size, block_size) * w_scale.unsqueeze(1).unsqueeze(-1)
    weight = weight.reshape(n, k)
    out = F.linear(x, weight)
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
    
    a, avg_a = run_torch(x, weight, x_scale, w_scale, block_shape_n, dtype)
    b, avg_b = run_gemm_ck(x, weight, x_scale, w_scale, dtype)

    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b -1:<5.1%}"
    checkAllclose(a, b, msg="a,b: "+msg, rtol=1e-2, atol=0.01)


for dtype in [torch.bfloat16]:
    # qkv_proj
    for (m, n, k) in [
        (1, 1280, 8192),
        (32, 1280, 8192), # debug
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
    ]:
        test_gemm(dtype, m, n, k)
    # # attn_out
    # for (m, n, k) in [
    #     (1, 8192, 1024),
    #     (32, 8192, 1024),
    #     (64, 8192, 1024),
    #     (128, 8192, 1024),
    #     (192, 8192, 1024),
    #     (256, 8192, 1024),
    #     (320, 8192, 1024),
    #     (512, 8192, 1024),
    #     (1024, 8192, 1024),
    #     (2048, 8192, 1024),
    #     (4096, 8192, 1024),
    #     (8192, 8192, 1024),
    #     (16384, 8192, 1024),
    # ]:
    #     test_gemm(dtype, m, n, k)
