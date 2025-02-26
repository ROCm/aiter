# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from aiter.test_common import checkAllclose, perftest, tensor_dump
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import aiter

# refer from https://github.com/sgl-project/sglang/blob/main/test/srt/test_fp8_kernel.py

QUANT_TYPE=torch.float8_e4m3fnuz # on AMD 

def _make_A(M, K, block_size, out_dtype):
    quant_A = torch.rand(
        M, K // block_size, block_size, dtype=torch.float32, device="cuda"
    )
    # -1 ~ 1
    quant_A = quant_A * 2 - 1
    # scaling abs max to fmax
    finfo = torch.finfo(out_dtype)
    fmax = finfo.max
    scaling = fmax / quant_A.abs().amax(-1, keepdim=True)
    quant_A *= scaling
    quant_A = quant_A.to(out_dtype).to(torch.float32)

    # create scale and A
    scale = torch.rand(M, K // block_size, dtype=torch.float32, device="cuda")
    scale /= fmax
    A = quant_A * scale[..., None]

    A = A.reshape(M, K)
    quant_A = quant_A.reshape(M, K).to(out_dtype)
    return A, quant_A, scale    

def _make_B(N, K, block_size, out_dtype):
    def _aligned_size(a, b):
        return (a + b - 1) // b * b

    N_aligned = _aligned_size(N, block_size)
    K_aligned = _aligned_size(K, block_size)

    quant_B = torch.rand(
        N_aligned // block_size,
        block_size,
        K_aligned // block_size,
        block_size,
        dtype=torch.float32,
        device="cuda",
    )
    quant_B = quant_B * 2 - 1

    # scaling abs max to fmax
    finfo = torch.finfo(out_dtype)
    fmax = finfo.max
    scaling = fmax / quant_B.abs().amax((1, 3), keepdim=True)
    quant_B *= scaling
    quant_B = quant_B.to(out_dtype).to(torch.float32)

    scale = torch.rand(
        N_aligned // block_size,
        1,
        K_aligned // block_size,
        1,
        dtype=torch.float32,
        device="cuda",
    )
    scale /= fmax

    B = quant_B * scale

    B = B.reshape(N_aligned, K_aligned)[:N, :K]
    quant_B = quant_B.reshape(N_aligned, K_aligned).to(out_dtype)[:N, :K]
    scale = scale.reshape(N_aligned // block_size, K_aligned // block_size)
    return B, quant_B, scale


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
    return aiter.gemm_a8w8_blockscale_CK(x, weight, x_scale, w_scale)

def test_gemm(dtype, m, n, k):
    block_size=128
    x_gt, x_quant, x_scale = _make_A(
        M=m, K=k, block_size=block_size, out_dtype=QUANT_TYPE
    )
    w_gt, w_quant, w_scale = _make_B(
        N=n, K=k, block_size=block_size, out_dtype=QUANT_TYPE
    )
    # C_gt = A.to(self.output_type) @ B.to(self.output_type)
    a, avg_a = run_torch(x_quant, w_quant, x_scale, w_scale, block_size, dtype)
    b, avg_b = run_gemm_ck(x_quant, w_quant, x_scale, w_scale, dtype)

    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b -1:<5.1%}"
    checkAllclose(a, b, msg="a,b: "+msg, rtol=1e-2, atol=0.01)
    if c != None:
        checkAllclose(a, c, msg="\033[1A\033[2K" + "a,c: "+ msg, rtol=1e-2, atol=0.01)


for dtype in [torch.bfloat16]:
    # qkv_proj
    for (m, n, k) in [
        (1, 1280, 8192),
        (32, 1280, 8192), # debug
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
        (256, 1280, 8192),
        (320, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        (8192, 1280, 8192),
        (16384, 1280, 8192),
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
