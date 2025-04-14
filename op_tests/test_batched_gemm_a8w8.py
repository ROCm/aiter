# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from aiter.test_common import checkAllclose, perftest, tensor_dump
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import aiter


@perftest(num_iters=5)
def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    B = x.size(0)
    M = x.size(1)
    N = weight.size(1)
    out = torch.empty(B, M, N, dtype=torch.bfloat16, device="cuda")
    for b in range(B):
        b_x = F.linear(x[b, :, :].to(torch.float32), weight[b, :, :].to(torch.float32))
        b_scale = torch.matmul(x_scale[b, :, :], w_scale[b, :, :])
        b_out = torch.mul(b_x, b_scale)
        if bias is not None:
            b_out = b_out.to(bias[b, :, :]) + bias[b, :, :]
        out[b, :, :] = b_out
    return out.to(dtype)

@perftest()
def run_gemm_ck(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    return aiter.batched_gemm_a8w8_CK(x, weight, x_scale, w_scale, bias)


def test_gemm(dtype, b, m, n, k):
    dim = (b, m, n, k)
    x = torch.randint(-20, 20, (b, m, k), dtype=torch.int8).cuda()
    weight = torch.randint(-20, 20, (b, n, k), dtype=torch.int8).cuda()
    x_scale = torch.rand([b, m, 1], dtype=torch.float32).cuda() + 1e-6
    w_scale = torch.rand([b, 1, n], dtype=torch.float32).cuda() + 1e-6

    a, avg_a = run_torch(x, weight, x_scale, w_scale, None, dtype)
    b, avg_b = run_gemm_ck(x, weight, x_scale, w_scale, None, dtype)
    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, msg="a,b: "+msg, rtol=1e-2, atol=0.01)


for dtype in [torch.bfloat16]:
    # qkv_proj
    for (b, m, n, k) in [
        (16, 1, 1280, 8192),
        (16, 32, 1280, 8192),
        (16, 64, 1280, 8192),
        (16, 128, 1280, 8192),
        (16, 192, 1280, 8192),
        (16, 256, 1280, 8192),
        (16, 320, 1280, 8192),
        (16, 512, 1280, 8192),
        (16, 1024, 1280, 8192),
        (16, 2048, 1280, 8192),
        (16, 4096, 1280, 8192),
        (16, 8192, 1280, 8192),
    ]:
        test_gemm(dtype, b, m, n, k)
    # attn_out
    for (b, m, n, k) in [
        (16, 1, 8192, 1024),
        (16, 32, 8192, 1024),
        (16, 64, 8192, 1024),
        (16, 128, 8192, 1024),
        (16, 192, 8192, 1024),
        (16, 256, 8192, 1024),
        (16, 320, 8192, 1024),
        (16, 512, 8192, 1024),
        (16, 1024, 8192, 1024),
        (16, 2048, 8192, 1024),
        (16, 4096, 8192, 1024),
        (16, 8192, 8192, 1024),
    ]:
        test_gemm(dtype, b, m, n, k)
