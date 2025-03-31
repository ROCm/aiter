# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from aiter.test_common import checkAllclose, perftest, tensor_dump
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import aiter
from aiter.ops.shuffle import shuffle_weight


@perftest()
def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    x = x.to(torch.float32) * x_scale
    weight = weight.to(torch.float32) * w_scale
    out = F.linear(x, weight)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


@perftest()
def run_gemm_ck(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    return aiter.gemm_a8w8_CK(x, weight, x_scale, w_scale, bias, dtype)

@perftest()
def run_gemm_asm(x, weightshuffle, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    return aiter.gemm_a8w8_ASM(x, weightshuffle, x_scale, w_scale, bias)


def test_gemm(dtype, m, n, k, quantDtype=torch.int8):
    dim = (m, n, k)
    x = torch.randn((m, k), dtype=dtype, device="cuda")
    weight = torch.randn((n, k), dtype=dtype, device="cuda")
    x, x_scale = aiter.pertoken_quant(x, quant_dtype=quantDtype)
    weight, w_scale = aiter.pertoken_quant(weight, quant_dtype=quantDtype)

    bias = torch.rand([1, n], dtype=dtype, device="cuda") * 10

    # x_pad, _ = F.pad(x,(0,128), "constant", 0).split([x.shape[1], 128],dim=1)
    # print(f"{x_pad.shape=}{x_pad.stride()}")

    a, avg_a = run_torch(x, weight, x_scale, w_scale, bias, dtype)
    b, avg_b = run_gemm_ck(x, weight, x_scale, w_scale, bias, dtype)
    if dtype == torch.bfloat16 and quantDtype == torch.int8 and bias is not None:
        weightshuffle = shuffle_weight(weight,layout=(32,16))
        bias_f32 = bias.to(torch.float)
        c, avg_c = run_gemm_asm(x, weightshuffle, x_scale, w_scale, bias_f32, dtype)
    else:
        c = None
    if c is None:
        msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, {quantDtype=} torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, asm : not support, uplift: {avg_a/avg_b-1:<5.1%}"
    else:
        msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, {quantDtype=} torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, asm avg: {avg_c:<8.2f} us, uplift: {avg_a/min(avg_b,avg_c)-1:<5.1%}"
    checkAllclose(a, b, msg="a,b: "+msg, rtol=1e-2, atol=0.01)
    if c != None:
        checkAllclose(a, c, msg="\033[1A\033[2K" + "a,c: "+ msg, rtol=1e-2, atol=0.01)

for dtype in [torch.bfloat16, torch.float16]:
    for quantDtype in [torch.int8, torch.float8_e4m3fnuz]:
        # qkv_proj
        for (m, n, k) in [
            (1, 1280, 8192),
            (32, 1280, 8192),
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
            test_gemm(dtype, m, n, k, quantDtype)
        # attn_out
        for (m, n, k) in [
            (1, 8192, 1024),
            (32, 8192, 1024),
            (64, 8192, 1024),
            (128, 8192, 1024),
            (192, 8192, 1024),
            (256, 8192, 1024),
            (320, 8192, 1024),
            (512, 8192, 1024),
            (1024, 8192, 1024),
            (2048, 8192, 1024),
            (4096, 8192, 1024),
            (8192, 8192, 1024),
            (16384, 8192, 1024),
        ]:
            test_gemm(dtype, m, n, k, quantDtype)
