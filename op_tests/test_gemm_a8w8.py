# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose, perftest, benchmark
import pandas as pd


@perftest(num_iters=5)
def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    x = x.to(dtypes.fp32) * x_scale
    weight = weight.to(dtypes.fp32) * w_scale
    out = F.linear(x, weight)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


@perftest()
def run_gemm_ck(x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_CK(x, weight, x_scale, w_scale, bias, dtype)


@perftest()
def run_gemm_ck_bpreshuffle(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_bpreshuffle_CK(x, weight, x_scale, w_scale, dtype)


@perftest()
def run_gemm_asm(x, weightshuffle, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_ASM(x, weightshuffle, x_scale, w_scale, bias)


@benchmark()
def test_gemm(dtype, m, n, k, quantDtype=dtypes.i8):
    dim = (m, n, k)
    x = torch.randn((m, k), dtype=dtype, device="cuda")
    weight = torch.randn((n, k), dtype=dtype, device="cuda")
    x, x_scale = aiter.pertoken_quant(x, quant_dtype=quantDtype)
    weight, w_scale = aiter.pertoken_quant(weight, quant_dtype=quantDtype)
    weightshuffle = shuffle_weight(weight, layout=(16, 16))
    bias = torch.rand([1, n], dtype=dtype, device="cuda") * 10

    # x_pad, _ = F.pad(x,(0,128), "constant", 0).split([x.shape[1], 128],dim=1)
    # print(f"{x_pad.shape=}{x_pad.stride()}")

    a, avg_a = run_torch(x, weight, x_scale, w_scale, bias, dtype)
    b, avg_b = run_gemm_ck(x, weight, x_scale, w_scale, bias, dtype)
    err_b = checkAllclose(a, b, msg="ck: ", rtol=1e-2, atol=1e-2)
    if quantDtype != dtypes.i8:
        c, avg_c = run_gemm_ck_bpreshuffle(x, weightshuffle, x_scale, w_scale, dtype)
        c = c + bias
        err_c = checkAllclose(a, c, msg="ck bpreshuffle: ", rtol=1e-2, atol=1e-2)
    else:
        avg_c = None
        err_c = None

    avg_d = None
    err_d = None
    if dtype == dtypes.bf16 and quantDtype == dtypes.i8 and bias is not None:
        weightshuffle_asm = shuffle_weight(weight, layout=(32, 16))
        bias_f32 = bias.to(dtypes.fp32)
        d, avg_d = run_gemm_asm(x, weightshuffle_asm, x_scale, w_scale, bias_f32, dtype)
        if d is not None:
            err_d = checkAllclose(a, d, msg="asm: ", rtol=1e-2, atol=1e-2)
        else:
            avg_d = None

    return {
        "ck us": avg_b,
        "ck err": err_b,
        "ck bpreshuffle us": avg_c,
        "ck bpreshuffle err": err_c,
        "asm us": avg_d,
        "asm err": err_d,
    }


df = []
for dtype in [dtypes.bf16, dtypes.fp16]:
    for quantDtype in [dtypes.i8, dtypes.fp8]:
        for m, n, k in [
            # qkv_proj
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
            # attn_out
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
            ret = test_gemm(dtype, m, n, k, quantDtype)
            df.append(ret)
df = pd.DataFrame(df)
aiter.logger.info(f"summary:\n{df}")
