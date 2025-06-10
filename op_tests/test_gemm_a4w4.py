# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, benchmark, perftest
from aiter import dtypes
from aiter.utility import fp4_utils
import random
import itertools
import argparse
from aiter import dtypes

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
SCALE_GROUP_SIZE = 32


@perftest(num_iters=5)
def run_torch(x, w, x_scales, w_scales, dtype):
    m, k = x.shape
    n, k = w.shape
    # First convert the x and w inputs to f32.
    x_f32 = fp4_utils.mxfp4_to_f32(x)
    w_f32 = fp4_utils.mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales[:m]
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_scales_f32 = fp4_utils.e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales[:n]
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_scales_f32 = fp4_utils.e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)[:m, :n]


@perftest()
def run_triton(x, w, x_scales, w_scales, out, dtype=dtypes.bf16):
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4

    gemm_afp4wfp4(x, w, out, x_scales, w_scales, dtype)
    return out


@perftest()
def run_gemm_asm(x, weightshuffle, x_scale, w_scale, out, bias=None, dtype=dtypes.bf16):
    return aiter.gemm_a4w4_asm(x, weightshuffle, x_scale, w_scale, out, bias)


@benchmark()
def test_gemm(dtype, M, N, K):
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() not in ["gfx950"]:
        return
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    _, x_scales = quant_func(x, shuffle=False)
    _, w_scales = quant_func(w, shuffle=False)
    x, x_scales_shuffle = quant_func(x, shuffle=True)
    w, w_scales_shuffle = quant_func(w, shuffle=True)
    out1 = torch.empty(M, N, dtype=dtype)
    out2 = torch.empty((M + 255) // 256 * 256, N, dtype=dtype)
    bias_f32 = torch.zeros(M, N, dtype=dtype)
    x_scales = x_scales.view(torch.uint8)
    w_scales = w_scales.view(torch.uint8)

    a, avg_a = run_torch(x, w, x_scales, w_scales, dtype)
    b, avg_b = run_triton(x, w.T, x_scales, w_scales, out1, dtype)
    err0 = checkAllclose(a, b, msg="triton")
    avg_c = None
    tflops_c = None
    tbs_c = None
    c, avg_c = run_gemm_asm(x, w, x_scales_shuffle, w_scales_shuffle, out2, bias_f32)
    err1 = checkAllclose(a, c[:M], msg="asm   ")
    tflops_c = M * N * K * 2 / avg_c / 1e6
    tbs_c = (x.nbytes + w.nbytes) / avg_c / 1e6
    return {
        "triton": avg_b,
        "asm": avg_c,
        "triton err": err0,
        "asm err": err1,
        "asm TFLPOS": tflops_c,
        "asm TB/s": tbs_c,
    }


import pandas as pd

l_dtype = ["bf16"]
l_mnk = [
    # pure_compute
    (16384, 16384, 16384),
    (32768, 106496, 16384),
    (32768, 16384, 53248),
    (32768, 18432, 16384),
    (32768, 16384, 16384),
    (128, 106496, 16384),
    (128, 16384, 53248),
    (128, 18432, 16384),
    (128, 16384, 16384),
    (64, 106496, 16384),
    (64, 16384, 53248),
    (64, 18432, 16384),
    (64, 16384, 16384),
    (32, 106496, 16384),
    (32, 16384, 53248),
    (32, 18432, 16384),
    (32, 16384, 16384),
    # qkv_proj
    (1, 1280, 8192),
    (64, 1280, 8192),
    (127, 1280, 8192),
    (129, 1280, 8192),
    (65, 1280, 8192),
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
]

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
    "-s",
    "--shape",
    type=dtypes.str2tuple,
    choices=l_mnk,
    nargs="?",
    const=None,
    default=None,
    help="shape",
)

args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]
if args.shape is not None:
    l_mnk = [args.shape]

df = []
for dtype in l_dtype:
    for m, n, k in l_mnk:
        ret = test_gemm(dtype, m, n, k)
        df.append(ret)
df = pd.DataFrame(df)
aiter.logger.info(f"summary:\n{df}")
