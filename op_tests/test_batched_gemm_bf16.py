# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose, perftest
import argparse


@perftest(num_iters=5)
def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    B = x.size(0)
    M = x.size(1)
    N = weight.size(1)
    out = torch.empty(B, M, N, dtype=dtypes.bf16, device="cuda")
    for b in range(B):
        b_out = F.linear(x[b, :, :].to(dtypes.fp32), weight[b, :, :].to(dtypes.fp32))
        if bias is not None:
            b_out = b_out.to(bias[b, :, :]) + bias[b, :, :]
        out[b, :, :] = b_out
    return out.to(dtype)


@perftest()
def run_gemm_ck(x, weight, bias=None, dtype=dtypes.bf16):
    return aiter.batched_gemm_bf16_CK(x, weight, bias)


def test_gemm(dtype, b, m, n, k):
    dim = (b, m, n, k)
    x = torch.randint(-20, 20, (b, m, k), dtype=dtypes.bf16).cuda()
    weight = torch.randint(-20, 20, (b, n, k), dtype=dtypes.bf16).cuda()

    a, avg_a = run_torch(x, weight, None, dtype)
    b, avg_b = run_gemm_ck(x, weight, None, dtype)
    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, msg="a,b: " + msg, rtol=1e-2, atol=0.01)


l_dtype = ["bf16"]
l_b = [16]
l_mnk = [
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
]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-b",
    "--batch",
    type=int,
    choices=l_b,
    nargs="?",
    const=None,
    default=None,
    help="""Batch size.
    e.g.: -b 16""",
)
parser.add_argument(
    "-mnk",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""shape of mnk.
    e.g. -mnk 1280,8192,1024""",
)
args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]
if args.batch is not None:
    l_b = [args.bitch]
if args.mnk is not None:
    l_mnk = [args.m]


for dtype in l_dtype:
    for b in l_b:
        for m, n, k in l_mnk:
            test_gemm(dtype, b, m, n, k)
