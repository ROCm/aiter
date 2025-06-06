# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose, perftest
import  argparse
import itertools


@perftest(num_iters=5)
def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    B = x.size(0)
    M = x.size(1)
    N = weight.size(1)
    out = torch.empty(B, M, N, dtype=dtypes.bf16, device="cuda")
    for b in range(B):
        b_x = F.linear(x[b, :, :].to(dtypes.fp32), weight[b, :, :].to(dtypes.fp32))
        b_scale = torch.matmul(x_scale[b, :, :], w_scale[b, :, :])
        b_out = torch.mul(b_x, b_scale)
        if bias is not None:
            b_out = b_out.to(bias[b, :, :]) + bias[b, :, :]
        out[b, :, :] = b_out
    return out.to(dtype)


@perftest()
def run_gemm_ck(x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    return aiter.batched_gemm_a8w8_CK(x, weight, x_scale, w_scale, bias)


def test_gemm(dtype, b, m, n, k):
    dim = (b, m, n, k)
    x = torch.randint(-20, 20, (b, m, k), dtype=dtypes.i8).cuda()
    weight = torch.randint(-20, 20, (b, n, k), dtype=dtypes.i8).cuda()
    x_scale = torch.rand([b, m, 1], dtype=dtypes.fp32).cuda() + 1e-6
    w_scale = torch.rand([b, 1, n], dtype=dtypes.fp32).cuda() + 1e-6

    a, avg_a = run_torch(x, weight, x_scale, w_scale, None, dtype)
    b, avg_b = run_gemm_ck(x, weight, x_scale, w_scale, None, dtype)
    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, msg="a,b: " + msg, rtol=1e-2, atol=0.01)

l_dtype = ['bf16']
l_b = [16]
l_m = [1, 32, 64, 128, 192, 256, 320, 512, 1024, 2048, 4096, 8192]
l_n = [1028, 8192]
l_k = [8192, 1024]

parser = argparse.ArgumentParser(description='config input of test')
parser.add_argument('-d', '--dtype',
                    type=str,
                    choices=l_dtype,
                    nargs='?',
                    const=None,
                    default=None,
                    help='data type')
parser.add_argument('-b', '--batch',
                    type=int,
                    choices=l_b,
                    nargs='?',
                    const=None,
                    default=None,
                    help='batch size')
parser.add_argument('-m',
                    type=int,
                    choices=l_m,
                    nargs='?',
                    const=None,
                    default=None,
                    help='m: Represents the number of rows in the output matrix ( C ) and the first input matrix ( A ).')
parser.add_argument('-n',
                    type=int,
                    choices=l_n,
                    nargs='?',
                    const=None,
                    default=None,
                    help='n: Represents the number of columns in the output matrix ( C ) and the second input matrix ( B ).')
parser.add_argument('-k',
                    type=int,
                    choices=l_k,
                    nargs='?',
                    const=None,
                    default=None,
                    help='k: Represents the number of columns in the first input matrix ( A ) and the number of rows in the second input matrix ( B ).')
args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]
if args.batch is not None:
    l_b = [args.bitch]
if args.m is not None:
    l_m = [args.m]
if args.n is not None:
    l_n = [args.n]
if args.k is not None:
    l_k = [args.k]

for dtype in l_dtype:
    for b, m, n, k in itertools.product(l_b, l_m, l_n, l_k):
        test_gemm(dtype, b, m, n, k)