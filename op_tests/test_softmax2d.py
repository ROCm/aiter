# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import checkAllclose, perftest
from aiter import dtypes
import argparse


@perftest()
def run_torch(input, dim=-1):
    output = F.softmax(input=input, dim=dim)
    return output


@perftest()
def run_ck(input, dim=[-1]):
    # output = aiter.softmax2d_hip(input, dim=dim)
    output = aiter.softmax2d_hip(input)
    return output


@perftest()
def run_asm(input, dim=-1):
    output = aiter.softmax(input, dim=dim)
    return output


def test_softmax2d(dtype, m, n):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    hidden_stats = torch.randn(m, n * 8, dtype=dtype, device="cuda")
    q, k, v = torch.split(hidden_stats, [6 * n, n, n], dim=1)
    input = k
    (a, *_), avg_a = run_torch(input)
    (b, *_), avg_b = run_ck(input)
    # import pdb; pdb.set_trace()
    msg = f"[perf] dim: {str(dim):<20}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, msg=msg)


l_dtype = ["bf16"]
parser = argparse.ArgumentParser(
    description="Test softmax2d performance and correctness",
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
    "-m",
    type=int,
    nargs="?",
    default=128,
    help="""Number of rows in the input tensor.
    e.g.: -m 128""",
)
parser.add_argument(
    "-n",
    type=int,
    nargs="?",
    default=8192,
    help="""Number of columns in the input tensor.
    e.g.: -n 8192""",
)
args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]
# for dtype in [dtypes.fp16, dtypes.bf16]:
#     for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
#         for n in [4096, 8192, 16384, 32768, 65536]:
#             test_softmax2d(dtype, m, n)
test_softmax2d(dtypes.bf16, 1024, 8192)

