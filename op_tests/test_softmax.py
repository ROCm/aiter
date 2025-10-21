# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import checkAllclose, perftest
from aiter import dtypes
from aiter.ops.triton.softmax import softmax
import argparse


@perftest()
def run_torch(x, dim=-1):
    return F.softmax(x, dim=dim)


@perftest()
def run_triton(x, dim=-1):
    return softmax(x)


def test_softmax(dtype, m, n):
    shape = (m, n)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    (a, *_), avg_a = run_torch(x)
    (b, *_), avg_b = run_triton(x)
    msg = f"[perf] dim: {str(-1):<20}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, triton avg: {avg_b:<8.2f} us"
    checkAllclose(a, b, msg=msg)


l_dtype = ["fp16", "bf16"]
l_m = [1, 2, 4, 8, 16, 32, 64, 128, 256]
l_n = [4096, 8192, 16384, 32768, 65536]
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
    "-m",
    "--m",
    type=int,
    nargs="?",
    default=None,
    help="""M of mnk.
    e.g.: -m 32""",
)
parser.add_argument(
    "-n",
    "--n",
    type=int,
    nargs="?",
    default=None,
    help="""N of mnk.
    e.g.: -n 1024""",
)

args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]
if args.m is not None:
    l_m = [args.m]
if args.n is not None:
    l_n = [args.n]

print("\nstart softmax triton test")
for dtype in l_dtype:
    for m in l_m:
        for n in l_n:
            test_softmax(dtype, m, n)
