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
def run_torch(x):
    return torch.softmax(x, axis=1)


@perftest()
def run_triton(x):
    return softmax(x)


class ComputeGBps:
    def __init__(self, total_bytes):
        self.total_bytes = total_bytes
    def __call__(self, time_us):
        return self.total_bytes / time_us / 1000


def test_softmax(dtype, m, n):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    (a, *_), avg_a = run_torch(input)
    cgbps = ComputeGBps(input.nbytes + a.nbytes)
    (b, *_), avg_b = run_triton(input)
    msg = f"[perf] dim: {str(dim):<20}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, aiter.triton avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    msg += f"\n[throughput] dim: {str(dim):<20}, dtype: {dtype}, torch avg: {cgbps(avg_a):<8.2f} GBps, aiter.triton avg: {cgbps(avg_b):<8.2f} GBps"
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

print("\nstart test")
for dtype in l_dtype:
    for m in l_m:
        for n in l_n:
            test_softmax(dtype, m, n)
