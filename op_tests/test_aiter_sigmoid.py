# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose
import argparse

# from ater.test_common import checkAllclose, perftest
from torch.profiler import profile, ProfilerActivity

# input shape: torch.Size([4096, 64, 160]) (20480, 1, 128)
# other shape: torch.Size([4096, 64, 160]) (10240, 160, 1)

# input shape: torch.Size([4096, 64, 160]) (47360, 1, 296)
# other shape: torch.Size([4096, 64, 160]) (10240, 160, 1)

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-s",
    "--shape",
    type=dtypes.str2tuple,
    default=(4096, 880),
    help="""Input shape.
    e.g.: -s 4096,880""",
)
parser.add_argument(
    "-st",
    "--stride",
    type=dtypes.str2tuple,
    default=(880, 1),
    help="""Input stride.
    e.g.: -st 880,1""",
)
args = parser.parse_args()


tensor0 = torch.empty_strided(args.shape, args.stride, dtype=dtypes.fp16, device="cuda")
random_data0 = torch.rand(args.shape)
tensor0.copy_(random_data0)
# tensor0.fill_(1)

print("Shape", args.shape)
print("Stride:", args.stride)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    with_stack=True,
    with_modules=True,
    record_shapes=True,
) as prof:
    for j in range(100):
        # cache_flush1 = torch.randn(10000, 10000, requires_grad=True, device="cuda", dtype=dtypes.fp32).to(dtypes.i32)
        result = torch.sigmoid(tensor0)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    with_stack=True,
    with_modules=True,
    record_shapes=True,
) as prof:
    for j in range(100):
        # cache_flush1 = torch.randn(10000, 10000, requires_grad=True, device="cuda", dtype=dtypes.fp32).to(dtypes.i32)
        output = aiter.sigmoid(tensor0)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print(torch.equal(result, output))
checkAllclose(result, output, msg="sigmoid")
print("result:", result)
print("output:", output)
