# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import numpy as np
import aiter
from aiter.test_common import checkAllclose, perftest
import argparse


@perftest()
def run_torch(input, x_scale, y_scale_dtype=torch.float32):
    output, y_scale = aiter.pertoken_quant(
        input, x_scale=x_scale, y_scale_dtype=y_scale_dtype)
    return output, y_scale


@perftest()
def run_ck(input, x_scale, y_scale_dtype=torch.float32):
    # pad stride
    output = torch.empty_strided(input.shape, (input.shape[1]+128, 1), dtype=torch.int8,
                         layout=input.layout, device=input.device)
    y_scale = torch.empty(
        input.shape[0], 1, device="cuda", dtype=y_scale_dtype)
    aiter.smoothquant_fwd(output,
                         input,
                         x_scale,
                         y_scale)

    return output, y_scale


def test_Smoothquant_instance(dtype, m, n, xscaleType):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    xscale = torch.randn(n, dtype=xscaleType, device="cuda")
    (a, yscale_a), avg_a = run_torch(input, x_scale=xscale)
    (b, yscale_b), avg_b = run_ck(input, x_scale=xscale)

    print(
        f"[perf] dim: {dim}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}")
    checkAllclose(a, b, rtol=0, atol=1)
    checkAllclose(yscale_a, yscale_b, rtol=1e-3, atol=1e-3)


def test_Smoothquant():
    print('\nstart layernorm2d fuse Smoothquant test')
    for scaleType in [torch.float32]:
        for dtype in [torch.float16, torch.bfloat16]:
            for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                for n in [10, 4096, 8192]:
                    test_Smoothquant_instance(
                        dtype, m, n, xscaleType=scaleType)


if __name__ == "__main__":
    test_Smoothquant()
