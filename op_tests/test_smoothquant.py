# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, perftest
from aiter import dtypes
import argparse


@perftest()
def run_torch(input, x_scale, y_scale_dtype=dtypes.fp32):
    output, y_scale = aiter.pertoken_quant(
        input, x_scale=x_scale, scale_dtype=y_scale_dtype
    )
    return output, y_scale


@perftest()
def run_ck(input, x_scale, y_scale_dtype=dtypes.fp32):
    # pad stride
    output = torch.empty_strided(
        input.shape,
        (input.shape[1] + 128, 1),
        dtype=dtypes.i8,
        layout=input.layout,
        device=input.device,
    )
    y_scale = torch.empty(input.shape[0], 1, device="cuda", dtype=y_scale_dtype)
    aiter.smoothquant_fwd(output, input, x_scale, y_scale)

    return output, y_scale


def test_Smoothquant_instance(dtype, m, n, xscaleType):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    xscale = torch.randn(n, dtype=xscaleType, device="cuda")
    (a, yscale_a), avg_a = run_torch(input, x_scale=xscale)
    (b, yscale_b), avg_b = run_ck(input, x_scale=xscale)

    print(
        f"[perf] dim: {dim}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    )
    checkAllclose(a, b, rtol=0, atol=1)
    checkAllclose(yscale_a, yscale_b, rtol=1e-3, atol=1e-3)


def test_Smoothquant(l_dtype: list, l_m: list, l_n: list):
    print("\nstart layernorm2d fuse Smoothquant test")
    for scaleType in [dtypes.fp32]:
        for dtype in [dtypes.fp16, dtypes.bf16]:
            for m in l_m:
                for n in l_n:
                    test_Smoothquant_instance(dtype, m, n, xscaleType=scaleType)


if __name__ == "__main__":
    l_dtype = ["bf16", "fp16"]
    parser = argparse.ArgumentParser()
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
        "-m",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        nargs="*",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=[10, 1024, 2048],
        nargs="*",
    )
    args = parser.parse_args()
    if args.dtype is None:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    test_Smoothquant(l_dtype, args.m, args.n)
