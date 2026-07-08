# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.ops.gemm_op_a6w6 import (
    dequant_mxfp6_torch,
    quant_mxfp6_gemm,
    quant_mxfp6_torch,
)
from aiter.test_common import benchmark, checkAllclose, perftest, run_perftest

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
SCALE_GROUP_SIZE = 32
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 30)


@perftest(num_iters=5)
def run_torch(x, w, dtype):
    # fp32 reference (on GPU) that matches the mxfp6 (E2M3, per-1x32 blockscale)
    # math the kernel approximates: quantize both operands, dequantize, matmul.
    xc, xs = quant_mxfp6_torch(x)
    wc, ws = quant_mxfp6_torch(w)
    xf = dequant_mxfp6_torch(xc, xs)
    wf = dequant_mxfp6_torch(wc, ws)
    return torch.mm(xf, wf.T).to(dtype)


@benchmark()
def test_gemm(dtype, M, N, K):
    from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

    if get_gfx() not in ["gfx950"]:
        return
    ret = {}
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)

    a, avg_a = run_torch(x, w, dtype)

    # pack operands + scales into the kernel's mxfp6 layout (done once, untimed)
    xq, xs = quant_mxfp6_gemm(x)
    wq, ws = quant_mxfp6_gemm(w)

    c, us = run_perftest(
        aiter.gemm_a6w6,
        xq,
        wq,
        xs,
        ws,
        M,
        N,
        K,
    )
    err = checkAllclose(a, c, msg="unified api", catastrophic_check=True)
    ret["us"] = us
    ret["TFLOPS"] = M * N * K * 2 / us / 1e6
    ret["TB/s"] = (x.nbytes + w.nbytes) / us / 1e6
    ret["err"] = err
    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    nargs="*",
    choices=[dtypes.d_dtypes["bf16"]],
    metavar="{bf16}",
    default=[dtypes.d_dtypes["bf16"]],
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-mnk",
    "--shape",
    type=dtypes.str2tuple,
    nargs="*",
    default=[
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (16384, 16384, 16384),
        # transformer shapes
        (9450, 13824, 5120),
        (9450, 5120, 13824),
    ],
    help="""Shape of mnk.
    e.g. -mnk 8192,8192,8192""",
)

args = parser.parse_args()

df = []
for dtype in args.dtype:
    for m, n, k in args.shape:
        ret = test_gemm(dtype, m, n, k)
        df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("gemm_a6w6 summary (markdown):\n%s", df_md)
