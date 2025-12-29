# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import itertools
import random
import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose, benchmark, run_perftest
from aiter.jit.utils.chip_info import get_gfx
from aiter import deepgemm
import pandas as pd
import argparse

# pd.set_option('display.max_rows', 200)
# pd.set_option('display.max_columns', 100)
# pd.set_option('display.width', 1000)
TEST_NUM_ITERS = 100


# @perftest(num_iters=TEST_NUM_ITERS)
def run_torch(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    if x_scale is not None:
        x = x.to(dtypes.fp32) * x_scale
    if w_scale is not None:
        weight = weight.to(dtypes.fp32) * w_scale

    out = torch.einsum("gmk,gnk->gmn", x, weight).to(dtype)

    return out.to(dtype)


@benchmark()
def test_deepgemm(
    num_groups: int,
    expect_m: int,
    k: int,
    n: int,
    XQDType,
    WQDType,
    quant_dtype=aiter.dtypes.fp8,
    dtypes=torch.bfloat16,
):
    # TODO: add support for gfx950
    if get_gfx() not in ["gfx942"]:
        return
    max_m = 256 if expect_m < 128 else 2 * expect_m
    x = torch.randn((num_groups, max_m, k), device="cuda", dtype=dtypes)
    weight = torch.randn((num_groups, n, k), device="cuda", dtype=dtypes)
    out = torch.zeros((num_groups, max_m, n), device="cuda", dtype=dtypes)

    torch_quant = aiter.get_torch_quant(quant_dtype)

    x, x_scale = torch_quant(x, quant_dtype=XQDType)
    weight, w_scale = torch_quant(weight, quant_dtype=WQDType)

    ref_out = run_torch(x, weight, x_scale, w_scale, dtype=dtypes)

    masked_m = torch.empty((num_groups,), device="cuda", dtype=torch.int)
    for j in range(num_groups):
        masked_m[j] = int(expect_m * random.uniform(0.7, 1.3))
        ref_out[j][masked_m[j] :] = 0.0
    assert masked_m.amax().item() <= max_m

    weightshuffle = shuffle_weight(weight, layout=(16, 16))

    out, us = run_perftest(
        deepgemm,
        x,
        weightshuffle,
        out,
        masked_m,
        x_scale,
        w_scale,
    )

    err = checkAllclose(out, ref_out, msg="")

    tflops = masked_m.sum() * k * n * 2 / us / 1e6
    size_a = masked_m.sum() * k * x.element_size()
    size_b = (
        min(masked_m.sum() / num_groups, 1) * num_groups * k * n * weight.element_size()
    )
    size_c = masked_m.sum() * n * out.element_size()

    bandwidth = (size_a + size_b + size_c) / us / 1e3

    return {
        "us": us,
        "err": err,
        "tflops": f"{tflops.item():.2f}TFLOPs",
        "bandwidth": f"{bandwidth.item():.2f}GB/s",
    }


l_dtype = ["bf16", "fp16"]
l_num_groups = [
    16,
]
l_expect_m = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
]
l_dim = [(7168, 4096)]
l_quant = [
    (aiter.QuantType.No, None, None),  # a16w16
    (aiter.QuantType.per_Token, dtypes.fp8, dtypes.fp8),  # a8w8
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
    "-num_groups",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""num of groups.
    e.g.: -num_groups 128""",
)
parser.add_argument(
    "-expect_m",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""expect m of each groups.
    e.g.: -expect_m 1024""",
)
parser.add_argument(
    "-dim",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""k, n of gemm.
    e.g.: -dim 6144,4096""",
)

parser.add_argument(
    "-q",
    "--quant",
    type=int,
    choices=range(len(l_quant)),
    help="""select quantization type:
    0 : aiter.QuantType.No, None, None),  # a16w16
    1 : aiter.QuantType.per_Token, dtypes.fp8, dtypes.fp8  # a8w8""",
)

args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]

if args.dim is not None:
    l_dim = [args.dim]

if args.num_groups is not None:
    l_num_groups = [args.num_groups]

if args.expect_m is not None:
    l_expect_m = [args.expect_m]

l_quant = [l_quant[args.quant]] if args.quant is not None else l_quant

for (
    dtype,
    num_groups,
    (quant_type, aq_dtype, wq_dtype),
    (k, n),
) in itertools.product(l_dtype, l_num_groups, l_quant, l_dim):
    df = []
    for expect_m in l_expect_m:
        ret = test_deepgemm(
            num_groups,
            expect_m,
            k,
            n,
            aq_dtype,
            wq_dtype,
            quant_type,
            dtype,
        )
        df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("deepgemm summary (markdown):\n%s", df_md)
