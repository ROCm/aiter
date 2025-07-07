# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from aiter.test_common import (
    checkAllclose,
    benchmark,
    run_perftest,
)
import torch
import aiter
from aiter import dtypes
from aiter import get_hip_quant, get_torch_quant, get_triton_quant
import itertools
import argparse

torch.set_default_device("cuda")


@benchmark()
def test_quant(m, n, q_type, q_dtype, h_dtype):
    dim = (m, n)

    input = torch.randn(dim, dtype=h_dtype)
    ref, ref_scale = get_torch_quant(q_type)(input, quant_dtype=q_dtype)

    q_funcs = {
        "triton": get_triton_quant,
        "hip": get_hip_quant,
    }
    ret = {}
    for name, q_func in q_funcs.items():
        q_func = q_func(q_type)
        (out, scale), us1 = run_perftest(q_func, input, quant_dtype=q_dtype)
        err1 = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-3,
            atol=1e-3,
            msg=f"{name}: dynamic quant",
        )
        checkAllclose(
            ref_scale.to(dtypes.fp32),
            scale.to(dtypes.fp32),
            rtol=1e-3,
            atol=1e-3,
            msg=f"{name}: dynamic quant scale",
        )
        ret[f"{name} dq"] = us1
        ret[f"{name} dq err"] = err1
        if q_type == aiter.QuantType.per_Tensor:
            (out, scale), us2 = run_perftest(
                q_func, input, ref_scale, quant_dtype=q_dtype
            )
            err2 = checkAllclose(
                ref.to(dtypes.fp32),
                out.to(dtypes.fp32),
                rtol=1e-3,
                atol=1e-3,
                msg=f"{name}: static  quant",
            )
            ret[f"{name} sq"] = us2
            ret[f"{name} sq err"] = err2

    return ret


d_quant = {
    "fp8_tensor": (aiter.QuantType.per_Tensor, dtypes.fp8),
    "fp8_token": (aiter.QuantType.per_Token, dtypes.fp8),
    "fp8_1x128": (aiter.QuantType.per_1x128, dtypes.fp8),
    "i8_token": (aiter.QuantType.per_Token, dtypes.i8),
    # 'fp4x2-1x32': (aiter.QuantType.per_1x32, dtypes.fp4x2),  # 注释的选项
}
list_dtype = ["fp16", "bf16"]
l_n = [4096, 8192]
l_m = [1, 2, 16, 32, 64, 128, 192, 256, 512, 1024, 16384, 163840]
import pandas as pd

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=list_dtype,
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-n",
    "--n",
    type=int,
    nargs="*",
    default=None,
    help="""N of mnk.
    e.g.: -n 1024""",
)
parser.add_argument(
    "-m",
    "--m",
    type=int,
    nargs="*",
    default=None,
    help="""M of mnk.
    e.g.: -m 32""",
)
parser.add_argument(
    "-q",
    "--quant",
    type=str,
    choices=list(d_quant.keys()),
    nargs="*",
    default=list(d_quant.keys()),
    help="""Quantization type.
    e.g.: -q fp8_tensor""",
)

args = parser.parse_args()
if args.dtype is None:
    list_dtype = [dtypes.d_dtypes[key] for key in list_dtype]
else:
    list_dtype = [dtypes.d_dtypes[args.dtype]]
list_quant = [d_quant[key] for key in args.quant]
if args.n is not None:
    l_n = args.n
if args.m is not None:
    l_m = args.m

for (
    (q_type, q_dtype),
    h_dtype,
) in itertools.product(list_quant, list_dtype):
    df = []
    for n in l_n:
        for m in l_m:
            ret = test_quant(m, n, q_type, q_dtype, h_dtype)
            df.append(ret)
    df = pd.DataFrame(df)
    aiter.logger.info(f"summary:\n{df}")
