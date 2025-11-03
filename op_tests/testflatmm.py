# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import random
import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose, perftest, benchmark
import pandas as pd
import argparse

TEST_NUM_ITERS = 100


@perftest(num_iters=TEST_NUM_ITERS)
def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    x = x.to(dtypes.fp32) * x_scale
    weight = weight.to(dtypes.fp32) * w_scale
    out = F.linear(x, weight)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


@perftest(num_iters=TEST_NUM_ITERS)
def run_gemm_ck(x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_CK(x, weight, x_scale, w_scale, bias, dtype)


@perftest()
def run_gemm_ck_bpreshuffle(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_bpreshuffle(x, weight, x_scale, w_scale, None, dtype)

@perftest()
def run_gemm_cktile_bpreshuffle(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_bpreshuffle_CKTILE(x, weight, x_scale, w_scale, None, dtype)

@perftest()
def run_gemm_asm(x, weightshuffle, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_ASM(x, weightshuffle, x_scale, w_scale, bias)

@benchmark()
def test_gemm(dtype, m, n, k, quantDtype=dtypes.i8):
    dim = (m, n, k)
    x = torch.randn((m, k), dtype=dtype, device="cuda")
    weight = torch.randn((n, k), dtype=dtype, device="cuda")
    x, x_scale = aiter.pertoken_quant(x, quant_dtype=quantDtype)
    weight, w_scale = aiter.pertoken_quant(weight, quant_dtype=quantDtype)
    weightshuffle = shuffle_weight(weight, layout=(16, 16))
    bias = torch.rand([1, n], dtype=dtype, device="cuda") * 10

    # x_pad, _ = F.pad(x,(0,128), "constant", 0).split([x.shape[1], 128],dim=1)
    # print(f"{x_pad.shape=}{x_pad.stride()}")

    a, avg_a = run_torch(x, weight, x_scale, w_scale, bias, dtype)
    # b, avg_b = run_gemm_ck(x, weight, x_scale, w_scale, bias, dtype)
    # err_b = checkAllclose(a, b, msg="ck: ", rtol=1e-2, atol=1e-2)
    if quantDtype != dtypes.i8:
        c, avg_c = run_gemm_ck_bpreshuffle(x, weightshuffle, x_scale, w_scale, dtype)
        c = c + bias
        err_c = checkAllclose(a, c, msg="ck bpreshuffle: ", rtol=1e-2, atol=1e-2)
        f, avg_f = run_gemm_cktile_bpreshuffle(x, weightshuffle, x_scale, w_scale, dtype)
        f = f + bias
        err_f = checkAllclose(a, f, msg="cktile bpreshuffle: ", rtol=1e-2, atol=1e-2)
    else:
        avg_c = None
        err_c = None
        avg_f = None
        err_f = None
    gpu = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(gpu)
    cu_num = device_properties.multi_processor_count
    cu_num = 80

    return {
        # "ck us": avg_b,
        # "ck err": err_b,
        "ck bpreshuffle us": avg_c,
        "ck bpreshuffle err": err_c,
        # "asm us": avg_d,
        # "asm err": err_d,
        "cktile bpreshuffle us": avg_f,
        "cktile bpreshuffle err": err_f,
    }


def test_normal_gemm_a8w8_pertoken_quant(l_dtype, l_quantDtype, l_mnk):
    df = []
    for dtype in l_dtype:
        for quantDtype in l_quantDtype:
            for m, n, k in l_mnk:
                ret = test_gemm(dtype, m, n, k, quantDtype)
                df.append(ret)
    df = pd.DataFrame(df)
    aiter.logger.info(f"summary:\n{df}")


l_dtype = ["fp16"]
l_quantDtype = ["fp8"]
# l_mnk_nm = [
#     # (2048, 4096, 5120),
#     (1024, 1024, 1024),
# ]
l_m = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
l_nk = [
    (3072, 1536),
    (7168, 256),
    (7168, 2048),
    (4608, 7168),
    (7168, 2304),
    (512, 7168),
]
l_mnk_nm = [(m, n, k) for m in l_m for n, k in l_nk]

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
    "-q",
    "--quantDtype",
    type=str,
    choices=l_quantDtype,
    nargs="?",
    const=None,
    default=None,
    help="""Date type of quantization.
    e.g.: -q fp8""",
)
parser.add_argument(
    "-mnk",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""shape of mnk.
    e.g. -mnk 1280,8192,1024""",
)

args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]
if args.quantDtype is None:
    l_quantDtype = [dtypes.d_dtypes[key] for key in l_quantDtype]
else:
    l_quantDtype = [dtypes.d_dtypes[args.quantDtype]]
if args.mnk is not None:
    l_mnk_nm = [args.mnk]

test_normal_gemm_a8w8_pertoken_quant(l_dtype, l_quantDtype, l_mnk_nm)
