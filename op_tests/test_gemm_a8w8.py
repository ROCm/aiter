# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

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
    return aiter.gemm_a8w8_bpreshuffle_CK(x, weight, x_scale, w_scale, dtype)


@perftest()
def run_gemm_asm(x, weightshuffle, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_ASM(x, weightshuffle, x_scale, w_scale, bias)


@perftest(num_iters=TEST_NUM_ITERS)
def run_gemm_skinny(
    x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16, cu_count=1
):
    out = torch.empty(x.shape[0], weight.shape[0], dtype=dtype, device=x.device)
    aiter.wvSplitKQ(weight, x, out, w_scale, x_scale, cu_count)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


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
    b, avg_b = run_gemm_ck(x, weight, x_scale, w_scale, bias, dtype)
    err_b = checkAllclose(a, b, msg="ck: ", rtol=1e-2, atol=1e-2)
    if quantDtype != dtypes.i8:
        c, avg_c = run_gemm_ck_bpreshuffle(x, weightshuffle, x_scale, w_scale, dtype)
        c = c + bias
        err_c = checkAllclose(a, c, msg="ck bpreshuffle: ", rtol=1e-2, atol=1e-2)
    else:
        avg_c = None
        err_c = None

    avg_d = None
    err_d = None
    if dtype == dtypes.bf16 and quantDtype == dtypes.i8 and bias is not None:
        weightshuffle_asm = shuffle_weight(weight, layout=(32, 16))
        bias_f32 = bias.to(dtypes.fp32)
        d, avg_d = run_gemm_asm(x, weightshuffle_asm, x_scale, w_scale, bias_f32, dtype)
        if d is not None:
            err_d = checkAllclose(a, d, msg="asm: ", rtol=1e-2, atol=1e-2)
        else:
            avg_d = None

    return {
        "ck us": avg_b,
        "ck err": err_b,
        "ck bpreshuffle us": avg_c,
        "ck bpreshuffle err": err_c,
        "asm us": avg_d,
        "asm err": err_d,
    }


def test_skinny_gemm(dtype, m, n, k, quantDtype=dtypes.fp8, cu_count=80):
    dim = (m, n, k)
    x = torch.randn((m, k), dtype=dtype, device="cuda")
    weight = torch.randn((n, k), dtype=dtype, device="cuda")
    x, x_scale = aiter.per_tensor_quant(x, quant_dtype=quantDtype)
    weight, w_scale = aiter.per_tensor_quant(weight, quant_dtype=quantDtype)
    bias = None

    if m <= 2:
        a, avg_a = run_torch(x, weight, x_scale, w_scale, bias, dtype)
    else:
        a, avg_a = run_gemm_ck(x, weight, x_scale, w_scale, bias, dtype)

    b, avg_b = run_gemm_skinny(x, weight, x_scale, w_scale, None, dtype, cu_count)

    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, quantDtype: {quantDtype}, torch avg: {avg_a:<8.2f} us, skinny_gemm avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, msg="a,b: " + msg, rtol=1e-2, atol=0.01)


def get_boundary_test_cases(cu_count, aligned_k):
    """
    Generate a list of boundary test cases (m, n, k) for the GEMM kernel.
    These test cases cover the edges of each valid region and transition points between regions.
    All k values are divisible by 8.

    Returns:
        list: A list of tuples (m, n, k) representing boundary conditions.
    """
    boundary_cases = []

    # Region 1: m=1 and m in [2,4]
    # m = 1 boundaries
    boundary_cases.extend(
        [
            (1, 1, aligned_k),  # min m, min n, min k
            (1, 1, 9216),  # min m, min n, max k
            (1, 2 * cu_count, aligned_k),  # min m, max n, min k
            (1, 2 * cu_count, 9216),  # min m, max n, max k
        ]
    )

    # m = 2 boundaries (min in [2,4])
    boundary_cases.extend(
        [
            (2, 1, aligned_k),  # min m in range, min n, min k
            (2, 1, 9216),  # min m in range, min n, max k
            (2, cu_count, aligned_k),  # min m in range, max n, min k
            (2, cu_count, 9216),  # min m in range, max n, max k
        ]
    )

    # # m = 4 boundaries (max in [2,4])
    # boundary_cases.extend(
    #     [
    #         (4, 1, aligned_k),  # max m in range, min n, min k
    #         (4, 1, 9216),  # max m in range, min n, max k
    #         (4, cu_count, aligned_k),  # max m in range, max n, min k
    #         (4, cu_count, 9216),  # max m in range, max n, max k
    #         (4, cu_count - 1, 9216),  # max m in range, max n-1, max k
    #     ]
    # )

    return boundary_cases


def generate_test_cases(cu_count, ratio, aligned_k):
    """
    Generate a list of (m, n, k) tuples that satisfy the kernel's constraints,
    sampling the valid parameter space at a given ratio. All generated k values
    will be divisible by 8.

    Args:
        ratio (float): Sampling ratio (0.0 to 1.0). Determines the proportion of valid
                       (m, n, k) tuples to include in the output.

    Returns:
        list: A list of tuples (m, n, k) that meet the kernel constraints,
              sampled according to the ratio.

    Raises:
        ValueError: If ratio is not in [0.0, 1.0].
    """
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError("ratio must be a float between 0.0 and 1.0")

    test_cases = []

    # Region 1: m=1 and m in [2,4]
    # m = 1
    m = 1
    for n in range(1, 2 * cu_count + 1):  # n: 1 to 2 * cu_count
        for k in range(
            16, 9217, aligned_k
        ):  # k: multiples of aligned_k from aligned_k to 9216
            if random.random() <= ratio:
                test_cases.append((m, n, k))

    # m in [2, 4]
    for m in range(2, 5):  # m: 2, 3, 4
        for n in range(1, cu_count + 1):  # n: 1 to cu_count
            for k in range(
                16, 9217, aligned_k
            ):  # k: multiples of aligned_k from aligned_k to 9216
                if random.random() <= ratio:
                    test_cases.append((m, n, k))

    return test_cases


def calculate_total_valid_points(cu_count, aligned_k):
    """Calculate the total number of valid (m, n, k) tuples that satisfy the kernel constraints with k divisible by 8."""
    total = 0

    # Region 1: m=1
    total += (
        2 * cu_count * (9216 // aligned_k)
    )  # m=1, n=1..2*cu_count, k=aligned_k,16,...,9216

    # Region 1: m in [2,4]
    total += (
        3 * cu_count * (9216 // aligned_k)
    )  # m=2,3,4; n=1..cu_count; k=aligned_k,16,...,9216

    return total


def test_normal_gemm_a8w8_pertoken_quant(l_dtype, l_quantDtype, l_mnk):
    df = []
    for dtype in l_dtype:
        for quantDtype in l_quantDtype:
            for m, n, k in l_mnk:
                ret = test_gemm(dtype, m, n, k, quantDtype)
                df.append(ret)
    df = pd.DataFrame(df)
    aiter.logger.info(f"summary:\n{df}")


def test_skinny_gemm_a8w8_pertoken_quant():
    # seed = 8779
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    random.seed(137)

    aligned_k = 16
    # cu_count = 80
    cu_count = torch.cuda.get_device_properties(device="cuda").multi_processor_count
    # ratio = 0.002
    ratio = 0.0002

    # Calculate and print total valid points
    total_points = calculate_total_valid_points(cu_count, aligned_k)
    boundary_mnk_list = get_boundary_test_cases(cu_count, aligned_k)
    mnk_list = generate_test_cases(cu_count, ratio, aligned_k)
    test_mnk_list = []
    test_mnk_list.extend(
        [
            # [3, 1, 8192],
            # [4, 1, 8192],
            # [4, 32, 8192],
            # [4, 32, 9216],
            [2, 320, 32768],
            [2, 640, 32768],
            [2, 1280, 32768],
            [2, 320, 32768 + 1024],
            [2, 320, 2 * 32768],
            [1, 4, 1264],
            [1, 12, 8720],
            [1, 17, 6192],
            [1, 40, 9024],
            [2, 27, 4544],
            [2, 28, 6544],
            [2, 57, 1952],
            [2, 60, 96],
        ]
    )
    test_mnk_list.extend(boundary_mnk_list)
    # test_mnk_list.extend(mnk_list)
    print(f"cu_count={cu_count}")
    print(f"len(boundary_mnk_list)={len(boundary_mnk_list)}")
    print(f"len(mnk_list)={len(mnk_list)}")
    print(
        f"total valid (m, n, k) tuples with k divisible by {aligned_k}: {total_points}"
    )
    print(f"total test case count: {2 * len(test_mnk_list)}")

    loop_count = 1
    for i in range(loop_count):
        for mnk in test_mnk_list:
            m, n, k = mnk
            for quant_dtype in [dtypes.fp8]:
                for dtype in [dtypes.fp16, dtypes.bf16]:
                    test_skinny_gemm(dtype, m, n, k, quant_dtype, cu_count)
                    # test_gemm(dtype, m, n, k, quant_dtype)


l_dtype = ["bf16", "fp16"]
l_quantDtype = ["i8", "fp8"]
l_mnk_nm = [
    # qkv_proj
    (1, 1280, 8192),
    (32, 1280, 8192),
    (64, 1280, 8192),
    (128, 1280, 8192),
    (192, 1280, 8192),
    (256, 1280, 8192),
    (320, 1280, 8192),
    (512, 1280, 8192),
    (1024, 1280, 8192),
    (2048, 1280, 8192),
    (4096, 1280, 8192),
    (8192, 1280, 8192),
    (16384, 1280, 8192),
    # attn_out
    (1, 8192, 1024),
    (32, 8192, 1024),
    (64, 8192, 1024),
    (128, 8192, 1024),
    (192, 8192, 1024),
    (256, 8192, 1024),
    (320, 8192, 1024),
    (512, 8192, 1024),
    (1024, 8192, 1024),
    (2048, 8192, 1024),
    (4096, 8192, 1024),
    (8192, 8192, 1024),
    (16384, 8192, 1024),
]

parser = argparse.ArgumentParser(description="config input of test")
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
    "-q",
    "--quantDtype",
    type=str,
    choices=l_quantDtype,
    nargs="?",
    const=None,
    default=None,
    help="shape",
)
parser.add_argument(
    "-mnk",
    type=dtypes.str2tuple,
    choices=l_mnk_nm,
    nargs="?",
    const=None,
    default=None,
    help="shape",
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
test_skinny_gemm_a8w8_pertoken_quant()
