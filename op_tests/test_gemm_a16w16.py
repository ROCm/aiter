# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import os
import random
import sys
from functools import lru_cache

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes, hipb_create_extension, hipb_mm
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, perftest
from aiter.tuned_gemm import tgemm, triton_gemm

# TEST_NUM_ITERS = 10
TEST_NUM_ITERS = 100


@perftest(num_iters=TEST_NUM_ITERS)
def run_torch(x, weight, bias=None, otype=None, scaleA=None, scaleB=None):
    if x.dtype == dtypes.fp8:
        if scaleA is None:
            scaleA = torch.ones(1, dtype=dtypes.fp32, device=x.device)
        if scaleB is None:
            scaleB = torch.ones(1, dtype=dtypes.fp32, device=x.device)

        try:
            out = torch._scaled_mm(
                x,
                weight.t(),
                out_dtype=otype,
                scale_a=scaleA,
                scale_b=scaleB,
                bias=bias,
            )
        except RuntimeError:
            out = F.linear(x.to(dtypes.fp32), weight.to(dtypes.fp32)) * scaleA * scaleB
            out = (out.to(otype) + bias) if bias is not None else out.to(otype)
        return out
    if scaleA is not None:
        x = x * scaleA
    if scaleB is not None:
        weight = weight * scaleB
    return F.linear(x, weight, bias).to(otype)


@perftest(num_iters=TEST_NUM_ITERS)
def run_gemm_b(x, weight, bias=None, otype=None, scaleA=None, scaleB=None):
    return tgemm.mm(x, weight, bias, otype, scaleA, scaleB)


@perftest(num_iters=TEST_NUM_ITERS)
def run_bf16gemm_asm(
    x, weight, out_asm, bias=None, splitK=None, kernelName=None, bpreshuffle=False
):
    return aiter.gemm_a16w16_asm(
        x, weight, out_asm, bias, splitK, kernelName, bpreshuffle
    )


@perftest(num_iters=TEST_NUM_ITERS)
def aiter_hip_bpreshuffle(inp, weights, scaleA, scaleB, dtype):
    if scaleB is not None:
        scaleB = scaleB.t()
    return hipb_mm(
        inp,
        weights.t(),
        solution_index=-1,
        bias=None,
        out_dtype=dtype,
        scaleA=scaleA,
        scaleB=scaleB,
        scaleOut=None,
        bpreshuffle=True,
    )


@perftest(num_iters=TEST_NUM_ITERS)
def run_gemm_triton(x, weight, bias=None, otype=None, scaleA=None, scaleB=None):
    return triton_gemm(x, weight, 0, bias=bias, otype=otype)


@lru_cache(maxsize=1)
def init_hipblas():
    hipb_create_extension()


@benchmark()
def test_gemm(dtype, m, n, k, bias=False, otype=None, scaleA=None, scaleB=None):
    ret = {}
    dim = (m, n, k)
    x = torch.randn(m, k, dtype=otype, device="cuda").to(dtype)
    weight = torch.randn(n, k, dtype=otype, device="cuda").to(dtype)
    if otype is None:
        otype = dtype
    if bias:
        bias = torch.rand(n, dtype=dtype, device="cuda")
    else:
        bias = None
    if scaleA is not None:
        scaleA = torch.tensor(scaleA, dtype=dtypes.fp32, device="cuda")
    if scaleB is not None:
        scaleB = torch.tensor(scaleB, dtype=dtypes.fp32, device="cuda")
    a, avg_a = run_torch(x, weight, bias, otype, scaleA, scaleB)
    b, avg_b = run_gemm_b(x, weight, bias, otype, scaleA, scaleB)
    assert (
        a.dtype == b.dtype
    ), f"Expected a.dtype == b.dtype, but a={a.dtype}, b={b.dtype}, input dtype={dtype}"
    if otype is not None:
        assert (
            a.dtype == otype
        ), f"a={a.dtype}, expected output dtype={otype}, input dtype={dtype}"
        assert (
            b.dtype == otype
        ), f"b={b.dtype}, expected output dtype={otype}, input dtype={dtype}"

    msg_b = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, B avg: {avg_b:<8.2f} us,B uplift: {avg_a/avg_b-1:<5.1%}, "
    err_tgemm = checkAllclose(a, b, msg=msg_b)
    ret["torch us"] = avg_a
    ret["tgemm us"] = avg_b
    ret["tgemm err"] = err_tgemm

    if (
        n % 16 == 0
        and k % 32 == 0
        and dtype == otype
        and otype == dtypes.bf16
        and get_gfx() == "gfx942"
    ):
        init_hipblas()
        weight_bpreshuffle = shuffle_weight(weight, layout=(16, 16), use_int4=False)
        c, avg_c = aiter_hip_bpreshuffle(x, weight_bpreshuffle, None, None, otype)
        if bias is not None:
            c = c + bias
    else:
        c = None
        avg_c = None
    if c is not None and avg_c is not None:
        assert (
            c.dtype == otype
        ), f"c={c.dtype}, expected output dtype={otype}, input dtype={dtype}"
        msg_c = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, C avg: {avg_c:<8.2f} us, C uplift: {avg_a/avg_c-1:<5.1%}, "
        err_hipb = checkAllclose(a, c, msg=msg_c) if c is not None else None
        ret["hipb us"] = avg_c
        ret["hipb err"] = err_hipb

    #### asm a16w16 gemm -- huan
    ### run bf16gemm_f32 asm
    if (
        dtype == dtypes.bf16
        and (otype == dtypes.fp32 or otype == dtypes.bf16)
        and k % 64 == 0
        and n % 64 == 0
    ):
        out_asm = torch.empty(m, n, dtype=otype, device=x.device)
        ### b preshuffle
        wshuffle = shuffle_weight(weight, layout=(16, 16))
        d, avg_d = run_bf16gemm_asm(
            x, wshuffle, out_asm, bias, bpreshuffle=wshuffle.is_shuffled
        )
        msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, B avg: {avg_b:<8.2f} us, asm-bpreshuffle avg: {avg_d:<8.2f} us, uplift: {avg_b/avg_d-1:<5.1%}"
        err_asm = checkAllclose(a, d, msg=msg)
        ### no shuffle
        e, avg_e = run_bf16gemm_asm(x, weight, out_asm, bias)
        msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, B avg: {avg_b:<8.2f} us, asm-noshuffle avg: {avg_e:<8.2f} us, uplift: {avg_b/avg_e-1:<5.1%}"
        err_asm_noshuffle = checkAllclose(a, e, msg=msg)
        ret["asm-bpshuff us"] = avg_d
        ret["asm-bpshuff err"] = err_asm
        ret["asm-nshuff us"] = avg_e
        ret["asm-nshuff err"] = err_asm_noshuffle

    a, us = run_gemm_triton(x, weight, bias, otype, scaleA, scaleB)
    err = checkAllclose(b, a)
    ret["triton us"] = us

    return ret


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

    # m = 4 boundaries (max in [2,4])
    boundary_cases.extend(
        [
            (4, 1, aligned_k),  # max m in range, min n, min k
            (4, 1, 9216),  # max m in range, min n, max k
            (4, cu_count, aligned_k),  # max m in range, max n, min k
            (4, cu_count, 9216),  # max m in range, max n, max k
            (4, cu_count - 1, 9216),  # max m in range, max n-1, max k
        ]
    )

    # Region 2: m in [5,8]
    # m = 5 boundaries (min in [5,8])
    boundary_cases.extend(
        [
            (5, 1, aligned_k),  # min m in range, min n, min k
            (5, 1, 5120),  # min m in range, min n, max k
            (5, cu_count, aligned_k),  # min m in range, max n, min k
            (5, cu_count, 5120),  # min m in range, max n, max k
        ]
    )

    # m = 8 boundaries (max in [5,8])
    boundary_cases.extend(
        [
            (8, 1, aligned_k),  # max m in range, min n, min k
            (8, 1, 5120),  # max m in range, min n, max k
            (8, cu_count, aligned_k),  # max m in range, max n, min k
            (8, cu_count, 5120),  # max m in range, max n, max k
            (8, cu_count - 1, 5120),  # max m in range, max n-1, max k
        ]
    )

    # Region 3: m in [9,16]
    # m = 9 boundaries (min in [9,16])
    boundary_cases.extend(
        [
            (9, 1, aligned_k),  # min m in range, min n, min k
            (9, 1, 256),  # min m in range, min n, max k
            (9, cu_count, aligned_k),  # min m in range, max n, min k
            (9, cu_count, 256),  # min m in range, max n, max k
        ]
    )

    # m = 16 boundaries (max in [9,16])
    boundary_cases.extend(
        [
            (16, 1, aligned_k),  # max m in range, min n, min k
            (16, 1, 256),  # max m in range, min n, max k
            (16, cu_count, aligned_k),  # max m in range, max n, min k
            (16, cu_count, 256),  # max m in range, max n, max k
            (15, cu_count, 256),  # max m-1 in range, max n, max k
            (16, cu_count - 1, 256),  # max m in range, max n-1, max k
            (15, cu_count - 1, 256),  # max m-1 in range, max n-1, max k
        ]
    )

    # Region transition boundaries
    boundary_cases.extend(
        [
            (4, cu_count, 9216),  # Region1 max (m=4)
            (5, cu_count, 5120),  # Region2 min (m=5)
            (8, cu_count, 5120),  # Region2 max (m=aligned_k)
            (9, cu_count, 256),  # Region3 min (m=9)
        ]
    )

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
            8, 9217, aligned_k
        ):  # k: multiples of aligned_k from aligned_k to 9216
            if random.random() <= ratio:
                test_cases.append((m, n, k))

    # m in [2, 4]
    for m in range(2, 5):  # m: 2, 3, 4
        for n in range(1, cu_count + 1):  # n: 1 to cu_count
            for k in range(
                8, 9217, aligned_k
            ):  # k: multiples of aligned_k from aligned_k to 9216
                if random.random() <= ratio:
                    test_cases.append((m, n, k))

    # Region 2: m in [5, 8]
    for m in range(5, 9):  # m: 5, 6, 7, 8
        for n in range(1, cu_count + 1):  # n: 1 to cu_count
            for k in range(
                8, 5121, aligned_k
            ):  # k: multiples of aligned_k from aligned_k to 5120
                if random.random() <= ratio:
                    test_cases.append((m, n, k))

    # Region 3: m in [9, 16]
    for m in range(9, 17):  # m: 9 to 16
        for n in range(1, cu_count + 1):  # n: 1 to cu_count
            for k in range(
                8, 257, aligned_k
            ):  # k: multiples of aligned_k from aligned_k to 256
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

    # Region 2: m in [5,8]
    total += (
        4 * cu_count * (5120 // aligned_k)
    )  # m=5..8; n=1..cu_count; k=aligned_k,16,...,5120

    # Region 3: m in [9,16]
    total += (
        8 * cu_count * (256 // aligned_k)
    )  # m=9..16; n=1..cu_count; k=aligned_k,16,...,256

    return total


def test_skinny_gemm():
    df = []
    # seed = 8779
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    random.seed(137)

    aligned_k = 8
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
            [3, 1, 8192],
            [4, 1, 8192],
            [4, 32, 8192],
            [4, 32, 9216],
            [16, 7424, 8192],
            [32, 7424, 8192],
            [48, 7424, 8192],
            [64, 7424, 8192],
            [4096, 7424, 8192],
            [5120, 7424, 8192],
            [8192, 7424, 8192],
        ]
    )
    test_mnk_list.extend(boundary_mnk_list)
    test_mnk_list.extend(mnk_list)
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
            for dtype in [dtypes.fp16, dtypes.bf16]:
                for otype in [None, dtypes.fp16, dtypes.bf16, dtypes.fp32]:
                    ret = test_gemm(dtype, m, n, k, otype=otype)
                    df.append(ret)
    return df


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of a16w16_gemm_test",
)
parser.add_argument(
    "-t",
    "--test",
    type=str,
    nargs="*",
    choices=["normal", "skinny"],
    default=["normal"],
    help="""Select test to run.
    e.g.: -t normal    # default
          or -t skinny""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    # choices=["bf16", "fp16", "fp8"],
    default=[torch.bfloat16, torch.float16],
    help="""Data type. Support "bf16", "fp16", "fp8".
    e.g.: -d bf16
          or -d bf16,fp16    # Multiple comma-separated argus supported.""",
)
parser.add_argument(
    "-mnk",
    type=dtypes.str2tuple,
    nargs="+",
    const=None,
    default=[(128, 32, 8192), (64, 256, 5120)],  # (64, 256, 5120) in tuned_gemm.csv
    help="""Shape of mnk.
    e.g. -mnk 128,32,8192""",
)
parser.add_argument(
    "-b",
    "--bias",
    action="store_true",
    help="""bias of GEMM. Default is False.
    -b or --bias    # enable Bias""",
)
parser.add_argument(
    "-o",
    "--otype",
    type=dtypes.str2Dtype,
    default=[torch.float16, torch.bfloat16, torch.float32],
    help="""Data type of output.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-sa",
    "--scale_a",
    type=float,
    default=None,
    help="""Scale A.
    e.g.: -sa 0.5""",
)
parser.add_argument(
    "-sb",
    "--scale_b",
    type=float,
    default=None,
    help="""Scale B.
    e.g.: -sb 0.5""",
)
args = parser.parse_args()

df = []
for test in args.test:
    if test == "normal":
        for dtype in args.dtype:
            for otype in args.otype:
                for m, n, k in args.mnk:
                    ret = test_gemm(
                        dtype,
                        m,
                        n,
                        k,
                        bias=args.bias,
                        otype=otype,
                        scaleA=args.scale_a,
                        scaleB=args.scale_b,
                    )
                    df.append(ret)

    elif test == "skinny":
        ret = test_skinny_gemm()
        df += ret
df = pd.DataFrame(df)
aiter.logger.info(f"summary:\n{df}")
