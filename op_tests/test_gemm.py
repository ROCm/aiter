# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import sys
import os
import random
from aiter import dtypes
from aiter.test_common import checkAllclose, perftest

# TEST_NUM_ITERS = 10
TEST_NUM_ITERS = 100

if 1:
    _path = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, f"{_path}/../../")
    from aiter.tuned_gemm import tgemm


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


def test_gemm(dtype, m, n, k, bias=False, otype=None, scaleA=None, scaleB=None):
    dim = (m, n, k)
    x = torch.randn(m, k, dtype=otype, device="cuda").to(dtype)
    weight = torch.rand(n, k, dtype=otype, device="cuda").to(dtype)
    if bias:
        bias = torch.rand(n, dtype=otype, device="cuda")
    else:
        bias = None
    if scaleA is not None:
        scaleA = torch.tensor(scaleA, dtype=dtypes.fp32, device="cuda")
    if scaleB is not None:
        scaleB = torch.tensor(scaleB, dtype=dtypes.fp32, device="cuda")
    (a, *_), avg_a = run_torch(x, weight, bias, otype, scaleA, scaleB)
    (b, *_), avg_b = run_gemm_b(x, weight, bias, otype, scaleA, scaleB)
    
    assert a.dtype == b.dtype, f"Expected a.dtype == b.dtype, but a={a.dtype}, b={b.dtype}, input dtype={dtype}"
    if otype is not None:
        assert a.dtype == otype, f"a={a.dtype}, expected output dtype={otype}, input dtype={dtype}"
        assert b.dtype == otype, f"b={b.dtype}, expected output dtype={otype}, input dtype={dtype}"

    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, B avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, msg=msg)


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


def test_normal_gemm():
    test_gemm(
        dtypes.fp8,
        128,
        768,
        4096,
        bias=False,
        otype=dtypes.bf16,
        scaleA=0.5,
        scaleB=0.5,
    )
    test_gemm(dtypes.bf16, 128, 32, 8192)
    for dtype in [dtypes.fp16, dtypes.bf16]:
        for otype in [None, dtypes.fp16, dtypes.bf16, dtypes.fp32]:
            test_gemm(dtype, 128, 32, 8192, otype=otype)
        # # qkv_proj
        # for (m, n, k) in [(4096, 1280, 8192),
        #                   (128, 1280, 8192),
        #                   (128, 1024, 8192),
        #                   (128, 128, 8192),
        #                   ]:
        #     test_gemm(dtype, m, n, k)
        # # attn_out
        # for (m, n, k) in [(4096, 8192, 1024),
        #                   (128, 8192, 1024)]:
        #     test_gemm(dtype, m, n, k)
        # test_gemm(dtype, 128, 1024, 8192)
        # # gating
        # for (m, n, k) in [(4096, 32, 8192),
        #                   (128, 32, 8192)]:
        #     test_gemm(dtype, m, n, k)
        # # gating
        # for (m, n, k) in [(1, 19392, 8192),
        #                   (128, 19392, 8192)]:
        #     test_gemm(dtype, m, n, k)


def test_skinny_gemm():
    # seed = 8779
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    random.seed(137)

    aligned_k = 8
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
            [3, 1, 8192],
            [4, 1, 8192],
            [4, 32, 8192],
            [4, 32, 9216],
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
                    test_gemm(dtype, m, n, k, otype=otype)


# test_normal_gemm()
test_skinny_gemm()
