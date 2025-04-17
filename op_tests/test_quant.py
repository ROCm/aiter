# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from aiter.test_common import checkAllclose, perftest, tensor_dump, benchmark
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import aiter
from aiter.ops.shuffle import shuffle_weight
from aiter import get_hip_quant, get_torch_quant, get_triton_quant
from aiter import QuantType


@perftest()
def test_aiter_perTensorQuantFp8(input, scale=None):
    q_func = get_hip_quant(QuantType.per_Tensor)
    out, scale = q_func(input, scale=scale)
    return out, scale


@perftest()
def test_torch_perTensorQuantFp8(input, scale=None):
    q_func = get_torch_quant(QuantType.per_Tensor)
    out, scale = q_func(input, scale=scale, quant_dtype=torch.float8_e4m3fnuz)
    return out, scale.view(1)


@perftest()
def test_aiter_perTokenQuantFp8(input):
    q_func = get_hip_quant(QuantType.per_Token)
    out, scale = q_func(input, quant_dtype=torch.float8_e4m3fnuz)
    return out, scale


@perftest()
def test_torch_perTokenQuantFp8(input):
    q_func = get_torch_quant(QuantType.per_Token)
    out, scale = q_func(input, quant_dtype=torch.float8_e4m3fnuz)
    return out, scale


@perftest()
def test_triton_perTokenQuantFp8(input):
    q_func = get_triton_quant(QuantType.per_Token)
    out, scale = q_func(input, quant_dtype=torch.float8_e4m3fnuz)
    return out, scale


# @perftest()
# def test_ck_perTokenQuanti8(input):
#     M, N = input.shape
#     device = input.device
#     out = torch.empty((M, N), dtype=torch.int8, device=device)
#     scale = torch.empty(M, dtype=torch.float, device=device)
#     smooth_scale = torch.ones(N, dtype=torch.float, device=device)
#     aiter.smoothquant_fwd(out, input, smooth_scale, scale)
#     return out, scale


@benchmark()
def test_quant(m, n, dtype=torch.bfloat16):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    max_val = torch.max(torch.abs(input.view(-1)))
    print(f"{max_val=}")

    (a, a_scale), avg_a = test_torch_perTensorQuantFp8(input)
    (b, b_scale), avg_b = test_aiter_perTensorQuantFp8(input)
    msg = f"[perf] dynamic_perTensorQuantFp8 dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, aiter avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a.to(torch.float), b.to(torch.float), rtol=0.125, atol=1e-3, msg=msg)
    checkAllclose(a_scale, b_scale, msg="scale")
    # print(f'{a_scale=}, {b_scale=}')

    scale = a_scale
    (a, a_scale), avg_a = test_torch_perTensorQuantFp8(input, scale)
    (b, b_scale), avg_b = test_aiter_perTensorQuantFp8(input, scale)
    msg = f"[perf] static_perTensorQuantFp8 dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, aiter  avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a.to(torch.float), b.to(torch.float), rtol=0.125, atol=1e-3, msg=msg)

    (a, a_scale), avg_a = test_torch_perTokenQuantFp8(input)
    (b, b_scale), avg_b = test_aiter_perTokenQuantFp8(input)
    (c, c_scale), avg_c = test_triton_perTokenQuantFp8(input)
    msg = f"[perf] dynamic_perTokenQuantFp8 dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, aiter  avg: {avg_b:<8.2f} us, triton avg: {avg_c:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a.to(torch.float), b.to(torch.float), rtol=0.125, atol=1e-3, msg=msg)
    checkAllclose(a_scale.view(-1), b_scale, msg="scale")
    msg = f"[perf] dynamic_perTokenQuantFp8 dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us,                               triton avg: {avg_c:<8.2f} us, uplift: {avg_a/avg_c-1:<5.1%}"
    checkAllclose(a.to(torch.float), c.to(torch.float), rtol=0.125, atol=1e-3, msg=msg)
    checkAllclose(a_scale.view(-1), c_scale, msg="scale")

    # (d, d_scale), avg_d = test_ck_perTokenQuanti8(input)
    # msg = f"[perf] dynamic_perTokenQuantFp8 dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck     avg: {avg_d:<8.2f} us, uplift: {avg_a/avg_d-1:<5.1%}"
    # checkAllclose(a.to(torch.float), d.to(torch.float), rtol=0.125, atol=1e-3, msg=msg)
    # checkAllclose(a_scale.view(-1), d_scale, msg="scale")


for dtype in [torch.float16, torch.bfloat16][1:]:
    for m in [1, 16, 32, 64, 128, 192, 256]:
        for n in [4096, 8192]:
            test_quant(m, n, dtype=dtype)
