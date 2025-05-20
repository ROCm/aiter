# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import pytest
import torch.nn.functional as F
from aiter.ops.triton.gemm_a8w8 import gemm_a8w8


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    x = F.linear(x.to(torch.float32), weight.to(torch.float32))
    scale = torch.matmul(x_scale, w_scale)
    out = torch.mul(x, scale)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


def run_triton(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16, y=None):
    return gemm_a8w8(x, weight, x_scale, w_scale, bias, dtype, y)


def is_cdna4():
    return triton.runtime.driver.active.get_current_target().arch == "gfx950"


e5m2_type = torch.float8_e5m2 if is_cdna4() else torch.float8_e5m2fnuz
e4m3_type = torch.float8_e4m3fn if is_cdna4() else torch.float8_e4m3fnuz

name_to_torch_types = {
    "int8": torch.int8,
    "int32": torch.int32,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp8e5": e5m2_type,
    "fp8e4": e4m3_type,
}


dtype_max = {
    dtype: (torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)).max
    for dtype in [
        e5m2_type,
        e4m3_type,
        torch.int8,
    ]
}


def get_x_vals():

    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    x_vals += [(4864, 4096, 8192), (9728, 8192, 65536), (4864, 8192, 4160)]
    x_vals += [
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
    return x_vals

def generate_gemm_a8w8_inputs(M, N, K, in_dtype, out_dtype, output=False):

    x = torch.randn((M, K), dtype=torch.float32, device="cuda")
    max_x = x.abs().float().amax(dim=1, keepdim=True)
    x_scale = max_x / dtype_max[in_dtype]
    x = x / x_scale
    x = x.to(in_dtype)

    weight = torch.randn((N, K), dtype=torch.float32, device="cuda")
    max_weight = weight.abs().float().amax(dim=1, keepdim=True).T.contiguous()
    w_scale = max_weight / dtype_max[in_dtype]
    weight = weight / w_scale.T
    weight = weight.to(in_dtype)

    bias = torch.rand([1, N], dtype=torch.float32).cuda() * 10

    y = None
    if output:
        y = torch.empty((M, N), dtype=out_dtype).cuda()

    return x, weight, x_scale, w_scale, bias, y


@pytest.mark.parametrize(
    "in_dtype, out_dtype, m, n, k, output",
    [
        (in_dtype, out_dtype, *shape, output)
        for in_dtype in ["fp8e4", "fp8e5", "int8"]
        for out_dtype in ["bf16"]
        for shape in get_x_vals()
        for output in [True, False]
    ],
)
def test_gemm(in_dtype, out_dtype, m, n, k, output):
    in_dtype = name_to_torch_types[in_dtype]
    out_dtype = name_to_torch_types[out_dtype]
    x, weight, x_scale, w_scale, bias, y = generate_gemm_a8w8_inputs(
        m, n, k, in_dtype, out_dtype, output
    )

    a = run_torch(x, weight, x_scale, w_scale, bias, out_dtype)
    b = run_triton(x, weight, x_scale, w_scale, bias, out_dtype, y)

    triton.testing.assert_close(a, b, atol=0.01, rtol=1e-2)
