# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from aiter.ops.triton.fused_gemm_afp4wfp4_split_cat import fused_gemm_afp4wfp4_split_cat
from op_tests.triton_tests.gemm.batched.test_batched_gemm_afp4wfp4 import (
    mxfp4_to_f32,
    e8m0_to_f32,
)

from aiter.ops.triton.utils.types import str_to_torch_dtype, get_fp8_dtypes
import torch.nn.functional as F

import aiter.ops.triton.utils._triton.arch_info as arch_info


DEVICE_ARCH = arch_info.get_arch()

SCALE_GROUP_SIZE = 32


def run_torch(x, w, y, x_scale, w_scale, S1, S2, D, dtype=torch.bfloat16):
    # First convert the x and w inputs to f32.
    x_f32 = mxfp4_to_f32(x)
    w_f32 = mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    x_scale = x_scale.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    x_scale_f32 = e8m0_to_f32(x_scale)
    x_f32 = x_f32 * x_scale_f32
    w_scale = w_scale.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    w_scale_f32 = e8m0_to_f32(w_scale)
    w_f32 = w_f32 * w_scale_f32
    c = torch.mm(x_f32, w_f32.T).to(dtype)

    c = c.view(-1, D, S1 + S2)
    c1, c2 = c.split([S1, S2], dim=-1)
    c1 = torch.cat([c1, y.expand((*c1.shape[:-1], -1))], dim=-1)

    return c1.to(dtype), c2.to(dtype)


def run_triton(x, w, y, x_scale, w_scale, S1, S2, D, dtype=torch.bfloat16):
    m = x.shape[0]

    return fused_gemm_afp4wfp4_split_cat(
        x, w, y.expand(m, D, -1), x_scale, w_scale, S1, S2, dtype
    )


e5m2_type, e4m3_type = get_fp8_dtypes()


def get_shapes():
    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    x_vals += [(4864, 4096, 8192), (9728, 8192, 65536)]
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
        (2048, 2048, 2049),
        (159, 17389, 597),
        (16, 576, 7168),
    ]
    x_vals += [
        (256, 8192, 1024),
        (256, 1024, 8192),
        (256, 32768, 8192),
        (256, 8192, 32768),
    ]
    x_vals += [
        (16, 2112, 7168),
        (32, 2112, 7168),
        (64, 2112, 7168),
        (128, 2112, 7168),
        (16, 3072, 1536),
        (32, 3072, 1536),
        (64, 3072, 1536),
        (128, 3072, 1536),
        (16, 7168, 2048),
        (32, 7168, 2048),
        (64, 7168, 2048),
        (128, 7168, 2048),
        (16, 4096, 7168),
        (32, 4096, 7168),
        (64, 4096, 7168),
        (128, 4096, 7168),
        (16, 7168, 256),
        (32, 7168, 256),
        (64, 7168, 256),
        (128, 7168, 256),
    ]
    x_vals = [
        (4096, 256, 512),
    ]
    return x_vals


def generate_fused_gemm_afp4wfp4_split_cat_inputs(
    M: int,
    N: int,
    K: int,
    S3: int,
    dtype=torch.bfloat16,
    layout: str = "TN",
):
    """
    The GEMM kernel expects:
    - x: (M, K) -> row-major format
    - w: (N, K) -> column-major format
    - y: (M, D, S3)
    """

    torch.manual_seed(5)
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]

    if layout[0] == "T":
        # 34 is two packed e2m1 values 0010 which is 1.0.
        x_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8)
        x_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8)
    else:
        x_low = torch.randint(0, 16, (K // 2, M), dtype=torch.uint8).T
        x_high = torch.randint(0, 16, (K // 2, M), dtype=torch.uint8).T

    if layout[1] == "N":
        w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
        w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
    else:
        w_low = torch.randint(0, 16, (K // 2, N), dtype=torch.uint8, device="cuda").T
        w_high = torch.randint(0, 16, (K // 2, N), dtype=torch.uint8, device="cuda").T

    x = (
        x_high << 4 | x_low
    )  # Doing this computation on GPU tensors results in NaNs, so move it to GPU afterwards
    x = x.to(device="cuda")

    w = w_low | w_high << 4
    # Scale of 1.0 in e8m0, bias 127.
    M_pad = (M + 255) // 256 * 256
    x_scale = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, M_pad), dtype=torch.uint8, device="cuda"
    )
    w_scale = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device="cuda"
    )
    x_scale = x_scale.T
    w_scale = w_scale.T

    y = torch.rand((M, S3), dtype=torch.bfloat16, device="cuda").unsqueeze(1)

    return x, w, y, x_scale[:M], w_scale


@pytest.mark.parametrize(
    "dtype, M, N, K, D, S3, layout",
    [
        (dtype, *shape, d, s3, layout)
        for dtype in ["bf16"]
        for shape in get_shapes()
        for d in [16]
        for s3 in [64]
        for layout in ["TN", "TT", "NN", "NT"]
    ],
)
def test_fused_gemm_afp4wfp4_split_cat(dtype, M, N, K, D, S3, layout):

    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests
    torch.cuda.synchronize()
    # skip tests
    if N % D != 0:
        pytest.skip("N must be divisible by D as N = D * (S1 + S2)")

    # deconstruct N
    S = N // D
    S1 = S // 2
    S2 = S - S1

    dtype = str_to_torch_dtype[dtype]
    x, w, y, x_scale, w_scale = generate_fused_gemm_afp4wfp4_split_cat_inputs(
        M,
        N,
        K,
        S3,
        dtype=dtype,
        layout=layout,
    )

    c1_torch, c2_torch = run_torch(x, w, y, x_scale, w_scale, S1, S2, D, dtype)
    c1_triton, c2_triton = run_triton(x, w, y, x_scale, w_scale, S1, S2, D, dtype)

    torch.testing.assert_close(c1_torch, c1_triton, atol=0.01, rtol=1e-2)
    torch.testing.assert_close(c2_torch, c2_triton, atol=0.01, rtol=1e-2)
