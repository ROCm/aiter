# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from aiter.ops.triton.fused_gemm_a8w8_blockscale_split_cat import fused_gemm_a8w8_blockscale_split_cat

from aiter.ops.triton.utils.types import str_to_torch_dtype, get_fp8_dtypes
import torch.nn.functional as F

import aiter.ops.triton.utils._triton.arch_info as arch_info


block_shape = (128, 128)
DEVICE_ARCH = arch_info.get_device()


def run_torch(
    x, w, y,
    x_scale, w_scale,
    S1, S2, D,
    dtype=torch.bfloat16
):
    block_shape_n, block_shape_k = block_shape
    m, c1 = x.shape
    n = w.shape[0]

    x_scale = x_scale.repeat_interleave(block_shape_k, dim=1)
    x = x.to(x_scale.dtype) * x_scale[:m, :c1]
    x = x.view(m, c1)

    w_scale = w_scale.repeat_interleave(block_shape_n, dim=0)
    w_scale = w_scale.repeat_interleave(block_shape_k, dim=1)
    w_scale = w_scale[:n, :c1]
    w = w.to(w_scale.dtype) * w_scale

    c = F.linear(x.to(torch.float32), w.to(torch.float32))
    c = c.view(-1, D, S1 + S2)
    c1, c2 = c.split([S1, S2], dim=-1)
    c1 = torch.cat([c1, y.expand((*c1.shape[:-1], -1))], dim=-1)

    return c1.to(dtype), c2.to(dtype)


def run_triton(
    x, w, y,
    x_scale, w_scale,
    S1, S2, D,
    dtype=torch.bfloat16
):
    m = x.shape[0]

    return fused_gemm_a8w8_blockscale_split_cat(
        x, w, y.expand(m, D, -1),
        x_scale, w_scale,
        S1, S2,
        dtype
    )


e5m2_type, e4m3_type = get_fp8_dtypes()


def get_shapes():
    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    x_vals += [
        (4864, 4096, 8192),
        (9728, 8192, 65536)
    ]
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
    return x_vals


def generate_fused_gemm_a8w8_blockscale_split_cat_inputs(
    M: int,
    N: int,
    K: int,
    S3: int,
    block_shape_n: int,
    block_shape_k: int,
    dtype=torch.bfloat16,
    layout: str = "TN",
):
    """
    The GEMM kernel expects:
    - x: (M, K) -> row-major format
    - w: (N, K) -> column-major format
    - y: (M, D, S3)
    """
    scale_n = (N + block_shape_n - 1) // block_shape_n
    scale_k = (K + block_shape_k - 1) // block_shape_k

    if layout[0] == "T":
        x = (torch.rand((M, K), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)
    else:
        x = (
            (torch.rand((K, M), dtype=torch.float16, device="cuda") / 10)
            .to(e4m3_type)
            .T
        )

    if layout[1] == "N":
        w = (torch.rand((N, K), dtype=torch.float16, device="cuda") / 10).to(
            e4m3_type
        )
    else:
        w = (
            (torch.rand((K, N), dtype=torch.float16, device="cuda") / 10)
            .to(e4m3_type)
            .T
        )

    x_scale = torch.rand([M, scale_k], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device="cuda")

    y = torch.rand((M, S3), dtype=torch.bfloat16, device="cuda").unsqueeze(1)

    return x, w, y, x_scale, w_scale


@pytest.mark.parametrize(
    "dtype, M, N, K, D, S3, layout",
    [
        (dtype, *shape, d, s3, layout)
        for dtype in ["bf16"]
        for shape in get_shapes()
        for d in [16, 32, 64, 128]
        for s3 in [16, 32, 64, 128]
        for layout in ["TN", "TT", "NN", "NT"]
    ],
)
def test_fused_gemm_a8w8_blockscale_split_cat(dtype, M, N, K, D, S3, layout):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests
    torch.cuda.synchronize()

    block_shape_n, block_shape_k = block_shape

    # skip tests
    if K % block_shape_k != 0:
        pytest.skip(
            "Latest upstream compiler as of Aug 22 (necessary for Gluon) causes"
            " infinite hang when EVEN_K is false. Try seeing if it's fixed if it's been a while."
        )
    if N % D != 0:
        pytest.skip(
            "N must be divisible by D as N = D * (S1 + S2)"
        )

    # deconstruct N
    S = N // D
    S1 = S // 2
    S2 = S - S1

    dtype = str_to_torch_dtype[dtype]
    x, w, y, x_scale, w_scale = generate_fused_gemm_a8w8_blockscale_split_cat_inputs(
        M,
        N,
        K,
        S3,
        block_shape_n,
        block_shape_k,
        dtype=dtype,
        layout=layout,
    )

    c1_torch, c2_torch = run_torch(x, w, y, x_scale, w_scale, S1, S2, D, dtype)
    c1_triton, c2_triton = run_triton(x, w, y, x_scale, w_scale, S1, S2, D, dtype)

    torch.testing.assert_close(c1_torch, c1_triton, atol=0.01, rtol=1e-2)
    torch.testing.assert_close(c2_torch, c2_triton, atol=0.01, rtol=1e-2)
