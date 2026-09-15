# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest

from aiter.ops.triton.gemm.fused.fused_gemm_a16w16_split_cat import (
    fused_gemm_a16w16_split_cat,
)
from aiter.ops.triton.utils.types import str_to_torch_dtype


def run_torch(x, w, y, S1, S2, D, out_dtype):
    """Reference: GEMM (fp32 accumulate) -> reshape [D, S1+S2] -> split ->
    concat y onto the K (nope) part -> cast to out_dtype."""
    c = torch.mm(x.to(torch.float32), w.to(torch.float32).T)
    c = c.view(-1, D, S1 + S2)
    c1, c2 = c.split([S1, S2], dim=-1)
    c1 = torch.cat([c1, y.to(torch.float32).expand((*c1.shape[:-1], -1))], dim=-1)
    return c1.to(out_dtype), c2.to(out_dtype)


def run_triton(x, w, y, S1, S2, D, out_dtype):
    m = x.shape[0]
    return fused_gemm_a16w16_split_cat(x, w, y.expand(m, D, -1), S1, S2, out_dtype)


def generate_inputs(M, N, K, S3, in_dtype):
    torch.manual_seed(5)
    x = torch.randn((M, K), dtype=in_dtype, device="cuda")
    w = torch.randn((N, K), dtype=in_dtype, device="cuda")
    y = torch.randn((M, S3), dtype=in_dtype, device="cuda").unsqueeze(1)  # (M, 1, S3)
    return x, w, y


def get_shapes():
    # (M, N, K): N must equal D * (S1 + S2)
    return [
        (16, 4096, 512),
        (64, 4096, 512),
        (128, 4096, 512),
        (256, 4096, 512),
        (1024, 4096, 512),
        (64, 8192, 1024),
        (256, 32768, 512),  # DeepSeek-R1 kv_b shape (with D=128)
    ]


@pytest.mark.parametrize(
    "in_dtype, out_dtype, M, N, K, D, S3",
    [
        (in_dtype, out_dtype, *shape, d, s3)
        for in_dtype in ["bf16"]
        for out_dtype in ["bf16", "float8_e4m3fn"]
        for shape in get_shapes()
        for d in [16, 128]
        for s3 in [64]
    ],
)
def test_fused_gemm_a16w16_split_cat(in_dtype, out_dtype, M, N, K, D, S3):
    if N % D != 0:
        pytest.skip("N must be divisible by D as N = D * (S1 + S2)")

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    in_dtype = str_to_torch_dtype[in_dtype]
    out_dtype = str_to_torch_dtype[out_dtype]

    S = N // D
    S1 = S // 2
    S2 = S - S1

    x, w, y = generate_inputs(M, N, K, S3, in_dtype)

    c1_torch, c2_torch = run_torch(x, w, y, S1, S2, D, out_dtype)
    c1_triton, c2_triton = run_triton(x, w, y, S1, S2, D, out_dtype)

    # fp8 e4m3 has ~3 mantissa bits (relative step ~1/8). The kernel accumulates
    # in fp32 from bf16 inputs and casts to fp8; the reference matmul precision
    # differs slightly, so a handful of boundary-straddling elements may land one
    # e4m3 step apart. Allow a single-step tolerance for fp8 output.
    if out_dtype == torch.float8_e4m3fn:
        atol, rtol = 0.125, 0.125
    else:
        atol, rtol = 0.01, 1e-2

    torch.testing.assert_close(
        c1_torch.to(torch.float32), c1_triton.to(torch.float32), atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        c2_torch.to(torch.float32), c2_triton.to(torch.float32), atol=atol, rtol=rtol
    )
