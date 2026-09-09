# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for fast_transpose_2d Triton kernel."""

import pytest
import torch

from aiter.ops.triton.quant.fast_transpose import fast_transpose_2d


@pytest.mark.parametrize(
    "M, N",
    [
        (1, 1),
        (32, 32),
        (64, 128),
        (128, 64),
        (1024, 512),
        (511, 257),  # non-power-of-2
        (1, 4096),
        (4096, 1),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.float16, torch.float8_e4m3fnuz],
)
def test_fast_transpose_2d_correctness(M, N, dtype):
    torch.manual_seed(0)
    if dtype == torch.float8_e4m3fnuz:
        x = torch.randn(M, N, device="cuda").to(dtype)
        ref = x.view(torch.int8).t().contiguous().view(dtype)
    else:
        x = torch.randn(M, N, dtype=dtype, device="cuda")
        ref = x.t().contiguous()

    out = fast_transpose_2d(x)

    assert out.shape == (N, M), f"shape mismatch: {out.shape} vs {(N, M)}"
    assert out.is_contiguous(), "output is not contiguous"
    assert out.dtype == dtype

    if dtype == torch.float8_e4m3fnuz:
        assert torch.equal(
            out.view(torch.int8), ref.view(torch.int8)
        ), "FP8 bit pattern mismatch"
    else:
        torch.testing.assert_close(out, ref)
