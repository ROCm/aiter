# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest

from aiter.ops.triton.quant import (
    static_per_tensor_quant_fp8_i8,
    dynamic_per_tensor_quant_fp8_i8,
    dynamic_per_token_quant_fp8_i8,
)
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

DEBUG = False


def torch_static_per_tensor_quant_fp8_i8(out, x, scale, dtype_quant):
    out = (x / scale).to(dtype_quant)

    return out


@pytest.mark.parametrize(
    "M, N",
    [
        (1, 32),
        (32, 32),
        (2, 16),
        (10, 128),
        (32, 8192),
        (1024, 128),
        (2048, 1024),
        (193, 75),
    ],
)
@pytest.mark.parametrize("dtype_in", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dtype_quant", [torch.int8, get_fp8_e4m3_dtype()])
def test_static_per_tensor_quant(M: int, N: int, dtype_in, dtype_quant):
    torch.manual_seed(20)
    x = torch.randn((M, N), dtype=dtype_in, device="cuda")
    scale = torch.randn(1, dtype=torch.float32, device="cuda")

    torch_out = torch.zeros((M, N), dtype=dtype_quant, device="cuda")
    torch_out = torch_static_per_tensor_quant_fp8_i8(torch_out, x, scale, dtype_quant)

    triton_out = torch.empty_like(x, dtype=dtype_quant, device="cuda")
    triton_out = static_per_tensor_quant_fp8_i8(triton_out, x, scale)

    # Note: Torch doesn't support comparing fp8 type
    torch.testing.assert_close(
        triton_out.to(dtype=torch.float32),
        torch_out.to(dtype=torch.float32),
        atol=1e-02,
        rtol=1e-02,
    )


@pytest.mark.parametrize("dtype_quant", [torch.int8, get_fp8_e4m3_dtype()])
def test_static_per_tensor_quant_processes_row_zero(dtype_quant):
    x = torch.arange(1, 14, dtype=torch.float32, device="cuda").reshape(1, 13)
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    expected = x.to(dtype_quant)
    output = torch.zeros_like(x, dtype=dtype_quant)

    static_per_tensor_quant_fp8_i8(output, x, scale)

    torch.testing.assert_close(
        output.to(torch.float32), expected.to(torch.float32), atol=0, rtol=0
    )


def torch_dynamic_per_tensor_quant_fp8_i8(x, dtype_quant):
    # Triton does max and scale in f32 so we need to match precision here.
    x_f32 = x.to(torch.float32)
    x_max = torch.max(torch.abs(x_f32))
    dtype_max = (
        torch.iinfo(dtype_quant).max
        if dtype_quant == torch.int8
        else torch.finfo(dtype_quant).max
    )
    scale_out = x_max.to(torch.float32) / dtype_max

    out = (x_f32 / scale_out).to(dtype=dtype_quant)

    return out, torch.tensor([scale_out], dtype=torch.float32, device=x.device)


@pytest.mark.parametrize(
    "M, N", [(1, 32), (32, 32), (2, 16), (10, 128), (32, 8192), (93, 75)]
)
@pytest.mark.parametrize("dtype_in", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dtype_quant", [torch.int8, get_fp8_e4m3_dtype()])
def test_dynamic_per_tensor_quant(M: int, N: int, dtype_in, dtype_quant):
    torch.manual_seed(20)
    x = torch.randn((M, N), dtype=dtype_in, device="cuda")

    torch_out, torch_scale_out = torch_dynamic_per_tensor_quant_fp8_i8(x, dtype_quant)

    triton_out = torch.empty_like(x, dtype=dtype_quant, device="cuda")
    triton_scale_out = torch.zeros(1, dtype=torch.float32, device="cuda")
    triton_out, triton_scale_out = dynamic_per_tensor_quant_fp8_i8(
        triton_out, x, triton_scale_out
    )

    torch.testing.assert_close(
        triton_scale_out,
        torch_scale_out,
        atol=1e-01,
        rtol=1e-01,
    )

    # Note: Torch doesn't support comparing fp8 type yet
    torch.testing.assert_close(
        triton_out.to(dtype=torch.float32),
        torch_out.to(dtype=torch.float32),
        atol=1e-01,
        rtol=1e-01,
    )


@pytest.mark.parametrize("dtype_in", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dtype_quant", [torch.int8, get_fp8_e4m3_dtype()])
def test_dynamic_per_tensor_quant_zero_input(dtype_in, dtype_quant):
    x = torch.zeros((1, 13), dtype=dtype_in, device="cuda")
    output = torch.empty_like(x, dtype=dtype_quant)
    scale = torch.zeros(1, dtype=torch.float32, device="cuda")

    dynamic_per_tensor_quant_fp8_i8(output, x, scale)

    assert torch.isfinite(scale).all()
    assert (scale > 0).all()
    torch.testing.assert_close(
        output.to(torch.float32),
        torch.zeros_like(x, dtype=torch.float32),
        atol=0,
        rtol=0,
    )


def torch_dynamic_per_token_quant_fp8_i8(x, dtype_quant):
    x_max, _ = torch.max(torch.abs(x), axis=-1)
    dtype_max = (
        torch.iinfo(dtype_quant).max
        if dtype_quant == torch.int8
        else torch.finfo(dtype_quant).max
    )
    scale_out = x_max.to(torch.float32) / dtype_max

    scale_recip = 1 / scale_out[:, None]
    out = x * scale_recip
    out = out.to(dtype_quant)

    return out, scale_out


@pytest.mark.parametrize(
    "M, N",
    [
        (256, 13),
        (2, 16),
        (1, 32),
        (32, 32),
        (192, 96),
        (1024, 128),
        (48, 96),
        (400, 400),
        (32, 4096),
    ],
)
@pytest.mark.parametrize("dtype_in", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dtype_quant", [torch.int8, get_fp8_e4m3_dtype()])
def test_dynamic_per_token_quant(M: int, N: int, dtype_in, dtype_quant):
    torch.manual_seed(20)
    torch.set_printoptions(precision=7, threshold=4000)
    x = torch.rand((M, N), dtype=dtype_in, device="cuda")

    torch_out, torch_scale_out = torch_dynamic_per_token_quant_fp8_i8(x, dtype_quant)

    triton_scale_out = torch.zeros(M, dtype=torch.float32, device="cuda")
    triton_out = torch.empty_like(x, dtype=dtype_quant, device="cuda")
    triton_out, triton_scale_out = dynamic_per_token_quant_fp8_i8(
        triton_out, x, triton_scale_out
    )

    if DEBUG:
        print(f"Torch_Scale={torch_scale_out}")
        print(f"Triton_Scale={triton_scale_out}")

        print(f"x={x}")
        print(f"Torch_out={torch_out}")
        print(f"Triton_out={triton_out}")

    torch.testing.assert_close(
        triton_scale_out,
        torch_scale_out,
        atol=1e-01,
        rtol=1e-01,
    )

    # Note: Torch doesn't support comparing fp8 type yet
    torch.testing.assert_close(
        triton_out.to(dtype=torch.float32),
        torch_out.to(dtype=torch.float32),
        atol=1e-01,
        rtol=1e-01,
    )


@pytest.mark.parametrize("dtype_in", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dtype_quant", [torch.int8, get_fp8_e4m3_dtype()])
def test_dynamic_per_token_quant_zero_and_nonzero_rows(dtype_in, dtype_quant):
    x = torch.stack(
        (
            torch.zeros(13, dtype=dtype_in, device="cuda"),
            torch.arange(-6, 7, dtype=dtype_in, device="cuda"),
        )
    )
    dtype_max = (
        torch.iinfo(dtype_quant).max
        if dtype_quant == torch.int8
        else torch.finfo(dtype_quant).max
    )
    expected_max = torch.maximum(
        x.float().abs().amax(dim=-1), torch.tensor(1e-10, device=x.device)
    )
    expected_scale = expected_max / dtype_max
    expected_output = (x / expected_scale[:, None]).to(dtype_quant)
    output = torch.empty_like(x, dtype=dtype_quant)
    scale = torch.empty(x.shape[0], dtype=torch.float32, device="cuda")

    dynamic_per_token_quant_fp8_i8(output, x, scale)

    assert torch.isfinite(scale).all()
    torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)
    torch.testing.assert_close(
        output.to(torch.float32), expected_output.to(torch.float32), atol=0, rtol=0
    )
