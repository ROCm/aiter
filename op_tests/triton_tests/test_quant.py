# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest

from aiter.ops.triton.quant import static_per_tensor_fp8_quant
from aiter.ops.triton.quant import dynamic_per_tensor_fp8_quant
from aiter.ops.triton.quant import dynamic_per_token_fp8_quant


def torch_static_per_tensor_fp8_quant(out, x, scale):
    out = (x / scale).to(torch.float8_e4m3fnuz)

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
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_static_per_tensor_fp8_quant(M: int, N: int, dtype):
    torch.manual_seed(20)
    x = torch.randn((M, N), dtype=dtype, device="cuda")
    scale = torch.randn(1, dtype=torch.float32, device="cuda")

    torch_out = torch.zeros((M, N), dtype=torch.float8_e4m3fnuz, device="cuda")
    torch_out = torch_static_per_tensor_fp8_quant(torch_out, x, scale)

    triton_out = torch.empty_like(x, dtype=torch.float8_e4m3fnuz, device="cuda")
    triton_out = static_per_tensor_fp8_quant(triton_out, x, scale)

    # Note: Torch doesn't support comparing fp8 type
    torch.testing.assert_close(
        triton_out.to(dtype=torch.float32),
        torch_out.to(dtype=torch.float32),
        atol=1e-02,
        rtol=1e-02,
    )


def torch_dynamic_per_tensor_fp8_quant(x):
    # Triton does max and scale in f32 so we need to match precision here.
    x_f32 = x.to(torch.float32)
    x_max = torch.max(torch.abs(x_f32))
    scale_out = x_max / torch.finfo(torch.float8_e4m3fnuz).max

    out = (x_f32 / scale_out).to(torch.float8_e4m3fnuz)

    return out, torch.tensor([scale_out], dtype=x.dtype, device=x.device)


@pytest.mark.parametrize(
    "M, N", [(1, 32), (32, 32), (2, 16), (10, 128), (32, 8192), (93, 75)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_dynamic_per_tensor_fp8_quant(M: int, N: int, dtype):
    torch.manual_seed(20)
    x = torch.randn((M, N), dtype=dtype, device="cuda")

    torch_out, torch_scale_out = torch_dynamic_per_tensor_fp8_quant(x)

    triton_out = torch.empty_like(x, dtype=torch.float8_e4m3fnuz, device="cuda")
    triton_scale_out = torch.zeros(1, dtype=torch.float32, device="cuda")
    triton_out, triton_scale_out = dynamic_per_tensor_fp8_quant(
        triton_out, x, triton_scale_out
    )

    # Note: Torch doesn't support comparing fp8 type
    torch.testing.assert_close(
        triton_scale_out.to(dtype=torch.float32),
        torch_scale_out.to(dtype=torch.float32),
        atol=1e-01,
        rtol=1e-01,
    )
    torch.testing.assert_close(
        triton_out.to(dtype=torch.float32),
        torch_out.to(dtype=torch.float32),
        atol=1e-01,
        rtol=1e-01,
    )


def torch_dynamic_per_token_fp8_quant(x):
    x_max, _ = torch.max(torch.abs(x), axis=-1)
    scale_out = x_max / torch.finfo(torch.float8_e4m3fnuz).max

    out = x / scale_out[:, None]
    out = out.to(torch.float8_e4m3fnuz)

    return out, scale_out


@pytest.mark.parametrize(
    "M, N",
    [
        (1, 32),
        (32, 32),
        (2, 16),
        (10, 128),
        (32, 4096),
        (1024, 128),
        (193, 76),
        (256, 13),
        (400, 400),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
# TODO Fix accuracy issues. Some shapes produce very close but not
# exact values in Triton vs pytorch. The quant to fp8 then produces a large difference.
# @pytest.mark.parametrize('M, N', [(1,32), (32,32), (2,16), (10,128), (32, 4096), (1024,128), (193,76), (256,13), (400,400)])
# @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_dynamic_per_token_fp8_quant(M: int, N: int, dtype):
    torch.manual_seed(20)
    torch.set_printoptions(precision=7, threshold=4000)
    x = torch.randn((M, N), dtype=dtype, device="cuda")

    torch_out, torch_scale_out = torch_dynamic_per_token_fp8_quant(x)

    triton_scale_out = torch.zeros(M, dtype=torch.float32, device="cuda")
    triton_out = torch.empty_like(x, dtype=torch.float8_e4m3fnuz, device="cuda")
    triton_out, triton_scale_out = dynamic_per_token_fp8_quant(
        triton_out, x, triton_scale_out
    )

    # Note: Torch doesn't support comparing fp8 type
    torch.testing.assert_close(
        triton_scale_out.to(dtype=torch.float32),
        torch_scale_out.to(dtype=torch.float32),
        atol=1e-01,
        rtol=1e-01,
    )
    torch.testing.assert_close(
        triton_out.to(dtype=torch.float32),
        torch_out.to(dtype=torch.float32),
        atol=1e-01,
        rtol=1e-01,
    )
