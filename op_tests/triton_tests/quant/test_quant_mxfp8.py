# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from aiter.ops.triton.quant.mxfp8_quant import downcast_to_mxfp8, upcast_from_mxfp8


def get_max_quant_val(dtype):
    if dtype == torch.float8_e4m3fn:
        return 448.0
    else:
        return 57344.0


def torch_downcast_to_mxfp8(
    x: torch.Tensor, dtype: torch.dtype, axis: int, SCALE_ROUNDING_MODE: int = 0
):
    # returns tensor and scale in fp32 post quantization

    ndim = x.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim

    x = x.to(torch.float32)
    x = x.transpose(axis, x.ndim - 1)
    orig_shape = x.shape
    quant_dim = orig_shape[-1]
    pad_length = 32 - quant_dim % 32
    if pad_length == 32:
        pad_length = 0
    padding = torch.empty(x.shape[:-1] + (pad_length,), dtype=x.dtype, device="cuda")
    padding.fill_(-1.0)
    x_padded = torch.cat((x, padding), -1)
    x_abs_padded = torch.cat((torch.abs(x), padding), -1)
    padded_shape = x_padded.shape

    new_shape = padded_shape[:-1] + (padded_shape[-1] // 32, 32)
    x_padded = x_padded.reshape(new_shape)
    x_abs_padded = x_abs_padded.reshape(new_shape)
    scale = torch.amax(x_abs_padded, -1)
    scale = scale / get_max_quant_val(dtype)
    if SCALE_ROUNDING_MODE == 0:
        scale = (scale.view(torch.int32) + 0x007FFFFF) & 0x7F800000
    else:
        scale = scale.view(torch.int32) & 0x7F800000
    scale = scale.view(torch.float32)

    scale_inv = torch.where(scale == 0.0, 0.0, 1.0 / scale).unsqueeze(-1)
    x_padded = x_padded * scale_inv
    x_padded = x_padded.reshape(padded_shape)
    x = x_padded[..., :quant_dim].clone()
    x = x.to(dtype).to(torch.float32)
    x = x.transpose(axis, x.ndim - 1)
    scale = scale.transpose(axis, x.ndim - 1)
    return x, scale


def upcast_scale(scale):
    scale = scale.to(torch.int32) << 23
    scale = scale.view(torch.float32)
    return scale


def torch_upcast_from_mxfp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    axis: int,
):
    ndim = x.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim

    x = x.to(torch.float32)
    x = x.transpose(axis, x.ndim - 1)
    scale = scale.transpose(axis, x.ndim - 1)
    orig_shape = x.shape
    quant_dim = orig_shape[-1]
    pad_length = 32 - quant_dim % 32
    if pad_length == 32:
        pad_length = 0
    padding = torch.empty(x.shape[:-1] + (pad_length,), dtype=x.dtype, device="cuda")
    padding.fill_(-1.0)
    x_padded = torch.cat((x, padding), -1)
    padded_shape = x_padded.shape

    new_shape = padded_shape[:-1] + (padded_shape[-1] // 32, 32)
    x_padded = x_padded.reshape(new_shape)
    scale = upcast_scale(scale).unsqueeze(-1)
    x_padded = x_padded * scale
    x_padded = x_padded.reshape(padded_shape)
    x = x_padded[..., :quant_dim].clone()
    x = x.transpose(axis, x.ndim - 1)
    x = x.to(dtype)
    return x


@pytest.mark.parametrize(
    "shape, axis",
    [
        ((1, 4), -1),
        ((1, 28), -1),
        ((1, 32), -1),
        ((1, 64), -1),
        ((1, 68), -1),
        ((2, 4), -1),
        ((2, 28), -1),
        ((2, 32), -1),
        ((2, 200, 64), 1),
        ((2, 68), -1),
        ((128, 4), 0),
        ((128, 28), -1),
        ((128, 32), -1),
        ((128, 64), -1),
        ((128, 68), -1),
        ((256, 32), -1),
        ((160, 40), -1),
        ((280, 20), -1),
    ],
)
@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("out_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("SCALE_ROUNDING_MODE", [0, 1])
def test_downcast_to_mxfp8(shape, axis, in_dtype, out_dtype, SCALE_ROUNDING_MODE):
    torch.cuda.empty_cache()
    torch.manual_seed(20)
    x = torch.randn(*shape, dtype=in_dtype, device="cuda")

    out_torch, out_scale_torch = torch_downcast_to_mxfp8(
        x, out_dtype, axis, SCALE_ROUNDING_MODE
    )
    out_triton, out_scale_triton = downcast_to_mxfp8(
        x, out_dtype, axis, SCALE_ROUNDING_MODE
    )
    out_triton = out_triton.to(torch.float32)
    out_scale_triton = upcast_scale(out_scale_triton)

    torch.testing.assert_close(out_triton, out_torch, atol=0.01, rtol=0.01)
    torch.testing.assert_close(out_scale_triton, out_scale_torch, atol=0.01, rtol=0.01)


@pytest.mark.parametrize(
    "shape, axis",
    [
        ((1, 4), -1),
        ((1, 28), -1),
        ((1, 32), -1),
        ((1, 64), -1),
        ((1, 68), -1),
        ((2, 4), -1),
        ((2, 28), -1),
        ((2, 32), -1),
        ((2, 200, 64), 1),
        ((2, 68), -1),
        ((128, 4), 0),
        ((128, 28), -1),
        ((128, 32), -1),
        ((128, 64), -1),
        ((128, 68), -1),
        ((256, 32), -1),
        ((160, 40), -1),
        ((280, 20), -1),
    ],
)
@pytest.mark.parametrize("in_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_upcast_from_mxfp8(shape, axis, in_dtype, out_dtype):
    torch.cuda.empty_cache()
    torch.manual_seed(20)
    x = torch.randn(*shape, dtype=out_dtype, device="cuda")
    x, x_scale = downcast_to_mxfp8(x, in_dtype, axis)
    out_triton = upcast_from_mxfp8(x, x_scale, out_dtype, axis)
    out_torch = torch_upcast_from_mxfp8(x, x_scale, out_dtype, axis)

    torch.testing.assert_close(out_triton, out_torch, atol=0.01, rtol=0.01)
