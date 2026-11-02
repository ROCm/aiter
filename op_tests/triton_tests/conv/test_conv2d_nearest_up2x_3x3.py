# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.conv.conv2d_nearest_up2x_3x3 import conv2d_nearest_up2x_3x3
from aiter.ops.triton.utils._triton.arch_info import get_arch

# Archs with a CONV2D-NEAREST-UP2X-3X3 config.
SUPPORTED_ARCHS = ("gfx1150", "gfx1151", "gfx1201")

# bf16 / fp16: the op rounds the folded 2x2 weights once to the input dtype and
# sums in another order than the conv on the upsampled tensor.
_TOL = {torch.bfloat16: (5e-2, 2e-2), torch.float16: (1e-2, 1e-2)}


def torch_nearest_up2x_conv3x3_eager(x, weight, bias, mode="nearest-exact"):
    """Eager: upsample x2 (as the Wan VAE WanUpsample), then the 3x3 conv."""
    return F.conv2d(
        F.interpolate(x, scale_factor=2.0, mode=mode), weight, bias, padding=1
    )


def torch_nearest_up2x_conv3x3_ref(x, weight, bias, mode="nearest-exact"):
    """Reference: the same in fp32, rounded once to the input dtype."""
    b = bias.float() if bias is not None else None
    xu = F.interpolate(x.float(), scale_factor=2.0, mode=mode)
    return F.conv2d(xu, weight.float(), b, padding=1).to(x.dtype)


def generate_nearest_up2x_conv3x3_inputs(N, C, K, H, W, dtype, has_bias, c_pad=0):
    """(x, weight, bias). Shared with the benchmark. With c_pad, x is a channel
    slice of a larger tensor (strided batch dim)."""
    torch.manual_seed(0)
    x = torch.randn((N, C + c_pad, H, W), dtype=dtype, device="cuda")[:, :C]
    weight = torch.randn((K, C, 3, 3), dtype=dtype, device="cuda") / (9 * C) ** 0.5
    bias = torch.randn((K,), dtype=dtype, device="cuda") if has_bias else None
    return x, weight, bias


@pytest.mark.parametrize(
    "N, C, K, H, W",
    [
        (1, 5, 7, 3, 4),
        (2, 70, 33, 9, 13),
        (1, 64, 64, 16, 16),
        (3, 128, 96, 7, 5),
        (1, 16, 8, 1, 1),
        (2, 384, 192, 30, 52),
        (1, 1024, 1024, 11, 20),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("c_pad", [0, 3])
def test_conv2d_nearest_up2x_3x3(N, C, K, H, W, dtype, has_bias, c_pad):
    if get_arch() not in SUPPORTED_ARCHS:
        pytest.skip(f"no CONV2D-NEAREST-UP2X-3X3 config for {get_arch()}")
    x, weight, bias = generate_nearest_up2x_conv3x3_inputs(
        N, C, K, H, W, dtype, has_bias, c_pad
    )
    out = conv2d_nearest_up2x_3x3(x, weight, bias)
    assert out.shape == (N, K, 2 * H, 2 * W) and out.dtype == dtype
    assert out.is_contiguous()
    atol, rtol = _TOL[dtype]
    # For a scale of 2, "nearest" and "nearest-exact" pick the same pixel.
    for mode in ("nearest-exact", "nearest"):
        ref = torch_nearest_up2x_conv3x3_ref(x, weight, bias, mode)
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


def test_conv2d_nearest_up2x_3x3_weight_update():
    """The folded weights are cached; an in-place weight update must refold."""
    if get_arch() not in SUPPORTED_ARCHS:
        pytest.skip(f"no CONV2D-NEAREST-UP2X-3X3 config for {get_arch()}")
    x, weight, bias = generate_nearest_up2x_conv3x3_inputs(
        1, 32, 16, 8, 8, torch.bfloat16, True
    )
    conv2d_nearest_up2x_3x3(x, weight, bias)
    weight.mul_(-2.0)
    out = conv2d_nearest_up2x_3x3(x, weight, bias)
    ref = torch_nearest_up2x_conv3x3_ref(x, weight, bias)
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=2e-2)
