# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.conv.conv2d_depthwise import conv2d_depthwise
from aiter.ops.triton.utils._triton.arch_info import get_arch

# Archs with a CONV2D-DEPTHWISE config.
SUPPORTED_ARCHS = ("gfx1150", "gfx1151", "gfx1201")

_ACT = {"silu": F.silu, "gelu": F.gelu}


def torch_conv2d_depthwise_eager(
    x, weight, bias, stride, padding, dilation, activation=None, glu_activation=None
):
    """Eager depthwise conv in the input dtype, then the activation or the SANA
    GLUMBConv GLU. The benchmark baseline."""
    y = F.conv2d(x, weight, bias, stride, padding, dilation, groups=x.shape[1])
    return _apply_epilogue(y, activation, glu_activation)


def torch_conv2d_depthwise_ref(
    x, weight, bias, stride, padding, dilation, activation=None, glu_activation=None
):
    """Reference: the conv in fp32 rounded once to the input dtype (with bf16 /
    fp16 input, MIOpen rounds the conv and then adds the bias, a second
    rounding), then the eager activation or GLU."""
    b = bias.float() if bias is not None else None
    y = F.conv2d(
        x.float(), weight.float(), b, stride, padding, dilation, groups=x.shape[1]
    ).to(x.dtype)
    return _apply_epilogue(y, activation, glu_activation)


def _apply_epilogue(y, activation, glu_activation):
    if glu_activation:
        a, gate = torch.chunk(y, 2, dim=1)
        return a * _ACT[glu_activation](gate)
    if activation:
        return _ACT[activation](y)
    return y


def generate_conv2d_depthwise_inputs(N, C, H, W, R, S, dtype, has_bias):
    """(x, weight, bias). Shared with the benchmark."""
    torch.manual_seed(0)
    x = torch.randn((N, C, H, W), dtype=dtype, device="cuda")
    weight = torch.randn((C, 1, R, S), dtype=dtype, device="cuda") / (R * S) ** 0.5
    bias = torch.randn((C,), dtype=dtype, device="cuda") if has_bias else None
    return x, weight, bias


@pytest.mark.parametrize(
    "N, C, H, W, R, S, stride, padding, dilation",
    [
        (2, 64, 32, 32, 3, 3, 1, 1, 1),
        (1, 96, 33, 47, 3, 3, 1, 1, 1),
        (1, 48, 30, 30, 5, 5, 1, 2, 1),
        (1, 32, 40, 40, 7, 7, 1, 3, 1),
        (2, 40, 37, 29, 3, 3, 2, 1, 1),
        (1, 24, 31, 31, 3, 3, 1, 2, 2),
        (1, 16, 9, 64, 3, 5, 1, (1, 2), 1),
        (1, 8, 5, 5, 3, 3, 1, 0, 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("mode", ["none", "silu", "gelu", "glu_silu"])
def test_conv2d_depthwise(
    N, C, H, W, R, S, stride, padding, dilation, dtype, has_bias, mode
):
    if get_arch() not in SUPPORTED_ARCHS:
        pytest.skip(f"no CONV2D-DEPTHWISE config for {get_arch()}")
    activation = mode if mode in _ACT else None
    glu = "silu" if mode == "glu_silu" else None
    x, weight, bias = generate_conv2d_depthwise_inputs(
        N, C, H, W, R, S, dtype, has_bias
    )
    ref = torch_conv2d_depthwise_ref(
        x, weight, bias, stride, padding, dilation, activation, glu
    )
    out = conv2d_depthwise(x, weight, bias, stride, padding, dilation, activation, glu)
    assert out.shape == ref.shape and out.dtype == ref.dtype and out.is_contiguous()
    if dtype == torch.float32:
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    else:
        # The fp32 tap order differs from the eager conv, so a conv output can
        # round one step apart before the activation.
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
