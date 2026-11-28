# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.normalization.channel_rmsnorm import channel_rmsnorm
from aiter.ops.triton.utils._triton.arch_info import get_arch

# Archs with a channel_rmsnorm config.
SUPPORTED_ARCHS = ("gfx1150", "gfx1151", "gfx1201")

_ACT = {None: lambda t: t, "silu": F.silu, "gelu": F.gelu}


def torch_channel_rmsnorm_ref(x, gamma, bias=None, activation=None, eps=1e-12):
    """diffusers WanRMS_norm(channel_first=True), then the activation."""
    C = x.shape[1]
    shape = (C,) + (1,) * (x.dim() - 2)
    xf = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    y = F.normalize(xf, dim=1, eps=eps).to(x.dtype)
    y = y * C**0.5 * gamma.view(shape)
    if bias is not None:
        y = y + bias.view(shape)
    return _ACT[activation](y)


def generate_channel_rmsnorm_inputs(shape, dtype, has_bias, device="cuda"):
    """(x, gamma, bias) with a non-default gamma. Shared with the benchmark."""
    torch.manual_seed(0)
    C = shape[1]
    x = torch.randn(shape, dtype=dtype, device=device)
    gamma = 1.0 + 0.5 * torch.randn(C, dtype=dtype, device=device)
    bias = 0.1 * torch.randn(C, dtype=dtype, device=device) if has_bias else None
    return x, gamma, bias


@pytest.mark.parametrize(
    "shape",
    [
        (1, 96, 2, 30, 52),
        (1, 384, 1, 60, 104),
        (2, 192, 3, 17, 33),
        (1, 1024, 1, 22, 40),
        (1, 1000, 2, 9, 11),
        (1, 2048, 1, 8, 8),
        (2, 64, 17, 33),
        (5, 3),
        (1, 16, 0, 4, 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("activation", [None, "silu"])
def test_channel_rmsnorm(shape, dtype, has_bias, activation):
    if get_arch() not in SUPPORTED_ARCHS:
        pytest.skip(f"no channel_rmsnorm config for {get_arch()}")
    x, gamma, bias = generate_channel_rmsnorm_inputs(shape, dtype, has_bias)
    ref = torch_channel_rmsnorm_ref(x, gamma, bias, activation)
    out = channel_rmsnorm(x, gamma, bias, activation)
    assert out.shape == x.shape and out.dtype == x.dtype
    if dtype == torch.float32:
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    else:
        # The fp32 sum order differs from PyTorch. A few normalized values round
        # one step apart, and sqrt(C) * gamma makes that up to three output steps.
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_channel_rmsnorm_zero_rows(dtype):
    """A zero channel vector gives zero, not NaN: the norm is bounded by eps."""
    if get_arch() not in SUPPORTED_ARCHS:
        pytest.skip(f"no channel_rmsnorm config for {get_arch()}")
    x, gamma, _ = generate_channel_rmsnorm_inputs((1, 96, 1, 8, 8), dtype, False)
    x[:, :, :, 0, :] = 0
    ref = torch_channel_rmsnorm_ref(x, gamma, None, "silu")
    out = channel_rmsnorm(x, gamma, None, "silu")
    assert not out.isnan().any()
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
