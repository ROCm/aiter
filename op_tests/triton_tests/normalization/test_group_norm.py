# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.normalization.group_norm import group_norm


def generate_group_norm_inputs(shape, num_groups, dtype, affine, temb, seed=0):
    torch.manual_seed(seed)
    N, C = shape[0], shape[1]
    x = torch.randn(shape, dtype=dtype, device="cuda")
    weight = torch.randn(C, dtype=dtype, device="cuda") if affine else None
    bias = torch.randn(C, dtype=dtype, device="cuda") if affine else None
    t = torch.randn(N, C, dtype=dtype, device="cuda") if temb else None
    return x, weight, bias, t


def run_torch(x, num_groups, weight, bias, eps, temb, activation):
    """The unfused expression, in fp32."""
    h = x.float()
    if temb is not None:
        h = h + temb.float().view(*temb.shape, *([1] * (x.ndim - 2)))
    h = F.group_norm(
        h,
        num_groups,
        None if weight is None else weight.float(),
        None if bias is None else bias.float(),
        eps,
    )
    if activation == "silu":
        h = F.silu(h)
    elif activation == "gelu":
        h = F.gelu(h)
    return h.to(x.dtype)


def get_shapes():
    shapes = [((1, 32, 1, 2), 32)]  # two elements per group
    shapes += [((2, 64, 3, 5), 32)]  # odd spatial size
    shapes += [((3, 96, 17), 8)]  # 1-D spatial, 12 channels per group
    shapes += [((2, 320, 64, 64), 32)]  # SD1.5 UNet, 512px
    shapes += [((2, 1280, 16, 16), 32)]  # SD1.5 UNet, lowest level
    shapes += [((1, 128, 256, 256), 32)]  # VAE decoder, split groups
    shapes += [((1, 64, 4, 32, 32), 32)]  # 3-D VAE
    return shapes


@pytest.mark.parametrize("shape, num_groups", get_shapes())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("temb", [False, True])
@pytest.mark.parametrize("activation", [None, "silu", "gelu"])
def test_group_norm(shape, num_groups, dtype, affine, temb, activation):
    x, weight, bias, t = generate_group_norm_inputs(
        shape, num_groups, dtype, affine, temb
    )
    torch_out = run_torch(x, num_groups, weight, bias, 1e-5, t, activation)
    triton_out = group_norm(
        x, num_groups, weight, bias, 1e-5, temb=t, activation=activation
    )
    tol = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 5e-2}[dtype]
    torch.testing.assert_close(triton_out, torch_out, atol=tol, rtol=tol)


def test_group_norm_large_mean():
    """A group with a large mean and a small spread, against fp64."""
    torch.manual_seed(0)
    x = torch.randn(1, 128, 256, 256, device="cuda") * 0.01 + 1000.0
    got = group_norm(x, 32, eps=1e-5)
    want = F.group_norm(x.double(), 32, eps=1e-5)
    torch.testing.assert_close(got.double(), want, atol=1e-3, rtol=1e-3)


def test_group_norm_out_parameter():
    x, weight, bias, t = generate_group_norm_inputs(
        (2, 320, 32, 32), 32, torch.float16, True, True
    )
    dst = torch.empty_like(x)
    got = group_norm(x, 32, weight, bias, temb=t, activation="silu", out=dst)
    assert got.data_ptr() == dst.data_ptr()
    torch.testing.assert_close(
        dst, run_torch(x, 32, weight, bias, 1e-5, t, "silu"), atol=1e-2, rtol=1e-2
    )


def test_group_norm_rejects_bad_inputs():
    x, weight, bias, _ = generate_group_norm_inputs(
        (2, 64, 8, 8), 32, torch.float16, True, False
    )
    with pytest.raises(AssertionError):
        group_norm(x, 24, weight, bias)  # 64 is not a multiple of 24
    with pytest.raises(AssertionError):
        group_norm(x, 32, weight[:-1], bias)
    with pytest.raises(AssertionError):
        group_norm(x.transpose(2, 3), 32, weight, bias)  # not contiguous
