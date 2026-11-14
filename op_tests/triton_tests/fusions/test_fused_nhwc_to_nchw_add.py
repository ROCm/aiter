# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.fusions.fused_nhwc_to_nchw_add import fused_nhwc_to_nchw_add
from aiter.ops.triton.utils._triton.arch_info import get_arch

# Archs with a FUSED_NHWC_TO_NCHW_ADD config.
SUPPORTED_ARCHS = ("gfx1150", "gfx1151", "gfx1201")


def torch_nhwc_to_nchw_add(x, residual, height, width):
    """diffusers Transformer2DModel exit: reshape, permute, contiguous, add."""
    B, C = x.shape[0], x.shape[-1]
    out = x.reshape(B, height, width, C).permute(0, 3, 1, 2).contiguous()
    return out if residual is None else out + residual


def generate_nhwc_to_nchw_add_inputs(B, C, H, W, dtype, has_residual, row_pad=0):
    """(x as (B, H * W, C), residual (B, C, H, W) or None). Shared with the benchmark.

    With row_pad, x is a view whose rows have a stride of C + row_pad.
    """
    torch.manual_seed(0)
    x = torch.randn((B, H * W, C + row_pad), dtype=dtype, device="cuda")[..., :C]
    residual = (
        torch.randn((B, C, H, W), dtype=dtype, device="cuda") if has_residual else None
    )
    return x, residual


@pytest.mark.parametrize(
    "B, C, H, W",
    [
        (2, 320, 64, 64),
        (2, 1280, 16, 16),
        (1, 640, 33, 47),
        (3, 77, 9, 13),
        (1, 3, 1, 1),
        (1, 8, 0, 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_residual", [True, False])
@pytest.mark.parametrize("x_layout", ["tokens", "nhwc", "strided_rows"])
def test_fused_nhwc_to_nchw_add(B, C, H, W, dtype, has_residual, x_layout):
    if get_arch() not in SUPPORTED_ARCHS:
        pytest.skip(f"no FUSED_NHWC_TO_NCHW_ADD config for {get_arch()}")
    row_pad = 5 if x_layout == "strided_rows" else 0
    x, residual = generate_nhwc_to_nchw_add_inputs(
        B, C, H, W, dtype, has_residual, row_pad
    )
    if x_layout == "nhwc":
        x = x.reshape(B, H, W, C)
    ref = torch_nhwc_to_nchw_add(x, residual, H, W)
    out = fused_nhwc_to_nchw_add(x, residual, H, W)
    assert out.shape == ref.shape and out.dtype == ref.dtype and out.is_contiguous()
    # Eager also adds in fp32 and rounds once, so the result is bit-identical.
    assert torch.equal(out, ref)
