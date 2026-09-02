# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for the convert_to_mxfp8 / convert_from_mxfp8 wrappers.

These cover ``aiter.ops.triton.quant.quant_mxfp8`` (full-tensor MXFP8 convert),
which is distinct from the fused/dynamic MXFP8 quant ops in
``test_quant_mxfp8.py``.

The ASM fast path uses gfx950-only ``v_cvt_scalef32_*`` instructions, so these
tests exercise the portable ``use_asm=False`` path and verify the convert →
deconvert roundtrip stays within MXFP8 (block-scaled e8m0) precision.
"""

import pytest
import torch

from aiter.ops.triton.quant.quant_mxfp8 import convert_from_mxfp8, convert_to_mxfp8

# e4m3 keeps 3 mantissa bits, e5m2 only 2 — allow a looser bound for e5m2.
_TOL = {torch.float8_e4m3fn: 0.16, torch.float8_e5m2: 0.35}


@pytest.mark.parametrize("M, N", [(64, 64), (128, 256), (256, 128)])
@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.bfloat16])
def test_mxfp8_roundtrip(M, N, is_2d_block, fp8_dtype, in_dtype):
    torch.manual_seed(0)
    qbs = 32
    x = torch.randn(M, N, device="cuda", dtype=in_dtype) * 3.0

    y, s = convert_to_mxfp8(
        x,
        fp8_dtype,
        quant_block_size=qbs,
        is_2d_block=is_2d_block,
        use_asm=False,
    )
    assert y.shape == (M, N)
    assert y.dtype == fp8_dtype
    assert s.dtype == torch.uint8

    xr = convert_from_mxfp8(
        y,
        s,
        in_dtype,
        quant_block_size=qbs,
        is_2d_block=is_2d_block,
        use_asm=False,
    )
    assert xr.shape == (M, N)
    assert xr.dtype == in_dtype

    scale = x.abs().max().clamp_min(1e-4)
    err = (xr.float() - x.float()).abs().max()
    assert err <= _TOL[fp8_dtype] * scale, f"max abs err {err:.4f} vs {scale:.4f}"


def test_mxfp8_rejects_unaligned_shape():
    """M/N not aligned to the tile must raise (kernel loads full tiles unmasked)."""
    x = torch.randn(100, 100, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        convert_to_mxfp8(x, torch.float8_e4m3fn, use_asm=False)
