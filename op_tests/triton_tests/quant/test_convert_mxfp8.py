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
from aiter.ops.triton.utils._triton import arch_info

# Decorator for tests that require the gfx950 ASM path.
_gfx950_only = pytest.mark.skipif(
    arch_info.get_arch() != "gfx950",
    reason="ASM v_cvt_scalef32_* instructions require gfx950",
)

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


@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_mxfp8_scale_independent_dequant(is_2d_block, fp8_dtype):
    """Dequantize with an independent e8m0 decode (2**(s-127)), not convert_from.

    convert_from shares the kernel's scale interpretation, so a matching
    encode/decode bug could cancel in the roundtrip test. Decoding the e8m0
    scale independently and applying it in plain torch catches a wrong scale
    value or a wrong block layout.
    """
    torch.manual_seed(2)
    M, N, qbs = 128, 256, 32
    x = torch.randn(M, N, device="cuda", dtype=torch.float32) * 3.0
    y, s = convert_to_mxfp8(
        x, fp8_dtype, quant_block_size=qbs, is_2d_block=is_2d_block, use_asm=False
    )
    # e8m0 biased byte -> dequant scale 2**(s-127), broadcast over the block.
    scale = torch.exp2(s.float() - 127.0)
    if is_2d_block:
        scale = scale.repeat_interleave(qbs, dim=0).repeat_interleave(qbs, dim=1)
    else:
        scale = scale.repeat_interleave(qbs, dim=1)
    deq = y.float() * scale
    bound = _TOL[fp8_dtype] * x.abs().max().clamp_min(1e-4)
    err = (deq - x).abs().max()
    assert err <= bound, f"independent dequant err {err:.4f} > {bound:.4f}"


def test_mxfp8_rejects_unaligned_shape():
    """M/N not aligned to the tile must raise (kernel loads full tiles unmasked)."""
    x = torch.randn(100, 100, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        convert_to_mxfp8(x, torch.float8_e4m3fn, use_asm=False)


def test_mxfp8_rejects_unsupported_dtype():
    """fp8_dtype outside {e4m3fn, e5m2} must raise ValueError."""
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="fp8_dtype"):
        convert_to_mxfp8(x, torch.float8_e4m3fnuz, use_asm=False)


def test_mxfp8_sr_unbiased_portable():
    """SR output is unbiased: mean(dequant(SR(x))) ≈ mean(x) over a large tensor.

    The kernel uses a fixed philox seed with position-dependent offsets, so
    each element receives a distinct random value — a single large forward
    pass acts as many independent draws.  The test also verifies that SR
    produces different bits than deterministic rounding, confirming the path
    is actually exercised.
    """
    torch.manual_seed(7)
    M, N, qbs = 512, 512, 32
    fp8_dtype = torch.float8_e4m3fn
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)

    y_sr, s_sr = convert_to_mxfp8(
        x, fp8_dtype, quant_block_size=qbs, use_sr=True, use_asm=False
    )
    y_det, _ = convert_to_mxfp8(
        x, fp8_dtype, quant_block_size=qbs, use_sr=False, use_asm=False
    )

    # SR must activate — same input, different rounding decisions.
    assert not torch.equal(y_sr, y_det), "SR and deterministic produced identical bits"

    # Mean of SR dequantized output tracks the input mean (unbiasedness).
    xr = convert_from_mxfp8(
        y_sr, s_sr, torch.float32, quant_block_size=qbs, use_asm=False
    )
    mean_err = (xr.mean() - x.mean()).abs().item()
    # Over 512×512 elements the CLT gives std(err) ≈ 0.16/sqrt(262144) ≈ 3e-4;
    # 0.02 is a very conservative bound.
    assert mean_err < 0.02, (
        f"SR mean {xr.mean():.4f} diverges from input mean {x.mean():.4f} "
        f"by {mean_err:.4f}"
    )


@_gfx950_only
@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.bfloat16])
def test_mxfp8_asm_matches_portable(is_2d_block, in_dtype):
    """On gfx950, the ASM path must produce results equivalent to portable Triton.

    Scales are expected to be bitwise identical (same _calculate_scales logic).
    Dequantized values must agree within FP8 precision.
    """
    torch.manual_seed(5)
    M, N, qbs = 128, 256, 32
    fp8_dtype = torch.float8_e4m3fn
    x = torch.randn(M, N, device="cuda", dtype=in_dtype)

    y_asm, s_asm = convert_to_mxfp8(
        x, fp8_dtype, quant_block_size=qbs, is_2d_block=is_2d_block, use_asm=True
    )
    y_port, s_port = convert_to_mxfp8(
        x, fp8_dtype, quant_block_size=qbs, is_2d_block=is_2d_block, use_asm=False
    )

    assert torch.equal(s_asm, s_port), "ASM and portable e8m0 scales differ"

    xr_asm = convert_from_mxfp8(
        y_asm,
        s_asm,
        in_dtype,
        quant_block_size=qbs,
        is_2d_block=is_2d_block,
        use_asm=True,
    )
    xr_port = convert_from_mxfp8(
        y_port,
        s_port,
        in_dtype,
        quant_block_size=qbs,
        is_2d_block=is_2d_block,
        use_asm=False,
    )

    scale = x.abs().max().clamp_min(1e-4)
    err = (xr_asm.float() - xr_port.float()).abs().max()
    assert (
        err <= _TOL[fp8_dtype] * scale
    ), f"ASM vs portable max err {err:.4f} > {_TOL[fp8_dtype]} * {scale:.4f}"
