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


def _te_mxfp8_reference(x, is_2d_block, fp8_dtype, qbs=32):
    """Transformer Engine MXFP8BlockScaling in plain torch: byte =
    ceil(log2(amax / max_pos)) + 127 clamped to [0, 254] (0 for amax 0), then a
    saturating RNE cast of x * 2^(127 - byte)."""
    max_pos = torch.finfo(fp8_dtype).max
    M, N = x.shape
    xf = x.float()
    if is_2d_block:
        blocks = xf.view(M // qbs, qbs, N // qbs, qbs).abs().amax((1, 3))
    else:
        blocks = xf.view(M, N // qbs, qbs).abs().amax(-1)
    v = blocks * torch.tensor(1.0 / max_pos, dtype=torch.float32)
    mant, exp = torch.frexp(v)
    ceil_log2 = torch.where(mant == 0.5, exp - 1, exp)
    byte = torch.where(v > 0, (ceil_log2 + 127).clamp(0, 254), torch.zeros_like(exp))
    rcp = torch.exp2((127 - byte).float())
    if is_2d_block:
        rcp = rcp.repeat_interleave(qbs, 0).repeat_interleave(qbs, 1)
    else:
        rcp = rcp.repeat_interleave(qbs, 1)
    q = (xf * rcp).clamp(-max_pos, max_pos).to(fp8_dtype)
    return q, byte.to(torch.uint8)


@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.bfloat16])
def test_mxfp8_te_scale_mode_bitexact(is_2d_block, fp8_dtype, in_dtype):
    """scale_mode="te" is bit-exact with TE: scale bytes and fp8 data."""
    torch.manual_seed(3)
    M, N = 128, 256
    x = torch.randn(M, N, device="cuda") * torch.logspace(
        -30, 30, M, device="cuda"
    ).unsqueeze(1)
    # block maxima where torchao's rule picks a smaller power of two than TE's
    x[64, 0], x[65, 0], x[66, 0], x[67, :32] = 460.0, 230.0, 448.0, 0.0
    x = x.to(in_dtype)
    y, s = convert_to_mxfp8(x, fp8_dtype, is_2d_block=is_2d_block, scale_mode="te")
    y_ref, s_ref = _te_mxfp8_reference(x, is_2d_block, fp8_dtype)
    assert torch.equal(s, s_ref), "TE-mode e8m0 bytes differ from TE"
    assert torch.equal(y.view(torch.uint8), y_ref.view(torch.uint8)), "fp8 data"


def test_mxfp8_default_scale_mode_unchanged():
    """The default (torchao) rule still gives amax 460 the scale 2^0 (the block max
    does not fit e4m3); TE mode gives 2^1 and stores 460 / 2 -> 224."""
    x = torch.zeros(64, 64, device="cuda")
    x[0, 0] = 460.0
    _, s = convert_to_mxfp8(x, torch.float8_e4m3fn, use_asm=False)
    y_te, s_te = convert_to_mxfp8(x, torch.float8_e4m3fn, scale_mode="te")
    assert s[0, 0].item() == 127
    assert s_te[0, 0].item() == 128 and y_te[0, 0].float().item() == 224.0


def test_mxfp8_te_scale_mode_rejects_asm_and_sr():
    x = torch.randn(64, 64, device="cuda")
    for kw in ({"use_asm": True}, {"use_sr": True}):
        with pytest.raises(ValueError):
            convert_to_mxfp8(x, torch.float8_e4m3fn, scale_mode="te", **kw)


@pytest.mark.parametrize("M, N", [(64, 128), (192, 256), (128, 7168)])
@pytest.mark.parametrize("transposed", [False, True])
def test_mxfp8_preshuffle_matches_shuffle_weight(M, N, transposed):
    """preshuffle=True writes exactly shuffle_weight(convert_to_mxfp8(x), (16, 16))."""
    from aiter.ops.shuffle import shuffle_weight

    torch.manual_seed(4)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 7
    if transposed:
        x = x.t().contiguous().t()  # a strided view, as a transposed weight
    y, s = convert_to_mxfp8(x, torch.float8_e4m3fn, scale_mode="te")
    y_pre, s_pre = convert_to_mxfp8(
        x, torch.float8_e4m3fn, scale_mode="te", preshuffle=True
    )
    ref = shuffle_weight(y, layout=(16, 16))
    assert torch.equal(y_pre.view(torch.uint8), ref.view(torch.uint8))
    assert torch.equal(s_pre, s)


def test_mxfp8_fast_launch_matches_dispatch():
    """The cached launcher gives the same bits as Triton's ordinary dispatch,
    across repeated shapes and the TE-mode options."""
    from aiter.ops.triton.utils import fast_launch

    torch.manual_seed(5)
    cases = [
        (torch.randn(64, 128, device="cuda"), {}),
        (torch.randn(128, 256, device="cuda").bfloat16(), {"preshuffle": True}),
        (torch.randn(192, 64, device="cuda"), {"round_input_bf16": True}),
        (torch.randn(64, 192, device="cuda").t(), {"dequant_dtype": torch.bfloat16}),
    ]
    for _ in range(2):  # the second pass hits the launcher cache
        for x, kw in cases:
            fast = convert_to_mxfp8(x, torch.float8_e4m3fn, scale_mode="te", **kw)
            with fast_launch.bypassed():
                ref = convert_to_mxfp8(x, torch.float8_e4m3fn, scale_mode="te", **kw)
            for f, r in zip(fast, ref):
                assert torch.equal(f.view(torch.uint8), r.view(torch.uint8))


@pytest.mark.parametrize(
    "M, N, out_cols",
    [(100, 96, 96), (100, 100, 128), (37, 256, 256), (130, 70, 96), (48, 128, 128)],
)
@pytest.mark.parametrize("mode", ["fp8", "dequant", "round", "preshuffle"])
def test_mxfp8_unaligned_matches_zero_padding(M, N, out_cols, mode):
    """allow_unaligned (masked loads/stores) gives the bits of the zero-padded cast."""
    import torch.nn.functional as F

    if mode == "preshuffle" and (M % 16 or N != out_cols or N % 32):
        pytest.skip("preshuffle needs 16-row tiles and 32-column blocks")
    torch.manual_seed(6)
    x = torch.randn(M, N, device="cuda") * 5
    kw = {
        "fp8": {},
        "dequant": {"dequant_dtype": torch.bfloat16},
        "round": {"round_input_bf16": True},
        "preshuffle": {"preshuffle": True},
    }[mode]
    if mode != "round":
        x = x.bfloat16()
    y, s = convert_to_mxfp8(
        x,
        torch.float8_e4m3fn,
        scale_mode="te",
        allow_unaligned=True,
        out_cols=out_cols,
        **kw,
    )
    xp = F.pad(x, (0, -(-out_cols // 64) * 64 - N, 0, (-M) % 64))
    yp, sp = convert_to_mxfp8(xp, torch.float8_e4m3fn, scale_mode="te", **kw)
    if mode == "preshuffle":
        # whole 16-row tiles: the prefix of the padded output
        assert torch.equal(
            y.view(torch.uint8).flatten(), yp.view(torch.uint8).flatten()[: M * N]
        )
    else:
        bits = torch.int16 if y.element_size() == 2 else torch.uint8
        assert torch.equal(y.view(bits), yp[:M, :out_cols].contiguous().view(bits))
    assert torch.equal(s, sp[:M, : -(-out_cols // 32)])
