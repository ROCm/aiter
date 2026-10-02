# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import triton
import triton.language as tl

from aiter import logger
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
from aiter.ops.triton.quant import dynamic_mxfp4_quant, dynamic_nvfp4_quant
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.utility.fp4_utils import (
    dynamic_mxfp4_quant as fp4_utils_dynamic_mxfp4_quant,
)
from aiter.utility.fp4_utils import (
    e8m0_to_f32,
    mxfp4_to_f32,
)

DEVICE_ARCH = arch_info.get_arch()
_REQUIRES_GFX950 = pytest.mark.skipif(
    DEVICE_ARCH != "gfx950",
    reason="MXFP4 stochastic conversion requires gfx950",
)


def _dequantize_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize row-wise MXFP4 payload and E8M0 scales."""
    values = mxfp4_to_f32(packed)
    scale_f32 = e8m0_to_f32(scales).repeat_interleave(32, dim=-1)
    return values * scale_f32


def torch_dynamic_mxfp4_quant(
    x: torch.Tensor,
    scaling_mode: str = "even",
    is_nvfp4: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to MX FP4 format based of AMD Quark Spec.

    Math equivalent:
        blockscale_e8m0 = 2^(floor(log2(rounding(max_abs(x_block)))-max_exp))
        x_block_fp4 = x_block / blockscale_e8m0
        where max_exp = 2 for fp4_e2m1.

    Args:
        x: The input tensor, typically fp16 or bf16.
        scaling_mode: The method to calculate MX block scaling.
            - "even" (default): `even_round`.
    Returns:
        A tuple of (x_fp4, blockscale_e8m0).
    """
    # Create padded x. Needed because mxfp4 works with block of 32 elements
    QUANT_BLOCK_SIZE = 16 if is_nvfp4 else 32
    EXP_BIAS_FP32 = 127
    EXP_BIAS_FP4 = 1
    EBITS_F32 = 8
    EBITS_FP4 = 2
    MBITS_F32 = 23
    MBITS_FP4 = 1
    max_normal = 6
    min_normal = 1
    sign_mask = 1 << (EBITS_FP4 + MBITS_FP4)

    x_shape = x.shape
    if x.shape[-1] % QUANT_BLOCK_SIZE != 0:
        shape = list(x_shape)
        shape = shape[:-1] + [
            ((shape[-1] - 1 + QUANT_BLOCK_SIZE) // QUANT_BLOCK_SIZE) * QUANT_BLOCK_SIZE
        ]
        shape = tuple(shape)
        x_padded = torch.zeros((shape), device=x.device, dtype=x.dtype)
        x_padded[..., : x.shape[-1]] = x
    else:
        x_padded = x

    # Calculate scale
    x_padded = x_padded.reshape(
        -1, x_padded.shape[-1] // QUANT_BLOCK_SIZE, QUANT_BLOCK_SIZE
    ).to(torch.float32)
    amax, _ = torch.max(torch.abs(x_padded), dim=-1)
    if is_nvfp4:
        scale_e4m3 = amax.to(torch.float32) / 6.0

        # Compute quantized x
        qx = x_padded * (1.0 / scale_e4m3).unsqueeze(-1)

        block_scales = scale_e4m3.to(e4m3_dtype)
    else:
        amax = amax.view(torch.int32)
        amax = (amax + 0x200000) & 0xFF800000
        amax = amax.view(torch.float32)
        scale_e8m0_unbiased = torch.log2(amax).floor() - 2
        scale_e8m0_unbiased = torch.clamp(scale_e8m0_unbiased, min=-127, max=127)
        quant_scale = torch.exp2(-scale_e8m0_unbiased)

        # Compute quantized x
        qx = x_padded * quant_scale.unsqueeze(-1)

        # blockscale_e8m0
        block_scales = scale_e8m0_unbiased.to(torch.uint8) + 127

    # Convert to mxfp4 format
    #
    # Note: This code is adapted from Triton Bench numerics mxfp4 code
    #
    # Note: MXFP4  S:1-bit, E:2-bit, M:1-bit
    #   Zeros: S000 -> +/-0
    #   Denormal Numbers: S001 -> +/- 0.5
    #   Normal Numbers:
    #           S010 -> +/- 1.0
    #           S011 -> +/- 1.5
    #           S100 -> +/- 2.0
    #           S101 -> +/- 3.0
    #           S110 -> +/- 4.0
    #           S111 -> +/- 6.0
    # Convert quantized fp32 tensor to int32 before converting to mxfp4 format
    qx = qx.view(torch.int32)

    # Extract sign
    s = qx & 0x80000000
    # Set everything to positive, will add sign back at the end
    qx = qx ^ s

    qx_fp32 = qx.view(torch.float32)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = torch.logical_and(
        torch.logical_not(saturate_mask), qx_fp32 < min_normal
    )
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    # Denormal numbers
    denorm_exp = (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    denorm_mask_int = denorm_exp << MBITS_F32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(
        torch.float32
    )

    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    # Normal numbers
    normal_x = qx
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
    normal_x = normal_x.to(torch.uint8)

    # Merge results
    e2m1_value = torch.full_like(qx, 0x7, dtype=torch.uint8)
    e2m1_value = torch.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = torch.where(denormal_mask, denormal_x, e2m1_value)

    # add sign back
    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(torch.uint8)
    # Right shift of a negative signed integer can fill the least significant
    # bits with either 1s or 0s, depending on the implementation. Since PyTorch
    # doesn't have an uint32 dtype, we mask out these bits to get just the
    # f4 sign bit
    sign_lp = sign_lp & sign_mask
    e2m1_value = e2m1_value | sign_lp

    # Pack 2 4-bit values into 8-bit
    x_mxfp4 = e2m1_value[..., ::2] | (e2m1_value[..., 1::2] << 4)

    # Recover last dimension's shape
    x_mxfp4 = torch.flatten(x_mxfp4, -2, -1)

    # Remove padded values
    if x.shape[-1] % QUANT_BLOCK_SIZE != 0:
        x_mxfp4 = x_mxfp4[..., : x.shape[-1] // 2]

    # Reshape back to original
    mxfp4_shape = list(x_shape)
    mxfp4_shape = tuple(mxfp4_shape[:-1] + [mxfp4_shape[-1] // 2])
    x_mxfp4 = x_mxfp4.reshape(mxfp4_shape)

    return x_mxfp4, block_scales


def torch_dequant_nvfp4(
    x: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Tutorial-style dequant: unpack OCP e2m1 nibbles, decode with OCP rules, multiply by
    per-block float8 scale broadcast (same construction as `random_nvfp4_tensor` reference).
    """
    NVFP4_QUANT_BLOCK_SIZE = 16
    assert x.dtype == torch.uint8
    assert scale.dtype == torch.float8_e4m3fn
    assert (
        scale.shape[-1] == x.shape[-1] * 2 // NVFP4_QUANT_BLOCK_SIZE
    ), f"Expected scale last dim {x.shape[-1]*2 // NVFP4_QUANT_BLOCK_SIZE}, got {scale.shape[-1]}"
    # high = (x >> 4) & 0xF
    # low = x & 0xF
    # raw = torch.stack((low, high), dim=-1).reshape(*p.shape[:-1], p.shape[-1] * 2)
    # ref = _ocp_e2m1_to_f32(raw.to(torch.uint8))
    # ref = mxfp4_to_f32(_pack_e2m1_along_dim(raw, dim=1))
    ref = mxfp4_to_f32(x)
    sc = (
        scale.to(torch.float32)
        .unsqueeze(-1)
        .expand(*scale.shape, NVFP4_QUANT_BLOCK_SIZE)
        .reshape(*scale.shape[:-1], x.shape[-1] * 2)
    )
    out = ref * sc
    return out.contiguous().to(out_dtype)


@pytest.mark.parametrize(
    "M, N",
    [
        (1, 4),
        (1, 28),
        (1, 32),
        (1, 64),
        (1, 68),
        (128, 4),
        (128, 28),
        (128, 32),
        (128, 64),
        (128, 68),
        (256, 32),
        (160, 40),
        (280, 20),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_dynamic_mxfp4_quant(M: int, N: int, dtype):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests
    torch.manual_seed(20)
    x = torch.randn((M, N), dtype=dtype, device="cuda")

    logger.debug("x.shape=%s x=%s", x.shape, x)

    triton_out, triton_scale = dynamic_mxfp4_quant(x)
    logger.debug("triton_out.shape=%s triton_out=%s", triton_out.shape, triton_out)
    logger.debug(
        "triton_scale.shape=%s triton_scale=%s", triton_scale.shape, triton_scale
    )

    torch_out, torch_scale = torch_dynamic_mxfp4_quant(x)
    logger.debug("torch_out.shape=%s torch_out=%s", torch_out.shape, torch_out)
    logger.debug("torch_scale.shape=%s torch_scale=%s", torch_scale.shape, torch_scale)

    torch.testing.assert_close(triton_scale, torch_scale)
    torch.testing.assert_close(triton_out, torch_out)


@pytest.mark.parametrize(
    "M, N",
    [
        (1, 4),
        (1, 28),
        (1, 32),
        (1, 64),
        (1, 68),
        (128, 4),
        (128, 28),
        (128, 32),
        (128, 64),
        (128, 68),
        (256, 32),
        (160, 40),
        (280, 20),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fp4_utils_dynamic_mxfp4_quant(M: int, N: int, dtype):
    torch.cuda.empty_cache()
    torch.manual_seed(20)
    x = torch.randn((M, N), dtype=dtype, device="cuda")

    logger.debug("x.shape=%s x=%s", x.shape, x)

    fp4_utils_out, fp4_utils_scale = fp4_utils_dynamic_mxfp4_quant(x)
    logger.debug(
        "fp4_utils_out.shape=%s fp4_utils_out=%s", fp4_utils_out.shape, fp4_utils_out
    )
    logger.debug(
        "fp4_utils_scale.shape=%s fp4_utils_scale=%s",
        fp4_utils_scale.shape,
        fp4_utils_scale,
    )

    torch_out, torch_scale = torch_dynamic_mxfp4_quant(x)
    logger.debug("torch_out.shape=%s torch_out=%s", torch_out.shape, torch_out)
    logger.debug("torch_scale.shape=%s torch_scale=%s", torch_scale.shape, torch_scale)

    torch.testing.assert_close(
        fp4_utils_scale.view(torch.uint8).cpu(), torch_scale.cpu()
    )
    torch.testing.assert_close(fp4_utils_out.view(torch.uint8).cpu(), torch_out.cpu())


@pytest.mark.parametrize("M", [1, 4, 16, 32, 64, 128])
@pytest.mark.parametrize("N", [16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_nvfp4_quant(
    M: int,
    N: int,
    dtype: torch.dtype,
):
    torch.cuda.empty_cache()
    if DEVICE_ARCH not in ("gfx1250",):
        pytest.skip("NVFP4 quantization is only supported on GFX1250")

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_og = torch.randn(M, N, device=device, dtype=dtype) / 20
    x_q_triton, x_s_triton = dynamic_nvfp4_quant(x_og)
    x_q_torch, x_s_torch = torch_dynamic_mxfp4_quant(x_og, is_nvfp4=True)

    x_dq_triton = torch_dequant_nvfp4(x_q_triton, x_s_triton, out_dtype=dtype)
    x_dq_torch = torch_dequant_nvfp4(x_q_torch, x_s_torch, out_dtype=dtype)

    atol = None
    rtol = None
    if dtype == torch.bfloat16:
        atol = 1.5e-2
        rtol = 1.5e-2
    torch.testing.assert_close(x_dq_triton, x_dq_torch, atol=atol, rtol=rtol)


@_REQUIRES_GFX950
def test_dynamic_mxfp4_quant_sr_validates_contract():
    with pytest.raises(ValueError, match="2-D"):
        dynamic_mxfp4_quant(
            torch.zeros(32, dtype=torch.bfloat16, device="cuda"),
            use_sr=True,
            philox_seed=1,
        )

    x = torch.zeros((3, 32), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(TypeError, match="bfloat16 or float32"):
        dynamic_mxfp4_quant(x.to(torch.float16), use_sr=True, philox_seed=1)
    with pytest.raises(ValueError, match="scaling_mode='even'"):
        dynamic_mxfp4_quant(
            x,
            scaling_mode="ceil",
            use_sr=True,
            philox_seed=1,
        )
    with pytest.raises(ValueError, match="divisible by 32"):
        dynamic_mxfp4_quant(x[:, :6], use_sr=True, philox_seed=1)
    with pytest.raises(ValueError, match="philox_seed is required"):
        dynamic_mxfp4_quant(x, use_sr=True)
    with pytest.raises(ValueError, match="philox_seed must be"):
        dynamic_mxfp4_quant(x, use_sr=True, philox_seed=-1)

    counters_used = x.numel() // 8
    max_valid_offset = (1 << 63) - counters_used
    dynamic_mxfp4_quant(
        x,
        use_sr=True,
        philox_seed=1,
        philox_offset=max_valid_offset,
    )
    with pytest.raises(ValueError, match="leave room"):
        dynamic_mxfp4_quant(
            x,
            use_sr=True,
            philox_seed=1,
            philox_offset=max_valid_offset + 1,
        )
    with pytest.raises(ValueError, match="only valid when use_sr=True"):
        dynamic_mxfp4_quant(x, philox_seed=1)


@_REQUIRES_GFX950
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 32), (6, 96), (64, 256)])
def test_dynamic_mxfp4_quant_sr_is_reproducible_and_reuses_rtn_scales(
    shape,
    dtype,
):
    torch.manual_seed(17)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    kwargs = {"use_sr": True, "philox_seed": 1234, "philox_offset": 5678}

    packed, scales = dynamic_mxfp4_quant(x, **kwargs)
    packed_repeat, scales_repeat = dynamic_mxfp4_quant(x, **kwargs)
    packed_next, scales_next = dynamic_mxfp4_quant(
        x,
        use_sr=True,
        philox_seed=1234,
        philox_offset=5679,
    )
    _, scales_rtn = dynamic_mxfp4_quant(x)

    assert packed.shape == (shape[0], shape[1] // 2)
    assert scales.shape == (shape[0], shape[1] // 32)
    assert packed.dtype == torch.uint8 and scales.dtype == torch.uint8
    assert scales.stride() == scales_rtn.stride()
    torch.testing.assert_close(packed, packed_repeat, atol=0, rtol=0)
    torch.testing.assert_close(scales, scales_repeat, atol=0, rtol=0)
    torch.testing.assert_close(scales, scales_next, atol=0, rtol=0)
    torch.testing.assert_close(scales, scales_rtn, atol=0, rtol=0)
    assert not torch.equal(packed, packed_next)


@_REQUIRES_GFX950
def test_dynamic_mxfp4_quant_sr_bf16_exact_values_match_known_encoding():
    values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
            0.0,
        ],
        dtype=torch.bfloat16,
        device="cuda",
    ).repeat(2)[None, :]
    expected_packed = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0xA9, 0xCB, 0xED, 0x0F] * 2,
        dtype=torch.uint8,
        device="cuda",
    )[None, :]

    packed, scales = dynamic_mxfp4_quant(
        values,
        use_sr=True,
        philox_seed=1234,
        philox_offset=5678,
    )

    torch.testing.assert_close(packed, expected_packed, atol=0, rtol=0)
    assert torch.all(scales == 127)
    torch.testing.assert_close(
        _dequantize_mxfp4(packed, scales), values.float(), atol=0, rtol=0
    )


@_REQUIRES_GFX950
def test_dynamic_mxfp4_quant_sr_accepts_noncontiguous_input():
    torch.manual_seed(19)
    x = torch.randn((96, 6), dtype=torch.float32, device="cuda").T
    assert not x.is_contiguous()

    actual = dynamic_mxfp4_quant(
        x,
        use_sr=True,
        philox_seed=7,
        philox_offset=11,
    )
    expected = dynamic_mxfp4_quant(
        x.contiguous(),
        use_sr=True,
        philox_seed=7,
        philox_offset=11,
    )

    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)


@_REQUIRES_GFX950
def test_dynamic_mxfp4_quant_sr_preserves_raw_zero_scale_endpoint():
    # Raw E8M0 zero represents 2^-127, not the 2^-126 minimum normal value.
    x = torch.full((32, 32), 2.0**-125, dtype=torch.float32, device="cuda")
    packed, scales = dynamic_mxfp4_quant(
        x,
        use_sr=True,
        philox_seed=1,
    )

    assert torch.count_nonzero(scales).item() == 0
    assert torch.all(packed == 0x66)
    torch.testing.assert_close(_dequantize_mxfp4(packed, scales), x, atol=0, rtol=0)


@_REQUIRES_GFX950
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dynamic_mxfp4_quant_sr_never_emits_e8m0_nan(dtype):
    x = torch.full(
        (2, 32),
        torch.finfo(dtype).max,
        dtype=dtype,
        device="cuda",
    )
    _, scales = dynamic_mxfp4_quant(x, use_sr=True, philox_seed=1)

    assert torch.all(scales == 254)


@_REQUIRES_GFX950
def test_dynamic_mxfp4_quant_sr_uses_distinct_global_counters():
    torch.manual_seed(23)
    tile = torch.randn((32, 128), dtype=torch.bfloat16, device="cuda")
    x = tile.repeat(2, 16)

    packed, scales = dynamic_mxfp4_quant(
        x,
        use_sr=True,
        philox_seed=1234,
    )
    packed_high_offset, scales_high_offset = dynamic_mxfp4_quant(
        x,
        use_sr=True,
        philox_seed=1234,
        philox_offset=(1 << 32),
    )

    torch.testing.assert_close(scales[:, :4], scales[:, 4:8], atol=0, rtol=0)
    assert not torch.equal(packed[:, :64], packed[:, 64:128])
    torch.testing.assert_close(scales[:32], scales[32:], atol=0, rtol=0)
    assert not torch.equal(packed[:32], packed[32:])
    torch.testing.assert_close(scales, scales_high_offset, atol=0, rtol=0)
    assert not torch.equal(packed, packed_high_offset)


@_REQUIRES_GFX950
def test_dynamic_mxfp4_quant_sr_rounds_midpoints_without_bias():
    torch.manual_seed(29)
    x = torch.full((256, 256), 1.25, dtype=torch.float32, device="cuda")
    x[:, 0::32] = 4.0
    x[:, 1::32] = 6.0

    packed, scales = dynamic_mxfp4_quant(
        x,
        use_sr=True,
        philox_seed=1234,
    )
    dequantized = _dequantize_mxfp4(packed, scales)
    midpoint_mask = torch.ones_like(x, dtype=torch.bool)
    midpoint_mask[:, 0::32] = False
    midpoint_mask[:, 1::32] = False
    midpoint_values = dequantized[midpoint_mask]
    round_up_fraction = (midpoint_values == 1.5).float().mean()

    assert packed[0, 0].item() == 0x76
    assert torch.all(scales == 127)
    assert torch.all((midpoint_values == 1.0) | (midpoint_values == 1.5))
    assert abs(round_up_fraction.item() - 0.5) < 0.02


@triton.jit
def _mxfp4_quant_op_kernel(
    x_ptr,
    fp4_ptr,
    scale_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    SCALING_MODE: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    rows = tl.program_id(0) * M + tl.arange(0, M)
    x = tl.load(x_ptr + rows[:, None] * N + tl.arange(0, N)[None, :])
    x_fp4, scales = _mxfp4_quant_op(x, N, M, 32, SCALING_MODE, USE_ASM)
    fp4_cols = tl.arange(0, N // 2)
    tl.store(fp4_ptr + rows[:, None] * (N // 2) + fp4_cols[None, :], x_fp4)
    scale_cols = tl.arange(0, N // 32)
    tl.store(scale_ptr + rows[:, None] * (N // 32) + scale_cols[None, :], scales)


@pytest.mark.parametrize("scaling_mode", [0, 1])
def test_mxfp4_quant_op_asm_matches_bits(scaling_mode):
    torch.manual_seed(scaling_mode)
    m, n = 1024, 128
    x = torch.randn(m, n, device="cuda") * torch.exp2(
        torch.randint(-140, 120, (m, 1), device="cuda").float()
    )
    # Rounding ties, -0.0 and an all-zero block.
    x[0, :8] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.0])
    x[1, :32] = 0.0
    outs = []
    for use_asm in (True, False):
        x_fp4 = torch.empty(m, n // 2, dtype=torch.uint8, device="cuda")
        scales = torch.empty(m, n // 32, dtype=torch.uint8, device="cuda")
        _mxfp4_quant_op_kernel[(m // 16,)](
            x, x_fp4, scales, M=16, N=n, SCALING_MODE=scaling_mode, USE_ASM=use_asm
        )
        outs.append((x_fp4, scales))
    torch.testing.assert_close(outs[0][0], outs[1][0], atol=0, rtol=0)
    torch.testing.assert_close(outs[0][1], outs[1][1], atol=0, rtol=0)
    if scaling_mode == 1:
        amax = x.reshape(m, -1, 32).abs().amax(-1).double().clamp(min=6 * 2**-126)
        ref = torch.ceil(torch.log2(amax / 6)).to(torch.int32) + 127
        torch.testing.assert_close(outs[0][1].int(), ref, atol=0, rtol=0)
