# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
import triton

from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.quant import dynamic_nvfp4_quant
from aiter.utility.fp4_utils import (
    dynamic_mxfp4_quant as fp4_utils_dynamic_mxfp4_quant,
)
from aiter.utility.fp4_utils import mxfp4_to_f32
from aiter.ops.triton.utils.types import e4m3_dtype
import aiter.ops.triton.utils._triton.arch_info as arch_info

DEVICE_ARCH = arch_info.get_arch()

DEBUG_MODE = False


def batched_swizzle_scales_gfx1250(data, scale_k_width=None):
    """
    Batched swizzle scales for FP4 blockscale16 format.
       https://github.com/triton-lang/triton/blob/main/third_party/amd/python/examples/gluon/mxfp_gemm_gfx1250.py#L1014
    """
    data_shape = data.shape
    N = data_shape[-2]
    SCALE_K = data_shape[-1]
    PRESHUFFLE_FACTOR = 128
    if scale_k_width is None:
        SCALE_KWIDTH = (
            min(16, triton.next_power_of_2(SCALE_K)) if SCALE_K >= 4 else SCALE_K
        )
    else:
        assert scale_k_width in [4, 8, 16]
        SCALE_KWIDTH = scale_k_width if SCALE_K >= 4 else SCALE_K
    data = data.view(
        -1,
        N // PRESHUFFLE_FACTOR,
        4,
        PRESHUFFLE_FACTOR // 4,
        SCALE_K // SCALE_KWIDTH,
        SCALE_KWIDTH,
    )
    data = data.permute(0, 1, 4, 3, 2, 5).contiguous()
    data = data.view(
        *data_shape[:-2], N // PRESHUFFLE_FACTOR, SCALE_K * PRESHUFFLE_FACTOR
    )
    return data


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
        (2, 4),
        (2, 28),
        (2, 32),
        (2, 64),
        (2, 68),
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

    if DEBUG_MODE:
        print(f"x.shape={x.shape} x={x}")

    triton_out, triton_scale = dynamic_mxfp4_quant(x)
    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape} triton_out={triton_out}")
        print(f"triton_scale.shape={triton_scale.shape} triton_scale={triton_scale}")

    torch_out, torch_scale = torch_dynamic_mxfp4_quant(x)
    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape} torch_out={torch_out}")
        print(f"torch_scale.shape={torch_scale.shape} torch_scale={torch_scale}")

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
        (2, 4),
        (2, 28),
        (2, 32),
        (2, 64),
        (2, 68),
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

    if DEBUG_MODE:
        print(f"x.shape={x.shape} x={x}")

    fp4_utils_out, fp4_utils_scale = fp4_utils_dynamic_mxfp4_quant(x)
    if DEBUG_MODE:
        print(
            f"fp4_utils_out.shape={fp4_utils_out.shape} fp4_utils_out={fp4_utils_out}"
        )
        print(
            f"fp4_utils_scale.shape={fp4_utils_scale.shape} fp4_utils_scale={fp4_utils_scale}"
        )

    torch_out, torch_scale = torch_dynamic_mxfp4_quant(x)
    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape} torch_out={torch_out}")
        print(f"torch_scale.shape={torch_scale.shape} torch_scale={torch_scale}")

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
