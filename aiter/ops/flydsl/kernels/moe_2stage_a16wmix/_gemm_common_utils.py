# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import torch
import triton
import triton.language as tl
from torch import Tensor

_8bit_fallback = torch.uint8
fp8_e8m0 = getattr(torch, "float8_e8m0fnu", _8bit_fallback)
fp4x2 = getattr(torch, "float4_e2m1fn_x2", _8bit_fallback)
fp32 = torch.float32


def f32_to_mxfp4(x):
    FP4_EBITS, FP4_MBITS = 2, 1
    x = _f32_to_floatx_unpacked(x.float(), FP4_EBITS, FP4_MBITS)
    x = pack_uint4(x)
    x = x.view(fp4x2)  # to(fp32) for this datatype gives all 0 for torch...
    # x = x.view(torch.uint8)
    return x


def mxfp4_to_f32(x):
    if x.dtype == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)

    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device=x.device)
    return mxfp4_in_f32[x.long()]


def f32_to_e8m0(x):
    u32 = x.view(torch.int32)
    exponent = ((u32 >> 23) & 0xFF).view(torch.uint32).to(torch.uint8)
    nan_case = exponent == 0xFF
    round_case = ((u32 & 0x400000) > 0) & (
        ((u32 & 0x200000) > 0) | ((u32 & 0x1FFFFF) > 0) | (exponent > 0)
    )
    exponent[round_case] += 1
    exponent[nan_case] = 0xFF
    return exponent.view(fp8_e8m0)


def e8m0_to_f32(scale_e8m0_biased):
    scale_e8m0_biased = scale_e8m0_biased.view(torch.uint8)
    zero_case = scale_e8m0_biased == 0
    nan_case = scale_e8m0_biased == 0xFF
    scale_f32 = scale_e8m0_biased.to(torch.int32) << 23
    scale_f32[zero_case] = 0x00400000
    scale_f32[nan_case] = 0x7F800001
    scale_f32 = scale_f32.view(fp32)
    return scale_f32


def random_e8m0(
    rows: int, cols: int, *, low_exp=127, high_exp=132, device="cpu"
) -> torch.Tensor:
    """Generate random E8M0 scale bytes [rows, cols] uint8."""
    return torch.randint(
        low_exp, high_exp + 1, (rows, cols), dtype=torch.uint8, device=device
    )


def random_fp4_packed(rows: int, cols: int, *, device="cpu") -> torch.Tensor:
    """Generate random packed FP4 data [rows, cols//2] uint8."""
    assert cols % 2 == 0
    unpacked = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device=device)
    return pack_uint4(unpacked)


def fp8_e4m3_to_f32(x):
    """Convert FP8/E4M3 (OCP E4M3, no infinity) uint8 tensor to float32.

    E4M3 layout: [sign:1][exp:4][mantissa:3], bias=7.
    Special: exp=0b1111 + mantissa=0b111 → NaN.
    """
    x = x.view(torch.uint8).to(torch.int32)
    sign = (x >> 7) & 1
    exp = (x >> 3) & 0xF
    mant = x & 0x7

    # Normal: value = (-1)^sign * 2^(exp - 7) * (1 + mant/8)
    # Denormal (exp==0): value = (-1)^sign * 2^(-6) * (mant/8)
    # NaN: exp==15 and mant==7
    is_nan = (exp == 15) & (mant == 7)
    is_denorm = exp == 0

    # Normal path
    f_normal = (1.0 + mant.float() / 8.0) * torch.pow(2.0, (exp.float() - 7.0))
    # Denormal path
    f_denorm = (mant.float() / 8.0) * (2.0**-6)

    result = torch.where(is_denorm, f_denorm, f_normal)
    result = torch.where(is_nan, torch.tensor(float("nan")), result)
    result = torch.where(sign == 1, -result, result)
    return result


def e8m0_shuffle(scale):
    if scale is None:
        return scale
    if scale.dtype == torch.float32:
        return scale
    assert scale.ndim == 2, "scale must be a 2D tensor"
    m, n = scale.shape
    scale_padded = torch.empty(
        (m + 255) // 256 * 256,
        (n + 7) // 8 * 8,
        dtype=scale.dtype,
        device=scale.device,
    )

    scale_padded[:m, :n] = scale
    scale = scale_padded
    sm, sn = scale.shape
    scale = scale.view(sm // 32, 2, 16, sn // 8, 2, 4)
    scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
    scale = scale.view(sm, sn)
    return scale


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def pack_uint4(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))


# copy-pasted from
# https://github.com/pytorch/ao/blob/bc4f51da86956275da7db0da6e420c506df97820/torchao/prototype/custom_fp_utils.py#L27C1-L142C29
def _n_ones(n: int) -> int:
    return (1 << n) - 1


EBITS_F32, MBITS_F32 = 8, 23
F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)


# copy-pasted from
# https://github.com/pytorch/ao/blob/bc4f51da86956275da7db0da6e420c506df97820/torchao/prototype/custom_fp_utils.py#L27C1-L142C29
def _f32_to_floatx_unpacked(x: Tensor, ebits: int, mbits: int) -> Tensor:
    """Convert FP32 numbers to sub-byte floating point numbers with the given
    number of exponent and mantissa bits.

    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, where the bit encoding is stored
    in the least significant bits. e.g.
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in fp6_e2m3 or fp6_e3m2 encoding

    Note: there are no special values (NaN, inf) support in this code. Values
    outside the representable range of Floatx after rounding are clamped to the
    maximum Floatx magnitude (sign is preserved).

    Code below is an adaptation of https://fburl.com/code/ciwofcg4

    Background 1: last answer in https://stackoverflow.com/q/8981913
    Background 2: Computer Organization and Design, RISC-V edition, Chapter 3.5
    """
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    # calculate constants
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)

    # TODO document this better
    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    # all E bits and M bits are 1s
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))

    # E bits = 1, M bits = 0
    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (
        # exp bias conversion between formats
        (F32_EXP_BIAS - exp_bias)
        # mantissa length difference between formats
        + (MBITS_F32 - mbits)
        # add one to encoded exponent for denormalized numbers
        + 1
    )
    denorm_mask_int = denorm_exp << MBITS_F32

    # reinterpret int32 as float32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(
        torch.float32
    )

    # save the sign
    # Note that we have torch.uint32, but some ops like cpu bit shifts
    # do not work on it. So, we stay in int32.
    x = x.view(torch.int32)
    sign = x & 0x80000000

    # set everything to positive, will add sign back at the end
    x = x ^ sign

    # TODO: can the branch floating point comparisons below be done without
    # converting to float? probably but need to verify
    x = x.view(torch.float)

    # rewrite saturate/denorm/norm branches without explicit data dependent
    # control flow, to be more compiler friendly
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    #
    # branch 1: saturate to max val - handled later in the code which combines
    #   the branches
    #

    #
    # branch 2: to conversion to denormal as well as rounding up to normal
    #
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    #
    # branch 3: stay in normal range, adjust the exponent and round
    #
    normal_x = x.view(torch.int32)
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    #
    # combine the branches
    #
    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    # add sign back
    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    # Right shift of a negative signed integer can fill the least significant
    # bits with either 1s or 0s, depending on the implementation. Since PyTorch
    # doesn't have an uint32 dtype, we mask out these bits to get just the
    # f4 sign bit
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


@triton.jit
def _dynamic_mxfp4_quant_kernel_asm_layout(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m,
    stride_x_n,
    stride_x_fp4_m,
    stride_x_fp4_n,
    stride_bs_m,
    stride_bs_n,
    M: tl.constexpr,
    N: tl.constexpr,
    scaleN: tl.constexpr,
    scaleM_pad: tl.constexpr,
    scaleN_pad: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    SCALING_MODE: tl.constexpr,
    SHUFFLE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    stride_x_m = tl.cast(stride_x_m, tl.int64)
    stride_x_n = tl.cast(stride_x_n, tl.int64)
    stride_x_fp4_m = tl.cast(stride_x_fp4_m, tl.int64)
    stride_x_fp4_n = tl.cast(stride_x_fp4_n, tl.int64)

    x_offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_offs_n = pid_n * MXFP4_QUANT_BLOCK_SIZE + tl.arange(0, MXFP4_QUANT_BLOCK_SIZE)
    x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n
    x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
    x = tl.load(x_ptr + x_offs, mask=x_mask).to(tl.float32)

    # Calculate scale
    amax = tl.max(tl.abs(x), axis=1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    quant_scale = tl.exp2(-scale_e8m0_unbiased)

    # Compute quantized x
    qx = x * quant_scale

    # blockscale_e8m0
    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127

    # Convert quantized fp32 tensor to uint32 before converting to mxfp4 format
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
    qx = qx.to(tl.uint32, bitcast=True)

    # Extract sign, exponents and mantissa fields from FP32
    s = qx & 0x80000000
    e = (qx >> 23) & 0xFF
    m = qx & 0x7FFFFF

    E8_BIAS: tl.constexpr = 127
    E2_BIAS: tl.constexpr = 1

    # Denormal numbers
    # If exponent is less than 127, then it's a denormal number
    # See above, for denormal number mantissa is always 1 and we set bit 1 of mantissa
    adjusted_exponents = tl.core.sub(E8_BIAS, e + 1, sanitize_overflow=False)
    m = tl.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)

    # For normal numbers, bias is changed from 127 to 1, and for subnormals, we keep exponent as 0.
    # Note: E8_BIAS - E2_BIAS = 126, so for normals we subtract that.
    e = tl.maximum(e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

    # Combine sign, exponent, and mantissa, while saturating
    # rounding nearest with tie breaking up by adding +1 to one bit right of the LSB, then shift right
    e2m1_tmp = tl.minimum((((e << 2) | (m >> 21)) + 1) >> 1, 0x7)
    e2m1_value = ((s >> 28) | e2m1_tmp).to(tl.uint8)

    e2m1_value = tl.reshape(e2m1_value, [BLOCK_SIZE, MXFP4_QUANT_BLOCK_SIZE // 2, 2])
    evens, odds = tl.split(e2m1_value)
    out_tensor = evens | (odds << 4)

    out_offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_offs_n = pid_n * MXFP4_QUANT_BLOCK_SIZE // 2 + tl.arange(
        0, MXFP4_QUANT_BLOCK_SIZE // 2
    )
    out_offs = (
        out_offs_m[:, None] * stride_x_fp4_m + out_offs_n[None, :] * stride_x_fp4_n
    )
    out_mask = (out_offs_m < M)[:, None] & (out_offs_n < (N // 2))[None, :]
    tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask)

    bs_offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    bs_offs_n = pid_n

    if SHUFFLE:
        bs_offs_0 = bs_offs_m[:, None] // 32
        bs_offs_1 = bs_offs_m[:, None] % 32
        bs_offs_2 = bs_offs_1 % 16
        bs_offs_1 = bs_offs_1 // 16
        bs_offs_3 = bs_offs_n[None, :] // 8
        bs_offs_4 = bs_offs_n[None, :] % 8
        bs_offs_5 = bs_offs_4 % 4
        bs_offs_4 = bs_offs_4 // 4
        bs_offs = (
            bs_offs_1
            + bs_offs_4 * 2
            + bs_offs_2 * 2 * 2
            + bs_offs_5 * 2 * 2 * 16
            + bs_offs_3 * 2 * 2 * 16 * 4
            + bs_offs_0 * 2 * 16 * scaleN
        )
        bs_mask1 = (bs_offs_m < M)[:, None] & (bs_offs_n < scaleN)[None, :]
        bs_mask2 = (bs_offs_m < scaleM_pad)[:, None] & (bs_offs_n < scaleN_pad)[None, :]
        bs_e8m0 = tl.where(bs_mask1, bs_e8m0, 127)
        tl.store(bs_ptr + bs_offs, bs_e8m0, mask=bs_mask2)
    else:
        bs_offs = bs_offs_m[:, None] * stride_bs_m + bs_offs_n[None, :] * stride_bs_n
        bs_mask = (bs_offs_m < M)[:, None] & (bs_offs_n < N)[None, :]
        tl.store(bs_ptr + bs_offs, bs_e8m0, mask=bs_mask)


def dynamic_mxfp4_quant(
    x: torch.Tensor, scaling_mode: str = "even", shuffle: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to MX FP4 format.

    Args:
        x: The input tensor, typically fp16 or bf16.
        scaling_mode: The method to calculate MX block scaling.
            - "even" (default): `even_round` in `quark.torch.quantization.utils`.
            - etc.
    Returns:
        A tuple of (x_fp4, blockscale_e8m0).
    """
    # Assume x is 2D-Tensor for now
    M, N = x.shape

    assert (N // 2) % 2 == 0

    # This is fixed by spec for MXFP4. Do not tune this.
    # For performance, perhaps, we should look at passing multiple of 32 column blocks
    # that a triton program can process
    MXFP4_QUANT_BLOCK_SIZE = 32

    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    scaleM = triton.cdiv(M, 32) * 32
    scaleN_valid = triton.cdiv(N, MXFP4_QUANT_BLOCK_SIZE)
    scaleN = triton.cdiv(scaleN_valid, 8) * 8
    blockscale_e8m0 = torch.empty(
        (
            triton.cdiv(M, 256) * 256,
            scaleN,
        ),
        dtype=torch.uint8,
        device=x.device,
    )

    BLOCK_SIZE = 128
    grid = (triton.cdiv(M, BLOCK_SIZE), scaleN)
    _dynamic_mxfp4_quant_kernel_asm_layout[grid](
        x,
        x_fp4,
        blockscale_e8m0,
        *x.stride(),
        *x_fp4.stride(),
        *blockscale_e8m0.stride(),
        M=M,
        N=N,
        scaleN=scaleN_valid,
        scaleM_pad=scaleM,
        scaleN_pad=scaleN,
        BLOCK_SIZE=BLOCK_SIZE,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        SCALING_MODE=0,
        SHUFFLE=shuffle,
    )

    return (x_fp4.view(fp4x2), blockscale_e8m0.view(fp8_e8m0))


@triton.jit
def _moe_mxfp4_sort_kernel(
    blockscale_e8m0_ptr,
    sorted_ids_ptr,
    num_valid_ids_ptr,
    blockscale_e8m0_sorted_ptr,
    stride_blockscale_e8m0_m: tl.constexpr,
    stride_blockscale_e8m0_n: tl.constexpr,
    stride_o3: tl.constexpr,
    stride_o2: tl.constexpr,
    stride_o1: tl.constexpr,
    stride_o0: tl.constexpr,
    token_num: tl.constexpr,
    M_i: tl.constexpr,
    N_i: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TOPK: tl.constexpr,
):
    pid_m = tl.program_id(0) * 2
    pid_n = tl.program_id(1) * 2
    num_valid_ids = tl.load(num_valid_ids_ptr)
    if pid_m * BLOCK_SIZE_M >= num_valid_ids:
        return
    stride_blockscale_e8m0_m = tl.cast(stride_blockscale_e8m0_m, tl.int64)
    stride_blockscale_e8m0_n = tl.cast(stride_blockscale_e8m0_n, tl.int64)
    stride_o0 = tl.cast(stride_o0, tl.int64)
    stride_o1 = tl.cast(stride_o1, tl.int64)
    stride_o2 = tl.cast(stride_o2, tl.int64)
    stride_o3 = tl.cast(stride_o3, tl.int64)

    out = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.uint32)
    for i in range(4):
        m = i % 2 * BLOCK_SIZE_M
        n = i // 2 * BLOCK_SIZE_N
        sorted_ids_offs_m = pid_m * BLOCK_SIZE_M + m + tl.arange(0, BLOCK_SIZE_M)
        sorted_ids_offs = sorted_ids_offs_m
        sorted_ids_mask = sorted_ids_offs_m < num_valid_ids
        sorted_ids = tl.load(
            sorted_ids_ptr + sorted_ids_offs, mask=sorted_ids_mask, other=token_num
        )
        topk_ids = sorted_ids >> 24
        sorted_ids = sorted_ids & 0xFFFFFF

        # Sort the blockscale tensor based on the sorted ids
        if TOPK == 1:
            blockscale_e8m0_offs_m = sorted_ids
        else:
            blockscale_e8m0_offs_m = sorted_ids * TOPK + topk_ids
        blockscale_e8m0_offs_n = pid_n * BLOCK_SIZE_N + n + tl.arange(0, BLOCK_SIZE_N)
        blockscale_e8m0_offs = (
            blockscale_e8m0_offs_m[:, None] * stride_blockscale_e8m0_m
            + blockscale_e8m0_offs_n[None, :] * stride_blockscale_e8m0_n
        )
        blockscale_e8m0_mask = (sorted_ids < token_num)[:, None] & (
            blockscale_e8m0_offs_n < N_i
        )[None, :]
        blockscale_e8m0_sub = tl.load(
            blockscale_e8m0_ptr + blockscale_e8m0_offs,
            mask=blockscale_e8m0_mask,
        ).to(tl.uint8, bitcast=True)
        out = out | (blockscale_e8m0_sub.to(tl.uint32) << (i * 8))

    # Store the result
    # 16x4 uint32 -> 32x2 uint8
    offs_0 = tl.arange(0, BLOCK_SIZE_M)
    offs_1 = tl.arange(0, BLOCK_SIZE_N)
    offs_2 = pid_n // 2
    offs_3 = pid_m // 2
    offs = (
        offs_0[:, None] * stride_o0
        + offs_1[None, :] * stride_o1  # * BLOCK_SIZE_M
        + offs_2 * stride_o2  # * BLOCK_SIZE_M * BLOCK_SIZE_N
        + offs_3 * stride_o3  # * BLOCK_SIZE_M * BLOCK_SIZE_N * N_i // BLOCK_SIZE_N
    )
    # blockscale_e8m0_sorted_mask = (blockscale_e8m0_sorted_offs_m < M_o)[:, None] & (
    #     blockscale_e8m0_sorted_offs_n < N_o
    # )[None, :]
    tl.store(
        blockscale_e8m0_sorted_ptr + offs,
        out,
        # mask=blockscale_e8m0_sorted_mask,
    )


def moe_mxfp4_sort(
    blockscale_e8m0: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    block_size: int = 32,
) -> torch.Tensor:
    """
    Sort the blockscale_e8m0 tensor based on the sorted_ids tensor.

    Args:
        blockscale_e8m0: The input tensor to be sorted.
        sorted_ids: The indices used for sorting.

    Returns:
        A sorted tensor.
    """
    # This is fixed by spec for MXFP4. Do not tune this.
    BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 8
    BLOCK_SIZE_M_u32, BLOCK_SIZE_N_u32 = 16, 4

    # Assume blockscale_e8m0 is 2D-Tensor for now
    topk = 1
    if len(blockscale_e8m0.shape) == 3:
        topk = blockscale_e8m0.shape[1]
        blockscale_e8m0 = blockscale_e8m0.view(-1, blockscale_e8m0.shape[-1])
    M_i, N_i = blockscale_e8m0.shape
    M_o, N_o = sorted_ids.shape[0], N_i
    assert (N_i // 2) % 2 == 0
    assert block_size % BLOCK_SIZE_M == 0

    blockscale_e8m0_sorted = torch.empty(
        (
            triton.cdiv(M_o, BLOCK_SIZE_M),
            triton.cdiv(N_o, BLOCK_SIZE_N),
            BLOCK_SIZE_N_u32,
            BLOCK_SIZE_M_u32,
        ),
        dtype=torch.uint32,
        device=blockscale_e8m0.device,
    )  # .fill_(0)

    grid = (triton.cdiv(M_o, BLOCK_SIZE_M), triton.cdiv(N_i, BLOCK_SIZE_N))
    _moe_mxfp4_sort_kernel[grid](
        blockscale_e8m0.view(torch.uint8),
        sorted_ids,
        num_valid_ids,
        blockscale_e8m0_sorted,
        *blockscale_e8m0.stride(),
        *blockscale_e8m0_sorted.stride(),
        token_num=token_num,
        M_i=M_i,
        N_i=N_i,
        BLOCK_SIZE_M=BLOCK_SIZE_M // 2,
        BLOCK_SIZE_N=BLOCK_SIZE_N // 2,
        TOPK=topk,
    )

    # Reshape the output to the final shape
    return blockscale_e8m0_sorted.view(fp8_e8m0).view(-1, N_o)


def shuffle_weight_w4(
    src: torch.Tensor, NLane: int, gate_up: bool, moe_gemm: bool
) -> torch.Tensor:
    """
    src: shape [experts_cnt, N, K_pk], where K_pk = K // 2
    Returns: shuffled tensor of shape [experts_cnt, N0*2, K0, KLane, NLane, KPack]
    """
    # print("gemm shape:", src.shape)
    src_type = src.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and src_type == torch.float4_e2m1fn_x2:
        src = src.view(torch.uint8)
    if moe_gemm:
        experts_cnt, N, K_pk = src.shape
        if gate_up:
            N = N // 2
        KPack = 16
        KLane = 64 // NLane  # 4
        N0 = N // NLane
        K0 = K_pk // (KLane * KPack)
        if gate_up:
            src_reshaped = src.view(
                experts_cnt, 2, N0, NLane, K0, KLane, KPack
            )  # [E,2, N0, NLane ,K0, KLane, KPack]
            src_reshaped = src_reshaped.permute(
                0, 2, 1, 4, 5, 3, 6
            ).contiguous()  # [E, N0, 2, K0, KLane, NLane, KPack]
            interleaved = src_reshaped.view(*src.shape)
        else:
            src_reshaped = src.view(experts_cnt, N0, NLane, K0, KLane, KPack)
            interleaved = (
                src_reshaped.permute(0, 1, 3, 4, 2, 5).contiguous().view(*src.shape)
            )
        # print("interleaved shape:", interleaved.shape)
        return interleaved.contiguous().view(src_type)
    else:
        N, K_pk = src.shape
        KPack = 16
        KLane = 64 // NLane  # 4
        N0 = N // NLane
        K0 = K_pk // (KLane * KPack)
        src_reshaped = src.view(N0, NLane, K0, KLane, KPack)
        interleaved = src_reshaped.permute(0, 2, 3, 1, 4).contiguous().view(*src.shape)
        # print("interleaved shape:", interleaved.shape)
        return interleaved.contiguous().view(src_type)


def shuffle_scale_w4(
    src: torch.Tensor, experts_cnt: int, gate_up: bool
) -> torch.Tensor:
    n_experts, k_ = src.shape
    n_ = n_experts // experts_cnt
    # MXFP4 constants
    K_Pack = 2
    N_Pack = 2
    N_Lane = 16
    K_Lane = 64 // N_Lane  # 4

    # Basic dimensions
    K1 = k_ // K_Pack // K_Lane  # k_ // 8
    N1 = n_ // N_Lane // N_Pack  # n_ // 32
    real_k = 32 * k_ * K_Pack * K_Lane  # 1x32 quant
    assert real_k >= 256, f"K {real_k} must be larger than Tile_K(256)"
    # print("src shape", src.shape)
    # Reshape based on moe_kind
    if gate_up:
        # Reshape to: [E, N_Pack, N1, N_Lane, K1, K_Pack, K_Lane]
        shfl_scale = src.view(experts_cnt, N_Pack, N1, N_Lane, K1, K_Pack, K_Lane)
        # Permute to: [E, N1, K1, K_Lane, N_Lane, K_Pack, N_Pack]
        shfl_scale = shfl_scale.permute(0, 2, 4, 6, 3, 5, 1).contiguous()
    else:
        # Reshape to: [E, K1, K_Pack, K_Lane, N1, N_Pack, N_Lane]
        shfl_scale = src.view(experts_cnt, N1, N_Pack, N_Lane, K1, K_Pack, K_Lane)
        # Permute to: [E, N1, K1, K_Lane, N_Lane, K_Pack, N_Pack]
        shfl_scale = shfl_scale.permute(0, 1, 4, 6, 3, 5, 2).contiguous()
    # print("shf_scale shape:", shfl_scale.shape)
    return shfl_scale.view(*src.shape).contiguous()


def per_1x32_f4_quant(x, scale=None, quant_dtype=fp4x2, shuffle=False):
    assert quant_dtype == fp4x2
    block_size = 32
    F4E2M1_MAX = 6.0
    MAX_POW2 = int(torch.log2(torch.tensor(F4E2M1_MAX, dtype=torch.float32)).item())
    # dtypeMax = F4E2M1_MAX
    dtypeMax = 2.0**MAX_POW2

    shape_original = x.shape
    x = x.view(-1, shape_original[-1])

    m, _ = x.shape
    x = x.view(-1, block_size)
    max_abs = torch.amax(torch.abs(x.float()), 1)
    # max_abs = max_abs.view(torch.int32)
    # max_abs = ((max_abs + 0x200000) & 0xFF800000).view(torch.float32)

    # fp8e8m0fnu_from_fp32_value
    scale_e8m0_biased = f32_to_e8m0(max_abs / dtypeMax)

    # Float8_e8m0fnu to float
    scale_f32 = e8m0_to_f32(scale_e8m0_biased)

    y = x.float() / scale_f32.view(-1, 1)
    y_fp4 = f32_to_mxfp4(y)
    y_fp4 = y_fp4.view(*shape_original[:-1], -1)
    scale = scale_e8m0_biased.view(m, -1).view(torch.uint8)
    if shuffle:
        scale = e8m0_shuffle(scale)
    return y_fp4, scale.view(fp8_e8m0), y


# MXFP6 (E2M3) helpers - A operand for the W4A6 preshuffle GEMM
def pack_fp6_e2m3(x_unpacked: Tensor) -> Tensor:
    """Pack uint8 (low 6 bits = E2M3) 4-at-a-time into 3 dense bytes.

    Input (..., 4*G) uint8 -> output (..., 3*G) uint8 (little-endian groups:
    b0 = e1[1:0]<<6 | e0, b1 = e2[3:0]<<4 | e1>>2, b2 = e3<<2 | e2>>4).
    """
    assert x_unpacked.dtype == torch.uint8 and x_unpacked.shape[-1] % 4 == 0
    g = x_unpacked.unflatten(-1, (-1, 4)).to(torch.int32) & 0x3F
    e0, e1, e2, e3 = g.unbind(dim=-1)
    b0 = ((e1 & 0x03) << 6) | e0
    b1 = ((e2 & 0x0F) << 4) | (e1 >> 2)
    b2 = (e3 << 2) | (e2 >> 4)
    out = torch.stack([b0, b1, b2], dim=-1).to(torch.uint8)
    return out.reshape(
        *x_unpacked.shape[:-1], x_unpacked.shape[-1] // 4 * 3
    ).contiguous()


_FP6_E2M3_LUT: dict = {}


def fp6_e2m3_to_f32(x_unpacked: Tensor) -> Tensor:
    """Decode uint8 (low 6 bits = E2M3, 1 sign / 2 exp / 3 mant, bias 1) to fp32."""
    dev = x_unpacked.device
    lut = _FP6_E2M3_LUT.get(dev)
    if lut is None:
        vals = torch.empty(64, dtype=torch.float32)
        for c in range(64):
            sign = -1.0 if (c & 0x20) else 1.0
            exp = (c >> 3) & 0x3
            mant = c & 0x7
            mag = (mant / 8.0) if exp == 0 else (2.0 ** (exp - 1)) * (1.0 + mant / 8.0)
            vals[c] = sign * mag
        lut = vals.to(dev)
        _FP6_E2M3_LUT[dev] = lut
    return lut[(x_unpacked & 0x3F).long()]


def per_1x32_f6_quant(x):
    """Per-1x32 MXFP6 (E2M3) quant of an A operand, FP8-padded packed FP6.

    Returns:
      a_pad:      (M, K) uint8 - FP8-padded packed FP6 (24 B codes + 8 B zero
                  per K=32 chunk), the exact layout the kernel reads.
      scale:      (M, K//32) e8m0 (unshuffled; caller applies shuffle_scale_w4).
      a_unpacked: (M, K) uint8 - low-6-bit E2M3 codes (for the dequant reference).
    """
    block = 32
    F6E2M3_MAX = 7.5
    dtypeMax = 2.0 ** int(
        torch.log2(torch.tensor(F6E2M3_MAX, dtype=torch.float32)).item()
    )
    shape_original = x.shape
    xb = x.view(-1, shape_original[-1]).reshape(-1, block)
    max_abs = torch.amax(torch.abs(xb.float()), 1)
    scale_e8m0 = f32_to_e8m0(max_abs / dtypeMax)
    scale_f32 = e8m0_to_f32(scale_e8m0)
    y = xb.float() / scale_f32.view(-1, 1)
    codes = _f32_to_floatx_unpacked(y, 2, 3).to(torch.uint8)  # (.., 32) low6
    a_unpacked = codes.view(*shape_original).contiguous()
    M, K = a_unpacked.shape[0], a_unpacked.shape[-1]
    packed = pack_fp6_e2m3(a_unpacked).view(M, K // 32, 24)
    a_pad = torch.zeros(M, K // 32, 32, dtype=torch.uint8, device=x.device)
    a_pad[:, :, :24] = packed
    a_pad = a_pad.view(M, K)
    scale = scale_e8m0.view(M, -1).view(torch.uint8)
    return a_pad, scale, a_unpacked


def per_1x32_f8_quant(x):
    """Per-1x32 MXFP8 (E4M3) quant of an operand, 1 byte/code (no packing).

    Returns:
      x_q:   (.., K) float8_e4m3fn - the fp8 codes the kernel reads.
      scale: (.., K//32) e8m0 uint8 (unshuffled; caller applies shuffle_scale_w4).
    """
    block = 32
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    shape_original = x.shape
    x_flat = x.contiguous().view(-1, block).float()
    max_abs = torch.amax(torch.abs(x_flat), dim=-1).clamp_min(1e-30)
    scale_e8m0 = f32_to_e8m0(max_abs / fp8_max)
    scale_f32 = e8m0_to_f32(scale_e8m0).clamp_min(1e-30)
    x_q = (
        (x_flat / scale_f32.view(-1, 1))
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
    )
    x_q = x_q.view(shape_original).contiguous()
    scale = (
        scale_e8m0.view(*shape_original[:-1], shape_original[-1] // block)
        .view(torch.uint8)
        .contiguous()
    )
    return x_q, scale


def preshuffle_b_16x16(b: Tensor, rows: int, cols: int) -> Tensor:
    """Preshuffle B data into 16x16 byte tiles for WMMA-friendly LDS loads.

    Works for both FP4 (cols = K//2) and FP8 (cols = K).
    """
    assert rows % 16 == 0, f"rows must be a multiple of 16, got {rows}"
    assert cols % 16 == 0, f"cols must be a multiple of 16, got {cols}"
    b = b.view(rows, cols)
    b = b.view(rows // 16, 16, cols // 16, 16)
    b = b.permute(0, 2, 1, 3).contiguous()
    return b.view(rows, cols)
