# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import torch
from . import dtypes
from torch import Tensor
# import triton
# import triton.language as tl


def f32_to_mxfp4(x):
    FP4_EBITS, FP4_MBITS = 2, 1
    x = _f32_to_floatx_unpacked(x.float(), FP4_EBITS, FP4_MBITS)
    x = pack_uint4(x)
    x = x.view(dtypes.fp4x2)  # to(fp32) for this datatype gives all 0 for torch...
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
    exponent = torch.where(round_case, exponent + 1, exponent)
    exponent = torch.where(nan_case, torch.full_like(exponent, 0xFF), exponent)
    return exponent.view(dtypes.fp8_e8m0)


def e8m0_to_f32(scale_e8m0_biased):
    scale_e8m0_biased = scale_e8m0_biased.view(torch.uint8)
    zero_case = scale_e8m0_biased == 0
    nan_case = scale_e8m0_biased == 0xFF
    scale_f32 = scale_e8m0_biased.to(torch.int32) << 23
    scale_f32 = torch.where(zero_case, torch.full_like(scale_f32, 0x00400000), scale_f32)
    scale_f32 = torch.where(nan_case, torch.full_like(scale_f32, 0x7F800001), scale_f32)
    scale_f32 = scale_f32.view(dtypes.fp32)
    return scale_f32


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


# --------------------------------------------------------------------------
# Triton kernels below were removed for the pure-PyTorch path; the wrappers
# ``dynamic_mxfp4_quant`` and ``moe_mxfp4_sort`` further down are now plain
# PyTorch. The original kernels were left as a doc-string for reference.
# --------------------------------------------------------------------------


def dynamic_mxfp4_quant(
    x: torch.Tensor, scaling_mode: str = "even", shuffle: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch port of the triton MXFP4 dynamic quant kernel.

    Quantize a 2-D tensor to MXFP4 (e2m1) with per-1x32 e8m0 block scales.
    Output shapes match the original triton wrapper:
        x_fp4:           (M, N // 2)    fp4x2
        blockscale_e8m0: (M, ceil(N/32)) e8m0    when shuffle=False
        blockscale_e8m0: padded + permuted layout when shuffle=True
    """
    M, N = x.shape
    assert (N // 2) % 2 == 0

    MXFP4_QUANT_BLOCK_SIZE = 32
    F4E2M1_MAX = 6.0
    MAX_POW2 = int(torch.log2(torch.tensor(F4E2M1_MAX, dtype=torch.float32)).item())
    dtype_max_pow2 = 2.0 ** MAX_POW2

    scaleN_valid = (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE

    x_blk = x.float().contiguous().view(-1, MXFP4_QUANT_BLOCK_SIZE)
    max_abs = torch.amax(torch.abs(x_blk), dim=1)
    scale_e8m0 = f32_to_e8m0(max_abs / dtype_max_pow2)
    scale_f32 = e8m0_to_f32(scale_e8m0)
    scale_f32 = torch.where(scale_f32 == 0, torch.ones_like(scale_f32), scale_f32)
    y_f32 = x_blk / scale_f32.view(-1, 1)
    y_fp4 = f32_to_mxfp4(y_f32).view(M, N // 2)

    scale = scale_e8m0.view(M, scaleN_valid).view(torch.uint8)
    if shuffle:
        scale = e8m0_shuffle(scale)
    return y_fp4.view(dtypes.fp4x2), scale.view(dtypes.fp8_e8m0)


# --------------------------------------------------------------------------
# Triton sort kernels (kept as a docstring for reference; pytorch port lives
# in ``moe_mxfp4_sort`` below).
# --------------------------------------------------------------------------


def moe_mxfp4_sort(
    blockscale_e8m0: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    block_size: int = 32,
) -> torch.Tensor:
    """Pure-PyTorch port of the triton MoE block-scale sort kernel.

    Gathers ``blockscale_e8m0`` rows by ``sorted_ids`` and packs them into
    32x8 super-tiles laid out as (PM, PN, 4, 16) uint32, matching the
    consumer kernel's expected byte stream when reinterpreted as e8m0.

    Args:
        blockscale_e8m0: 2-D (M_i, N_i) or 3-D (token_num, topk, N_i) e8m0
            scales (uint8 view-equivalent).
        sorted_ids: (M_o,) int — low 24 bits = token id, high 8 bits = topk
            index. Entries beyond ``num_valid_ids`` are ignored.
        num_valid_ids: scalar tensor with the count of valid entries in
            ``sorted_ids``.
        token_num: source ``token_num`` for masking out-of-range gathers.
        block_size: must be a multiple of 32.

    Returns:
        e8m0 tensor of shape ``(-1, N_i)`` in the tile-shuffled byte layout.
    """
    BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 8

    topk = 1
    if blockscale_e8m0.ndim == 3:
        topk = blockscale_e8m0.shape[1]
        blockscale_e8m0 = blockscale_e8m0.reshape(-1, blockscale_e8m0.shape[-1])
    M_i, N_i = blockscale_e8m0.shape
    M_o, N_o = sorted_ids.shape[0], N_i
    assert (N_i // 2) % 2 == 0
    assert block_size % BLOCK_SIZE_M == 0

    device = blockscale_e8m0.device
    bse_u8 = blockscale_e8m0.contiguous().view(torch.uint8)

    raw = sorted_ids.to(torch.int64)
    token_ids = raw & 0xFFFFFF
    if topk == 1:
        src_rows = token_ids
    else:
        src_rows = token_ids * topk + (raw >> 24)

    valid_pos = torch.arange(M_o, device=device, dtype=torch.int64) < num_valid_ids.to(torch.int64)
    valid_tok = token_ids < token_num
    valid = valid_pos & valid_tok

    # Append a zero-row sentinel for safe out-of-bounds gather.
    bse_padded_src = torch.cat(
        [bse_u8, torch.zeros((1, N_i), dtype=torch.uint8, device=device)], dim=0
    )
    src_rows_safe = torch.where(
        valid, src_rows, torch.full_like(src_rows, M_i)
    )
    gathered = bse_padded_src[src_rows_safe]  # (M_o, N_i) uint8

    PM = (M_o + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    PN = (N_o + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    M_o_pad = PM * BLOCK_SIZE_M
    N_o_pad = PN * BLOCK_SIZE_N
    if M_o_pad != M_o or N_o_pad != N_o:
        padded = torch.zeros((M_o_pad, N_o_pad), dtype=torch.uint8, device=device)
        padded[:M_o, :N_o] = gathered
        gathered = padded

    # (PM, 32, PN, 8) -> (PM, PN, 32, 8) super-tiles.
    tile = gathered.view(PM, BLOCK_SIZE_M, PN, BLOCK_SIZE_N).permute(0, 2, 1, 3).contiguous()
    # Split rows into (m_idx in {0,1}, i in 0..15), cols into (n_idx in {0,1}, j in 0..3).
    sub = tile.view(PM, PN, 2, 16, 2, 4)
    b0 = sub[:, :, 0, :, 0, :].to(torch.int64)  # (PM, PN, 16, 4)
    b1 = sub[:, :, 1, :, 0, :].to(torch.int64)
    b2 = sub[:, :, 0, :, 1, :].to(torch.int64)
    b3 = sub[:, :, 1, :, 1, :].to(torch.int64)
    # Pack 4 bytes into one uint32 (little-endian: b0 is LSB).
    packed64 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    out_i32 = packed64.to(torch.int32)
    # Permute (PM, PN, 16, 4) -> (PM, PN, 4, 16) to match storage layout (j, i).
    out_i32 = out_i32.permute(0, 1, 3, 2).contiguous()

    # Reinterpret bytes and reshape to (-1, N_o).
    out_bytes = out_i32.view(torch.uint8).reshape(-1, N_o)
    return out_bytes.view(dtypes.fp8_e8m0)
