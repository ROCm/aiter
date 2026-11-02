# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.quant.quant import (
    _mxfp4_e8m0_to_fp32,
    _mxfp4_pack_op,
    _mxfp4_scale_from_amax,
    _mxfp4_sr_pack,
    _mxfp4_sr_random_words,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.mxfp4 import (
    mxfp4_apply_quant_scale,
    mxfp4_shuffled_data_offsets,
    mxfp4_swizzled_scale_offsets,
    normalized_hadamard16,
)


@triton.jit
def _decode_e2m1(code):
    magnitude = code & 0x07
    value = tl.where(
        magnitude == 0,
        0.0,
        tl.where(
            magnitude == 1,
            0.5,
            tl.where(
                magnitude == 2,
                1.0,
                tl.where(
                    magnitude == 3,
                    1.5,
                    tl.where(
                        magnitude == 4,
                        2.0,
                        tl.where(
                            magnitude == 5,
                            3.0,
                            tl.where(magnitude == 6, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return tl.where((code & 0x08) != 0, -value, value)


_dequant_hadamard_quant_mxfp4_repr = make_kernel_repr(
    "_dequant_hadamard_quant_mxfp4_kernel",
    [
        "BLOCK_SIZE",
        "HADAMARD_SIZE",
        "USE_SR",
        "SWIZZLE_SCALE",
        "SHUFFLE_DATA",
        "IN_SCALE_SWIZZLED",
        "num_warps",
        "num_stages",
    ],
)


@triton.jit(repr=_dequant_hadamard_quant_mxfp4_repr)
def _dequant_hadamard_quant_mxfp4_kernel(
    packed_ptr,
    input_scale_ptr,
    sign_ptr,
    output_ptr,
    output_scale_ptr,
    stride_packed_m_in,
    stride_packed_k_in,
    stride_input_scale_m_in,
    stride_input_scale_k_in,
    stride_output_k_in,
    stride_output_m_in,
    stride_output_scale_k_in,
    stride_output_scale_m_in,
    M,
    K,
    philox_seed,
    philox_offset,
    BLOCK_SIZE: tl.constexpr,
    HADAMARD_SIZE: tl.constexpr,
    USE_SR: tl.constexpr,
    SWIZZLE_SCALE: tl.constexpr,
    SHUFFLE_DATA: tl.constexpr,
    IN_SCALE_SWIZZLED: tl.constexpr,
    SCALE_STRIPE: tl.constexpr,
    SCALE_KCHUNK: tl.constexpr,
    SHUFFLE_TILE_ROWS: tl.constexpr,
    SHUFFLE_UNIT_BYTES: tl.constexpr,
):
    """Dequantize, transpose, H16-rotate, and requantize one 32x32 tile."""
    tl.static_assert(BLOCK_SIZE == 32)
    tl.static_assert(HADAMARD_SIZE == 16)

    pid_m = tl.cast(tl.program_id(0), tl.int64)
    pid_k = tl.cast(tl.program_id(1), tl.int64)

    stride_packed_m = tl.cast(stride_packed_m_in, tl.int64)
    stride_packed_k = tl.cast(stride_packed_k_in, tl.int64)
    stride_input_scale_m = tl.cast(stride_input_scale_m_in, tl.int64)
    stride_input_scale_k = tl.cast(stride_input_scale_k_in, tl.int64)
    stride_output_k = tl.cast(stride_output_k_in, tl.int64)
    stride_output_m = tl.cast(stride_output_m_in, tl.int64)
    stride_output_scale_k = tl.cast(stride_output_scale_k_in, tl.int64)
    stride_output_scale_m = tl.cast(stride_output_scale_m_in, tl.int64)

    offsets_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    offsets_k = pid_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    offsets_k_packed = pid_k * (BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2).to(
        tl.int64
    )
    packed_offsets = (
        offsets_m[:, None] * stride_packed_m
        + offsets_k_packed[None, :] * stride_packed_k
    )
    packed = tl.load(packed_ptr + packed_offsets, cache_modifier=".cg").to(tl.uint8)

    low = _decode_e2m1(packed & 0x0F)
    high = _decode_e2m1((packed >> 4) & 0x0F)
    values = tl.reshape(tl.join(low, high), (BLOCK_SIZE, BLOCK_SIZE))

    input_scale_cols = K // BLOCK_SIZE
    if IN_SCALE_SWIZZLED:
        input_scale_offsets = mxfp4_swizzled_scale_offsets(
            offsets_m,
            pid_k,
            input_scale_cols,
            stripe=SCALE_STRIPE,
            kchunk=SCALE_KCHUNK,
        )
    else:
        input_scale_offsets = (
            offsets_m * stride_input_scale_m + pid_k * stride_input_scale_k
        )
    input_scale_raw = tl.load(
        input_scale_ptr + input_scale_offsets, cache_modifier=".cg"
    ).to(tl.uint8)
    invalid_input_scale = tl.max((input_scale_raw == 255).to(tl.int32), axis=0) != 0
    input_scale = _mxfp4_e8m0_to_fp32(input_scale_raw)
    input_scale = tl.where(input_scale_raw == 255, 0.0, input_scale)
    dequantized = values * input_scale[:, None]

    sign = tl.load(sign_ptr + tl.arange(0, HADAMARD_SIZE)).to(tl.float32)
    rows: tl.constexpr = BLOCK_SIZE * (BLOCK_SIZE // HADAMARD_SIZE)
    # The Lumen composition materializes this stage as BF16 before H16. E2M1
    # times a power-of-two scale is exact in BF16 whenever it is finite.
    transposed = tl.trans(dequantized.to(tl.bfloat16)).to(tl.float32)
    signed = tl.reshape(transposed, (rows, HADAMARD_SIZE)) * sign[None, :]
    rotated = tl.reshape(
        normalized_hadamard16(signed, rows=rows),
        (BLOCK_SIZE, BLOCK_SIZE),
    )
    # Lumen's fused production path quantizes the FP32 H16 accumulator directly.
    # Its matrix multiply canonicalizes exact cancellation and signed zero to
    # +0, so preserve that byte-level behavior before E2M1 packing.
    rotated = tl.where(rotated == 0.0, 0.0, rotated)

    rotated_bits = rotated.to(tl.uint32, bitcast=True)
    invalid_value = (rotated_bits & 0x7F800000) == 0x7F800000
    invalid_row = tl.max(invalid_value.to(tl.int32), axis=1) != 0
    finite_rotated = tl.where(invalid_value, 0.0, rotated)
    row_amax = tl.max(tl.abs(finite_rotated), axis=1, keep_dims=True)
    output_scale_raw, quant_scale = _mxfp4_scale_from_amax(row_amax)

    if USE_SR:
        random_words = _mxfp4_sr_random_words(
            offsets_k,
            pid_m,
            M,
            philox_seed,
            philox_offset,
            BLOCK_SIZE_M=BLOCK_SIZE,
            BLOCK_SIZE_N=BLOCK_SIZE,
        )
        output = _mxfp4_sr_pack(
            finite_rotated,
            output_scale_raw,
            random_words,
            BLOCK_SIZE_M=BLOCK_SIZE,
            BLOCK_SIZE_N=BLOCK_SIZE,
            MXFP4_QUANT_BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        normalized = mxfp4_apply_quant_scale(
            finite_rotated, output_scale_raw, quant_scale
        )
        output = _mxfp4_pack_op(
            normalized,
            BLOCK_SIZE,
            BLOCK_SIZE,
            BLOCK_SIZE,
            USE_ASM=True,
        )

    output_scale_raw = tl.reshape(output_scale_raw, (BLOCK_SIZE,))
    invalid_row = invalid_row | invalid_input_scale
    output_scale_raw = tl.where(invalid_row, 255, output_scale_raw).to(tl.uint8)
    output = tl.where(invalid_row[:, None], 0, output)

    offsets_m_packed = pid_m * (BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2).to(
        tl.int64
    )
    if SHUFFLE_DATA:
        output_offsets = mxfp4_shuffled_data_offsets(
            offsets_k[:, None],
            offsets_m_packed[None, :],
            M // 2,
            tile_rows=SHUFFLE_TILE_ROWS,
            unit_bytes=SHUFFLE_UNIT_BYTES,
        )
    else:
        output_offsets = (
            offsets_k[:, None] * stride_output_k
            + offsets_m_packed[None, :] * stride_output_m
        )
    tl.store(output_ptr + output_offsets, output)

    output_scale_cols = M // BLOCK_SIZE
    if SWIZZLE_SCALE:
        output_scale_offsets = mxfp4_swizzled_scale_offsets(
            offsets_k,
            pid_m,
            output_scale_cols,
            stripe=SCALE_STRIPE,
            kchunk=SCALE_KCHUNK,
        )
    else:
        output_scale_offsets = (
            offsets_k * stride_output_scale_k + pid_m * stride_output_scale_m
        )
    tl.store(output_scale_ptr + output_scale_offsets, output_scale_raw)
