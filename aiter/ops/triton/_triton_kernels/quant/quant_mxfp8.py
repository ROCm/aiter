# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# MXFP8 quantization Triton kernels: scale calculation, pack/unpack, and
# full-tensor convert-to / convert-from helpers for the MX FP8 format.

import triton
import triton.language as tl


@triton.jit
def _calculate_scales(
    x,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr = False,
    float8_dtype: tl.constexpr = tl.float8e4nv,
):
    E8M0_EXPONENT_BIAS: tl.constexpr = 127

    if IS_2D_BLOCK:
        tl.static_assert(
            BLOCK_M % QUANT_BLOCK_SIZE == 0,
            "block m must be divided by QUANT_BLOCK_SIZE",
        )
    tl.static_assert(
        BLOCK_N % QUANT_BLOCK_SIZE == 0, "block N must be divided by QUANT_BLOCK_SIZE"
    )
    if x.type.element_ty == tl.float32:
        hp_int_dtype = tl.int32
        hp_mbits = 23
        hp_ebits = 8
        hp_exp_bias = 127
    else:
        # bf16
        hp_int_dtype = tl.int16
        hp_mbits = 7
        hp_ebits = 8
        hp_exp_bias = 127

    sbits = 1
    if float8_dtype == tl.float8e4nv:
        mbits = 3
        target_max_pow2 = 8
    elif float8_dtype == tl.float8e5:
        mbits = 2
        target_max_pow2 = 15

    NEW_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    if IS_2D_BLOCK:
        NEW_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
        x = x.reshape(NEW_BLOCK_M, QUANT_BLOCK_SIZE, NEW_BLOCK_N, QUANT_BLOCK_SIZE)
        x = tl.permute(x, (0, 2, 1, 3))
        max_abs = tl.max(tl.abs(x), axis=-1)
        max_abs = tl.max(max_abs, axis=-1)
    else:
        x = x.reshape(BLOCK_M, NEW_BLOCK_N, QUANT_BLOCK_SIZE)
        max_abs = tl.max(tl.abs(x), axis=-1)
    max_abs = max_abs.to(x.type.element_ty)

    # round even (adaptive)
    # https://github.com/pytorch/ao/blob/a5f2693089b4c6528f019a0fb17235c9f22180a9/torchao/prototype/mx_formats/mx_tensor.py#L148
    max_abs = max_abs.to(hp_int_dtype, bitcast=True)
    val_to_add = 1 << (hp_mbits - mbits - 1)
    mask = ((1 << (hp_ebits + sbits)) - 1) << hp_mbits
    max_abs = (max_abs + val_to_add) & mask

    extracted_pow2 = ((max_abs >> hp_mbits) & 0b11111111) - hp_exp_bias
    scale_e8m0_unbiased = extracted_pow2 - target_max_pow2

    scale_e8m0_unbiased = tl.minimum(
        tl.maximum(scale_e8m0_unbiased, -1 * E8M0_EXPONENT_BIAS), E8M0_EXPONENT_BIAS + 1
    )
    scale_e8m0_biased = scale_e8m0_unbiased + E8M0_EXPONENT_BIAS

    return scale_e8m0_biased.to(tl.uint8)


@triton.jit
def _generate_randval(m, n, philox_seed, philox_offset):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    rng_offsets = philox_offset + ms[:, None] * n + ns[None, :]
    r1, _, _, _ = tl.randint4x(philox_seed, rng_offsets)
    return r1


@triton.jit
def _pack_fp8(
    x,
    scales,
    philox_seed,
    philox_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr = False,
    USE_SR: tl.constexpr = False,
    USE_ASM: tl.constexpr = False,
    float8_dtype: tl.constexpr = tl.float8e4nv,
):
    if float8_dtype == tl.float8e4nv:
        max_pos: tl.constexpr = 448.0
    else:
        max_pos: tl.constexpr = 57344.0

    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
    HALF_QUANT_BLOCK_SIZE: tl.constexpr = QUANT_BLOCK_SIZE // 2
    SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    x0, x1 = tl.split(x.reshape(BLOCK_M, HALF_BLOCK_N, 2))

    scales = tl.where(scales < 1, 1, scales)
    scales_fp32 = (scales.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    F32_MIN_NORMAL = tl.constexpr(2**-126)
    min_frag = (F32_MIN_NORMAL).to(tl.float32)
    scales_fp32 = tl.where(scales_fp32 < min_frag, min_frag, scales_fp32)

    if IS_2D_BLOCK:
        scales_fp32 = (
            scales_fp32.expand_dims(axis=(1, 3))
            .broadcast_to(
                SCALE_BLOCK_M, QUANT_BLOCK_SIZE, SCALE_BLOCK_N, HALF_QUANT_BLOCK_SIZE
            )
            .reshape(BLOCK_M, HALF_BLOCK_N)
        )
    else:
        scales_fp32 = (
            scales_fp32.expand_dims(axis=2)
            .broadcast_to(BLOCK_M, SCALE_BLOCK_N, HALF_QUANT_BLOCK_SIZE)
            .reshape(BLOCK_M, HALF_BLOCK_N)
        )

    if USE_SR:
        randval0 = _generate_randval(BLOCK_M, BLOCK_N, philox_seed, philox_offset)
    else:
        randval0 = 0

    if USE_ASM:
        tl.static_assert(float8_dtype == tl.float8e4nv)
        if x0.type.element_ty == tl.float32:
            if not USE_SR:
                y = tl.inline_asm_elementwise(
                    asm="""
                    v_cvt_scalef32_pk_fp8_f32 $0, $1, $2, $3 op_sel:[0,0,0,0];
                    """,
                    constraints="=&v,v,v,v",
                    args=[x0, x1, scales_fp32],
                    dtype=tl.uint16,
                    is_pure=True,
                    pack=1,
                )
            else:
                x0 = (x1.to(tl.uint32, bitcast=True).to(tl.uint64) << 32) | x0.to(
                    tl.uint32, bitcast=True
                )
                y = tl.inline_asm_elementwise(
                    asm="""
                    v_cvt_scalef32_sr_pk_fp8_f32 $0, $1, $2, $3 op_sel:[0,0,0,0];
                    """,
                    constraints="=&v,v,v,v",
                    args=[x0, randval0, scales_fp32],
                    dtype=tl.uint16,
                    is_pure=True,
                    pack=1,
                )
        else:
            if not USE_SR:
                x0 = (x1.to(tl.uint16, bitcast=True).to(tl.uint32) << 16) | x0.to(
                    tl.uint16, bitcast=True
                )
                y = tl.inline_asm_elementwise(
                    asm="""
                    v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2 op_sel:[0,0,0,0];
                    """,
                    constraints="=&v,v,v",
                    args=[x0, scales_fp32],
                    dtype=tl.uint16,
                    is_pure=True,
                    pack=1,
                )
            else:
                scales_uint8 = scales.to(tl.uint8)
                scales_fp32 = (
                    scales_fp32.expand_dims(axis=2)
                    .broadcast_to(BLOCK_M, HALF_BLOCK_N, 2)
                    .reshape(BLOCK_M, BLOCK_N)
                )
                y = tl.inline_asm_elementwise(
                    asm="""
                    v_cvt_scalef32_sr_fp8_bf16 $0, $1, $2, $3 op_sel:[0,0,0,0];
                    """,
                    constraints="=&v,v,v,v",
                    args=[x, randval0, scales_uint8],
                    dtype=float8_dtype,
                    is_pure=True,
                    pack=1,
                )

        if x0.type.element_ty == tl.float32 or not USE_SR:
            y1 = (y >> 8).to(tl.uint8).to(float8_dtype, bitcast=True)
            y0 = (y & 0x00FF).to(tl.uint8).to(float8_dtype, bitcast=True)
            y = tl.join(y0, y1).reshape(BLOCK_M, BLOCK_N)
        y = y.to(tl.float32)
        mask_pos = (y != y) & (x > 0)  # noqa: PLR0124
        mask_neg = (y != y) & (x < 0)  # noqa: PLR0124
        y = tl.where(mask_neg, -max_pos, y)
        y = tl.where(mask_pos, max_pos, y)
        y = y.to(float8_dtype)
    else:
        x_scaled = x / scales_fp32.expand_dims(axis=2).broadcast_to(
            BLOCK_M, HALF_BLOCK_N, 2
        ).reshape(BLOCK_M, BLOCK_N)

        if USE_SR:
            noise = randval0.to(tl.float32) * (1.0 / 4294967296.0)
            x_scaled = x_scaled + (noise - 0.5) * 0.01

        y = x_scaled.to(float8_dtype)

    return y.reshape(BLOCK_M, BLOCK_N)


@triton.jit
def _unpack_fp8(
    x,
    scales,
    output_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr,
    USE_ASM: tl.constexpr = False,
):
    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
    HALF_QUANT_BLOCK_SIZE: tl.constexpr = QUANT_BLOCK_SIZE // 2
    SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    x0, x1 = tl.split(x.reshape(BLOCK_M, HALF_BLOCK_N, 2))

    scales = tl.where(scales < 1, 1, scales)
    scales_fp32 = (scales.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    if IS_2D_BLOCK:
        scales_fp32 = (
            scales_fp32.expand_dims(axis=(1, 3))
            .broadcast_to(
                SCALE_BLOCK_M, QUANT_BLOCK_SIZE, SCALE_BLOCK_N, HALF_QUANT_BLOCK_SIZE
            )
            .reshape(BLOCK_M, HALF_BLOCK_N)
        )
    else:
        scales_fp32 = (
            scales_fp32.expand_dims(axis=2)
            .broadcast_to(BLOCK_M, SCALE_BLOCK_N, HALF_QUANT_BLOCK_SIZE)
            .reshape(BLOCK_M, HALF_BLOCK_N)
        )

    if USE_ASM:
        tl.static_assert(x.type.element_ty == tl.float8e4nv)
        x0 = (x1.to(tl.uint8, bitcast=True).to(tl.uint16) << 8) | x0.to(
            tl.uint8, bitcast=True
        )
        if output_dtype == tl.float32:
            y_packed = tl.inline_asm_elementwise(
                asm="""
                v_cvt_scalef32_pk_f32_fp8 $0, $1, $2 op_sel:[0,0];
                """,
                constraints="=&v,v,v",
                args=[x0, scales_fp32],
                dtype=tl.uint64,
                is_pure=True,
                pack=1,
            )
            y1 = (y_packed >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
            y0 = (
                (y_packed & 0x00000000FFFFFFFF)
                .to(tl.uint32)
                .to(tl.float32, bitcast=True)
            )
        else:
            y_packed = tl.inline_asm_elementwise(
                asm="""
                v_cvt_scalef32_pk_bf16_fp8 $0, $1, $2 op_sel:[0,0];
                """,
                constraints="=&v,v,v",
                args=[x0, scales_fp32],
                dtype=tl.uint32,
                is_pure=True,
                pack=1,
            )
            y1 = (y_packed >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
            y0 = (y_packed & 0x0000FFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
        y = tl.join(y0, y1).reshape(BLOCK_M, BLOCK_N)
    else:
        y = x.to(output_dtype)
        y = y * scales_fp32.expand_dims(axis=2).broadcast_to(
            BLOCK_M, HALF_BLOCK_N, 2
        ).reshape(BLOCK_M, BLOCK_N)

    return y


@triton.jit
def _convert_to_mxfp8_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    philox_seed,
    philox_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr,
    USE_SR: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_xn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_sn = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
    if IS_2D_BLOCK:
        offs_sm = pid_m * SCALE_BLOCK_M + tl.arange(0, SCALE_BLOCK_M)
    else:
        offs_sm = offs_m

    offs_x = offs_m[:, None] * stride_xm + offs_xn[None, :] * stride_xn
    offs_s = offs_sm[:, None] * stride_sm + offs_sn[None, :] * stride_sn

    tl.static_assert(
        x_ptr.type.element_ty == tl.float32 or x_ptr.type.element_ty == tl.bfloat16
    )

    float8_dtype = y_ptr.type.element_ty
    tl.static_assert(float8_dtype == tl.float8e4nv or float8_dtype == tl.float8e5)

    x = tl.load(x_ptr + offs_x)

    scales = _calculate_scales(
        x,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=IS_2D_BLOCK,
        float8_dtype=float8_dtype,
    )
    y = _pack_fp8(
        x,
        scales,
        philox_seed,
        philox_offset,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=IS_2D_BLOCK,
        USE_SR=USE_SR,
        USE_ASM=USE_ASM,
        float8_dtype=float8_dtype,
    )

    offs_y = offs_m[:, None] * stride_ym + offs_yn[None, :] * stride_yn
    tl.store(y_ptr + offs_y, y.to(y_ptr.type.element_ty))
    tl.store(s_ptr + offs_s, scales)


@triton.jit
def _convert_from_mxfp8_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_xn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_sn = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
    if IS_2D_BLOCK:
        offs_sm = pid_m * SCALE_BLOCK_M + tl.arange(0, SCALE_BLOCK_M)
    else:
        offs_sm = offs_m

    offs_x = offs_m[:, None] * stride_xm + offs_xn[None, :] * stride_xn
    offs_s = offs_sm[:, None] * stride_sm + offs_sn[None, :] * stride_sn

    x = tl.load(x_ptr + offs_x)
    s = tl.load(s_ptr + offs_s)

    tl.static_assert(
        y_ptr.type.element_ty == tl.float32 or y_ptr.type.element_ty == tl.bfloat16
    )
    tl.static_assert(
        x_ptr.type.element_ty == tl.float8e4nv or x_ptr.type.element_ty == tl.float8e5
    )

    y = _unpack_fp8(
        x,
        s,
        y_ptr.type.element_ty,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=IS_2D_BLOCK,
        USE_ASM=USE_ASM,
    )
    offs_y = offs_m[:, None] * stride_ym + offs_yn[None, :] * stride_yn
    tl.store(y_ptr + offs_y, y)
