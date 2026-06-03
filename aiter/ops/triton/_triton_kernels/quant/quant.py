# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _static_per_tensor_quant_fp8_i8_kernel(
    qx_ptr,
    x_in_ptr,
    scale_in_ptr,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    scale = tl.load(scale_in_ptr)
    scale_recip = 1 / scale

    qx = (x * scale_recip).to(qx_ptr.dtype.element_ty)

    tl.store(qx_ptr + offs, qx, mask=mask)


@triton.jit
def _dynamic_per_tensor_quant_fp8_i8_kernel(
    x_in_ptr,
    scale_out_ptr,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    m = tl.max(tl.abs(x))
    tl.atomic_max(scale_out_ptr, m / DTYPE_MAX, sem="relaxed")


@triton.jit
def _dynamic_per_token_quant_fp8_i8_kernel(
    qx_ptr,
    scale_out_ptr,
    x_in_ptr,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    m = tl.max(tl.abs(x), axis=-1)
    scale_out = m.to(tl.float32) / DTYPE_MAX
    scale_recip = 1 / scale_out

    qx = x * scale_recip
    qx = qx.to(qx_ptr.dtype.element_ty)

    scale_offs = pid
    tl.store(scale_out_ptr + scale_offs, scale_out)

    tl.store(qx_ptr + offs, qx, mask=mask, cache_modifier=".cs")


@triton.jit
def _mxfp4_quant_op(
    x,
    BLOCK_SIZE_N,
    BLOCK_SIZE_M,
    MXFP4_QUANT_BLOCK_SIZE,
):
    """
    Converts given x (in fp32) to mxfp4 format.
    x: [BLOCK_SIZE_M, BLOCK_SIZE_N], fp32

    """
    EXP_BIAS_FP32: tl.constexpr = 127
    EXP_BIAS_FP4: tl.constexpr = 1
    EBITS_F32: tl.constexpr = 8
    EBITS_FP4: tl.constexpr = 2
    MBITS_F32: tl.constexpr = 23
    MBITS_FP4: tl.constexpr = 1

    max_normal: tl.constexpr = 6
    min_normal: tl.constexpr = 1

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)
    # Calculate scale
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)

    # blockscale_e8m0
    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127  # in fp32, we have 2&(e - 127)

    quant_scale = tl.exp2(-scale_e8m0_unbiased)

    # Compute quantized x
    qx = x * quant_scale

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

    # Extract sign
    s = qx & 0x80000000
    # Set everything to positive, will add sign back at the end
    qx = qx ^ s

    qx_fp32 = qx.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    # Denormal numbers
    denorm_exp: tl.constexpr = (
        (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    )
    denorm_mask_int: tl.constexpr = denorm_exp << MBITS_F32
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)

    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)

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
    normal_x = normal_x.to(tl.uint8)

    # Merge results
    e2m1_value = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)
    # add sign back
    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1_value = e2m1_value | sign_lp
    e2m1_value = tl.reshape(
        e2m1_value, [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2]
    )
    evens, odds = tl.split(e2m1_value)
    x_fp4 = evens | (odds << 4)
    x_fp4 = x_fp4.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)


@triton.jit
def _nvfp4_quant_op(
    x,
    BLOCK_SIZE_N,
    BLOCK_SIZE_M,
    NVFP4_QUANT_BLOCK_SIZE,
):
    """
    Converts given x (in fp32) to nvfp4 format.
    x: [BLOCK_SIZE_M, BLOCK_SIZE_N], fp32

    """
    EXP_BIAS_FP32: tl.constexpr = 127
    EXP_BIAS_FP4: tl.constexpr = 1
    EBITS_F32: tl.constexpr = 8
    EBITS_FP4: tl.constexpr = 2
    MBITS_F32: tl.constexpr = 23
    MBITS_FP4: tl.constexpr = 1

    max_normal: tl.constexpr = 6
    min_normal: tl.constexpr = 1

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // NVFP4_QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, NVFP4_QUANT_BLOCK_SIZE)
    # Calculate scale
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
    scale_e4m3 = amax.to(tl.float32) / 6.0
    quant_scale = 1.0 / scale_e4m3

    # Compute quantized x
    qx = x * quant_scale

    # Convert quantized fp32 tensor to uint32 before converting to nvfp4 format
    # Note: NVFP4  S:1-bit, E:2-bit, M:1-bit
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

    # Extract sign
    s = qx & 0x80000000
    # Set everything to positive, will add sign back at the end
    qx = qx ^ s

    qx_fp32 = qx.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    # Denormal numbers
    denorm_exp: tl.constexpr = (
        (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    )
    denorm_mask_int: tl.constexpr = denorm_exp << MBITS_F32
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)

    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)

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
    normal_x = normal_x.to(tl.uint8)

    # Merge results
    e2m1_value = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)
    # add sign back
    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1_value = e2m1_value | sign_lp
    e2m1_value = tl.reshape(
        e2m1_value, [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, NVFP4_QUANT_BLOCK_SIZE // 2, 2]
    )
    evens, odds = tl.split(e2m1_value)
    x_fp4 = evens | (odds << 4)
    x_fp4 = x_fp4.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

    return x_fp4, scale_e4m3.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)


@triton.heuristics(
    {
        "EVEN_M_N": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N"] % (args["BLOCK_SIZE_N"] * args["NUM_ITER"]) == 0,
    }
)
@triton.jit
def _dynamic_mxfp4_quant_kernel(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_x_fp4_m_in,
    stride_x_fp4_n_in,
    stride_bs_m_in,
    stride_bs_n_in,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_ITER: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    EVEN_M_N: tl.constexpr,
    SCALING_MODE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    start_n = tl.program_id(1) * NUM_ITER
    # cast strides to int64, in case M*N > max int32
    stride_x_m = tl.cast(stride_x_m_in, tl.int64)
    stride_x_n = tl.cast(stride_x_n_in, tl.int64)
    stride_x_fp4_m = tl.cast(stride_x_fp4_m_in, tl.int64)
    stride_x_fp4_n = tl.cast(stride_x_fp4_n_in, tl.int64)
    stride_bs_m = tl.cast(stride_bs_m_in, tl.int64)
    stride_bs_n = tl.cast(stride_bs_n_in, tl.int64)

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE

    for pid_n in tl.range(start_n, min(start_n + NUM_ITER, N), num_stages=NUM_STAGES):
        x_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        x_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n

        if EVEN_M_N:
            x = tl.load(x_ptr + x_offs, cache_modifier=".cg").to(tl.float32)
        else:
            x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
            x = tl.load(x_ptr + x_offs, mask=x_mask, cache_modifier=".cg").to(
                tl.float32
            )

        out_tensor, bs_e8m0 = _mxfp4_quant_op(
            x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
        )

        out_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        out_offs_n = pid_n * BLOCK_SIZE_N // 2 + tl.arange(0, BLOCK_SIZE_N // 2)
        out_offs = (
            out_offs_m[:, None] * stride_x_fp4_m + out_offs_n[None, :] * stride_x_fp4_n
        )

        if EVEN_M_N:
            tl.store(x_fp4_ptr + out_offs, out_tensor)
        else:
            out_mask = (out_offs_m < M)[:, None] & (out_offs_n < (N // 2))[None, :]
            tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask)

        bs_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        bs_offs_n = pid_n * NUM_QUANT_BLOCKS + tl.arange(0, NUM_QUANT_BLOCKS)
        bs_offs = bs_offs_m[:, None] * stride_bs_m + bs_offs_n[None, :] * stride_bs_n
        if EVEN_M_N:
            tl.store(bs_ptr + bs_offs, bs_e8m0)
        else:
            bs_mask = (bs_offs_m < M)[:, None] & (
                bs_offs_n < (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
            )[None, :]
            tl.store(
                bs_ptr + bs_offs,
                bs_e8m0,
                mask=bs_mask,
            )


# MXFP8 (1x32 e8m0) quant: derives a per-block uint8 e8m0 scale + FP8 e4m3
# values. The bit-trick (bitcast amax to int32, add 0x200000, mask 0xFF800000,
# bitcast back to fp32) rounds amax up to a power of 2; log2(amax).floor() - 8
# is the unbiased e8m0 exponent (dtypeMax = 2**8).


@triton.jit
def _mxfp8_quant_op(x_grouped, QUANT_AXIS: tl.constexpr):
    """Shared MXFP8 (1x32 e8m0) scale derivation.

    Given a fp32 tile where the QUANT_AXIS dim is sized QUANT_BLOCK_SIZE (=32),
    returns (scale_e8m0, quant_scale): the per-group uint8 e8m0 scale and the
    matching fp32 multiplicative scale. Both outputs keep QUANT_AXIS with size 1
    so they broadcast against the input for in-place quantization.
    """
    amax = tl.max(tl.abs(x_grouped), axis=QUANT_AXIS, keep_dims=True)
    amax_i32 = amax.to(tl.int32, bitcast=True)
    amax_i32 = (amax_i32 + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax_p2 = amax_i32.to(tl.float32, bitcast=True)
    scale_unbiased = tl.log2(amax_p2).floor() - 8
    scale_unbiased = tl.clamp(scale_unbiased, min=-127, max=127)
    scale_e8m0 = (scale_unbiased.to(tl.int32) + 127).to(tl.uint8)
    quant_scale = tl.exp2(-scale_unbiased)
    return scale_e8m0, quant_scale


@triton.jit
def _dynamic_mxfp8_quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    BLOCK_SIZE_N: tl.constexpr,  # power-of-2 covering full N
    QUANT_BLOCK_SIZE: tl.constexpr,  # =32
    NUM_PRGMS: tl.constexpr,  # row-loop range (usually =M)
):
    """
    Per-1x32 MXFP8 quant. One program per row, holding the full row in
    registers so a single launch handles all K-groups. Mirrors
    _fused_rms_mxfp8_kernel shape (in fused_mxfp8_quant.py) and minimizes
    grid overhead.
    """
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = col_offsets < N
    n_groups: tl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE

    for row_idx in tl.range(row_start, M, NUM_PRGMS, num_stages=2):
        x = tl.load(
            x_ptr + row_idx * stride_xm + col_offsets * stride_xn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        # (BLOCK_SIZE_N,) -> (n_groups, QUANT_BLOCK_SIZE)
        x_2d = tl.reshape(x, (n_groups, QUANT_BLOCK_SIZE))
        scale_e8m0, quant_scale = _mxfp8_quant_op(x_2d, QUANT_AXIS=1)

        qx_2d = x_2d * quant_scale
        qx = tl.reshape(qx_2d, (BLOCK_SIZE_N,))
        y = qx.to(y_ptr.type.element_ty)

        tl.store(
            y_ptr + row_idx * stride_ym + col_offsets * stride_yn,
            y,
            mask=mask,
        )

        group_offsets = tl.arange(0, n_groups)
        group_mask = group_offsets < (N // QUANT_BLOCK_SIZE)
        scale_flat = tl.reshape(scale_e8m0, (n_groups,))
        tl.store(
            s_ptr + row_idx * stride_sm + group_offsets * stride_sn,
            scale_flat,
            mask=group_mask,
        )


# Transcoder: (FP8 fnuz, fp32 1x128 scale) -> (FP8 fn, e8m0 1x32 scale).
# Replaces the Python dequant+requant cascade (fp32 cast + multiply + bf16 cast
# + per_1x32_mxfp8 quant) used in linear.py's MXFP8 fallback path for MLA wq_b
# when q_norm emits the legacy fp8 fnuz + fp32 1x128 format.
#
# In: x_fp8_fnuz (M, N) — fp8 e4m3fnuz bits (interpreted with bias 8 -> value)
#     x_scale_fp32 (M, N//128) — fp32 per-token-block scale
# Out: y_fp8_fn (M, N) — fp8 e4m3fn bits (NV format, bias 7)
#      y_scale_e8m0 (M, N//32) — uint8 e8m0 (1x32 MX scale)


@triton.jit
def _fp8_legacy_to_mxfp8_kernel(
    x_fnuz_ptr,
    x_scale_fp32_ptr,
    y_fn_ptr,
    y_scale_e8m0_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_xsm,
    stride_xsn,
    stride_ym,
    stride_yn,
    stride_ysm,
    stride_ysn,
    BLOCK_SIZE_M: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,  # =32 (MXFP8 group)
    LEGACY_BLOCK_SIZE: tl.constexpr,  # =128 (input scale group)
):
    """
    One program per (BLOCK_SIZE_M rows, QUANT_BLOCK_SIZE-element column window).
    For each 1x32 block, dequantize fnuz fp8 values using the corresponding
    1x128 fp32 scale, derive the e8m0 (1x32) scale, then re-quantize to fp8 fn.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * QUANT_BLOCK_SIZE + tl.arange(0, QUANT_BLOCK_SIZE)

    x_offs = offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load fp8 fnuz values; .to(fp32) decodes via fnuz bias 8 semantically.
    x_fnuz = tl.load(x_fnuz_ptr + x_offs, mask=x_mask, other=0.0).to(tl.float32)

    # Which legacy 1x128 group does this 1x32 block fall into?
    legacy_n = (pid_n * QUANT_BLOCK_SIZE) // LEGACY_BLOCK_SIZE
    xs_offs = offs_m * stride_xsm + legacy_n * stride_xsn
    xs_mask = offs_m < M
    x_scale = tl.load(x_scale_fp32_ptr + xs_offs, mask=xs_mask, other=1.0)

    # Dequantize: bf16-equivalent reconstruction.
    x_dq = x_fnuz * x_scale[:, None]

    # Derive new e8m0 (1x32) scale from x_dq amax.
    scale_e8m0, quant_scale = _mxfp8_quant_op(x_dq, QUANT_AXIS=1)

    # Re-quantize to fp8 fn.
    qx = x_dq * quant_scale
    y = qx.to(y_fn_ptr.type.element_ty)

    y_offs = offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_fn_ptr + y_offs, y, mask=x_mask)

    s_offs = offs_m[:, None] * stride_ysm + pid_n * stride_ysn
    s_mask = offs_m[:, None] < M
    tl.store(y_scale_e8m0_ptr + s_offs, scale_e8m0, mask=s_mask)


@triton.heuristics(
    {
        "EVEN_M_N": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N"] % (args["BLOCK_SIZE_N"] * args["NUM_ITER"]) == 0,
    }
)
@triton.jit
def _dynamic_nvfp4_quant_kernel(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_x_fp4_m_in,
    stride_x_fp4_n_in,
    stride_bs_m_in,
    stride_bs_n_in,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_ITER: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NVFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    EVEN_M_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    start_n = tl.program_id(1) * NUM_ITER
    # cast strides to int64, in case M*N > max int32
    stride_x_m = tl.cast(stride_x_m_in, tl.int64)
    stride_x_n = tl.cast(stride_x_n_in, tl.int64)
    stride_x_fp4_m = tl.cast(stride_x_fp4_m_in, tl.int64)
    stride_x_fp4_n = tl.cast(stride_x_fp4_n_in, tl.int64)
    stride_bs_m = tl.cast(stride_bs_m_in, tl.int64)
    stride_bs_n = tl.cast(stride_bs_n_in, tl.int64)

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // NVFP4_QUANT_BLOCK_SIZE

    for pid_n in tl.range(start_n, min(start_n + NUM_ITER, N), num_stages=NUM_STAGES):
        x_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        x_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n

        if EVEN_M_N:
            x = tl.load(x_ptr + x_offs, cache_modifier=".cg").to(tl.float32)
        else:
            x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
            x = tl.load(x_ptr + x_offs, mask=x_mask, cache_modifier=".cg").to(
                tl.float32
            )

        out_tensor, scale_e4m3 = _nvfp4_quant_op(
            x, BLOCK_SIZE_N, BLOCK_SIZE_M, NVFP4_QUANT_BLOCK_SIZE
        )

        out_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        out_offs_n = pid_n * BLOCK_SIZE_N // 2 + tl.arange(0, BLOCK_SIZE_N // 2)
        out_offs = (
            out_offs_m[:, None] * stride_x_fp4_m + out_offs_n[None, :] * stride_x_fp4_n
        )

        if EVEN_M_N:
            tl.store(x_fp4_ptr + out_offs, out_tensor)
        else:
            out_mask = (out_offs_m < M)[:, None] & (out_offs_n < (N // 2))[None, :]
            tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask)

        bs_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        bs_offs_n = pid_n * NUM_QUANT_BLOCKS + tl.arange(0, NUM_QUANT_BLOCKS)
        bs_offs = bs_offs_m[:, None] * stride_bs_m + bs_offs_n[None, :] * stride_bs_n
        if EVEN_M_N:
            tl.store(bs_ptr + bs_offs, scale_e4m3.to(bs_ptr.type.element_ty))
        else:
            bs_mask = (bs_offs_m < M)[:, None] & (
                bs_offs_n < (N + NVFP4_QUANT_BLOCK_SIZE - 1) // NVFP4_QUANT_BLOCK_SIZE
            )[None, :]
            tl.store(
                bs_ptr + bs_offs,
                scale_e4m3.to(bs_ptr.type.element_ty),
                mask=bs_mask,
            )
