# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _get_max_quant_val(dtype: tl.constexpr):
    if dtype == tl.float8e5:
        return 57344.0
    elif dtype == tl.float8e4nv:
        return 448.0
    else:
        tl.static_assert(False, f"Invalid {dtype=}")


@triton.jit
def _get_max_power_of_2_quant_val(dtype: tl.constexpr):
    if dtype == tl.float8e5:
        return 32768.0
    elif dtype == tl.float8e4nv:
        return 256.0
    else:
        tl.static_assert(False, f"Invalid {dtype=}")


@triton.jit
def _compute_mx_quant_and_scale(
    src_tensor,
    valid_src_mask,
    mx_tensor_dtype: tl.constexpr,
    SCALE_ROUNDING_MODE: tl.constexpr = 0,
):
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // 32

    # Explicit cast to fp32 since most ops are not supported on bfloat16. We avoid needless conversions to and from bf16
    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(
        valid_src_mask, abs_tensor, -1.0
    )  # Don't consider padding tensors in scale computation
    abs_tensor = tl.reshape(
        abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32]
    )
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)
    if SCALE_ROUNDING_MODE == 0:
        # ROUND_UP
        # compute 2 ** ceil(log2(dequant_scale))
        # Adding 0x007FFFFF adds exponent by 1 unless mantissa is all zeros
        # A corner case: exponent is 0xFF that will overflow but that's already
        # NaN so assume we don't care.
        dequant_scale = max_val / _get_max_quant_val(mx_tensor_dtype)
        dequant_scale_exponent = (
            dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF
        ) & 0x7F800000
    else:
        # ROUND_DOWN
        # compute 2 ** floor(log2(dequant_scale))
        assert SCALE_ROUNDING_MODE == 1
        dequant_scale = max_val / _get_max_power_of_2_quant_val(mx_tensor_dtype)
        dequant_scale_exponent = dequant_scale.to(tl.uint32, bitcast=True) & 0x7F800000
    dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
    quant_scale = tl.where(dequant_scale_rounded == 0, 0, 1.0 / dequant_scale_rounded)

    f32_tensor = tl.reshape(
        f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32]
    )
    quant_tensor = f32_tensor * quant_scale

    # Reshape the tensors after scaling
    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    # Set the invalid portions of the tensor to 0. This will ensure that any padding tensors are 0 in the mx format.
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0)
    dequant_scale_exponent = dequant_scale_exponent.reshape(
        [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE]
    )

    # First, we simply extract the exponent part of the scales and store the result
    dequant_scale_exponent = (dequant_scale_exponent >> 23).to(tl.uint8)
    # Now we must convert the tensors to the mx format.
    out_tensor = quant_tensor.to(mx_tensor_dtype)

    return out_tensor, dequant_scale_exponent


@triton.jit
def _downcast_to_mxfp8(
    mx_tensor_ptr,
    stride_mxt_outer,
    stride_mxt_quant: tl.constexpr,
    mx_scale_ptr,
    stride_mx_scale_outer,
    stride_mx_scale_quant,
    src_ptr,
    stride_src_outer,
    stride_src_quant,
    outer_dim,
    quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    SCALE_ROUNDING_MODE: tl.constexpr,
):

    tl.static_assert(
        stride_mxt_quant == 1, f"Output stride, {stride_mxt_quant=} must be 1."
    )
    tl.static_assert(
        BLOCK_SIZE_QUANT_DIM % 32 == 0,
        f"{BLOCK_SIZE_QUANT_DIM=} must be a multiple of 32",
    )

    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    tl.static_assert(
        mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5,
        f"Invalid {mx_tensor_dtype=}. Must be float8.",
    )

    src_dtype: tl.constexpr = src_ptr.dtype.element_ty
    tl.static_assert(
        mx_scale_ptr.dtype.element_ty == tl.uint8,
        f"{mx_scale_ptr.dtype.element_ty=} must be uint8",
    )
    tl.static_assert(
        (src_dtype == tl.float32)
        or (src_dtype == tl.bfloat16)
        or (src_dtype == tl.float16),
        f"{src_dtype=} must be float32 or bfloat16 or float16",
    )

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 32
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM

    start_src_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_mx_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    src_ptr += start_src_quant * stride_src_quant + start_out * stride_src_outer
    mx_scale_ptr += (
        start_mx_scale_quant * stride_mx_scale_quant + start_out * stride_mx_scale_outer
    )
    mx_tensor_ptr += start_mx_quant * stride_mxt_quant + start_out * stride_mxt_outer

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_mxt_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_scale_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)

    mask_src_quant = start_src_quant + offs_src_quant < quant_dim
    mask_n = start_out + offs_outer < outer_dim
    full_mask_src = mask_src_quant & mask_n

    mask_mxt_quant = start_mx_quant + offs_mxt_quant < quant_dim
    full_mask_mxt = mask_mxt_quant & mask_n

    scale_mask_k = start_mx_scale_quant + offs_scale_quant < tl.cdiv(quant_dim, 32)
    full_scale_mask = scale_mask_k & mask_n

    src_tensor_offsets = (
        offs_src_quant * stride_src_quant + offs_outer * stride_src_outer
    )
    mx_scale_offsets = (
        offs_scale_quant * stride_mx_scale_quant + offs_outer * stride_mx_scale_outer
    )
    mx_tensor_offsets = (
        offs_mxt_quant * stride_mxt_quant + offs_outer * stride_mxt_outer
    )
    src_tensor = tl.load(src_ptr + src_tensor_offsets, mask=full_mask_src)

    out_tensor, scale_tensor = _compute_mx_quant_and_scale(
        src_tensor, full_mask_src, mx_tensor_dtype, SCALE_ROUNDING_MODE
    )

    tl.store(mx_scale_ptr + mx_scale_offsets, scale_tensor, mask=full_scale_mask)
    tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)


@triton.jit
def _upcast_from_mxfp8(
    out_ptr,
    stride_o_outer,
    stride_o_quant: tl.constexpr,
    mx_scale_ptr,
    stride_scale_outer,
    stride_scale_quant,
    mx_tensor_ptr,
    stride_tensor_outer,
    stride_tensor_quant: tl.constexpr,
    outer_dim,
    quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
):

    tl.static_assert(
        stride_o_quant == 1, "the weight must be contiguous in the k dimension for mx"
    )
    tl.static_assert(
        BLOCK_SIZE_QUANT_DIM % 32 == 0, "BLOCK_SIZE_K must be a multiple of 32"
    )
    # uint8 signifies two fp4 e2m1 values packed into a single byte
    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    dst_dtype: tl.constexpr = out_ptr.dtype.element_ty
    tl.static_assert(
        dst_dtype == tl.float32 or dst_dtype == tl.float16 or dst_dtype == tl.bfloat16
    )
    tl.static_assert(
        (mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5)
        or mx_tensor_dtype == dst_dtype,
        "mx_tensor_ptr must be float8 or dst_dtype",
    )
    tl.static_assert(
        mx_scale_ptr.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8"
    )

    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 32
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM

    # Compute starting indices for the quantized (packed) dimension and the outer dimension.
    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    start_mxt_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    mx_tensor_ptr += (
        start_mxt_quant * stride_tensor_quant + start_out * stride_tensor_outer
    )
    mx_scale_ptr += (
        start_mx_scale_quant * stride_scale_quant + start_out * stride_scale_outer
    )
    out_ptr += start_out * stride_o_outer + start_out_quant * stride_o_quant

    # Compute offsets and masks.
    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_out_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)
    offs_scale = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)

    mask_outer = start_out + offs_outer < outer_dim
    mask_out_quant = start_out_quant + offs_out_quant < quant_dim
    full_mask_out = mask_out_quant & mask_outer

    mask_src_quant = start_mxt_quant + offs_src_quant < quant_dim
    full_mask_src = mask_src_quant & mask_outer

    mask_scale = start_mx_scale_quant + offs_scale < tl.cdiv(quant_dim, 32)
    full_scale_mask = mask_scale & mask_outer

    tensor_offsets = (
        offs_src_quant * stride_tensor_quant + offs_outer * stride_tensor_outer
    )
    scale_offsets = offs_scale * stride_scale_quant + offs_outer * stride_scale_outer
    out_offsets = offs_out_quant * stride_o_quant + offs_outer * stride_o_outer

    # Load the packed tensor and scale.
    tensor = tl.load(mx_tensor_ptr + tensor_offsets, mask=full_mask_src)
    scale = tl.load(mx_scale_ptr + scale_offsets, mask=full_scale_mask)

    # Upcast the scale.
    dst_scale = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)

    # Now upcast the tensor.
    dst_tensor = tensor.to(tl.float32)

    # Reshape for proper broadcasting: the scale was stored with a 32-sized "inner" grouping.
    dst_tensor = dst_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32])
    dst_scale = dst_scale.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1])
    scale = scale.reshape(dst_scale.shape)

    out_tensor = dst_tensor * dst_scale
    # Correct any NaNs encoded via the scale.
    out_tensor = tl.where(scale == 0xFF, float("nan"), out_tensor)
    out_tensor = out_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    out_tensor = out_tensor.to(dst_dtype)
    tl.store(out_ptr + out_offsets, out_tensor, mask=full_mask_out)


@triton.jit
def _compute_mx_quant_and_scale_2d(
    src_tensor,
    valid_src_mask,
    mx_tensor_dtype: tl.constexpr,
    SCALE_ROUNDING_MODE: tl.constexpr = 0,
):
    BLOCK_SIZE_M: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_N: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_M_SCALE: tl.constexpr = BLOCK_SIZE_M // 32
    BLOCK_SIZE_N_SCALE: tl.constexpr = BLOCK_SIZE_N // 32

    # Explicit cast to fp32 since most ops are not supported on bfloat16.
    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    # Don't consider padding tensors in scale computation
    abs_tensor = tl.where(valid_src_mask, abs_tensor, -1.0)

    # Reshape to (M_SCALE, 32, N_SCALE, 32) so that for each (i, j) scale block,
    # the elements live at abs_4d[i, :, j, :] — a 32x32 sub-block.
    abs_4d = tl.reshape(abs_tensor, [BLOCK_SIZE_M_SCALE, 32, BLOCK_SIZE_N_SCALE, 32])
    # Two sequential reductions to compute the max over each 32x32 block.
    max_val = tl.max(abs_4d, axis=3, keep_dims=True)
    max_val = tl.max(max_val, axis=1, keep_dims=True)

    dequant_scale = max_val / _get_max_quant_val(mx_tensor_dtype)
    if SCALE_ROUNDING_MODE == 0:
        # ROUND_UP
        dequant_scale_exponent = (
            dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF
        ) & 0x7F800000
    else:
        # ROUND_DOWN
        tl.static_assert(SCALE_ROUNDING_MODE == 1)
        dequant_scale_exponent = dequant_scale.to(tl.uint32, bitcast=True) & 0x7F800000
    dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
    quant_scale = tl.where(dequant_scale_rounded == 0, 0, 1.0 / dequant_scale_rounded)

    # Broadcast (M_SCALE, 1, N_SCALE, 1) over (M_SCALE, 32, N_SCALE, 32).
    f32_tensor_4d = tl.reshape(
        f32_tensor, [BLOCK_SIZE_M_SCALE, 32, BLOCK_SIZE_N_SCALE, 32]
    )
    quant_tensor = f32_tensor_4d * quant_scale

    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_M, BLOCK_SIZE_N])
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0)

    dequant_scale_exponent = dequant_scale_exponent.reshape(
        [BLOCK_SIZE_M_SCALE, BLOCK_SIZE_N_SCALE]
    )
    dequant_scale_exponent = (dequant_scale_exponent >> 23).to(tl.uint8)
    out_tensor = quant_tensor.to(mx_tensor_dtype)

    return out_tensor, dequant_scale_exponent


@triton.jit
def _downcast_to_mxfp8_2d(
    mx_tensor_ptr,
    stride_mxt_b,
    stride_mxt_m,
    stride_mxt_n: tl.constexpr,
    mx_scale_ptr,
    stride_mx_scale_b,
    stride_mx_scale_m,
    stride_mx_scale_n,
    src_ptr,
    stride_src_b,
    stride_src_m,
    stride_src_n,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCALE_ROUNDING_MODE: tl.constexpr,
):

    tl.static_assert(stride_mxt_n == 1, f"Output stride, {stride_mxt_n=} must be 1.")
    tl.static_assert(
        BLOCK_SIZE_M % 32 == 0, f"{BLOCK_SIZE_M=} must be a multiple of 32"
    )
    tl.static_assert(
        BLOCK_SIZE_N % 32 == 0, f"{BLOCK_SIZE_N=} must be a multiple of 32"
    )

    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    tl.static_assert(
        mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5,
        f"Invalid {mx_tensor_dtype=}. Must be float8.",
    )

    src_dtype: tl.constexpr = src_ptr.dtype.element_ty
    tl.static_assert(
        mx_scale_ptr.dtype.element_ty == tl.uint8,
        f"{mx_scale_ptr.dtype.element_ty=} must be uint8",
    )
    tl.static_assert(
        (src_dtype == tl.float32)
        or (src_dtype == tl.bfloat16)
        or (src_dtype == tl.float16),
        f"{src_dtype=} must be float32 or bfloat16 or float16",
    )

    batch = tl.program_id(0).to(tl.int64)
    m_block = tl.program_id(1).to(tl.int64)
    n_block = tl.program_id(2).to(tl.int64)

    BLOCK_SIZE_M_SCALE: tl.constexpr = BLOCK_SIZE_M // 32
    BLOCK_SIZE_N_SCALE: tl.constexpr = BLOCK_SIZE_N // 32

    start_m = m_block * BLOCK_SIZE_M
    start_n = n_block * BLOCK_SIZE_N
    start_scale_m = m_block * BLOCK_SIZE_M_SCALE
    start_scale_n = n_block * BLOCK_SIZE_N_SCALE

    src_ptr += batch * stride_src_b + start_m * stride_src_m + start_n * stride_src_n
    mx_tensor_ptr += (
        batch * stride_mxt_b + start_m * stride_mxt_m + start_n * stride_mxt_n
    )
    mx_scale_ptr += (
        batch * stride_mx_scale_b
        + start_scale_m * stride_mx_scale_m
        + start_scale_n * stride_mx_scale_n
    )

    offs_m = tl.arange(0, BLOCK_SIZE_M)[:, None].to(tl.int64)
    offs_n = tl.arange(0, BLOCK_SIZE_N)[None, :].to(tl.int64)
    offs_scale_m = tl.arange(0, BLOCK_SIZE_M_SCALE)[:, None].to(tl.int64)
    offs_scale_n = tl.arange(0, BLOCK_SIZE_N_SCALE)[None, :].to(tl.int64)

    mask_m = start_m + offs_m < M
    mask_n = start_n + offs_n < N
    full_mask = mask_m & mask_n

    mask_scale_m = start_scale_m + offs_scale_m < tl.cdiv(M, 32)
    mask_scale_n = start_scale_n + offs_scale_n < tl.cdiv(N, 32)
    full_scale_mask = mask_scale_m & mask_scale_n

    src_offsets = offs_m * stride_src_m + offs_n * stride_src_n
    mx_tensor_offsets = offs_m * stride_mxt_m + offs_n * stride_mxt_n
    scale_offsets = offs_scale_m * stride_mx_scale_m + offs_scale_n * stride_mx_scale_n

    src_tensor = tl.load(src_ptr + src_offsets, mask=full_mask)

    out_tensor, scale_tensor = _compute_mx_quant_and_scale_2d(
        src_tensor, full_mask, mx_tensor_dtype, SCALE_ROUNDING_MODE
    )

    tl.store(mx_scale_ptr + scale_offsets, scale_tensor, mask=full_scale_mask)
    tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask)


@triton.jit
def _upcast_from_mxfp8_2d(
    out_ptr,
    stride_o_b,
    stride_o_m,
    stride_o_n: tl.constexpr,
    mx_scale_ptr,
    stride_scale_b,
    stride_scale_m,
    stride_scale_n,
    mx_tensor_ptr,
    stride_tensor_b,
    stride_tensor_m,
    stride_tensor_n: tl.constexpr,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):

    tl.static_assert(
        stride_o_n == 1,
        "the weight must be contiguous in the n dimension for mx",
    )
    tl.static_assert(BLOCK_SIZE_M % 32 == 0, "BLOCK_SIZE_M must be a multiple of 32")
    tl.static_assert(BLOCK_SIZE_N % 32 == 0, "BLOCK_SIZE_N must be a multiple of 32")
    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    dst_dtype: tl.constexpr = out_ptr.dtype.element_ty
    tl.static_assert(
        dst_dtype == tl.float32 or dst_dtype == tl.float16 or dst_dtype == tl.bfloat16
    )
    tl.static_assert(
        (mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5)
        or mx_tensor_dtype == dst_dtype,
        "mx_tensor_ptr must be float8 or dst_dtype",
    )
    tl.static_assert(
        mx_scale_ptr.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8"
    )

    BLOCK_SIZE_M_SCALE: tl.constexpr = BLOCK_SIZE_M // 32
    BLOCK_SIZE_N_SCALE: tl.constexpr = BLOCK_SIZE_N // 32

    batch = tl.program_id(0).to(tl.int64)
    m_block = tl.program_id(1).to(tl.int64)
    n_block = tl.program_id(2).to(tl.int64)

    start_m = m_block * BLOCK_SIZE_M
    start_n = n_block * BLOCK_SIZE_N
    start_scale_m = m_block * BLOCK_SIZE_M_SCALE
    start_scale_n = n_block * BLOCK_SIZE_N_SCALE

    mx_tensor_ptr += (
        batch * stride_tensor_b + start_m * stride_tensor_m + start_n * stride_tensor_n
    )
    mx_scale_ptr += (
        batch * stride_scale_b
        + start_scale_m * stride_scale_m
        + start_scale_n * stride_scale_n
    )
    out_ptr += batch * stride_o_b + start_m * stride_o_m + start_n * stride_o_n

    offs_m = tl.arange(0, BLOCK_SIZE_M)[:, None].to(tl.int64)
    offs_n = tl.arange(0, BLOCK_SIZE_N)[None, :].to(tl.int64)
    offs_scale_m = tl.arange(0, BLOCK_SIZE_M_SCALE)[:, None].to(tl.int64)
    offs_scale_n = tl.arange(0, BLOCK_SIZE_N_SCALE)[None, :].to(tl.int64)

    mask_m = start_m + offs_m < M
    mask_n = start_n + offs_n < N
    full_mask = mask_m & mask_n

    mask_scale_m = start_scale_m + offs_scale_m < tl.cdiv(M, 32)
    mask_scale_n = start_scale_n + offs_scale_n < tl.cdiv(N, 32)
    full_scale_mask = mask_scale_m & mask_scale_n

    tensor_offsets = offs_m * stride_tensor_m + offs_n * stride_tensor_n
    scale_offsets = offs_scale_m * stride_scale_m + offs_scale_n * stride_scale_n
    out_offsets = offs_m * stride_o_m + offs_n * stride_o_n

    tensor = tl.load(mx_tensor_ptr + tensor_offsets, mask=full_mask)
    scale = tl.load(mx_scale_ptr + scale_offsets, mask=full_scale_mask)

    dst_scale = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    dst_tensor = tensor.to(tl.float32)

    # Broadcast the per-32x32-block scale across the full tile.
    dst_tensor = dst_tensor.reshape([BLOCK_SIZE_M_SCALE, 32, BLOCK_SIZE_N_SCALE, 32])
    dst_scale = dst_scale.reshape([BLOCK_SIZE_M_SCALE, 1, BLOCK_SIZE_N_SCALE, 1])
    scale_4d = scale.reshape([BLOCK_SIZE_M_SCALE, 1, BLOCK_SIZE_N_SCALE, 1])

    out_tensor = dst_tensor * dst_scale
    out_tensor = tl.where(scale_4d == 0xFF, float("nan"), out_tensor)
    out_tensor = out_tensor.reshape([BLOCK_SIZE_M, BLOCK_SIZE_N])
    out_tensor = out_tensor.to(dst_dtype)
    tl.store(out_ptr + out_offsets, out_tensor, mask=full_mask)
