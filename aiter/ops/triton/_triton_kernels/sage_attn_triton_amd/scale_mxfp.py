from enum import Enum
import torch
import triton
import triton.language as tl

class DequantScaleRoundingMode(Enum):
    ROUND_RNE = 2

@triton.jit
def _get_max_quant_val(dtype: tl.constexpr):
    if dtype == tl.uint8:
        return 6.0
    elif dtype == tl.float8e5:
        return 57344.0
    elif dtype == tl.float8e4nv:
        return 448.0
    else:
        tl.static_assert(False, f"Invalid {dtype=}")

def downcast_to_mxfp_rne(
    src_tensor: torch.Tensor,
    out_quant_type: torch.dtype,
    axis: int,
    DEQUANT_SCALE_ROUNDING_MODE: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_RNE,
):
    """
    Convert the src weights to mx format. The src weight is quantized along the axis dimension.

    If weight_quant_type is torch.uint8, we output mxfp4 where two e2m1 values are packed into a single byte.
    Note that this means the k_dim of the tensor will be half of the logical k_dim.

    If weight_quant_type is torch.float8_e4m3fn or torch.float8_e5m2, we output mxfp8 with the float8s are stored
    in their respective formats.
    """

    assert DEQUANT_SCALE_ROUNDING_MODE == DequantScaleRoundingMode.ROUND_RNE, f"DEQUANT_SCALE_ROUNDING_MODE should be {DequantScaleRoundingMode.ROUND_RNE}."

    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim
    # downcast
    src_tensor = src_tensor.transpose(axis, src_tensor.ndim - 1)
    is_fp4 = out_quant_type == torch.uint8
    is_fp8 = out_quant_type in (torch.float8_e4m3fn, torch.float8_e5m2)
    assert is_fp4 or is_fp8
    divisor = 2 if is_fp4 else 1
    L = src_tensor.shape[-1]
    if is_fp4:
        assert L % 2 == 0, f"axis dim must be divisible by 2 for e2m1. Got {L}"
    out_shape = src_tensor.shape[:-1] + (L // divisor,)
    out_scale_shape = src_tensor.shape[:-1] + (triton.cdiv(L, 32),)

    out_quant_tensor = src_tensor.new_empty(out_shape, dtype=out_quant_type)
    out_scale = src_tensor.new_empty(out_scale_shape, dtype=torch.uint8)

    kernel_src_tensor = src_tensor.reshape(-1, src_tensor.shape[-1])
    kernel_quant_tensor = out_quant_tensor.view(-1, out_quant_tensor.shape[-1])
    kernel_scale = out_scale.view(-1, out_scale.shape[-1])

    BLOCK_OUT_DIM = 128
    BLOCK_QUANT_DIM = 32
    grid_out = triton.cdiv(kernel_src_tensor.shape[0], BLOCK_OUT_DIM)
    grid_quant = triton.cdiv(kernel_src_tensor.shape[1], BLOCK_QUANT_DIM)

    _downcast_to_mxfp[(grid_out, grid_quant)](
        kernel_quant_tensor,
        *kernel_quant_tensor.stride(),
        kernel_scale,
        *kernel_scale.stride(),
        kernel_src_tensor,
        *kernel_src_tensor.stride(),
        *kernel_src_tensor.shape,
        BLOCK_OUT_DIM,
        BLOCK_QUANT_DIM,
        DEQUANT_SCALE_ROUNDING_MODE.value,
        num_warps=8,
    )

    out_quant_tensor = out_quant_tensor.transpose(axis, src_tensor.ndim - 1)
    out_scale = out_scale.transpose(axis, src_tensor.ndim - 1)
    return out_quant_tensor, out_scale

@triton.jit
def _downcast_to_mxfp(
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
    DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr,
):
    tl.static_assert(
        stride_mxt_quant == 1, f"Output stride, {stride_mxt_quant=} must be 1."
    )
    tl.static_assert(
        BLOCK_SIZE_QUANT_DIM % 32 == 0,
        f"{BLOCK_SIZE_QUANT_DIM=} must be a multiple of 32",
    )

    # uint8 signifies two fp4 e2m1 values packed into a single byte
    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    tl.static_assert(
        mx_tensor_dtype == tl.uint8
        or (mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5),
        f"Invalid {mx_tensor_dtype=}. Must be uint8 or float8.",
    )

    src_dtype: tl.constexpr = src_ptr.dtype.element_ty
    tl.static_assert(
        mx_scale_ptr.dtype.element_ty == tl.uint8,
        f"{mx_scale_ptr.dtype.element_ty=} must be uint8",
    )
    tl.static_assert(
        (src_dtype == tl.bfloat16) or (src_dtype == tl.float16),
        f"{src_dtype=} must be bfloat16 or float16",
    )
    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    K_DIVISOR: tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 32
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

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

    mask_mxt_quant = start_mx_quant + offs_mxt_quant < tl.cdiv(quant_dim, K_DIVISOR)
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

    out_tensor, scale_tensor = _compute_mx_quant_and_scale_rne(
        src_tensor, full_mask_src, mx_tensor_dtype
    )

    tl.store(mx_scale_ptr + mx_scale_offsets, scale_tensor, mask=full_scale_mask)
    tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)

@triton.jit
def _compute_mx_quant_and_scale_rne(
    src_tensor,
    valid_src_mask,
    mx_tensor_dtype: tl.constexpr,
):
    """
    Compute MX quantization with RNE (Round to Nearest Even) rounding for the scale.
    
    RNE is applied when converting max_abs/max_quant_val to E8M0 format (nearest power of 2).
    This is equivalent to computing: scale = 2^(clip(floor(log2(RNE(max_abs/max_quant_val))), -127, 127) - 2)
    where RNE rounds to the nearest power of 2, with ties going to even exponent.
    """
    is_fp8: tl.constexpr = (
        mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    )
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // 32

    # Explicit cast to fp32 since most ops are not supported on bfloat16
    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(
        valid_src_mask, abs_tensor, -1.0
    )  # Don't consider padding tensors in scale computation
    abs_tensor = tl.reshape(
        abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32]
    )
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)
    dequant_scale = max_val / _get_max_quant_val(mx_tensor_dtype)
    
    # RNE (Round to Nearest Even) rounding when converting max_abs to E8M0 format
    # E8M0 stores only exponent (no mantissa), so we round to nearest power of 2
    # Extract exponent and mantissa from float32
    dequant_scale_bits = dequant_scale.to(tl.uint32, bitcast=True)
    exponent = (dequant_scale_bits >> 23) & 0xFF
    mantissa = dequant_scale_bits & 0x7FFFFF
    
    # RNE to nearest power of 2:
    # For value 2^n * (1 + m/2^23), the threshold is at m = 0.5 * 2^23 = 0x400000
    # - If mantissa < 0x400000: round to 2^n (keep exponent)
    # - If mantissa > 0x400000: round to 2^(n+1) (increment exponent)
    # - If mantissa == 0x400000: tie case, round to even exponent (RNE)
    
    # Determine if we should round up
    should_round_up = (mantissa > 0x400000) | (
        (mantissa == 0x400000) & ((exponent & 1) == 1)
    )
    
    rounded_exponent = tl.where(should_round_up, exponent + 1, exponent)
    
    # Construct the rounded scale as a power of 2
    dequant_scale_exponent = (rounded_exponent << 23) & 0x7F800000
    dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
    quant_scale = tl.where(dequant_scale_rounded == 0, 0, 1.0 / dequant_scale_rounded)

    f32_tensor = tl.reshape(
        f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32]
    )
    quant_tensor = f32_tensor * quant_scale

    # Reshape the tensors after scaling
    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    # Set the invalid portions of the tensor to 0
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0)
    dequant_scale_exponent = dequant_scale_exponent.reshape(
        [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE]
    )

    # Extract the exponent part of the scales and store the result
    dequant_scale_exponent = (dequant_scale_exponent >> 23).to(tl.uint8)
    
    # Convert the tensors to the mx format
    if is_fp8:
        out_tensor = quant_tensor.to(mx_tensor_dtype)
    else:
        quant_tensor = quant_tensor.to(tl.uint32, bitcast=True)
        signs = quant_tensor & 0x80000000
        exponents = (quant_tensor >> 23) & 0xFF
        mantissas = quant_tensor & 0x7FFFFF

        # 0.25 <= x < 0.75 maps to 0.5, a denormal number
        E8_BIAS = 127
        E2_BIAS = 1
        # Move implicit bit 1 at the beginning to mantissa for denormals
        adjusted_exponents = tl.core.sub(
            E8_BIAS, exponents + 1, sanitize_overflow=False
        )
        mantissas = tl.where(
            exponents < E8_BIAS,
            (0x400000 | (mantissas >> 1)) >> adjusted_exponents,
            mantissas,
        )

        # For normal numbers, we change the bias from 127 to 1, and for subnormals, we keep exponent as 0.
        exponents = tl.maximum(exponents, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

        # Combine sign, exponent, and mantissa, while saturating
        # rounding nearest with tie breaking up by adding +1 to one bit right of the LSB, then shift right
        e2m1_tmp = tl.minimum((((exponents << 2) | (mantissas >> 21)) + 1) >> 1, 0x7)
        e2m1_value = ((signs >> 28) | e2m1_tmp).to(tl.uint8)

        e2m1_value = tl.reshape(
            e2m1_value, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2]
        )
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)

    return out_tensor, dequant_scale_exponent
