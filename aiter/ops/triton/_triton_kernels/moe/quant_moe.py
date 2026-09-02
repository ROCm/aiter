import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.quant.quant import (
    _mxfp4_quant_op,
    _mxfp8_quant_op,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit
def _compute_static_fp8_quant(tensor, scale):
    tensor = tensor.to(tl.float32)
    tensor = tensor / scale
    tensor = tensor.to(tl.float8e4nv)
    return tensor


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


@triton.jit
def _compute_mx_quant_and_scale(
    src_tensor,
    valid_src_mask,
    mx_tensor_dtype: tl.constexpr,
    DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr = 0,
    POW2_SCALE: tl.constexpr = False,
):
    """Quantize a [OUT_DIM, QUANT_DIM] fp32/bf16/fp16 tile to MXFP4 or MXFP8.

    Two scale-derivation schemes, selected by POW2_SCALE:

    * ``POW2_SCALE=False`` (default): ``scale = amax / dtype_max`` with the
      exponent rounded per DEQUANT_SCALE_ROUNDING_MODE. Never saturates, but
      leaves up to 2x of the dtype's range unused when amax sits just above a
      power of two.
    * ``POW2_SCALE=True``: the ``even_round`` scheme shared with
      :func:`_mxfp4_quant_op` / :func:`_mxfp8_quant_op` -- amax is rounded to the
      nearest power of two and the scale is taken relative to the largest power
      of two the dtype holds (4 for e2m1, 256 for e4m3, 32768 for e5m2). Uses
      the range better in the bulk at the cost of clipping the odd outlier.

    The two are NOT bit-compatible: they disagree on roughly 12% of fp4 block
    scales and 0.3% of fp8 ones, so a tensor quantized with one must be
    dequantized against the same one. DEQUANT_SCALE_ROUNDING_MODE is ignored
    when POW2_SCALE is set.
    """
    is_fp8: tl.constexpr = (
        mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    )
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // 32

    if POW2_SCALE:
        # Padding lanes are zeroed rather than set to -1: zero is neutral for the
        # group amax and is also what the tile stores for them either way.
        masked = tl.where(valid_src_mask, src_tensor.to(tl.float32), 0.0)
        if is_fp8:
            LOG2_DTYPE_MAX: tl.constexpr = 8 if mx_tensor_dtype == tl.float8e4nv else 15
            grouped = tl.reshape(
                masked, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32]
            )
            scale_e8m0, quant_scale = _mxfp8_quant_op(grouped, 2, LOG2_DTYPE_MAX)
            out_tensor = (
                (grouped * quant_scale)
                .reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
                .to(mx_tensor_dtype)
            )
            dequant_scale_exponent = scale_e8m0.reshape(
                [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE]
            )
        else:
            out_tensor, dequant_scale_exponent = _mxfp4_quant_op(
                masked, BLOCK_SIZE_QUANT_DIM, BLOCK_SIZE_OUT_DIM, 32
            )
        return out_tensor, dequant_scale_exponent

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
    dequant_scale = max_val / _get_max_quant_val(mx_tensor_dtype)
    if DEQUANT_SCALE_ROUNDING_MODE == 0:
        # DequantScaleRoundingMode.ROUND_UP
        # compute 2 ** ceil(log2(dequant_scale))
        # Adding 0x007FFFFF adds exponent by 1 unless mantissa is all zeros
        # A corner case: exponent is 0xFF that will overflow but that's already
        # NaN so assume we don't care.
        dequant_scale_exponent = (
            dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF
        ) & 0x7F800000
    else:
        # DequantScaleRoundingMode.ROUND_DOWN
        # compute 2 ** floor(log2(dequant_scale))
        assert DEQUANT_SCALE_ROUNDING_MODE == 1
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


_downcast_to_mxfp_repr = make_kernel_repr(
    "_downcast_to_mxfp",
    [
        "BLOCK_SIZE_OUT_DIM",
        "BLOCK_SIZE_QUANT_DIM",
        "DEQUANT_SCALE_ROUNDING_MODE",
        "POW2_SCALE",
    ],
)


@triton.jit(repr=_downcast_to_mxfp_repr)
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
    POW2_SCALE: tl.constexpr,
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

    out_tensor, scale_tensor = _compute_mx_quant_and_scale(
        src_tensor,
        full_mask_src,
        mx_tensor_dtype,
        DEQUANT_SCALE_ROUNDING_MODE,
        POW2_SCALE,
    )

    tl.store(mx_scale_ptr + mx_scale_offsets, scale_tensor, mask=full_scale_mask)
    tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)


_smoothquant_fuse_quant_kernel_repr = make_kernel_repr(
    "_smoothquant_fuse_quant_kernel",
    [
        "BLOCK_M",
        "BLOCK_K",
    ],
)


@triton.jit(repr=_smoothquant_fuse_quant_kernel_repr)
def _smoothquant_fuse_quant_kernel(
    # Input tensors
    X_ptr,  # bf16 input [M, K]
    stride_x_m,
    stride_x_k,
    SmoothScale_ptr,  # fp32 smooth scale [K] - one per column
    # Output tensors
    Y_ptr,  # int8 output [M, K]
    stride_y_m,
    stride_y_k,
    RowScale_ptr,  # fp32 per-row scale [M]
    stride_row_scale,
    # Dimensions
    M,
    K,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused smoothquant quantization kernel.

    Performs:
    1. Multiply bf16 input by fp32 smooth scale (per column)
    2. Compute per-row max absolute value
    3. Calculate per-row quantization scale
    4. Quantize to int8

    Args:
        X_ptr: Input tensor in bf16 [M, K]
        SmoothScale_ptr: Smooth scale vector in fp32 [K]
        Y_ptr: Output tensor in int8 [M, K]
        RowScale_ptr: Per-row quantization scales in fp32 [M]
    """
    # Program ID
    pid_m = tl.program_id(0)

    # Row offset
    row_start = pid_m * BLOCK_M

    # Initialize per-row max values
    row_max = tl.zeros([BLOCK_M], dtype=tl.float32)

    # First pass: apply smooth scale and compute row max
    for k_start in range(0, K, BLOCK_K):
        # Compute offsets
        offs_m = row_start + tl.arange(0, BLOCK_M)
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Create masks
        mask_m = offs_m < M
        mask_k = offs_k < K
        mask = mask_m[:, None] & mask_k[None, :]

        # Load input values
        x_ptrs = X_ptr + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        # Load smooth scales for this K block
        smooth_scale = tl.load(SmoothScale_ptr + offs_k, mask=mask_k, other=1.0)

        # Apply smooth scale: x_smooth = x * smooth_scale (broadcast over M)
        x_fp32 = x.to(tl.float32) * smooth_scale[None, :]

        # Update row max with absolute values
        abs_x = tl.abs(x_fp32)
        # Max across K dimension for each row
        block_max = tl.max(abs_x, axis=1)
        row_max = tl.maximum(row_max, block_max)

    # Compute per-row quantization scale
    # scale = max_val / 127.0 (int8 max)
    INT8_MAX: tl.constexpr = 127.0
    eps = 1e-12  # Small epsilon to avoid division by zero
    row_scale = row_max / INT8_MAX + eps

    # Store row scales
    offs_m = row_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    scale_ptrs = RowScale_ptr + offs_m * stride_row_scale
    tl.store(scale_ptrs, row_scale, mask=mask_m)

    # Second pass: quantize values using computed scales
    for k_start in range(0, K, BLOCK_K):
        # Compute offsets
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Create masks
        mask_k = offs_k < K
        mask = mask_m[:, None] & mask_k[None, :]

        # Load input values again
        x_ptrs = X_ptr + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        # Load smooth scales
        smooth_scale = tl.load(SmoothScale_ptr + offs_k, mask=mask_k, other=1.0)

        # Apply smooth scale
        x_fp32 = x.to(tl.float32) * smooth_scale[None, :]

        # Quantize: x_int8 = round(x_fp32 / row_scale)
        x_scaled = x_fp32 / row_scale[:, None]

        # Clamp to int8 range and round
        x_clamped = tl.minimum(tl.maximum(x_scaled, -127.0), 127.0)
        # Use standard rounding: add 0.5 and truncate for positive, subtract 0.5 for negative
        x_rounded = tl.where(
            x_clamped >= 0,
            (x_clamped + 0.5).to(tl.int32),
            (x_clamped - 0.5).to(tl.int32),
        )
        x_int8 = x_rounded.to(tl.int8)

        # Store quantized values
        y_ptrs = Y_ptr + offs_m[:, None] * stride_y_m + offs_k[None, :] * stride_y_k
        tl.store(y_ptrs, x_int8, mask=mask)


_smoothquant_fuse_quant_kernel_single_pass_repr = make_kernel_repr(
    "_smoothquant_fuse_quant_kernel_single_pass",
    [
        "BLOCK_M",
        "BLOCK_K",
    ],
)


@triton.jit(repr=_smoothquant_fuse_quant_kernel_single_pass_repr)
def _smoothquant_fuse_quant_kernel_single_pass(
    # Input tensors
    X_ptr,  # bf16 input [M, K]
    stride_x_m,
    stride_x_k,
    SmoothScale_ptr,  # fp32 smooth scale [K] - one per column
    # Output tensors
    Y_ptr,  # int8 output [M, K]
    stride_y_m,
    stride_y_k,
    RowScale_ptr,  # fp32 per-row scale [M]
    stride_row_scale,
    # Dimensions
    M,
    K,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Single-pass smoothquant quantization kernel for small K dimensions.

    Loads entire rows into registers, applies smooth scaling, computes row max,
    and quantizes in a single pass. More efficient when K fits in shared memory.
    """
    pid_m = tl.program_id(0)
    row_start = pid_m * BLOCK_M

    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_k[None, :]

    # Load entire block
    x_ptrs = X_ptr + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Load smooth scales
    smooth_scale = tl.load(SmoothScale_ptr + offs_k, mask=mask_k, other=1.0)

    # Apply smooth scale
    x_fp32 = x.to(tl.float32) * smooth_scale[None, :]

    # Compute per-row max
    abs_x = tl.abs(x_fp32)
    # Set masked values to 0 for max computation
    abs_x = tl.where(mask, abs_x, 0.0)
    row_max = tl.max(abs_x, axis=1)

    # Compute scale
    INT8_MAX: tl.constexpr = 127.0
    eps = 1e-12
    row_scale = row_max / INT8_MAX + eps

    # Store row scales
    scale_ptrs = RowScale_ptr + offs_m * stride_row_scale
    tl.store(scale_ptrs, row_scale, mask=mask_m)

    # Quantize
    x_scaled = x_fp32 / row_scale[:, None]
    x_clamped = tl.minimum(tl.maximum(x_scaled, -127.0), 127.0)
    # Use standard rounding: add 0.5 and truncate for positive, subtract 0.5 for negative
    x_rounded = tl.where(
        x_clamped >= 0, (x_clamped + 0.5).to(tl.int32), (x_clamped - 0.5).to(tl.int32)
    )
    x_int8 = x_rounded.to(tl.int8)

    # Store quantized values
    y_ptrs = Y_ptr + offs_m[:, None] * stride_y_m + offs_k[None, :] * stride_y_k
    tl.store(y_ptrs, x_int8, mask=mask)


@triton.jit
def _dequant_int8_to_fp32_kernel(
    # Input tensors
    X_ptr,  # int8 input [M, N]
    stride_x_m,
    stride_x_n,
    Scale_ptr,  # fp32 scale [M]
    stride_scale,
    # Output tensor
    Y_ptr,  # fp32 output [M, N]
    stride_y_m,
    stride_y_n,
    # Dimensions
    M,
    N,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Dequantize int8 tensor to fp32 using per-row scales.

    Y = X.to(fp32) * scale
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = col_start + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Load int8 values
    x_ptrs = X_ptr + offs_m[:, None] * stride_x_m + offs_n[None, :] * stride_x_n
    x = tl.load(x_ptrs, mask=mask, other=0)

    # Load scales
    scale = tl.load(Scale_ptr + offs_m * stride_scale, mask=mask_m, other=1.0)

    # Dequantize
    y = x.to(tl.float32) * scale[:, None]

    # Store result
    y_ptrs = Y_ptr + offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
    tl.store(y_ptrs, y, mask=mask)
