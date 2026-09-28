"""MX dequantization used to build bf16 references in the MoE GEMM tests.

``downcast_to_mxfp`` throws away precision, so a bf16 torch reference fed the
original tensor would charge the kernel for quantization error that the kernel
did not cause. These tests instead round-trip the quantized tensor back to bf16
and hand the reference exactly the values the kernel will read, which is what
lets them keep tight tolerances.

This is test scaffolding, not an inference path: nothing under ``aiter/`` calls
it. It lived in ``aiter/ops/triton/moe/quant_moe.py`` until it was moved here.
"""

import torch
import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_upcast_from_mxfp_repr = make_kernel_repr(
    "_upcast_from_mxfp",
    [
        "BLOCK_SIZE_OUT_DIM",
        "BLOCK_SIZE_QUANT_DIM",
    ],
)


@triton.jit(repr=_upcast_from_mxfp_repr)
def _upcast_from_mxfp(
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
    tl.static_assert(dst_dtype == tl.float16 or dst_dtype == tl.bfloat16)
    tl.static_assert(
        mx_tensor_dtype == tl.uint8
        or (
            (mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5)
            or mx_tensor_dtype == dst_dtype
        ),
        "mx_tensor_ptr must be uint8 or float8 or dst_dtype",
    )
    tl.static_assert(
        mx_scale_ptr.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8"
    )

    # Determine if we are dealing with fp8 types.
    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8
    is_fp8: tl.constexpr = (
        mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    )
    K_DIVISOR: tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 32
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

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

    mask_src_quant = start_mxt_quant + offs_src_quant < tl.cdiv(quant_dim, K_DIVISOR)
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

    # Upcast the scale to the destination type.
    if dst_dtype == tl.bfloat16:
        dst_scale = (scale.to(tl.uint16) << 7).to(dst_dtype, bitcast=True)
    else:
        tl.static_assert(dst_dtype == tl.float16)
        dst_scale = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        dst_scale = dst_scale.to(tl.float16)

    # Now upcast the tensor.
    if is_fp8:
        dst_tensor = tensor.to(dst_dtype)
        if tensor.dtype == tl.float8e5:
            from_e_bits: tl.constexpr = 5
            from_m_bits: tl.constexpr = 2
            to_e_bits: tl.constexpr = 8 if dst_dtype == tl.bfloat16 else 5
            to_m_bits: tl.constexpr = 7 if dst_dtype == tl.bfloat16 else 10

            # Preserve infs and nans. FIXME Fp8E5M2_to_Bf16 doesn't preserve them!
            non_finite_mask_src: tl.constexpr = ((1 << from_e_bits) - 1) << from_m_bits
            non_finite_mask_dst: tl.constexpr = ((1 << to_e_bits) - 1) << to_m_bits
            dst_tensor = tl.where(
                (tensor.to(tl.uint8, bitcast=True) & non_finite_mask_src)
                == non_finite_mask_src,
                (dst_tensor.to(tl.uint16, bitcast=True) | non_finite_mask_dst).to(
                    dst_dtype, bitcast=True
                ),
                dst_tensor,
            )
    else:
        assert is_fp4
        dst_bias: tl.constexpr = 127 if dst_dtype == tl.bfloat16 else 15
        dst_0p5: tl.constexpr = 16128 if dst_dtype == tl.bfloat16 else 0x3800
        dst_m_bits: tl.constexpr = 7 if dst_dtype == tl.bfloat16 else 10
        # e2m1
        em0 = tensor & 0x07
        em1 = tensor & 0x70
        x0 = (em0.to(tl.uint16) << (dst_m_bits - 1)) | (
            (tensor & 0x08).to(tl.uint16) << 12
        )
        x1 = (em1.to(tl.uint16) << (dst_m_bits - 5)) | (
            (tensor & 0x80).to(tl.uint16) << 8
        )
        # Three cases:
        # 1) x is normal and non-zero: Correct bias
        x0 = tl.where((em0 & 0x06) != 0, x0 + ((dst_bias - 1) << dst_m_bits), x0)
        x1 = tl.where((em1 & 0x60) != 0, x1 + ((dst_bias - 1) << dst_m_bits), x1)
        # 2) x is subnormal (x == 0bs001 where s is the sign): Map to +-0.5 in the dst type
        x0 = tl.where(em0 == 0x01, dst_0p5 | (x0 & 0x8000), x0)
        x1 = tl.where(em1 == 0x10, dst_0p5 | (x1 & 0x8000), x1)
        # 3) x is zero, do nothing
        dst_tensor = tl.interleave(x0, x1).to(dst_dtype, bitcast=True)

    # Reshape for proper broadcasting: the scale was stored with a 32-sized "inner" grouping.
    dst_tensor = dst_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32])
    dst_scale = dst_scale.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1])
    scale = scale.reshape(dst_scale.shape)

    out_tensor = dst_tensor * dst_scale
    # Correct any NaNs encoded via the scale.
    out_tensor = tl.where(scale == 0xFF, float("nan"), out_tensor)
    out_tensor = out_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    tl.store(out_ptr + out_offsets, out_tensor, mask=full_mask_out)


def upcast_from_mxfp(
    tensor: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, axis: int
):
    """
    Upcasts an mxfp (packed) weight tensor back to float16 or bfloat16.

    The function assumes that the tensors were quantized along the given axis.
    It permutes the tensor so that the quantized axis is last, reshapes to 2D,
    launches the Triton upcast kernel, and then unpermutes back to the original order.
    """
    ndim = tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim
    assert tensor.ndim == scale.ndim, (
        f"Weight and scale must have the same number of dimensions. "
        f"Got {tensor.ndim=} and {scale.ndim=}"
    )
    # dtype checks
    assert tensor.dtype in {
        torch.uint8,
        torch.float8_e5m2,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    }, f"Invalid tensor dtype {tensor.dtype=}"
    assert scale.dtype == torch.uint8, f"Invalid scale dtype {scale.dtype=}"
    assert dtype in (torch.float16, torch.bfloat16), f"Invalid output dtype {dtype=}"
    # upcast
    logical_quant_dim = tensor.shape[axis] * (2 if tensor.dtype == torch.uint8 else 1)
    tensor = tensor.transpose(axis, tensor.ndim - 1).contiguous()
    scale = scale.transpose(axis, scale.ndim - 1).contiguous()
    out = torch.empty(
        (*tensor.shape[:-1], logical_quant_dim), dtype=dtype, device=tensor.device
    )
    reshaped_out = out.view(-1, out.shape[-1])
    reshaped_tensor = tensor.view(-1, tensor.shape[-1])
    reshaped_scale = scale.view(-1, scale.shape[-1])
    BLOCK_OUT_DIM = 128
    BLOCK_QUANT_DIM = 32
    blocks_out_dim = triton.cdiv(reshaped_out.shape[0], BLOCK_OUT_DIM)
    blocks_quant_dim = triton.cdiv(reshaped_out.shape[1], BLOCK_QUANT_DIM)
    _upcast_from_mxfp[(blocks_out_dim, blocks_quant_dim)](
        reshaped_out,
        *reshaped_out.stride(),
        reshaped_scale,
        *reshaped_scale.stride(),
        reshaped_tensor,
        *reshaped_tensor.stride(),
        *reshaped_out.shape,
        BLOCK_OUT_DIM,
        BLOCK_QUANT_DIM,
        num_warps=8,
    )
    out = out.transpose(axis, scale.ndim - 1).contiguous()
    return out
