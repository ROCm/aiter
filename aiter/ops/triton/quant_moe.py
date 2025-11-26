from enum import Enum
import triton
import torch
from aiter.ops.triton._triton_kernels.quant_moe import (
    _downcast_to_static_fp8,
    _downcast_to_mxfp,
    _upcast_from_mxfp,
)


def downcast_to_static_fp8_3d(x: torch.Tensor, scale: torch.Tensor):
    assert x.ndim == 3
    E, M, N = x.shape

    x2d = x.reshape(E * M, N).contiguous()

    y2d = downcast_to_static_fp8(x2d, scale)
    y3d = y2d.reshape(E, M, N)
    return y3d


def downcast_to_static_fp8(x: torch.Tensor, scale: torch.Tensor):
    M, N = x.shape
    y = torch.empty((M, N), dtype=torch.float8_e4m3fn, device="cuda")

    BLOCK_M = min(triton.next_power_of_2(M), 128)
    if M <= 4096:
        BLOCK_N = 32
    else:
        BLOCK_N = 64
    grid_m = triton.cdiv(x.shape[0], BLOCK_M)
    grid_n = triton.cdiv(x.shape[1], BLOCK_N)

    _downcast_to_static_fp8[(grid_m, grid_n)](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        scale,
        M,
        N,
        BLOCK_M,
        BLOCK_N,
        num_warps=8,
    )

    return y


class DequantScaleRoundingMode(Enum):
    ROUND_UP = 0
    ROUND_DOWN = 1


def downcast_to_mxfp(
    src_tensor: torch.Tensor,
    out_quant_type: torch.dtype,
    axis: int,
    DEQUANT_SCALE_ROUNDING_MODE: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_UP,
):
    """
    Convert the src weights to mx format. The src weight is quantized along the axis dimension.

    If weight_quant_type is torch.uint8, we output mxfp4 where two e2m1 values are packed into a single byte.
    Note that this means the k_dim of the tensor will be half of the logical k_dim.

    If weight_quant_type is torch.float8_e4m3fn or torch.float8_e5m2, we output mxfp8 with the float8s are stored
    in their respective formats.
    """
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
