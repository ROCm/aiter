from enum import Enum
import triton
import torch
from aiter.ops.triton._triton_kernels.moe.quant_moe import (
    _downcast_to_static_fp8,
    _downcast_to_mxfp,
    _upcast_from_mxfp,
    _smoothquant_fuse_quant_kernel,
    _smoothquant_fuse_quant_kernel_single_pass,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch


def downcast_to_static_fp8_3d(x: torch.Tensor, scale: torch.Tensor):
    assert x.ndim == 3
    E, M, N = x.shape

    x2d = x.reshape(E * M, N).contiguous()

    y2d = downcast_to_static_fp8(x2d, scale)
    y3d = y2d.reshape(E, M, N)
    return y3d


def downcast_to_static_fp8(x: torch.Tensor, scale: torch.Tensor):
    M, N = x.shape
    if get_arch() != "gfx942":
        dtype = torch.float8_e4m3fn
    else:
        dtype = torch.float8_e4m3fnuz
    y = torch.empty((M, N), dtype=dtype, device="cuda")

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
    is_fp8 = out_quant_type in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
    )
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


def dequant_x_blockscale(x, x_scales, per_row_x_scale, group_shape):
    assert x_scales is not None
    group_shape_m, _, group_shape_k = group_shape
    M, K = x.shape

    K_blocks = (K + group_shape_k - 1) // group_shape_k
    if per_row_x_scale:
        assert x_scales.shape == (M, K_blocks)
        K_pad = K_blocks * group_shape_k
        if K_pad != K:
            x_pad = x.new_zeros((M, K_pad))
            x_pad[:, :K] = x
            x = x_pad

        x = x.to(torch.float32).view(M, K_blocks, group_shape_k) * x_scales.to(
            torch.float32
        ).view(M, K_blocks, 1)
        x = x.view(M, K_pad)[:, :K]
    else:
        M_blocks = (M + group_shape_m - 1) // group_shape_m
        assert x_scales.shape == (M_blocks, K_blocks)
        M_pad = M_blocks * group_shape_m
        K_pad = K_blocks * group_shape_k
        if M_pad != M or K_pad != K:
            x_pad = x.new_zeros((M_pad, K_pad))
            x_pad[:M, :K] = x
            x = x_pad

        x = x.to(torch.float32).view(M_blocks, group_shape_m, K_blocks, group_shape_k)
        scales = x_scales.to(torch.float32).view(M_blocks, 1, K_blocks, 1)
        x = x * scales
        x = x.view(M_pad, K_pad)[:M, :K]
    return x


def dequant_w_blockscale(w, w_scales, group_shape):
    assert w_scales is not None
    _, group_shape_n, group_shape_k = group_shape
    E, K, N = w.shape

    K_blocks = (K + group_shape_k - 1) // group_shape_k
    N_blocks = (N + group_shape_n - 1) // group_shape_n

    assert w_scales.shape == (E, K_blocks, N_blocks)

    K_pad = K_blocks * group_shape_k
    N_pad = N_blocks * group_shape_n
    if K_pad != K or N_pad != N:
        w_pad = w.new_zeros((E, K_pad, N_pad))
        w_pad[:, :K, :N] = w
        w = w_pad
    w = w.to(torch.float32).view(E, K_blocks, group_shape_k, N_blocks, group_shape_n)
    scales = w_scales.to(torch.float32).view(E, K_blocks, 1, N_blocks, 1)
    w = w * scales
    w = w.view(E, K_pad, N_pad)[:, :K, :N]
    return w


def smoothquant_quantize(
    x: torch.Tensor,
    smooth_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply smoothquant quantization to convert bf16/fp16 tensor to int8.

    Args:
        x: Input tensor in bf16/fp16 [M, K]
        smooth_scale: Per-column smooth scale in fp32 [K]

    Returns:
        x_int8: Quantized int8 tensor [M, K]
        x_scale: Per-row quantization scale in fp32 [M]

    The operation performs:
    1. x_smooth = x * smooth_scale (per column)
    2. row_scale = max(abs(x_smooth), dim=1) / 127
    3. x_int8 = round(x_smooth / row_scale)
    """
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    assert smooth_scale.ndim == 1, f"Expected 1D smooth_scale, got {smooth_scale.ndim}D"
    assert (
        x.shape[1] == smooth_scale.shape[0]
    ), f"Dimension mismatch: x.shape[1]={x.shape[1]}, smooth_scale.shape[0]={smooth_scale.shape[0]}"

    M, K = x.shape
    device = x.device

    x_int8 = torch.empty((M, K), dtype=torch.int8, device=device)
    x_scale = torch.empty((M,), dtype=torch.float32, device=device)

    smooth_scale = smooth_scale.to(torch.float32).contiguous()

    MAX_SINGLE_PASS_K = 1024
    BLOCK_M = min(triton.next_power_of_2(M), 32)

    if K <= MAX_SINGLE_PASS_K:
        # Single pass: load entire row at once
        BLOCK_K = triton.next_power_of_2(K)
        grid = (triton.cdiv(M, BLOCK_M),)

        _smoothquant_fuse_quant_kernel_single_pass[grid](
            x,
            x.stride(0),
            x.stride(1),
            smooth_scale,
            x_int8,
            x_int8.stride(0),
            x_int8.stride(1),
            x_scale,
            1,
            M,
            K,
            BLOCK_M,
            BLOCK_K,
            num_warps=4,
        )
    else:
        BLOCK_K = 256
        grid = (triton.cdiv(M, BLOCK_M),)
        _smoothquant_fuse_quant_kernel[grid](
            x,
            x.stride(0),
            x.stride(1),
            smooth_scale,
            x_int8,
            x_int8.stride(0),
            x_int8.stride(1),
            x_scale,
            1,
            M,
            K,
            BLOCK_M,
            BLOCK_K,
            num_warps=4,
        )

    return x_int8, x_scale


def quantize_weights_int8(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weights to int8 with per-output-channel scaling.

    Args:
        w: Weight tensor in bf16/fp16/fp32 [E, K, N] or [K, N]

    Returns:
        w_int8: Quantized int8 weights (contiguous)
        w_scale: Per-output-channel scale [E, N] or [N] (contiguous)
    """
    if w.ndim == 2:
        # [K, N] -> [1, K, N]
        w = w.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    w_fp32 = w.to(torch.float32)
    w_abs_max = w_fp32.abs().max(dim=1).values
    INT8_MAX = 127.0
    w_scale = w_abs_max / INT8_MAX + 1e-12
    w_scaled = w_fp32 / w_scale[:, None, :]
    w_int8 = w_scaled.round().clamp(-127, 127).to(torch.int8)

    # Layout [E, K, N] with N contiguous
    w_int8 = w_int8.contiguous()
    w_scale = w_scale.contiguous()

    if squeeze_output:
        w_int8 = w_int8.squeeze(0)
        w_scale = w_scale.squeeze(0)

    return w_int8, w_scale
