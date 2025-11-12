import torch
import triton
import triton.language as tl
from typing import Optional

from aiter.ops.triton._triton_kernels.fused_mxfp4_quant import (
    _rmsmorm_op,
    _fused_rms_mxfp4_quant_kernel,
    _fused_flatten_mxfp4_quant,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_rms_mxfp4_quant(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: float = 0.0,
    res1: Optional[torch.Tensor] = None,
    shuffle: Optional[bool] = False,
    scale_shuffle_padding: Optional[bool] = False,
):
    """
    This op contains several steps:
        1. if res1 is not None, x1 = x1 + res1, and store x1 to out_res1
        2. perform RMS norm along the last dimenion for x1
        3. if x2 is not None, perform RMS norm along the last dimenion for x2
        4. perform mxfp4 quantization for x1 only

    Key parameters:
    - x: Matrix X with shape (M, N1, N2).

    Returns:
    - out1_fp4: The output matrix with shape (M, N1 // 2).
    - out1_bs: The output matrix with shape (M, cdiv(N1, MXFP4_QUANT_BLOCK_SIZE)).
    - out2: The output matrix with shape (M, N2).
    - out_res1: The output matrix with shape (M, N1).

        if both x2 and res1 provided, return (out1_fp4, out1_bs), out2, out_res1
        if x2 provided, return (out1_fp4, out1_bs), out2
        if res1 provided, return (out1_fp4, out1_bs), out_res1
        if both x2 and res1 not provided, return (out1_fp4, out1_bs)
    """
    _LOGGER.info(f"FUSED_RMS_MXFP4_QUANT: inp1={tuple(x1.shape)}")

    MXFP4_QUANT_BLOCK_SIZE = 32
    M, N1 = x1.shape
    BLOCK_SIZE_N = max(triton.next_power_of_2(N1), MXFP4_QUANT_BLOCK_SIZE)
    BLOCK_SIZE_N2 = 1
    if x2 is not None:
        N2 = x2.shape[1]
        BLOCK_SIZE_N2 = triton.next_power_of_2(N2)
    else:
        N2 = 0
    # as we merge 2 fp4s to 1 uint8
    assert N1 % 2 == 0
    BLOCK_SIZE_M = 1
    # BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = max(BLOCK_SIZE_N, MXFP4_QUANT_BLOCK_SIZE)
    out1_fp4 = torch.empty((M, N1 // 2), dtype=torch.uint8, device=x1.device)
    SCALE_N_valid = triton.cdiv(N1, MXFP4_QUANT_BLOCK_SIZE)
    use_scale_shuffle_padding = shuffle or scale_shuffle_padding
    if use_scale_shuffle_padding:
        SCALE_M = triton.cdiv(M, 256) * 256
        SCALE_N = triton.cdiv(SCALE_N_valid, 8) * 8
        # BLOCK_SIZE_M = triton.cdiv(BLOCK_SIZE_M, 32) * 32
        BLOCK_SIZE_N = triton.cdiv(BLOCK_SIZE_N, 32) * 32
    else:
        SCALE_M = M
        SCALE_N = SCALE_N_valid
    out1_bs = torch.empty(
        (SCALE_M, SCALE_N),
        dtype=torch.uint8,
        device=x1.device,
    )

    out_res1 = None
    res1_stride_m = 0
    out_res1_stride_m = 0
    if res1 is not None:
        out_res1 = torch.empty((M, N1), dtype=x1.dtype, device=x1.device)
        res1_stride_m = res1.stride(0)
        out_res1_stride_m = out_res1.stride(0)

    out2 = None
    out2_stride_m = 0
    x2_stride_m = 0
    if x2 is not None:
        out2 = torch.empty((M, N2), dtype=x1.dtype, device=x1.device)
        x2_stride_m = x2.stride(0)
        out2_stride_m = out2.stride(0)

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * (2 if (x2 is not None) else 1),)
    _fused_rms_mxfp4_quant_kernel[grid](
        x1,
        x1_weight,
        x2,
        x2_weight,
        res1,
        out1_fp4,
        out1_bs,
        out2,
        out_res1,
        x1_epsilon,
        x2_epsilon,
        M,
        N1,
        N2,
        x1.stride(0),
        x2_stride_m,
        res1_stride_m,
        out1_fp4.stride(0),
        *out1_bs.stride(),
        out2_stride_m,
        out_res1_stride_m,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_N2=BLOCK_SIZE_N2,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        HAS_SECOND_INPUT=(x2 is not None),
        FIRST_INPUT_RES=(res1 is not None),
        SCALE_N=SCALE_N_valid,
        SCALE_M_PAD=(SCALE_M if use_scale_shuffle_padding else 1),
        SCALE_N_PAD=SCALE_N,
        SHUFFLE=shuffle,
        SHUFFLE_PAD=use_scale_shuffle_padding,
    )

    return (out1_fp4, out1_bs), out2, out_res1


def fused_flatten_mxfp4_quant(
    x: torch.Tensor,
):
    """
    Flatten the last two dimension of x and perform mxfp4 quantization along the last dimension

    Key parameters:
    - x: Matrix X with shape (M, N1, N2).

    Returns:
    - out: The output matrix with shape (M, (N1 * N2) // 2).
    - out_block_scales: The output matrix with shape (M, cdiv(N1 * N2, MXFP4_QUANT_BLOCK_SIZE)).
    """
    _LOGGER.info(f"FUSED_FLATTEN_MXFP4_QUANT: x={tuple(x.shape)}")
    M, N1, N2 = x.shape

    MXFP4_QUANT_BLOCK_SIZE = 32
    BLOCK_SIZE_N2 = max(triton.next_power_of_2(N2), MXFP4_QUANT_BLOCK_SIZE)
    N = N1 * N2
    out = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    out_block_scales = torch.empty(
        (triton.cdiv(N, MXFP4_QUANT_BLOCK_SIZE), M),
        dtype=torch.uint8,
        device=x.device,
    ).T

    grid = (
        M,
        N1,
    )
    _fused_flatten_mxfp4_quant[grid](
        x,
        out,
        out_block_scales,
        *x.stride(),
        *out.stride(),
        *out_block_scales.stride(),
        N2,
        BLOCK_SIZE_N2,
        MXFP4_QUANT_BLOCK_SIZE,
    )

    return out, out_block_scales
