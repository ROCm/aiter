import torch
import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.fused_mxfp4_quant import (
    _rmsmorm_op,
    _fused_rms_mxfp4_quant_kernel,
    _fused_flatten_mxfp4_quant,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_rms_mxfp4_quant(
    inp1,
    inp1_weight,
    inp1_epsilon,
    inp2=None,
    inp2_weight=None,
    inp2_epsilon=0.0,
    res1=None,
):
    """
    This op contains several steps:
        1. if res1 is not None, inp1 = inp1 + res1, and store inp1 to out_res1
        2. perform RMS norm along the last dimenion for inp1
        3. if inp2 is not None, perform RMS norm along the last dimenion for inp2
        4. perform mxfp4 quantization for inp1 only

    Key parameters:
    - x: Matrix X with shape (M, N1, N2).

    Returns:
    - out1_fp4: The output matrix with shape (M, N1 // 2).
    - out1_bs: The output matrix with shape (M, cdiv(N1, MXFP4_QUANT_BLOCK_SIZE)).
    - out2: The output matrix with shape (M, N2).
    - out_res1: The output matrix with shape (M, N1).

        if both inp2 and res1 provided, return (out1_fp4, out1_bs), out2, out_res1
        if inp2 provided, return (out1_fp4, out1_bs), out2
        if res1 provided, return (out1_fp4, out1_bs), out_res1
        if both inp2 and res1 not provided, return (out1_fp4, out1_bs)
    """
    _LOGGER.info(f"FUSED_RMS_MXFP4_QUANT: inp1={tuple(inp1.shape)}")
    MXFP4_QUANT_BLOCK_SIZE = 32
    M, N1 = inp1.shape
    BLOCK_SIZE = max(triton.next_power_of_2(N1), MXFP4_QUANT_BLOCK_SIZE)
    if inp2 is not None:
        N2 = inp2.shape[1]
        BLOCK_SIZE = max(triton.next_power_of_2(N2), BLOCK_SIZE)
    else:
        N2 = 0
    # as we merge 2 fp4s to 1 uint8
    assert N1 % 2 == 0

    BLOCK_SIZE = max(BLOCK_SIZE, MXFP4_QUANT_BLOCK_SIZE)
    out1_fp4 = torch.empty((M, N1 // 2), dtype=torch.uint8, device=inp1.device)
    out1_bs = torch.empty(
        ((N1 + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE, M),
        dtype=torch.uint8,
        device=inp1.device,
    ).T

    out_res1 = None
    res1_row_stride = 0
    out_res1_row_stride = 0
    if res1 is not None:
        out_res1 = torch.empty((M, N1), dtype=inp1.dtype, device=inp1.device)
        res1_row_stride = res1.stride(0)
        out_res1_row_stride = out_res1.stride(0)

    out2 = None
    out2_row_stride = 0
    inp2_row_stride = 0
    if inp2 is not None:
        out2 = torch.empty((M, N2), dtype=inp1.dtype, device=inp1.device)
        inp2_row_stride = inp2.stride(0)
        out2_row_stride = out2.stride(0)

    _fused_rms_mxfp4_quant_kernel[(M,)](
        inp1,
        inp1_weight,
        inp2,
        inp2_weight,
        res1,
        out1_fp4,
        out1_bs,
        out2,
        out_res1,
        inp1_epsilon,
        inp2_epsilon,
        M,
        N1,
        N2,
        inp1.stride(0),
        inp2_row_stride,
        res1_row_stride,
        out1_fp4.stride(0),
        *out1_bs.stride(),
        out2_row_stride,
        out_res1_row_stride,
        BLOCK_SIZE=BLOCK_SIZE,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        SKIP_SECOND_INPUT=(inp2 is None),
        FIRST_INPUT_RES=(res1 is not None),
    )
    if res1 is not None:
        if inp2 is None:
            return (out1_fp4, out1_bs), out_res1
        else:
            return (out1_fp4, out1_bs), out2, out_res1
    else:
        if inp2 is None:
            return (out1_fp4, out1_bs)
        else:
            return (out1_fp4, out1_bs), out2


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
