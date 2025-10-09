import torch
import triton
import aiter
from aiter.ops.triton._triton_kernels.fused_fp8_quant import (
    _fused_rms_fp8_group_quant_kernel,
    _fused_flatten_fp8_group_quant_kernel,
)

fp8_dtype = aiter.dtypes.fp8


def fused_rms_fp8_group_quant(
    inp1,
    inp1_weight,
    inp1_epsilon,
    inp2=None,
    inp2_weight=None,
    inp2_epsilon=None,
    group_size=128,
    dtype_quant=fp8_dtype,
    res1=None,
    output_unquantized_inp1=False,
):
    """
    This op contains several steps:
        1. if res1 is not None, inp1 = inp1 + res1, and store inp1 to out_res1
        2. perform RMS norm along the last dimenion for inp1
        3. if inp2 is not None, perform RMS norm along the last dimenion for inp2
        4. perform fp8 quantization for inp1 only

    Key parameters:
    - x: Matrix X with shape (M, N1, N2).

    Returns:
    - out1_fp8: The output matrix with shape (M, N1).
    - out1_bs: The output matrix with shape (M, cdiv(N1, group_size)).
    - out1: The output matrix with shape (M, N1).
    - out2: The output matrix with shape (M, N2).
    - out_res1: The output matrix with shape (M, N1).
    - out1: The output matrix with shape (M, N1).
    """

    M, N1 = inp1.shape
    BLOCK_SIZE_N = max(triton.next_power_of_2(N1), group_size)
    if inp2 is not None:
        M2, N2 = inp2.shape
        BLOCK_SIZE_N = max(triton.next_power_of_2(N2), BLOCK_SIZE_N)
        assert (
            M == M2
        ), "The leading dimension should be identical between inp1 and inp2"
    else:
        N2 = 0
    out1_fp8 = torch.empty((M, N1), dtype=dtype_quant, device=inp1.device)
    out1_bs = torch.empty(
        (M, (N1 + group_size - 1) // group_size),
        dtype=torch.float32,
        device=inp1.device,
    )

    out2 = None
    out2_row_stride = 0
    out2_col_stride = 0
    inp2_row_stride = 0
    inp2_col_stride = 0
    if inp2 is not None:
        out2 = torch.empty((M, N2), dtype=inp1.dtype, device=inp1.device)
        inp2_row_stride = inp2.stride(0)
        inp2_col_stride = inp2.stride(1)
        out2_row_stride = out2.stride(0)
        out2_col_stride = out2.stride(1)

    out1 = None
    out1_row_stride = 0
    out1_col_stride = 0
    if output_unquantized_inp1:
        out1 = torch.empty((M, N1), dtype=inp1.dtype, device=inp1.device)
        out1_row_stride = out1.stride(0)
        out1_col_stride = out1.stride(1)

    BLOCK_SIZE_N = max(BLOCK_SIZE_N, group_size)
    out_res1 = None
    res1_row_stride = 0
    res1_col_stride = 0
    out_res1_row_stride = 0
    out_res1_col_stride = 0
    if res1 is not None:
        Mr, Nr = res1.shape
        assert (
            M == Mr and N1 == Nr
        ), "The shape should be identical between inp1 and res1"
        out_res1 = torch.empty((M, N1), dtype=inp1.dtype, device=inp1.device)
        res1_row_stride = res1.stride(0)
        res1_col_stride = res1.stride(1)
        out_res1_row_stride = out_res1.stride(0)
        out_res1_col_stride = out_res1.stride(1)

    DTYPE_MAX = (
        torch.finfo(out1_fp8.dtype).max
        if torch.is_floating_point(out1_fp8)
        else torch.iinfo(out1_fp8.dtype).max
    )
    _fused_rms_fp8_group_quant_kernel[(M,)](
        inp1,
        inp1_weight,
        inp2,
        inp2_weight,
        res1,
        out1_fp8,
        out1_bs,
        out2,
        out_res1,
        out1,
        inp1_epsilon,
        inp2_epsilon,
        M,
        N1,
        N2,
        inp1.stride(0),
        inp2_row_stride,
        inp1.stride(1),
        inp2_col_stride,
        res1_row_stride,
        res1_col_stride,
        out1_fp8.stride(0),
        out1_fp8.stride(1),
        out1_bs.stride(0),
        out1_bs.stride(1),
        out2_row_stride,
        out2_col_stride,
        out_res1_row_stride,
        out_res1_col_stride,
        out1_row_stride,
        out1_col_stride,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        QUANT_BLOCK_SIZE=group_size,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
        HAVE_SECOND_INPUT=(inp2 is not None),
        FIRST_INPUT_RES=(res1 is not None),
        FIRST_INPUT_OUT=output_unquantized_inp1,
    )

    return (out1_fp8, out1_bs), out1, out2, out_res1


def fused_flatten_fp8_group_quant(
    x: torch.Tensor,
    group_size,
    dtype_quant=fp8_dtype,
):
    M, N1, N2 = x.shape

    BLOCK_SIZE_N2 = max(triton.next_power_of_2(N2), group_size)
    N = N1 * N2
    out = torch.empty((M, N), dtype=dtype_quant, device=x.device)
    out_block_scales = torch.empty(
        (M, triton.cdiv(N, group_size)), dtype=torch.float32, device=x.device
    )

    DTYPE_MAX = (
        torch.finfo(out.dtype).max
        if torch.is_floating_point(out)
        else torch.iinfo(out.dtype).max
    )
    grid = (
        M,
        N1,
    )
    _fused_flatten_fp8_group_quant_kernel[grid](
        x,
        out,
        out_block_scales,
        *x.stride(),
        *out.stride(),
        *out_block_scales.stride(),
        N2,
        BLOCK_SIZE_N2=BLOCK_SIZE_N2,
        QUANT_BLOCK_SIZE=group_size,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
    )

    return out, out_block_scales
