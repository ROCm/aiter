import torch
import triton
import triton.language as tl

from aiter.ops.triton.quant import _mxfp4_quant_op
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


@triton.jit
def _rmsmorm_op(row, weight, n_cols, epsilon):
    row_norm = row * row
    row_norm = tl.sum(row_norm, axis=-1)
    norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

    rms_norm = row * norm_factor * weight
    return rms_norm


@triton.jit
def _fused_rms_mxfp4_quant_kernel(
    inp1_ptr,
    weight1_ptr,
    inp2_ptr,
    weight2_ptr,
    res1_ptr,
    out1_fp4_ptr,
    out1_bs_ptr,
    out2_ptr,
    out_res1_ptr,
    eps1,
    eps2,
    n_rows,
    inp1_n_cols,
    inp2_n_cols,
    inp1_row_stride,
    inp2_row_stride,
    res1_row_stride,
    out1_fp4_row_stride,
    out1_bs_row_stride,
    out1_bs_col_stride,
    out2_row_stride,
    out_res1_row_stride,
    BLOCK_SIZE: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    SKIP_SECOND_INPUT: tl.constexpr,
    FIRST_INPUT_RES: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE // MXFP4_QUANT_BLOCK_SIZE
    block_inds = tl.arange(0, BLOCK_SIZE)

    mask1 = block_inds < inp1_n_cols
    inp1 = tl.load(
        inp1_ptr + pid * inp1_row_stride + block_inds,
        mask=mask1,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)
    if FIRST_INPUT_RES:
        res1 = tl.load(
            res1_ptr + pid * res1_row_stride + block_inds,
            mask=mask1,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        inp1 = inp1 + res1

    w1 = tl.load(weight1_ptr + block_inds, mask=mask1, other=0.0).to(tl.float32)

    norm1 = _rmsmorm_op(inp1, w1, inp1_n_cols, eps1)
    out1_fp4, out1_block_scales = _mxfp4_quant_op(
        norm1, BLOCK_SIZE, 1, MXFP4_QUANT_BLOCK_SIZE
    )
    out1_fp4 = tl.ravel(out1_fp4)
    out1_block_scales = tl.ravel(out1_block_scales)

    # store the results
    half_block_inds = tl.arange(0, BLOCK_SIZE // 2)
    tl.store(
        out1_fp4_ptr + pid * out1_fp4_row_stride + half_block_inds,
        out1_fp4,
        mask=half_block_inds < (inp1_n_cols // 2),
    )
    bs_inds = tl.arange(0, NUM_QUANT_BLOCKS)
    num_bs_cols = (inp1_n_cols + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
    tl.store(
        out1_bs_ptr + pid * out1_bs_row_stride + bs_inds * out1_bs_col_stride,
        out1_block_scales,
        mask=bs_inds < num_bs_cols,
    )
    if not SKIP_SECOND_INPUT:
        mask2 = block_inds < inp2_n_cols
        inp2 = tl.load(
            inp2_ptr + pid * inp2_row_stride + block_inds,
            mask=mask2,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        w2 = tl.load(weight2_ptr + block_inds, mask=mask2, other=0.0).to(tl.float32)
        norm2 = _rmsmorm_op(inp2, w2, inp2_n_cols, eps2)
        tl.store(out2_ptr + pid * out2_row_stride + block_inds, norm2, mask=mask2)
    if FIRST_INPUT_RES:
        inp1 = inp1.to(out_res1_ptr.dtype.element_ty)
        tl.store(
            out_res1_ptr + pid * out_res1_row_stride + block_inds, inp1, mask=mask1
        )


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


@triton.jit
def _fused_flatten_mxfp4_quant(
    x_ptr,
    out_ptr,
    out_scales_ptr,
    x_stride_m,
    x_stride_n1,
    x_stride_n2,
    out_stride_m,
    out_stride_n,
    out_scales_stride_m,
    out_scales_stride_n,
    N2,
    BLOCK_SIZE_N2: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
):
    m = tl.program_id(0)
    n1 = tl.program_id(1)

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N2 // MXFP4_QUANT_BLOCK_SIZE
    n2_offs = tl.arange(0, BLOCK_SIZE_N2)
    x_offs = m * x_stride_m + n1 * x_stride_n1 + n2_offs * x_stride_n2
    x = tl.load(x_ptr + x_offs, mask=n2_offs < N2)

    out, out_block_scales = _mxfp4_quant_op(x, BLOCK_SIZE_N2, 1, MXFP4_QUANT_BLOCK_SIZE)
    out = tl.ravel(out)
    out_block_scales = tl.ravel(out_block_scales)

    half_block_offs = tl.arange(0, BLOCK_SIZE_N2 // 2)
    tl.store(
        out_ptr
        + m * out_stride_m
        + (n1 * (BLOCK_SIZE_N2 // 2) + half_block_offs) * out_stride_n,
        out,
        mask=half_block_offs < (N2 // 2),
    )
    block_scale_offs = tl.arange(0, NUM_QUANT_BLOCKS)
    tl.store(
        out_scales_ptr
        + m * out_scales_stride_m
        + (n1 * NUM_QUANT_BLOCKS + block_scale_offs) * out_scales_stride_n,
        out_block_scales,
        mask=block_scale_offs < tl.cdiv(N2, MXFP4_QUANT_BLOCK_SIZE),
    )


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
