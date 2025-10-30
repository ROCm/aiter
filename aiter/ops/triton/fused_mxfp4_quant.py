import torch
import triton
import triton.language as tl
from typing import Optional
from aiter.ops.triton.quant import _mxfp4_quant_op


@triton.jit
def _rmsmorm_op(row, weight, n_cols, epsilon):
    row_norm = row * row
    row_norm = tl.sum(row_norm, axis=-1)
    norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

    rms_norm = row * norm_factor[:, None] * weight
    return rms_norm


@triton.heuristics(
    {
        "EVEN_M_N": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N1"] % (args["BLOCK_SIZE_N"]) == 0,     
        "EVEN_M_N2": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N2"] % (args["BLOCK_SIZE_N2"]) == 0,      
    }
)
@triton.jit
def _fused_rms_mxfp4_quant_kernel(
    x1_ptr,
    w1_ptr,
    x2_ptr,
    w2_ptr,
    res1_ptr,
    out1_fp4_ptr,
    out1_bs_ptr,
    out2_ptr,
    out_res1_ptr,
    eps1,
    eps2,
    M,
    N1,
    N2,
    x1_stride_m,
    x2_stride_m,
    res1_stride_m,
    out1_fp4_stride_m,
    out1_bs_stride_m,
    out1_bs_stride_n,
    out2_stride_m,
    out_res1_stride_m,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_N2: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    HAS_SECOND_INPUT: tl.constexpr,
    FIRST_INPUT_RES: tl.constexpr,
    SCALE_N: tl.constexpr,
    SCALE_M_PAD: tl.constexpr,
    SCALE_N_PAD: tl.constexpr,
    SHUFFLE: tl.constexpr,
    SHUFFLE_PAD: tl.constexpr,
    EVEN_M_N: tl.constexpr,
    EVEN_M_N2: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)

    if pid >= num_pid_m:
        if HAS_SECOND_INPUT:
            pid -= num_pid_m
            x_offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            x_offs_n2 = tl.arange(0, BLOCK_SIZE_N2)
            mask2 = None
            other2 = None
            if not EVEN_M_N2:
                mask2 = (x_offs_m < M)[:, None] & (x_offs_n2 < N2)[None, :]
                other2 = 0.0

            x2 = tl.load(
                x2_ptr + x_offs_m[:, None] * x2_stride_m + x_offs_n2[None, :],
                mask=mask2,
                other=other2,
                cache_modifier=".cg",
            ).to(tl.float32)

            w_mask2 = None
            w_other2 = None
            if not EVEN_M_N2:
                w_mask2 = x_offs_n2 < N2
                w_other2 = 0.0

            w2 = tl.load(w2_ptr + x_offs_n2, mask=w_mask2, other=w_other2).to(tl.float32)

            norm2 = _rmsmorm_op(x2, w2, N2, eps2)

            tl.store(
                out2_ptr + x_offs_m[:, None] * out2_stride_m + x_offs_n2[None, :],
                norm2.to(out2_ptr.type.element_ty),
                mask=mask2,
            )
        return

    x_offs_n = tl.arange(0, BLOCK_SIZE_N)
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x_offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    mask1 = None
    other1 = None
    if not EVEN_M_N:
        mask1 = (x_offs_m < M)[:, None] & (x_offs_n < N1)[None, :]
        other1 = 0.0

    x1 = tl.load(
        x1_ptr + x_offs_m[:, None] * x1_stride_m + x_offs_n[None, :],
        mask=mask1,
        other=other1,
        cache_modifier=".cg",
    ).to(tl.float32)

    if FIRST_INPUT_RES:
        res1 = tl.load(
            res1_ptr + x_offs_m[:, None] * res1_stride_m + x_offs_n[None, :],
            mask=mask1,
            other=other1,
            cache_modifier=".cg",
        ).to(tl.float32)
        x1 = x1 + res1

    w_mask1 = None
    w_other1 = None
    if not EVEN_M_N:
        w_mask1 = x_offs_n < N1
        w_other1 = 0.0

    w1 = tl.load(w1_ptr + x_offs_n, mask=w_mask1, other=w_other1).to(tl.float32)

    norm1 = _rmsmorm_op(x1, w1, N1, eps1)
    out1_fp4, bs_e8m0 = _mxfp4_quant_op(
        norm1, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
    )

    # store the results
    half_x_offs_n = tl.arange(0, BLOCK_SIZE_N // 2)
    out_mask1 = None
    if not EVEN_M_N:
        out_mask1 = (x_offs_m < M)[:, None] & (half_x_offs_n < (N1 // 2))[None, :]

    tl.store(
        out1_fp4_ptr + x_offs_m[:, None] * out1_fp4_stride_m + half_x_offs_n[None, :],
        out1_fp4,
        mask=out_mask1,
    )

    bs_offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    bs_offs_n = tl.arange(0, NUM_QUANT_BLOCKS)
    num_bs_cols = (N1 + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
    if SHUFFLE:
        bs_offs_0 = bs_offs_m[:, None] // 32
        bs_offs_1 = bs_offs_m[:, None] % 32
        bs_offs_2 = bs_offs_1 % 16
        bs_offs_1 = bs_offs_1 // 16
        bs_offs_3 = bs_offs_n[None, :] // 8
        bs_offs_4 = bs_offs_n[None, :] % 8
        bs_offs_5 = bs_offs_4 % 4
        bs_offs_4 = bs_offs_4 // 4
        bs_offs = (
            bs_offs_1
            + bs_offs_4 * 2
            + bs_offs_2 * 2 * 2
            + bs_offs_5 * 2 * 2 * 16
            + bs_offs_3 * 2 * 2 * 16 * 4
            + bs_offs_0 * 2 * 16 * SCALE_N_PAD
        )
        bs_mask_127 = (bs_offs_m < M)[:, None] & (bs_offs_n < num_bs_cols)[None, :]
        bs_e8m0 = tl.where(bs_mask_127, bs_e8m0, 127)
    else:
        bs_offs = (
            bs_offs_m[:, None] * out1_bs_stride_m + bs_offs_n[None, :] * out1_bs_stride_n
        )
        
    bs_mask = None
    if not EVEN_M_N:
        if SHUFFLE_PAD:
            bs_mask = (bs_offs_m < SCALE_M_PAD)[:, None] & (bs_offs_n < SCALE_N_PAD)[None, :]
        else:
            bs_mask = (bs_offs_m < M)[:, None] & (bs_offs_n < SCALE_N)[None, :]

    tl.store(
        out1_bs_ptr + bs_offs,
        bs_e8m0.to(out1_bs_ptr.type.element_ty),
        mask=bs_mask,
    )

    if FIRST_INPUT_RES:
        tl.store(
            out_res1_ptr + x_offs_m[:, None] * out_res1_stride_m + x_offs_n[None, :],
            x1.to(out_res1_ptr.dtype.element_ty),
            mask=mask1,
        )


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

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M) * (2 if (x2 is not None) else 1),
    )
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
