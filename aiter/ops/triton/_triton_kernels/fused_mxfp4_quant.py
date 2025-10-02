import triton
import triton.language as tl

from .quant import _mxfp4_quant_op


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
