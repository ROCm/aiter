import triton
import triton.language as tl


@triton.jit
def _rmsmorm_op(row, weight, n_cols, epsilon):
    row_norm = row * row
    row_norm = tl.sum(row_norm, axis=-1)
    norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

    rms_norm = row * norm_factor * weight
    return rms_norm


@triton.jit
def _fp8_quant_op(
    x,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
):
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, QUANT_BLOCK_SIZE)
    m = tl.maximum(tl.max(tl.abs(x), axis=-1), 1e-10)
    scale_out = m.to(tl.float32) / DTYPE_MAX
    scale_recip = 1.0 / scale_out.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, 1)
    x = tl.clamp(x * scale_recip, DTYPE_MIN, DTYPE_MAX)

    return x, scale_out


@triton.jit
def _fused_rms_fp8_group_quant_kernel(
    inp1_ptr,
    weight1_ptr,
    inp2_ptr,
    weight2_ptr,
    res1_ptr,
    out1_fp8_ptr,
    out1_bs_ptr,
    out2_ptr,
    out_res1_ptr,
    out1_ptr,
    eps1,
    eps2,
    n_rows,
    inp1_n_cols,
    inp2_n_cols,
    inp1_row_stride,
    inp2_row_stride,
    inp1_col_stride,
    inp2_col_stride,
    res1_row_stride,
    res1_col_stride,
    out1_fp8_row_stride,
    out1_fp8_col_stride,
    out1_bs_row_stride,
    out1_bs_col_stride,
    out2_row_stride,
    out2_col_stride,
    out_res1_row_stride,
    out_res1_col_stride,
    out1_row_stride,
    out1_col_stride,
    BLOCK_SIZE_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
    HAVE_SECOND_INPUT: tl.constexpr,
    FIRST_INPUT_RES: tl.constexpr,
    FIRST_INPUT_OUT: tl.constexpr,
):
    m_pid = tl.program_id(0)
    n_offs = tl.arange(0, BLOCK_SIZE_N)
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE

    mask1 = n_offs < inp1_n_cols
    inp1 = tl.load(
        inp1_ptr + m_pid * inp1_row_stride + n_offs * inp1_col_stride,
        mask=mask1,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)
    if FIRST_INPUT_RES:
        res1 = tl.load(
            res1_ptr + m_pid * res1_row_stride + n_offs * res1_col_stride,
            mask=mask1,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        inp1 = inp1 + res1

    w1 = tl.load(weight1_ptr + n_offs, mask=mask1, other=0.0).to(tl.float32)

    norm1 = _rmsmorm_op(inp1, w1, inp1_n_cols, eps1)

    if FIRST_INPUT_OUT:
        mask1 = n_offs < inp1_n_cols
        tl.store(
            out1_ptr + m_pid * out1_row_stride + n_offs * out1_col_stride,
            norm1,
            mask=mask1,
        )

    out1_fp8, out1_block_scales = _fp8_quant_op(
        norm1, 1, BLOCK_SIZE_N, QUANT_BLOCK_SIZE, DTYPE_MAX, DTYPE_MIN
    )
    out1_fp8 = tl.ravel(out1_fp8)
    out1_block_scales = tl.ravel(out1_block_scales)

    # store the results
    tl.store(
        out1_fp8_ptr + m_pid * out1_fp8_row_stride + n_offs * out1_fp8_col_stride,
        out1_fp8.to(out1_fp8_ptr.dtype.element_ty),
        mask=mask1,
    )
    g_offs = tl.arange(0, NUM_QUANT_BLOCKS)
    num_bs_cols = (inp1_n_cols + QUANT_BLOCK_SIZE - 1) // QUANT_BLOCK_SIZE
    tl.store(
        out1_bs_ptr + m_pid * out1_bs_row_stride + g_offs * out1_bs_col_stride,
        out1_block_scales.to(out1_bs_ptr.dtype.element_ty),
        mask=g_offs < num_bs_cols,
    )
    if HAVE_SECOND_INPUT:
        mask2 = n_offs < inp2_n_cols
        inp2 = tl.load(
            inp2_ptr + m_pid * inp2_row_stride + n_offs * inp2_col_stride,
            mask=mask2,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        w2 = tl.load(weight2_ptr + n_offs, mask=mask2, other=0.0).to(tl.float32)
        norm2 = _rmsmorm_op(inp2, w2, inp2_n_cols, eps2)
        tl.store(
            out2_ptr + m_pid * out2_row_stride + n_offs * out2_col_stride,
            norm2,
            mask=mask2,
        )

    if FIRST_INPUT_RES:
        inp1 = inp1.to(out_res1_ptr.dtype.element_ty)
        tl.store(
            out_res1_ptr + m_pid * out_res1_row_stride + n_offs * out_res1_col_stride,
            inp1,
            mask=mask1,
        )


@triton.jit
def _fused_flatten_fp8_group_quant_kernel(
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
    QUANT_BLOCK_SIZE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
):
    m = tl.program_id(0)
    n1 = tl.program_id(1)

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N2 // QUANT_BLOCK_SIZE

    n2_offs = tl.arange(0, BLOCK_SIZE_N2)
    x_offs = m * x_stride_m + n1 * x_stride_n1 + n2_offs * x_stride_n2
    x = tl.load(x_ptr + x_offs, mask=n2_offs < N2)

    out, out_block_scales = _fp8_quant_op(
        x, 1, BLOCK_SIZE_N2, QUANT_BLOCK_SIZE, DTYPE_MAX, DTYPE_MIN
    )
    out = tl.ravel(out)
    out_block_scales = tl.ravel(out_block_scales)

    tl.store(
        out_ptr + m * out_stride_m + (n1 * BLOCK_SIZE_N2 + n2_offs) * out_stride_n,
        out.to(out_ptr.dtype.element_ty),
        mask=n2_offs < N2,
    )
    block_scale_offs = tl.arange(0, NUM_QUANT_BLOCKS)
    tl.store(
        out_scales_ptr
        + m * out_scales_stride_m
        + (n1 * NUM_QUANT_BLOCKS + block_scale_offs) * out_scales_stride_n,
        out_block_scales.to(out_scales_ptr.dtype.element_ty),
        mask=block_scale_offs < tl.cdiv(N2, QUANT_BLOCK_SIZE),
    )
