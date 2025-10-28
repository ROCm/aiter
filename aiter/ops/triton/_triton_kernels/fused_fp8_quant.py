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


@triton.jit
def _fused_reduce_act_mul_fp8_group_quant(
    x_ptr,
    y_ptr,
    y_scale_ptr,
    x2_ptr,
    y2_ptr,
    M,
    N1,
    N2,
    stride_x_spk,
    stride_x_m,
    stride_x_n,
    stride_y_m,
    stride_y_n,
    stride_y_scale_m,
    stride_y_scale_n,
    stride_x2_spk,
    stride_x2_m,
    stride_x2_n,
    stride_y2_m,
    stride_y2_n,
    # Meta-parameters
    ACTIVATION: tl.constexpr,
    BLOCK_SIZE_M2: tl.constexpr,
    BLOCK_SIZE_N1: tl.constexpr,
    BLOCK_SIZE_N2: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
    X_HAS_SPLITK: tl.constexpr,
    X_NUM_KSPLIT: tl.constexpr,
    X_NUM_KSPLIT_POW2: tl.constexpr,
    X_MASK: tl.constexpr,
):

    tl.assume(stride_x_spk > 0)
    tl.assume(stride_x_m > 0)
    tl.assume(stride_x_n > 0)
    tl.assume(stride_y_m > 0)
    tl.assume(stride_y_n > 0)
    tl.assume(stride_y_scale_m > 0)
    tl.assume(stride_y_scale_n > 0)
    tl.assume(stride_x2_spk > 0)
    tl.assume(stride_x2_m > 0)
    tl.assume(stride_x2_n > 0)
    tl.assume(stride_y2_m > 0)
    tl.assume(stride_y2_n > 0)

    m_pid = tl.program_id(axis=0)
    if X_HAS_SPLITK and m_pid >= M:
        pid2 = m_pid - M
        num_pid_n2 = tl.cdiv(N2, BLOCK_SIZE_N2)
        pid_m2 = pid2 // num_pid_n2
        pid_n2 = pid2 % num_pid_n2
        offs_m2 = (pid_m2 * BLOCK_SIZE_M2 + tl.arange(0, BLOCK_SIZE_M2)) % M
        offs_n2 = (pid_n2 * BLOCK_SIZE_N2 + tl.arange(0, BLOCK_SIZE_N2)) % N2
        offs_spk = tl.arange(0, X_NUM_KSPLIT_POW2)
        x2_ptrs = (
            x2_ptr
            + offs_spk[:, None, None] * stride_x2_spk
            + offs_m2[None, :, None] * stride_x2_m
            + offs_n2[None, None, :] * stride_x2_n
        )
        if X_NUM_KSPLIT_POW2 == X_NUM_KSPLIT:
            x2 = tl.load(x2_ptrs)
        else:
            x2 = tl.load(
                x2_ptrs, mask=offs_spk[:, None, None] < X_NUM_KSPLIT, other=0.0
            )
        x2 = tl.sum(x2, axis=0)

        x2 = x2.to(y2_ptr.type.element_ty)

        y2_out_ptrs = (
            y2_ptr + (offs_m2[:, None] * stride_y2_m) + (offs_n2[None, :] * stride_y2_n)
        )

        tl.store(y2_out_ptrs, x2)
        return

    n_offs = tl.arange(0, BLOCK_SIZE_N1)
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N1 // QUANT_BLOCK_SIZE

    mask = None
    other = None
    if X_HAS_SPLITK:
        offs_spk = tl.arange(0, X_NUM_KSPLIT_POW2)
        x_ptrs = (
            x_ptr
            + offs_spk[:, None] * stride_x_spk
            + m_pid * stride_x_m
            + n_offs[None, :] * stride_x_n
        )
        if X_MASK:
            mask = (offs_spk[:, None] < X_NUM_KSPLIT) & (n_offs[None, :] < N1)
            other = 0.0
        else:
            mask = offs_spk[:, None] < X_NUM_KSPLIT
            other = 0.0
    else:
        x_ptrs = x_ptr + m_pid * stride_x_m + n_offs * stride_x_n
        if X_MASK:
            mask = n_offs < N1
            other = 0.0

    x = tl.load(
        x_ptrs,
        mask=mask,
        other=other,
        cache_modifier=".cg",
    ).to(tl.float32)
    x_mul = tl.load(
        x_ptrs + N1 * stride_x_n,
        mask=mask,
        other=other,
        cache_modifier=".cg",
    ).to(tl.float32)

    if X_HAS_SPLITK:
        x = tl.sum(x, axis=0)
        x_mul = tl.sum(x_mul, axis=0)

    x = ACTIVATION(x) * x_mul

    y, y_scale = _fp8_quant_op(
        x, 1, BLOCK_SIZE_N1, QUANT_BLOCK_SIZE, DTYPE_MAX, DTYPE_MIN
    )
    y = tl.ravel(y)
    y_scale = tl.ravel(y_scale)

    if X_MASK:
        mask = n_offs < N1
    else:
        mask = n_offs < N1
    tl.store(
        y_ptr + m_pid * stride_y_m + n_offs * stride_y_n,
        y.to(y_ptr.dtype.element_ty),
        mask=mask,
    )
    g_offs = tl.arange(0, NUM_QUANT_BLOCKS)
    num_bs_cols = (N1 + QUANT_BLOCK_SIZE - 1) // QUANT_BLOCK_SIZE
    tl.store(
        y_scale_ptr + m_pid * stride_y_scale_m + g_offs * stride_y_scale_n,
        y_scale.to(y_scale_ptr.dtype.element_ty),
        mask=g_offs < num_bs_cols,
    )


@triton.jit
def _fused_reduce_rms_fp8_group_quant_kernel(
    inp1_ptr,
    weight1_ptr,
    inp2_ptr,
    weight2_ptr,
    inp3_ptr,
    res1_ptr,
    out1_fp8_ptr,
    out1_bs_ptr,
    out2_ptr,
    out_res1_ptr,
    out1_ptr,
    out3_ptr,
    eps1,
    eps2,
    n_rows,
    inp1_n_cols,
    inp2_n_cols,
    inp3_n_cols,
    inp1_spk_stride,
    inp2_spk_stride,
    inp3_spk_stride,
    inp1_row_stride,
    inp2_row_stride,
    inp3_row_stride,
    inp1_col_stride,
    inp2_col_stride,
    inp3_col_stride,
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
    out3_row_stride,
    out3_col_stride,
    BLOCK_SIZE_N1: tl.constexpr,
    BLOCK_SIZE_N2: tl.constexpr,
    BLOCK_SIZE_N3: tl.constexpr,
    N_MASK1: tl.constexpr,
    N_MASK2: tl.constexpr,
    N_MASK3: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
    HAVE_SECOND_INPUT: tl.constexpr,
    FIRST_INPUT_RES: tl.constexpr,
    FIRST_INPUT_OUT: tl.constexpr,
    HAS_SPLITK: tl.constexpr,
    NUM_SPLITK: tl.constexpr,
    NUM_SPLITK_POW2: tl.constexpr,
):
    m_pid = tl.program_id(0)

    if m_pid < n_rows:
        n1_offs = tl.arange(0, BLOCK_SIZE_N1)
        NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N1 // QUANT_BLOCK_SIZE

        if N_MASK1:
            mask1 = n1_offs < inp1_n_cols
            other1 = 0.0
        else:
            mask1 = None
            other1 = None
        if HAS_SPLITK:
            spk_offs = tl.arange(0, NUM_SPLITK_POW2)
            if NUM_SPLITK_POW2 != NUM_SPLITK:
                if N_MASK1:
                    mask1_in = (spk_offs[:, None] < NUM_SPLITK) & (
                        n1_offs[None, :] < inp1_n_cols
                    )
                else:
                    mask1_in = spk_offs[:, None] < NUM_SPLITK
                other1_in = 0.0
            else:
                if N_MASK1:
                    mask1_in = mask1[None, :]
                else:
                    mask1_in = mask1
                other1_in = other1

            inp1 = tl.load(
                inp1_ptr
                + spk_offs[:, None] * inp1_spk_stride
                + m_pid * inp1_row_stride
                + n1_offs[None, :] * inp1_col_stride,
                mask=mask1_in,
                other=other1_in,
                cache_modifier=".cg",
            ).to(tl.float32)
            inp1 = tl.sum(inp1, axis=0)
        else:
            inp1 = tl.load(
                inp1_ptr + m_pid * inp1_row_stride + n1_offs * inp1_col_stride,
                mask=mask1,
                other=other1,
                cache_modifier=".cg",
            ).to(tl.float32)

        if FIRST_INPUT_RES:
            res1 = tl.load(
                res1_ptr + m_pid * res1_row_stride + n1_offs * res1_col_stride,
                mask=mask1,
                other=other1,
                cache_modifier=".cg",
            ).to(tl.float32)
            inp1 = inp1 + res1

        w1 = tl.load(weight1_ptr + n1_offs, mask=mask1, other=other1).to(tl.float32)

        norm1 = _rmsmorm_op(inp1, w1, inp1_n_cols, eps1)

        if FIRST_INPUT_OUT:
            tl.store(
                out1_ptr + m_pid * out1_row_stride + n1_offs * out1_col_stride,
                norm1,
                mask=mask1,
            )

        out1_fp8, out1_block_scales = _fp8_quant_op(
            norm1, 1, BLOCK_SIZE_N1, QUANT_BLOCK_SIZE, DTYPE_MAX, DTYPE_MIN
        )
        out1_fp8 = tl.ravel(out1_fp8)
        out1_block_scales = tl.ravel(out1_block_scales)

        # store the results
        tl.store(
            out1_fp8_ptr + m_pid * out1_fp8_row_stride + n1_offs * out1_fp8_col_stride,
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

        if FIRST_INPUT_RES:
            inp1 = inp1.to(out_res1_ptr.dtype.element_ty)
            tl.store(
                out_res1_ptr
                + m_pid * out_res1_row_stride
                + n1_offs * out_res1_col_stride,
                inp1,
                mask=mask1,
            )
    elif m_pid < 2 * n_rows:
        m_pid -= n_rows
        if HAS_SPLITK:
            spk_offs = tl.arange(0, NUM_SPLITK_POW2)
        if HAVE_SECOND_INPUT:
            n2_offs = tl.arange(0, BLOCK_SIZE_N2)
            if N_MASK2:
                mask2 = n2_offs < inp1_n_cols
                other2 = 0.0
            else:
                mask2 = None
                other2 = None
            if HAS_SPLITK:
                if NUM_SPLITK_POW2 != NUM_SPLITK:
                    if N_MASK2:
                        mask2_in = (spk_offs[:, None] < NUM_SPLITK) & (
                            n2_offs[None, :] < inp2_n_cols
                        )
                    else:
                        mask2_in = spk_offs[:, None] < NUM_SPLITK
                    other2_in = 0.0
                else:
                    if N_MASK2:
                        mask2_in = mask2[None, :]
                    else:
                        mask2_in = mask2
                    other2_in = other2
                inp2 = tl.load(
                    inp2_ptr
                    + spk_offs[:, None] * inp2_spk_stride
                    + m_pid * inp2_row_stride
                    + n2_offs[None, :] * inp2_col_stride,
                    mask=mask2_in,
                    other=other2_in,
                    cache_modifier=".cg",
                ).to(tl.float32)
                inp2 = tl.sum(inp2, axis=0)
            else:
                inp2 = tl.load(
                    inp2_ptr + m_pid * inp2_row_stride + n2_offs * inp2_col_stride,
                    mask=mask2,
                    other=other2,
                    cache_modifier=".cg",
                ).to(tl.float32)
            w2 = tl.load(weight2_ptr + n2_offs, mask=mask2, other=other2).to(tl.float32)
            norm2 = _rmsmorm_op(inp2, w2, inp2_n_cols, eps2)
            tl.store(
                out2_ptr + m_pid * out2_row_stride + n2_offs * out2_col_stride,
                norm2,
                mask=mask2,
            )
    elif m_pid < 3 * n_rows:
        m_pid -= 2 * n_rows
        if HAS_SPLITK:
            spk_offs = tl.arange(0, NUM_SPLITK_POW2)
            n3_offs = tl.arange(0, BLOCK_SIZE_N3)
            if N_MASK3:
                mask3 = n3_offs < inp3_n_cols
                other3 = 0.0
            else:
                mask3 = None
                other3 = None
            if NUM_SPLITK_POW2 != NUM_SPLITK:
                if N_MASK3:
                    mask3_in = (spk_offs[:, None] < NUM_SPLITK) & (
                        n3_offs[None, :] < inp3_n_cols
                    )
                else:
                    mask3_in = spk_offs[:, None] < NUM_SPLITK
                other3_in = 0.0
            else:
                if N_MASK3:
                    mask3_in = mask3[None, :]
                else:
                    mask3_in = mask3
                other3_in = other3
            inp3 = tl.load(
                inp3_ptr
                + spk_offs[:, None] * inp3_spk_stride
                + m_pid * inp3_row_stride
                + n3_offs[None, :] * inp3_col_stride,
                mask=mask3_in,
                other=other3_in,
                cache_modifier=".cg",
            ).to(tl.float32)
            inp3 = tl.sum(inp3, axis=0)
            tl.store(
                out3_ptr + m_pid * out3_row_stride + n3_offs * out3_col_stride,
                inp3,
                mask=mask3,
            )
