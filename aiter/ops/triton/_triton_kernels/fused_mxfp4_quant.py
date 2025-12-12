import triton
import triton.language as tl

from .quant import _mxfp4_quant_op


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
    out1_ptr,
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
    out1_stride_m,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_N2: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    HAS_SECOND_INPUT: tl.constexpr,
    FIRST_INPUT_RES: tl.constexpr,
    FIRST_INPUT_OUT: tl.constexpr,
    SCALE_N: tl.constexpr,
    SCALE_M_PAD: tl.constexpr,
    SCALE_N_PAD: tl.constexpr,
    SHUFFLE: tl.constexpr,
    SHUFFLE_PAD: tl.constexpr,
    EVEN_M_N: tl.constexpr,
    EVEN_M_N2: tl.constexpr,
):
    # TODO: XCD remapping where every 32-token block should share the same XCD
    # TODO: debug for large M
    # TODO: investigate cache_modifier='.cg' on tl.store
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

            w2 = tl.load(w2_ptr + x_offs_n2, mask=w_mask2, other=w_other2).to(
                tl.float32
            )

            norm2 = _rmsmorm_op(x2, w2, N2, eps2)

            tl.store(
                out2_ptr + x_offs_m[:, None] * out2_stride_m + x_offs_n2[None, :],
                norm2.to(out2_ptr.type.element_ty),
                mask=mask2,
                cache_modifier=".cg",
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

    if FIRST_INPUT_OUT:
        tl.store(
            out1_ptr + x_offs_m[:, None] * out1_stride_m + x_offs_n[None, :],
            norm1,
            mask=mask1,
        )

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
        cache_modifier=".cg",
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
            bs_offs_m[:, None] * out1_bs_stride_m
            + bs_offs_n[None, :] * out1_bs_stride_n
        )

    bs_mask = None
    if not EVEN_M_N:
        if SHUFFLE_PAD:
            bs_mask = (bs_offs_m < SCALE_M_PAD)[:, None] & (bs_offs_n < SCALE_N_PAD)[
                None, :
            ]
        else:
            bs_mask = (bs_offs_m < M)[:, None] & (bs_offs_n < SCALE_N)[None, :]

    tl.store(
        out1_bs_ptr + bs_offs,
        bs_e8m0.to(out1_bs_ptr.type.element_ty),
        mask=bs_mask,
        cache_modifier=".cg",
    )

    if FIRST_INPUT_RES:
        tl.store(
            out_res1_ptr + x_offs_m[:, None] * out_res1_stride_m + x_offs_n[None, :],
            x1.to(out_res1_ptr.dtype.element_ty),
            mask=mask1,
            cache_modifier=".cg",
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


@triton.heuristics(
    {
        "EVEN_M_N": lambda args: args["M"] % args["BLOCK_SIZE_M1"] == 0
        and args["N1"] % (args["BLOCK_SIZE_N1"] * args["NUM_ITER"]) == 0,
    }
)
@triton.jit
def _fused_reduce_act_mul_and_dynamic_mxfp4_quant_kernel(
    x_ptr,
    y_ptr,
    y_scale_ptr,
    x2_ptr,
    y2_ptr,
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
    M,
    N1,
    N2,
    BLOCK_SIZE_M1: tl.constexpr,
    BLOCK_SIZE_N1: tl.constexpr,
    BLOCK_SIZE_M2: tl.constexpr,
    BLOCK_SIZE_N2: tl.constexpr,
    NUM_ITER: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    EVEN_M_N: tl.constexpr,
    SCALING_MODE: tl.constexpr,
    ACTIVATION: tl.constexpr,
    scaleN: tl.constexpr,
    scaleM_pad: tl.constexpr,
    scaleN_pad: tl.constexpr,
    SHUFFLE: tl.constexpr,
    X_HAS_SPLITK: tl.constexpr,
    X_NUM_KSPLIT: tl.constexpr,
    X_NUM_KSPLIT_POW2: tl.constexpr,
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

    all_pid = tl.program_id(axis=0)
    num_pid_m1 = tl.cdiv(M, BLOCK_SIZE_M1)
    num_pid_n1 = tl.cdiv(N1, BLOCK_SIZE_N1 * NUM_ITER)
    num_pid_1 = num_pid_m1 * num_pid_n1

    if X_HAS_SPLITK and all_pid >= num_pid_1:
        pid2 = all_pid - num_pid_1
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

    pid_m = all_pid // num_pid_n1
    start_n = all_pid % num_pid_n1 * NUM_ITER
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N1 // MXFP4_QUANT_BLOCK_SIZE

    offs_spk = None
    if X_HAS_SPLITK:
        offs_spk = tl.arange(0, X_NUM_KSPLIT_POW2)

    for pid_n in tl.range(start_n, min(start_n + NUM_ITER, N1), num_stages=NUM_STAGES):
        x_offs_m = pid_m * BLOCK_SIZE_M1 + tl.arange(0, BLOCK_SIZE_M1)
        x_offs_n = pid_n * BLOCK_SIZE_N1 + tl.arange(0, BLOCK_SIZE_N1)

        mask = None
        other = None
        if X_HAS_SPLITK:
            x_ptrs = (
                x_ptr
                + offs_spk[:, None, None] * stride_x_spk
                + x_offs_m[None, :, None] * stride_x_m
                + x_offs_n[None, None, :] * stride_x_n
            )
            if X_NUM_KSPLIT_POW2 != X_NUM_KSPLIT and not EVEN_M_N:
                mask = (
                    (offs_spk[:, None, None] < X_NUM_KSPLIT)
                    & (x_offs_m[None, :, None] < M)
                    & (x_offs_n[None, None, :] < N1)
                )
                other = 0.0
            elif not (X_NUM_KSPLIT_POW2 == X_NUM_KSPLIT):
                mask = offs_spk[:, None, None] < X_NUM_KSPLIT
                other = 0.0
            elif not EVEN_M_N:
                mask = (x_offs_m[None, :, None] < M) & (x_offs_n[None, None, :] < N1)
                other = 0.0
        else:
            x_ptrs = (
                x_ptr + x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n
            )
            if not EVEN_M_N:
                mask = (x_offs_m[:, None] < M) & (x_offs_n[None, :] < N1)
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

        # x = _apply_activation_from_str(a, ACTIVATION) * b
        x = ACTIVATION(x) * x_mul

        y, y_scale = _mxfp4_quant_op(
            x, BLOCK_SIZE_N1, BLOCK_SIZE_M1, MXFP4_QUANT_BLOCK_SIZE
        )

        out_offs_m = pid_m * BLOCK_SIZE_M1 + tl.arange(0, BLOCK_SIZE_M1)
        # out_offs_m = x_offs_m
        out_offs_n = pid_n * BLOCK_SIZE_N1 // 2 + tl.arange(0, BLOCK_SIZE_N1 // 2)
        out_offs = out_offs_m[:, None] * stride_y_m + out_offs_n[None, :] * stride_y_n

        if EVEN_M_N:
            tl.store(y_ptr + out_offs, y)
        else:
            out_mask = (out_offs_m < M)[:, None] & (out_offs_n < (N1 // 2))[None, :]
            tl.store(y_ptr + out_offs, y, mask=out_mask)

        bs_offs_m = pid_m * BLOCK_SIZE_M1 + tl.arange(0, BLOCK_SIZE_M1)
        # bs_offs_m = x_offs_m
        bs_offs_n = pid_n * NUM_QUANT_BLOCKS + tl.arange(0, NUM_QUANT_BLOCKS)
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
                + bs_offs_0 * 2 * 16 * scaleN
            )
            bs_mask1 = (bs_offs_m < M)[:, None] & (bs_offs_n < scaleN)[None, :]
            bs_mask = (bs_offs_m < scaleM_pad)[:, None] & (bs_offs_n < scaleN_pad)[
                None, :
            ]
            y_scale = tl.where(bs_mask1, y_scale, 127)
        else:
            bs_offs = (
                bs_offs_m[:, None] * stride_y_scale_m
                + bs_offs_n[None, :] * stride_y_scale_n
            )
            bs_mask = (bs_offs_m < M)[:, None] & (bs_offs_n < scaleN)[None, :]
        if EVEN_M_N:
            tl.store(y_scale_ptr + bs_offs, y_scale)
        else:
            tl.store(
                y_scale_ptr + bs_offs,
                y_scale,
                mask=bs_mask,
            )


@triton.heuristics(
    {
        "EVEN_M_N": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N1"] % (args["BLOCK_SIZE_N"]) == 0,
        "EVEN_M_N2": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N2"] % (args["BLOCK_SIZE_N2"]) == 0,
        "EVEN_M_N3": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N3"] % (args["BLOCK_SIZE_N3"]) == 0,
    }
)
@triton.jit
def _fused_reduce_rms_mxfp4_quant_kernel(
    x1_ptr,
    w1_ptr,
    x2_ptr,
    w2_ptr,
    x3_ptr,
    res1_ptr,
    out1_fp4_ptr,
    out1_bs_ptr,
    out1_ptr,
    out2_ptr,
    out3_ptr,
    out_res1_ptr,
    eps1,
    eps2,
    M,
    N1,
    N2,
    N3,
    x1_stride_spk,
    x1_stride_m,
    x2_stride_spk,
    x2_stride_m,
    x3_stride_spk,
    x3_stride_m,
    res1_stride_m,
    out1_fp4_stride_m,
    out1_bs_stride_m,
    out1_bs_stride_n,
    out1_stride_m,
    out2_stride_m,
    out3_stride_m,
    out_res1_stride_m,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_N2: tl.constexpr,
    BLOCK_SIZE_N3: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    HAS_SECOND_INPUT: tl.constexpr,
    FIRST_INPUT_RES: tl.constexpr,
    FIRST_INPUT_OUT: tl.constexpr,
    HAS_SPLITK: tl.constexpr,
    NUM_SPLITK: tl.constexpr,
    NUM_SPLITK_POW2: tl.constexpr,
    SCALE_N: tl.constexpr,
    SCALE_M_PAD: tl.constexpr,
    SCALE_N_PAD: tl.constexpr,
    SHUFFLE: tl.constexpr,
    SHUFFLE_PAD: tl.constexpr,
    EVEN_M_N: tl.constexpr,
    EVEN_M_N2: tl.constexpr,
    EVEN_M_N3: tl.constexpr,
):
    # TODO: XCD remapping where every 32-token block should share the same XCD
    # TODO: debug for large M
    # TODO: investigate cache_modifier='.cg' on tl.store
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)

    if pid >= 2 * num_pid_m:
        pid -= 2 * num_pid_m
        if HAS_SPLITK:
            spk_offs = tl.arange(0, NUM_SPLITK_POW2)
            x_offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            x_offs_n3 = tl.arange(0, BLOCK_SIZE_N3)
            mask3 = None
            mask3_out = None
            other3 = None
            if not EVEN_M_N3:
                other3 = 0.0
                mask3_out = (x_offs_m < M)[:, None] & (x_offs_n3 < N3)[None, :]
                if NUM_SPLITK_POW2 != NUM_SPLITK:
                    mask3 = (
                        (spk_offs < NUM_SPLITK)[:, None, None]
                        & (x_offs_m < M)[None, :, None]
                        & (x_offs_n3 < N3)[None, None, :]
                    )
                else:
                    mask3 = (x_offs_m < M)[None, :, None] & (x_offs_n3 < N3)[
                        None, None, :
                    ]
            elif NUM_SPLITK_POW2 != NUM_SPLITK:
                other3 = 0.0
                mask3 = (spk_offs < NUM_SPLITK)[:, None, None]

            x3 = tl.load(
                x3_ptr
                + spk_offs[:, None, None] * x3_stride_spk
                + x_offs_m[None, :, None] * x3_stride_m
                + x_offs_n3[None, None, :],
                mask=mask3,
                other=other3,
                cache_modifier=".cg",
            ).to(tl.float32)
            x3 = tl.sum(x3, axis=0)
            tl.store(
                out3_ptr + x_offs_m[:, None] * out3_stride_m + x_offs_n3[None, :],
                x3.to(out3_ptr.dtype.element_ty),
                mask=mask3_out,
            )
        return

    if pid >= num_pid_m:
        pid -= num_pid_m
        if HAS_SECOND_INPUT:
            if HAS_SPLITK:
                spk_offs = tl.arange(0, NUM_SPLITK_POW2)
            x_offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            x_offs_n2 = tl.arange(0, BLOCK_SIZE_N2)
            mask2 = None
            mask2_out = None
            other2 = None

            if HAS_SPLITK:
                if not EVEN_M_N2:
                    other2 = 0.0
                    mask2_out = (x_offs_m < M)[:, None] & (x_offs_n2 < N2)[None, :]
                    if NUM_SPLITK_POW2 != NUM_SPLITK:
                        mask2 = (
                            (spk_offs < NUM_SPLITK)[:, None, None]
                            & (x_offs_m < M)[None, :, None]
                            & (x_offs_n2 < N2)[None, None, :]
                        )
                    else:
                        mask2 = (x_offs_m < M)[None, :, None] & (x_offs_n2 < N2)[
                            None, None, :
                        ]
                elif NUM_SPLITK_POW2 != NUM_SPLITK:
                    other2 = 0.0
                    mask2 = (spk_offs < NUM_SPLITK)[:, None, None]

                x2_ptrs = (
                    x2_ptr
                    + spk_offs[:, None, None] * x2_stride_spk
                    + x_offs_m[None, :, None] * x2_stride_m
                    + x_offs_n2[None, None, :]
                )
            else:
                if not EVEN_M_N2:
                    other2 = 0.0
                    mask2_out = (x_offs_m < M)[:, None] & (x_offs_n2 < N2)[None, :]
                    mask2 = (x_offs_m < M)[:, None] & (x_offs_n2 < N2)[None, :]

                x2_ptrs = x2_ptr + x_offs_m[:, None] * x2_stride_m + x_offs_n2[None, :]

            x2 = tl.load(
                x2_ptrs,
                mask=mask2,
                other=other2,
                cache_modifier=".cg",
            ).to(tl.float32)

            if HAS_SPLITK:
                x2 = tl.sum(x2, axis=0)

            w_mask2 = None
            w_other2 = None
            if not EVEN_M_N2:
                w_mask2 = x_offs_n2 < N2
                w_other2 = 0.0

            w2 = tl.load(w2_ptr + x_offs_n2, mask=w_mask2, other=w_other2).to(
                tl.float32
            )

            norm2 = _rmsmorm_op(x2, w2, N2, eps2)

            tl.store(
                out2_ptr + x_offs_m[:, None] * out2_stride_m + x_offs_n2[None, :],
                norm2.to(out2_ptr.type.element_ty),
                mask=mask2_out,
            )
        return

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x_offs_n = tl.arange(0, BLOCK_SIZE_N)
    x_offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    if HAS_SPLITK:
        spk_offs = tl.arange(0, NUM_SPLITK_POW2)

    mask1 = None
    mask1_out = None
    other1 = None
    if HAS_SPLITK:
        if not EVEN_M_N:
            other1 = 0.0
            mask1_out = (x_offs_m < M)[:, None] & (x_offs_n < N1)[None, :]
            if NUM_SPLITK_POW2 != NUM_SPLITK:
                mask1 = (
                    (spk_offs < NUM_SPLITK)[:, None, None]
                    & (x_offs_m < M)[None, :, None]
                    & (x_offs_n < N1)[None, None, :]
                )
            else:
                mask1 = (x_offs_m < M)[None, :, None] & (x_offs_n < N1)[None, None, :]
        elif NUM_SPLITK_POW2 != NUM_SPLITK:
            other1 = 0.0
            mask1 = (spk_offs < NUM_SPLITK)[:, None, None]

        x1_ptrs = (
            x1_ptr
            + spk_offs[:, None, None] * x1_stride_spk
            + x_offs_m[None, :, None] * x1_stride_m
            + x_offs_n[None, None, :]
        )
    else:
        if not EVEN_M_N:
            other1 = 0.0
            mask1_out = (x_offs_m < M)[:, None] & (x_offs_n < N1)[None, :]
            mask1 = (x_offs_m < M)[:, None] & (x_offs_n < N1)[None, :]

        x1_ptrs = x1_ptr + x_offs_m[:, None] * x1_stride_m + x_offs_n[None, :]

    x1 = tl.load(
        x1_ptrs,
        mask=mask1,
        other=other1,
        cache_modifier=".cg",
    ).to(tl.float32)

    if HAS_SPLITK:
        x1 = tl.sum(x1, axis=0)

    if FIRST_INPUT_RES:
        other1_res = None
        mask1_res = None
        if not EVEN_M_N:
            other1_res = 0.0
            mask1_res = (x_offs_m < M)[:, None] & (x_offs_n < N1)[None, :]

        res1 = tl.load(
            res1_ptr + x_offs_m[:, None] * res1_stride_m + x_offs_n[None, :],
            mask=mask1_res,
            other=other1_res,
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

    if FIRST_INPUT_OUT:
        tl.store(
            out1_ptr + x_offs_m[:, None] * out1_stride_m + x_offs_n[None, :],
            norm1.to(out1_ptr.dtype.element_ty),
            mask=mask1_out,
        )

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
        cache_modifier=".cg",
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
            bs_offs_m[:, None] * out1_bs_stride_m
            + bs_offs_n[None, :] * out1_bs_stride_n
        )

    bs_mask = None
    if not EVEN_M_N:
        if SHUFFLE_PAD:
            bs_mask = (bs_offs_m < SCALE_M_PAD)[:, None] & (bs_offs_n < SCALE_N_PAD)[
                None, :
            ]
        else:
            bs_mask = (bs_offs_m < M)[:, None] & (bs_offs_n < SCALE_N)[None, :]

    tl.store(
        out1_bs_ptr + bs_offs,
        bs_e8m0.to(out1_bs_ptr.type.element_ty),
        mask=bs_mask,
        cache_modifier=".cg",
    )

    if FIRST_INPUT_RES:
        tl.store(
            out_res1_ptr + x_offs_m[:, None] * out_res1_stride_m + x_offs_n[None, :],
            x1.to(out_res1_ptr.dtype.element_ty),
            mask=mask1_out,
            cache_modifier=".cg",
        )


@triton.jit
def _fused_dynamic_mxfp4_quant_moe_sort_kernel(
    x_ptr,
    x_fp4_ptr,
    sorted_ids_ptr,
    num_valid_ids_ptr,
    blockscale_e8m0_sorted_ptr,
    Mx,
    Nx,
    scaleNx,
    stride_x_m,
    stride_x_n,
    stride_x_fp4_m,
    stride_x_fp4_n,
    stride_o3,  #: tl.constexpr,
    stride_o2,  #: tl.constexpr,
    stride_o1,  #: tl.constexpr,
    stride_o0,  #: tl.constexpr,
    stride_o4,  #: tl.constexpr,
    token_num,  #: tl.constexpr,
    M_i,  #: tl.constexpr,
    N_i,  #: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_Mx: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TOPK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_x = tl.cdiv(Mx, BLOCK_SIZE_Mx) * scaleNx

    stride_x_m = tl.cast(stride_x_m, tl.int64)
    stride_x_n = tl.cast(stride_x_n, tl.int64)
    stride_x_fp4_m = tl.cast(stride_x_fp4_m, tl.int64)
    stride_x_fp4_n = tl.cast(stride_x_fp4_n, tl.int64)

    if pid < num_pid_x:
        pid_m = pid // scaleNx
        pid_n = pid % scaleNx

        x_offs_m = pid_m * BLOCK_SIZE_Mx + tl.arange(0, BLOCK_SIZE_Mx)
        x_offs_n = pid_n * MXFP4_QUANT_BLOCK_SIZE + tl.arange(0, MXFP4_QUANT_BLOCK_SIZE)
        x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n
        x_mask = (x_offs_m < Mx)[:, None] & (x_offs_n < Nx)[None, :]
        x = tl.load(x_ptr + x_offs, mask=x_mask).to(tl.float32)

        # Calculate scale
        amax = tl.max(tl.abs(x), axis=1, keep_dims=True)
        amax = amax.to(tl.int32, bitcast=True)
        amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
        amax = amax.to(tl.float32, bitcast=True)
        scale_e8m0_unbiased = tl.log2(amax).floor() - 2
        scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
        quant_scale = tl.exp2(-scale_e8m0_unbiased)

        # Compute quantized x
        qx = x * quant_scale

        # blockscale_e8m0
        # bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127

        # Convert quantized fp32 tensor to uint32 before converting to mxfp4 format
        # Note: MXFP4  S:1-bit, E:2-bit, M:1-bit
        #   Zeros: S000 -> +/-0
        #   Denormal Numbers: S001 -> +/- 0.5
        #   Normal Numbers:
        #           S010 -> +/- 1.0
        #           S011 -> +/- 1.5
        #           S100 -> +/- 2.0
        #           S101 -> +/- 3.0
        #           S110 -> +/- 4.0
        #           S111 -> +/- 6.0
        qx = qx.to(tl.uint32, bitcast=True)

        # Extract sign, exponents and mantissa fields from FP32
        s = qx & 0x80000000
        e = (qx >> 23) & 0xFF
        m = qx & 0x7FFFFF

        E8_BIAS: tl.constexpr = 127
        E2_BIAS: tl.constexpr = 1

        # Denormal numbers
        # If exponent is less than 127, then it's a denormal number
        # See above, for denormal number mantissa is always 1 and we set bit 1 of mantissa
        adjusted_exponents = tl.core.sub(E8_BIAS, e + 1, sanitize_overflow=False)
        m = tl.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)

        # For normal numbers, bias is changed from 127 to 1, and for subnormals, we keep exponent as 0.
        # Note: E8_BIAS - E2_BIAS = 126, so for normals we subtract that.
        e = tl.maximum(e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

        # Combine sign, exponent, and mantissa, while saturating
        # rounding nearest with tie breaking up by adding +1 to one bit right of the LSB, then shift right
        e2m1_tmp = tl.minimum((((e << 2) | (m >> 21)) + 1) >> 1, 0x7)
        e2m1_value = ((s >> 28) | e2m1_tmp).to(tl.uint8)

        e2m1_value = tl.reshape(
            e2m1_value, [BLOCK_SIZE_Mx, MXFP4_QUANT_BLOCK_SIZE // 2, 2]
        )
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)

        out_offs_m = pid_m * BLOCK_SIZE_Mx + tl.arange(0, BLOCK_SIZE_Mx)
        out_offs_n = pid_n * MXFP4_QUANT_BLOCK_SIZE // 2 + tl.arange(
            0, MXFP4_QUANT_BLOCK_SIZE // 2
        )
        out_offs = (
            out_offs_m[:, None] * stride_x_fp4_m + out_offs_n[None, :] * stride_x_fp4_n
        )
        out_mask = (out_offs_m < Mx)[:, None] & (out_offs_n < (Nx // 2))[None, :]
        tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask)

        return

    pid -= num_pid_x
    num_pid_n = tl.cdiv(N_i, BLOCK_SIZE_N * 2)
    pid_m = pid // num_pid_n  # * 2
    pid_n = pid % num_pid_n  # * 2
    # pid_m = tl.program_id(0) * 2
    # pid_n = tl.program_id(1) * 2
    num_valid_ids = tl.load(num_valid_ids_ptr)
    if pid_m * BLOCK_SIZE_M * 2 >= num_valid_ids:
        return
    stride_o0 = tl.cast(stride_o0, tl.int64)
    stride_o1 = tl.cast(stride_o1, tl.int64)
    stride_o2 = tl.cast(stride_o2, tl.int64)
    stride_o3 = tl.cast(stride_o3, tl.int64)
    stride_o4 = tl.cast(stride_o4, tl.int64)

    BLOCK_SIZE_Nb: tl.constexpr = BLOCK_SIZE_N * 2 * MXFP4_QUANT_BLOCK_SIZE
    sorted_ids_offs_m = pid_m * BLOCK_SIZE_M * 2 + tl.arange(0, BLOCK_SIZE_M * 2)
    sorted_ids_offs = sorted_ids_offs_m
    sorted_ids_mask = sorted_ids_offs_m < num_valid_ids
    sorted_ids = tl.load(
        sorted_ids_ptr + sorted_ids_offs,
        mask=sorted_ids_mask,
        other=token_num,
        # sorted_ids_ptr + sorted_ids_offs, mask=sorted_ids_mask, other=Mx
    )
    topk_ids = sorted_ids >> 24
    sorted_ids = sorted_ids & 0xFFFFFF
    if TOPK == 1:
        x_offs_m = sorted_ids
    else:
        x_offs_m = sorted_ids * TOPK + topk_ids
    # if pid == 0:
    #     tl.device_print("x_offs_m", x_offs_m)
    x_offs_n = pid_n * BLOCK_SIZE_Nb + tl.arange(0, BLOCK_SIZE_Nb)
    x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n
    x_mask = (sorted_ids < token_num)[:, None] & (x_offs_n < Nx)[None, :]
    # x_mask = (x_offs_m < Mx)[:, None] & (x_offs_n < Nx)[None, :]
    x = tl.load(x_ptr + x_offs, mask=x_mask).to(tl.float32)
    x = x.reshape(BLOCK_SIZE_M * 2, BLOCK_SIZE_N * 2, MXFP4_QUANT_BLOCK_SIZE)

    # Calculate scale
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    # blockscale_e8m0
    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127
    bs_e8m0 = (
        bs_e8m0.reshape(2, BLOCK_SIZE_M, 2, BLOCK_SIZE_N)
        .permute(1, 3, 2, 0)
        .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N, 4)
    )
    out = bs_e8m0

    # Store the result
    # 16x4 uint32 -> 32x2 uint8
    offs_0 = tl.arange(0, BLOCK_SIZE_M)
    offs_1 = tl.arange(0, BLOCK_SIZE_N)
    offs_2 = pid_n  # // 2
    offs_3 = pid_m  # // 2
    offs_4 = tl.arange(0, 4)
    offs = (
        offs_0[:, None, None] * stride_o0
        + offs_1[None, :, None] * stride_o1  # * BLOCK_SIZE_M
        + offs_2 * stride_o2  # * BLOCK_SIZE_M * BLOCK_SIZE_N
        + offs_3 * stride_o3  # * BLOCK_SIZE_M * BLOCK_SIZE_N * N_i // BLOCK_SIZE_N
        + offs_4[None, None, :] * stride_o4
    )
    # blockscale_e8m0_sorted_mask = (blockscale_e8m0_sorted_offs_m < M_o)[:, None] & (
    #     blockscale_e8m0_sorted_offs_n < N_o
    # )[None, :]
    tl.store(
        blockscale_e8m0_sorted_ptr + offs,
        out,
        # mask=blockscale_e8m0_sorted_mask,
    )
