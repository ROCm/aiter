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
