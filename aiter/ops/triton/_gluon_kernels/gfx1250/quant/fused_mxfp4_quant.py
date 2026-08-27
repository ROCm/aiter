import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


# rms norm op copied from triton
@triton.jit
def _rmsmorm_op(row, weight, n_cols, epsilon):
    row_norm = row * row
    row_norm = tl.sum(row_norm, axis=-1)
    norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)
    if weight is not None:
        rms_norm = row * norm_factor[:, None] * weight
    else:
        rms_norm = row * norm_factor[:, None]
    return rms_norm


# mxfp4 quant op copied from triton and modified for gluon
@triton.jit
def _mxfp4_quant_op(
    x,
    BLOCK_SIZE_N,
    BLOCK_SIZE_M,
    MXFP4_QUANT_BLOCK_SIZE,
):
    """
    Converts given x (in fp32) to mxfp4 format.
    x: [BLOCK_SIZE_M, BLOCK_SIZE_N], fp32

    """
    EXP_BIAS_FP32: tl.constexpr = 127
    EXP_BIAS_FP4: tl.constexpr = 1
    EBITS_F32: tl.constexpr = 8
    EBITS_FP4: tl.constexpr = 2
    MBITS_F32: tl.constexpr = 23
    MBITS_FP4: tl.constexpr = 1

    max_normal: tl.constexpr = 6
    min_normal: tl.constexpr = 1

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)
    # Calculate scale
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)

    # blockscale_e8m0
    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127  # in fp32, we have 2&(e - 127)

    quant_scale = tl.exp2(-scale_e8m0_unbiased)

    # Compute quantized x
    qx = x * quant_scale

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

    # Extract sign
    s = qx & 0x80000000
    # Set everything to positive, will add sign back at the end
    qx = qx ^ s

    qx_fp32 = qx.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    # Denormal numbers
    denorm_exp: tl.constexpr = (
        (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    )
    denorm_mask_int: tl.constexpr = denorm_exp << MBITS_F32
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)

    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)

    # Normal numbers
    normal_x = qx
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
    normal_x = normal_x.to(tl.uint8)

    # Merge results
    e2m1_value = gl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)
    # add sign back
    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1_value = e2m1_value | sign_lp
    e2m1_value = tl.reshape(
        e2m1_value, [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2]
    )
    evens, odds = tl.split(e2m1_value)
    x_fp4 = evens | (odds << 4)
    x_fp4 = x_fp4.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)


@triton.heuristics(
    {
        "EVEN_M_N": lambda args: args["M"] % args["ROWS_PER_CTA"] == 0
        and args["N1"] % (args["BLOCK_SIZE_N"]) == 0,
    }
)
@gluon.jit
def _gluon_fused_rms_mxfp4_quant_kernel(
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
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_N2: gl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: gl.constexpr,
    HAS_SECOND_INPUT: gl.constexpr,
    FIRST_INPUT_RES: gl.constexpr,
    FIRST_INPUT_OUT: gl.constexpr,
    SCALE_N: gl.constexpr,
    SCALE_M_PAD: gl.constexpr,
    SCALE_N_PAD: gl.constexpr,
    SHUFFLE: gl.constexpr,
    SHUFFLE_PAD: gl.constexpr,
    EVEN_M_N: gl.constexpr,
    ROWS_PER_CTA: gl.constexpr,
):
    start_pid = gl.program_id(0)
    # Calculate numbers of grouped CTAs and the base row index
    num_pid_m = gl.cdiv(M, ROWS_PER_CTA)
    cta_base = start_pid * ROWS_PER_CTA

    # Layout descriptors for the first input
    X1_SPT: gl.constexpr = min(16, BLOCK_SIZE_N // 128)
    gLayout2D_x1: gl.constexpr = gl.BlockedLayout(
        [1, X1_SPT],
        [1, 32],
        [1, 4],
        [1, 0],
    )
    gLayoutN_x1: gl.constexpr = gl.SliceLayout(0, gLayout2D_x1)

    # Layout descriptors for the second input
    X2_SPT: gl.constexpr = min(16, BLOCK_SIZE_N2 // 128)
    gLayout2D_x2: gl.constexpr = gl.BlockedLayout(
        [1, X2_SPT],
        [1, 32],
        [1, 4],
        [1, 0],
    )

    gLayoutN_x2: gl.constexpr = gl.SliceLayout(0, gLayout2D_x2)
    sharedLayout2D: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    sharedLayoutN: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])

    x1_desec = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        x1_ptr,
        [M, N1],
        [x1_stride_m, 1],
        [BLOCK_SIZE_M, BLOCK_SIZE_N],
        sharedLayout2D,
    )
    w1_desec = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        w1_ptr,
        [N1],
        [1],
        [BLOCK_SIZE_N],
        sharedLayoutN,
    )

    smemX1 = gl.allocate_shared_memory(
        x1_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N], sharedLayout2D
    )
    smemW1 = gl.allocate_shared_memory(
        w1_ptr.dtype.element_ty, [BLOCK_SIZE_N], sharedLayoutN
    )

    # Creates tensor descriptors and preloads residual input and output into shared memory if present
    if FIRST_INPUT_RES:
        res1_desec = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            res1_ptr,
            [M, N1],
            [res1_stride_m, 1],
            [BLOCK_SIZE_M, BLOCK_SIZE_N],
            sharedLayout2D,
        )
        out_res1_desec = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            out_res1_ptr,
            [M, N1],
            [out_res1_stride_m, 1],
            [BLOCK_SIZE_M, BLOCK_SIZE_N],
            sharedLayout2D,
        )
        smemRes1 = gl.allocate_shared_memory(
            res1_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N], sharedLayout2D
        )
        smemOutRes1 = gl.allocate_shared_memory(
            out_res1_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N], sharedLayout2D
        )

    # Handles second input path if present
    if HAS_SECOND_INPUT:
        x2_desec = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            x2_ptr,
            [M, N2],
            [x2_stride_m, 1],
            [BLOCK_SIZE_M, BLOCK_SIZE_N2],
            sharedLayout2D,
        )
        w2_desec = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            w2_ptr,
            [N2],
            [1],
            [BLOCK_SIZE_N2],
            sharedLayoutN,
        )
        out2_desec = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            out2_ptr,
            [M, N2],
            [out2_stride_m, 1],
            [BLOCK_SIZE_M, BLOCK_SIZE_N2],
            sharedLayout2D,
        )
        smemX2 = gl.allocate_shared_memory(
            x2_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N2], sharedLayout2D
        )
        smemW2 = gl.allocate_shared_memory(
            w2_ptr.dtype.element_ty, [BLOCK_SIZE_N2], sharedLayoutN
        )
        smemOut2 = gl.allocate_shared_memory(
            out2_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N2], sharedLayout2D
        )

    # Checks if the current PID is in the second input path
    if start_pid >= num_pid_m:
        if HAS_SECOND_INPUT:
            x2_local_pid = start_pid - num_pid_m
            x2_cta_base = x2_local_pid * ROWS_PER_CTA

            gl.amd.gfx1250.tdm.async_load(x2_desec, [x2_cta_base, 0], smemX2)
            gl.amd.gfx1250.tdm.async_load(w2_desec, [0], smemW2)
            gl.amd.gfx1250.tdm.async_wait(0)

            w2 = smemW2.load(gLayoutN_x2).to(gl.float32)
            w2 = w2.reshape(1, BLOCK_SIZE_N2)
            w2 = gl.convert_layout(w2, gLayout2D_x2)

            for i in range(ROWS_PER_CTA):
                x2_row_abs = x2_cta_base + i
                x2 = smemX2.load(gLayout2D_x2).to(gl.float32)
                if i + 1 < ROWS_PER_CTA:
                    gl.amd.gfx1250.tdm.async_load(x2_desec, [x2_row_abs + 1, 0], smemX2)
                norm2 = _rmsmorm_op(x2, w2, N2, eps2)
                smemOut2.store(norm2.to(out2_ptr.dtype.element_ty))
                gl.amd.gfx1250.tdm.async_store(out2_desec, [x2_row_abs, 0], smemOut2)
                gl.amd.gfx1250.tdm.async_wait(0)
        return

    gl.amd.gfx1250.tdm.async_load(x1_desec, [cta_base, 0], smemX1)
    if FIRST_INPUT_RES:
        gl.amd.gfx1250.tdm.async_load(res1_desec, [cta_base, 0], smemRes1)
    gl.amd.gfx1250.tdm.async_load(w1_desec, [0], smemW1)

    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE

    out1_fp4_desec = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        out1_fp4_ptr,
        [M, N1 // 2],
        [out1_fp4_stride_m, 1],
        [BLOCK_SIZE_M, BLOCK_SIZE_N // 2],
        sharedLayout2D,
    )
    smemOutFp4 = gl.allocate_shared_memory(
        out1_fp4_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2], sharedLayout2D
    )

    if FIRST_INPUT_OUT:
        out1_desec = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            out1_ptr,
            [M, N1],
            [out1_stride_m, 1],
            [BLOCK_SIZE_M, BLOCK_SIZE_N],
            sharedLayout2D,
        )
        smemOut1 = gl.allocate_shared_memory(
            out1_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N], sharedLayout2D
        )

    num_bs_cols = (N1 + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
    bs_offs_n = gl.arange(0, NUM_QUANT_BLOCKS)
    gl.amd.gfx1250.tdm.async_wait(0)

    w1 = smemW1.load(gLayoutN_x1).to(gl.float32)
    w1 = w1.reshape(1, BLOCK_SIZE_N)
    w1 = gl.convert_layout(w1, gLayout2D_x1)

    # Loop through each row in the CTA
    for i in range(ROWS_PER_CTA):
        row_abs = cta_base + i
        bs_offs_m_i = row_abs + gl.arange(0, BLOCK_SIZE_M)

        # Blockscale offset computation.
        if SHUFFLE:
            bs_offs_0 = bs_offs_m_i[:, None] >> 5
            bs_offs_1 = bs_offs_m_i[:, None] & 31
            bs_offs_2 = bs_offs_1 & 15
            bs_offs_1 = bs_offs_1 >> 4
            bs_offs_3 = bs_offs_n[None, :] >> 3
            bs_offs_4 = bs_offs_n[None, :] & 7
            bs_offs_5 = bs_offs_4 & 3
            bs_offs_4 = bs_offs_4 >> 2
            bs_offs_i = (
                bs_offs_1
                + bs_offs_4 * 2
                + bs_offs_2 * 2 * 2
                + bs_offs_5 * 2 * 2 * 16
                + bs_offs_3 * 2 * 2 * 16 * 4
                + bs_offs_0 * 2 * 16 * SCALE_N_PAD
            )
            bs_mask_127_i = (bs_offs_m_i < M)[:, None] & (bs_offs_n < num_bs_cols)[
                None, :
            ]
        else:
            bs_offs_i = (
                bs_offs_m_i[:, None] * out1_bs_stride_m
                + bs_offs_n[None, :] * out1_bs_stride_n
            )

        bs_mask_i = None
        if not EVEN_M_N:
            if SHUFFLE_PAD:
                bs_mask_i = (bs_offs_m_i < SCALE_M_PAD)[:, None] & (
                    bs_offs_n < SCALE_N_PAD
                )[None, :]
            else:
                bs_mask_i = (bs_offs_m_i < M)[:, None] & (bs_offs_n < SCALE_N)[None, :]

        x1 = smemX1.load(gLayout2D_x1).to(gl.float32)
        if FIRST_INPUT_RES:
            res1_loaded = smemRes1.load(gLayout2D_x1).to(gl.float32)

        # Fetch the next row while the current row is being processed
        if i + 1 < ROWS_PER_CTA:
            gl.amd.gfx1250.tdm.async_load(x1_desec, [row_abs + 1, 0], smemX1)
            if FIRST_INPUT_RES:
                gl.amd.gfx1250.tdm.async_load(res1_desec, [row_abs + 1, 0], smemRes1)

        if FIRST_INPUT_RES:
            x1 = x1 + res1_loaded
            smemOutRes1.store(x1.to(out_res1_ptr.dtype.element_ty))
            gl.amd.gfx1250.tdm.async_store(out_res1_desec, [row_abs, 0], smemOutRes1)

        norm1 = _rmsmorm_op(x1, w1, N1, eps1)

        if FIRST_INPUT_OUT:
            smemOut1.store(norm1.to(out1_ptr.dtype.element_ty))
            gl.amd.gfx1250.tdm.async_store(out1_desec, [row_abs, 0], smemOut1)

        out1_fp4, bs_e8m0 = _mxfp4_quant_op(
            norm1, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
        )

        if SHUFFLE:
            bs_e8m0 = gl.where(bs_mask_127_i, bs_e8m0, 127)

        # Store the quantized and blockscale values
        smemOutFp4.store(out1_fp4)
        gl.amd.gfx1250.tdm.async_store(out1_fp4_desec, [row_abs, 0], smemOutFp4)
        gl.store(
            out1_bs_ptr + bs_offs_i,
            bs_e8m0.to(out1_bs_ptr.type.element_ty),
            mask=bs_mask_i,
        )

        gl.amd.gfx1250.tdm.async_wait(0)


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
@gluon.jit
def _gluon_fused_reduce_rms_mxfp4_quant_kernel(
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
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_N2: gl.constexpr,
    BLOCK_SIZE_N3: gl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: gl.constexpr,
    HAS_SECOND_INPUT: gl.constexpr,
    FIRST_INPUT_RES: gl.constexpr,
    FIRST_INPUT_OUT: gl.constexpr,
    HAS_SPLITK: gl.constexpr,
    NUM_SPLITK: gl.constexpr,
    NUM_SPLITK_POW2: gl.constexpr,
    SCALE_N: gl.constexpr,
    SCALE_M_PAD: gl.constexpr,
    SCALE_N_PAD: gl.constexpr,
    SHUFFLE: gl.constexpr,
    SHUFFLE_PAD: gl.constexpr,
    EVEN_M_N: gl.constexpr,
    EVEN_M_N2: gl.constexpr,
    EVEN_M_N3: gl.constexpr,
):
    # setup grid 1d
    start_pid = gl.program_id(0)
    # first dimension is the same as input tensor since BLOCK_SIZE_M is hardcoded to 1
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)

    # layout descriptors for the first input
    gLayout3d_spk: gl.constexpr = gl.BlockedLayout(
        [1, 1, 2],
        [1, 1, 32],
        [1, 1, 4],
        [2, 1, 0],
    )

    # memory layout descriptors for the main branch
    sharedLayout2D: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    sharedLayoutN: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])

    gLayoutN3_2: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, gLayout3d_spk))
    gLayout2d_x2: gl.constexpr = gl.SliceLayout(0, gLayout3d_spk)
    gLayoutN2: gl.constexpr = gl.SliceLayout(0, gLayout2d_x2)

    # TDM descriptor for main branch
    w1_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        w1_ptr,
        [N1],
        [1],
        [BLOCK_SIZE_N],
        sharedLayoutN,
    )
    out1_fp4_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        out1_fp4_ptr,
        [M, N1 // 2],
        [out1_fp4_stride_m, 1],
        [BLOCK_SIZE_M, BLOCK_SIZE_N // 2],
        sharedLayout2D,
    )

    if FIRST_INPUT_RES:
        res1_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            res1_ptr,
            [M, N1],
            [res1_stride_m, 1],
            [BLOCK_SIZE_M, BLOCK_SIZE_N],
            sharedLayout2D,
        )
        out_res1_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            out_res1_ptr,
            [M, N1],
            [out_res1_stride_m, 1],
            [BLOCK_SIZE_M, BLOCK_SIZE_N],
            sharedLayout2D,
        )
    if FIRST_INPUT_OUT:
        out1_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            out1_ptr,
            [M, N1],
            [out1_stride_m, 1],
            [BLOCK_SIZE_M, BLOCK_SIZE_N],
            sharedLayout2D,
        )
    if HAS_SPLITK:
        out3_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            out3_ptr,
            [M, N3],
            [out3_stride_m, 1],
            [BLOCK_SIZE_M, BLOCK_SIZE_N3],
            sharedLayout2D,
        )

    smemW1 = gl.allocate_shared_memory(
        w1_ptr.dtype.element_ty,
        [BLOCK_SIZE_N],
        sharedLayoutN,
    )
    smemOut1_fp4 = gl.allocate_shared_memory(
        out1_fp4_ptr.dtype.element_ty,
        [BLOCK_SIZE_M, BLOCK_SIZE_N // 2],
        sharedLayout2D,
    )
    if FIRST_INPUT_RES:
        smemRes1 = gl.allocate_shared_memory(
            res1_ptr.dtype.element_ty,
            [BLOCK_SIZE_M, BLOCK_SIZE_N],
            sharedLayout2D,
        )
        smemOut_res1 = gl.allocate_shared_memory(
            out_res1_ptr.dtype.element_ty,
            [BLOCK_SIZE_M, BLOCK_SIZE_N],
            sharedLayout2D,
        )

    if FIRST_INPUT_OUT:
        smemOut1 = gl.allocate_shared_memory(
            out1_ptr.dtype.element_ty,
            [BLOCK_SIZE_M, BLOCK_SIZE_N],
            sharedLayout2D,
        )

    if HAS_SPLITK:
        smemOut3 = gl.allocate_shared_memory(
            out3_ptr.dtype.element_ty,
            [BLOCK_SIZE_M, BLOCK_SIZE_N3],
            sharedLayout2D,
        )

    gLayoutSPK: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, gLayout3d_spk))
    gLayoutM3: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, gLayout3d_spk))
    gLayoutN3: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, gLayout3d_spk))

    # 2d layouts
    gLayout2d: gl.constexpr = gl.SliceLayout(0, gLayout3d_spk)
    gLayoutM: gl.constexpr = gl.SliceLayout(1, gLayout2d)
    gLayoutN: gl.constexpr = gl.SliceLayout(0, gLayout2d)

    if start_pid >= 2 * num_pid_m:
        start_pid -= 2 * num_pid_m
        if HAS_SPLITK:
            spk_offs = gl.arange(0, NUM_SPLITK_POW2, layout=gLayoutSPK)
            x_offs_m = start_pid * BLOCK_SIZE_M + gl.arange(
                0, BLOCK_SIZE_M, layout=gLayoutM3
            )
            x_offs_n3 = gl.arange(0, BLOCK_SIZE_N3, layout=gLayoutN3)
            mask3 = None
            other3 = None
            if not EVEN_M_N3:
                other3 = 0.0
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

            x3 = gl.amd.gfx1250.buffer_load(
                x3_ptr,
                spk_offs[:, None, None] * x3_stride_spk
                + x_offs_m[None, :, None] * x3_stride_m
                + x_offs_n3[None, None, :],
                mask3,
                other3,
            ).to(gl.float32)

            x3 = gl.sum(x3, axis=0)

            start_row = start_pid * BLOCK_SIZE_M
            smemOut3.store(x3.to(out3_ptr.dtype.element_ty))
            gl.amd.gfx1250.tdm.async_store(out3_desc, [start_row, 0], smemOut3)
            gl.amd.gfx1250.tdm.async_wait(0)

        return

    if start_pid >= num_pid_m:
        start_pid -= num_pid_m
        if HAS_SECOND_INPUT:
            w2_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                w2_ptr,
                [N2],
                [1],
                [BLOCK_SIZE_N2],
                sharedLayoutN,
            )
            out2_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                out2_ptr,
                [M, N2],
                [out2_stride_m, 1],
                [BLOCK_SIZE_M, BLOCK_SIZE_N2],
                sharedLayout2D,
            )
            smemW2 = gl.allocate_shared_memory(
                w2_ptr.dtype.element_ty,
                [BLOCK_SIZE_N2],
                sharedLayoutN,
            )
            smemOut2 = gl.allocate_shared_memory(
                out2_ptr.dtype.element_ty,
                [BLOCK_SIZE_M, BLOCK_SIZE_N2],
                sharedLayout2D,
            )
            if HAS_SPLITK:
                spk_offs = gl.arange(0, NUM_SPLITK_POW2, layout=gLayoutSPK)
            x_offs_m = start_pid * BLOCK_SIZE_M + gl.arange(
                0, BLOCK_SIZE_M, layout=gLayoutM3
            )
            x_offs_n2 = gl.arange(0, BLOCK_SIZE_N2, layout=gLayoutN3_2)
            x_offs_m_2d = gl.convert_layout(x_offs_m, gLayoutM)
            x_offs_n2_2d = gl.convert_layout(x_offs_n2, gLayoutN2)
            start_row = start_pid * BLOCK_SIZE_M
            gl.amd.gfx1250.tdm.async_load(
                w2_desc,
                [0],
                smemW2,
            )
            mask2 = None
            other2 = None

            if HAS_SPLITK:
                if not EVEN_M_N2:
                    other2 = 0.0
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

            else:
                if not EVEN_M_N2:
                    other2 = 0.0
                    mask2 = (x_offs_m_2d < M)[:, None] & (x_offs_n2_2d < N2)[None, :]

            if HAS_SPLITK:
                x2 = gl.amd.gfx1250.buffer_load(
                    x2_ptr,
                    spk_offs[:, None, None] * x2_stride_spk
                    + x_offs_m[None, :, None] * x2_stride_m
                    + x_offs_n2[None, None, :],
                    mask2,
                    other2,
                    ".cg",
                ).to(gl.float32)
                x2 = gl.sum(x2, axis=0)
            else:
                x2 = gl.amd.gfx1250.buffer_load(
                    x2_ptr,
                    x_offs_m_2d[:, None] * x2_stride_m + x_offs_n2_2d[None, :],
                    mask2,
                    other2,
                    ".cg",
                ).to(gl.float32)

            gl.amd.gfx1250.tdm.async_wait(0)
            w2 = smemW2.load(gLayoutN2).to(gl.float32)
            w2 = w2[None, :]
            norm2 = _rmsmorm_op(x2, w2, N2, eps2)

            smemOut2.store(norm2.to(out2_ptr.dtype.element_ty))
            gl.amd.gfx1250.tdm.async_store(out2_desc, [start_row, 0], smemOut2)
            gl.amd.gfx1250.tdm.async_wait(0)

        return

    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x_offs_n = gl.arange(0, BLOCK_SIZE_N, layout=gLayoutN3)
    x_offs_m = start_pid * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gLayoutM3)
    x_offs_m_2d = gl.convert_layout(x_offs_m, gLayoutM)
    x_offs_n_2d = gl.convert_layout(x_offs_n, gLayoutN)
    if HAS_SPLITK:
        spk_offs = gl.arange(0, NUM_SPLITK_POW2, layout=gLayoutSPK)

    start_row = start_pid * BLOCK_SIZE_M

    gl.amd.gfx1250.tdm.async_load(
        w1_desc,
        [0],
        smemW1,
    )
    if FIRST_INPUT_RES:
        gl.amd.gfx1250.tdm.async_load(
            res1_desc,
            [start_row, 0],
            smemRes1,
        )

    mask1 = None
    other1 = None
    if HAS_SPLITK:
        if not EVEN_M_N:
            other1 = 0.0
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
    else:
        if not EVEN_M_N:
            other1 = 0.0
            mask1 = (x_offs_m_2d < M)[:, None] & (x_offs_n_2d < N1)[None, :]

    if HAS_SPLITK:
        x1 = gl.amd.gfx1250.buffer_load(
            x1_ptr,
            spk_offs[:, None, None] * x1_stride_spk
            + x_offs_m[None, :, None] * x1_stride_m
            + x_offs_n[None, None, :],
            mask1,
            other1,
            ".cg",
        ).to(gl.float32)
        x1 = gl.sum(x1, axis=0)
    else:
        x1 = gl.amd.gfx1250.buffer_load(
            x1_ptr,
            x_offs_m_2d[:, None] * x1_stride_m + x_offs_n_2d[None, :],
            mask1,
            other1,
            ".cg",
        ).to(gl.float32)

    gl.amd.gfx1250.tdm.async_wait(0)
    gl.barrier()
    if FIRST_INPUT_RES:
        res1 = smemRes1.load(gLayout2d).to(gl.float32)
        x1 = x1 + res1

    w1 = smemW1.load(gLayoutN).to(gl.float32)
    w1 = w1[None, :]
    norm1 = _rmsmorm_op(x1, w1, N1, eps1)

    if FIRST_INPUT_OUT:
        smemOut1.store(norm1.to(out1_ptr.dtype.element_ty))
        gl.amd.gfx1250.tdm.async_store(out1_desc, [start_row, 0], smemOut1)

    out1_fp4, bs_e8m0 = _mxfp4_quant_op(
        norm1, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
    )

    # store the results
    smemOut1_fp4.store(out1_fp4)
    gl.amd.gfx1250.tdm.async_store(out1_fp4_desc, [start_row, 0], smemOut1_fp4)

    # Quantization block store

    bs_offs_m = start_pid * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M)
    bs_offs_n = gl.arange(0, NUM_QUANT_BLOCKS)
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
        bs_e8m0 = gl.where(bs_mask_127, bs_e8m0, 127)
    else:
        bs_offs = (
            bs_offs_m[:, None] * out1_bs_stride_m
            + bs_offs_n[None, :] * out1_bs_stride_n
        )

    # Shuffle optional check
    bs_mask = None
    if not EVEN_M_N:
        if SHUFFLE_PAD:
            bs_mask = (bs_offs_m < SCALE_M_PAD)[:, None] & (bs_offs_n < SCALE_N_PAD)[
                None, :
            ]
        else:
            bs_mask = (bs_offs_m < M)[:, None] & (bs_offs_n < SCALE_N)[None, :]

    gl.amd.gfx1250.buffer_store(
        bs_e8m0.to(out1_bs_ptr.dtype.element_ty),
        out1_bs_ptr,
        bs_offs,
        bs_mask,
    )

    if FIRST_INPUT_RES:
        smemOut_res1.store(x1.to(out_res1_ptr.dtype.element_ty))
        gl.amd.gfx1250.tdm.async_store(out_res1_desc, [start_row, 0], smemOut_res1)

    gl.amd.gfx1250.tdm.async_wait(0)


@gluon.jit
def _gluon_fused_dynamic_mxfp4_quant_moe_sort_kernel(
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
    stride_o3,  #: gl.constexpr,
    stride_o2,  #: gl.constexpr,
    stride_o1,  #: gl.constexpr,
    stride_o0,  #: gl.constexpr,
    stride_o4,  #: gl.constexpr,
    token_num,  #: gl.constexpr,
    N_i,  #: gl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: gl.constexpr,
    BLOCK_SIZE_Mx: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    TOPK: gl.constexpr,
):

    # pid contains both phase 1 and phase 2 ids
    pid = gl.program_id(0)

    # number of phase 1 ids (quantize stage)
    num_pid_x = gl.cdiv(Mx, BLOCK_SIZE_Mx) * scaleNx

    # block size for the second input (sort stage)
    BLOCK_SIZE_Nb: gl.constexpr = BLOCK_SIZE_N * 2 * MXFP4_QUANT_BLOCK_SIZE

    # layout descriptor for quantize stage
    gLayout2d: gl.constexpr = gl.BlockedLayout(
        [BLOCK_SIZE_Mx // 32, 8],
        [8, 4],
        [4, 1],
        [1, 0],
    )

    # layout descriptor for sort stage
    gLayout1D_id: gl.constexpr = gl.BlockedLayout(
        [1],
        [32],
        [4],
        [0],
    )

    # layout descriptor for sort stage
    gLayout2d_phase2: gl.constexpr = gl.BlockedLayout(
        [4, 16],
        [2, 16],
        [4, 1],
        [1, 0],
    )

    x_row_layout: gl.constexpr = gl.SliceLayout(1, gLayout2d_phase2)
    x_col_layout: gl.constexpr = gl.SliceLayout(0, gLayout2d_phase2)

    # memory layouts
    sharedLayout2D: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    # tdm descriptors
    x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        x_ptr,
        [Mx, Nx],
        [stride_x_m, stride_x_n],
        [BLOCK_SIZE_Mx, MXFP4_QUANT_BLOCK_SIZE],
        sharedLayout2D,
    )

    x_fp4_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        x_fp4_ptr,
        [Mx, Nx // 2],
        [stride_x_fp4_m, stride_x_fp4_n],
        [BLOCK_SIZE_Mx, MXFP4_QUANT_BLOCK_SIZE // 2],
        sharedLayout2D,
    )

    # shared memory for the input tensor
    smem_x = gl.allocate_shared_memory(
        x_ptr.dtype.element_ty, [BLOCK_SIZE_Mx, MXFP4_QUANT_BLOCK_SIZE], sharedLayout2D
    )
    # shared memory for the quantized tensor
    smem_x_fp4 = gl.allocate_shared_memory(
        gl.uint8, [BLOCK_SIZE_Mx, MXFP4_QUANT_BLOCK_SIZE // 2], sharedLayout2D
    )

    stride_x_m = gl.cast(stride_x_m, gl.int64)
    stride_x_n = gl.cast(stride_x_n, gl.int64)
    stride_x_fp4_m = gl.cast(stride_x_fp4_m, gl.int64)
    stride_x_fp4_n = gl.cast(stride_x_fp4_n, gl.int64)

    # phase 1: quantize the input tensor
    if pid < num_pid_x:
        pid_m = pid // scaleNx
        pid_n = pid % scaleNx

        gl.amd.gfx1250.tdm.async_load(
            x_desc,
            [pid_m * BLOCK_SIZE_Mx, pid_n * MXFP4_QUANT_BLOCK_SIZE],
            smem_x,
        )
        gl.amd.gfx1250.tdm.async_wait(0)
        x = smem_x.load(gLayout2d).to(gl.float32)

        # Calculate scale
        amax = gl.max(gl.abs(x), axis=1, keep_dims=True)
        amax = amax.to(gl.int32, bitcast=True)
        amax = (amax + 0x200000).to(gl.uint32, bitcast=True) & 0xFF800000
        amax = amax.to(gl.float32, bitcast=True)
        scale_e8m0_unbiased = gl.log2(amax).floor() - 2
        scale_e8m0_unbiased = gl.maximum(gl.minimum(scale_e8m0_unbiased, 127), -127)
        quant_scale = gl.exp2(-scale_e8m0_unbiased)

        # Compute quantized x
        qx = x * quant_scale

        # blockscale_e8m0
        # bs_e8m0 = scale_e8m0_unbiased.to(gl.uint8) + 127

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
        qx = qx.to(gl.uint32, bitcast=True)

        # Extract sign, exponents and mantissa fields from FP32
        s = qx & 0x80000000
        e = (qx >> 23) & 0xFF
        m = qx & 0x7FFFFF

        E8_BIAS: gl.constexpr = 127
        E2_BIAS: gl.constexpr = 1

        # Denormal numbers
        # If exponent is less than 127, then it's a denormal number
        # See above, for denormal number mantissa is always 1 and we set bit 1 of mantissa
        adjusted_exponents = gl.sub(E8_BIAS, e + 1, sanitize_overflow=False)
        m = gl.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)

        # For normal numbers, bias is changed from 127 to 1, and for subnormals, we keep exponent as 0.
        # Note: E8_BIAS - E2_BIAS = 126, so for normals we subtract that.
        e = gl.maximum(e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

        # Combine sign, exponent, and mantissa, while saturating
        # rounding nearest with tie breaking up by adding +1 to one bit right of the LSB, then shift right
        e2m1_tmp = gl.minimum((((e << 2) | (m >> 21)) + 1) >> 1, 0x7)
        e2m1_value = ((s >> 28) | e2m1_tmp).to(gl.uint8)

        e2m1_value = gl.reshape(
            e2m1_value, [BLOCK_SIZE_Mx, MXFP4_QUANT_BLOCK_SIZE // 2, 2]
        )
        evens, odds = gl.split(e2m1_value)
        out_tensor = evens | (odds << 4)

        smem_x_fp4.store(out_tensor)
        gl.amd.gfx1250.tdm.async_store(
            x_fp4_desc,
            [pid_m * BLOCK_SIZE_Mx, pid_n * MXFP4_QUANT_BLOCK_SIZE // 2],
            smem_x_fp4,
        )

        return

    # phase 2: sort block scale tensor
    pid -= num_pid_x
    num_pid_n = gl.cdiv(N_i, BLOCK_SIZE_N * 2)
    pid_m = pid // num_pid_n  # * 2
    pid_n = pid % num_pid_n  # * 2
    num_valid_ids = gl.load(num_valid_ids_ptr)
    if pid_m * BLOCK_SIZE_M * 2 >= num_valid_ids:
        return
    stride_o0 = gl.cast(stride_o0, gl.int64)
    stride_o1 = gl.cast(stride_o1, gl.int64)
    stride_o2 = gl.cast(stride_o2, gl.int64)
    stride_o3 = gl.cast(stride_o3, gl.int64)
    stride_o4 = gl.cast(stride_o4, gl.int64)

    sorted_ids_offs = pid_m * BLOCK_SIZE_M * 2 + gl.arange(
        0, BLOCK_SIZE_M * 2, gLayout1D_id
    )
    sorted_ids_mask = sorted_ids_offs < num_valid_ids
    sorted_ids = gl.amd.gfx1250.buffer_load(
        sorted_ids_ptr,
        sorted_ids_offs,
        sorted_ids_mask,
        token_num,
    ).to(gl.uint32)

    topk_ids = sorted_ids >> 24
    sorted_ids = sorted_ids & 0xFFFFFF
    if TOPK == 1:
        x_offs_m = sorted_ids
    else:
        x_offs_m = sorted_ids * TOPK + topk_ids

    x_offs_m_row = gl.convert_layout(x_offs_m, x_row_layout)
    row_valid = gl.convert_layout(sorted_ids < token_num, x_row_layout)
    x_offs_n = pid_n * BLOCK_SIZE_Nb + gl.arange(0, BLOCK_SIZE_Nb, x_col_layout)
    col_valid = x_offs_n < Nx

    x_offs_2d = (
        x_offs_m_row.to(gl.int64)[:, None] * stride_x_m
        + x_offs_n.to(gl.int64)[None, :] * stride_x_n
    ).to(gl.int32)
    x_mask_2d = row_valid[:, None] & col_valid[None, :]

    x = gl.amd.gfx1250.buffer_load(x_ptr, x_offs_2d, x_mask_2d, 0.0).to(gl.float32)
    x = x.reshape(BLOCK_SIZE_M * 2, BLOCK_SIZE_N * 2, MXFP4_QUANT_BLOCK_SIZE)

    # Calculate scale
    amax = gl.max(gl.abs(x), axis=-1, keep_dims=True)
    amax = amax.to(gl.int32, bitcast=True)
    amax = (amax + 0x200000).to(gl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(gl.float32, bitcast=True)
    scale_e8m0_unbiased = gl.log2(amax).floor() - 2
    scale_e8m0_unbiased = gl.maximum(gl.minimum(scale_e8m0_unbiased, 127), -127)
    # blockscale_e8m0
    bs_e8m0 = scale_e8m0_unbiased.to(gl.uint8) + 127
    bs_e8m0 = (
        bs_e8m0.reshape(2, BLOCK_SIZE_M, 2, BLOCK_SIZE_N)
        .permute(1, 3, 2, 0)
        .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N, 4)
    )
    out = bs_e8m0

    offs_0 = gl.arange(0, BLOCK_SIZE_M)
    offs_1 = gl.arange(0, BLOCK_SIZE_N)
    offs_4 = gl.arange(0, 4)

    offs = (
        offs_0[:, None, None] * stride_o0
        + offs_1[None, :, None] * stride_o1
        + pid_n * stride_o2
        + pid_m * stride_o3
        + offs_4[None, None, :] * stride_o4
    ).to(gl.int32)

    gl.store(blockscale_e8m0_sorted_ptr + offs, out)
