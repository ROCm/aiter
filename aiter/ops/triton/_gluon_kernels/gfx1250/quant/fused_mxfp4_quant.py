import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@triton.jit
def _rmsnorm_op(
    row,
    weights,
    n_cols,
    epsilon,
):
    row_norm = row * row
    row_norm = gl.sum(row_norm, axis=-1, keep_dims=True)
    norm_factor = gl.rsqrt((row_norm / n_cols) + epsilon)
    rms_norm = row * norm_factor * weights
    return rms_norm


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
    e2m1_value = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
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
                norm2 = _rmsnorm_op(x2, w2, N2, eps2)
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

        norm1 = _rmsnorm_op(x1, w1, N1, eps1)

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
