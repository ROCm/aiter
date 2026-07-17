from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _mxfp4_quant_op(
    x,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: gl.constexpr,
):
    """
    Converts given x (in fp32) to mxfp4 format.
    x: [BLOCK_SIZE_M, BLOCK_SIZE_N], fp32

    """
    EXP_BIAS_FP32: gl.constexpr = 127
    EXP_BIAS_FP4: gl.constexpr = 1
    EBITS_F32: gl.constexpr = 8
    EBITS_FP4: gl.constexpr = 2
    MBITS_F32: gl.constexpr = 23
    MBITS_FP4: gl.constexpr = 1

    max_normal: gl.constexpr = 6
    min_normal: gl.constexpr = 1
    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)
    # Calculate scale
    amax = gl.max(gl.abs(x), axis=-1, keep_dims=True)
    amax = amax.to(gl.int32, bitcast=True)
    amax = (amax + 0x200000).to(gl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(gl.float32, bitcast=True)
    scale_e8m0_unbiased = gl.log2(amax).floor() - 2
    scale_e8m0_unbiased = gl.maximum(-127, gl.minimum(scale_e8m0_unbiased, 127))

    # blockscale_e8m0
    bs_e8m0 = scale_e8m0_unbiased.to(gl.uint8) + 127  # in fp32, we have 2&(e - 127)

    quant_scale = gl.exp2(-scale_e8m0_unbiased)

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
    qx = qx.to(gl.uint32, bitcast=True)

    # Extract sign
    s = qx & 0x80000000
    # Set everything to positive, will add sign back at the end
    qx = qx ^ s

    qx_fp32 = qx.to(gl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (~saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = ~(saturate_mask | denormal_mask)

    # Denormal numbers
    denorm_exp: gl.constexpr = (
        (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    )
    denorm_mask_int: gl.constexpr = denorm_exp << MBITS_F32
    denorm_mask_float: gl.constexpr = gl.cast(denorm_mask_int, gl.float32, bitcast=True)

    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(gl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(gl.uint8)

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
    normal_x = normal_x.to(gl.uint8)

    # Merge results
    e2m1_value = gl.full(qx.type.get_block_shapes(), 0x7, dtype=gl.uint8)
    e2m1_value = gl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = gl.where(denormal_mask, denormal_x, e2m1_value)
    # add sign back
    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(gl.uint8)
    e2m1_value = e2m1_value | sign_lp
    e2m1_value = gl.reshape(
        e2m1_value, [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2]
    )
    shift = gl.reshape(gl.arange(0, 2) * 4, [1, 1, 1, 2])
    x_fp4 = gl.sum((e2m1_value.to(gl.int32) << shift), axis=-1).to(gl.uint8)
    x_fp4 = x_fp4.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)

@gluon.jit
def gluon_dynamic_mxfp4_quant_kernel_gfx1250(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_x_fp4_m_in,
    stride_x_fp4_n_in,
    stride_bs_m_in,
    stride_bs_n_in,
    M,
    N,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    NUM_ITER: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    num_warps: gl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: gl.constexpr,
    EVEN_M_N: gl.constexpr,
    SCALING_MODE: gl.constexpr,
    NUM_BUFFERS: gl.constexpr = 2,
):
    gl.static_assert(NUM_BUFFERS >= 2, "LDS kernel requires NUM_BUFFERS >= 2")

    pid_m = gl.program_id(0)
    start_n = gl.program_id(1) * NUM_ITER

    stride_x_m = gl.cast(stride_x_m_in, gl.int64)
    stride_x_n = gl.cast(stride_x_n_in, gl.int64)
    stride_x_fp4_m = gl.cast(stride_x_fp4_m_in, gl.int64)
    stride_x_fp4_n = gl.cast(stride_x_fp4_n_in, gl.int64)
    stride_bs_m = gl.cast(stride_bs_m_in, gl.int64)
    stride_bs_n = gl.cast(stride_bs_n_in, gl.int64)

    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE

    # LDS layout: row-major, vec=8 elements = 128-bit stores, padded to avoid bank conflicts
    SHARED_LAYOUT_X: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_N, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_N], [1, 0]
    )

    # Register layout for LDS reads: order=[1,0] = N fastest, matches row-major LDS
    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 8],
        warps_per_cta=[1, num_warps],
        order=[1, 0],
    )

    # LDS ring buffer
    x_buffer = gl.allocate_shared_memory(
        x_ptr.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_SIZE_M, BLOCK_SIZE_N],
        layout=SHARED_LAYOUT_X,
    )
    SHARED_LAYOUT_O: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_N // 2, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_N // 2], [1, 0]
    )
    out_smem = gl.allocate_shared_memory(
        x_fp4_ptr.type.element_ty,
        shape=[BLOCK_SIZE_M, BLOCK_SIZE_N // 2],
        layout=SHARED_LAYOUT_O,
    )

    # TDM descriptor: base at this CTA's (M, N) origin
    x_base = (
        x_ptr + pid_m * BLOCK_SIZE_M * stride_x_m + start_n * BLOCK_SIZE_N * stride_x_n
    )
    x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=x_base,
        shape=(M - pid_m * BLOCK_SIZE_M, N - start_n * BLOCK_SIZE_N),
        strides=(stride_x_m, stride_x_n),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        layout=SHARED_LAYOUT_X,
    )
    out_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=x_fp4_ptr,
        shape=(M, N // 2),
        strides=(stride_x_fp4_m, stride_x_fp4_n),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N // 2),
        layout=SHARED_LAYOUT_O,
    )

    load_idx = 0
    compute_idx = 0
    num_tiles = min(NUM_ITER, gl.cdiv(N, BLOCK_SIZE_N) - start_n)
    # ---- Prologue: fill NUM_BUFFERS-1 slots ----
    for _ in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_load(
            x_desc, [0, 0], x_buffer.index(load_idx % NUM_BUFFERS)
        )
        x_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            x_desc, add_offsets=[0, BLOCK_SIZE_N]
        )
        load_idx += 1

    # ---- Main loop: load tile ahead, wait for oldest, quant, store ----
    for _ in range(num_tiles - (NUM_BUFFERS - 1)):
        gl.amd.gfx1250.tdm.async_load(
            x_desc, [0, 0], x_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_wait(NUM_BUFFERS - 1)  # 1 TDM op/tile
        x_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            x_desc, add_offsets=[0, BLOCK_SIZE_N]
        )
        load_idx += 1

        x_reg = (
            x_buffer.index(compute_idx % NUM_BUFFERS)
            .load(layout=blocked_layout)
            .to(gl.float32)
        )

        out_fp4, bs_e8m0 = _mxfp4_quant_op(
            x_reg, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
        )

        pid_n = start_n + compute_idx
        out_smem.store(out_fp4)
        gl.barrier()
        gl.amd.gfx1250.tdm.async_store(
            out_desc,
            [pid_m * BLOCK_SIZE_M, pid_n * (BLOCK_SIZE_N // 2)],
            out_smem,
        )
        gl.amd.gfx1250.tdm.async_wait(0)

        bs_offs_m = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M)
        bs_offs_n = pid_n * NUM_QUANT_BLOCKS + gl.arange(0, NUM_QUANT_BLOCKS)
        bs_offs = bs_offs_m[:, None] * stride_bs_m + bs_offs_n[None, :] * stride_bs_n
        if EVEN_M_N:
            gl.store(bs_ptr + bs_offs, bs_e8m0)
        else:
            gl.store(
                bs_ptr + bs_offs,
                bs_e8m0,
                mask=(bs_offs_m < M)[:, None]
                & (
                    bs_offs_n
                    < (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
                )[None, :],
            )
        compute_idx += 1

    # ---- Epilogue: drain remaining NUM_BUFFERS-1 tiles ----
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait(NUM_BUFFERS - 2 - i)

        x_reg = (
            x_buffer.index(compute_idx % NUM_BUFFERS)
            .load(layout=blocked_layout)
            .to(gl.float32)
        )

        out_fp4, bs_e8m0 = _mxfp4_quant_op(
            x_reg, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
        )

        pid_n = start_n + compute_idx
        out_smem.store(out_fp4)
        gl.barrier()
        gl.amd.gfx1250.tdm.async_store(
            out_desc,
            [pid_m * BLOCK_SIZE_M, pid_n * (BLOCK_SIZE_N // 2)],
            out_smem,
        )
        gl.amd.gfx1250.tdm.async_wait(0)

        bs_offs_m = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M)
        bs_offs_n = pid_n * NUM_QUANT_BLOCKS + gl.arange(0, NUM_QUANT_BLOCKS)
        bs_offs = bs_offs_m[:, None] * stride_bs_m + bs_offs_n[None, :] * stride_bs_n
        if EVEN_M_N:
            gl.store(bs_ptr + bs_offs, bs_e8m0)
        else:
            gl.store(
                bs_ptr + bs_offs,
                bs_e8m0,
                mask=(bs_offs_m < M)[:, None]
                & (
                    bs_offs_n
                    < (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
                )[None, :],
            )
        compute_idx += 1
