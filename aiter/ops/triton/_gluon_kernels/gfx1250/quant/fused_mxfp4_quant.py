from triton.experimental import gluon
from triton.experimental.gluon import language as gl


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
        [BLOCK_SIZE_Mx // 8, 2],
        [2, 16],
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
        gl.amd.gfx1250.tdm.async_wait(0)

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
