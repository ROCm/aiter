import triton
from triton.experimental import gluon
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
from triton.experimental.gluon import language as gl


@gluon.jit
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


@triton.heuristics(
    {
        "EVEN_M_N": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
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
):
    start_pid = gl.program_id(0)
    # get number of programs to determine is 1 or 2 passes
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)

    # create block layouts
    gLayout2D: gl.constexpr = gl.BlockedLayout(
        [1, 2],  # sizePerThread
        [1, 32],  # threadsPerWarp
        [1, 4],  # warpsPerCTA
        [1, 0],  # order
    )

    gLayoutN: gl.constexpr = gl.SliceLayout(0, gLayout2D)

    # 2D shared layout for matrix rows; 1D shared layout for weight vectors
    sharedLayout2D: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    sharedLayoutN: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])

    # Tensor descriptors for first input and its weights
    x1_desec = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        x1_ptr,
        [M, N1],
        [x1_stride_m, 1],
        [BLOCK_SIZE_M, BLOCK_SIZE_N],
        sharedLayout2D,
    )

    # tensor descriptor for weight 1
    w1_desec = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        w1_ptr,
        [N1],
        [1],
        [BLOCK_SIZE_N],
        sharedLayoutN,
    )

    # Shared memory for first input and its weights
    smemX1 = gl.allocate_shared_memory(
        x1_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N], sharedLayout2D
    )
    smemW1 = gl.allocate_shared_memory(
        w1_ptr.dtype.element_ty, [BLOCK_SIZE_N], sharedLayoutN
    )

    # x1 load issued unconditionally for early latency hiding (OOB-safe for second-input programs)
    gl.amd.gfx1250.tdm.async_load(x1_desec, [start_pid * BLOCK_SIZE_M, 0], smemX1)

    # Tensor descriptor and shared memory for optional residual input
    if FIRST_INPUT_RES:
        res1_desec = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            res1_ptr,
            [M, N1],
            [res1_stride_m, 1],
            [BLOCK_SIZE_M, BLOCK_SIZE_N],
            sharedLayout2D,
        )

        smemRes1 = gl.allocate_shared_memory(
            res1_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N], sharedLayout2D
        )

        gl.amd.gfx1250.tdm.async_load(
            res1_desec, [start_pid * BLOCK_SIZE_M, 0], smemRes1
        )

        out_res1_desec = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            out_res1_ptr,
            [M, N1],
            [out_res1_stride_m, 1],
            [BLOCK_SIZE_M, BLOCK_SIZE_N],
            sharedLayout2D,
        )

        smemOutRes1 = gl.allocate_shared_memory(
            out_res1_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N], sharedLayout2D
        )

    # Second input path — programs with id >= num_pid_m handle x2
    if start_pid >= num_pid_m:
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
            smemX2 = gl.allocate_shared_memory(
                x2_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N2], sharedLayout2D
            )
            smemW2 = gl.allocate_shared_memory(
                w2_ptr.dtype.element_ty, [BLOCK_SIZE_N2], sharedLayoutN
            )
            start_pid -= num_pid_m

            gl.amd.gfx1250.tdm.async_load(
                x2_desec, [start_pid * BLOCK_SIZE_M, 0], smemX2
            )
            gl.amd.gfx1250.tdm.async_load(w2_desec, [0], smemW2)
            gl.amd.gfx1250.tdm.async_wait(0)

            x2 = smemX2.load(gLayout2D).to(gl.float32)
            w2 = smemW2.load(gLayoutN).to(gl.float32)
            w2 = w2.reshape(1, BLOCK_SIZE_N2)
            w2 = gl.convert_layout(w2, gLayout2D)
            norm2 = _rmsnorm_op(x2, w2, N2, eps2)

            # Store norm2 output via TDM
            out2_desec = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                out2_ptr,
                [M, N2],
                [out2_stride_m, 1],
                [BLOCK_SIZE_M, BLOCK_SIZE_N2],
                sharedLayout2D,
            )
            smemOut2 = gl.allocate_shared_memory(
                out2_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N2], sharedLayout2D
            )
            smemOut2.store(norm2.to(out2_ptr.dtype.element_ty))
            gl.amd.gfx1250.tdm.async_store(
                out2_desec, [start_pid * BLOCK_SIZE_M, 0], smemOut2
            )
            gl.amd.gfx1250.tdm.async_wait(0)
        return

    gl.amd.gfx1250.tdm.async_load(w1_desec, [0], smemW1)

    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE

    # Descriptor and smem for fp4 TDM async_store
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

    bs_offs_m = start_pid * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M)
    bs_offs_n = gl.arange(0, NUM_QUANT_BLOCKS)
    num_bs_cols = (N1 + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
    if SHUFFLE:
        bs_offs_0 = bs_offs_m[:, None] >> 5  # // 32
        bs_offs_1 = bs_offs_m[:, None] & 31  # % 32
        bs_offs_2 = bs_offs_1 & 15  # % 16
        bs_offs_1 = bs_offs_1 >> 4  # // 16
        bs_offs_3 = bs_offs_n[None, :] >> 3  # // 8
        bs_offs_4 = bs_offs_n[None, :] & 7  # % 8
        bs_offs_5 = bs_offs_4 & 3  # % 4
        bs_offs_4 = bs_offs_4 >> 2  # // 4
        bs_offs = (
            bs_offs_1
            + bs_offs_4 * 2
            + bs_offs_2 * 2 * 2
            + bs_offs_5 * 2 * 2 * 16
            + bs_offs_3 * 2 * 2 * 16 * 4
            + bs_offs_0 * 2 * 16 * SCALE_N_PAD
        )
        bs_mask_127 = (bs_offs_m < M)[:, None] & (bs_offs_n < num_bs_cols)[None, :]
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

    x1 = smemX1.load(gLayout2D).to(gl.float32)

    if FIRST_INPUT_RES:
        res1_loaded = smemRes1.load(gLayout2D).to(gl.float32)
        x1 = x1 + res1_loaded
        smemOutRes1.store(x1.to(out_res1_ptr.dtype.element_ty))
        gl.amd.gfx1250.tdm.async_store(
            out_res1_desec, [start_pid * BLOCK_SIZE_M, 0], smemOutRes1
        )

    w1 = smemW1.load(gLayoutN).to(gl.float32)
    w1 = w1.reshape(1, BLOCK_SIZE_N)
    w1 = gl.convert_layout(w1, gLayout2D)
    norm1 = _rmsnorm_op(x1, w1, N1, eps1)

    # Store unquantized output via TDM (optional)
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
        smemOut1.store(norm1.to(out1_ptr.dtype.element_ty))
        gl.amd.gfx1250.tdm.async_store(
            out1_desec, [start_pid * BLOCK_SIZE_M, 0], smemOut1
        )

    out1_fp4, bs_e8m0 = _mxfp4_quant_op(
        norm1, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
    )

    # Apply out-of-range scale mask
    if SHUFFLE:
        bs_e8m0 = gl.where(bs_mask_127, bs_e8m0, 127)

    smemOutFp4.store(out1_fp4)
    gl.amd.gfx1250.tdm.async_store(
        out1_fp4_desec, [start_pid * BLOCK_SIZE_M, 0], smemOutFp4
    )

    gl.store(
        out1_bs_ptr + bs_offs, bs_e8m0.to(out1_bs_ptr.type.element_ty), mask=bs_mask
    )

    gl.amd.gfx1250.tdm.async_wait(0)
