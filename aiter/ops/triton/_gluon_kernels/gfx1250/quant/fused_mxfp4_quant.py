import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op


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

    # layout descriptors for the second input
    gLayout3d_spk2: gl.constexpr = gl.BlockedLayout(
        [1, 1, 2],
        [1, 1, 32],
        [1, 1, 4],
        [2, 1, 0],
    )

    gLayoutN3_2: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, gLayout3d_spk2))
    gLayout2d_x2: gl.constexpr = gl.SliceLayout(0, gLayout3d_spk2)
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

    # placeholder
    gl.amd.gfx1250.tdm.async_wait(0)
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
