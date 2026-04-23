import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@triton.heuristics(
    {
        "UNROLL_TIMES": lambda args: triton.cdiv(args["K"], args["BLOCK_K"]),
    }
)
@gluon.jit
def _gluon_moe_gemm_int8_smoothquant(
    Y,
    stride_y_k,
    stride_y_m,
    stride_y_n,
    X,
    stride_x_m,
    stride_x_k,
    XScale,
    stride_x_scale,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WScale,
    stride_w_scale_e,
    stride_w_scale_n,
    B,
    stride_b_e,  # Bias
    Gammas,
    N,
    K,
    # expt data
    GatherIndx,
    ExptHist,
    ExptOffs,
    ExptOffsSum,
    ExptData,
    # true grid size
    grid_m,
    grid_n,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: gl.constexpr,
    APPLY_ACTIVATION: gl.constexpr,
    ADD_RESIDUAL: gl.constexpr,
    # MoE config
    N_EXPTS_ACT: gl.constexpr,
    # optimization config
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_M: gl.constexpr,
    EVEN_K: gl.constexpr,
    MASK_K_LIMIT: gl.constexpr,
    # Gluon-specific
    UNROLL_TIMES: gl.constexpr,
    num_warps: gl.constexpr,
):
    """
    Gluon-optimized Int8 MoE GEMM with SmoothQuant for small K dimensions.

    Key optimizations over the standard _moe_gemm_int8_smoothquant:
    - Manual LICM: A matrix, x_scale, and gammas pre-loaded outside N loop
    - K-dimension unrolling via gl.static_range eliminates loop overhead
    - Explicit BlockedLayout + MFMA instructions for optimal register usage
    - SUB_BLOCK_SIZE_N inner loop processes large BLOCK_N in 64-wide chunks
    """
    SUB_BLOCK_SIZE_N: gl.constexpr = 64

    # -- Layouts --
    # INT8 on CDNA3 uses v_mfma_i32_32x32x16_i8 instruction
    blocked_a: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[32, 2],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    blocked_b: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, num_warps],
        order=[0, 1],
    )
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )
    mfma_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    mfma_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    blocked_d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[16, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )

    # -- PID mapping (new path style) --
    pid = gl.program_id(axis=0)

    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # Unpack expert data
    expt_data = gl.load(ExptData + pid_m)
    if expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = gl.load(ExptHist + expt_id)
    start_m = gl.load(ExptOffs + expt_id)

    # -- A row offsets --
    offs_x_m_raw = block_id * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, blocked_a)
    )
    offs_x_m = offs_x_m_raw % M
    mask_m = offs_x_m_raw < M

    if GatherIndx is not None:
        # Indirect indexing via gather
        offs_x_m = (
            gl.amd.cdna3.buffer_load(
                GatherIndx + start_m, offs_x_m, mask=mask_m, other=0
            )
            // N_EXPTS_ACT
        )
    else:
        X += start_m * stride_x_m
        XScale += start_m * stride_x_scale

    offs_ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, blocked_a))
    offs_bk = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, blocked_b))

    # =====================================================
    # LICM: Pre-load all A matrix K-blocks outside N loop
    # =====================================================
    a_converted = ()
    for k in gl.static_range(UNROLL_TIMES):
        if EVEN_K:
            a = gl.amd.cdna3.buffer_load(
                X + k * BLOCK_K * stride_x_k,
                offs_x_m[:, None] * stride_x_m + offs_ak[None, :] * stride_x_k,
                mask=mask_m[:, None],
                other=0.0,
            )
        else:
            a = gl.amd.cdna3.buffer_load(
                X + k * BLOCK_K * stride_x_k,
                offs_x_m[:, None] * stride_x_m + offs_ak[None, :] * stride_x_k,
                mask=mask_m[:, None] & (offs_ak[None, :] < K - k * BLOCK_K),
                other=0.0,
            )
        a_converted = a_converted + (gl.convert_layout(a, mfma_a_layout),)

    # =====================================================
    # LICM: Pre-load per-token x_scale outside N loop
    # =====================================================
    x_scale = gl.amd.cdna3.buffer_load(
        XScale, offs_x_m * stride_x_scale, mask=mask_m, other=1.0
    )
    x_scale_converted = gl.convert_layout(x_scale, gl.SliceLayout(1, mfma_layout))

    # =====================================================
    # LICM: Pre-load gammas outside N loop
    # =====================================================
    if Gammas is not None:
        offs_gamma = block_id * BLOCK_M + gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(1, mfma_layout)
        )
        gamma_mask = offs_gamma < M
        gamma_vals = gl.amd.cdna3.buffer_load(
            Gammas + start_m,
            offs_gamma,
            mask=gamma_mask,
            other=0.0,
        )

    # =====================================================
    # N-dimension loop (SUB_BLOCK_SIZE_N chunks)
    # =====================================================
    W_base = W + expt_id * stride_w_e

    for n_start in range(0, BLOCK_N, SUB_BLOCK_SIZE_N):
        offs_bn = (
            pid_n * BLOCK_N
            + n_start
            + gl.arange(0, SUB_BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_b))
        ) % N

        # Accumulator in int32 for int8 x int8
        accumulator = gl.zeros(
            (BLOCK_M, SUB_BLOCK_SIZE_N), dtype=gl.int32, layout=mfma_layout
        )

        # -- Load B and compute MFMA (unrolled K via static_range) --
        for k in gl.static_range(UNROLL_TIMES):
            if EVEN_K:
                b = gl.amd.cdna3.buffer_load(
                    W_base + k * BLOCK_K * stride_w_k,
                    offs_bk[:, None] * stride_w_k + offs_bn[None, :] * stride_w_n,
                )
            else:
                b = gl.amd.cdna3.buffer_load(
                    W_base + k * BLOCK_K * stride_w_k,
                    offs_bk[:, None] * stride_w_k + offs_bn[None, :] * stride_w_n,
                    mask=offs_bk[:, None] < K - k * BLOCK_K,
                    other=0.0,
                )
            b_converted = gl.convert_layout(b, mfma_b_layout)
            accumulator = gl.amd.cdna3.mfma(a_converted[k], b_converted, accumulator)

        # -- Apply SmoothQuant scales: acc_fp32 = acc_int32 * x_scale * w_scale --
        # Load per-channel weight scale for this N sub-block
        offs_wscale_n = (
            pid_n * BLOCK_N
            + n_start
            + gl.arange(0, SUB_BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout))
        )
        w_scale = gl.amd.cdna3.buffer_load(
            WScale + expt_id * stride_w_scale_e,
            offs_wscale_n * stride_w_scale_n,
            mask=offs_wscale_n < N,
            other=1.0,
        )
        w_scale_converted = gl.convert_layout(w_scale, gl.SliceLayout(0, mfma_layout))
        acc_fp32 = (
            accumulator.to(gl.float32)
            * x_scale_converted[:, None]
            * w_scale_converted[None, :]
        )

        # -- Bias --
        if B is not None:
            bias = gl.amd.cdna3.buffer_load(
                B + expt_id * stride_b_e,
                offs_wscale_n,
                mask=offs_wscale_n < N,
                other=0.0,
            )
            bias_converted = gl.convert_layout(bias, gl.SliceLayout(0, mfma_layout))
            acc_fp32 = acc_fp32 + bias_converted[None, :]

        # -- Apply gammas (pre-loaded outside N loop) --
        if Gammas is not None:
            acc_fp32 = acc_fp32 * gamma_vals[:, None]

        # -- Write back --
        offs_cn = (
            pid_n * BLOCK_N
            + n_start
            + gl.arange(0, SUB_BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_d))
        )
        offs_ym = block_id * BLOCK_M + gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(1, blocked_d)
        )
        ym_mask = offs_ym < M
        cn_mask = offs_cn < N

        out = gl.convert_layout(acc_fp32, blocked_d).to(Y.dtype.element_ty)
        gl.amd.cdna3.buffer_store(
            out,
            Y + start_m * stride_y_m,
            offs_ym[:, None] * stride_y_m + offs_cn[None, :] * stride_y_n,
            mask=ym_mask[:, None] & cn_mask[None, :],
        )
