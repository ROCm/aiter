# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from .quant import _mxfp8_quant_op

# Fused RMSNorm + MXFP8 (1x32 e8m0) quant. Replaces the separate
# rmsnorm_quant(fp8 fnuz + fp32 1x128) + transcode-to-MXFP8 sequence used
# upstream of MXFP8-aware GEMMs (e.g. V4 q_norm -> wq_b).
#
# One program per row. Holds the full row in registers, so K is constrained
# by the BLOCK_SIZE_K constexpr (must be a power of two >= K).
#
# In:  x (M, K) bf16 or fp16
#      g (K,)  bf16 or fp16 weight
# Out: y (M, K) fp8 e4m3fn
#      scale (M, K // 32) uint8 e8m0


@triton.jit
def _fused_rms_mxfp8_kernel(
    x_ptr,
    g_ptr,
    y_ptr,
    s_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_ym,
    stride_yk,
    stride_sm,
    stride_sn,
    epsilon,
    BLOCK_SIZE_K: tl.constexpr,  # power-of-2 covering full K
    QUANT_BLOCK_SIZE: tl.constexpr,  # =32
    NUM_PRGMS: tl.constexpr,  # for persistent-loop variant; usually =M
):
    """One program processes one row: rmsnorm then MXFP8 quant in registers."""
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE_K)
    mask = col_offsets < K

    for row_idx in tl.range(row_start, M, NUM_PRGMS, num_stages=2):
        # Load full row, cast to fp32
        x = tl.load(
            x_ptr + row_idx * stride_xm + col_offsets * stride_xk,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

        # RMS norm
        ss = tl.sum(x * x, axis=-1)
        norm_factor = tl.math.rsqrt((ss / K) + epsilon)
        y_fp32 = x * norm_factor * g  # (BLOCK_SIZE_K,)

        # Reshape into (K // QUANT_BLOCK_SIZE, QUANT_BLOCK_SIZE) groups for amax.
        # BLOCK_SIZE_K is the power-of-2 padded size; we keep OOB lanes masked to 0
        # via the load above, so amax over them is 0 (won't affect the in-bounds max).
        y_2d = tl.reshape(y_fp32, (BLOCK_SIZE_K // QUANT_BLOCK_SIZE, QUANT_BLOCK_SIZE))
        scale_e8m0, quant_scale = _mxfp8_quant_op(y_2d, QUANT_AXIS=1)

        # Quantize: y_quant = y_fp32 * quant_scale (broadcast along inner 32).
        qx_2d = y_2d * quant_scale
        qx = tl.reshape(qx_2d, (BLOCK_SIZE_K,))
        y_fp8 = qx.to(y_ptr.type.element_ty)

        # Store y (mask OOB).
        tl.store(
            y_ptr + row_idx * stride_ym + col_offsets * stride_yk,
            y_fp8,
            mask=mask,
        )

        # Store scales: G entries for this row.
        n_groups: tl.constexpr = BLOCK_SIZE_K // QUANT_BLOCK_SIZE
        group_offsets = tl.arange(0, n_groups)
        group_mask = group_offsets < (K // QUANT_BLOCK_SIZE)
        scale_flat = tl.reshape(scale_e8m0, (n_groups,))
        tl.store(
            s_ptr + row_idx * stride_sm + group_offsets * stride_sn,
            scale_flat,
            mask=group_mask,
        )


# Dual fused RMSNorm: Q-side (MXFP8 quant + e8m0 scale emit) + K-side (bf16 out).
# Replaces the CK `fused_qk_rmsnorm_group_quant` semantics in one Triton launch
# for the MXFP8 GEMM path (Task #77). The two halves are independent (different
# weight, different K dim) so they're packed into one program per row to amortize
# launch overhead: same kernel launch loads both rows, normalizes both, stores Q
# fp8 + scale, stores K bf16. Each row's Q and K are independently RMSNorm'd
# (separate weights, separate eps, separate K dim) -- this kernel does NOT fuse
# their normalization arithmetic, only their launch.
#
# In:  q     (M, KQ) bf16 or fp16
#      kv    (M, KK) bf16 or fp16
#      gq    (KQ,)   bf16 or fp16 Q-RMSNorm weight
#      gk    (KK,)   bf16 or fp16 K-RMSNorm weight
# Out: yq    (M, KQ) fp8 e4m3fn
#      sq    (M, KQ // 32) uint8 e8m0
#      yk    (M, KK) bf16


@triton.jit
def _fused_dual_rmsnorm_mxfp8_quant_kernel(
    q_ptr,
    k_ptr,
    gq_ptr,
    gk_ptr,
    yq_ptr,
    sq_ptr,
    yk_ptr,
    M,
    KQ,
    KK,
    stride_qm,
    stride_qn,
    stride_km,
    stride_kn,
    stride_yqm,
    stride_yqn,
    stride_sqm,
    stride_sqn,
    stride_ykm,
    stride_ykn,
    eps_q,
    eps_k,
    BLOCK_SIZE_KQ: tl.constexpr,  # power-of-2 covering full KQ
    BLOCK_SIZE_KK: tl.constexpr,  # power-of-2 covering full KK
    QUANT_BLOCK_SIZE: tl.constexpr,  # =32 (MXFP8 group size)
    NUM_PRGMS: tl.constexpr,  # row-loop bound (usually =M)
):
    """One program per row: do Q-side RMSNorm+MXFP8 quant AND K-side RMSNorm
    (bf16 out) in one launch. Mirrors the CK `fused_qk_rmsnorm_group_quant`
    fusion topology but emits MXFP8 1x32 (e8m0) scales for Q directly."""
    row_start = tl.program_id(0)

    q_col_offsets = tl.arange(0, BLOCK_SIZE_KQ)
    q_mask = q_col_offsets < KQ
    k_col_offsets = tl.arange(0, BLOCK_SIZE_KK)
    k_mask = k_col_offsets < KK

    n_q_groups: tl.constexpr = BLOCK_SIZE_KQ // QUANT_BLOCK_SIZE

    for row_idx in tl.range(row_start, M, NUM_PRGMS, num_stages=2):
        # ===== Q side: RMSNorm + MXFP8 quant =====
        x_q = tl.load(
            q_ptr + row_idx * stride_qm + q_col_offsets * stride_qn,
            mask=q_mask,
            other=0.0,
        ).to(tl.float32)
        g_q = tl.load(gq_ptr + q_col_offsets, mask=q_mask, other=0.0).to(tl.float32)

        ss_q = tl.sum(x_q * x_q, axis=-1)
        norm_q = tl.math.rsqrt((ss_q / KQ) + eps_q)
        y_q_fp32 = x_q * norm_q * g_q

        y_q_2d = tl.reshape(y_q_fp32, (n_q_groups, QUANT_BLOCK_SIZE))
        scale_q_e8m0, quant_scale_q = _mxfp8_quant_op(y_q_2d, QUANT_AXIS=1)

        qx_q_2d = y_q_2d * quant_scale_q
        qx_q = tl.reshape(qx_q_2d, (BLOCK_SIZE_KQ,))
        y_q_fp8 = qx_q.to(yq_ptr.type.element_ty)

        tl.store(
            yq_ptr + row_idx * stride_yqm + q_col_offsets * stride_yqn,
            y_q_fp8,
            mask=q_mask,
        )

        q_group_offsets = tl.arange(0, n_q_groups)
        q_group_mask = q_group_offsets < (KQ // QUANT_BLOCK_SIZE)
        scale_q_flat = tl.reshape(scale_q_e8m0, (n_q_groups,))
        tl.store(
            sq_ptr + row_idx * stride_sqm + q_group_offsets * stride_sqn,
            scale_q_flat,
            mask=q_group_mask,
        )

        # ===== K side: RMSNorm only, bf16 out =====
        x_k = tl.load(
            k_ptr + row_idx * stride_km + k_col_offsets * stride_kn,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        g_k = tl.load(gk_ptr + k_col_offsets, mask=k_mask, other=0.0).to(tl.float32)

        ss_k = tl.sum(x_k * x_k, axis=-1)
        norm_k = tl.math.rsqrt((ss_k / KK) + eps_k)
        y_k_fp32 = x_k * norm_k * g_k
        y_k_out = y_k_fp32.to(yk_ptr.type.element_ty)

        tl.store(
            yk_ptr + row_idx * stride_ykm + k_col_offsets * stride_ykn,
            y_k_out,
            mask=k_mask,
        )


# Flatten-then-MXFP8 quant. Takes (M, N1, N2) input, flattens the trailing two
# dims into N = N1 * N2, and emits per-1x32 MXFP8 (FP8 e4m3fn values + uint8
# e8m0 scales) along the flattened axis. One program per (m, n1); each program
# handles a row of N2 elements that contributes BLOCK_SIZE_N2 // 32 groups to
# the M-th row of the (M, N) flattened output.


@triton.jit
def _fused_flatten_mxfp8_quant_kernel(
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
):
    m = tl.program_id(0)
    n1 = tl.program_id(1)

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N2 // QUANT_BLOCK_SIZE
    # In the flattened (M, N1 * N2) output, each n1 segment is exactly N2 wide
    # (not BLOCK_SIZE_N2), so stride between n1 segments must use N2 — otherwise
    # non-power-of-2 N2 (e.g. 7168) would gap-write the output.
    n2_groups = N2 // QUANT_BLOCK_SIZE

    n2_offs = tl.arange(0, BLOCK_SIZE_N2)
    x_mask = n2_offs < N2
    x_offs = m * x_stride_m + n1 * x_stride_n1 + n2_offs * x_stride_n2
    x = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0).to(tl.float32)

    x_2d = tl.reshape(x, (NUM_QUANT_BLOCKS, QUANT_BLOCK_SIZE))
    scale_e8m0, quant_scale = _mxfp8_quant_op(x_2d, QUANT_AXIS=1)

    qx_2d = x_2d * quant_scale
    qx = tl.reshape(qx_2d, (BLOCK_SIZE_N2,))
    tl.store(
        out_ptr + m * out_stride_m + (n1 * N2 + n2_offs) * out_stride_n,
        qx.to(out_ptr.type.element_ty),
        mask=x_mask,
    )

    block_scale_offs = tl.arange(0, NUM_QUANT_BLOCKS)
    scale_flat = tl.reshape(scale_e8m0, (NUM_QUANT_BLOCKS,))
    tl.store(
        out_scales_ptr
        + m * out_scales_stride_m
        + (n1 * n2_groups + block_scale_offs) * out_scales_stride_n,
        scale_flat,
        mask=block_scale_offs < n2_groups,
    )
