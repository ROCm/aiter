# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

# MXFP8 activation quant: per-1x32 e8m0 scale + FP8 e4m3 values.
# Follows aiter.ops.quant.per_1x32_f8_scale_f8_quant:
#   MAX_POW2 = int(log2(448)) = 8
#   dtypeMax = 2 ** 8 = 256.0
#   scale_f32 = max_abs / dtypeMax
#   scale_e8m0 = round_up_to_pow2(scale_f32) → e8m0 biased
#   y = round(x_fp32 / e8m0_to_f32(scale_e8m0)) cast to fp8 e4m3
#
# Per-block e8m0 derivation done with the same trick as the existing mxfp4 quant:
#   - bitcast amax to int32
#   - add 0x200000 (round up to a power of 2 with respect to fp4-style rounding)
#   - mask 0xFF800000 (keep only sign+exponent bits)
#   - bitcast back to fp32
# This delivers a pure power-of-2 amax. Then log2(amax).floor() - 8 gives the
# unbiased e8m0 exponent for MXFP8 (since dtypeMax = 2**8).


@triton.jit
def _mxfp8_quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    BLOCK_SIZE_N: tl.constexpr,  # power-of-2 covering full N
    QUANT_BLOCK_SIZE: tl.constexpr,  # =32
    NUM_PRGMS: tl.constexpr,  # row-loop range (usually =M)
):
    """
    Per-1x32 MXFP8 quant. One program per row, holding the full row in
    registers so a single launch handles all K-groups. Mirrors
    _rmsnorm_mxfp8_quant_kernel shape and minimizes grid overhead.
    """
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = col_offsets < N
    n_groups: tl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE

    for row_idx in tl.range(row_start, M, NUM_PRGMS, num_stages=2):
        x = tl.load(
            x_ptr + row_idx * stride_xm + col_offsets * stride_xn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        # (BLOCK_SIZE_N,) -> (n_groups, QUANT_BLOCK_SIZE)
        x_2d = tl.reshape(x, (n_groups, QUANT_BLOCK_SIZE))
        amax = tl.max(tl.abs(x_2d), axis=1, keep_dims=True)

        amax_i32 = amax.to(tl.int32, bitcast=True)
        amax_i32 = (amax_i32 + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
        amax_p2 = amax_i32.to(tl.float32, bitcast=True)
        scale_unbiased = tl.log2(amax_p2).floor() - 8
        scale_unbiased = tl.clamp(scale_unbiased, min=-127, max=127)
        scale_e8m0 = (scale_unbiased.to(tl.int32) + 127).to(tl.uint8)
        quant_scale = tl.exp2(-scale_unbiased)

        qx_2d = x_2d * quant_scale
        qx = tl.reshape(qx_2d, (BLOCK_SIZE_N,))
        y = qx.to(y_ptr.type.element_ty)

        tl.store(
            y_ptr + row_idx * stride_ym + col_offsets * stride_yn,
            y,
            mask=mask,
        )

        group_offsets = tl.arange(0, n_groups)
        group_mask = group_offsets < (N // QUANT_BLOCK_SIZE)
        scale_flat = tl.reshape(scale_e8m0, (n_groups,))
        tl.store(
            s_ptr + row_idx * stride_sm + group_offsets * stride_sn,
            scale_flat,
            mask=group_mask,
        )


# Transcoder: (FP8 fnuz, fp32 1x128 scale) -> (FP8 fn, e8m0 1x32 scale).
# Replaces the Python dequant+requant cascade (fp32 cast + multiply + bf16 cast
# + per_1x32_mxfp8 quant) used in linear.py's MXFP8 fallback path for MLA wq_b
# when q_norm emits the legacy fp8 fnuz + fp32 1x128 format.
#
# In: x_fp8_fnuz (M, N) — fp8 e4m3fnuz bits (interpreted with bias 8 -> value)
#     x_scale_fp32 (M, N//128) — fp32 per-token-block scale
# Out: y_fp8_fn (M, N) — fp8 e4m3fn bits (NV format, bias 7)
#      y_scale_e8m0 (M, N//32) — uint8 e8m0 (1x32 MX scale)


@triton.jit
def _fp8_legacy_to_mxfp8_kernel(
    x_fnuz_ptr,
    x_scale_fp32_ptr,
    y_fn_ptr,
    y_scale_e8m0_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_xsm,
    stride_xsn,
    stride_ym,
    stride_yn,
    stride_ysm,
    stride_ysn,
    BLOCK_SIZE_M: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,  # =32 (MXFP8 group)
    LEGACY_BLOCK_SIZE: tl.constexpr,  # =128 (input scale group)
):
    """
    One program per (BLOCK_SIZE_M rows, QUANT_BLOCK_SIZE-element column window).
    For each 1x32 block, dequantize fnuz fp8 values using the corresponding
    1x128 fp32 scale, derive the e8m0 (1x32) scale, then re-quantize to fp8 fn.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * QUANT_BLOCK_SIZE + tl.arange(0, QUANT_BLOCK_SIZE)

    x_offs = offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load fp8 fnuz values; .to(fp32) decodes via fnuz bias 8 semantically.
    x_fnuz = tl.load(x_fnuz_ptr + x_offs, mask=x_mask, other=0.0).to(tl.float32)

    # Which legacy 1x128 group does this 1x32 block fall into?
    legacy_n = (pid_n * QUANT_BLOCK_SIZE) // LEGACY_BLOCK_SIZE
    xs_offs = offs_m * stride_xsm + legacy_n * stride_xsn
    xs_mask = offs_m < M
    x_scale = tl.load(x_scale_fp32_ptr + xs_offs, mask=xs_mask, other=1.0)

    # Dequantize: bf16-equivalent reconstruction.
    x_dq = x_fnuz * x_scale[:, None]

    # Derive new e8m0 (1x32) scale from x_dq amax. Same recipe as
    # _mxfp8_quant_kernel above.
    amax = tl.max(tl.abs(x_dq), axis=1, keep_dims=True)
    amax_i32 = amax.to(tl.int32, bitcast=True)
    amax_i32 = (amax_i32 + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax_p2 = amax_i32.to(tl.float32, bitcast=True)
    scale_unbiased = tl.log2(amax_p2).floor() - 8
    scale_unbiased = tl.clamp(scale_unbiased, min=-127, max=127)
    scale_e8m0 = (scale_unbiased.to(tl.int32) + 127).to(tl.uint8)
    quant_scale = tl.exp2(-scale_unbiased)

    # Re-quantize to fp8 fn.
    qx = x_dq * quant_scale
    y = qx.to(y_fn_ptr.type.element_ty)

    y_offs = offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_fn_ptr + y_offs, y, mask=x_mask)

    s_offs = offs_m[:, None] * stride_ysm + pid_n * stride_ysn
    s_mask = offs_m[:, None] < M
    tl.store(y_scale_e8m0_ptr + s_offs, scale_e8m0, mask=s_mask)


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
def _rmsnorm_mxfp8_quant_kernel(
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
        amax = tl.max(tl.abs(y_2d), axis=1, keep_dims=True)  # (G, 1)

        # e8m0 scale derivation (same recipe as _mxfp8_quant_kernel).
        amax_i32 = amax.to(tl.int32, bitcast=True)
        amax_i32 = (amax_i32 + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
        amax_p2 = amax_i32.to(tl.float32, bitcast=True)
        scale_unbiased = tl.log2(amax_p2).floor() - 8
        scale_unbiased = tl.clamp(scale_unbiased, min=-127, max=127)
        scale_e8m0 = (scale_unbiased.to(tl.int32) + 127).to(tl.uint8)  # (G, 1)
        quant_scale = tl.exp2(-scale_unbiased)  # (G, 1)

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
def _dual_rmsnorm_mxfp8_quant_kernel(
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
        amax_q = tl.max(tl.abs(y_q_2d), axis=1, keep_dims=True)

        amax_qi32 = amax_q.to(tl.int32, bitcast=True)
        amax_qi32 = (amax_qi32 + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
        amax_qp2 = amax_qi32.to(tl.float32, bitcast=True)
        scale_q_unbiased = tl.log2(amax_qp2).floor() - 8
        scale_q_unbiased = tl.clamp(scale_q_unbiased, min=-127, max=127)
        scale_q_e8m0 = (scale_q_unbiased.to(tl.int32) + 127).to(tl.uint8)
        quant_scale_q = tl.exp2(-scale_q_unbiased)

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
