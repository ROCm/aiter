# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Triton kernels for sage_quant_mxfp4 with fp8 q/k input.
#
# The only difference from the bf16 kernels in sage_attention_quant.py is that
# immediately after tl.load the tile is cast to tl.float32 before any
# arithmetic (smoothing, Hadamard rotation). This avoids materialising an
# intermediate bf16 tensor in HBM; the fp8 values are widened on-chip.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.quant.sage_attention_quant import (
    _compute_mx_quant_and_scale_rne,
)


@triton.jit
def _rotate_quantize_q_fp8_kernel(
    Q,               # fp8 input  [B, S, H, D] or [B, H, S, D]
    Q_q,             # uint8 output (packed fp4x2)
    Q_descale,       # uint8 output (e8m0 scales)
    Q_mean,          # fp32 output (only used when q_smoothing=True)
    R,               # Hadamard matrix [BLOCK_R, BLOCK_R] in bf16
    sm_scale: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_qqb,
    stride_qqm,
    stride_qqh,
    stride_qqd,
    stride_qsb,
    stride_qsm,
    stride_qsh,
    stride_qsd,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_md,
    batch,
    heads_q,
    seqlen_q,
    d_model,
    q_smoothing: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
    D: tl.constexpr,
):
    SCALE_GROUP_SIZE: tl.constexpr = 32
    pid = tl.program_id(0).to(tl.int64)
    pid_b = pid % batch
    pid_h = pid // batch % heads_q
    pid_m = pid // (batch * heads_q)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_dq = tl.arange(0, D // 2)
    offs_ds = tl.arange(0, D // SCALE_GROUP_SIZE)

    qk_ptr = Q + (
        pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    qk_descale_ptr = Q_descale + (
        pid_b * stride_qsb
        + pid_h * stride_qsh
        + offs_m[:, None] * stride_qsm
        + offs_ds[None, :] * stride_qsd
    )
    qk_quant_ptr = Q_q + (
        pid_b * stride_qqb
        + pid_h * stride_qqh
        + offs_m[:, None] * stride_qqm
        + offs_dq[None, :] * stride_qqd
    )

    # Load fp8, widen to fp32 on-chip — no intermediate bf16 in HBM
    qk_tile = tl.load(
        qk_ptr,
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < d_model),
        other=0.0,
    ).to(tl.float32)

    if q_smoothing:
        ACTUAL_BLOCK_M = tl.minimum(BLOCK_M, seqlen_q - pid_m * BLOCK_M)
        m_row_mean = tl.sum(qk_tile, axis=0) / ACTUAL_BLOCK_M
        qk_tile -= m_row_mean[None, :]
        mean_ptr = (
            Q_mean
            + pid_b * stride_mb
            + pid_h * stride_mh
            + pid_m * stride_mm
            + offs_d * stride_md
        )
        tl.store(mean_ptr, m_row_mean * sm_scale)

    r_ptr = (
        R
        + tl.arange(0, BLOCK_R)[:, None] * BLOCK_R
        + tl.arange(0, BLOCK_R)[None, :]
    )
    r_mat = tl.load(r_ptr)  # bf16 [BLOCK_R, BLOCK_R]

    shape0: tl.constexpr = BLOCK_M * D // BLOCK_R
    qk_rot_tile = tl.dot(qk_tile.reshape((shape0, BLOCK_R)).to(r_mat.dtype), r_mat)
    qk_rot_tile = qk_rot_tile.reshape((BLOCK_M, D))
    qk_rot_tile *= sm_scale

    qk_quant_tile, qk_descale = _compute_mx_quant_and_scale_rne(
        qk_rot_tile, offs_m[:, None] < seqlen_q, tl.uint8
    )

    tl.store(qk_descale_ptr, qk_descale, mask=(offs_m[:, None] < seqlen_q))
    tl.store(qk_quant_ptr, qk_quant_tile, mask=(offs_m[:, None] < seqlen_q))


@triton.jit
def _rotate_quantize_qk_fp8_kernel(
    Q,               # fp8 input
    Q_q,             # uint8 output (packed fp4x2)
    Q_descale,       # uint8 output (e8m0 scales)
    K,               # fp8 input
    K_q,             # uint8 output (packed fp4x2)
    K_descale,       # uint8 output (e8m0 scales)
    K_mean,          # fp32 [B, H, D] — per-(B,H,D) mean to subtract on-chip
    R,               # Hadamard matrix [BLOCK_R, BLOCK_R] in bf16
    Q_input_scale,   # fp32 scalar: x_bf16 = x_fp8 * scale (per-tensor comms)
    K_input_scale,   # fp32 scalar
    apply_input_scale: tl.constexpr,
    sm_scale: tl.constexpr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_qqb, stride_qqm, stride_qqh, stride_qqd,
    stride_qsb, stride_qsm, stride_qsh, stride_qsd,
    stride_kb, stride_kh, stride_km, stride_kd,
    stride_kqb, stride_kqn, stride_kqh, stride_kqd,
    stride_ksb, stride_ksn, stride_ksh, stride_ksd,
    stride_meanb, stride_meanh, stride_meand,
    batch,
    heads_q,
    heads_k,
    seqlen_q,
    seqlen_k,
    d_model,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
    D: tl.constexpr,
):
    SCALE_GROUP_SIZE: tl.constexpr = 32

    q_pids = batch * heads_q * tl.cdiv(seqlen_q, BLOCK_M)
    pid = tl.program_id(0).to(tl.int64)
    is_q_pid = pid < q_pids

    if is_q_pid:
        pid_b = pid % batch
        pid_h = pid // batch % heads_q
        pid_m = pid // (batch * heads_q)
    else:
        _pid = pid - q_pids
        pid_b = _pid % batch
        pid_h = _pid // batch % heads_k
        pid_m = _pid // (batch * heads_k)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_dq = tl.arange(0, D // 2)
    offs_ds = tl.arange(0, D // SCALE_GROUP_SIZE)

    if is_q_pid:
        qk_ptr = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        qk_descale_ptr = Q_descale + pid_b * stride_qsb + pid_h * stride_qsh + offs_m[:, None] * stride_qsm + offs_ds[None, :] * stride_qsd
        qk_quant_ptr = Q_q + pid_b * stride_qqb + pid_h * stride_qqh + offs_m[:, None] * stride_qqm + offs_dq[None, :] * stride_qqd
        seqlen = seqlen_q
    else:
        qk_ptr = K + pid_b * stride_kb + pid_h * stride_kh + offs_m[:, None] * stride_km + offs_d[None, :] * stride_kd
        qk_descale_ptr = K_descale + pid_b * stride_ksb + pid_h * stride_ksh + offs_m[:, None] * stride_ksn + offs_ds[None, :] * stride_ksd
        qk_quant_ptr = K_q + pid_b * stride_kqb + pid_h * stride_kqh + offs_m[:, None] * stride_kqn + offs_dq[None, :] * stride_kqd
        seqlen = seqlen_k

    # Load fp8, widen to fp32 on-chip
    qk_tile = tl.load(
        qk_ptr,
        mask=(offs_m[:, None] < seqlen) & (offs_d[None, :] < d_model),
        other=0.0,
    ).to(tl.float32)
    if apply_input_scale:
        if is_q_pid:
            qk_tile *= tl.load(Q_input_scale)
        else:
            qk_tile *= tl.load(K_input_scale)

    # K-smoothing: subtract per-(B,H,D) mean on-chip for K pids
    if not is_q_pid:
        mean_ptr = K_mean + pid_b * stride_meanb + pid_h * stride_meanh + offs_d[None, :] * stride_meand
        k_mean = tl.load(mean_ptr)
        qk_tile -= k_mean

    r_ptr = R + tl.arange(0, BLOCK_R)[:, None] * BLOCK_R + tl.arange(0, BLOCK_R)[None, :]
    r_mat = tl.load(r_ptr)

    shape0: tl.constexpr = BLOCK_M * D // BLOCK_R
    qk_rot_tile = tl.dot(qk_tile.reshape((shape0, BLOCK_R)).to(r_mat.dtype), r_mat)
    qk_rot_tile = qk_rot_tile.reshape((BLOCK_M, D))

    if is_q_pid:
        qk_rot_tile *= sm_scale

    qk_quant_tile, qk_descale = _compute_mx_quant_and_scale_rne(
        qk_rot_tile, offs_m[:, None] < seqlen, tl.uint8
    )

    tl.store(qk_descale_ptr, qk_descale, mask=(offs_m[:, None] < seqlen))
    tl.store(qk_quant_ptr, qk_quant_tile, mask=(offs_m[:, None] < seqlen))


@triton.jit
def _rotate_quantize_k_fp8_kernel(
    K,               # fp8 input
    K_q,             # uint8 output (packed fp4x2)
    K_descale,       # uint8 output (e8m0 scales)
    K_mean,          # fp32 [B, 1, H, D] or [B, H, 1, D] — per-(B,H,D) mean to subtract
    R,               # Hadamard matrix [BLOCK_R, BLOCK_R] in bf16
    stride_kb,
    stride_kh,
    stride_km,
    stride_kd,
    stride_kqb,
    stride_kqn,
    stride_kqh,
    stride_kqd,
    stride_ksb,
    stride_ksn,
    stride_ksh,
    stride_ksd,
    stride_meanb,
    stride_meanh,
    stride_meand,
    batch,
    heads_k,
    seqlen_k,
    d_model,
    smooth_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
    D: tl.constexpr,
):
    SCALE_GROUP_SIZE: tl.constexpr = 32
    pid = tl.program_id(0).to(tl.int64)
    pid_b = pid % batch
    pid_h = pid // batch % heads_k
    pid_m = pid // (batch * heads_k)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_dq = tl.arange(0, D // 2)
    offs_ds = tl.arange(0, D // SCALE_GROUP_SIZE)

    qk_ptr = K + (
        pid_b * stride_kb
        + pid_h * stride_kh
        + offs_m[:, None] * stride_km
        + offs_d[None, :] * stride_kd
    )
    qk_descale_ptr = K_descale + (
        pid_b * stride_ksb
        + pid_h * stride_ksh
        + offs_m[:, None] * stride_ksn
        + offs_ds[None, :] * stride_ksd
    )
    qk_quant_ptr = K_q + (
        pid_b * stride_kqb
        + pid_h * stride_kqh
        + offs_m[:, None] * stride_kqn
        + offs_dq[None, :] * stride_kqd
    )

    # Load fp8, widen to fp32 on-chip — no intermediate bf16 tensor in HBM
    qk_tile = tl.load(
        qk_ptr,
        mask=(offs_m[:, None] < seqlen_k) & (offs_d[None, :] < d_model),
        other=0.0,
    ).to(tl.float32)

    # Subtract per-(B,H,D) mean on-chip — K_mean is fp32, shape [B, H, D]
    if smooth_k:
        mean_ptr = (
            K_mean
            + pid_b * stride_meanb
            + pid_h * stride_meanh
            + offs_d[None, :] * stride_meand
        )
        k_mean = tl.load(mean_ptr)  # [1, D] broadcast over BLOCK_M
        qk_tile -= k_mean

    r_ptr = (
        R
        + tl.arange(0, BLOCK_R)[:, None] * BLOCK_R
        + tl.arange(0, BLOCK_R)[None, :]
    )
    r_mat = tl.load(r_ptr)

    shape0: tl.constexpr = BLOCK_M * D // BLOCK_R
    qk_rot_tile = tl.dot(qk_tile.reshape((shape0, BLOCK_R)).to(r_mat.dtype), r_mat)
    qk_rot_tile = qk_rot_tile.reshape((BLOCK_M, D))

    qk_quant_tile, qk_descale = _compute_mx_quant_and_scale_rne(
        qk_rot_tile, offs_m[:, None] < seqlen_k, tl.uint8
    )

    tl.store(qk_descale_ptr, qk_descale, mask=(offs_m[:, None] < seqlen_k))
    tl.store(qk_quant_ptr, qk_quant_tile, mask=(offs_m[:, None] < seqlen_k))
