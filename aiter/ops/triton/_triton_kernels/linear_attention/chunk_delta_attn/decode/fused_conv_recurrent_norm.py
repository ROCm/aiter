# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused KDA decode kernel: conv1d + delta-rule recurrence + gated RMSNorm.

Single Triton kernel. Grid: (batch, heads).
"""

import triton
import triton.language as tl


@triton.jit
def fused_conv_recurrent_norm_kernel(
    # Conv1d inputs
    x_ptr,  # [B, 3*lp] bf16 (may be strided slice)
    conv_weight_ptr,  # [3*lp, W] or [3, W, lp]
    conv_state_ptr,  # [N, 3*lp, W-1] bf16
    # Recurrence inputs
    gate_ptr,  # [1, B, H, K] bf16
    beta_ptr,  # [1, B, H] bf16 (may be strided)
    A_log_ptr,  # [H] fp32
    dt_bias_ptr,  # [H*K] fp32
    # State
    ssm_state_ptr,  # [N, H, V, K] fp32
    ssm_state_indices_ptr,  # [B] int32
    cu_seqlens_ptr,  # [B+1] int64
    # RMSNorm + output
    norm_weight_ptr,  # [K] fp32
    out_gate_ptr,  # [B, H*K] bf16 (may be strided)
    out_ptr,  # [B, H*K] bf16
    # Scalars
    lower_bound,
    norm_eps,
    qk_scale,
    T: tl.int64,
    # Constexprs
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    W: tl.constexpr,
    # Strides
    stride_x_tok: tl.int64,
    stride_cw_group: tl.int64,
    stride_cw_width: tl.int64,
    stride_cw_ch: tl.int64,
    stride_cs_slot: tl.int64,
    stride_cs_dim: tl.int64,
    stride_cs_pos: tl.int64,
    stride_beta_tok: tl.int64,
    stride_og_tok: tl.int64,
    stride_ssm_slot: tl.int64,
):
    i_n = tl.program_id(0)
    i_h = tl.program_id(1)

    bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int64)
    seq_T = eos - bos
    if seq_T == 0:
        return
    state_idx = tl.load(ssm_state_indices_ptr + i_n).to(tl.int64)
    if state_idx < 0:
        return

    lp = H * K
    q_off = i_h * K
    k_off = lp + i_h * K
    v_off = 2 * lp + i_h * V
    cw_q = 0 * stride_cw_group + i_h * K * stride_cw_ch
    cw_k = 1 * stride_cw_group + i_h * K * stride_cw_ch
    cw_v = 2 * stride_cw_group + i_h * V * stride_cw_ch
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)

    for i_t in range(seq_T):
        tok = bos + i_t
        p_x = x_ptr + tok * stride_x_tok
        p_cs = conv_state_ptr + state_idx * stride_cs_slot

        # ========== Conv1d Q ==========
        b_x_q = tl.load(p_x + q_off + o_k).to(tl.float32)
        p_csq = p_cs + (q_off + o_k) * stride_cs_dim
        b_q = b_x_q * tl.load(
            conv_weight_ptr + cw_q + o_k * stride_cw_ch + (W - 1) * stride_cw_width
        ).to(tl.float32)
        for j in tl.static_range(W - 1):
            b_q += tl.load(p_csq + j * stride_cs_pos).to(tl.float32) * tl.load(
                conv_weight_ptr + cw_q + o_k * stride_cw_ch + j * stride_cw_width
            ).to(tl.float32)
        b_q = b_q * tl.sigmoid(b_q)
        for j in tl.static_range(W - 2):
            tl.store(
                p_csq + j * stride_cs_pos, tl.load(p_csq + (j + 1) * stride_cs_pos)
            )
        tl.store(
            p_csq + (W - 2) * stride_cs_pos,
            b_x_q.to(p_csq.dtype.element_ty),
        )

        # ========== Conv1d K ==========
        b_x_k = tl.load(p_x + k_off + o_k).to(tl.float32)
        p_csk = p_cs + (k_off + o_k) * stride_cs_dim
        b_k = b_x_k * tl.load(
            conv_weight_ptr + cw_k + o_k * stride_cw_ch + (W - 1) * stride_cw_width
        ).to(tl.float32)
        for j in tl.static_range(W - 1):
            b_k += tl.load(p_csk + j * stride_cs_pos).to(tl.float32) * tl.load(
                conv_weight_ptr + cw_k + o_k * stride_cw_ch + j * stride_cw_width
            ).to(tl.float32)
        b_k = b_k * tl.sigmoid(b_k)
        for j in tl.static_range(W - 2):
            tl.store(
                p_csk + j * stride_cs_pos, tl.load(p_csk + (j + 1) * stride_cs_pos)
            )
        tl.store(
            p_csk + (W - 2) * stride_cs_pos,
            b_x_k.to(p_csk.dtype.element_ty),
        )

        # ========== Conv1d V ==========
        b_x_v = tl.load(p_x + v_off + o_v).to(tl.float32)
        p_csv = p_cs + (v_off + o_v) * stride_cs_dim
        b_v = b_x_v * tl.load(
            conv_weight_ptr + cw_v + o_v * stride_cw_ch + (W - 1) * stride_cw_width
        ).to(tl.float32)
        for j in tl.static_range(W - 1):
            b_v += tl.load(p_csv + j * stride_cs_pos).to(tl.float32) * tl.load(
                conv_weight_ptr + cw_v + o_v * stride_cw_ch + j * stride_cw_width
            ).to(tl.float32)
        b_v = b_v * tl.sigmoid(b_v)
        for j in tl.static_range(W - 2):
            tl.store(
                p_csv + j * stride_cs_pos, tl.load(p_csv + (j + 1) * stride_cs_pos)
            )
        tl.store(
            p_csv + (W - 2) * stride_cs_pos,
            b_x_v.to(p_csv.dtype.element_ty),
        )

        # ========== QK L2 Norm + Decay + Beta ==========
        b_q = b_q * tl.math.rsqrt(tl.sum(b_q * b_q) + 1e-6) * qk_scale
        b_k = b_k * tl.math.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_a = tl.load(gate_ptr + (tok * H + i_h) * K + o_k).to(tl.float32)
        b_dt = tl.load(dt_bias_ptr + i_h * K + o_k).to(tl.float32)
        b_A = tl.load(A_log_ptr + i_h).to(tl.float32)
        b_g = lower_bound * tl.sigmoid(tl.exp(b_A) * (b_a + b_dt))
        b_beta = tl.sigmoid(
            tl.load(beta_ptr + tok * stride_beta_tok + i_h).to(tl.float32)
        )

        # ========== Delta Rule ==========
        p_h = (
            ssm_state_ptr
            + state_idx * stride_ssm_slot
            + i_h * V * K
            + o_v[:, None] * K
            + o_k[None, :]
        )
        b_h = tl.load(p_h).to(tl.float32)
        b_h = b_h * tl.exp(b_g[None, :])
        b_dot = tl.sum(b_h * b_k[None, :], 1)
        b_v = b_v - b_dot
        b_v = b_v * b_beta
        b_h = b_h + b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_h, b_h.to(p_h.dtype.element_ty))

        # ========== Gated RMSNorm ==========
        b_o_rounded = b_o.to(tl.bfloat16).to(tl.float32)
        o_sumsq = tl.sum(b_o_rounded * b_o_rounded)
        rstd = tl.math.rsqrt(o_sumsq / V + norm_eps)
        b_w = tl.load(norm_weight_ptr + o_v).to(tl.float32)
        b_og = tl.load(out_gate_ptr + tok * stride_og_tok + i_h * V + o_v).to(
            tl.float32
        )
        b_y = b_o_rounded * rstd * b_w * tl.sigmoid(b_og)
        tl.store(
            out_ptr + tok * (H * V) + i_h * V + o_v,
            b_y.to(out_ptr.dtype.element_ty),
        )
