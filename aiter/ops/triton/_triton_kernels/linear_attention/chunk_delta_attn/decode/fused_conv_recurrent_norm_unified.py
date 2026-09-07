# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unified fused KDA decode kernel: conv1d + recurrence + gated RMSNorm.

One kernel, four modes controlled by two constexpr flags:

  USE_REPLAY=False, IS_SPEC=False  →  normal decode (load state from ssm_state)
  USE_REPLAY=False, IS_SPEC=True   →  DSpark spec decode (2D indices, snapshot)
  USE_REPLAY=True,  IS_SPEC=False  →  ReplaySSM (checkpoint rebuild + records)
  USE_REPLAY=True,  IS_SPEC=True   →  DSpark + ReplaySSM

USE_REPLAY=False: state from ssm_state[slot], write back to same/snapshot slots.
USE_REPLAY=True:  state from ckpt[slot] + ring buffer replay, write records,
                  optionally flush checkpoint.

Functional-first — correctness over performance.
"""

import triton
import triton.language as tl


@triton.jit
def _fused_kda_decode_unified_kernel(
    # Conv1d
    x_ptr,
    conv_weight_ptr,
    conv_state_ptr,
    # Recurrence gate inputs
    gate_ptr,
    beta_ptr,
    A_log_ptr,
    dt_bias_ptr,
    # State (ssm_state when !USE_REPLAY, ckpt when USE_REPLAY)
    state_ptr,
    # State indices (1D [B] normal, 2D [B, num_spec+1] spec)
    state_indices_ptr,
    cu_seqlens_ptr,
    # RMSNorm + output
    norm_weight_ptr,
    out_gate_ptr,
    out_ptr,
    # Spec decode
    num_accepted_tokens_ptr,
    conv_state_indices_ptr,
    # ReplaySSM ring buffers
    buf_k_ptr,
    buf_u_ptr,
    buf_g_ptr,
    write_pos_ptr,
    slot_idx_ptr,
    # Scalars
    lower_bound,
    norm_eps,
    qk_scale,
    T_tot: tl.int64,
    # Constexprs — dimensions
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    W: tl.constexpr,
    STATE_LEN: tl.constexpr,
    # Constexprs — replay
    CAP: tl.constexpr,
    T_MAX: tl.constexpr,
    BH: tl.constexpr,
    # Constexprs — mode flags
    USE_REPLAY: tl.constexpr,
    IS_SPEC: tl.constexpr,
    # Strides — conv
    stride_x_tok: tl.int64,
    stride_cw_group: tl.int64,
    stride_cw_width: tl.int64,
    stride_cw_ch: tl.int64,
    stride_cs_slot: tl.int64,
    stride_cs_dim: tl.int64,
    stride_cs_pos: tl.int64,
    # Strides — recurrence
    stride_beta_tok: tl.int64,
    stride_og_tok: tl.int64,
    # Strides — state
    stride_state_slot: tl.int64,
    stride_indices_seq: tl.int64,
    stride_indices_tok: tl.int64,
    # Strides — ring buffers (only used when USE_REPLAY)
    stride_bufk_slot: tl.int64,
    stride_bufk_hv: tl.int64,
    stride_bufk_pos: tl.int64,
    stride_bufu_slot: tl.int64,
    stride_bufu_hv: tl.int64,
    stride_bufu_pos: tl.int64,
    stride_bufg_slot: tl.int64,
    stride_bufg_hv: tl.int64,
    stride_bufg_pos: tl.int64,
):
    i_n = tl.program_id(0)
    i_h = tl.program_id(1)

    bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int64)
    seq_T = eos - bos
    if seq_T == 0:
        return

    lp = H * K
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)

    # ================================================================
    # Resolve slot + load initial state
    # ================================================================
    if USE_REPLAY:
        slot = tl.load(slot_idx_ptr + i_n).to(tl.int64)
        if slot < 0:
            return
        if IS_SPEC:
            conv_slot = tl.load(conv_state_indices_ptr + i_n).to(tl.int64)
        else:
            conv_slot = slot

        p_ckpt = (
            state_ptr
            + slot * stride_state_slot
            + i_h * V * K
            + o_v[:, None] * K
            + o_k[None, :]
        )
        b_h = tl.load(p_ckpt).to(tl.float32)

        h_cursor = tl.load(write_pos_ptr + slot).to(tl.int32)
        h_cursor = tl.maximum(h_cursor, 0)
        do_flush = h_cursor + 2 * T_MAX > CAP
        base = tl.where(do_flush, tl.zeros([], dtype=tl.int64), h_cursor.to(tl.int64))

        if h_cursor > 0:
            o_h = tl.arange(0, BH)
            m_h = o_h < h_cursor
            b_rec_g = tl.load(
                buf_g_ptr
                + slot * stride_bufg_slot
                + i_h * stride_bufg_hv
                + o_h[:, None] * stride_bufg_pos
                + o_k[None, :],
                mask=m_h[:, None],
                other=0.0,
            ).to(tl.float32)
            b_c = tl.cumsum(b_rec_g, axis=0)
            b_ctot = tl.sum(b_rec_g, axis=0)
            b_w = tl.exp(b_ctot[None, :] - b_c)
            b_rec_k = tl.load(
                buf_k_ptr
                + slot * stride_bufk_slot
                + i_h * stride_bufk_hv
                + o_h[:, None] * stride_bufk_pos
                + o_k[None, :],
                mask=m_h[:, None],
                other=0.0,
            ).to(tl.float32)
            b_rec_u = tl.load(
                buf_u_ptr
                + slot * stride_bufu_slot
                + i_h * stride_bufu_hv
                + o_h[:, None] * stride_bufu_pos
                + o_v[None, :],
                mask=m_h[:, None],
                other=0.0,
            ).to(tl.float32)
            b_kw = b_rec_k * b_w
            b_h = b_h * tl.exp(b_ctot)[None, :]
            b_h += tl.dot(tl.trans(b_rec_u), b_kw)

        if do_flush:
            tl.store(p_ckpt, b_h.to(state_ptr.dtype.element_ty))

    else:
        if IS_SPEC:
            i_t_start = tl.load(num_accepted_tokens_ptr + i_n).to(tl.int64)
            i_t_start = tl.maximum(i_t_start - 1, tl.zeros([], dtype=tl.int64))
            state_idx = tl.load(
                state_indices_ptr
                + i_n * stride_indices_seq
                + i_t_start * stride_indices_tok
            ).to(tl.int64)
            conv_slot = tl.load(conv_state_indices_ptr + i_n).to(tl.int64)
        else:
            state_idx = tl.load(state_indices_ptr + i_n).to(tl.int64)
            conv_slot = state_idx

        if state_idx < 0:
            return

        p_h_init = (
            state_ptr
            + state_idx * stride_state_slot
            + i_h * V * K
            + o_v[:, None] * K
            + o_k[None, :]
        )
        b_h = tl.load(p_h_init).to(tl.float32)
        base = tl.zeros([], dtype=tl.int64)

    # ================================================================
    # Conv offsets
    # ================================================================
    q_off = i_h * K
    k_off = lp + i_h * K
    v_off = 2 * lp + i_h * V
    cw_q = 0 * stride_cw_group + i_h * K * stride_cw_ch
    cw_k = 1 * stride_cw_group + i_h * K * stride_cw_ch
    cw_v = 2 * stride_cw_group + i_h * V * stride_cw_ch

    # ================================================================
    # Per-token loop
    # ================================================================
    for i_t in range(seq_T):
        tok = bos + i_t
        p_x = x_ptr + tok * stride_x_tok
        p_cs = conv_state_ptr + conv_slot * stride_cs_slot

        # -------- Conv1d Q --------
        b_x_q = tl.load(p_x + q_off + o_k).to(tl.float32)
        p_csq = p_cs + (q_off + o_k) * stride_cs_dim
        b_q = b_x_q * tl.load(
            conv_weight_ptr + cw_q + o_k * stride_cw_ch + (W - 1) * stride_cw_width
        ).to(tl.float32)
        for j in tl.static_range(W - 1):
            cs_pos = (STATE_LEN - W + 1 + j) if IS_SPEC else j
            b_q += tl.load(p_csq + cs_pos * stride_cs_pos).to(tl.float32) * tl.load(
                conv_weight_ptr + cw_q + o_k * stride_cw_ch + j * stride_cw_width
            ).to(tl.float32)
        b_q = b_q * tl.sigmoid(b_q)
        if IS_SPEC:
            for j in tl.static_range(STATE_LEN - 1):
                tl.store(
                    p_csq + j * stride_cs_pos, tl.load(p_csq + (j + 1) * stride_cs_pos)
                )
            tl.store(
                p_csq + (STATE_LEN - 1) * stride_cs_pos,
                b_x_q.to(p_csq.dtype.element_ty),
            )
        else:
            for j in tl.static_range(W - 2):
                tl.store(
                    p_csq + j * stride_cs_pos, tl.load(p_csq + (j + 1) * stride_cs_pos)
                )
            tl.store(p_csq + (W - 2) * stride_cs_pos, b_x_q.to(p_csq.dtype.element_ty))

        # -------- Conv1d K --------
        b_x_k = tl.load(p_x + k_off + o_k).to(tl.float32)
        p_csk = p_cs + (k_off + o_k) * stride_cs_dim
        b_k = b_x_k * tl.load(
            conv_weight_ptr + cw_k + o_k * stride_cw_ch + (W - 1) * stride_cw_width
        ).to(tl.float32)
        for j in tl.static_range(W - 1):
            cs_pos = (STATE_LEN - W + 1 + j) if IS_SPEC else j
            b_k += tl.load(p_csk + cs_pos * stride_cs_pos).to(tl.float32) * tl.load(
                conv_weight_ptr + cw_k + o_k * stride_cw_ch + j * stride_cw_width
            ).to(tl.float32)
        b_k = b_k * tl.sigmoid(b_k)
        if IS_SPEC:
            for j in tl.static_range(STATE_LEN - 1):
                tl.store(
                    p_csk + j * stride_cs_pos, tl.load(p_csk + (j + 1) * stride_cs_pos)
                )
            tl.store(
                p_csk + (STATE_LEN - 1) * stride_cs_pos,
                b_x_k.to(p_csk.dtype.element_ty),
            )
        else:
            for j in tl.static_range(W - 2):
                tl.store(
                    p_csk + j * stride_cs_pos, tl.load(p_csk + (j + 1) * stride_cs_pos)
                )
            tl.store(p_csk + (W - 2) * stride_cs_pos, b_x_k.to(p_csk.dtype.element_ty))

        # -------- Conv1d V --------
        b_x_v = tl.load(p_x + v_off + o_v).to(tl.float32)
        p_csv = p_cs + (v_off + o_v) * stride_cs_dim
        b_v = b_x_v * tl.load(
            conv_weight_ptr + cw_v + o_v * stride_cw_ch + (W - 1) * stride_cw_width
        ).to(tl.float32)
        for j in tl.static_range(W - 1):
            cs_pos = (STATE_LEN - W + 1 + j) if IS_SPEC else j
            b_v += tl.load(p_csv + cs_pos * stride_cs_pos).to(tl.float32) * tl.load(
                conv_weight_ptr + cw_v + o_v * stride_cw_ch + j * stride_cw_width
            ).to(tl.float32)
        b_v = b_v * tl.sigmoid(b_v)
        if IS_SPEC:
            for j in tl.static_range(STATE_LEN - 1):
                tl.store(
                    p_csv + j * stride_cs_pos, tl.load(p_csv + (j + 1) * stride_cs_pos)
                )
            tl.store(
                p_csv + (STATE_LEN - 1) * stride_cs_pos,
                b_x_v.to(p_csv.dtype.element_ty),
            )
        else:
            for j in tl.static_range(W - 2):
                tl.store(
                    p_csv + j * stride_cs_pos, tl.load(p_csv + (j + 1) * stride_cs_pos)
                )
            tl.store(p_csv + (W - 2) * stride_cs_pos, b_x_v.to(p_csv.dtype.element_ty))

        # -------- QK L2 Norm + Decay + Beta --------
        b_q = b_q * tl.math.rsqrt(tl.sum(b_q * b_q) + 1e-6) * qk_scale
        b_k = b_k * tl.math.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_a = tl.load(gate_ptr + (tok * H + i_h) * K + o_k).to(tl.float32)
        b_dt = tl.load(dt_bias_ptr + i_h * K + o_k).to(tl.float32)
        b_A = tl.load(A_log_ptr + i_h).to(tl.float32)
        b_g = lower_bound * tl.sigmoid(tl.exp(b_A) * (b_a + b_dt))
        b_beta = tl.sigmoid(
            tl.load(beta_ptr + tok * stride_beta_tok + i_h).to(tl.float32)
        )

        # -------- Delta Rule --------
        b_h = b_h * tl.exp(b_g[None, :])
        b_dot = tl.sum(b_h * b_k[None, :], 1)
        b_u = (b_v - b_dot) * b_beta
        b_h = b_h + b_u[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)

        # -------- State write-back --------
        if USE_REPLAY:
            pos = base + i_t
            tl.store(
                buf_k_ptr
                + slot * stride_bufk_slot
                + i_h * stride_bufk_hv
                + pos * stride_bufk_pos
                + o_k,
                b_k.to(buf_k_ptr.dtype.element_ty),
            )
            tl.store(
                buf_u_ptr
                + slot * stride_bufu_slot
                + i_h * stride_bufu_hv
                + pos * stride_bufu_pos
                + o_v,
                b_u.to(buf_u_ptr.dtype.element_ty),
            )
            tl.store(
                buf_g_ptr
                + slot * stride_bufg_slot
                + i_h * stride_bufg_hv
                + pos * stride_bufg_pos
                + o_k,
                b_g.to(buf_g_ptr.dtype.element_ty),
            )
        else:
            if IS_SPEC:
                final_idx = tl.load(
                    state_indices_ptr
                    + i_n * stride_indices_seq
                    + i_t * stride_indices_tok
                ).to(tl.int64)
                if final_idx >= 0:
                    p_h_out = (
                        state_ptr
                        + final_idx * stride_state_slot
                        + i_h * V * K
                        + o_v[:, None] * K
                        + o_k[None, :]
                    )
                    tl.store(p_h_out, b_h.to(state_ptr.dtype.element_ty))
            else:
                p_h_out = (
                    state_ptr
                    + state_idx * stride_state_slot
                    + i_h * V * K
                    + o_v[:, None] * K
                    + o_k[None, :]
                )
                tl.store(p_h_out, b_h.to(state_ptr.dtype.element_ty))

        # -------- Gated RMSNorm --------
        b_o_r = b_o.to(tl.bfloat16).to(tl.float32)
        o_sumsq = tl.sum(b_o_r * b_o_r)
        rstd = tl.math.rsqrt(o_sumsq / V + norm_eps)
        b_w = tl.load(norm_weight_ptr + o_v).to(tl.float32)
        b_og = tl.load(out_gate_ptr + tok * stride_og_tok + i_h * V + o_v).to(
            tl.float32
        )
        b_y = b_o_r * rstd * b_w * tl.sigmoid(b_og)
        tl.store(
            out_ptr + tok * (H * V) + i_h * V + o_v,
            b_y.to(out_ptr.dtype.element_ty),
        )
