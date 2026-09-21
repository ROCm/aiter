# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused KDA decode kernel: conv1d + delta-rule recurrence + gated RMSNorm.

Single Triton kernel. Grid: (batch, heads).
"""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_fused_conv_recurrent_norm_repr = make_kernel_repr(
    "fused_conv_recurrent_norm_kernel",
    ["H", "K", "V", "W", "STATE_LEN", "IS_SPEC_DECODING"],
)
_fused_kda_spec_parallel_v_repr = make_kernel_repr(
    "fused_kda_spec_parallel_v_kernel",
    ["H", "K", "V", "W", "BV", "SPEC_LEN"],
)
_fused_kda_spec_finalize_repr = make_kernel_repr(
    "fused_kda_spec_finalize_kernel",
    ["H", "K", "V", "W", "SPEC_LEN"],
)


@triton.jit(repr=_fused_conv_recurrent_norm_repr)
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
    ssm_state_indices_ptr,  # [B] or [B, num_spec+1] int32
    num_accepted_tokens_ptr,  # [B] int32 (spec decode only)
    conv_state_indices_ptr,  # [B] int32 (spec decode only)
    cu_seqlens_ptr,  # [B+1] int64
    # RMSNorm + output
    norm_weight_ptr,  # [K] fp32
    out_gate_ptr,  # [B, H*K] bf16 (may be strided)
    out_ptr,  # [B, H*K] bf16
    # Scalars
    lower_bound,
    norm_eps,
    qk_scale,
    # Constexprs
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    W: tl.constexpr,
    STATE_LEN: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    # Strides
    stride_x_tok,
    stride_cw_group,
    stride_cw_width,
    stride_cw_ch,
    stride_cs_slot,
    stride_cs_dim,
    stride_cs_pos,
    stride_beta_tok,
    stride_og_tok,
    stride_ssm_slot,
    stride_indices_seq,
    stride_indices_tok,
):
    i_n = tl.program_id(0)
    i_h = tl.program_id(1)
    tl.assume(i_n >= 0)
    tl.assume(i_h >= 0)
    tl.assume(stride_x_tok > 0)
    tl.assume(stride_cw_group > 0)
    tl.assume(stride_cw_width > 0)
    tl.assume(stride_cw_ch > 0)
    tl.assume(stride_cs_slot > 0)
    tl.assume(stride_cs_dim > 0)
    tl.assume(stride_cs_pos > 0)
    tl.assume(stride_beta_tok > 0)
    tl.assume(stride_og_tok > 0)
    tl.assume(stride_ssm_slot > 0)
    tl.assume(stride_indices_seq > 0)
    tl.assume(stride_indices_tok > 0)

    bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int64)
    tl.assume(bos >= 0)
    tl.assume(eos >= bos)
    seq_T = eos - bos
    if seq_T == 0:
        return
    if IS_SPEC_DECODING:
        checkpoint = tl.load(num_accepted_tokens_ptr + i_n).to(tl.int64) - 1
        checkpoint = tl.maximum(checkpoint, 0)
        tl.assume(checkpoint >= 0)
        state_idx = tl.load(
            ssm_state_indices_ptr
            + i_n * stride_indices_seq
            + checkpoint * stride_indices_tok
        ).to(tl.int64)
        conv_state_idx = tl.load(conv_state_indices_ptr + i_n).to(tl.int64)
    else:
        state_idx = tl.load(ssm_state_indices_ptr + i_n).to(tl.int64)
        conv_state_idx = state_idx
    if state_idx < 0:
        return
    tl.assume(state_idx >= 0)
    tl.assume(conv_state_idx >= 0)

    lp = H * K
    q_off = i_h * K
    k_off = lp + i_h * K
    v_off = 2 * lp + i_h * V
    cw_q = 0 * stride_cw_group + i_h * K * stride_cw_ch
    cw_k = 1 * stride_cw_group + i_h * K * stride_cw_ch
    cw_v = 2 * stride_cw_group + i_h * V * stride_cw_ch
    o_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, K), K), K)
    o_v = tl.max_contiguous(tl.multiple_of(tl.arange(0, V), V), V)

    if IS_SPEC_DECODING:
        p_h = (
            ssm_state_ptr
            + state_idx * stride_ssm_slot
            + i_h * V * K
            + o_v[:, None] * K
            + o_k[None, :]
        )
        b_h = tl.load(p_h).to(tl.float32)

    p_cs = conv_state_ptr + conv_state_idx * stride_cs_slot
    p_csq = p_cs + (q_off + o_k) * stride_cs_dim
    p_csk = p_cs + (k_off + o_k) * stride_cs_dim
    p_csv = p_cs + (v_off + o_v) * stride_cs_dim
    if IS_SPEC_DECODING:
        # The expanded speculative conv state is
        #   [history(W-1), previous draft candidates(num_spec)].
        # Roll back to the accepted checkpoint, keep the selected W-1 history
        # values in registers across all candidates, and materialize the next
        # expanded state exactly once. The old implementation read only the
        # final W-1 slots and shifted all STATE_LEN slots for every token.
        q_h0 = tl.load(p_csq + checkpoint * stride_cs_pos).to(tl.float32)
        q_h1 = tl.load(p_csq + (checkpoint + 1) * stride_cs_pos).to(tl.float32)
        q_h2 = tl.load(p_csq + (checkpoint + 2) * stride_cs_pos).to(tl.float32)
        k_h0 = tl.load(p_csk + checkpoint * stride_cs_pos).to(tl.float32)
        k_h1 = tl.load(p_csk + (checkpoint + 1) * stride_cs_pos).to(tl.float32)
        k_h2 = tl.load(p_csk + (checkpoint + 2) * stride_cs_pos).to(tl.float32)
        v_h0 = tl.load(p_csv + checkpoint * stride_cs_pos).to(tl.float32)
        v_h1 = tl.load(p_csv + (checkpoint + 1) * stride_cs_pos).to(tl.float32)
        v_h2 = tl.load(p_csv + (checkpoint + 2) * stride_cs_pos).to(tl.float32)
        tl.debug_barrier()
        tl.store(p_csq, q_h1.to(p_csq.dtype.element_ty))
        tl.store(p_csq + stride_cs_pos, q_h2.to(p_csq.dtype.element_ty))
        tl.store(p_csk, k_h1.to(p_csk.dtype.element_ty))
        tl.store(p_csk + stride_cs_pos, k_h2.to(p_csk.dtype.element_ty))
        tl.store(p_csv, v_h1.to(p_csv.dtype.element_ty))
        tl.store(p_csv + stride_cs_pos, v_h2.to(p_csv.dtype.element_ty))

    b_A = tl.load(A_log_ptr + i_h).to(tl.float32)

    for i_t in range(seq_T):
        tok = bos + i_t
        tl.assume(tok >= 0)
        p_x = x_ptr + tok * stride_x_tok

        # ========== Conv1d Q ==========
        b_x_q = tl.load(p_x + q_off + o_k).to(tl.float32)
        b_q = b_x_q * tl.load(
            conv_weight_ptr + cw_q + o_k * stride_cw_ch + (W - 1) * stride_cw_width
        ).to(tl.float32)
        if IS_SPEC_DECODING:
            b_q += q_h0 * tl.load(conv_weight_ptr + cw_q + o_k * stride_cw_ch).to(
                tl.float32
            )
            b_q += q_h1 * tl.load(
                conv_weight_ptr + cw_q + o_k * stride_cw_ch + stride_cw_width
            ).to(tl.float32)
            b_q += q_h2 * tl.load(
                conv_weight_ptr + cw_q + o_k * stride_cw_ch + 2 * stride_cw_width
            ).to(tl.float32)
        else:
            for j in tl.static_range(W - 1):
                cs_pos = STATE_LEN - W + 1 + j
                b_q += tl.load(p_csq + cs_pos * stride_cs_pos).to(tl.float32) * tl.load(
                    conv_weight_ptr + cw_q + o_k * stride_cw_ch + j * stride_cw_width
                ).to(tl.float32)
        b_q = b_q * tl.sigmoid(b_q)

        # ========== Conv1d K ==========
        b_x_k = tl.load(p_x + k_off + o_k).to(tl.float32)
        b_k = b_x_k * tl.load(
            conv_weight_ptr + cw_k + o_k * stride_cw_ch + (W - 1) * stride_cw_width
        ).to(tl.float32)
        if IS_SPEC_DECODING:
            b_k += k_h0 * tl.load(conv_weight_ptr + cw_k + o_k * stride_cw_ch).to(
                tl.float32
            )
            b_k += k_h1 * tl.load(
                conv_weight_ptr + cw_k + o_k * stride_cw_ch + stride_cw_width
            ).to(tl.float32)
            b_k += k_h2 * tl.load(
                conv_weight_ptr + cw_k + o_k * stride_cw_ch + 2 * stride_cw_width
            ).to(tl.float32)
        else:
            for j in tl.static_range(W - 1):
                cs_pos = STATE_LEN - W + 1 + j
                b_k += tl.load(p_csk + cs_pos * stride_cs_pos).to(tl.float32) * tl.load(
                    conv_weight_ptr + cw_k + o_k * stride_cw_ch + j * stride_cw_width
                ).to(tl.float32)
        b_k = b_k * tl.sigmoid(b_k)

        # ========== Conv1d V ==========
        b_x_v = tl.load(p_x + v_off + o_v).to(tl.float32)
        b_v = b_x_v * tl.load(
            conv_weight_ptr + cw_v + o_v * stride_cw_ch + (W - 1) * stride_cw_width
        ).to(tl.float32)
        if IS_SPEC_DECODING:
            b_v += v_h0 * tl.load(conv_weight_ptr + cw_v + o_v * stride_cw_ch).to(
                tl.float32
            )
            b_v += v_h1 * tl.load(
                conv_weight_ptr + cw_v + o_v * stride_cw_ch + stride_cw_width
            ).to(tl.float32)
            b_v += v_h2 * tl.load(
                conv_weight_ptr + cw_v + o_v * stride_cw_ch + 2 * stride_cw_width
            ).to(tl.float32)
        else:
            for j in tl.static_range(W - 1):
                cs_pos = STATE_LEN - W + 1 + j
                b_v += tl.load(p_csv + cs_pos * stride_cs_pos).to(tl.float32) * tl.load(
                    conv_weight_ptr + cw_v + o_v * stride_cw_ch + j * stride_cw_width
                ).to(tl.float32)
        b_v = b_v * tl.sigmoid(b_v)

        if IS_SPEC_DECODING:
            q_h0, q_h1, q_h2 = q_h1, q_h2, b_x_q
            k_h0, k_h1, k_h2 = k_h1, k_h2, b_x_k
            v_h0, v_h1, v_h2 = v_h1, v_h2, b_x_v
            tl.store(
                p_csq + (W - 2 + i_t) * stride_cs_pos,
                b_x_q.to(p_csq.dtype.element_ty),
            )
            tl.store(
                p_csk + (W - 2 + i_t) * stride_cs_pos,
                b_x_k.to(p_csk.dtype.element_ty),
            )
            tl.store(
                p_csv + (W - 2 + i_t) * stride_cs_pos,
                b_x_v.to(p_csv.dtype.element_ty),
            )
        else:
            # Layouts with more threads than channels replicate conv vectors.
            # Finish all history reads before any owner updates the in-place state.
            tl.debug_barrier()
            for j in tl.static_range(STATE_LEN - 1):
                tl.store(
                    p_csq + j * stride_cs_pos,
                    tl.load(p_csq + (j + 1) * stride_cs_pos),
                )
                tl.store(
                    p_csk + j * stride_cs_pos,
                    tl.load(p_csk + (j + 1) * stride_cs_pos),
                )
                tl.store(
                    p_csv + j * stride_cs_pos,
                    tl.load(p_csv + (j + 1) * stride_cs_pos),
                )
            tl.store(
                p_csq + (STATE_LEN - 1) * stride_cs_pos,
                b_x_q.to(p_csq.dtype.element_ty),
            )
            tl.store(
                p_csk + (STATE_LEN - 1) * stride_cs_pos,
                b_x_k.to(p_csk.dtype.element_ty),
            )
            tl.store(
                p_csv + (STATE_LEN - 1) * stride_cs_pos,
                b_x_v.to(p_csv.dtype.element_ty),
            )
            tl.debug_barrier()

        # ========== QK L2 Norm + Decay + Beta ==========
        b_q = b_q * tl.math.rsqrt(tl.sum(b_q * b_q) + 1e-6) * qk_scale
        b_k = b_k * tl.math.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_a = tl.load(gate_ptr + (tok * H + i_h) * K + o_k).to(tl.float32)
        b_dt = tl.load(dt_bias_ptr + i_h * K + o_k).to(tl.float32)
        b_g = lower_bound * tl.sigmoid(tl.exp(b_A) * (b_a + b_dt))
        b_beta = tl.sigmoid(
            tl.load(beta_ptr + tok * stride_beta_tok + i_h).to(tl.float32)
        )

        # ========== Delta Rule ==========
        if not IS_SPEC_DECODING:
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
        if IS_SPEC_DECODING:
            output_state_idx = tl.load(
                ssm_state_indices_ptr
                + i_n * stride_indices_seq
                + i_t * stride_indices_tok
            ).to(tl.int64)
            if output_state_idx >= 0:
                tl.assume(output_state_idx >= 0)
                p_h_out = (
                    ssm_state_ptr
                    + output_state_idx * stride_ssm_slot
                    + i_h * V * K
                    + o_v[:, None] * K
                    + o_k[None, :]
                )
                tl.store(p_h_out, b_h.to(p_h_out.dtype.element_ty))
        else:
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


@triton.jit(repr=_fused_kda_spec_parallel_v_repr)
def fused_kda_spec_parallel_v_kernel(
    x_ptr,
    conv_weight_ptr,
    conv_state_ptr,
    conv_carry_ptr,
    gate_ptr,
    beta_ptr,
    A_log_ptr,
    dt_bias_ptr,
    ssm_state_ptr,
    ssm_state_indices_ptr,
    num_accepted_tokens_ptr,
    conv_state_indices_ptr,
    cu_seqlens_ptr,
    out_ptr,
    lower_bound,
    qk_scale,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    W: tl.constexpr,
    BV: tl.constexpr,
    SPEC_LEN: tl.constexpr,
    stride_x_tok,
    stride_cw_group,
    stride_cw_width,
    stride_cw_ch,
    stride_cs_slot,
    stride_cs_dim,
    stride_cs_pos,
    stride_beta_tok,
    stride_ssm_slot,
    stride_indices_seq,
    stride_indices_tok,
):
    """Run the SPEC_LEN-token speculative recurrence in parallel V tiles.

    Every sequence is expected to hold exactly SPEC_LEN tokens; ssm_state_indices
    is only SPEC_LEN wide, so the token loop is clamped to that width to keep a
    longer sequence from indexing past the row.
    """
    i_n = tl.program_id(0)
    i_h = tl.program_id(1)
    i_vt = tl.program_id(2)
    tl.assume(i_n >= 0)
    tl.assume(i_h >= 0)
    tl.assume(i_vt >= 0)
    tl.assume(stride_x_tok > 0)
    tl.assume(stride_cw_group > 0)
    tl.assume(stride_cw_width > 0)
    tl.assume(stride_cw_ch > 0)
    tl.assume(stride_cs_slot > 0)
    tl.assume(stride_cs_dim > 0)
    tl.assume(stride_cs_pos > 0)
    tl.assume(stride_beta_tok > 0)
    tl.assume(stride_ssm_slot > 0)
    tl.assume(stride_indices_seq > 0)
    tl.assume(stride_indices_tok > 0)
    bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int64)
    seq_t = eos - bos
    if seq_t == 0:
        return

    checkpoint = tl.maximum(tl.load(num_accepted_tokens_ptr + i_n).to(tl.int64) - 1, 0)
    tl.assume(checkpoint >= 0)
    read_state_idx = tl.load(
        ssm_state_indices_ptr
        + i_n * stride_indices_seq
        + checkpoint * stride_indices_tok
    ).to(tl.int64)
    conv_state_idx = tl.load(conv_state_indices_ptr + i_n).to(tl.int64)
    tl.assume(conv_state_idx >= 0)
    if read_state_idx < 0:
        return
    tl.assume(read_state_idx >= 0)

    lp: tl.constexpr = H * K
    q_off = i_h * K
    k_off = lp + i_h * K
    v_base = i_vt * BV
    v_off = 2 * lp + i_h * V + v_base
    cw_q = i_h * K * stride_cw_ch
    cw_k = stride_cw_group + i_h * K * stride_cw_ch
    cw_v = 2 * stride_cw_group + (i_h * V + v_base) * stride_cw_ch
    o_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, K), K), K)
    o_v = tl.max_contiguous(tl.multiple_of(tl.arange(0, BV), BV), BV)

    p_cs = conv_state_ptr + conv_state_idx * stride_cs_slot
    p_csq = p_cs + (q_off + o_k) * stride_cs_dim
    p_csk = p_cs + (k_off + o_k) * stride_cs_dim
    p_csv = p_cs + (v_off + o_v) * stride_cs_dim
    q_h0 = tl.load(p_csq + checkpoint * stride_cs_pos).to(tl.float32)
    q_h1 = tl.load(p_csq + (checkpoint + 1) * stride_cs_pos).to(tl.float32)
    q_h2 = tl.load(p_csq + (checkpoint + 2) * stride_cs_pos).to(tl.float32)
    k_h0 = tl.load(p_csk + checkpoint * stride_cs_pos).to(tl.float32)
    k_h1 = tl.load(p_csk + (checkpoint + 1) * stride_cs_pos).to(tl.float32)
    k_h2 = tl.load(p_csk + (checkpoint + 2) * stride_cs_pos).to(tl.float32)
    v_h0 = tl.load(p_csv + checkpoint * stride_cs_pos).to(tl.float32)
    v_h1 = tl.load(p_csv + (checkpoint + 1) * stride_cs_pos).to(tl.float32)
    v_h2 = tl.load(p_csv + (checkpoint + 2) * stride_cs_pos).to(tl.float32)
    carry_base = i_n * (3 * lp * 2)
    if i_vt == 0:
        tl.store(conv_carry_ptr + carry_base + (q_off + o_k) * 2, q_h1)
        tl.store(conv_carry_ptr + carry_base + (q_off + o_k) * 2 + 1, q_h2)
        tl.store(conv_carry_ptr + carry_base + (k_off + o_k) * 2, k_h1)
        tl.store(conv_carry_ptr + carry_base + (k_off + o_k) * 2 + 1, k_h2)
    tl.store(conv_carry_ptr + carry_base + (v_off + o_v) * 2, v_h1)
    tl.store(conv_carry_ptr + carry_base + (v_off + o_v) * 2 + 1, v_h2)

    p_h = (
        ssm_state_ptr
        + read_state_idx * stride_ssm_slot
        + i_h * V * K
        + (v_base + o_v[:, None]) * K
        + o_k[None, :]
    )
    b_h = tl.load(p_h).to(tl.float32)

    # The convolution weights and decay parameters are invariant across the
    # speculative-token loop. Loading them in the loop creates eight identical
    # VMEM dependency chains per V tile.
    w_q0 = tl.load(conv_weight_ptr + cw_q + o_k * stride_cw_ch).to(tl.float32)
    w_q1 = tl.load(conv_weight_ptr + cw_q + o_k * stride_cw_ch + stride_cw_width).to(
        tl.float32
    )
    w_q2 = tl.load(
        conv_weight_ptr + cw_q + o_k * stride_cw_ch + 2 * stride_cw_width
    ).to(tl.float32)
    w_q3 = tl.load(
        conv_weight_ptr + cw_q + o_k * stride_cw_ch + (W - 1) * stride_cw_width
    ).to(tl.float32)
    w_k0 = tl.load(conv_weight_ptr + cw_k + o_k * stride_cw_ch).to(tl.float32)
    w_k1 = tl.load(conv_weight_ptr + cw_k + o_k * stride_cw_ch + stride_cw_width).to(
        tl.float32
    )
    w_k2 = tl.load(
        conv_weight_ptr + cw_k + o_k * stride_cw_ch + 2 * stride_cw_width
    ).to(tl.float32)
    w_k3 = tl.load(
        conv_weight_ptr + cw_k + o_k * stride_cw_ch + (W - 1) * stride_cw_width
    ).to(tl.float32)
    w_v0 = tl.load(conv_weight_ptr + cw_v + o_v * stride_cw_ch).to(tl.float32)
    w_v1 = tl.load(conv_weight_ptr + cw_v + o_v * stride_cw_ch + stride_cw_width).to(
        tl.float32
    )
    w_v2 = tl.load(
        conv_weight_ptr + cw_v + o_v * stride_cw_ch + 2 * stride_cw_width
    ).to(tl.float32)
    w_v3 = tl.load(
        conv_weight_ptr + cw_v + o_v * stride_cw_ch + (W - 1) * stride_cw_width
    ).to(tl.float32)
    b_dt = tl.load(dt_bias_ptr + i_h * K + o_k).to(tl.float32)
    b_A = tl.load(A_log_ptr + i_h).to(tl.float32)

    # Software-pipeline token-dependent loads while preserving the loop-carried
    # recurrent state dependency.
    n_tok = tl.minimum(seq_t, SPEC_LEN)
    for i_t in tl.range(0, n_tok, num_stages=2):
        tok = bos + i_t
        p_x = x_ptr + tok * stride_x_tok
        b_x_q = tl.load(p_x + q_off + o_k).to(tl.float32)
        b_x_k = tl.load(p_x + k_off + o_k).to(tl.float32)
        b_x_v = tl.load(p_x + v_off + o_v).to(tl.float32)
        b_a = tl.load(gate_ptr + (tok * H + i_h) * K + o_k).to(tl.float32)
        b_beta_raw = tl.load(beta_ptr + tok * stride_beta_tok + i_h).to(tl.float32)

        b_q = q_h0 * w_q0 + q_h1 * w_q1 + q_h2 * w_q2 + b_x_q * w_q3
        b_k = k_h0 * w_k0 + k_h1 * w_k1 + k_h2 * w_k2 + b_x_k * w_k3
        b_v = v_h0 * w_v0 + v_h1 * w_v1 + v_h2 * w_v2 + b_x_v * w_v3
        b_q = b_q * tl.sigmoid(b_q)
        b_k = b_k * tl.sigmoid(b_k)
        b_v = b_v * tl.sigmoid(b_v)
        q_h0, q_h1, q_h2 = q_h1, q_h2, b_x_q
        k_h0, k_h1, k_h2 = k_h1, k_h2, b_x_k
        v_h0, v_h1, v_h2 = v_h1, v_h2, b_x_v

        b_q = b_q * tl.math.rsqrt(tl.sum(b_q * b_q) + 1e-6) * qk_scale
        b_k = b_k * tl.math.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_g = lower_bound * tl.sigmoid(tl.exp(b_A) * (b_a + b_dt))
        b_beta = tl.sigmoid(b_beta_raw)
        b_h = b_h * tl.exp(b_g[None, :])
        b_dot = tl.sum(b_h * b_k[None, :], 1)
        b_u = (b_v - b_dot) * b_beta
        b_h = b_h + b_u[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)

        output_state_idx = tl.load(
            ssm_state_indices_ptr + i_n * stride_indices_seq + i_t * stride_indices_tok
        ).to(tl.int64)
        if output_state_idx >= 0:
            p_h_out = (
                ssm_state_ptr
                + output_state_idx * stride_ssm_slot
                + i_h * V * K
                + (v_base + o_v[:, None]) * K
                + o_k[None, :]
            )
            tl.store(p_h_out, b_h.to(p_h_out.dtype.element_ty))
        tl.store(
            out_ptr + tok * (H * V) + i_h * V + v_base + o_v,
            b_o.to(out_ptr.dtype.element_ty),
        )


@triton.jit(repr=_fused_kda_spec_finalize_repr)
def fused_kda_spec_finalize_kernel(
    x_ptr,
    conv_state_ptr,
    conv_carry_ptr,
    ssm_state_indices_ptr,
    num_accepted_tokens_ptr,
    conv_state_indices_ptr,
    cu_seqlens_ptr,
    norm_weight_ptr,
    out_gate_ptr,
    out_ptr,
    norm_eps,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    W: tl.constexpr,
    SPEC_LEN: tl.constexpr,
    stride_x_tok: tl.constexpr,
    stride_cs_slot: tl.constexpr,
    stride_cs_dim: tl.constexpr,
    stride_cs_pos: tl.constexpr,
    stride_og_tok: tl.constexpr,
):
    """Normalize tiled output and commit speculative convolution state once.

    Launched with SPEC_LEN programs on the token axis, matching the width of the
    conv state reserved for the speculative window.
    """
    i_n = tl.program_id(0)
    i_h = tl.program_id(1)
    i_t = tl.program_id(2)
    tl.assume(stride_x_tok > 0)
    tl.assume(stride_cs_slot > 0)
    tl.assume(stride_cs_dim > 0)
    tl.assume(stride_cs_pos > 0)
    tl.assume(stride_og_tok > 0)
    bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int64)
    seq_t = eos - bos
    if i_t >= seq_t:
        return

    conv_state_idx = tl.load(conv_state_indices_ptr + i_n).to(tl.int64)
    lp: tl.constexpr = H * K
    o_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, K), K), K)
    o_v = tl.max_contiguous(tl.multiple_of(tl.arange(0, V), V), V)
    q_off = i_h * K
    k_off = lp + i_h * K
    v_off = 2 * lp + i_h * V
    p_cs = conv_state_ptr + conv_state_idx * stride_cs_slot
    p_csq = p_cs + (q_off + o_k) * stride_cs_dim
    p_csk = p_cs + (k_off + o_k) * stride_cs_dim
    p_csv = p_cs + (v_off + o_v) * stride_cs_dim

    if i_t == 0:
        carry_base = i_n * (3 * lp * 2)
        tl.assume(carry_base >= 0)
        q_h1 = tl.load(conv_carry_ptr + carry_base + (q_off + o_k) * 2)
        q_h2 = tl.load(conv_carry_ptr + carry_base + (q_off + o_k) * 2 + 1)
        k_h1 = tl.load(conv_carry_ptr + carry_base + (k_off + o_k) * 2)
        k_h2 = tl.load(conv_carry_ptr + carry_base + (k_off + o_k) * 2 + 1)
        v_h1 = tl.load(conv_carry_ptr + carry_base + (v_off + o_v) * 2)
        v_h2 = tl.load(conv_carry_ptr + carry_base + (v_off + o_v) * 2 + 1)
        tl.store(p_csq, q_h1)
        tl.store(p_csq + stride_cs_pos, q_h2)
        tl.store(p_csk, k_h1)
        tl.store(p_csk + stride_cs_pos, k_h2)
        tl.store(p_csv, v_h1)
        tl.store(p_csv + stride_cs_pos, v_h2)

    tok = bos + i_t
    p_x = x_ptr + tok * stride_x_tok
    tl.store(
        p_csq + (W - 2 + i_t) * stride_cs_pos,
        tl.load(p_x + q_off + o_k),
    )
    tl.store(
        p_csk + (W - 2 + i_t) * stride_cs_pos,
        tl.load(p_x + k_off + o_k),
    )
    tl.store(
        p_csv + (W - 2 + i_t) * stride_cs_pos,
        tl.load(p_x + v_off + o_v),
    )

    p_o = out_ptr + tok * (H * V) + i_h * V + o_v
    b_o = tl.load(p_o).to(tl.float32)
    rstd = tl.math.rsqrt(tl.sum(b_o * b_o) / V + norm_eps)
    b_w = tl.load(norm_weight_ptr + o_v).to(tl.float32)
    b_og = tl.load(out_gate_ptr + tok * stride_og_tok + i_h * V + o_v).to(tl.float32)
    b_y = b_o * rstd * b_w * tl.sigmoid(b_og)
    tl.store(p_o, b_y.to(p_o.dtype.element_ty))
