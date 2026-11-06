# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import triton
import triton.language as tl


@triton.jit
def _gated_delta_rule_step(
    b_h,
    b_k,
    b_v,
    b_a,
    b_b,
    b_A_log,
    b_dt_bias,
    beta,
    threshold,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    x = b_a + b_dt_bias
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    b_g = -tl.exp(b_A_log) * softplus_x

    b_beta = tl.sigmoid(b_b)

    if USE_QK_L2NORM_IN_KERNEL:
        b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
    if not IS_KDA:
        b_h *= tl.exp(b_g)
    else:
        b_h *= tl.exp(b_g[None, :])
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= b_beta
    b_h += b_v[:, None] * b_k[None, :]
    return b_h


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_rearrange_sigmoid_gated_delta_rule_update_kernel(
    A_log,
    a,
    b,
    dt_bias,
    beta,
    threshold,
    qkv,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_qkv_l: tl.constexpr,
    stride_qkv_hd: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
    DEFER_SPEC_STATE: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = qkv + bos * stride_qkv_l + ((i_h * K) + o_k) * stride_qkv_hd
    p_k = qkv + bos * stride_qkv_l + (H * K + (i_h * K) + o_k) * stride_qkv_hd
    p_v = qkv + bos * stride_qkv_l + (2 * H * K + (i_hv * V) + o_v) * stride_qkv_hd

    p_A_log = A_log + i_hv
    if not IS_KDA:
        p_a = a + bos * HV + i_hv
        p_dt_bias = dt_bias + i_hv
    else:
        p_a = a + (bos * HV + i_hv) * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k

    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    b_A_log = tl.load(p_A_log).to(tl.float32)
    b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

    # DEFER_SPEC_STATE stores the state only after the first token (slot 0) and
    # records the inputs of the other tokens in slot 1. The next step replays the
    # accepted records, so each step stores one state instead of one per token.
    STASH_W: tl.constexpr = BK + BV + 2
    o_r = tl.arange(0, BV)
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if DEFER_SPEC_STATE:
        state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
        if state_idx < 0:
            return
        p_h0 = h0 + state_idx * stride_init_state_token
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
        stash_idx = tl.load(
            ssm_state_indices + i_n * stride_indices_seq + stride_indices_tok
        ).to(tl.int64)
        # A row without a record slot (-1) neither replays nor records; the
        # clamp keeps its record pointer inside the state pool.
        has_stash = stash_idx >= 0
        # Each program has its own records, so replay reads only its own data.
        p_stash = (
            h0
            + tl.maximum(stash_idx, 0) * stride_init_state_token
            + (i_hv * tl.num_programs(1) + i_v) * STASH_W
        )
        stash_stride = HV * tl.num_programs(1) * STASH_W
        num_replay = tl.where(
            has_stash, tl.load(num_accepted_tokens + i_n).to(tl.int32) - 1, 0
        )
        for i_r in range(num_replay):
            p_r = p_stash + i_r * stash_stride
            b_h = _gated_delta_rule_step(
                b_h,
                tl.load(p_r + o_k, mask=mask_k, other=0).to(tl.float32),
                tl.load(p_r + BK + o_r, mask=mask_v, other=0).to(tl.float32),
                tl.load(p_r + BK + BV).to(tl.float32),
                tl.load(p_r + BK + BV + 1).to(tl.float32),
                b_A_log,
                b_dt_bias,
                beta,
                threshold,
                USE_QK_L2NORM_IN_KERNEL,
                IS_KDA,
            )
        # Read all records before this step overwrites them.
        tl.debug_barrier()
    elif USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t * stride_indices_tok
            ).to(tl.int64)
            if state_idx < 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)
        if DEFER_SPEC_STATE and i_t > 0:
            p_r = p_stash + (i_t - 1) * stash_stride
            tl.store(p_r + o_k, b_k.to(p_r.dtype.element_ty), mask=mask_k & has_stash)
            tl.store(
                p_r + BK + o_r, b_v.to(p_r.dtype.element_ty), mask=mask_v & has_stash
            )
            tl.store(p_r + BK + BV, b_a.to(p_r.dtype.element_ty), mask=has_stash)
            tl.store(p_r + BK + BV + 1, b_b.to(p_r.dtype.element_ty), mask=has_stash)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
        b_q = b_q * scale
        b_h = _gated_delta_rule_step(
            b_h,
            b_k,
            b_v,
            b_a,
            b_b,
            b_A_log,
            b_dt_bias,
            beta,
            threshold,
            USE_QK_L2NORM_IN_KERNEL,
            IS_KDA,
        )
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if DEFER_SPEC_STATE:
            if i_t == 0:
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)
        elif INPLACE_FINAL_STATE:
            final_state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t * stride_indices_tok
            ).to(tl.int64)
            if final_state_idx >= 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += stride_qkv_l
        p_k += stride_qkv_l
        p_v += stride_qkv_l
        p_o += HV * V
        p_b += HV
        if not IS_KDA:
            p_a += HV
        else:
            p_a += HV * K
