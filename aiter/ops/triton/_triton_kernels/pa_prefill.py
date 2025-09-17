# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# The kernels in this file are adapted from LightLLM's context_attention_fwd:
# https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    K_cache,
    V_cache,
    B_Loc,
    sm_scale,
    k_scale,
    v_scale,
    B_Start_Loc,
    B_Seqlen,
    block_size,
    x,
    Out,
    stride_b_loc_b,
    stride_b_loc_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_k_cache_d,
    stride_k_cache_bl,
    stride_k_cache_x,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_v_cache_d,
    stride_v_cache_bl,
    num_queries_per_kv: int,
    IN_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,  # head size
    BLOCK_DMODEL_PADDED: tl.constexpr,  # head size padded to a power of 2
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    SKIP_DECODE: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // num_queries_per_kv

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

    if SKIP_DECODE and cur_batch_query_len == 1:
        return

    # start position inside of the query
    # generally, N goes over kv, while M goes over query_len
    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    # [N]; starts at 0
    offs_n = tl.arange(0, BLOCK_N)
    # [D]; starts at 0
    offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
    # [M]; starts at current position in query
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # [M,D]
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )

    dim_mask = tl.where(tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0).to(
        tl.int1
    )  # [D]

    q = tl.load(
        Q + off_q,
        mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len),
        other=0.0,
    )  # [M,D]

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # [M]
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # [M]
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)  # [M,D]

    # compute query against context (no causal mask here)
    for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        bn = tl.load(
            B_Loc
            + cur_batch * stride_b_loc_b
            + ((start_n + offs_n) // block_size) * stride_b_loc_s,
            mask=(start_n + offs_n) < cur_batch_ctx_len,
            other=0,
        )  # [N]
        # [D,N]
        off_k = (
            bn[None, :] * stride_k_cache_bs
            + cur_kv_head * stride_k_cache_h
            + (offs_d[:, None] // x) * stride_k_cache_d
            + ((start_n + offs_n[None, :]) % block_size) * stride_k_cache_bl
            + (offs_d[:, None] % x) * stride_k_cache_x
        )
        # [N,D]
        off_v = (
            bn[:, None] * stride_v_cache_bs
            + cur_kv_head * stride_v_cache_h
            + offs_d[None, :] * stride_v_cache_d
            + (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl
        )
        k_load = tl.load(
            K_cache + off_k,
            mask=dim_mask[:, None] & ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
            other=0.0,
        )  # [D,N]

        if k_load.dtype.is_fp8():
            k = (k_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
        else:
            k = k_load

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)  # [M,N]
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        qk = tl.where(
            (start_n + offs_n[None, :]) < cur_batch_ctx_len, qk, float("-inf")
        )
        qk *= sm_scale
        if SLIDING_WINDOW > 0:
            # (cur_batch_ctx_len + offs_m[:, None]) are the positions of
            # Q entries in sequence
            # (start_n + offs_n[None, :]) are the positions of
            # KV entries in sequence
            # So the condition makes sure each entry in Q only attends
            # to KV entries not more than SLIDING_WINDOW away.
            #
            # We can't use -inf here, because the
            # sliding window may lead to the entire row being masked.
            # This then makes m_ij contain -inf, which causes NaNs in
            # exp().
            qk = tl.where(
                (cur_batch_ctx_len + offs_m[:, None]) - (start_n + offs_n[None, :])
                < SLIDING_WINDOW,
                qk,
                -10000,
            )

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)  # [M]
        p = tl.exp(qk - m_ij[:, None])  # [M,N]
        l_ij = tl.sum(p, 1)  # [M]
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)  # [M]
        alpha = tl.exp(m_i - m_i_new)  # [M]
        beta = tl.exp(m_ij - m_i_new)  # [M]
        l_i_new = alpha * l_i + beta * l_ij  # [M]

        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v_load = tl.load(
            V_cache + off_v,
            mask=dim_mask[None, :] & ((start_n + offs_n[:, None]) < cur_batch_ctx_len),
            other=0.0,
        )  # [N,D]
        if v_load.dtype.is_fp8():
            v = (v_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
        else:
            v = v_load
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
        # # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    off_k = (
        offs_n[None, :] * stride_kbs
        + cur_kv_head * stride_kh
        + offs_d[:, None] * stride_kd
    )
    off_v = (
        offs_n[:, None] * stride_vbs
        + cur_kv_head * stride_vh
        + offs_d[None, :] * stride_vd
    )
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # block_mask is 0 when we're already past the current query length
    block_mask = tl.where(block_start_loc < cur_batch_query_len, 1, 0)

    # compute query against itself (with causal mask)
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=dim_mask[:, None]
            & ((start_n + offs_n[None, :]) < cur_batch_query_len),
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        qk *= sm_scale
        # apply causal mask
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        if SLIDING_WINDOW > 0:
            qk = tl.where(
                offs_m[:, None] - (start_n + offs_n[None, :]) < SLIDING_WINDOW,
                qk,
                -10000,
            )

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=dim_mask[None, :]
            & ((start_n + offs_n[:, None]) < cur_batch_query_len),
            other=0.0,
        )
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs,
        acc,
        mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len),
    )
    return


@triton.jit
def _fwd_kernel_alibi(
    Q,
    K,
    V,
    K_cache,
    V_cache,
    B_Loc,
    sm_scale,
    k_scale,
    v_scale,
    B_Start_Loc,
    B_Seqlen,
    Alibi_slopes,
    block_size,
    x,
    Out,
    stride_b_loc_b,
    stride_b_loc_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_k_cache_d,
    stride_k_cache_bl,
    stride_k_cache_x,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_v_cache_d,
    stride_v_cache_bl,
    num_queries_per_kv: int,
    IN_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,  # head size
    BLOCK_DMODEL_PADDED: tl.constexpr,  # head size padded to a power of 2
    BLOCK_N: tl.constexpr,
    SKIP_DECODE: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    # attn_bias[]
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // num_queries_per_kv

    # cur_batch_seq_len: the length of prompts
    # cur_batch_ctx_len: the length of prefix
    # cur_batch_in_all_start_index: the start id of the dim=0
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

    if SKIP_DECODE and cur_batch_query_len == 1:
        return

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )

    dim_mask = tl.where(tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0).to(
        tl.int1
    )

    q = tl.load(
        Q + off_q,
        mask=dim_mask[None, :]
        & (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len),
        other=0.0,
    )

    # # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)

    alibi_slope = tl.load(Alibi_slopes + cur_head)
    alibi_start_q = tl.arange(0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
    alibi_start_k = 0
    for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        bn = tl.load(
            B_Loc
            + cur_batch * stride_b_loc_b
            + ((start_n + offs_n) // block_size) * stride_b_loc_s,
            mask=(start_n + offs_n) < cur_batch_ctx_len,
            other=0,
        )
        off_k = (
            bn[None, :] * stride_k_cache_bs
            + cur_kv_head * stride_k_cache_h
            + (offs_d[:, None] // x) * stride_k_cache_d
            + ((start_n + offs_n[None, :]) % block_size) * stride_k_cache_bl
            + (offs_d[:, None] % x) * stride_k_cache_x
        )
        off_v = (
            bn[:, None] * stride_v_cache_bs
            + cur_kv_head * stride_v_cache_h
            + offs_d[None, :] * stride_v_cache_d
            + (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl
        )
        k_load = tl.load(
            K_cache + off_k,
            mask=dim_mask[:, None] & ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
            other=0.0,
        )  # [D,N]

        if k_load.dtype.is_fp8():
            k = (k_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
        else:
            k = k_load

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        qk = tl.where(
            (start_n + offs_n[None, :]) < cur_batch_ctx_len, qk, float("-inf")
        )
        qk *= sm_scale

        # load alibi
        alibi = (
            tl.arange(0, BLOCK_N)[None, :] + alibi_start_k - alibi_start_q[:, None]
        ) * alibi_slope
        alibi = tl.where(
            (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
            alibi,
            float("-inf"),
        )
        qk += alibi
        alibi_start_k += BLOCK_N

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.math.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i

        alpha = tl.math.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij
        # -- update output accumulator --
        # scale p
        # scale acc
        acc_scale = alpha
        # acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v_load = tl.load(
            V_cache + off_v,
            mask=dim_mask[None, :] & ((start_n + offs_n[:, None]) < cur_batch_ctx_len),
            other=0.0,
        )
        if v_load.dtype.is_fp8():
            v = (v_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
        else:
            v = v_load
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision="ieee")
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    off_k = (
        offs_n[None, :] * stride_kbs
        + cur_kv_head * stride_kh
        + offs_d[:, None] * stride_kd
    )
    off_v = (
        offs_n[:, None] * stride_vbs
        + cur_kv_head * stride_vh
        + offs_d[None, :] * stride_vd
    )
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    block_mask = tl.where(block_start_loc < cur_batch_seq_len - cur_batch_ctx_len, 1, 0)

    # init alibi
    alibi_slope = tl.load(Alibi_slopes + cur_head)
    alibi_start_q = tl.arange(0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
    alibi_start_k = cur_batch_ctx_len
    # # init debugger
    # offset_db_q = tl.arange(0, BLOCK_M) + block_start_loc
    # offset_db_k = tl.arange(0, BLOCK_N)
    # calc q[BLOCK_M, BLOCK_MODEL] mul k[prefix_len: , BLOCK_DMODEL]
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=dim_mask[:, None]
            & ((start_n + offs_n[None, :]) < cur_batch_seq_len - cur_batch_ctx_len),
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision="ieee")
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        # load alibi
        alibi = (
            tl.arange(0, BLOCK_N)[None, :] + alibi_start_k - alibi_start_q[:, None]
        ) * alibi_slope
        alibi = tl.where(
            (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
            alibi,
            float("-inf"),
        )
        qk += alibi
        alibi_start_k += BLOCK_N

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.math.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i

        alpha = tl.math.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij
        # -- update output accumulator --
        # scale p
        # scale acc
        acc_scale = alpha
        # acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=dim_mask[None, :]
            & ((start_n + offs_n[:, None]) < cur_batch_seq_len - cur_batch_ctx_len),
            other=0.0,
        )
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision="ieee")
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    acc = acc / l_i[:, None]

    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs,
        acc,
        mask=dim_mask[None, :]
        & (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len),
    )
    return
