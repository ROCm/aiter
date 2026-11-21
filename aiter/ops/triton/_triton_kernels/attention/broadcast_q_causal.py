# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_broadcast_q_causal_forward_repr = make_kernel_repr(
    "_broadcast_q_causal_forward",
    ["BLOCK_M", "BLOCK_N", "HEAD_DIM"],
)
_broadcast_q_causal_dq_repr = make_kernel_repr(
    "_broadcast_q_causal_dq",
    ["BLOCK_M", "BLOCK_N", "HEAD_DIM", "BATCH_PER_GROUP"],
)
_broadcast_q_causal_dkdv_repr = make_kernel_repr(
    "_broadcast_q_causal_dkdv",
    ["BLOCK_M", "BLOCK_N", "HEAD_DIM"],
)


@triton.jit(repr=_broadcast_q_causal_forward_repr)
def _broadcast_q_causal_forward(
    q_ptr,
    k_ptr,
    v_ptr,
    cu_seqlens_k_ptr,
    out_ptr,
    lse_ptr,
    q_len,
    softmax_scale,
    stride_q_m,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_h,
    stride_k_d,
    stride_v_n,
    stride_v_h,
    stride_v_d,
    stride_o_m,
    stride_o_h,
    stride_o_d,
    stride_lse_m,
    stride_lse_h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch = tl.program_id(0)
    query_block = tl.program_id(1)
    head = tl.program_id(2)

    kv_start = tl.load(cu_seqlens_k_ptr + batch)
    kv_end = tl.load(cu_seqlens_k_ptr + batch + 1)
    history_len = kv_end - kv_start
    active_len = tl.minimum(q_len, history_len)
    if query_block * BLOCK_M >= q_len:
        return

    offs_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q_mask = offs_m[:, None] < q_len
    q_ptrs = (
        q_ptr
        + offs_m[:, None] * stride_q_m
        + head * stride_q_h
        + offs_d[None, :] * stride_q_d
    )
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    key_limit = tl.minimum(active_len, (query_block + 1) * BLOCK_M)

    for key_start in tl.range(0, key_limit, BLOCK_N):
        key_offsets = key_start + offs_n
        key_mask = key_offsets < active_len
        k_ptrs = (
            k_ptr
            + (kv_start + key_offsets[:, None]) * stride_k_n
            + head * stride_k_h
            + offs_d[None, :] * stride_k_d
        )
        v_ptrs = (
            v_ptr
            + (kv_start + key_offsets[:, None]) * stride_v_n
            + head * stride_v_h
            + offs_d[None, :] * stride_v_d
        )
        k = tl.load(k_ptrs, mask=key_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=key_mask[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        causal_mask = (
            (offs_m[:, None] >= key_offsets[None, :])
            & (offs_m[:, None] < active_len)
            & key_mask[None, :]
        )
        qk = tl.where(causal_mask, qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_new_safe)
        p = tl.exp(qk - m_new_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / l_safe[:, None]
    out = tl.where(l_i[:, None] > 0.0, out, 0.0)
    lse = tl.where(l_i > 0.0, m_i + tl.log(l_i), -float("inf"))

    out_rows = batch * q_len + offs_m
    out_mask = offs_m < q_len
    out_ptrs = (
        out_ptr
        + out_rows[:, None] * stride_o_m
        + head * stride_o_h
        + offs_d[None, :] * stride_o_d
    )
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=out_mask[:, None])
    lse_ptrs = out_rows * stride_lse_m + head * stride_lse_h
    tl.store(lse_ptr + lse_ptrs, lse, mask=out_mask)


@triton.jit(repr=_broadcast_q_causal_dq_repr)
def _broadcast_q_causal_dq(
    q_ptr,
    k_ptr,
    v_ptr,
    cu_seqlens_k_ptr,
    lse_ptr,
    out_ptr,
    do_ptr,
    dq_partial_ptr,
    q_len,
    batch_count,
    softmax_scale,
    stride_q_m,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_h,
    stride_k_d,
    stride_v_n,
    stride_v_h,
    stride_v_d,
    stride_do_m,
    stride_do_h,
    stride_do_d,
    stride_lse_m,
    stride_lse_h,
    stride_o_m,
    stride_o_h,
    stride_o_d,
    stride_dq_g,
    stride_dq_m,
    stride_dq_h,
    stride_dq_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BATCH_PER_GROUP: tl.constexpr,
):
    group = tl.program_id(0)
    query_block = tl.program_id(1)
    head = tl.program_id(2)

    offs_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q_mask = offs_m[:, None] < q_len
    q_ptrs = (
        q_ptr
        + offs_m[:, None] * stride_q_m
        + head * stride_q_h
        + offs_d[None, :] * stride_q_d
    )
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for group_offset in tl.range(0, BATCH_PER_GROUP):
        batch = group * BATCH_PER_GROUP + group_offset
        if batch < batch_count:
            kv_start = tl.load(cu_seqlens_k_ptr + batch)
            kv_end = tl.load(cu_seqlens_k_ptr + batch + 1)
            history_len = kv_end - kv_start
            active_len = tl.minimum(q_len, history_len)
            if query_block * BLOCK_M < active_len:
                out_rows = batch * q_len + offs_m
                do_mask = offs_m[:, None] < active_len
                do_ptrs = (
                    do_ptr
                    + out_rows[:, None] * stride_do_m
                    + head * stride_do_h
                    + offs_d[None, :] * stride_do_d
                )
                do = tl.load(do_ptrs, mask=do_mask, other=0.0)
                lse = tl.load(
                    lse_ptr + out_rows * stride_lse_m + head * stride_lse_h,
                    mask=offs_m < active_len,
                    other=-float("inf"),
                )
                out_ptrs = (
                    out_ptr
                    + out_rows[:, None] * stride_o_m
                    + head * stride_o_h
                    + offs_d[None, :] * stride_o_d
                )
                out = tl.load(out_ptrs, mask=do_mask, other=0.0)
                key_limit = tl.minimum(active_len, (query_block + 1) * BLOCK_M)

                for key_start in tl.range(0, key_limit, BLOCK_N):
                    key_offsets = key_start + offs_n
                    key_mask = key_offsets < active_len
                    k_ptrs = (
                        k_ptr
                        + (kv_start + key_offsets[:, None]) * stride_k_n
                        + head * stride_k_h
                        + offs_d[None, :] * stride_k_d
                    )
                    v_ptrs = (
                        v_ptr
                        + (kv_start + key_offsets[:, None]) * stride_v_n
                        + head * stride_v_h
                        + offs_d[None, :] * stride_v_d
                    )
                    k = tl.load(k_ptrs, mask=key_mask[:, None], other=0.0)
                    v = tl.load(v_ptrs, mask=key_mask[:, None], other=0.0)
                    qk = tl.dot(q, tl.trans(k)) * softmax_scale
                    causal_mask = (
                        (offs_m[:, None] >= key_offsets[None, :])
                        & (offs_m[:, None] < active_len)
                        & key_mask[None, :]
                    )
                    p = tl.where(causal_mask, tl.exp(qk - lse[:, None]), 0.0)
                    dp = tl.dot(do, tl.trans(v))
                    delta = tl.sum(do * out, axis=1)
                    ds = p * (dp - delta[:, None])
                    acc += tl.dot(ds.to(k.dtype), k) * softmax_scale

    dq_ptrs = (
        dq_partial_ptr
        + group * stride_dq_g
        + offs_m[:, None] * stride_dq_m
        + head * stride_dq_h
        + offs_d[None, :] * stride_dq_d
    )
    tl.store(dq_ptrs, acc, mask=q_mask)


@triton.jit(repr=_broadcast_q_causal_dkdv_repr)
def _broadcast_q_causal_dkdv(
    q_ptr,
    k_ptr,
    v_ptr,
    cu_seqlens_k_ptr,
    lse_ptr,
    out_ptr,
    do_ptr,
    dk_ptr,
    dv_ptr,
    q_len,
    softmax_scale,
    stride_q_m,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_h,
    stride_k_d,
    stride_v_n,
    stride_v_h,
    stride_v_d,
    stride_do_m,
    stride_do_h,
    stride_do_d,
    stride_lse_m,
    stride_lse_h,
    stride_o_m,
    stride_o_h,
    stride_o_d,
    stride_dk_n,
    stride_dk_h,
    stride_dk_d,
    stride_dv_n,
    stride_dv_h,
    stride_dv_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch = tl.program_id(0)
    key_block = tl.program_id(1)
    head = tl.program_id(2)

    kv_start = tl.load(cu_seqlens_k_ptr + batch)
    kv_end = tl.load(cu_seqlens_k_ptr + batch + 1)
    history_len = kv_end - kv_start
    active_len = tl.minimum(q_len, history_len)
    if key_block * BLOCK_N >= active_len:
        return

    offs_n = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    key_mask = offs_n < active_len
    k_ptrs = (
        k_ptr
        + (kv_start + offs_n[:, None]) * stride_k_n
        + head * stride_k_h
        + offs_d[None, :] * stride_k_d
    )
    v_ptrs = (
        v_ptr
        + (kv_start + offs_n[:, None]) * stride_v_n
        + head * stride_v_h
        + offs_d[None, :] * stride_v_d
    )
    k = tl.load(k_ptrs, mask=key_mask[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=key_mask[:, None], other=0.0)
    dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    query_start = (key_block * BLOCK_N // BLOCK_M) * BLOCK_M

    for query_block_start in tl.range(query_start, active_len, BLOCK_M):
        offs_m = query_block_start + tl.arange(0, BLOCK_M)
        out_rows = batch * q_len + offs_m
        q_mask = offs_m[:, None] < q_len
        do_mask = offs_m[:, None] < active_len
        q_ptrs = (
            q_ptr
            + offs_m[:, None] * stride_q_m
            + head * stride_q_h
            + offs_d[None, :] * stride_q_d
        )
        do_ptrs = (
            do_ptr
            + out_rows[:, None] * stride_do_m
            + head * stride_do_h
            + offs_d[None, :] * stride_do_d
        )
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        do = tl.load(do_ptrs, mask=do_mask, other=0.0)
        lse = tl.load(
            lse_ptr + out_rows * stride_lse_m + head * stride_lse_h,
            mask=offs_m < active_len,
            other=-float("inf"),
        )
        out_ptrs = (
            out_ptr
            + out_rows[:, None] * stride_o_m
            + head * stride_o_h
            + offs_d[None, :] * stride_o_d
        )
        out = tl.load(out_ptrs, mask=do_mask, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        causal_mask = (
            (offs_m[:, None] >= offs_n[None, :])
            & (offs_m[:, None] < active_len)
            & key_mask[None, :]
        )
        p = tl.where(causal_mask, tl.exp(qk - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        delta = tl.sum(do * out, axis=1)
        ds = p * (dp - delta[:, None])
        dk_acc += tl.dot(tl.trans(ds.to(q.dtype)), q) * softmax_scale
        dv_acc += tl.dot(tl.trans(p.to(do.dtype)), do)

    dk_ptrs = (
        dk_ptr
        + (kv_start + offs_n[:, None]) * stride_dk_n
        + head * stride_dk_h
        + offs_d[None, :] * stride_dk_d
    )
    dv_ptrs = (
        dv_ptr
        + (kv_start + offs_n[:, None]) * stride_dv_n
        + head * stride_dv_h
        + offs_d[None, :] * stride_dv_d
    )
    tl.store(dk_ptrs, dk_acc.to(dk_ptr.dtype.element_ty), mask=key_mask[:, None])
    tl.store(dv_ptrs, dv_acc.to(dv_ptr.dtype.element_ty), mask=key_mask[:, None])
