# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_mfalcon_attention_kernel_repr = make_kernel_repr(
    "_mfalcon_attention_kernel",
    ["BLOCK_M", "BLOCK_N", "BLOCK_D", "RETURN_LSE", "ALLOW_TF32"],
)


@triton.jit(repr=_mfalcon_attention_kernel_repr)
def _mfalcon_attention_kernel(
    Q,
    K,
    V,
    Out,
    LSE,
    seq_offsets,
    num_candidates,
    candidate_offsets,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    stride_lse_t,
    stride_lse_h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    seq_start = tl.load(seq_offsets + pid_b).to(tl.int64)
    seq_end = tl.load(seq_offsets + pid_b + 1).to(tl.int64)
    seq_len = (seq_end - seq_start).to(tl.int32)
    num_c = tl.load(num_candidates + pid_b).to(tl.int32)
    history_len = seq_len - num_c
    if history_len < 0:
        return

    candidate_start = seq_start + history_len.to(tl.int64)
    out_start = tl.load(candidate_offsets + pid_b).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < num_c
    if pid_m * BLOCK_M >= num_c:
        return

    offs_d = tl.arange(0, BLOCK_D)
    head = pid_h.to(tl.int64)
    q_ptrs = Q + (
        (candidate_start + offs_m.to(tl.int64))[:, None] * stride_qt
        + head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    m_i = tl.where(mask_m, float("-inf"), 0.0)
    l_i = tl.where(mask_m, 0.0, 1.0)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, history_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < history_len
        history_tokens = seq_start + offs_n.to(tl.int64)
        k_ptrs = K + (
            history_tokens[None, :] * stride_kt
            + head * stride_kh
            + offs_d[:, None] * stride_kd
        )
        v_ptrs = V + (
            history_tokens[:, None] * stride_vt
            + head * stride_vh
            + offs_d[None, :] * stride_vd
        )
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * sm_scale * 1.4426950408889634
        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        l_new = l_i * tl.math.exp2(m_i - m_new) + tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_new)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, allow_tf32=ALLOW_TF32)
        m_i = m_new
        l_i = l_new

    candidate_tokens = candidate_start + offs_m.to(tl.int64)
    k_ptrs = K + (
        candidate_tokens[None, :] * stride_kt
        + head * stride_kh
        + offs_d[:, None] * stride_kd
    )
    v_ptrs = V + (
        candidate_tokens[:, None] * stride_vt
        + head * stride_vh
        + offs_d[None, :] * stride_vd
    )
    k = tl.load(k_ptrs, mask=mask_m[None, :], other=0.0)
    v = tl.load(v_ptrs, mask=mask_m[:, None], other=0.0)
    self_qk = tl.sum(q * tl.trans(k), 1) * sm_scale * 1.4426950408889634
    m_new = tl.maximum(m_i, self_qk)
    p = tl.math.exp2(self_qk - m_new)
    l_new = l_i * tl.math.exp2(m_i - m_new) + p
    alpha = tl.math.exp2(m_i - m_new)
    acc = acc * alpha[:, None] + p[:, None] * v
    m_i = m_new
    l_i = l_new

    out_ptrs = Out + (
        (out_start + offs_m.to(tl.int64))[:, None] * stride_ot
        + head * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(
        out_ptrs,
        (acc / l_i[:, None]).to(Out.dtype.element_ty),
        mask=mask_m[:, None],
    )
    if RETURN_LSE:
        lse = m_i * 0.6931471805599453 + tl.log(l_i)
        lse_ptrs = LSE + (
            (out_start + offs_m.to(tl.int64)) * stride_lse_t + head * stride_lse_h
        )
        tl.store(lse_ptrs, lse, mask=mask_m)
