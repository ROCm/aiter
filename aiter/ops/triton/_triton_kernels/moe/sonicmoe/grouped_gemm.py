# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_grouped_gemm_repr = make_kernel_repr(
    "_grouped_gemm_kernel",
    [
        "N",
        "K",
        "E",
        "SCALE_BLOCK_SIZE",
        "BLOCKWISE_FP8",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "GROUP_SIZE_M",
        "HAS_BIAS",
        "HAS_GATHER_IDX",
        "HAS_SCATTER_IDX",
    ],
)
_grouped_gemm_dw_repr = make_kernel_repr(
    "_grouped_gemm_dw_kernel",
    [
        "N",
        "K",
        "E",
        "SCALE_BLOCK_SIZE",
        "BLOCKWISE_FP8",
        "BLOCK_K",
        "BLOCK_N",
        "BLOCK_T",
        "HAS_GATHER_IDX",
    ],
)


@triton.jit(repr=_grouped_gemm_repr)
def _grouped_gemm_kernel(
    A_ptr,
    B_ptr,
    A_scale_ptr,
    B_scale_ptr,
    C_ptr,
    cu_seqlens_ptr,
    bias_ptr,
    A_idx_ptr,
    scatter_idx_ptr,
    stride_ak,
    stride_am,
    stride_be,
    stride_bk,
    stride_bn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_cm,
    stride_cn,
    stride_bias_e,
    stride_bias_n,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,
    BLOCKWISE_FP8: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_GATHER_IDX: tl.constexpr,
    HAS_SCATTER_IDX: tl.constexpr,
):
    pid = tl.program_id(0)

    cumulative_blocks = 0
    expert_id = 0
    expert_start = 0
    expert_end = 0

    for e in range(E):
        s = tl.load(cu_seqlens_ptr + e).to(tl.int32)
        f = tl.load(cu_seqlens_ptr + e + 1).to(tl.int32)
        m_e = f - s
        blocks_m_e = tl.cdiv(m_e, BLOCK_M)
        blocks_this_expert = blocks_m_e * tl.cdiv(N, BLOCK_N)
        if pid >= cumulative_blocks and pid < cumulative_blocks + blocks_this_expert:
            expert_id = e
            expert_start = s
            expert_end = f
        cumulative_blocks += blocks_this_expert

    # Launching an upper bound avoids copying cu_seqlens to the CPU just to
    # calculate the exact grid size.
    if pid >= cumulative_blocks:
        return

    local_pid = pid
    for e in range(E):
        if e < expert_id:
            s = tl.load(cu_seqlens_ptr + e).to(tl.int32)
            f = tl.load(cu_seqlens_ptr + e + 1).to(tl.int32)
            m_e = f - s
            local_pid -= tl.cdiv(m_e, BLOCK_M) * tl.cdiv(N, BLOCK_N)

    M_expert = expert_end - expert_start
    num_pid_m = tl.cdiv(M_expert, BLOCK_M)
    num_pid_n: tl.constexpr = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = local_pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (local_pid % num_pid_in_group) % group_size_m
    pid_n = (local_pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M_expert
    global_m = expert_start + offs_m

    if HAS_GATHER_IDX:
        a_row_idx = tl.load(A_idx_ptr + global_m, mask=m_mask, other=0).to(tl.int64)
    else:
        a_row_idx = global_m.to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    expert_id_i64 = expert_id.to(tl.int64)
    a_dtype = A_ptr.dtype.element_ty

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < K
        a = tl.load(
            A_ptr
            + a_row_idx[:, None] * stride_ak
            + k_offs[None, :].to(tl.int64) * stride_am,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(a_dtype)
        b = tl.load(
            B_ptr
            + expert_id_i64 * stride_be
            + k_offs[:, None].to(tl.int64) * stride_bk
            + offs_n[None, :].to(tl.int64) * stride_bn,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
        ).to(a_dtype)
        dot = tl.dot(a, b)
        if BLOCKWISE_FP8:
            scale_k = k_start // SCALE_BLOCK_SIZE
            a_scale = tl.load(
                A_scale_ptr + a_row_idx * stride_asm + scale_k * stride_ask,
                mask=m_mask,
                other=0.0,
            )
            b_scale = tl.load(
                B_scale_ptr
                + expert_id_i64 * stride_bse
                + scale_k * stride_bsk
                + (offs_n // SCALE_BLOCK_SIZE).to(tl.int64) * stride_bsn,
                mask=offs_n < N,
                other=0.0,
            )
            dot *= a_scale[:, None] * b_scale[None, :]
        acc += dot

    if HAS_BIAS:
        bias_vals = tl.load(
            bias_ptr
            + expert_id_i64 * stride_bias_e
            + offs_n.to(tl.int64) * stride_bias_n,
            mask=offs_n < N,
            other=0.0,
        )
        acc += bias_vals[None, :]

    c = acc.to(C_ptr.dtype.element_ty)

    if HAS_SCATTER_IDX:
        c_row_idx = tl.load(scatter_idx_ptr + global_m, mask=m_mask, other=0).to(
            tl.int64
        )
    else:
        c_row_idx = global_m.to(tl.int64)

    c_ptrs = (
        C_ptr
        + c_row_idx[:, None] * stride_cm
        + offs_n[None, :].to(tl.int64) * stride_cn
    )
    c_mask = m_mask[:, None] & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit(repr=_grouped_gemm_dw_repr)
def _grouped_gemm_dw_kernel(
    A_ptr,
    B_ptr,
    A_scale_ptr,
    B_scale_ptr,
    C_ptr,
    cu_seqlens_ptr,
    A_idx_ptr,
    stride_ak,
    stride_am,
    stride_bm,
    stride_bn,
    stride_ast,
    stride_ask,
    stride_bst,
    stride_bsn,
    stride_ce,
    stride_ck,
    stride_cn,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,
    BLOCKWISE_FP8: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_GATHER_IDX: tl.constexpr,
):
    pid = tl.program_id(0)
    num_k_blocks: tl.constexpr = tl.cdiv(K, BLOCK_K)
    num_n_blocks: tl.constexpr = tl.cdiv(N, BLOCK_N)
    blocks_per_expert: tl.constexpr = num_k_blocks * num_n_blocks

    expert_id = pid // blocks_per_expert
    local_pid = pid % blocks_per_expert
    pid_k = local_pid // num_n_blocks
    pid_n = local_pid % num_n_blocks

    expert_start = tl.load(cu_seqlens_ptr + expert_id).to(tl.int32)
    expert_end = tl.load(cu_seqlens_ptr + expert_id + 1).to(tl.int32)
    M_expert = expert_end - expert_start
    scale_expert_start = 0
    if BLOCKWISE_FP8:
        for e in range(E):
            if e < expert_id:
                e_start = tl.load(cu_seqlens_ptr + e).to(tl.int32)
                e_end = tl.load(cu_seqlens_ptr + e + 1).to(tl.int32)
                scale_expert_start += tl.cdiv(e_end - e_start, SCALE_BLOCK_SIZE)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_t = tl.arange(0, BLOCK_T)

    k_mask = offs_k < K
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    a_dtype = A_ptr.dtype.element_ty

    for t_start in range(0, M_expert, BLOCK_T):
        t_offs = t_start + offs_t
        t_mask = t_offs < M_expert
        global_t = expert_start + t_offs

        if HAS_GATHER_IDX:
            a_row_idx = tl.load(A_idx_ptr + global_t, mask=t_mask, other=0).to(tl.int64)
        else:
            a_row_idx = global_t.to(tl.int64)

        a = tl.load(
            A_ptr
            + offs_k[:, None].to(tl.int64) * stride_am
            + a_row_idx[None, :] * stride_ak,
            mask=k_mask[:, None] & t_mask[None, :],
            other=0.0,
        ).to(a_dtype)

        b = tl.load(
            B_ptr
            + global_t[:, None].to(tl.int64) * stride_bm
            + offs_n[None, :].to(tl.int64) * stride_bn,
            mask=t_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(a_dtype)

        # Load A directly as [K, T]. Keeping the reduction dimension contiguous
        # in the dot operands avoids the very slow FP8 lowering of tl.trans(a).
        dot = tl.dot(a, b)
        if BLOCKWISE_FP8:
            scale_t = scale_expert_start + t_start // SCALE_BLOCK_SIZE
            a_scale = tl.load(
                A_scale_ptr + scale_t * stride_ast + offs_k.to(tl.int64) * stride_ask,
                mask=k_mask,
                other=0.0,
            )
            b_scale = tl.load(
                B_scale_ptr + scale_t * stride_bst + offs_n.to(tl.int64) * stride_bsn,
                mask=n_mask,
                other=0.0,
            )
            dot *= a_scale[:, None] * b_scale[None, :]
        acc += dot

    c = acc.to(C_ptr.dtype.element_ty)
    expert_id_i64 = expert_id.to(tl.int64)
    c_ptrs = (
        C_ptr
        + expert_id_i64 * stride_ce
        + offs_k[:, None].to(tl.int64) * stride_ck
        + offs_n[None, :].to(tl.int64) * stride_cn
    )
    c_mask = k_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, c, mask=c_mask)
