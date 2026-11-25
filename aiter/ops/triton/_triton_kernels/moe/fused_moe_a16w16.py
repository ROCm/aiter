# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# BF16 x BF16 stages of aiter.fused_moe_2stages. They read the moe_sorting
# output: sorted_token_ids holds (topk_slot << 24) | token, padded with
# token == num_tokens, and sorted_expert_ids holds one expert per BLOCK_M rows.
# Weights are [E, N, K], plain or in the shuffle_weight((16, 16)) layout.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit
def _load_b_tile(
    b_ptr,
    offs_n_tile,
    k0,
    K,
    stride_bn,
    N_LIMIT,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SHUFFLED: tl.constexpr,
):
    """Return the [BLOCK_K, BLOCK_N] tile W[n0:n0+BLOCK_N, k0:k0+BLOCK_K]^T."""
    if SHUFFLED:
        # Each 16 x 32 (n x k) block is stored as [k // 8][n][k % 8], so the
        # K slice of one 16-row block is BLOCK_K * 16 contiguous elements.
        NB: tl.constexpr = BLOCK_N // 16
        KB: tl.constexpr = BLOCK_K // 32
        nb = offs_n_tile // 16 + tl.arange(0, NB)
        j = tl.arange(0, BLOCK_K * 16)
        ptrs = b_ptr + nb[:, None] * (16 * K) + k0 * 16 + j[None, :]
        b = tl.load(ptrs, mask=(nb[:, None] * 16) < N_LIMIT, other=0.0)
        b = tl.reshape(b, (NB, KB, 4, 16, 8))
        b = tl.permute(b, (1, 2, 4, 0, 3))
        return tl.reshape(b, (BLOCK_K, BLOCK_N))
    else:
        offs_n = offs_n_tile + tl.arange(0, BLOCK_N)
        offs_k = k0 + tl.arange(0, BLOCK_K)
        ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None]
        return tl.load(ptrs, mask=offs_n[None, :] < N_LIMIT, other=0.0)


@triton.jit
def _block_ids(
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_M_BLOCKS,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(NUM_M_BLOCKS - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


_stage1_repr = make_kernel_repr(
    "_fused_moe_a16w16_stage1_kernel",
    ["TOPK", "BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M", "SHUFFLED"],
)
_stage2_repr = make_kernel_repr(
    "_fused_moe_a16w16_stage2_kernel",
    [
        "TOPK",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "GROUP_M",
        "SHUFFLED",
        "MUL_ROUTED_WEIGHT",
    ],
)


@triton.jit(repr=_stage1_repr)
def _fused_moe_a16w16_stage1_kernel(
    a_ptr,
    w1_ptr,
    out_ptr,
    sorted_token_ids_ptr,
    sorted_expert_ids_ptr,
    num_valid_ids_ptr,
    num_tokens,
    inter_dim,
    K,
    stride_am,
    stride_we,
    stride_wn,
    stride_om,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SHUFFLED: tl.constexpr,
    NUM_M_BLOCKS,
):
    """out[t, slot, n] = silu(x[t] @ w1[e, n]) * (x[t] @ w1[e, inter_dim + n])."""
    pid_m, pid_n = _block_ids(inter_dim, BLOCK_M, BLOCK_N, GROUP_M, NUM_M_BLOCKS)
    num_valid = tl.load(num_valid_ids_ptr)
    if pid_m * BLOCK_M >= num_valid:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    packed = tl.load(sorted_token_ids_ptr + offs_m)
    token = packed & 0xFFFFFF
    slot = packed >> 24
    row_ok = token < num_tokens
    expert = tl.load(sorted_expert_ids_ptr + pid_m).to(tl.int64)

    n0 = pid_n * BLOCK_N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + token.to(tl.int64)[:, None] * stride_am + offs_k[None, :]
    w_gate = w1_ptr + expert * stride_we
    w_up = w_gate + inter_dim.to(tl.int64) * K

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=row_ok[:, None], other=0.0)
        bg = _load_b_tile(
            w_gate, n0, k0, K, stride_wn, inter_dim, BLOCK_N, BLOCK_K, SHUFFLED
        )
        bu = _load_b_tile(
            w_up, n0, k0, K, stride_wn, inter_dim, BLOCK_N, BLOCK_K, SHUFFLED
        )
        acc_g = tl.dot(a, bg, acc_g)
        acc_u = tl.dot(a, bu, acc_u)
        a_ptrs += BLOCK_K

    act = acc_g / (1.0 + tl.exp(-acc_g)) * acc_u
    offs_n = n0 + tl.arange(0, BLOCK_N)
    out_row = token.to(tl.int64) * TOPK + slot
    o_ptrs = out_ptr + out_row[:, None] * stride_om + offs_n[None, :]
    tl.store(
        o_ptrs,
        act.to(out_ptr.dtype.element_ty),
        mask=row_ok[:, None] & (offs_n[None, :] < inter_dim),
    )


@triton.jit(repr=_stage2_repr)
def _fused_moe_a16w16_stage2_kernel(
    a_ptr,
    w2_ptr,
    out_ptr,
    sorted_token_ids_ptr,
    sorted_expert_ids_ptr,
    sorted_weights_ptr,
    num_valid_ids_ptr,
    num_tokens,
    N,
    K,
    stride_am,
    stride_we,
    stride_wn,
    stride_om,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SHUFFLED: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    NUM_M_BLOCKS,
):
    """out[t, n] += w_route * (inter[t, slot] @ w2[e, n])."""
    pid_m, pid_n = _block_ids(N, BLOCK_M, BLOCK_N, GROUP_M, NUM_M_BLOCKS)
    num_valid = tl.load(num_valid_ids_ptr)
    if pid_m * BLOCK_M >= num_valid:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    packed = tl.load(sorted_token_ids_ptr + offs_m)
    token = packed & 0xFFFFFF
    slot = packed >> 24
    row_ok = token < num_tokens
    expert = tl.load(sorted_expert_ids_ptr + pid_m).to(tl.int64)

    n0 = pid_n * BLOCK_N
    offs_k = tl.arange(0, BLOCK_K)
    a_row = token.to(tl.int64) * TOPK + slot
    a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_k[None, :]
    w = w2_ptr + expert * stride_we

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=row_ok[:, None], other=0.0)
        b = _load_b_tile(w, n0, k0, K, stride_wn, N, BLOCK_N, BLOCK_K, SHUFFLED)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K

    if MUL_ROUTED_WEIGHT:
        route_w = tl.load(sorted_weights_ptr + offs_m, mask=row_ok, other=0.0)
        acc = acc * route_w[:, None]
    offs_n = n0 + tl.arange(0, BLOCK_N)
    o_ptrs = out_ptr + token.to(tl.int64)[:, None] * stride_om + offs_n[None, :]
    tl.atomic_add(
        o_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=row_ok[:, None] & (offs_n[None, :] < N),
        sem="relaxed",
    )
