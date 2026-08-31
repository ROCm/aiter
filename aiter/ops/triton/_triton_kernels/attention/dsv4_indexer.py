# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# DSV4 Indexer Triton kernels (forward + backward).
# Fused scoring: einsum + relu + weighted-sum + causal-mask in one kernel.

import triton
import triton.language as tl


@triton.jit
def _indexer_fwd_kernel(
    q_ptr,
    k_ptr,
    w_ptr,
    scores_ptr,
    q_stride_s: tl.int64,
    q_stride_h: tl.int64,
    k_stride_p: tl.int64,
    w_stride_s: tl.int64,
    scores_stride_s: tl.int64,
    S: tl.int32,
    P: tl.int32,
    H: tl.constexpr,
    HD: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_p = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    s_mask = s_offs < S
    p_mask = p_offs < P
    hd_idx = tl.arange(0, HD)

    acc = tl.zeros((BLOCK_S, BLOCK_P), dtype=tl.float32)

    for h in tl.static_range(0, H):
        q_tile = tl.load(
            q_ptr + s_offs[:, None] * q_stride_s + h * q_stride_h + hd_idx[None, :],
            mask=s_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        k_tile = tl.load(
            k_ptr + p_offs[:, None] * k_stride_p + hd_idx[None, :],
            mask=p_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        dot = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32)
        dot = tl.maximum(dot, 0.0)

        w_h = tl.load(
            w_ptr + s_offs * w_stride_s + h,
            mask=s_mask,
            other=0.0,
        ).to(tl.float32)

        acc += dot * w_h[:, None]

    allowed = (p_offs[None, :] + 1) * COMPRESS_RATIO - 1 <= s_offs[:, None]
    acc = tl.where(allowed, acc, float("-inf"))

    tl.store(
        scores_ptr + s_offs[:, None] * scores_stride_s + p_offs[None, :],
        acc,
        mask=s_mask[:, None] & p_mask[None, :],
    )


@triton.jit
def _indexer_bwd_dq_dw_kernel(
    q_ptr,
    k_ptr,
    w_ptr,
    d_scores_ptr,
    dq_ptr,
    dw_ptr,
    q_stride_s: tl.int64,
    q_stride_h: tl.int64,
    k_stride_p: tl.int64,
    w_stride_s: tl.int64,
    ds_stride_s: tl.int64,
    dq_stride_s: tl.int64,
    dq_stride_h: tl.int64,
    dw_stride_s: tl.int64,
    S: tl.int32,
    P: tl.int32,
    H: tl.constexpr,
    HD: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_s = tl.program_id(0)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S
    hd_idx = tl.arange(0, HD)

    for h in tl.static_range(0, H):
        q_tile = tl.load(
            q_ptr + s_offs[:, None] * q_stride_s + h * q_stride_h + hd_idx[None, :],
            mask=s_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        w_h = tl.load(
            w_ptr + s_offs * w_stride_s + h,
            mask=s_mask,
            other=0.0,
        ).to(tl.float32)

        acc_dq = tl.zeros((BLOCK_S, HD), dtype=tl.float32)
        acc_dw = tl.zeros((BLOCK_S,), dtype=tl.float32)

        for p_start in tl.range(0, P, BLOCK_P):
            p_offs = p_start + tl.arange(0, BLOCK_P)
            p_mask = p_offs < P

            k_tile = tl.load(
                k_ptr + p_offs[:, None] * k_stride_p + hd_idx[None, :],
                mask=p_mask[:, None],
                other=0.0,
            ).to(tl.float32)

            d_scores_tile = tl.load(
                d_scores_ptr + s_offs[:, None] * ds_stride_s + p_offs[None, :],
                mask=s_mask[:, None] & p_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            allowed = (p_offs[None, :] + 1) * COMPRESS_RATIO - 1 <= s_offs[:, None]
            d_acc = tl.where(allowed, d_scores_tile, 0.0)

            dot = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32)
            relu_dot = tl.maximum(dot, 0.0)
            relu_mask = dot > 0.0

            acc_dw += tl.sum(d_acc * relu_dot, axis=1)

            d_relu_dot = d_acc * w_h[:, None]
            d_dot = tl.where(relu_mask, d_relu_dot, 0.0)

            acc_dq += tl.dot(d_dot, k_tile, out_dtype=tl.float32)

        tl.store(
            dq_ptr + s_offs[:, None] * dq_stride_s + h * dq_stride_h + hd_idx[None, :],
            acc_dq,
            mask=s_mask[:, None],
        )
        tl.store(
            dw_ptr + s_offs * dw_stride_s + h,
            acc_dw,
            mask=s_mask,
        )


@triton.jit
def _indexer_bwd_dk_kernel(
    q_ptr,
    k_ptr,
    w_ptr,
    d_scores_ptr,
    dk_ptr,
    q_stride_s: tl.int64,
    q_stride_h: tl.int64,
    k_stride_p: tl.int64,
    w_stride_s: tl.int64,
    ds_stride_s: tl.int64,
    dk_stride_p: tl.int64,
    S: tl.int32,
    P: tl.int32,
    H: tl.constexpr,
    HD: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_p = tl.program_id(0)

    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offs < P
    hd_idx = tl.arange(0, HD)

    k_tile = tl.load(
        k_ptr + p_offs[:, None] * k_stride_p + hd_idx[None, :],
        mask=p_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    acc_dk = tl.zeros((BLOCK_P, HD), dtype=tl.float32)

    for s_start in tl.range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S

        d_scores_tile = tl.load(
            d_scores_ptr + s_offs[:, None] * ds_stride_s + p_offs[None, :],
            mask=s_mask[:, None] & p_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        allowed = (p_offs[None, :] + 1) * COMPRESS_RATIO - 1 <= s_offs[:, None]
        d_acc = tl.where(allowed, d_scores_tile, 0.0)

        for h in tl.static_range(0, H):
            q_tile = tl.load(
                q_ptr + s_offs[:, None] * q_stride_s + h * q_stride_h + hd_idx[None, :],
                mask=s_mask[:, None],
                other=0.0,
            ).to(tl.float32)

            w_h = tl.load(
                w_ptr + s_offs * w_stride_s + h,
                mask=s_mask,
                other=0.0,
            ).to(tl.float32)

            dot = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32)
            relu_mask = dot > 0.0

            d_relu_dot = d_acc * w_h[:, None]
            d_dot = tl.where(relu_mask, d_relu_dot, 0.0)

            acc_dk += tl.dot(tl.trans(d_dot), q_tile, out_dtype=tl.float32)

    tl.store(
        dk_ptr + p_offs[:, None] * dk_stride_p + hd_idx[None, :],
        acc_dk,
        mask=p_mask[:, None],
    )
