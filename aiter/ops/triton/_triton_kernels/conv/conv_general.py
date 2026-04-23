# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
import triton
import triton.language as tl
from .helpers import _tanh, AUTOTUNE_GENERAL_CONFIGS


@triton.autotune(
    configs=AUTOTUNE_GENERAL_CONFIGS,
    key=["M_total", "K_out", "K_pad"],
    reset_to_zero=["Y"],
    warmup=50,
    rep=200,
    cache_results=True,
)
@triton.jit
def _conv2d_general_kernel(
    X,
    WK,
    BIAS,
    Y,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W_in: tl.constexpr,
    K_out: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    K_pad: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dil_h: tl.constexpr,
    dil_w: tl.constexpr,
    M_total: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT_TYPE: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    """General conv kernel with precomputed bases.
    LAYOUT: 0=NCHW, 1=NHWC
    """
    # WK is always [K_out, K_pad] contiguous
    stride_wk_kout: tl.constexpr = K_pad
    stride_wk_kred: tl.constexpr = 1
    if LAYOUT == 0:
        # NCHW: X[N, C, H, W_in], Y[N, K_out, P, Q]
        stride_x_n: tl.constexpr = C * H * W_in
        stride_x_c: tl.constexpr = H * W_in
        stride_x_h: tl.constexpr = W_in
        stride_x_w: tl.constexpr = 1
        stride_y_n: tl.constexpr = K_out * P * Q
        stride_y_k: tl.constexpr = P * Q
        stride_y_p: tl.constexpr = Q
        stride_y_q: tl.constexpr = 1
    else:
        # NHWC: X[N, H, W_in, C], Y[N, P, Q, K_out]
        stride_x_n: tl.constexpr = H * W_in * C
        stride_x_c: tl.constexpr = 1
        stride_x_h: tl.constexpr = W_in * C
        stride_x_w: tl.constexpr = C
        stride_y_n: tl.constexpr = P * Q * K_out
        stride_y_k: tl.constexpr = 1
        stride_y_p: tl.constexpr = Q * K_out
        stride_y_q: tl.constexpr = K_out

    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M_total, BLOCK_M)
    num_pid_n = tl.cdiv(K_out, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m >= num_pid_m:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    kout_mask = offs_n < K_out

    # Decode offs_m -> (n_idx, p_idx, q_idx)
    n_idx = offs_m[:, None] // (P * Q)
    pq = offs_m[:, None] % (P * Q)
    p_idx = pq // Q
    q_idx = pq % Q

    n_valid = n_idx < N

    # Precompute base positions
    base_oh = p_idx * stride_h - pad_h
    base_ow = q_idx * stride_w - pad_w
    stride_dh = dil_h * stride_x_h
    stride_dw = dil_w * stride_x_w
    x_base = X + n_idx * stride_x_n + base_oh * stride_x_h + base_ow * stride_x_w
    wk_base = WK + offs_n[None, :] * stride_wk_kout

    Y_ptrs = (
        Y
        + n_idx * stride_y_n
        + offs_n[None, :] * stride_y_k
        + p_idx * stride_y_p
        + q_idx * stride_y_q
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)
    rs_stride = R * S

    for k0 in range(0, K_pad, BLOCK_K):
        kred = k0 + offs_k

        WK_ptrs = wk_base + kred[:, None] * stride_wk_kred
        w_tile = tl.load(WK_ptrs, mask=kout_mask[None, :], other=0.0)

        c = kred // rs_stride
        rs = kred % rs_stride
        r = rs // S
        s = rs % S

        oh = base_oh + r * dil_h
        ow = base_ow + s * dil_w

        X_ptrs = x_base + c * stride_x_c + r * stride_dh + s * stride_dw
        x_mask = (
            n_valid & (oh >= 0) & (ow >= 0) & (oh < H) & (ow < W_in) & (c[None, :] < C)
        )
        x_tile = tl.load(X_ptrs, mask=x_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile, out_dtype=tl.float32)

    if HAS_BIAS:
        b = tl.load(BIAS + offs_n, mask=offs_n < K_out, other=0.0)
        acc += b[None, :]

    if ACT_TYPE == 1:
        acc = tl.maximum(acc, 0)
    elif ACT_TYPE == 2:
        acc = tl.minimum(tl.maximum(acc, 0), 6)
    elif ACT_TYPE == 3:
        acc = (
            0.5 * acc * (1.0 + _tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))
        )

    tl.store(
        Y_ptrs,
        acc,
        mask=(n_valid & (p_idx < P) & (q_idx < Q) & kout_mask[None, :]),
    )
