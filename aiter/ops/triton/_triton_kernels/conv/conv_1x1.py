# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
import triton
import triton.language as tl

from .helpers import _tanh, AUTOTUNE_1x1_CONFIGS


@triton.autotune(
    configs=AUTOTUNE_1x1_CONFIGS,
    key=["M_total", "K_out", "C"],
    reset_to_zero=["Y"],
    warmup=50,
    rep=200,
    cache_results=True,
)
@triton.jit
def _conv2d_1x1_kernel(
    X,
    W,
    BIAS,
    Y,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W_in: tl.constexpr,
    K_out: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    M_total: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT_TYPE: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    """
    Specialized 1x1 convolution kernel.
    - No R*S loop (R=S=1)
    - Direct channel reduction
    - Simplified pointer arithmetic
    LAYOUT: 0=NCHW, 1=NHWC
    """
    # W is always [K_out, C] contiguous
    stride_w_k: tl.constexpr = C
    stride_w_c: tl.constexpr = 1
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

    # M = N * P * Q (output spatial), N_dim = K_out (output channels)
    num_pid_m = tl.cdiv(M_total, BLOCK_M)
    num_pid_n = tl.cdiv(K_out, BLOCK_N)

    # L2 cache swizzle pattern
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m >= num_pid_m:
        return

    # Compute output tile indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Decode (n, p, q) from linear index
    n_idx = offs_m // (P * Q)
    pq = offs_m % (P * Q)
    p_idx = pq // Q
    q_idx = pq % Q

    # Valid output mask
    m_mask = offs_m < M_total
    n_mask = offs_n < K_out

    ih = p_idx * stride_h - pad_h
    iw = q_idx * stride_w - pad_w

    # Check spatial bounds
    spatial_valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W_in) & (n_idx < N)

    # Base pointers for this output tile
    x_base = X + n_idx * stride_x_n + ih * stride_x_h + iw * stride_x_w  # [BLOCK_M]
    w_base = W + offs_n * stride_w_k  # [BLOCK_N]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Channel reduction loop
    for k0 in range(0, C, BLOCK_K):
        k_offs = k0 + offs_k
        k_mask = k_offs < C

        # Load input: X[n, c, ih, iw] -> shape [BLOCK_M, BLOCK_K]
        x_ptrs = x_base[:, None] + k_offs[None, :] * stride_x_c
        x_mask = spatial_valid[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load weight: W[k_out, c] -> shape [BLOCK_K, BLOCK_N]
        w_ptrs = w_base[None, :] + k_offs[:, None] * stride_w_c
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc += tl.dot(x_tile, w_tile, out_dtype=tl.float32)

    # Bias
    if HAS_BIAS:
        b = tl.load(BIAS + offs_n, mask=n_mask, other=0.0)
        acc += b[None, :]

    # Activation
    if ACT_TYPE == 1:  # ReLU
        acc = tl.maximum(acc, 0)
    elif ACT_TYPE == 2:  # ReLU6
        acc = tl.minimum(tl.maximum(acc, 0), 6)
    elif ACT_TYPE == 3:  # GELU
        acc = (
            0.5 * acc * (1.0 + _tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))
        )

    # Store output: Y[n, k, p, q]
    y_ptrs = (
        Y
        + n_idx[:, None] * stride_y_n
        + offs_n[None, :] * stride_y_k
        + p_idx[:, None] * stride_y_p
        + q_idx[:, None] * stride_y_q
    )
    y_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)
