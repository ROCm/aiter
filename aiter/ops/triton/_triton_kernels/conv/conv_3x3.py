# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
import triton
import triton.language as tl
from .helpers import (
    _tanh,
    AUTOTUNE_3x3_NHWC_CONFIGS,
    AUTOTUNE_3x3_CBLOCKED_CONFIGS,
)


@triton.autotune(
    configs=AUTOTUNE_3x3_NHWC_CONFIGS,
    key=["M_total", "K_out", "C_pad"],
    reset_to_zero=["Y"],
    warmup=50,
    rep=200,
    cache_results=True,
)
@triton.jit
def _conv2d_3x3_nhwc_kernel(
    X,
    W3,
    BIAS,
    Y,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W_in: tl.constexpr,
    K_out: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    C_pad: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dil_h: tl.constexpr,
    dil_w: tl.constexpr,
    M_total: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    """Specialized 3x3 NHWC kernel: stride_x_c=1 and stride_y_k=1 hardcoded
    so the compiler can emit coalesced vector loads/stores."""
    # X layout: [N, H, W_in, C] contiguous NHWC (stride_x_c=1 hardcoded in load logic)
    stride_x_w: tl.constexpr = C
    stride_x_h: tl.constexpr = W_in * C
    stride_x_n: tl.constexpr = H * W_in * C
    # W3 layout: [K_out, 9, C_pad] contiguous
    stride_w3_c: tl.constexpr = 1
    stride_w3_rs: tl.constexpr = C_pad
    stride_w3_kout: tl.constexpr = 9 * C_pad
    # Y layout: [N, P, Q, K_out] contiguous NHWC (stride_y_k=1 hardcoded in store logic)
    stride_y_q: tl.constexpr = K_out
    stride_y_p: tl.constexpr = Q * K_out
    stride_y_n: tl.constexpr = P * Q * K_out

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

    # Decode (n, p, q) from linear index
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

    # Weight base: W3[K_out, 9, C_pad]
    w_base = W3 + offs_n[None, :] * stride_w3_kout

    Y_ptrs = (
        Y
        + n_idx * stride_y_n
        + offs_n[None, :]
        + p_idx * stride_y_p
        + q_idx * stride_y_q
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_c = tl.arange(0, BLOCK_C)

    for r in tl.static_range(3):
        oh = base_oh + r * dil_h
        valid_oh = n_valid & (oh >= 0) & (oh < H)
        x_off_r = r * stride_dh
        for s in tl.static_range(3):
            rs_idx = r * 3 + s
            ow = base_ow + s * dil_w
            valid = valid_oh & (ow >= 0) & (ow < W_in)

            for c0 in range(0, C_pad, BLOCK_C):
                c_offs = c0 + offs_c
                c_mask = c_offs < C

                x_ptrs = x_base + c_offs[None, :] + x_off_r + s * stride_dw
                w_ptrs = w_base + rs_idx * stride_w3_rs + c_offs[:, None] * stride_w3_c

                x_tile = tl.load(x_ptrs, mask=valid & c_mask[None, :], other=0.0)
                w_tile = tl.load(
                    w_ptrs, mask=c_mask[:, None] & kout_mask[None, :], other=0.0
                )
                acc += tl.dot(x_tile, w_tile, out_dtype=tl.float32)

    # Epilogue: bias + activation + store
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


@triton.autotune(
    configs=AUTOTUNE_3x3_CBLOCKED_CONFIGS,
    key=["M_total", "K_out", "C_pad"],
    reset_to_zero=["Y"],
    warmup=50,
    rep=200,
    cache_results=True,
)
@triton.jit
def _conv2d_3x3_cblocked_kernel(
    X,
    W3,
    BIAS,
    Y,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W_in: tl.constexpr,
    K_out: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    C_pad: tl.constexpr,
    Cb: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dil_h: tl.constexpr,
    dil_w: tl.constexpr,
    M_total: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    """Specialized 3x3 kernel for channel-blocked [N, C_blocks, H, W, Cb] input.
    stride_c_local=1 is hardcoded so the compiler emits coalesced vector loads."""
    # X layout: [N, C_blocks, H, W_in, Cb] where C_blocks = C_pad // Cb
    stride_x_w: tl.constexpr = Cb
    stride_x_h: tl.constexpr = W_in * Cb
    stride_x_cblock: tl.constexpr = H * W_in * Cb
    stride_x_n: tl.constexpr = (C_pad // Cb) * H * W_in * Cb
    # W3 layout: [K_out, 9, C_pad] contiguous
    stride_w3_c: tl.constexpr = 1
    stride_w3_rs: tl.constexpr = C_pad
    stride_w3_kout: tl.constexpr = 9 * C_pad
    # Y layout: [N, K_out, P, Q] contiguous NCHW
    stride_y_q: tl.constexpr = 1
    stride_y_p: tl.constexpr = Q
    stride_y_k: tl.constexpr = P * Q
    stride_y_n: tl.constexpr = K_out * P * Q

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

    # Decode (n, p, q) from linear index
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

    # x_base for channel-blocked layout: X[n, cblock, h, w, c_local]
    # base pointer accounts for n, h, w
    x_base = X + n_idx * stride_x_n + base_oh * stride_x_h + base_ow * stride_x_w

    # Weight base: W3[K_out, 9, C_pad]
    w_base = W3 + offs_n[None, :] * stride_w3_kout

    # Y pointers
    Y_ptrs = (
        Y
        + n_idx * stride_y_n
        + offs_n[None, :] * stride_y_k
        + p_idx * stride_y_p
        + q_idx * stride_y_q
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_c = tl.arange(0, BLOCK_C)

    for r in tl.static_range(3):
        oh = base_oh + r * dil_h
        valid_oh = n_valid & (oh >= 0) & (oh < H)
        x_off_r = r * stride_dh
        for s in tl.static_range(3):
            rs_idx = r * 3 + s
            ow = base_ow + s * dil_w
            valid = valid_oh & (ow >= 0) & (ow < W_in)

            for c0 in range(0, C_pad, BLOCK_C):
                c_offs = c0 + offs_c
                c_mask = c_offs < C

                # Compute cblock index and local offset within block
                cblock_idx = c_offs // Cb
                c_local = c_offs % Cb

                x_ptrs = (
                    x_base
                    + cblock_idx[None, :] * stride_x_cblock
                    + c_local[None, :]
                    + x_off_r
                    + s * stride_dw
                )
                w_ptrs = w_base + rs_idx * stride_w3_rs + c_offs[:, None] * stride_w3_c

                x_tile = tl.load(x_ptrs, mask=valid & c_mask[None, :], other=0.0)
                w_tile = tl.load(
                    w_ptrs, mask=c_mask[:, None] & kout_mask[None, :], other=0.0
                )
                acc += tl.dot(x_tile, w_tile, out_dtype=tl.float32)

    # Epilogue: bias + activation + store
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
