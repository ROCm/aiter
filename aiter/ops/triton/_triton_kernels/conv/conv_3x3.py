# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.conv_config_utils import get_conv_config

from ..activation import _apply_activation_from_str


def _get_config_nhwc(shape_key=None, M=None, variants=()):
    return get_conv_config("CONV-3X3-NHWC", shape_key=shape_key, M=M, variants=variants)


def _get_config_cblocked(shape_key=None, M=None, variants=()):
    return get_conv_config(
        "CONV-3X3-CBLOCKED", shape_key=shape_key, M=M, variants=variants
    )


def _get_config_nchw(shape_key=None, M=None, variants=()):
    return get_conv_config("CONV-3X3-NCHW", shape_key=shape_key, M=M, variants=variants)


_conv2d_3x3_nhwc_kernel_repr = make_kernel_repr(
    "_conv2d_3x3_nhwc_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "GROUP_SIZE_M",
        "HAS_BIAS",
        "ACTIVATION",
    ],
)


_conv2d_3x3_cblocked_kernel_repr = make_kernel_repr(
    "_conv2d_3x3_cblocked_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "GROUP_SIZE_M",
        "HAS_BIAS",
        "ACTIVATION",
    ],
)


_conv2d_3x3_nchw_kernel_repr = make_kernel_repr(
    "_conv2d_3x3_nchw_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "GROUP_SIZE_M",
        "HAS_BIAS",
        "ACTIVATION",
        "ROW_ALIGNED",
    ],
)


@triton.jit(repr=_conv2d_3x3_nhwc_kernel_repr)
def _conv2d_3x3_nhwc_kernel(
    X,
    W,
    BIAS,
    Y,
    N,
    C: tl.constexpr,
    H: tl.constexpr,
    W_in: tl.constexpr,
    K_out: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    C_pad: tl.constexpr,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    M_total,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Specialized 3x3 NHWC kernel: stride_x_c=1 and stride_y_k=1 hardcoded
    so the compiler can emit coalesced vector loads/stores."""
    # X layout: [N, H, W_in, C] contiguous NHWC (stride_x_c=1 hardcoded in load logic)
    stride_x_w: tl.constexpr = C
    stride_x_h: tl.constexpr = W_in * C
    stride_x_n: tl.constexpr = H * W_in * C
    # W layout: [K_out, 9, C_pad] contiguous
    stride_w_c: tl.constexpr = 1
    stride_w_rs: tl.constexpr = C_pad
    stride_w_kout: tl.constexpr = 9 * C_pad
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M_total
    kout_mask = offs_n < K_out

    # Decode (n, p, q) from linear index
    n_idx = offs_m[:, None] // (P * Q)
    pq = offs_m[:, None] % (P * Q)
    p_idx = pq // Q
    q_idx = pq % Q
    n_valid = n_idx < N

    # Precompute base positions
    base_ih = p_idx * stride_h - pad_h
    base_iw = q_idx * stride_w - pad_w
    stride_x_dh = dil_h * stride_x_h
    stride_x_dw = dil_w * stride_x_w

    x_base = X + n_idx * stride_x_n + base_ih * stride_x_h + base_iw * stride_x_w

    # Weight base: W[K_out, 9, C_pad]
    w_base = W + offs_n[None, :] * stride_w_kout

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for r in tl.static_range(3):
        ih = base_ih + r * dil_h
        valid_ih = n_valid & (ih >= 0) & (ih < H)
        for s in tl.static_range(3):
            rs_idx = r * 3 + s
            iw = base_iw + s * dil_w
            spatial_valid = valid_ih & (iw >= 0) & (iw < W_in)

            for k0 in range(0, C_pad, BLOCK_K):
                k_offs = k0 + offs_k
                k_mask = k_offs < C

                x_ptrs = x_base + k_offs[None, :] + r * stride_x_dh + s * stride_x_dw
                w_ptrs = w_base + rs_idx * stride_w_rs + k_offs[:, None] * stride_w_c

                x_tile = tl.load(
                    x_ptrs, mask=spatial_valid & k_mask[None, :], other=0.0
                )
                w_tile = tl.load(
                    w_ptrs, mask=k_mask[:, None] & kout_mask[None, :], other=0.0
                )
                acc = tl.dot(x_tile, w_tile, acc=acc)

    # Epilogue: bias + activation + store
    if HAS_BIAS:
        b = tl.load(BIAS + offs_n, mask=offs_n < K_out, other=0.0).to(tl.float32)
        acc += b[None, :]

    acc = _apply_activation_from_str(acc, ACTIVATION)

    y_ptrs = (
        Y
        + n_idx * stride_y_n
        + offs_n[None, :]
        + p_idx * stride_y_p
        + q_idx * stride_y_q
    )
    tl.store(y_ptrs, acc, mask=(m_mask[:, None] & kout_mask[None, :]))


@triton.jit(repr=_conv2d_3x3_cblocked_kernel_repr)
def _conv2d_3x3_cblocked_kernel(
    X,
    W,
    BIAS,
    Y,
    N,
    C: tl.constexpr,
    H: tl.constexpr,
    W_in: tl.constexpr,
    K_out: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    C_pad: tl.constexpr,
    Cb: tl.constexpr,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    M_total,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Specialized 3x3 kernel for channel-blocked [N, C_blocks, H, W, Cb] input.
    stride_x_cb=1 is hardcoded so the compiler emits coalesced vector loads."""
    # X layout: [N, C_blocks, H, W_in, Cb] where C_blocks = C_pad // Cb
    stride_x_cb: tl.constexpr = 1
    stride_x_w: tl.constexpr = Cb
    stride_x_h: tl.constexpr = W_in * Cb
    stride_x_cblock: tl.constexpr = H * W_in * Cb
    stride_x_n: tl.constexpr = (C_pad // Cb) * H * W_in * Cb
    # W layout: [K_out, 9, C_pad] contiguous
    stride_w_c: tl.constexpr = 1
    stride_w_rs: tl.constexpr = C_pad
    stride_w_kout: tl.constexpr = 9 * C_pad
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Valid output mask
    m_mask = offs_m < M_total
    kout_mask = offs_n < K_out

    # Decode (n, p, q) from linear index
    n_idx = offs_m[:, None] // (P * Q)
    pq = offs_m[:, None] % (P * Q)
    p_idx = pq // Q
    q_idx = pq % Q
    n_valid = n_idx < N

    # Precompute base positions
    base_ih = p_idx * stride_h - pad_h
    base_iw = q_idx * stride_w - pad_w
    stride_x_dh = dil_h * stride_x_h
    stride_x_dw = dil_w * stride_x_w

    # x_base for channel-blocked layout: X[n, cblock, h, w, k_local]
    # base pointer accounts for n, h, w
    x_base = X + n_idx * stride_x_n + base_ih * stride_x_h + base_iw * stride_x_w

    # Weight base: W[K_out, 9, C_pad]
    w_base = W + offs_n[None, :] * stride_w_kout

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for r in tl.static_range(3):
        ih = base_ih + r * dil_h
        valid_ih = n_valid & (ih >= 0) & (ih < H)
        for s in tl.static_range(3):
            rs_idx = r * 3 + s
            iw = base_iw + s * dil_w
            spatial_valid = valid_ih & (iw >= 0) & (iw < W_in)

            for k0 in range(0, C_pad, BLOCK_K):
                k_offs = k0 + offs_k
                k_mask = k_offs < C

                # Compute cblock index and local offset within block
                cblock_idx = k_offs // Cb
                k_local = k_offs % Cb

                x_ptrs = (
                    x_base
                    + cblock_idx[None, :] * stride_x_cblock
                    + k_local[None, :] * stride_x_cb
                    + r * stride_x_dh
                    + s * stride_x_dw
                )
                w_ptrs = w_base + rs_idx * stride_w_rs + k_offs[:, None] * stride_w_c

                x_tile = tl.load(
                    x_ptrs, mask=spatial_valid & k_mask[None, :], other=0.0
                )
                w_tile = tl.load(
                    w_ptrs, mask=k_mask[:, None] & kout_mask[None, :], other=0.0
                )
                acc = tl.dot(x_tile, w_tile, acc=acc)

    # Epilogue: bias + activation + store
    if HAS_BIAS:
        b = tl.load(BIAS + offs_n, mask=offs_n < K_out, other=0.0).to(tl.float32)
        acc += b[None, :]

    acc = _apply_activation_from_str(acc, ACTIVATION)

    y_ptrs = (
        Y
        + n_idx * stride_y_n
        + offs_n[None, :] * stride_y_k
        + p_idx * stride_y_p
        + q_idx * stride_y_q
    )
    tl.store(y_ptrs, acc, mask=(m_mask[:, None] & kout_mask[None, :]))


@triton.jit(repr=_conv2d_3x3_nchw_kernel_repr)
def _conv2d_3x3_nchw_kernel(
    X,
    W,
    BIAS,
    Y,
    N,
    C: tl.constexpr,
    H: tl.constexpr,
    W_in: tl.constexpr,
    K_out: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    C_pad: tl.constexpr,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    M_total,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    ROW_ALIGNED: tl.constexpr,
):
    """3x3 convolution reading contiguous NCHW without an activation pack.

    The dot operands are transposed so the pixel dimension, rather than the
    channel dimension, is contiguous across lanes. Weight padding is masked;
    the input keeps its true C-sized batch stride and is never channel-padded.
    """
    stride_x_w: tl.constexpr = 1
    stride_x_h: tl.constexpr = W_in
    stride_x_c: tl.constexpr = H * W_in
    stride_x_n: tl.constexpr = C * H * W_in
    stride_w_c: tl.constexpr = 1
    stride_w_rs: tl.constexpr = C_pad
    stride_w_kout: tl.constexpr = 9 * C_pad
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

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    kout_mask = offs_n < K_out

    if ROW_ALIGNED:
        row = (pid_m * BLOCK_M) // Q
        q_idx = (pid_m * BLOCK_M - row * Q) + tl.arange(0, BLOCK_M)
        p_idx = (row % P) + tl.zeros((BLOCK_M,), dtype=tl.int32)
        n_idx = (row // P) + tl.zeros((BLOCK_M,), dtype=tl.int32)
        m_mask = tl.full((BLOCK_M,), 1, dtype=tl.int1)
    else:
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < M_total
        n_idx = offs_m // (P * Q)
        pq = offs_m % (P * Q)
        p_idx = pq // Q
        q_idx = pq % Q
    n_valid = n_idx < N

    base_ih = p_idx * stride_h - pad_h
    base_iw = q_idx * stride_w - pad_w
    stride_x_dh = dil_h * stride_x_h
    stride_x_dw = dil_w * stride_x_w
    x_base = X + n_idx * stride_x_n + base_ih * stride_x_h + base_iw
    w_base = W + offs_n * stride_w_kout
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for r in tl.static_range(3):
        ih = base_ih + r * dil_h
        valid_ih = n_valid & (ih >= 0) & (ih < H)
        for s in tl.static_range(3):
            rs_idx = r * 3 + s
            iw = base_iw + s * dil_w
            spatial_valid = valid_ih & (iw >= 0) & (iw < W_in)
            x_tap = x_base + r * stride_x_dh + s * stride_x_dw

            for k0 in range(0, C_pad, BLOCK_K):
                k_offs = k0 + offs_k
                k_mask = k_offs < C
                x_ptrs = x_tap[None, :] + k_offs[:, None] * stride_x_c
                w_ptrs = (
                    w_base[:, None]
                    + rs_idx * stride_w_rs
                    + k_offs[None, :] * stride_w_c
                )
                x_tile = tl.load(
                    x_ptrs, mask=spatial_valid[None, :] & k_mask[:, None], other=0.0
                )
                w_tile = tl.load(
                    w_ptrs, mask=kout_mask[:, None] & k_mask[None, :], other=0.0
                )
                acc = tl.dot(w_tile, x_tile, acc=acc)

    if HAS_BIAS:
        b = tl.load(BIAS + offs_n, mask=kout_mask, other=0.0).to(tl.float32)
        acc += b[:, None]

    acc = _apply_activation_from_str(acc, ACTIVATION)

    y_ptrs = (
        Y
        + n_idx[None, :] * stride_y_n
        + offs_n[:, None] * stride_y_k
        + p_idx[None, :] * stride_y_p
        + q_idx[None, :] * stride_y_q
    )
    tl.store(y_ptrs, acc, mask=(kout_mask[:, None] & m_mask[None, :]))
