# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from aiter.ops.triton.utils.conv_config_utils import get_conv_config
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from .helpers import CONV_AUTOTUNE_ENABLED
from ..activation import _relu, _relu6, _gelu_tanh


def _get_config(shape_key=None, M=None):
    if CONV_AUTOTUNE_ENABLED:
        return {}
    return get_conv_config("CONV-GENERAL", shape_key=shape_key, M=M)


_conv2d_general_kernel_repr = make_kernel_repr(
    "_conv2d_general_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "GROUP_SIZE_M",
        "HAS_BIAS",
        "ACTIVATION",
        "LAYOUT",
    ],
)


@triton.jit(repr=_conv2d_general_kernel_repr)
def _conv2d_general_kernel(
    X,
    W,
    BIAS,
    Y,
    N,
    C: tl.constexpr,
    H: tl.constexpr,
    W_in: tl.constexpr,
    K_out: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    K_pad: tl.constexpr,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    M_total,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # blocks the flattened C*R*S GEMM-K axis (see loop below), not channels
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    """General conv kernel with precomputed bases.
    LAYOUT: "nchw" or "nhwc"
    """
    # W is always [K_out, K_pad] contiguous
    stride_w_kout: tl.constexpr = K_pad
    stride_w_kred: tl.constexpr = 1
    if LAYOUT == "nchw":
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M_total
    kout_mask = offs_n < K_out

    # Decode offs_m -> (n_idx, p_idx, q_idx)
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

    # Input base: X[N, C, H, W_in] (NCHW) or X[N, H, W_in, C] (NHWC)
    x_base = X + n_idx * stride_x_n + base_ih * stride_x_h + base_iw * stride_x_w
    # Weight base: W[K_out, K_pad] contiguous (K_pad = padded C*R*S)
    w_base = W + offs_n[None, :] * stride_w_kout

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    rs_stride = R * S

    # Reduction is over the flattened im2col GEMM-K axis (C*R*S = K_pad), not
    # channels alone: one BLOCK_K step can straddle channel/filter-tap
    # boundaries, so each kred is decoded below into (c, r, s). Here the GEMM-K
    # axis is genuinely C*R*S, whereas in 1x1/3x3 the same BLOCK_K names a
    # reduction that spans only channels (R=S=1, or the 3x3 taps looped
    # separately in static_range).
    for k0 in range(0, K_pad, BLOCK_K):
        kred = k0 + offs_k

        w_ptrs = w_base + kred[:, None] * stride_w_kred
        w_tile = tl.load(w_ptrs, mask=kout_mask[None, :], other=0.0)

        c = kred // rs_stride
        rs = kred % rs_stride
        r = rs // S
        s = rs % S

        ih = base_ih + r * dil_h
        iw = base_iw + s * dil_w

        X_ptrs = x_base + c * stride_x_c + r * stride_x_dh + s * stride_x_dw
        x_mask = (
            n_valid & (ih >= 0) & (iw >= 0) & (ih < H) & (iw < W_in) & (c[None, :] < C)
        )
        x_tile = tl.load(X_ptrs, mask=x_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Epilogue: bias + activation + store
    if HAS_BIAS:
        b = tl.load(BIAS + offs_n, mask=offs_n < K_out, other=0.0)
        acc += b[None, :]

    if ACTIVATION == "relu":
        acc = _relu(acc)
    elif ACTIVATION == "relu6":
        acc = _relu6(acc)
    elif ACTIVATION == "gelu":
        acc = _gelu_tanh(acc)

    y_ptrs = (
        Y
        + n_idx * stride_y_n
        + offs_n[None, :] * stride_y_k
        + p_idx * stride_y_p
        + q_idx * stride_y_q
    )
    tl.store(y_ptrs, acc, mask=(m_mask[:, None] & kout_mask[None, :]))


# Autotune search space (used when AITER_TRITON_CONV_AUTOTUNE=1).
AUTOTUNE_GENERAL_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=1,
    ),
]


if CONV_AUTOTUNE_ENABLED:
    _conv2d_general_kernel = triton.autotune(
        configs=AUTOTUNE_GENERAL_CONFIGS,
        key=["M_total", "K_out", "K_pad"],
        cache_results=True,
    )(_conv2d_general_kernel)
