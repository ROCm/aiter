# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_conv2d_nearest_up2x_3x3_repr = make_kernel_repr(
    "_conv2d_nearest_up2x_3x3_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "GROUP_SIZE_M",
        "HAS_BIAS",
        "num_warps",
    ],
)


@triton.jit(repr=_conv2d_nearest_up2x_3x3_repr)
def _conv2d_nearest_up2x_3x3_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    K,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_yn,
    stride_yk,
    stride_yh,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """One output phase (a, b) of conv3x3(nearest_up2x(x)), pad 1, for a tile of
    BLOCK_M input pixels x BLOCK_N output channels.

    Output pixel (2p + a, 2q + b) only sees input rows p + a - 1 + {0, 1} and
    columns q + b - 1 + {0, 1}, so each phase is a 2x2 conv on x with the folded
    weights w[phase, k, tap, c] (tap = 2 * dr + dc). Pixels are the contiguous
    dim of the dot, as in the direct NCHW 3x3 conv kernel.
    """
    pid = tl.program_id(0)
    phase = tl.program_id(1)
    a = phase // 2
    b = phase % 2

    M = N * H * W
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(K, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    n_idx = offs_m // (H * W)
    pq = offs_m % (H * W)
    p_idx = pq // W
    q_idx = pq % W
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < K
    offs_c = tl.arange(0, BLOCK_K)

    x_base = x_ptr + n_idx * stride_xn
    w_base = w_ptr + phase * (K * 4 * C) + offs_n * (4 * C)
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for c0 in range(0, C, BLOCK_K):
        cs = c0 + offs_c
        c_mask = cs < C
        for t in tl.static_range(4):
            ih = p_idx + a - 1 + t // 2
            iw = q_idx + b - 1 + t % 2
            valid = m_mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            x_tile = tl.load(
                x_base[None, :]
                + cs[:, None] * stride_xc
                + ih[None, :] * stride_xh
                + iw[None, :],
                mask=c_mask[:, None] & valid[None, :],
                other=0.0,
            )
            w_tile = tl.load(
                w_base[:, None] + t * C + cs[None, :],
                mask=n_mask[:, None] & c_mask[None, :],
                other=0.0,
            )
            acc = tl.dot(w_tile, x_tile, acc=acc)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc += bias[:, None]

    y_ptrs = (
        y_ptr
        + n_idx[None, :] * stride_yn
        + offs_n[:, None] * stride_yk
        + (2 * p_idx + a)[None, :] * stride_yh
        + (2 * q_idx + b)[None, :]
    )
    tl.store(
        y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=n_mask[:, None] & m_mask[None, :]
    )
