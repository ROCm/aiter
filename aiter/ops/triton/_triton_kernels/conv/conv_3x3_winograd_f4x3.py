# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.conv.winograd_f4x3_common import (
    _winograd_f4x3_input_transform,
    _winograd_f4x3_output_transform,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.conv_config_utils import get_conv_config


def _get_config_input(shape_key=None, M=None):
    return get_conv_config("CONV-WINO-F4X3-INPUT", shape_key=shape_key, M=M)


def _get_config_gemm(shape_key=None, M=None):
    return get_conv_config("CONV-WINO-F4X3-GEMM", shape_key=shape_key, M=M)


def _get_config_output(shape_key=None, M=None):
    return get_conv_config("CONV-WINO-F4X3-OUTPUT", shape_key=shape_key, M=M)


_winograd_f4x3_input_transform_kernel_repr = make_kernel_repr(
    "_winograd_f4x3_input_transform_kernel",
    ["BLOCK_C", "LAYOUT"],
)


_winograd_f4x3_cblocked_input_transform_kernel_repr = make_kernel_repr(
    "_winograd_f4x3_cblocked_input_transform_kernel",
    ["BLOCK_C"],
)


_winograd_f4x3_batched_gemm_kernel_repr = make_kernel_repr(
    "_winograd_f4x3_batched_gemm_kernel",
    ["BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_SIZE_M"],
)


_winograd_f4x3_output_transform_kernel_repr = make_kernel_repr(
    "_winograd_f4x3_output_transform_kernel",
    ["BLOCK_K", "HAS_BIAS", "ACTIVATION", "LAYOUT"],
)


@triton.jit(repr=_winograd_f4x3_input_transform_kernel_repr)
def _winograd_f4x3_input_transform_kernel(
    X,
    V,
    N,
    C: tl.constexpr,
    C_pad: tl.constexpr,
    H: tl.constexpr,
    W_in: tl.constexpr,
    tile_H: tl.constexpr,
    tile_W: tl.constexpr,
    T: tl.constexpr,
    pad_h,
    pad_w,
    BLOCK_C: tl.constexpr,
    LAYOUT: tl.constexpr = "nchw",
):
    if LAYOUT == "nchw":
        stride_x_w: tl.constexpr = 1
        stride_x_h: tl.constexpr = W_in
        stride_x_c: tl.constexpr = H * W_in
        stride_x_n: tl.constexpr = C * H * W_in
    else:
        stride_x_w: tl.constexpr = C
        stride_x_h: tl.constexpr = W_in * C
        stride_x_c: tl.constexpr = 1
        stride_x_n: tl.constexpr = H * W_in * C

    tile_idx = tl.program_id(0)
    c_block = tl.program_id(1)

    n = tile_idx // (tile_H * tile_W)
    rem = tile_idx % (tile_H * tile_W)
    th = rem // tile_W
    tw = rem % tile_W

    h_start = th * 4 - pad_h
    w_start = tw * 4 - pad_w

    offs_c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    base = X + n * stride_x_n + offs_c * stride_x_c

    _winograd_f4x3_input_transform(
        X,
        V,
        base,
        tile_idx,
        offs_c,
        c_mask,
        n < N,
        h_start,
        w_start,
        H,
        W_in,
        C_pad,
        T,
        stride_x_h,
        stride_x_w,
        BLOCK_C,
    )


@triton.jit(repr=_winograd_f4x3_cblocked_input_transform_kernel_repr)
def _winograd_f4x3_cblocked_input_transform_kernel(
    X,
    V,
    N,
    C: tl.constexpr,
    C_pad: tl.constexpr,
    H: tl.constexpr,
    W_in: tl.constexpr,
    tile_H: tl.constexpr,
    tile_W: tl.constexpr,
    T: tl.constexpr,
    pad_h,
    pad_w,
    Cb: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # X: NCHWc [N, C_blocks, H, W_in, Cb]
    stride_x_w: tl.constexpr = Cb
    stride_x_h: tl.constexpr = W_in * Cb
    stride_x_cblock: tl.constexpr = H * W_in * Cb
    stride_x_n: tl.constexpr = (C_pad // Cb) * H * W_in * Cb

    tile_idx = tl.program_id(0)
    c_block = tl.program_id(1)

    n = tile_idx // (tile_H * tile_W)
    rem = tile_idx % (tile_H * tile_W)
    th = rem // tile_W
    tw = rem % tile_W

    h_start = th * 4 - pad_h
    w_start = tw * 4 - pad_w

    offs_c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    cblock_idx = offs_c // Cb
    c_local = offs_c % Cb
    base = X + n * stride_x_n + cblock_idx * stride_x_cblock + c_local

    _winograd_f4x3_input_transform(
        X,
        V,
        base,
        tile_idx,
        offs_c,
        c_mask,
        n < N,
        h_start,
        w_start,
        H,
        W_in,
        C_pad,
        T,
        stride_x_h,
        stride_x_w,
        BLOCK_C,
    )


@triton.jit(repr=_winograd_f4x3_batched_gemm_kernel_repr)
def _winograd_f4x3_batched_gemm_kernel(
    V,
    U,
    M_out,
    T: tl.constexpr,
    K_out: tl.constexpr,
    C_pad: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Batched GEMM: M[alpha] = V[alpha] @ U[alpha]^T."""
    # V: [36, T, C_pad]; U: [36, K_out, C_pad]; M: [36, T, K_out]
    stride_v_c: tl.constexpr = 1
    stride_v_tile: tl.constexpr = C_pad
    stride_v_alpha: tl.constexpr = T * C_pad
    stride_u_c: tl.constexpr = 1
    stride_u_k: tl.constexpr = C_pad
    stride_u_alpha: tl.constexpr = K_out * C_pad
    stride_m_k: tl.constexpr = 1
    stride_m_tile: tl.constexpr = K_out
    stride_m_alpha: tl.constexpr = T * K_out

    pid = tl.program_id(0)
    alpha = tl.program_id(1)

    num_pid_m = tl.cdiv(T, BLOCK_M)
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

    v_base = V + alpha * stride_v_alpha
    u_base = U + alpha * stride_u_alpha
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, C_pad, BLOCK_K):
        k_offs = k0 + offs_k

        v_ptrs = v_base + offs_m[:, None] * stride_v_tile + k_offs[None, :] * stride_v_c
        v_mask = (offs_m[:, None] < T) & (k_offs[None, :] < C_pad)
        v_tile = tl.load(v_ptrs, mask=v_mask, other=0.0)

        u_ptrs = u_base + offs_n[:, None] * stride_u_k + k_offs[None, :] * stride_u_c
        u_mask = (offs_n[:, None] < K_out) & (k_offs[None, :] < C_pad)
        u_tile = tl.load(u_ptrs, mask=u_mask, other=0.0)

        acc = tl.dot(v_tile, tl.trans(u_tile), acc=acc)

    m_ptrs = (
        M_out
        + alpha * stride_m_alpha
        + offs_m[:, None] * stride_m_tile
        + offs_n[None, :] * stride_m_k
    )
    m_mask = (offs_m[:, None] < T) & (offs_n[None, :] < K_out)
    tl.store(m_ptrs, acc, mask=m_mask)


@triton.jit(repr=_winograd_f4x3_output_transform_kernel_repr)
def _winograd_f4x3_output_transform_kernel(
    M_in,
    BIAS,
    Y,
    N,
    K_out: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    tile_H: tl.constexpr,
    tile_W: tl.constexpr,
    T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    LAYOUT: tl.constexpr = "nchw",
):
    stride_m_tile: tl.constexpr = K_out
    stride_m_alpha: tl.constexpr = T * K_out
    if LAYOUT == "nchw":
        stride_y_q: tl.constexpr = 1
        stride_y_p: tl.constexpr = Q
        stride_y_k: tl.constexpr = P * Q
        stride_y_n: tl.constexpr = K_out * P * Q
    else:
        stride_y_q: tl.constexpr = K_out
        stride_y_p: tl.constexpr = Q * K_out
        stride_y_k: tl.constexpr = 1
        stride_y_n: tl.constexpr = P * Q * K_out

    tile_idx = tl.program_id(0)
    k_block = tl.program_id(1)

    n = tile_idx // (tile_H * tile_W)
    rem = tile_idx % (tile_H * tile_W)
    th = rem // tile_W
    tw = rem % tile_W

    p_start = th * 4
    q_start = tw * 4

    offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = offs_k < K_out
    m_base = M_in + tile_idx * stride_m_tile + offs_k
    y_base = Y + n * stride_y_n + offs_k * stride_y_k

    _winograd_f4x3_output_transform(
        BIAS,
        m_base,
        y_base,
        offs_k,
        k_mask,
        n < N,
        p_start,
        q_start,
        P,
        Q,
        stride_m_alpha,
        stride_y_p,
        stride_y_q,
        HAS_BIAS,
        ACTIVATION,
    )
