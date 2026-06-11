# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

from aiter.ops.triton.conv._utils import _out_hw, _is_winograd_eligible
from aiter.ops.triton.utils.conv_config_utils import format_shape_key
from aiter.ops.triton._triton_kernels.conv.conv_1x1 import (
    _conv2d_1x1_kernel,
    _get_config as _get_config_1x1,
)
from aiter.ops.triton._triton_kernels.conv.conv_general import (
    _conv2d_general_kernel,
    _get_config as _get_config_general,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3 import (
    _conv2d_3x3_nhwc_kernel,
    _conv2d_3x3_cblocked_kernel,
    _get_config_nhwc,
    _get_config_cblocked,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3_winograd_f4x3 import (
    _winograd_f4x3_input_transform_kernel,
    _winograd_f4x3_cblocked_input_transform_kernel,
    _winograd_f4x3_batched_gemm_kernel,
    _winograd_f4x3_output_transform_kernel,
    _winograd_f4x3_fused_gemm_output_kernel,
    _get_config_input as _get_config_wino_input,
    _get_config_gemm as _get_config_wino_gemm,
    _get_config_output as _get_config_wino_output,
    _get_config_fused as _get_config_wino_fused,
)


def _select_3x3_method(N, C, H, W, K_out, stride, dilation):
    """Pick the best 3x3 kernel method based on shape heuristics.

    Decision tree (from benchmark sweep on RDNA4):
    1. Non-Winograd-eligible (stride>1, dilation>1, or C<4) -> cblocked
    2. Winograd only wins when BOTH C and K >= 512 with enough tiles (T >= 98).
       At 256x256 channels, cblocked is tied or slightly better.
    3. Among Winograd variants: WF4cb (NCHWc input) beats WF4 (NCHW input)
       when T >= 392 (large batch * spatial gives more coalescing benefit).
       Below that, WF4 is slightly faster (less repacking overhead).
    """
    if not _is_winograd_eligible(3, 3, stride, dilation, C):
        return "cblocked"
    P, Q = _out_hw(H, W, 3, 3, stride, (1, 1), dilation)
    tile_H = (P + 3) // 4
    tile_W = (Q + 3) // 4
    T = N * tile_H * tile_W
    if C >= 512 and K_out >= 512 and T >= 98:
        if T >= 392:
            return "winograd_f4x3_cblocked"
        return "winograd_f4x3"
    return "cblocked"


def _launch_1x1(
    x,
    w_oihw,
    bias_fp32,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    stride,
    padding,
    activation,
    layout="nchw",
):
    """Launch specialized 1x1 kernel.
    layout: "nchw" or "nhwc" (case-insensitive).
    """
    if triton is None:
        raise RuntimeError("Triton not available")

    sh, sw = stride
    ph, pw = padding

    w = w_oihw.squeeze(-1).squeeze(-1).contiguous()  # [K_out, C]

    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(N * P * Q, BM) * triton.cdiv(K_out, BN),)

    bias_arg = bias_fp32 if bias_fp32 is not None else w.new_empty(1)

    M_total = N * P * Q

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=1,
        S=1,
        sh=sh,
        sw=sw,
        ph=ph,
        pw=pw,
        dh=1,
        dw=1,
    )
    config = _get_config_1x1(shape_key=shape_key, M=M_total)

    _conv2d_1x1_kernel[grid](
        x,
        w,
        bias_arg,
        y,
        N,
        C,
        H,
        W_in,
        K_out,
        P,
        Q,
        sh,
        sw,
        ph,
        pw,
        M_total,
        HAS_BIAS=1 if bias_fp32 is not None else 0,
        ACTIVATION=activation,
        LAYOUT=layout,
        **config,
    )


def _launch_3x3_nhwc(
    x,
    w_3x3,
    bias_fp32,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    C_pad,
    stride,
    padding,
    dilation,
    activation,
):
    """Launch specialized 3x3 NHWC kernel (hardcoded stride_c=1, stride_k=1)."""
    if triton is None:
        raise RuntimeError("Triton not available")

    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(N * P * Q, BM) * triton.cdiv(K_out, BN),)

    bias_arg = bias_fp32 if bias_fp32 is not None else w_3x3.new_empty(1)

    M_total = N * P * Q

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=3,
        S=3,
        sh=sh,
        sw=sw,
        ph=ph,
        pw=pw,
        dh=dh,
        dw=dw,
    )
    config = _get_config_nhwc(shape_key=shape_key, M=M_total)

    _conv2d_3x3_nhwc_kernel[grid](
        x,
        w_3x3,
        bias_arg,
        y,
        N,
        C,
        H,
        W_in,
        K_out,
        P,
        Q,
        C_pad,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        M_total,
        HAS_BIAS=1 if bias_fp32 is not None else 0,
        ACTIVATION=activation,
        **config,
    )


def _launch_3x3_cblocked(
    x_blocked,
    w_3x3,
    bias_fp32,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    C_pad,
    Cb,
    stride,
    padding,
    dilation,
    activation,
):
    """Launch specialized 3x3 kernel for channel-blocked input."""
    if triton is None:
        raise RuntimeError("Triton not available")

    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(N * P * Q, BM) * triton.cdiv(K_out, BN),)

    bias_arg = bias_fp32 if bias_fp32 is not None else w_3x3.new_empty(1)

    M_total = N * P * Q

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=3,
        S=3,
        sh=sh,
        sw=sw,
        ph=ph,
        pw=pw,
        dh=dh,
        dw=dw,
    )
    config = _get_config_cblocked(shape_key=shape_key, M=M_total)

    _conv2d_3x3_cblocked_kernel[grid](
        x_blocked,
        w_3x3,
        bias_arg,
        y,
        N,
        C,
        H,
        W_in,
        K_out,
        P,
        Q,
        C_pad,
        Cb,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        M_total,
        HAS_BIAS=1 if bias_fp32 is not None else 0,
        ACTIVATION=activation,
        **config,
    )


def _launch_general(
    x,
    w_k,
    bias_fp32,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    R,
    S,
    P,
    Q,
    K_pad,
    stride,
    padding,
    dilation,
    block_k,
    activation,
    layout="nchw",
):
    """Launch general conv kernel.
    layout: "nchw" or "nhwc" (case-insensitive).
    """
    if triton is None:
        raise RuntimeError("Triton not available")

    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(N * P * Q, BM) * triton.cdiv(K_out, BN),)

    bias_arg = bias_fp32 if bias_fp32 is not None else w_k.new_empty(1)

    M_total = N * P * Q

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=R,
        S=S,
        sh=sh,
        sw=sw,
        ph=ph,
        pw=pw,
        dh=dh,
        dw=dw,
    )
    config = _get_config_general(shape_key=shape_key, M=M_total)

    _conv2d_general_kernel[grid](
        x,
        w_k,
        bias_arg,
        y,
        N,
        C,
        H,
        W_in,
        K_out,
        R,
        S,
        P,
        Q,
        K_pad,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        M_total,
        HAS_BIAS=1 if bias_fp32 is not None else 0,
        ACTIVATION=activation,
        LAYOUT=layout,
        **config,
    )


def _launch_winograd_f4x3_fused(
    x,
    U,
    bias_fp32,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    C_pad,
    padding,
    activation,
    layout="nchw",
):
    """Launch Winograd F(4x4,3x3) with fused GEMM+output transform (2 kernels instead of 3)."""
    if triton is None:
        raise RuntimeError("Triton not available")
    ph, pw = padding
    tile_H = (P + 3) // 4
    tile_W = (Q + 3) // 4
    T = N * tile_H * tile_W

    input_dtype = x.dtype
    V = torch.empty((36, T, C_pad), device=x.device, dtype=input_dtype)

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=3,
        S=3,
        sh=1,
        sw=1,
        ph=ph,
        pw=pw,
        dh=1,
        dw=1,
    )
    input_config = _get_config_wino_input(shape_key=shape_key, M=T)
    fused_config = _get_config_wino_fused(shape_key=shape_key, M=T)

    # 1. Input transform
    def input_grid_f4(meta):
        return (T, triton.cdiv(C_pad, meta["BLOCK_C"]))

    _winograd_f4x3_input_transform_kernel[input_grid_f4](
        x,
        V,
        N,
        C,
        C_pad,
        H,
        W_in,
        tile_H,
        tile_W,
        T,
        ph,
        pw,
        LAYOUT=layout,
        **input_config,
    )

    # 2. Fused GEMM + output transform
    bias_arg = bias_fp32 if bias_fp32 is not None else x.new_empty(1)

    def fused_grid_f4(meta):
        return (triton.cdiv(T, meta["BLOCK_T"]), triton.cdiv(K_out, meta["BLOCK_K"]))

    _winograd_f4x3_fused_gemm_output_kernel[fused_grid_f4](
        V,
        U,
        bias_arg,
        y,
        N,
        K_out,
        P,
        Q,
        C_pad,
        tile_H,
        tile_W,
        T,
        HAS_BIAS=1 if bias_fp32 is not None else 0,
        ACTIVATION=activation,
        LAYOUT=layout,
        **fused_config,
    )


def _launch_winograd_f4x3(
    x,
    U,
    bias_fp32,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    C_pad,
    padding,
    activation,
    layout="nchw",
):
    """Launch Winograd F(4x4,3x3) pipeline: input transform -> batched GEMM -> output transform."""
    if triton is None:
        raise RuntimeError("Triton not available")
    ph, pw = padding
    tile_H = (P + 3) // 4
    tile_W = (Q + 3) // 4
    T = N * tile_H * tile_W

    input_dtype = x.dtype
    V = torch.empty((36, T, C_pad), device=x.device, dtype=input_dtype)
    M = torch.empty((36, T, K_out), device=x.device, dtype=torch.float32)

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=3,
        S=3,
        sh=1,
        sw=1,
        ph=ph,
        pw=pw,
        dh=1,
        dw=1,
    )
    input_config = _get_config_wino_input(shape_key=shape_key, M=T)
    gemm_config = _get_config_wino_gemm(shape_key=shape_key, M=T)
    output_config = _get_config_wino_output(shape_key=shape_key, M=T)

    # 1. Input transform
    def input_grid_f4(meta):
        return (T, triton.cdiv(C_pad, meta["BLOCK_C"]))

    _winograd_f4x3_input_transform_kernel[input_grid_f4](
        x,
        V,
        N,
        C,
        C_pad,
        H,
        W_in,
        tile_H,
        tile_W,
        T,
        ph,
        pw,
        LAYOUT=layout,
        **input_config,
    )

    # 2. Batched GEMM
    def gemm_grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(T, BM) * triton.cdiv(K_out, BN), 36)

    _winograd_f4x3_batched_gemm_kernel[gemm_grid](
        V,
        U,
        M,
        T,
        K_out,
        C_pad,
        **gemm_config,
    )

    # 3. Output transform
    bias_arg = bias_fp32 if bias_fp32 is not None else x.new_empty(1)

    def output_grid_f4(meta):
        return (T, triton.cdiv(K_out, meta["BLOCK_K"]))

    _winograd_f4x3_output_transform_kernel[output_grid_f4](
        M,
        bias_arg,
        y,
        N,
        K_out,
        P,
        Q,
        tile_H,
        tile_W,
        T,
        HAS_BIAS=1 if bias_fp32 is not None else 0,
        ACTIVATION=activation,
        LAYOUT=layout,
        **output_config,
    )


def _launch_winograd_f4x3_cblocked(
    x_blocked,
    C_pad_blocked,
    U,
    bias_fp32,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    C_pad,
    padding,
    activation,
    block_k,
):
    """Launch Winograd F(4x4,3x3) with NCHWc input layout: cblocked input transform -> batched GEMM -> output transform."""
    if triton is None:
        raise RuntimeError("Triton not available")
    ph, pw = padding
    tile_H = (P + 3) // 4
    tile_W = (Q + 3) // 4
    T = N * tile_H * tile_W

    Cb = block_k
    input_dtype = x_blocked.dtype
    V = torch.empty((36, T, C_pad), device=x_blocked.device, dtype=input_dtype)
    M = torch.empty((36, T, K_out), device=x_blocked.device, dtype=torch.float32)

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=3,
        S=3,
        sh=1,
        sw=1,
        ph=ph,
        pw=pw,
        dh=1,
        dw=1,
    )
    input_config = _get_config_wino_input(shape_key=shape_key, M=T)
    gemm_config = _get_config_wino_gemm(shape_key=shape_key, M=T)
    output_config = _get_config_wino_output(shape_key=shape_key, M=T)

    # 1. Cblocked input transform
    def input_grid_f4(meta):
        return (T, triton.cdiv(C_pad, meta["BLOCK_C"]))

    _winograd_f4x3_cblocked_input_transform_kernel[input_grid_f4](
        x_blocked,
        V,
        N,
        C,
        C_pad,
        H,
        W_in,
        tile_H,
        tile_W,
        T,
        ph,
        pw,
        Cb,
        **input_config,
    )

    def gemm_grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(T, BM) * triton.cdiv(K_out, BN), 36)

    _winograd_f4x3_batched_gemm_kernel[gemm_grid](
        V,
        U,
        M,
        T,
        K_out,
        C_pad,
        **gemm_config,
    )

    bias_arg = bias_fp32 if bias_fp32 is not None else x_blocked.new_empty(1)

    def output_grid_f4(meta):
        return (T, triton.cdiv(K_out, meta["BLOCK_K"]))

    _winograd_f4x3_output_transform_kernel[output_grid_f4](
        M,
        bias_arg,
        y,
        N,
        K_out,
        P,
        Q,
        tile_H,
        tile_W,
        T,
        HAS_BIAS=1 if bias_fp32 is not None else 0,
        ACTIVATION=activation,
        **output_config,
    )
