# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.conv.conv_1x1 import (
    _conv2d_1x1_kernel,
)
from aiter.ops.triton._triton_kernels.conv.conv_1x1 import (
    _get_config as _get_config_1x1,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3 import (
    _conv2d_3x3_cblocked_kernel,
    _conv2d_3x3_nchw_kernel,
    _conv2d_3x3_nhwc_kernel,
    _get_config_cblocked,
    _get_config_nchw,
    _get_config_nhwc,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3_winograd_f4x3 import (
    _get_config_gemm as _get_config_wino_gemm,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3_winograd_f4x3 import (
    _get_config_input as _get_config_wino_input,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3_winograd_f4x3 import (
    _get_config_output as _get_config_wino_output,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3_winograd_f4x3 import (
    _winograd_f4x3_batched_gemm_kernel,
    _winograd_f4x3_cblocked_input_transform_kernel,
    _winograd_f4x3_input_transform_kernel,
    _winograd_f4x3_output_transform_kernel,
)
from aiter.ops.triton._triton_kernels.conv.conv_general import (
    _conv2d_general_kernel,
)
from aiter.ops.triton._triton_kernels.conv.conv_general import (
    _get_config as _get_config_general,
)
from aiter.ops.triton._triton_kernels.conv.nchw_to_cblocked import (
    _get_config as _get_config_prepack,
)
from aiter.ops.triton._triton_kernels.conv.nchw_to_cblocked import (
    _nchw_to_cblocked_kernel,
)
from aiter.ops.triton.utils.conv_config_utils import (
    format_prepack_shape_key,
    format_shape_key,
)


def _dtype_variant(dtype):
    return "fp16" if dtype == torch.float16 else "bf16"


def _make_mn_grid(M_total, K_out):
    """Grid for the GEMM-style conv kernels (1x1, 3x3 nhwc/cblocked, general):
    one program per (BLOCK_M tile of M_total) x (BLOCK_N tile of K_out)."""

    def grid(meta):
        return (
            triton.cdiv(M_total, meta["BLOCK_M"]) * triton.cdiv(K_out, meta["BLOCK_N"]),
        )

    return grid


def _make_wino_input_grid(T, C_pad):
    """Grid for the Winograd F(4,3) input-transform kernels: one program per
    tile T x (BLOCK_C tile of C_pad)."""

    def grid(meta):
        return (T, triton.cdiv(C_pad, meta["BLOCK_C"]))

    return grid


def _make_wino_gemm_grid(T, K_out):
    """Grid for the Winograd F(4,3) batched GEMM: (BLOCK_M tile of T) x
    (BLOCK_N tile of K_out) program blocks, batched over the 36 tile elements."""

    def grid(meta):
        return (
            triton.cdiv(T, meta["BLOCK_M"]) * triton.cdiv(K_out, meta["BLOCK_N"]),
            36,
        )

    return grid


def _make_wino_output_grid(T, K_out):
    """Grid for the Winograd F(4,3) output-transform kernels: one program per
    tile T x (BLOCK_K tile of K_out)."""

    def grid(meta):
        return (T, triton.cdiv(K_out, meta["BLOCK_K"]))

    return grid


def _launch_nchw_to_cblocked(x, x_blocked, N, C, H, W, C_pad, block_c):
    """Launch the fused NCHW-to-NCHWc activation pack."""
    HW = H * W
    shape_key = format_prepack_shape_key(N, C, H, W, block_c)
    config = _get_config_prepack(shape_key=shape_key, M=HW)
    grid = lambda meta: (
        triton.cdiv(HW, meta["BLOCK_M"]),
        triton.cdiv(C_pad, meta["BLOCK_C"]),
        N,
    )
    _nchw_to_cblocked_kernel[grid](
        x,
        x_blocked,
        C,
        HW,
        C_PAD=C_pad,
        CB=block_c,
        **config,
    )


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
    sh, sw = stride
    ph, pw = padding

    w = w_oihw.squeeze(-1).squeeze(-1).contiguous()  # [K_out, C]

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
    dtype_variant = _dtype_variant(x.dtype)
    config = _get_config_1x1(
        shape_key=shape_key,
        M=M_total,
        variants=(f"{layout}_{dtype_variant}", layout, dtype_variant),
    )

    _conv2d_1x1_kernel[_make_mn_grid(M_total, K_out)](
        x,
        w,
        bias_fp32,
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
        HAS_BIAS=bias_fp32 is not None,
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
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

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
    config = _get_config_nhwc(
        shape_key=shape_key,
        M=M_total,
        variants=(_dtype_variant(x.dtype),),
    )

    _conv2d_3x3_nhwc_kernel[_make_mn_grid(M_total, K_out)](
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
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        M_total,
        HAS_BIAS=bias_fp32 is not None,
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
    """Launch the 3x3 kernel for a materialized 5-D NCHWc input."""
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

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
    config = _get_config_cblocked(
        shape_key=shape_key,
        M=M_total,
        variants=(_dtype_variant(x_blocked.dtype),),
    )

    _conv2d_3x3_cblocked_kernel[_make_mn_grid(M_total, K_out)](
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
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        M_total,
        HAS_BIAS=bias_fp32 is not None,
        ACTIVATION=activation,
        **config,
    )


def _launch_3x3_nchw(
    x,
    w_3x3,
    bias,
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
    """Launch the repack-free kernel on a contiguous NCHW activation."""
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
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
    config = _get_config_nchw(shape_key=shape_key, M=M_total)
    row_aligned = "BLOCK_M" in config and Q % config["BLOCK_M"] == 0

    _conv2d_3x3_nchw_kernel[_make_mn_grid(M_total, K_out)](
        x,
        w_3x3,
        bias,
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
        HAS_BIAS=bias is not None,
        ACTIVATION=activation,
        ROW_ALIGNED=row_aligned,
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
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

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
    dtype_variant = _dtype_variant(x.dtype)
    config = _get_config_general(
        shape_key=shape_key,
        M=M_total,
        variants=(f"{layout}_{dtype_variant}", layout, dtype_variant),
    )

    _conv2d_general_kernel[_make_mn_grid(M_total, K_out)](
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
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        M_total,
        HAS_BIAS=bias_fp32 is not None,
        ACTIVATION=activation,
        LAYOUT=layout,
        **config,
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
    _winograd_f4x3_input_transform_kernel[_make_wino_input_grid(T, C_pad)](
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
    _winograd_f4x3_batched_gemm_kernel[_make_wino_gemm_grid(T, K_out)](
        V,
        U,
        M,
        T,
        K_out,
        C_pad,
        **gemm_config,
    )

    # 3. Output transform
    _winograd_f4x3_output_transform_kernel[_make_wino_output_grid(T, K_out)](
        M,
        bias_fp32,
        y,
        N,
        K_out,
        P,
        Q,
        tile_H,
        tile_W,
        T,
        HAS_BIAS=bias_fp32 is not None,
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
    """Launch Winograd F(4x4,3x3) with a materialized NCHWc input."""
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
    _winograd_f4x3_cblocked_input_transform_kernel[_make_wino_input_grid(T, C_pad)](
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

    _winograd_f4x3_batched_gemm_kernel[_make_wino_gemm_grid(T, K_out)](
        V,
        U,
        M,
        T,
        K_out,
        C_pad,
        **gemm_config,
    )

    _winograd_f4x3_output_transform_kernel[_make_wino_output_grid(T, K_out)](
        M,
        bias_fp32,
        y,
        N,
        K_out,
        P,
        Q,
        tile_H,
        tile_W,
        T,
        HAS_BIAS=bias_fp32 is not None,
        ACTIVATION=activation,
        **output_config,
    )
