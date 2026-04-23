# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

from aiter.ops.triton.conv._utils import _out_hw, _is_winograd_eligible
from aiter.ops.triton._triton_kernels.conv.conv_1x1 import _conv2d_1x1_kernel
from aiter.ops.triton._triton_kernels.conv.conv_general import _conv2d_general_kernel
from aiter.ops.triton._triton_kernels.conv.conv_3x3 import (
    _conv2d_3x3_nhwc_kernel,
    _conv2d_3x3_cblocked_kernel,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3_winograd_f4x3 import (
    _winograd_f4x3_input_transform_kernel,
    _winograd_f4x3_cblocked_input_transform_kernel,
    _winograd_f4x3_batched_gemm_kernel,
    _winograd_f4x3_output_transform_kernel,
    _winograd_f4x3_fused_gemm_output_kernel,
)


def _torch_dtype_to_tl(dtype):
    """Map torch dtype to triton dtype for constexpr params."""
    if dtype == torch.float16:
        return tl.float16
    elif dtype == torch.bfloat16:
        return tl.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


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


def _layout_to_int(layout):
    """Convert layout string to kernel int: 0=NCHW, 1=NHWC."""
    layout = layout.lower()
    if layout not in ("nchw", "nhwc"):
        raise ValueError(f"layout must be 'nchw' or 'nhwc', got '{layout}'")
    return 0 if layout == "nchw" else 1


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
    out_dtype,
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
    layout = _layout_to_int(layout)

    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(N * P * Q, BM) * triton.cdiv(K_out, BN),)

    ACT_MAP = {"none": 0, "relu": 1, "relu6": 2, "gelu": 3}
    bias_arg = bias_fp32 if bias_fp32 is not None else w.new_empty(1)

    M_total = N * P * Q

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
        ACT_TYPE=ACT_MAP.get(activation, 0),
        LAYOUT=layout,
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
    out_dtype,
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

    ACT_MAP = {"none": 0, "relu": 1, "relu6": 2, "gelu": 3}
    bias_arg = bias_fp32 if bias_fp32 is not None else w_3x3.new_empty(1)

    M_total = N * P * Q

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
        ACT_TYPE=ACT_MAP.get(activation, 0),
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
    out_dtype,
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

    ACT_MAP = {"none": 0, "relu": 1, "relu6": 2, "gelu": 3}
    bias_arg = bias_fp32 if bias_fp32 is not None else w_3x3.new_empty(1)

    M_total = N * P * Q

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
        ACT_TYPE=ACT_MAP.get(activation, 0),
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
    out_dtype,
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
    layout = _layout_to_int(layout)

    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(N * P * Q, BM) * triton.cdiv(K_out, BN),)

    ACT_MAP = {"none": 0, "relu": 1, "relu6": 2, "gelu": 3}
    bias_arg = bias_fp32 if bias_fp32 is not None else w_k.new_empty(1)

    M_total = N * P * Q

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
        ACT_TYPE=ACT_MAP.get(activation, 0),
        LAYOUT=layout,
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
    out_dtype,
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
    layout_int = _layout_to_int(layout)

    input_dtype = x.dtype
    V = torch.empty((36, T, C_pad), device=x.device, dtype=input_dtype)

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
        INPUT_DTYPE=_torch_dtype_to_tl(input_dtype),
        LAYOUT=layout_int,
    )

    # 2. Fused GEMM + output transform
    ACT_MAP = {"none": 0, "relu": 1, "relu6": 2, "gelu": 3}
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
        ACT_TYPE=ACT_MAP.get(activation, 0),
        LAYOUT=layout_int,
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
    out_dtype,
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
    layout_int = _layout_to_int(layout)

    input_dtype = x.dtype
    V = torch.empty((36, T, C_pad), device=x.device, dtype=input_dtype)
    M = torch.empty((36, T, K_out), device=x.device, dtype=torch.float32)

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
        INPUT_DTYPE=_torch_dtype_to_tl(input_dtype),
        LAYOUT=layout_int,
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
    )

    # 3. Output transform
    ACT_MAP = {"none": 0, "relu": 1, "relu6": 2, "gelu": 3}
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
        ACT_TYPE=ACT_MAP.get(activation, 0),
        LAYOUT=layout_int,
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
    out_dtype,
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
        INPUT_DTYPE=_torch_dtype_to_tl(input_dtype),
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
    )

    ACT_MAP = {"none": 0, "relu": 1, "relu6": 2, "gelu": 3}
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
        ACT_TYPE=ACT_MAP.get(activation, 0),
    )
