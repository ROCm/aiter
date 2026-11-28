# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_conv2d_depthwise_repr = make_kernel_repr(
    "_conv2d_depthwise_kernel",
    [
        "R",
        "S",
        "BLOCK_H",
        "BLOCK_W",
        "HAS_BIAS",
        "USE_ACTIVATION",
        "USE_GLU",
        "num_warps",
    ],
)


@triton.jit
def _depthwise_plane(
    x_plane,
    w_ptr,
    bias_ptr,
    c,
    oh,
    ow,
    H,
    W,
    stride_xh,
    sh,
    sw,
    ph,
    pw,
    dh,
    dw,
    R: tl.constexpr,
    S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """fp32 depthwise conv of one channel plane at the output tile (oh, ow)."""
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    for r in tl.static_range(R):
        ih = oh * sh - ph + r * dh
        mask_h = (ih >= 0) & (ih < H)
        for s in tl.static_range(S):
            iw = ow * sw - pw + s * dw
            mask = mask_h & (iw >= 0) & (iw < W)
            x = tl.load(x_plane + ih * stride_xh + iw, mask=mask, other=0.0)
            wv = tl.load(w_ptr + c * (R * S) + r * S + s).to(tl.float32)
            acc += x.to(tl.float32) * wv
    if HAS_BIAS:
        acc += tl.load(bias_ptr + c).to(tl.float32)
    return acc


@triton.jit(repr=_conv2d_depthwise_repr)
def _conv2d_depthwise_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    C_out,
    H,
    W,
    OH,
    OW,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_yn,
    stride_yc,
    stride_yh,
    sh,
    sw,
    ph,
    pw,
    dh,
    dw,
    activation: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_ACTIVATION: tl.constexpr,
    USE_GLU: tl.constexpr,
):
    """One BLOCK_H x BLOCK_W output tile of one (n, c) plane. With USE_GLU, the
    output channel c is conv[c] * act(conv[c + C_out])."""
    pid_tile = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.program_id(2).to(tl.int64)
    tiles_w = tl.cdiv(OW, BLOCK_W)
    oh = ((pid_tile // tiles_w) * BLOCK_H + tl.arange(0, BLOCK_H))[:, None]
    ow = ((pid_tile % tiles_w) * BLOCK_W + tl.arange(0, BLOCK_W))[None, :]
    mask_out = (oh < OH) & (ow < OW)
    out_dtype = y_ptr.dtype.element_ty

    x_n = x_ptr + n * stride_xn
    y = _depthwise_plane(
        x_n + c.to(tl.int64) * stride_xc,
        w_ptr,
        bias_ptr,
        c,
        oh,
        ow,
        H,
        W,
        stride_xh,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        R,
        S,
        BLOCK_H,
        BLOCK_W,
        HAS_BIAS,
    )
    # Eager rounds the conv output to the activation dtype first.
    y = y.to(out_dtype).to(tl.float32)
    if USE_GLU:
        g = _depthwise_plane(
            x_n + (c + C_out).to(tl.int64) * stride_xc,
            w_ptr,
            bias_ptr,
            c + C_out,
            oh,
            ow,
            H,
            W,
            stride_xh,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw,
            R,
            S,
            BLOCK_H,
            BLOCK_W,
            HAS_BIAS,
        )
        g = activation(g.to(out_dtype).to(tl.float32)).to(out_dtype).to(tl.float32)
        y = y * g
    elif USE_ACTIVATION:
        y = activation(y)
    tl.store(
        y_ptr + n * stride_yn + c.to(tl.int64) * stride_yc + oh * stride_yh + ow,
        y.to(out_dtype),
        mask=mask_out,
    )
