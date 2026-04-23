# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
from typing import Optional
import torch

from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.conv._utils import (
    BLOCK_K,
    _out_hw,
    _is_1x1_conv,
    _is_3x3_conv,
    _is_winograd_eligible,
)
from aiter.ops.triton.conv._prepack import (
    get_or_make_weight_pack,
    get_or_make_weight_pack_3x3,
    get_or_make_input_pack_cblocked,
    get_or_make_winograd_filter_f4x3,
)
from aiter.ops.triton.conv._launch import (
    _launch_1x1,
    _launch_3x3_nhwc,
    _launch_3x3_cblocked,
    _launch_general,
    _launch_winograd_f4x3,
    _launch_winograd_f4x3_cblocked,
    _launch_winograd_f4x3_fused,
    _select_3x3_method,
)

_LOGGER = AiterTritonLogger()

# Tracks the last Triton kernel selected by conv2d_nchw smart routing.
_last_triton_kernel: Optional[str] = None


def conv2d(
    x,
    w_oihw,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    activation="none",
    out_dtype=torch.float16,
    layout="nchw",
):
    """Forward 2-D conv on AMD ROCm via Triton. Drop-in for the forward of
    ``torch.nn.functional.conv2d`` (no backward).

    A shape-driven router picks among five kernel families (1x1, 3x3 cblocked,
    3x3 NHWC, Winograd F(4x4,3x3), general) per call.

    Inputs must be fp16 or bf16. ``layout="nhwc"`` runs an NHWC-native kernel
    with no internal layout conversion.

    Notes
    -----
    - Only ``groups=1`` (depthwise/grouped raises ``AssertionError``).
    - Only ``padding_mode="zeros"`` (no reflect/replicate/circular).
    - ``bias=None`` skips the with-bias kernel path; passing a zero tensor
      instead routes through the with-bias kernel and times differently.
    """
    _LOGGER.info(
        f"CONV2D: x={tuple(x.shape)} w={tuple(w_oihw.shape)} stride={stride} "
        f"padding={padding} dilation={dilation} layout={layout} "
        f"dtype={x.dtype} out_dtype={out_dtype} bias={'yes' if bias is not None else 'no'} "
        f"act={activation}"
    )
    layout = layout.lower()
    if layout not in ("nchw", "nhwc"):
        raise ValueError(f"layout must be 'nchw' or 'nhwc', got '{layout}'")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"conv2d only supports fp16 and bf16 inputs, got {x.dtype}")

    if layout == "nhwc":
        return conv2d_nhwc(
            x, w_oihw, bias, stride, padding, dilation, activation, out_dtype
        )
    else:
        return conv2d_nchw(
            x, w_oihw, bias, stride, padding, dilation, activation, out_dtype
        )


def conv2d_winograd_f4x3(
    x,
    w_oihw,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    activation="none",
    out_dtype=torch.float16,
    block_k=BLOCK_K,
    layout="nchw",
):
    """NCHW/NHWC conv2d using Winograd F(4x4,3x3). Raises ValueError for non-eligible convs."""
    assert x.is_cuda and w_oihw.is_cuda
    N, C, H, W_in = x.shape
    K_out, Cw, R, S = w_oihw.shape
    assert Cw == C
    P, Q = _out_hw(H, W_in, R, S, stride, padding, dilation)

    if not _is_winograd_eligible(R, S, stride, dilation, C):
        raise ValueError(
            f"conv2d_winograd_f4x3 requires 3x3 kernel with stride=1, dilation=1, "
            f"and C >= 4 (F(4,3) output transform amplifies rounding by up to "
            f"361x; C<4 has too few reduction terms to absorb it), "
            f"got {R}x{S} stride={stride} dilation={dilation} C={C}"
        )

    if layout == "nhwc":
        y = torch.empty((N, K_out, P, Q), device=x.device, dtype=out_dtype).to(
            memory_format=torch.channels_last
        )
    else:
        y = torch.empty((N, K_out, P, Q), device=x.device, dtype=out_dtype)
    bias_fp32 = bias.float().contiguous() if bias is not None else None
    U, (_, C_pad) = get_or_make_winograd_filter_f4x3(w_oihw.contiguous(), block_k)
    _launch_winograd_f4x3(
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
        layout=layout,
    )
    return y


def conv2d_winograd_f4x3_cblocked(
    x,
    w_oihw,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    activation="none",
    out_dtype=torch.float16,
    block_k=BLOCK_K,
):
    """NCHW conv2d using Winograd F(4x4,3x3) with NCHWc input layout for coalesced loads.
    Raises ValueError for non-eligible convs."""
    assert x.is_cuda and w_oihw.is_cuda
    N, C, H, W_in = x.shape
    K_out, Cw, R, S = w_oihw.shape
    assert Cw == C
    P, Q = _out_hw(H, W_in, R, S, stride, padding, dilation)

    if not _is_winograd_eligible(R, S, stride, dilation, C):
        raise ValueError(
            f"conv2d_winograd_f4x3_cblocked requires 3x3 kernel with stride=1, dilation=1, "
            f"and C >= 4 (F(4,3) output transform amplifies rounding by up to "
            f"361x; C<4 has too few reduction terms to absorb it), "
            f"got {R}x{S} stride={stride} dilation={dilation} C={C}"
        )

    y = torch.empty((N, K_out, P, Q), device=x.device, dtype=out_dtype)
    bias_fp32 = bias.float().contiguous() if bias is not None else None
    U, (_, C_pad) = get_or_make_winograd_filter_f4x3(w_oihw.contiguous(), block_k)
    x_blocked, C_pad_blocked = get_or_make_input_pack_cblocked(x, block_k)
    _launch_winograd_f4x3_cblocked(
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
    )
    return y


def conv2d_winograd_f4x3_fused(
    x,
    w_oihw,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    activation="none",
    out_dtype=torch.float16,
    block_k=BLOCK_K,
):
    """NCHW conv2d using Winograd F(4x4,3x3) with fused GEMM+output transform.
    Raises ValueError for non-eligible convs."""
    assert x.is_cuda and w_oihw.is_cuda
    N, C, H, W_in = x.shape
    K_out, Cw, R, S = w_oihw.shape
    assert Cw == C
    P, Q = _out_hw(H, W_in, R, S, stride, padding, dilation)

    if not _is_winograd_eligible(R, S, stride, dilation, C):
        raise ValueError(
            f"conv2d_winograd_f4x3_fused requires 3x3 kernel with stride=1, dilation=1, "
            f"and C >= 4 (F(4,3) output transform amplifies rounding by up to "
            f"361x; C<4 has too few reduction terms to absorb it), "
            f"got {R}x{S} stride={stride} dilation={dilation} C={C}"
        )

    y = torch.empty((N, K_out, P, Q), device=x.device, dtype=out_dtype)
    bias_fp32 = bias.float().contiguous() if bias is not None else None
    U, (_, C_pad) = get_or_make_winograd_filter_f4x3(w_oihw.contiguous(), block_k)
    _launch_winograd_f4x3_fused(
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
    )
    return y


def conv2d_1x1(
    x,
    w_oihw,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    activation="none",
    out_dtype=torch.float16,
    block_k=BLOCK_K,
    layout="nchw",
):
    """NCHW/NHWC conv2d for 1x1 kernels. Raises ValueError for non-1x1."""
    assert x.is_cuda and w_oihw.is_cuda
    N, C, H, W_in = x.shape
    K_out, Cw, R, S = w_oihw.shape
    assert Cw == C
    if not _is_1x1_conv(R, S, dilation):
        raise ValueError(f"conv2d_1x1 requires 1x1 kernel, got {R}x{S}")
    P, Q = _out_hw(H, W_in, R, S, stride, padding, dilation)

    if layout == "nhwc":
        y = torch.empty((N, K_out, P, Q), device=x.device, dtype=out_dtype).to(
            memory_format=torch.channels_last
        )
    else:
        y = torch.empty((N, K_out, P, Q), device=x.device, dtype=out_dtype)
    bias_fp32 = bias.float().contiguous() if bias is not None else None
    _launch_1x1(
        x,
        w_oihw.contiguous(),
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
        layout=layout,
    )
    return y


def conv2d_general(
    x,
    w_oihw,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    activation="none",
    out_dtype=torch.float16,
    block_k=BLOCK_K,
    layout="nchw",
):
    """NCHW/NHWC conv2d using general kernel with prepacked weights (5x5, 7x7, etc.)."""
    assert x.is_cuda and w_oihw.is_cuda
    N, C, H, W_in = x.shape
    K_out, Cw, R, S = w_oihw.shape
    assert Cw == C
    P, Q = _out_hw(H, W_in, R, S, stride, padding, dilation)

    if layout == "nhwc":
        y = torch.empty((N, K_out, P, Q), device=x.device, dtype=out_dtype).to(
            memory_format=torch.channels_last
        )
    else:
        y = torch.empty((N, K_out, P, Q), device=x.device, dtype=out_dtype)
    bias_fp32 = bias.float().contiguous() if bias is not None else None
    w_k, (_, K_pad) = get_or_make_weight_pack(w_oihw.contiguous(), block_k)
    _launch_general(
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
        layout=layout,
    )
    return y


def conv2d_nhwc_3x3(
    x,
    w_oihw,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    activation="none",
    out_dtype=torch.float16,
    block_k=BLOCK_K,
):
    """NHWC conv2d for 3x3 kernels. Raises ValueError for non-3x3."""
    assert x.is_cuda and w_oihw.is_cuda
    N, C, H, W_in = x.shape
    K_out, Cw, R, S = w_oihw.shape
    assert Cw == C
    if not _is_3x3_conv(R, S):
        raise ValueError(f"conv2d_nhwc_3x3 requires 3x3 kernel, got {R}x{S}")
    P, Q = _out_hw(H, W_in, R, S, stride, padding, dilation)

    y = torch.empty((N, K_out, P, Q), device=x.device, dtype=out_dtype).to(
        memory_format=torch.channels_last
    )
    bias_fp32 = bias.float().contiguous() if bias is not None else None
    w_3x3, (_, C_pad) = get_or_make_weight_pack_3x3(w_oihw.contiguous(), block_k)
    _launch_3x3_nhwc(
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
    )
    return y


def conv2d_nchw(
    x,
    w_oihw,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    activation="none",
    out_dtype=torch.float16,
    block_k=BLOCK_K,
):
    """Hybrid NCHW conv2d: routes to specialized 1x1, 3x3, or general kernel."""
    assert x.is_cuda and w_oihw.is_cuda
    N, C, H, W_in = x.shape
    K_out, Cw, R, S = w_oihw.shape
    assert Cw == C

    global _last_triton_kernel
    if _is_1x1_conv(R, S, dilation):
        _last_triton_kernel = "_conv2d_1x1_kernel"
        return conv2d_1x1(
            x,
            w_oihw,
            bias,
            stride,
            padding,
            dilation,
            activation,
            out_dtype,
            block_k,
            layout="nchw",
        )
    elif _is_3x3_conv(R, S):
        method = _select_3x3_method(N, C, H, W_in, K_out, stride, dilation)
        if method == "winograd_f4x3_cblocked":
            _last_triton_kernel = "_winograd_f4x3_cblocked_* (3 kern)"
            return conv2d_winograd_f4x3_cblocked(
                x,
                w_oihw,
                bias,
                stride,
                padding,
                dilation,
                activation,
                out_dtype,
                block_k,
            )
        elif method == "winograd_f4x3":
            _last_triton_kernel = "_winograd_f4x3_* (3 kernels)"
            return conv2d_winograd_f4x3(
                x,
                w_oihw,
                bias,
                stride,
                padding,
                dilation,
                activation,
                out_dtype,
                block_k,
            )
        else:
            _last_triton_kernel = "_conv2d_3x3_cblocked_kernel"
            return conv2d_nchw_cblocked(
                x,
                w_oihw,
                bias,
                stride,
                padding,
                dilation,
                activation,
                out_dtype,
                block_k,
            )
    else:
        _last_triton_kernel = "_conv2d_general_kernel"
        return conv2d_general(
            x,
            w_oihw,
            bias,
            stride,
            padding,
            dilation,
            activation,
            out_dtype,
            block_k,
            layout="nchw",
        )


def conv2d_nhwc(
    x,
    w_oihw,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    activation="none",
    out_dtype=torch.float16,
    block_k=BLOCK_K,
):
    """Conv2d with NHWC (channels-last) input and output.

    Input x can be NCHW or NHWC — it will be converted to channels_last.
    Output y is allocated as channels_last (NHWC-contiguous) and returned
    in logical NCHW shape with channels_last strides.
    """
    assert x.is_cuda and w_oihw.is_cuda
    x = x.to(memory_format=torch.channels_last)
    N, C, H, W_in = x.shape
    K_out, Cw, R, S = w_oihw.shape
    assert Cw == C

    global _last_triton_kernel
    if _is_1x1_conv(R, S, dilation):
        _last_triton_kernel = "_conv2d_1x1_kernel"
        return conv2d_1x1(
            x,
            w_oihw,
            bias,
            stride,
            padding,
            dilation,
            activation,
            out_dtype,
            block_k,
            layout="nhwc",
        )
    elif _is_3x3_conv(R, S):
        method = _select_3x3_method(N, C, H, W_in, K_out, stride, dilation)
        if method in ("winograd_f4x3", "winograd_f4x3_cblocked"):
            _last_triton_kernel = "_winograd_f4x3_* (3 kernels)"
            return conv2d_winograd_f4x3(
                x,
                w_oihw,
                bias,
                stride,
                padding,
                dilation,
                activation,
                out_dtype,
                block_k,
                layout="nhwc",
            )
        else:
            _last_triton_kernel = "_conv2d_3x3_nhwc_kernel"
            return conv2d_nhwc_3x3(
                x,
                w_oihw,
                bias,
                stride,
                padding,
                dilation,
                activation,
                out_dtype,
                block_k,
            )
    else:
        _last_triton_kernel = "_conv2d_general_kernel"
        return conv2d_general(
            x,
            w_oihw,
            bias,
            stride,
            padding,
            dilation,
            activation,
            out_dtype,
            block_k,
            layout="nhwc",
        )


def conv2d_nchw_cblocked(
    x,
    w_oihw,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    activation="none",
    out_dtype=torch.float16,
    block_k=BLOCK_K,
):
    """NCHW conv2d with channel-blocked input packing for 3x3 kernels.
    Raises ValueError for non-3x3."""
    assert x.is_cuda and w_oihw.is_cuda
    N, C, H, W_in = x.shape
    K_out, Cw, R, S = w_oihw.shape
    assert Cw == C
    P, Q = _out_hw(H, W_in, R, S, stride, padding, dilation)

    if not _is_3x3_conv(R, S):
        raise ValueError(f"conv2d_nchw_cblocked requires 3x3 kernel, got {R}x{S}")

    y = torch.empty((N, K_out, P, Q), device=x.device, dtype=out_dtype)
    bias_fp32 = bias.float().contiguous() if bias is not None else None
    w_3x3, (_, C_pad) = get_or_make_weight_pack_3x3(w_oihw.contiguous(), block_k)
    Cb = block_k  # packing block size matches weight padding block
    x_blocked, C_pad_x = get_or_make_input_pack_cblocked(x, Cb)
    # Ensure channel padding is consistent
    assert (
        C_pad_x == C_pad
    ), f"Channel padding mismatch: input {C_pad_x} vs weight {C_pad}"
    _launch_3x3_cblocked(
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
    )
    return y
