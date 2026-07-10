# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from enum import Enum

import torch

from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.conv._utils import (
    BLOCK_K,
    _is_1x1_conv,
    _is_3x3_conv,
    _conv_dims,
    _alloc_output,
    _prep_bias,
    _require_winograd_eligible,
)
from aiter.ops.triton.conv._prepack import (
    get_or_make_weight_pack,
    get_or_make_weight_pack_3x3,
    prepack_nchw_to_cblocked,
    get_or_make_winograd_filter_f4x3,
)
from aiter.ops.triton.conv._launch import (
    _launch_1x1,
    _launch_3x3_nhwc,
    _launch_3x3_cblocked,
    _launch_general,
    _launch_winograd_f4x3,
    _launch_winograd_f4x3_cblocked,
    _select_3x3_method,
)

_LOGGER = AiterTritonLogger()


class Route(Enum):
    # Values are kernel display names; the bench substring-matches "winograd"/
    # "cblocked" on them to pick tolerances and timing paths — keep those tokens.
    ONE_X_ONE = "_conv2d_1x1_kernel"
    WF4X3_CBLOCKED = "_winograd_f4x3_cblocked_* (3 kern)"
    WF4X3 = "_winograd_f4x3_* (3 kernels)"
    CBLOCKED_NCHW = "_conv2d_3x3_cblocked_kernel"
    NHWC_3X3 = "_conv2d_3x3_nhwc_kernel"
    GENERAL = "_conv2d_general_kernel"


def _resolve_route(R, S, stride, dilation, N, C, H, W_in, K_out, layout):
    if _is_1x1_conv(R, S, dilation):
        return Route.ONE_X_ONE
    if _is_3x3_conv(R, S):
        method = _select_3x3_method(N, C, H, W_in, K_out, stride, dilation)
        if layout == "nhwc":
            # _select_3x3_method tunes for NCHW; the cblocked vs. non-cblocked
            # distinction is deliberately ignored here because cblocked is an
            # NCHW-only layout. The only NHWC question is winograd-or-not: any
            # winograd pick maps to the (NCHW-input) winograd kernel, a plain
            # "cblocked" pick falls through to the NHWC 3x3 kernel.
            if method in ("winograd_f4x3", "winograd_f4x3_cblocked"):
                return Route.WF4X3
            return Route.NHWC_3X3
        if method == "winograd_f4x3_cblocked":
            return Route.WF4X3_CBLOCKED
        if method == "winograd_f4x3":
            return Route.WF4X3
        return Route.CBLOCKED_NCHW
    return Route.GENERAL


def conv2d(
    x,
    w_oihw,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    activation="none",
    layout="nchw",
):
    """Forward 2-D conv on AMD ROCm via Triton. Drop-in for the forward of
    ``torch.nn.functional.conv2d`` (no backward).

    A shape-driven router picks among five kernel families (1x1, 3x3 cblocked,
    3x3 NHWC, Winograd F(4x4,3x3), general) per call.

    Inputs must be fp16 or bf16. ``layout="nhwc"`` runs an NHWC-native kernel
    with no internal layout conversion.

    Output dtype always matches the input dtype, matching
    ``torch.nn.Conv2d`` semantics.

    Notes
    -----
    - Only ``groups=1`` (depthwise/grouped raises ``AssertionError``).
    - Only ``padding_mode="zeros"`` (no reflect/replicate/circular).
    - ``bias=None`` skips the with-bias kernel path; passing a zero tensor
      instead routes through the with-bias kernel and times differently.
    """
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"conv2d only supports fp16 and bf16 inputs, got {x.dtype}")
    layout = layout.lower()
    if layout not in ("nchw", "nhwc"):
        raise ValueError(f"layout must be 'nchw' or 'nhwc', got '{layout}'")

    _LOGGER.info(
        f"CONV2D: x={tuple(x.shape)} w={tuple(w_oihw.shape)} stride={stride} "
        f"padding={padding} dilation={dilation} layout={layout} "
        f"dtype={x.dtype} bias={'yes' if bias is not None else 'no'} "
        f"act={activation}"
    )

    if layout == "nhwc":
        return conv2d_nhwc(x, w_oihw, bias, stride, padding, dilation, activation)
    else:
        return conv2d_nchw(x, w_oihw, bias, stride, padding, dilation, activation)


def conv2d_winograd_f4x3(
    x,
    w_oihw,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    activation="none",
    block_k=BLOCK_K,
    layout="nchw",
):
    """NCHW/NHWC conv2d using Winograd F(4x4,3x3). Raises ValueError for non-eligible convs."""
    N, C, H, W_in, K_out, R, S, P, Q = _conv_dims(x, w_oihw, stride, padding, dilation)
    _require_winograd_eligible("conv2d_winograd_f4x3", R, S, stride, dilation, C)

    y = _alloc_output(N, K_out, P, Q, x, layout)
    bias_fp32 = _prep_bias(bias)
    U, C_pad = get_or_make_winograd_filter_f4x3(w_oihw.contiguous(), block_k)
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
    block_k=BLOCK_K,
    x_blocked=None,
):
    """NCHW conv2d using Winograd F(4x4,3x3) with NCHWc input layout for coalesced loads.
    Raises ValueError for non-eligible convs.

    x_blocked: optional pre-packed NCHWc input. Used by the benchmark to time
    the kernel without host-side input packing; when None (the normal inference
    path) the input is packed here."""
    N, C, H, W_in, K_out, R, S, P, Q = _conv_dims(x, w_oihw, stride, padding, dilation)
    _require_winograd_eligible(
        "conv2d_winograd_f4x3_cblocked", R, S, stride, dilation, C
    )

    y = _alloc_output(N, K_out, P, Q, x, "nchw")
    bias_fp32 = _prep_bias(bias)
    U, C_pad = get_or_make_winograd_filter_f4x3(w_oihw.contiguous(), block_k)
    if x_blocked is None:
        x_blocked, C_pad_blocked = prepack_nchw_to_cblocked(x, block_k)
    else:
        C_pad_blocked = x_blocked.shape[-1] * x_blocked.shape[1]
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
        activation,
        block_k,
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
    block_k=BLOCK_K,
    layout="nchw",
):
    """NCHW/NHWC conv2d for 1x1 kernels. Raises ValueError for non-1x1."""
    N, C, H, W_in, K_out, R, S, P, Q = _conv_dims(x, w_oihw, stride, padding, dilation)
    if not _is_1x1_conv(R, S, dilation):
        raise ValueError(f"conv2d_1x1 requires 1x1 kernel, got {R}x{S}")

    y = _alloc_output(N, K_out, P, Q, x, layout)
    bias_fp32 = _prep_bias(bias)
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
    block_k=BLOCK_K,
    layout="nchw",
):
    """NCHW/NHWC conv2d using general kernel with prepacked weights (5x5, 7x7, etc.)."""
    N, C, H, W_in, K_out, R, S, P, Q = _conv_dims(x, w_oihw, stride, padding, dilation)

    y = _alloc_output(N, K_out, P, Q, x, layout)
    bias_fp32 = _prep_bias(bias)
    w_k, K_pad = get_or_make_weight_pack(w_oihw.contiguous(), block_k)
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
    block_k=BLOCK_K,
):
    """NHWC conv2d for 3x3 kernels. Raises ValueError for non-3x3."""
    N, C, H, W_in, K_out, R, S, P, Q = _conv_dims(x, w_oihw, stride, padding, dilation)
    if not _is_3x3_conv(R, S):
        raise ValueError(f"conv2d_nhwc_3x3 requires 3x3 kernel, got {R}x{S}")

    y = _alloc_output(N, K_out, P, Q, x, "nhwc")
    bias_fp32 = _prep_bias(bias)
    w_3x3, C_pad = get_or_make_weight_pack_3x3(w_oihw.contiguous(), block_k)
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
        activation,
    )
    return y


def _route_and_run(
    x, w_oihw, bias, stride, padding, dilation, activation, block_k, layout
):
    """Shared router body: resolve the route once and dispatch to the wrapper.
    Single source of dispatch shared by conv2d_nchw and conv2d_nhwc."""
    N, C, H, W_in = x.shape
    K_out, _, R, S = w_oihw.shape
    route = _resolve_route(R, S, stride, dilation, N, C, H, W_in, K_out, layout)

    if route == Route.ONE_X_ONE:
        return conv2d_1x1(
            x,
            w_oihw,
            bias,
            stride,
            padding,
            dilation,
            activation,
            block_k,
            layout=layout,
        )
    if route == Route.WF4X3_CBLOCKED:
        return conv2d_winograd_f4x3_cblocked(
            x, w_oihw, bias, stride, padding, dilation, activation, block_k
        )
    if route == Route.WF4X3:
        return conv2d_winograd_f4x3(
            x,
            w_oihw,
            bias,
            stride,
            padding,
            dilation,
            activation,
            block_k,
            layout=layout,
        )
    if route == Route.CBLOCKED_NCHW:
        return conv2d_nchw_cblocked(
            x, w_oihw, bias, stride, padding, dilation, activation, block_k
        )
    if route == Route.NHWC_3X3:
        return conv2d_nhwc_3x3(
            x, w_oihw, bias, stride, padding, dilation, activation, block_k
        )
    return conv2d_general(
        x,
        w_oihw,
        bias,
        stride,
        padding,
        dilation,
        activation,
        block_k,
        layout=layout,
    )


def conv2d_nchw(
    x,
    w_oihw,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    activation="none",
    block_k=BLOCK_K,
):
    """Hybrid NCHW conv2d: routes to specialized 1x1, 3x3, or general kernel."""
    assert x.is_cuda and w_oihw.is_cuda
    return _route_and_run(
        x,
        w_oihw,
        bias,
        stride,
        padding,
        dilation,
        activation,
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
    block_k=BLOCK_K,
):
    """Conv2d with NHWC (channels-last) input and output.

    Input x can be NCHW or NHWC — it will be converted to channels_last.
    Output y is allocated as channels_last (NHWC-contiguous) and returned
    in logical NCHW shape with channels_last strides.
    """
    assert x.is_cuda and w_oihw.is_cuda
    x = x.to(memory_format=torch.channels_last)
    return _route_and_run(
        x,
        w_oihw,
        bias,
        stride,
        padding,
        dilation,
        activation,
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
    block_k=BLOCK_K,
    x_blocked=None,
):
    """NCHW conv2d with channel-blocked input packing for 3x3 kernels.
    Raises ValueError for non-3x3.

    x_blocked: optional pre-packed NCHWc input. Used by the benchmark to time
    the kernel without host-side input packing; when None (the normal inference
    path) the input is packed here."""
    N, C, H, W_in, K_out, R, S, P, Q = _conv_dims(x, w_oihw, stride, padding, dilation)

    if not _is_3x3_conv(R, S):
        raise ValueError(f"conv2d_nchw_cblocked requires 3x3 kernel, got {R}x{S}")

    y = _alloc_output(N, K_out, P, Q, x, "nchw")
    bias_fp32 = _prep_bias(bias)
    w_3x3, C_pad = get_or_make_weight_pack_3x3(w_oihw.contiguous(), block_k)
    if x_blocked is None:
        # input channel-block size matches the weight padding block
        x_blocked, C_pad_x = prepack_nchw_to_cblocked(x, block_k)
    else:
        C_pad_x = x_blocked.shape[-1] * x_blocked.shape[1]
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
        block_k,
        stride,
        padding,
        dilation,
        activation,
    )
    return y
