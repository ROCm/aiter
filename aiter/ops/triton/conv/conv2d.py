# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import logging
import os
from enum import Enum

import torch
import triton

from aiter.ops.triton._triton_kernels.conv.helpers import CONV_AUTOTUNE_ENABLED
from aiter.ops.triton.conv._launch import (
    _launch_1x1,
    _launch_3x3_cblocked,
    _launch_3x3_nchw,
    _launch_3x3_nhwc,
    _launch_general,
    _launch_winograd_f4x3,
    _launch_winograd_f4x3_cblocked,
)
from aiter.ops.triton.conv._prepack import (
    get_or_make_weight_pack,
    get_or_make_weight_pack_3x3,
    get_or_make_winograd_filter_f4x3,
    prepack_nchw_to_cblocked,
)
from aiter.ops.triton.conv._utils import (
    BLOCK_K,
    _alloc_output,
    _conv_dims,
    _is_1x1_conv,
    _is_3x3_conv,
    _is_winograd_eligible,
    _out_hw,
    _prep_bias,
    _require_winograd_eligible,
)
from aiter.ops.triton.utils.conv_config_utils import has_conv_config
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


class Route(Enum):
    # Values are actual kernel display names. The benchmark substring-matches
    # "winograd" and "cblocked" to select tolerances and prepacked timing.
    ONE_X_ONE = "_conv2d_1x1_kernel"
    WF4X3_CBLOCKED = "_winograd_f4x3_cblocked_* (3 kern)"
    WF4X3 = "_winograd_f4x3_* (3 kernels)"
    DIRECT_NCHW_3X3 = "_conv2d_3x3_nchw_kernel"
    CBLOCKED_NCHW = "_conv2d_3x3_cblocked_kernel"
    NHWC_3X3 = "_conv2d_3x3_nhwc_kernel"
    GENERAL = "_conv2d_general_kernel"


def _is_amd_wave32():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.warp_size == 32


# On gfx1201, cblocked wins at 196 pixels while direct NCHW wins at 784.
# Allow other architectures and workloads to adjust the crossover at import.
_NCHW_DIRECT_MIN_PIXELS = int(
    os.getenv("AITER_TRITON_CONV_NCHW_DIRECT_MIN_PIXELS", "512")
)


def _nchw_direct_is_profitable(N: int, H: int, W: int) -> bool:
    """Whether direct NCHW should replace a materialized NCHWc input pack."""
    return N * H * W >= _NCHW_DIRECT_MIN_PIXELS and (
        CONV_AUTOTUNE_ENABLED or has_conv_config("CONV-3X3-NCHW")
    )


def _select_3x3_method(N, C, H, W, K_out, stride, dilation, block_c=BLOCK_K):
    """Pick the best 3x3 kernel method based on shape heuristics.

    RDNA wave32 uses the direct kernel. With Triton's RDNA backend fixed at
    ``num_stages=1``, the three Winograd launches and materialized V/M tensors
    cost more than the reduced arithmetic saves on the model shapes. Keep the
    previous Winograd heuristic for non-RDNA targets, where the pipeline and
    occupancy trade-off differs.
    """
    if C < block_c:
        return "general"
    if not _is_winograd_eligible(3, 3, stride, dilation, C):
        return "cblocked"
    P, Q = _out_hw(H, W, 3, 3, stride, (1, 1), dilation)
    tile_H = (P + 3) // 4
    tile_W = (Q + 3) // 4
    T = N * tile_H * tile_W
    if C >= 512 and K_out >= 512 and T >= 98:
        if _is_amd_wave32():
            return "cblocked"
        if T >= 392:
            return "winograd_f4x3_cblocked"
        return "winograd_f4x3"
    return "cblocked"


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
        if method == "general":
            return Route.GENERAL
        if method == "winograd_f4x3_cblocked":
            return Route.WF4X3_CBLOCKED
        if method == "winograd_f4x3":
            return Route.WF4X3
        if _nchw_direct_is_profitable(N, H, W_in):
            return Route.DIRECT_NCHW_3X3
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

    A shape-driven router picks among six kernel families (1x1, direct NCHW
    3x3, cblocked 3x3, NHWC 3x3, Winograd F(4x4,3x3), general) per call.

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

    if _LOGGER.get_logger().isEnabledFor(logging.INFO):
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
    bias = _prep_bias(bias)
    U, C_pad = get_or_make_winograd_filter_f4x3(w_oihw, block_k)
    _launch_winograd_f4x3(
        x,
        U,
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

    x_blocked: optional pre-packed NCHWc input. Routed execution supplies it
    explicitly; direct method calls may leave it unset to include packing."""
    N, C, H, W_in, K_out, R, S, P, Q = _conv_dims(x, w_oihw, stride, padding, dilation)
    _require_winograd_eligible(
        "conv2d_winograd_f4x3_cblocked", R, S, stride, dilation, C
    )

    y = _alloc_output(N, K_out, P, Q, x, "nchw")
    bias = _prep_bias(bias)
    U, C_pad = get_or_make_winograd_filter_f4x3(w_oihw, block_k)
    if x_blocked is None:
        x_blocked, C_pad_blocked = prepack_nchw_to_cblocked(x, block_k)
    else:
        if x_blocked.ndim != 5:
            raise ValueError(
                "conv2d_winograd_f4x3_cblocked requires a 5-D NCHWc "
                f"x_blocked tensor, got {x_blocked.ndim}-D"
            )
        C_pad_blocked = x_blocked.shape[-1] * x_blocked.shape[1]
    _launch_winograd_f4x3_cblocked(
        x_blocked,
        C_pad_blocked,
        U,
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
    bias = _prep_bias(bias)
    _launch_1x1(
        x,
        w_oihw.contiguous(),
        bias,
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
    bias = _prep_bias(bias)
    w_k, K_pad = get_or_make_weight_pack(w_oihw, block_k)
    _launch_general(
        x,
        w_k,
        bias,
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
    bias = _prep_bias(bias)
    w_3x3, C_pad = get_or_make_weight_pack_3x3(w_oihw, block_k)
    _launch_3x3_nhwc(
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
    )
    return y


def _route_and_run(
    x, w_oihw, bias, stride, padding, dilation, activation, block_k, layout
):
    """Resolve and execute one route, including required input preparation."""
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
        x_blocked, _ = prepack_nchw_to_cblocked(x, block_k)
        return conv2d_winograd_f4x3_cblocked(
            x,
            w_oihw,
            bias,
            stride,
            padding,
            dilation,
            activation,
            block_k,
            x_blocked=x_blocked,
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
    if route == Route.DIRECT_NCHW_3X3:
        return conv2d_nchw_3x3_direct(
            x, w_oihw, bias, stride, padding, dilation, activation, block_k
        )
    if route == Route.CBLOCKED_NCHW:
        x_blocked, _ = prepack_nchw_to_cblocked(x, block_k)
        return conv2d_nchw_cblocked(
            x,
            w_oihw,
            bias,
            stride,
            padding,
            dilation,
            activation,
            block_k,
            x_blocked=x_blocked,
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
    if not x.is_contiguous():
        x = x.contiguous()
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
    if not x.is_contiguous(memory_format=torch.channels_last):
        x = x.contiguous(memory_format=torch.channels_last)
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


def conv2d_nchw_3x3_direct(
    x,
    w_oihw,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    activation="none",
    block_k=BLOCK_K,
):
    """NCHW 3x3 convolution that reads the activation without repacking."""
    assert x.is_cuda and w_oihw.is_cuda
    if not x.is_contiguous():
        x = x.contiguous()
    N, C, H, W_in, K_out, R, S, P, Q = _conv_dims(x, w_oihw, stride, padding, dilation)
    if not _is_3x3_conv(R, S):
        raise ValueError(f"conv2d_nchw_3x3_direct requires 3x3 kernel, got {R}x{S}")

    y = _alloc_output(N, K_out, P, Q, x, "nchw")
    bias = _prep_bias(bias)
    w_3x3, C_pad = get_or_make_weight_pack_3x3(w_oihw, block_k)
    _launch_3x3_nchw(
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
    )
    return y


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

    x_blocked: optional pre-packed NCHWc input. Routed execution supplies it
    explicitly; direct method calls may leave it unset to include packing."""
    N, C, H, W_in, K_out, R, S, P, Q = _conv_dims(x, w_oihw, stride, padding, dilation)

    if not _is_3x3_conv(R, S):
        raise ValueError(f"conv2d_nchw_cblocked requires 3x3 kernel, got {R}x{S}")

    y = _alloc_output(N, K_out, P, Q, x, "nchw")
    bias = _prep_bias(bias)
    w_3x3, C_pad = get_or_make_weight_pack_3x3(w_oihw, block_k)
    if x_blocked is None:
        # input channel-block size matches the weight padding block
        x_blocked, C_pad_x = prepack_nchw_to_cblocked(x, block_k)
    else:
        if x_blocked.ndim != 5:
            raise ValueError(
                "conv2d_nchw_cblocked requires a 5-D NCHWc x_blocked tensor, "
                f"got {x_blocked.ndim}-D"
            )
        C_pad_x = x_blocked.shape[-1] * x_blocked.shape[1]
    # Ensure channel padding is consistent
    assert (
        C_pad_x == C_pad
    ), f"Channel padding mismatch: input {C_pad_x} vs weight {C_pad}"
    _launch_3x3_cblocked(
        x_blocked,
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
        block_k,
        stride,
        padding,
        dilation,
        activation,
    )
    return y
