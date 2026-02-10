# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton-based Conv2d matching the PyTorch ``torch.nn.Conv2d`` interface.

Supports:
  - NCHW and NHWC data layouts (selected via ``layout`` parameter)
  - arbitrary kernel sizes, strides, padding, dilation
  - grouped convolutions (including depthwise when groups == in_channels)
  - optional bias
  - ``padding_mode='zeros'`` only (other modes should be applied externally)

Limitations (current):
  - forward pass only (no autograd backward)
  - no ``padding_mode`` other than ``'zeros'``
"""

from typing import Optional, Tuple, Union

import torch
import triton

from aiter.ops.triton._triton_kernels.conv2d import (
    _conv2d_implicit_gemm_kernel,
    _conv2d_nhwc_kernel,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Normalise a scalar or 2-tuple to a 2-tuple."""
    if isinstance(x, int):
        return (x, x)
    assert len(x) == 2
    return tuple(x)


def _compute_output_size(
    in_h: int, in_w: int, kh: int, kw: int,
    stride: Tuple[int, int], padding: Tuple[int, int],
    dilation: Tuple[int, int],
) -> Tuple[int, int]:
    oh = (in_h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    ow = (in_w + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    return oh, ow


def _pick_block_k(K: int) -> int:
    """Choose BLOCK_K: power-of-two, >= 16 (tl.dot minimum), <= 64."""
    bk = min(32, K) if K > 0 else 1
    bk = triton.next_power_of_2(bk)
    return max(bk, 16)


# -------------------------------------------------------------------
# NCHW dispatch (original)
# -------------------------------------------------------------------

def _launch_nchw(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
) -> torch.Tensor:
    N, C_in, H_in, W_in = input.shape
    C_out, C_in_per_group, kH, kW = weight.shape

    H_out, W_out = _compute_output_size(H_in, W_in, kH, kW, stride, padding, dilation)

    input = input.contiguous()
    weight = weight.contiguous()

    output = torch.empty(
        (N, C_out, H_out, W_out), dtype=input.dtype, device=input.device
    )

    M = N * H_out * W_out
    N_gemm = C_out // groups
    K = C_in_per_group * kH * kW
    BLOCK_K = _pick_block_k(K)

    grid = (
        triton.cdiv(M, 64) * triton.cdiv(N_gemm, 64),
        groups,
    )

    _conv2d_implicit_gemm_kernel[grid](
        input, weight,
        bias if bias is not None else input,
        output,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kH, kW,
        stride[0], stride[1], padding[0], padding[1],
        dilation[0], dilation[1], groups,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        M, N_gemm, K, C_in_per_group,
        HAS_BIAS=bias is not None,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=BLOCK_K,
    )
    return output


# -------------------------------------------------------------------
# NHWC dispatch
# -------------------------------------------------------------------

def _launch_nhwc(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
) -> torch.Tensor:
    """Launch the NHWC-optimised kernel.

    Parameters
    ----------
    input : Tensor (N, H, W, C_in) – contiguous NHWC
    weight : Tensor (C_out, C_in/groups, kH, kW) – standard PyTorch format.
             Permuted internally to (C_out, kH, kW, C_in/groups) for coalesced
             weight loads.
    """
    N, H_in, W_in, C_in = input.shape
    C_out, C_in_per_group, kH, kW = weight.shape

    H_out, W_out = _compute_output_size(H_in, W_in, kH, kW, stride, padding, dilation)

    input = input.contiguous()

    # Permute weight: (Co, Ci/g, kH, kW) -> (Co, kH, kW, Ci/g) for coalesced K loads
    weight_nhwc = weight.permute(0, 2, 3, 1).contiguous()

    output = torch.empty(
        (N, H_out, W_out, C_out), dtype=input.dtype, device=input.device
    )

    M = N * H_out * W_out
    N_gemm = C_out // groups
    K = C_in_per_group * kH * kW
    BLOCK_K = _pick_block_k(K)

    grid = (
        triton.cdiv(M, 64) * triton.cdiv(N_gemm, 64),
        groups,
    )

    _conv2d_nhwc_kernel[grid](
        input, weight_nhwc,
        bias if bias is not None else input,
        output,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kH, kW,
        stride[0], stride[1], padding[0], padding[1],
        dilation[0], dilation[1], groups,
        # Input strides (N, H, W, C)
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        # Weight strides (Co, kH, kW, Ci/g)
        weight_nhwc.stride(0), weight_nhwc.stride(1),
        weight_nhwc.stride(2), weight_nhwc.stride(3),
        # Output strides (N, Ho, Wo, Co)
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        M, N_gemm, K, C_in_per_group,
        HAS_BIAS=bias is not None,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=BLOCK_K,
    )
    return output


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    layout: str = "nchw",
) -> torch.Tensor:
    """Triton Conv2d forward pass.

    Parameters
    ----------
    input : Tensor
        * ``layout="nchw"``: shape ``(N, C_in, H, W)`` or ``(C_in, H, W)``
        * ``layout="nhwc"``: shape ``(N, H, W, C_in)`` or ``(H, W, C_in)``
    weight : Tensor
        Always ``(C_out, C_in/groups, kH, kW)`` (standard PyTorch format).
    bias : Tensor, optional – shape ``(C_out,)``
    stride, padding, dilation : int or (int, int)
    groups : int
    layout : ``"nchw"`` or ``"nhwc"``

    Returns
    -------
    Tensor – same layout as *input*.
    """
    layout = layout.lower()
    assert layout in ("nchw", "nhwc"), f"layout must be 'nchw' or 'nhwc', got '{layout}'"

    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    # Handle unbatched input
    unbatched = input.dim() == 3
    if unbatched:
        input = input.unsqueeze(0)

    assert input.dim() == 4, f"Expected 4-D input, got {input.dim()}-D"
    assert weight.dim() == 4, f"Expected 4-D weight, got {weight.dim()}-D"

    # Extract dimensions according to layout
    if layout == "nchw":
        _N, C_in, _H, _W = input.shape
    else:
        _N, _H, _W, C_in = input.shape

    C_out, C_in_per_group, kH, kW = weight.shape

    assert C_in % groups == 0, f"in_channels ({C_in}) not divisible by groups ({groups})"
    assert C_out % groups == 0, f"out_channels ({C_out}) not divisible by groups ({groups})"
    assert C_in_per_group == C_in // groups, (
        f"weight dim-1 ({C_in_per_group}) != in_channels/groups ({C_in // groups})"
    )
    if bias is not None:
        assert bias.shape == (C_out,), f"bias shape {bias.shape} != ({C_out},)"

    H_out, W_out = _compute_output_size(_H, _W, kH, kW, stride, padding, dilation)
    assert H_out > 0 and W_out > 0, (
        f"Invalid output size ({H_out}, {W_out}). Check kernel/stride/pad/dilation."
    )

    # Dispatch
    if layout == "nchw":
        output = _launch_nchw(input, weight, bias, stride, padding, dilation, groups)
    else:
        output = _launch_nhwc(input, weight, bias, stride, padding, dilation, groups)

    if unbatched:
        output = output.squeeze(0)

    return output


def conv2d_nhwc(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
) -> torch.Tensor:
    """Convenience shorthand for ``conv2d(..., layout='nhwc')``.

    Input is ``(N, H, W, C_in)``; output is ``(N, H_out, W_out, C_out)``.
    Weight is always ``(C_out, C_in/groups, kH, kW)``.
    """
    return conv2d(input, weight, bias, stride, padding, dilation, groups, layout="nhwc")
