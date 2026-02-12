# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton-based Conv2d matching the PyTorch ``torch.nn.Conv2d`` interface.

Supports NCHW/NHWC layouts, grouped convolutions, bias, split-K for
full GPU occupancy on small-batch workloads.
"""

from typing import Optional, Tuple, Union

import torch
import triton

from aiter.ops.triton._triton_kernels.conv2d import (
    _conv2d_implicit_gemm_kernel,
    _conv2d_nhwc_kernel,
    pick_conv2d_block_config,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
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


# -------------------------------------------------------------------
# NCHW dispatch
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

    M = N * H_out * W_out
    N_gemm = C_out // groups
    K = C_in_per_group * kH * kW

    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, split_k = pick_conv2d_block_config(
        M, N_gemm, K, groups
    )
    k_per_split = triton.cdiv(K, split_k)

    # Output buffer
    if split_k == 1:
        output = torch.empty(
            (N, C_out, H_out, W_out), dtype=input.dtype, device=input.device
        )
        split_k_stride = 0
        out_strides = (output.stride(0), output.stride(1),
                       output.stride(2), output.stride(3))
    else:
        # fp32 temp: [split_k, N_batch, C_out, H_out, W_out]
        output = torch.empty(
            (split_k, N, C_out, H_out, W_out), dtype=torch.float32, device=input.device
        )
        split_k_stride = output.stride(0)
        out_strides = (output.stride(1), output.stride(2),
                       output.stride(3), output.stride(4))

    grid = (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N_gemm, BLOCK_N),
        groups,
        split_k,
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
        out_strides[0], out_strides[1], out_strides[2], out_strides[3],
        M, N_gemm, K, C_in_per_group,
        k_per_split, split_k_stride,
        HAS_BIAS=bias is not None,
        SPLIT_K=split_k,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )

    if split_k > 1:
        output = output.sum(dim=0)                      # reduce split-K
        if bias is not None:
            output = output + bias.view(1, C_out, 1, 1)  # add bias
        output = output.to(input.dtype)

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
    N, H_in, W_in, C_in = input.shape
    C_out, C_in_per_group, kH, kW = weight.shape
    H_out, W_out = _compute_output_size(H_in, W_in, kH, kW, stride, padding, dilation)

    input = input.contiguous()
    weight_nhwc = weight.permute(0, 2, 3, 1).contiguous()

    M = N * H_out * W_out
    N_gemm = C_out // groups
    K = C_in_per_group * kH * kW

    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, split_k = pick_conv2d_block_config(
        M, N_gemm, K, groups
    )
    k_per_split = triton.cdiv(K, split_k)

    if split_k == 1:
        output = torch.empty(
            (N, H_out, W_out, C_out), dtype=input.dtype, device=input.device
        )
        split_k_stride = 0
        out_strides = (output.stride(0), output.stride(1),
                       output.stride(2), output.stride(3))
    else:
        output = torch.empty(
            (split_k, N, H_out, W_out, C_out), dtype=torch.float32, device=input.device
        )
        split_k_stride = output.stride(0)
        out_strides = (output.stride(1), output.stride(2),
                       output.stride(3), output.stride(4))

    grid = (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N_gemm, BLOCK_N),
        groups,
        split_k,
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
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight_nhwc.stride(0), weight_nhwc.stride(1),
        weight_nhwc.stride(2), weight_nhwc.stride(3),
        out_strides[0], out_strides[1], out_strides[2], out_strides[3],
        M, N_gemm, K, C_in_per_group,
        k_per_split, split_k_stride,
        HAS_BIAS=bias is not None,
        SPLIT_K=split_k,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )

    if split_k > 1:
        output = output.sum(dim=0)
        if bias is not None:
            output = output + bias.view(1, 1, 1, C_out)
        output = output.to(input.dtype)

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
    weight : Tensor  ``(C_out, C_in/groups, kH, kW)``
    bias : Tensor, optional  ``(C_out,)``
    stride, padding, dilation : int or (int, int)
    groups : int
    layout : ``"nchw"`` or ``"nhwc"``
    """
    layout = layout.lower()
    assert layout in ("nchw", "nhwc"), f"layout must be 'nchw' or 'nhwc', got '{layout}'"

    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    unbatched = input.dim() == 3
    if unbatched:
        input = input.unsqueeze(0)

    assert input.dim() == 4, f"Expected 4-D input, got {input.dim()}-D"
    assert weight.dim() == 4, f"Expected 4-D weight, got {weight.dim()}-D"

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
    """Convenience shorthand for ``conv2d(..., layout='nhwc')``."""
    return conv2d(input, weight, bias, stride, padding, dilation, groups, layout="nhwc")
