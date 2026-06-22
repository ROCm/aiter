# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Opus conv2d implicit GEMM (opus-fused, gfx942) Python user-facing API.

Public entry point: ``conv2d_implicit_opus(input, weight, ...)``

Tensor layout: NHWC bf16.  The kernel requires padded tensors:
  - Cpg (= C/group) padded to multiple of 8
  - Kpg (= K/group) padded to multiple of 16
  - GEMM_K (= R*S*Cpg_pad) padded to multiple of 128
  - Weight pre-packed to [group, Kpg_pad, GEMM_K_pad] in RSC order

This module handles padding/packing transparently.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from ...jit.core import compile_ops


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _gen_conv2d_fake(
    input: Tensor, weight: Tensor, output: Tensor,
    N_batch: int, C: int, K: int,
    Hi: int, Wi: int,
    R: int, S: int,
    pad_h: int, pad_w: int,
    stride_h: int, stride_w: int,
    dil_h: int, dil_w: int,
    group: int,
) -> Tensor:
    return output


@compile_ops(
    "module_opus_conv2d",
    fc_name="opus_conv2d_implicit",
    gen_fake=_gen_conv2d_fake,
    develop=True,
)
def _opus_conv2d_implicit_raw(
    input: Tensor, weight: Tensor, output: Tensor,
    N_batch: int, C: int, K: int,
    Hi: int, Wi: int,
    R: int, S: int,
    pad_h: int, pad_w: int,
    stride_h: int, stride_w: int,
    dil_h: int, dil_w: int,
    group: int,
) -> Tensor: ...


def _pad_input(x: Tensor, C: int, C_pad: int) -> Tensor:
    """Pad channel dimension from C to C_pad: [N, Hi, Wi, C] -> [N, Hi, Wi, C_pad]."""
    if C == C_pad:
        return x.contiguous()
    return F.pad(x, (0, C_pad - C)).contiguous()


def _pack_weight_rsc(
    weight: Tensor,
    K: int, R: int, S: int, Cpg: int,
    group: int, Kpg_pad: int, GEMM_K_pad: int, Cpg_pad: int,
) -> Tensor:
    """Pack weight [K, R, S, Cpg] -> [group, Kpg_pad, GEMM_K_pad] in RSC order with padding."""
    Kpg = K // group
    device = weight.device
    packed = torch.zeros(group, Kpg_pad, GEMM_K_pad, dtype=weight.dtype, device=device)
    for ig in range(group):
        for ik in range(Kpg):
            for r in range(R):
                for s in range(S):
                    for c in range(Cpg):
                        k_idx = r * S * Cpg_pad + s * Cpg_pad + c
                        packed[ig, ik, k_idx] = weight[ig * Kpg + ik, r, s, c]
    return packed.contiguous()


def _pack_weight_rsc_vectorized(
    weight: Tensor,
    K: int, R: int, S: int, Cpg: int,
    group: int, Kpg_pad: int, GEMM_K_pad: int, Cpg_pad: int,
) -> Tensor:
    """Vectorized weight packing: [K, R, S, Cpg] -> [group, Kpg_pad, GEMM_K_pad]."""
    Kpg = K // group
    device = weight.device

    # weight is [K, R, S, Cpg] = [group*Kpg, R, S, Cpg]
    # reshape to [group, Kpg, R, S, Cpg]
    w = weight.view(group, Kpg, R, S, Cpg)

    # Pad Cpg -> Cpg_pad along last dim
    if Cpg_pad > Cpg:
        w = F.pad(w, (0, Cpg_pad - Cpg))

    # Now [group, Kpg, R, S, Cpg_pad] — flatten R*S*Cpg_pad -> GEMM_K_real
    w = w.reshape(group, Kpg, R * S * Cpg_pad)

    # Pad Kpg -> Kpg_pad
    if Kpg_pad > Kpg:
        w = F.pad(w, (0, 0, 0, Kpg_pad - Kpg))

    # Pad GEMM_K_real -> GEMM_K_pad
    GEMM_K_real = R * S * Cpg_pad
    if GEMM_K_pad > GEMM_K_real:
        w = F.pad(w, (0, GEMM_K_pad - GEMM_K_real))

    return w.contiguous()


def conv2d_implicit_opus(
    input: Tensor,
    weight: Tensor,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
) -> Tensor:
    """
    Conv2D forward using implicit GEMM fused into opus asm pipeline.

    Args:
        input:    [N, Hi, Wi, C] bf16 NHWC
        weight:   [K, R, S, C/groups] bf16 NHWC
        stride:   convolution stride (h, w)
        padding:  convolution padding (h, w)
        dilation: convolution dilation (h, w)
        groups:   number of groups

    Returns:
        output: [N, Ho, Wo, K] bf16 NHWC
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    N = input.shape[0]
    Hi = input.shape[1]
    Wi = input.shape[2]
    C = input.shape[3]
    K = weight.shape[0]
    R = weight.shape[1]
    S = weight.shape[2]
    Cpg = C // groups
    Kpg = K // groups

    Ho = (Hi + 2 * pad_h - dil_h * (R - 1) - 1) // stride_h + 1
    Wo = (Wi + 2 * pad_w - dil_w * (S - 1) - 1) // stride_w + 1

    Cpg_pad = _ceil_div(Cpg, 8) * 8
    C_pad = groups * Cpg_pad
    Kpg_pad = _ceil_div(Kpg, 16) * 16
    GEMM_K_real = R * S * Cpg_pad
    GEMM_K_pad = _ceil_div(GEMM_K_real, 128) * 128
    M = N * Ho * Wo
    stride_out = groups * Kpg_pad

    # Pad input channels
    input_padded = _pad_input(input, C, C_pad)

    # Pack weight
    weight_packed = _pack_weight_rsc_vectorized(
        weight, K, R, S, Cpg, groups, Kpg_pad, GEMM_K_pad, Cpg_pad
    )

    # Allocate output: [M, stride_out]
    output = torch.zeros(M, stride_out, dtype=input.dtype, device=input.device)

    _opus_conv2d_implicit_raw(
        input_padded, weight_packed, output,
        N, C, K, Hi, Wi, R, S,
        pad_h, pad_w, stride_h, stride_w, dil_h, dil_w, groups,
    )

    # Extract real output: [M, group*Kpg_pad] -> [N, Ho, Wo, K]
    # For each group, take the first Kpg columns out of Kpg_pad
    if groups == 1:
        result = output[:, :K].reshape(N, Ho, Wo, K)
    else:
        # output is [M, groups*Kpg_pad], need to gather [ig*Kpg_pad : ig*Kpg_pad+Kpg] for each ig
        slices = []
        for ig in range(groups):
            slices.append(output[:, ig * Kpg_pad : ig * Kpg_pad + Kpg])
        result = torch.cat(slices, dim=-1).reshape(N, Ho, Wo, K)

    return result.contiguous()
