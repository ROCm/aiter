# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl


@triton.jit
def _modulate(normed, scale, shift, indices, output, COUNT: tl.constexpr,
              WIDTH: tl.constexpr, SCALE_STRIDE: tl.constexpr,
              SHIFT_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < COUNT
    row = offsets // WIDTH
    column = offsets % WIDTH
    group = tl.load(indices + row, valid, other=0)
    scale_value = tl.load(scale + group * SCALE_STRIDE + column, valid, other=0).to(tl.float32)
    factor = (1.0 + scale_value).to(tl.bfloat16).to(tl.float32)
    normalized = tl.load(normed + offsets, valid, other=0).to(tl.float32)
    product = (normalized * factor).to(tl.bfloat16).to(tl.float32)
    shift_value = tl.load(shift + group * SHIFT_STRIDE + column, valid, other=0).to(tl.float32)
    tl.store(output + offsets, product + shift_value, valid)


def modulate(normed, scale, shift, indices):
    if normed.ndim != 2 or min(normed.shape) < 1 or not normed.is_cuda or not normed.is_contiguous():
        raise ValueError('Expected positive contiguous GPU [S,D] normalized input')
    if any(tensor.dtype != torch.bfloat16 or tensor.device != normed.device or tensor.requires_grad
           for tensor in (normed, scale, shift)):
        raise ValueError('Same-device BF16 inference tensors required')
    if scale.ndim != 2 or scale.shape != shift.shape or scale.shape[1] != normed.shape[1] or scale.shape[0] < 1:
        raise ValueError('Expected matching positive [groups,D] scale and shift')
    if any(tensor.stride(1) != 1 or tensor.stride(0) < tensor.shape[1] for tensor in (scale, shift)):
        raise ValueError('Modulation tables require contiguous columns and nonoverlapping rows')
    if indices.shape != (normed.shape[0],) or indices.device != normed.device or not indices.is_contiguous() or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError('Expected contiguous per-token int32/int64 indices on the same GPU')
    if normed.numel() >= 2**31 or max(scale.stride(0), shift.stride(0)) * scale.shape[0] >= 2**31:
        raise ValueError('32-bit index limit exceeded')
    output = torch.empty_like(normed)
    with torch.cuda.device(normed.device):
        _modulate[(triton.cdiv(normed.numel(), 1024),)](
            normed, scale, shift, indices, output, normed.numel(), normed.shape[1],
            scale.stride(0), shift.stride(0), BLOCK=1024, num_warps=4, enable_fp_fusion=False)
    return output
