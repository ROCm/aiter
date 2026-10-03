# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl


@triton.jit
def _gated_residual(residual, projected, gate, indices, output,
                    count: tl.constexpr, dimension: tl.constexpr,
                    gate_stride: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < count
    rows = offsets // dimension
    columns = offsets % dimension
    group = tl.load(indices + rows, mask=valid, other=0)
    gate_value = tl.load(gate + group * gate_stride + columns, mask=valid, other=0).to(tl.float32)
    branch = tl.load(projected + offsets, mask=valid, other=0).to(tl.float32)
    product = (gate_value * branch).to(tl.bfloat16).to(tl.float32)
    original = tl.load(residual + offsets, mask=valid, other=0).to(tl.float32)
    tl.store(output + offsets, original + product, mask=valid)


def gated_residual(residual, projected, gate, indices):
    if residual.ndim != 2 or residual.shape != projected.shape or residual.numel() == 0:
        raise ValueError('Expected matching positive [S,D] residual and projected tensors')
    if any(tensor.device != residual.device or tensor.dtype != torch.bfloat16 for tensor in (residual, projected, gate)):
        raise ValueError('Expected BF16 tensors on one GPU')
    if not residual.is_cuda or not residual.is_contiguous() or not projected.is_contiguous():
        raise ValueError('Expected contiguous GPU residual and projected tensors')
    if gate.ndim != 2 or gate.shape[1] != residual.shape[1] or gate.shape[0] < 1 or gate.stride(1) != 1:
        raise ValueError('Expected [groups,D] gate with contiguous columns')
    if indices.shape != (residual.shape[0],) or indices.device != residual.device or not indices.is_contiguous() or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError('Expected contiguous per-token int32/int64 indices on the same GPU')
    if any(tensor.requires_grad for tensor in (residual, projected, gate)):
        raise ValueError('Inference only')
    output = torch.empty_like(residual)
    with torch.cuda.device(residual.device):
        _gated_residual[(triton.cdiv(residual.numel(), 1024),)](
            residual, projected, gate, indices, output, residual.numel(), residual.shape[1], gate.stride(0),
            BLOCK=1024, num_warps=4, enable_fp_fusion=False)
    return output
