# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl


@triton.jit
def _up_swiglu(inputs, weights, output, rows: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                GROUP: tl.constexpr):
    reduction: tl.constexpr = 5376
    channels: tl.constexpr = 7168
    program = tl.program_id(0)
    row_tiles = tl.cdiv(rows, BM)
    column_tiles: tl.constexpr = channels // (BN // 2)
    group_size: tl.constexpr = GROUP * column_tiles
    first_row = (program // group_size) * GROUP
    active_rows = tl.minimum(row_tiles - first_row, GROUP)
    row_tile = first_row + program % active_rows
    column_tile = (program % group_size) // active_rows
    row = row_tile * BM + tl.arange(0, BM)
    paired_column = tl.arange(0, BN)
    column = column_tile * (BN // 2) + paired_column % (BN // 2)
    weight_column = column + (paired_column // (BN // 2)) * channels
    inner = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for block in range(reduction // BK):
        position = block * BK + inner
        activation = tl.load(inputs + row[:, None] * reduction + position[None, :],
                             row[:, None] < rows, other=0)
        weight = tl.load(weights + weight_column[None, :] * reduction + position[:, None])
        accumulator = tl.dot(activation, weight, accumulator)
    paired = accumulator.to(tl.bfloat16).to(tl.float32).reshape((BM, 2, BN // 2)).trans(0, 2, 1)
    value, gate = tl.split(paired)
    activated = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    out_column = column_tile * (BN // 2) + tl.arange(0, BN // 2)
    tl.store(output + row[:, None] * channels + out_column[None, :],
             (value * activated).to(tl.bfloat16), row[:, None] < rows)


def fused_up_out(inputs, weights, output):
    if inputs.ndim != 2 or inputs.shape[0] < 1 or inputs.shape[1] != 5376:
        raise ValueError('Expected positive [S,5376] input')
    if weights.shape != (14336, 5376) or output.shape != (inputs.shape[0], 7168):
        raise ValueError('Expected TP2 paired [14336,5376] weight and [S,7168] output')
    if any(tensor.dtype != torch.bfloat16 or not tensor.is_cuda or
           not tensor.is_contiguous() or tensor.requires_grad or tensor.device != inputs.device
           for tensor in (inputs, weights, output)):
        raise ValueError('Contiguous BF16 inference tensors on one GPU required')
    if max(inputs.numel(), output.numel()) >= 2**31:
        raise ValueError('32-bit index limit exceeded')
    with torch.cuda.device(inputs.device):
        return _up_swiglu[(triton.cdiv(inputs.shape[0], 128) * 56,)](
            inputs, weights, output, inputs.shape[0], BM=128, BN=256, BK=64,
            GROUP=32, num_warps=8, num_stages=2, enable_fp_fusion=False)


def fused_up(inputs, weights):
    output = torch.empty((inputs.shape[0], 7168), device=inputs.device, dtype=inputs.dtype)
    fused_up_out(inputs, weights, output)
    return output


@triton.jit
def _swiglu(projected, output, count: tl.constexpr, channels: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    source = (offsets // channels) * (2 * channels) + offsets % channels
    value = tl.load(projected + source, offsets < count, other=0).to(tl.float32)
    gate = tl.load(projected + source + channels, offsets < count, other=0).to(tl.float32)
    activated = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    tl.store(output + offsets, value * activated, offsets < count)


def fused_swiglu(projected):
    if projected.ndim != 2 or projected.shape[1] % 2 or projected.numel() == 0:
        raise ValueError('Expected positive [S,2*N] projection')
    if projected.dtype != torch.bfloat16 or not projected.is_cuda or not projected.is_contiguous() or projected.requires_grad:
        raise ValueError('Expected contiguous GPU BF16 inference projection')
    rows, width = projected.shape
    output = torch.empty((rows, width // 2), device=projected.device, dtype=projected.dtype)
    with torch.cuda.device(projected.device):
        _swiglu[(triton.cdiv(output.numel(), 1024),)](
            projected, output, output.numel(), width // 2, BLOCK=1024,
            num_warps=4, enable_fp_fusion=False)
    return output


def library_up(inputs, weights):
    return fused_swiglu(torch.mm(inputs, weights.t()))
