# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl


@triton.jit
def _down(inputs, weights, output, rows: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    reduction: tl.constexpr = 3584
    columns: tl.constexpr = 5376
    program = tl.program_id(0)
    row_tiles = tl.cdiv(rows, BM)
    column_tiles: tl.constexpr = tl.cdiv(columns, BN)
    group_size: tl.constexpr = GROUP * column_tiles
    group = program // group_size
    first_row = group * GROUP
    active_rows = tl.minimum(row_tiles - first_row, GROUP)
    row_tile = first_row + program % active_rows
    column_tile = (program % group_size) // active_rows
    row = row_tile * BM + tl.arange(0, BM)
    column = column_tile * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for block in range(reduction // BK):
        position = block * BK + inner
        activation = tl.load(inputs + row[:, None] * reduction + position[None, :],
                             row[:, None] < rows, other=0)
        weight = tl.load(weights + column[None, :] * reduction + position[:, None])
        accumulator = tl.dot(activation, weight, accumulator)
    tl.store(output + row[:, None] * columns + column[None, :],
             accumulator.to(tl.bfloat16), row[:, None] < rows)


def project_out(inputs, weights, output):
    if inputs.ndim != 2 or inputs.shape[0] < 1 or inputs.shape[1] != 3584:
        raise ValueError('Expected positive [S,3584] TP2 activation')
    if weights.shape != (3584, 5376) or weights.stride() != (1, 3584):
        raise ValueError('Expected transposed contiguous [3584,5376] local weight')
    if output.shape != (inputs.shape[0], 5376) or not output.is_contiguous() or not inputs.is_contiguous():
        raise ValueError('Expected contiguous input and [S,5376] output')
    if any(tensor.device != inputs.device or tensor.dtype != torch.bfloat16 or tensor.requires_grad
           for tensor in (inputs, weights, output)) or not inputs.is_cuda:
        raise ValueError('Expected inference BF16 tensors on the same GPU')
    if max(inputs.numel(), output.numel()) >= 2**31:
        raise ValueError('32-bit index limit exceeded')
    with torch.cuda.device(inputs.device):
        kernel = _down[(triton.cdiv(inputs.shape[0], 128) * 21,)](
            inputs, weights, output, inputs.shape[0], BM=128, BN=256, BK=64,
            GROUP=16, num_warps=8, num_stages=2)
    return kernel
