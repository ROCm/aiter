# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.fused_indices_gather import (
    _fused_indices_and_gather_kernel
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

def fuse_indices_and_gather(
    x, E, a, D, BLOCK_M=16, BLOCK_N=64, num_warps=4, num_stages=2
):
    x2d = x.view(-1, D)
    idx2d = torch.empty(E * a, D, dtype=torch.long, device=x.device)
    gather_out = torch.empty_like(idx2d, device=x.device, dtype=x.dtype)

    # compute strides for accessing elems in Triton
    sx2d0 = x2d.stride(0)
    sx2d1 = x2d.stride(1)
    sidx2d0 = idx2d.stride(0)
    sidx2d1 = idx2d.stride(1)
    sgatherout0 = gather_out.stride(0)
    sgatherout1 = gather_out.stride(1)

    grid = (triton.cdiv(E * a, BLOCK_M), triton.cdiv(D, BLOCK_N))

    _fused_indices_and_gather_kernel[grid](
        x2d,
        gather_out,
        idx2d,
        E,
        a,
        D,
        sx2d0,
        sx2d1,
        sgatherout0,
        sgatherout1,
        sidx2d0,
        sidx2d1,
        BLOCK_M,
        BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return gather_out, idx2d