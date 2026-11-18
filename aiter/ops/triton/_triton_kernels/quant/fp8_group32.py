# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Based on ROCm/ATOM 10b99a085ba46e5b3a125f0114cec7c71e401552,
# atom/model_ops/blockscale_kernels/quantization.py (MIT).

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit(
    repr=make_kernel_repr("_quantize_fp8_group32_kernel", ["PERSISTENT", "DEQUANT"])
)
def _quantize_fp8_group32_kernel(
    X, Y, S, SIZE, DEQUANT: tl.constexpr, PERSISTENT: tl.constexpr
):
    task = tl.program_id(0)
    if PERSISTENT:
        task += 16 * tl.program_id(1)
        stride = 256
    else:
        stride = tl.num_programs(0)
    for tile in range(task, tl.cdiv(SIZE, 1024), stride):
        groups = tile * 32 + tl.arange(0, 32)
        offsets = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
        x = tl.load(X + offsets, offsets < SIZE, other=0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x), 1), 1e-10)
        bits = (amax * (1.0 / 448.0)).to(tl.uint32, bitcast=True)
        code = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
        scale = (code << 23).to(tl.float32, bitcast=True)
        q = tl.minimum(tl.maximum(x / scale[:, None], -448.0), 448.0).to(tl.float8e4nv)
        if DEQUANT:
            tl.store(Y + offsets, q.to(tl.float32) * scale[:, None], offsets < SIZE)
        else:
            tl.store(Y + offsets, q, offsets < SIZE)
            tl.store(S + groups, code.to(tl.uint8), groups < SIZE // 32)
