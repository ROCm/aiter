# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Based on ROCm/ATOM 10b99a085ba46e5b3a125f0114cec7c71e401552,
# atom/model_ops/blockscale_kernels/quantization.py (MIT).

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna3 import buffer_load, buffer_store

from aiter.ops.triton._gluon_kernels.gfx1250.quant.mono_memory import (
    load_dev,
    store_dev_ht,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.jit
def _quantize_fp8_group32_tile(
    X,
    Y,
    S,
    SIZE,
    tile,
    DEQUANT: gl.constexpr,
    TILES_PER_ITER: gl.constexpr = 1,
    TILE_STRIDE: gl.constexpr = 1,
    MONO_MEMORY: gl.constexpr = False,
):
    # Keep each scale in its reduction layout to avoid an LDS copy across waves.
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [4, 1], [1, 0])
    groups = gl.arange(0, 32 * TILES_PER_ITER, layout=gl.SliceLayout(1, layout))
    groups = (groups // 32) * (32 * TILE_STRIDE) + groups % 32
    offsets = (
        groups[:, None] * 32
        + gl.arange(0, 32, layout=gl.SliceLayout(0, layout))[None, :]
    )
    # Buffer offsets are 32-bit; rebase each tile with 64-bit pointer arithmetic.
    tile_base = tile.to(gl.int64) * 1024
    scale_base = tile.to(gl.int64) * 32
    mask = tile_base + offsets < SIZE
    if MONO_MEMORY:
        x = load_dev(X + tile_base, offsets, mask).to(gl.float32)
    else:
        x = buffer_load(X + tile_base, offsets, mask=mask, other=0).to(gl.float32)
    amax = gl.maximum(gl.max(gl.abs(x), 1), 1e-10)
    bits = (amax * (1.0 / 448.0)).to(gl.uint32, bitcast=True)
    code = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(gl.uint32)
    scale = (code << 23).to(gl.float32, bitcast=True)
    # These power-of-two reciprocals are exact; keep the Inf and -0 scale cases.
    reciprocal_code = gl.maximum(254 - code.to(gl.int32), 0)
    reciprocal_bits = gl.where(code == 256, -8388608, reciprocal_code << 23)
    reciprocal_scale = reciprocal_bits.to(gl.float32, bitcast=True)
    q = gl.minimum(gl.maximum(x * reciprocal_scale[:, None], -448.0), 448.0).to(
        gl.float8e4nv
    )
    if DEQUANT:
        y = (q.to(gl.float32) * scale[:, None]).to(Y.dtype.element_ty)
        if MONO_MEMORY:
            store_dev_ht(Y + tile_base, offsets, y, mask)
        else:
            buffer_store(y, Y + tile_base, offsets, mask=mask)
    else:
        if MONO_MEMORY:
            store_dev_ht(Y + tile_base, offsets, q, mask)
            store_dev_ht(
                S + scale_base,
                groups,
                code.to(gl.uint8),
                scale_base + groups < SIZE // 32,
            )
        else:
            buffer_store(q, Y + tile_base, offsets, mask=mask)
            buffer_store(
                code.to(gl.uint8),
                S + scale_base,
                groups,
                mask=scale_base + groups < SIZE // 32,
            )


@gluon.jit(
    repr=make_kernel_repr(
        "_quantize_fp8_group32_kernel", ["PERSISTENT", "DEQUANT", "TILES_PER_ITER"]
    )
)
def _quantize_fp8_group32_kernel(
    X,
    Y,
    S,
    SIZE,
    DEQUANT: gl.constexpr,
    PERSISTENT: gl.constexpr,
    TILES_PER_ITER: gl.constexpr = 1,
):
    task = gl.program_id(0)
    if PERSISTENT:
        task = gl.program_id(1)
        for tile in range(task, gl.cdiv(SIZE, 1024), 256 * TILES_PER_ITER):
            _quantize_fp8_group32_tile(
                X,
                Y,
                S,
                SIZE,
                tile,
                DEQUANT,
                TILES_PER_ITER,
                TILE_STRIDE=256,
                MONO_MEMORY=True,
            )
    else:
        _quantize_fp8_group32_tile(X, Y, S, SIZE, task, DEQUANT)
