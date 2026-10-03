# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single-workgroup global-to-local expert scan and valid-route count."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n", "E", "topk"])
def _g2l_lut_kernel(
    Mask,
    Nvt,
    Lut,
    Counter,
    Nvr,
    n,
    E,
    topk,
    STRIDE: tl.constexpr,
    CLEAR_COUNTER: tl.constexpr,
    WRITE_NVR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    value = tl.load(Mask + i * STRIDE, mask=i < n, other=0)
    # Compare before casting, preserving nonzero int64 / fractional masks.
    enabled = (value != 0).to(tl.int32)
    prefix = tl.cumsum(enabled, axis=0)
    lut = tl.where(enabled != 0, prefix - 1, E).to(tl.int32)
    tl.store(Lut + i, lut, mask=i < n)

    if WRITE_NVR:
        # Read the device scalar on every execution, including graph replay.
        nvt = tl.load(Nvt).to(tl.int32)
        tl.store(Nvr, nvt * topk)

    if CLEAR_COUNTER:
        tl.store(Counter + i, tl.full((BLOCK,), 0, tl.int32), mask=i < E)
