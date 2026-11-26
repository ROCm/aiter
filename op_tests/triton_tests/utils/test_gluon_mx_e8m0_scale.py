# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math
import struct

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.common.utils import mx_e8m0_scale

N = 64


@gluon.jit
def _mx_e8m0_scale_kernel(
    x_ptr, out_ptr, N: gl.constexpr, EXP_OFFSET: gl.constexpr, THREADS: gl.constexpr
):
    layout: gl.constexpr = gl.BlockedLayout([1], [THREADS], [1], [0])
    offs = gl.arange(0, N, layout=layout)
    gl.store(out_ptr + offs, mx_e8m0_scale(gl.load(x_ptr + offs), EXP_OFFSET))


def _f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _edge_bits():
    bits = [0x00000000, 0x80000000]  # 0, -0
    bits += [0x00000001, 0x80000001]  # smallest denormal, its negative
    bits += [0x005FFFFF, 0x00600000, 0x007FFFFF]  # 0.75 x 2^-126, largest denormal
    bits += [0x00800000]  # 2^-126
    for k in (-126, -1, 0, 8, 126, 127):
        b = _bits(1.75 * 2.0**k)
        bits += [b - 1, b, b + 1]  # just below, at, just above 1.75 x 2^k
    bits += [_bits(1.0), _bits(6.0), _bits(-1.0)]
    bits += [0x7F7FFFFF, 0x7F800000, 0xFF800000]  # largest finite value, inf, -inf
    # NaNs: the rounding add keeps some at 255 and wraps others to an exponent of 0.
    bits += [0x7F800001, 0x7FC00000, 0x7FDFFFFF, 0x7FE00000, 0x7FFFFFFF]
    bits += [0xFFC00000, 0xFFE00000, 0xFFFFFFFF]
    return bits


def _reference(bits, exp_offset):
    """|value| rounds to a power of two 2^p: up from 1.75 x 2^k. An fp32 denormal
    rounds to 2^-126 from 0.75 x 2^-126, else to 0. The byte is p + 127 - exp_offset,
    at least 0. Inf, values that round to 2^128 and negative values that do not
    round to 0 give 254. A NaN gives 0 when bits + 0x200000 (mod 2^32) has an
    exponent field of 0, else 254."""
    value = _f32(bits)
    if math.isnan(value):
        return 0 if (bits + 0x200000) & 0x7F800000 == 0 else 254
    if math.isinf(value):
        return 254
    a = abs(value)
    if a < 2.0**-126:
        if a < 0.75 * 2.0**-126:
            return 0
        p = -126
    else:
        mant, exp = math.frexp(a)  # a = mant * 2^exp, mant in [0.5, 1)
        p = exp - 1 + (mant >= 0.875)
    if value < 0 or p == 128:
        return 254
    return max(p + 127 - exp_offset, 0)


@pytest.mark.parametrize("exp_offset", [2, 8])
def test_mx_e8m0_scale_edge_values(exp_offset):
    bits = _edge_bits()
    bits += [0] * (N - len(bits))
    signed = [b - (1 << 32) if b >= 1 << 31 else b for b in bits]
    x = torch.tensor(signed, dtype=torch.int32, device="cuda").view(torch.float32)
    out = torch.empty((N,), dtype=torch.uint8, device="cuda")
    threads = triton.runtime.driver.active.get_current_target().warp_size
    _mx_e8m0_scale_kernel[(1,)](
        x, out, N=N, EXP_OFFSET=exp_offset, THREADS=threads, num_warps=1
    )
    expected = [_reference(b, exp_offset) for b in bits]
    bad = [
        (hex(b), got, ref)
        for b, got, ref in zip(bits, out.tolist(), expected)
        if got != ref
    ]
    assert not bad, f"(input bits, kernel, reference): {bad}"
