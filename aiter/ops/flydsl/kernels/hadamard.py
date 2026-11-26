# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Register-resident block Hadamard rotation for GEMM epilogues."""

import math

import flydsl.expr as fx
from flydsl.expr import range_constexpr

from .dpp_utils import dpp_xor_f32


def _lane_xor(v, dist):
    if dist < 16:
        return fx.Float32(dpp_xor_f32(v, dist))
    return v.shuffle_xor(fx.Int32(dist), fx.Int32(64))


def block_hadamard_f32(vals, lane, lane_bits):
    """Multiply each row block by the normalized Sylvester Hadamard matrix.

    One block spans ``2**lane_bits`` consecutive lanes of a wavefront, each
    holding ``len(vals)`` consecutive columns, so the block width is
    ``len(vals) << lane_bits`` and column ``c`` of the block sits in element
    ``c % len(vals)`` of lane ``c // len(vals)`` (mod the block).
    """
    vals = list(vals)
    n = len(vals)
    assert n & (n - 1) == 0, f"values per lane must be a power of two, got {n}"
    h = 1
    while h < n:
        for i in range_constexpr(n):
            if not i & h:
                a, b = vals[i], vals[i | h]
                vals[i], vals[i | h] = a + b, a - b
        h *= 2
    for s in range_constexpr(lane_bits):
        dist = 1 << s
        upper = (lane & fx.Int32(dist)) != fx.Int32(0)
        for i in range_constexpr(n):
            peer = _lane_xor(vals[i], dist)
            vals[i] = upper.select(peer - vals[i], vals[i] + peer)
    scale = fx.Float32(1.0 / math.sqrt(n << lane_bits))
    return [v * scale for v in vals]
