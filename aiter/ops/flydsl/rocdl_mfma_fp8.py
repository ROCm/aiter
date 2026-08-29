# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CDNA4 (gfx950) scaled-MFMA fp8 helpers for FlyDSL kernels.

This module wraps the ``mfma_scale_f32_32x32x64_f8f6f4`` MFMA instruction
(fp8 E4M3 inputs, f32 accumulator) exposed by FlyDSL as
``fx.rocdl.cdna4.MFMA_Scale(32, 32, 64, fx.Float8E4M3FN)``.

It mirrors the ``Mfma16x16x128`` API in
``flydsl/kernels/fp8_gemm_utils.py`` (``zero_value``, ``accum_type``,
``call``) but for the 32x32x64 atom, plus pure-Python helpers that
describe / build the exact per-lane fragment layouts so host code can
pack A/B and unpack C correctly.

Exact fragment layouts (verified empirically on gfx950, see
``op_tests/test_flydsl_mfma_fp8_32x32x64.py``).  Lane ``L`` in
``[0, 64)`` decomposes as ``lo = L % 32`` (0..31) and
``hi = L // 32`` (0 or 1).

A tile is ``M=32 x K=64`` fp8, 32 fp8 / lane (vec<8xi32>):
    A_frag[L][v] = A[row = lo, col = hi * 32 + v]      v in [0, 32)

B tile is ``K=64 x N=32`` fp8 (K-major / ``KxN``), 32 fp8 / lane
(vec<8xi32>); same lane/value structure as A with K playing K's role:
    B_frag[L][v] = B[row(K) = hi * 32 + v, col(N) = lo]   v in [0, 32)

C tile is ``M=32 x N=32`` f32, 16 f32 / lane (vec<16xf32>):
    C_frag[L][v] = C[row = hi * 4 + (v % 4) + 8 * (v // 4), col = lo]

All three maps verified to bit-exact agreement on gfx950 (cosine 1.0)
by ``op_tests/test_flydsl_mfma_fp8_32x32x64.py``.

The default ``scale_a`` / ``scale_b`` operands are zero, which the
hardware treats as the identity scale (exponent bias 0), so a plain
matmul validates with the atom's default scales.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl.expr.typing import Vector as Vec

M = 32
N = 32
K = 64
NUM_LANES = 64
A_FP8_PER_LANE = 32  # vec<8 x i32> == 32 fp8
B_FP8_PER_LANE = 32  # vec<8 x i32> == 32 fp8
C_F32_PER_LANE = 16  # vec<16 x f32>

# 16x16x128 variant.  Same vec<8xi32> A/B operands, but only 4 f32 per lane
# of accumulator instead of 16.
C16_F32_PER_LANE = 4


class Mfma32x32x64:
    """One CDNA4 ``mfma_scale_f32_32x32x64_f8f6f4`` per ``call``.

    A is ``32x64`` fp8, B is ``64x32`` fp8 (K-major / ``KxN``), C is
    ``32x32`` f32.  ``a`` and ``b`` are each a ``vec<8xi32>`` (32 fp8)
    per lane; ``c`` is a ``vec<16xf32>`` per lane.
    """

    def __init__(self, elem_ty=fx.Float8E4M3FN):
        self.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(M, N, K, elem_ty))
        self.accum_type = Vec.make_type(C_F32_PER_LANE, fx.Float32)
        self.zero_value = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)

    def call(self, a, b, c):
        """Run one MFMA and return the updated ``vec<16xf32>`` accumulator."""
        return fly_dialect.mma_atom_call_ssa([self.accum_type], self.atom, a, b, c)


class Mfma16x16x128:
    """One CDNA4 ``mfma_scale_f32_16x16x128_f8f6f4`` per ``call``.

    A/B operands are the same shape as :class:`Mfma32x32x64` -- a
    ``vec<8xi32>`` (32 fp8) per lane -- but C is only a ``vec<4xf32>``,
    a quarter of the accumulator registers.

    The two atoms differ in which lanes fold into one output column:

        32x32x64  : column == lane % 32  -> sums lanes {n, n+32}
        16x16x128 : column == lane % 16  -> sums lanes {n, n+16, n+32, n+48}

    (Both verified on gfx950 by driving a ones-column A against an
    identical B fragment and reading C back lane-major.)

    That makes the 16x16x128 atom a drop-in replacement for a ones-column
    *reduction* over a 32-lane-period fragment only if A is masked so that
    each of the four 16-lane groups contributes a distinct output row.
    Feeding A = 1.0 exactly on the lanes where ``lane % 32`` is one of
    ``ONES_ROW_LANES`` and 0 elsewhere makes ``C[lane][0]`` bit-identical
    to the ``vec<16xf32>`` atom's ``C[lane][0]`` on all 64 lanes.  The
    other three C elements stay zero, so the readout is still element 0.
    """

    def __init__(self, elem_ty=fx.Float8E4M3FN):
        self.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, elem_ty))
        self.accum_type = Vec.make_type(C16_F32_PER_LANE, fx.Float32)
        self.zero_value = Vec.filled(C16_F32_PER_LANE, 0.0, fx.Float32)

    def call(self, a, b, c):
        """Run one MFMA and return the updated ``vec<4xf32>`` accumulator."""
        return fly_dialect.mma_atom_call_ssa([self.accum_type], self.atom, a, b, c)


# Lanes (mod 32) that carry the ones column for a 16x16x128 reduction; see
# ``Mfma16x16x128``.  Matches the assembly kernel's own ones-operand mask.
ONES_ROW_LANES = (0, 8, 20, 28)


# ---------------------------------------------------------------------------
# Pure-Python fragment <-> tile coordinate maps (host-side packing helpers).
# ---------------------------------------------------------------------------


def a_frag_coord(lane, v):
    """(row, col) in the 32x64 A tile for A_frag[lane][v], v in [0,32)."""
    lo = lane % 32
    hi = lane // 32
    return lo, hi * 32 + v


def b_frag_coord(lane, v):
    """(row(K), col(N)) in the 64x32 B tile for B_frag[lane][v], v in [0,32).

    B is K-major (``KxN``).  Same lane/value structure as A.
    """
    lo = lane % 32
    hi = lane // 32
    return hi * 32 + v, lo


def c_frag_coord(lane, v):
    """(row, col) in the 32x32 C tile for C_frag[lane][v], v in [0,16)."""
    lo = lane % 32
    hi = lane // 32
    return hi * 4 + (v % 4) + 8 * (v // 4), lo


def pack_a(a_tile):
    """Pack a (32, 64) fp8/np array into a (64, 32) lane-major fp8 fragment.

    ``a_tile`` is indexed ``[row, col]``; result is ``[lane, v]``.
    """
    import numpy as np

    out = np.empty((NUM_LANES, A_FP8_PER_LANE), dtype=a_tile.dtype)
    for lane in range(NUM_LANES):
        for v in range(A_FP8_PER_LANE):
            r, c = a_frag_coord(lane, v)
            out[lane, v] = a_tile[r, c]
    return out


def pack_b(b_tile):
    """Pack a (64, 32) fp8/np array into a (64, 32) lane-major fp8 fragment.

    ``b_tile`` is indexed ``[row(K), col(N)]``; result is ``[lane, v]``.
    """
    import numpy as np

    out = np.empty((NUM_LANES, B_FP8_PER_LANE), dtype=b_tile.dtype)
    for lane in range(NUM_LANES):
        for v in range(B_FP8_PER_LANE):
            r, c = b_frag_coord(lane, v)
            out[lane, v] = b_tile[r, c]
    return out


def unpack_c(c_frag):
    """Unpack a (64, 16) f32 lane-major C fragment into a (32, 32) tile."""
    import numpy as np

    out = np.empty((M, N), dtype=c_frag.dtype)
    for lane in range(NUM_LANES):
        for v in range(C_F32_PER_LANE):
            r, c = c_frag_coord(lane, v)
            out[r, c] = c_frag[lane, v]
    return out
