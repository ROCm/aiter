# SPDX-License-Identifier: MIT
"""CGA-aware layout derivation for gfx1250 Gluon kernels.

On gfx1250 a tile is computed by a cluster of ctas_per_cga[0] x ctas_per_cga[1] CTAs, so every
layout carries a block (CGA) component. These builders derive that component instead of
hand-rolling basis vectors per tile size.
"""

from __future__ import annotations

import triton.experimental.gluon.language as gl
from triton.runtime.jit import constexpr_function

#: The TDM descriptor holds log2(pad interval in dwords) in 3 bits, so the interval is at most
#: 256 dwords. An interval above the cap fails the pass pipeline with no useful diagnostic.
TDM_PAD_INTERVAL_MAX = 256

#: An LDS line, in 16-bit elements. A transposed A pads once per line instead of once per
#: per-CTA M row: a narrow row would otherwise pad every 32 elements, which costs more LDS
#: than the tile itself saves. B cannot use this, because a C stage that aliases the B ring
#: needs an interval equal to the per-CTA N.
LDS_LINE_ELEMS = 256 * 8 // 16

WMMA_INSTR_SHAPE = [16, 16, 32]
WMMA_VERSION = 3

#: Elements each lane reads per dot operand step.
K_WIDTH = 8


@constexpr_function
def plan_warps(BM, BN, num_warps):
    """Factor num_warps into [wpcM, wpcN] over a per-CTA tile.

    Doubles the dimension with fewer warps, but only while it still holds more than one WMMA
    sub-tile. Surplus warps replicate when both dimensions are exhausted.
    """
    wpcM, wpcN = 1, 1
    while wpcM * wpcN < num_warps:
        can_m = BM // (WMMA_INSTR_SHAPE[0] * wpcM) > 1
        can_n = BN // (WMMA_INSTR_SHAPE[1] * wpcN) > 1
        if can_m and can_n:
            if wpcM <= wpcN:
                wpcM *= 2
            else:
                wpcN *= 2
        elif can_m:
            wpcM *= 2
        elif can_n:
            wpcN *= 2
        else:
            # Both dimensions hold one WMMA sub-tile. Replicate the remaining
            # warps along M. The layout must carry num_warps warps: a layout
            # with fewer is rejected ("Layout has 2 warps per CTA, but the
            # context requires 4"), which made 16x16, 16x32 and 32x16 per-CTA
            # tiles fail to compile at num_warps=4.
            wpcM *= 2
    return [wpcM, wpcN]


@constexpr_function
def cga_bases(ctas_m, ctas_n):
    """Pure-python ``make_cga_layout([m, n], [m, n], [0, 1])``.

    Deliberately not a call into libtriton's ``make_cga_layout``: gluon's dependency walker
    inspects the globals of every function a kernel body references and rejects nanobind
    builtins outright ("Unsupported function referenced"), so even a constexpr_function wrapper
    around it cannot be used from inside a kernel. Equivalence is asserted in the op test.
    """
    bases = []
    for dim, ctas in ((0, ctas_m), (1, ctas_n)):
        bit = 1
        while bit < ctas:
            basis = [0, 0]
            basis[dim] = bit
            bases.append(basis)
            bit *= 2
    return bases


@constexpr_function
def warp_bases_from_grid(wpc):
    """Warp basis vectors for a [wpcM, wpcN] grid, N bases first then M bases."""
    bases = []
    step = 1
    while step < wpc[1]:
        bases.append([0, step])
        step *= 2
    step = 1
    while step < wpc[0]:
        bases.append([step, 0])
        step *= 2
    return bases


@constexpr_function
def wmma_layout(shape_per_cta, ctas_per_cga, num_warps):
    """Accumulator layout for a per-CTA tile. The accumulator splits across the whole cluster."""
    wpc = plan_warps(shape_per_cta[0], shape_per_cta[1], num_warps)
    cga = cga_bases(ctas_per_cga[0], ctas_per_cga[1])
    return gl.amd.AMDWMMALayout(
        WMMA_VERSION,
        True,
        warp_bases_from_grid(wpc),
        [],
        WMMA_INSTR_SHAPE,
        cga,
    )


@constexpr_function
def operand_bundle_cga(
    wmma, op_idx, shape_per_cta, ctas_per_cga, transpose_b=False, transpose_a=False
):
    """Derive (dot_operand, padded_shared) for one WMMA operand.

    shape_per_cta is the per-CTA LDS tile. For A it is [BM, BK], or [BK, BM] when
    transpose_a is set; for B it is [BK, BN], or [BN, BK] when transpose_b is set. Do not
    pass the kernel's BLOCK_M/BLOCK_N constexprs, because those are already CGA-wide.

    Each operand has two orientations, selected by the layout of the input tensor:
      K innermost  (a K-contiguous input): interval BLOCK_K, pad 8.
      M/N innermost (an M- or N-contiguous input): interval min(extent, cap), pad 16.
    The second form costs more LDS whenever the innermost extent is below 2 * BLOCK_K, by
    16 * BLOCK_K - 8 * extent elements. The orientation is not free: a C stage that aliases
    the B ring needs an N-oriented interval equal to the per-CTA innermost extent.
    """
    dot = gl.DotOperandLayout(op_idx, wmma, K_WIDTH)
    cga_op = dot.cga_layout

    # with_identity_for needs the CGA-wide shape, but the interval needs the per-CTA extent.
    if op_idx == 0 and transpose_a:  # A transposed: [BK, BM], M innermost
        shape_cga = [shape_per_cta[0], shape_per_cta[1] * ctas_per_cga[0]]
        # Pad once per LDS line, not once per per-CTA M row. A 32-wide row padded by 16
        # every row costs 50% of the tile; one pad of 8 per 128 elements costs 6%.
        interval = min(max(shape_per_cta[1], LDS_LINE_ELEMS), TDM_PAD_INTERVAL_MAX)
        # A's cga_layout splits M, which is dim 0 in [BM, BK]. Transposing moves M to dim 1,
        # so the rows must be swapped too. Passing it unswapped splits K across the cluster
        # and the backend rejects it with "does not support cross-CTA shared memory
        # transfers".
        cga_t = tuple((row[1], row[0]) for row in cga_op)
        shared = gl.PaddedSharedLayout.with_identity_for(
            [[interval, 8]], shape_cga, [1, 0], cga_t
        )
    elif op_idx == 0:  # A: [BM, BK], K innermost
        shape_cga = [shape_per_cta[0] * ctas_per_cga[0], shape_per_cta[1]]
        interval = shape_per_cta[1]  # K does not split across the cluster
        shared = gl.PaddedSharedLayout.with_identity_for(
            [[interval, 8]], shape_cga, [1, 0], cga_op
        )
    elif transpose_b:  # B transposed: [BN, BK], K innermost
        shape_cga = [shape_per_cta[0] * ctas_per_cga[1], shape_per_cta[1]]
        interval = shape_per_cta[1]
        cga_t = tuple((row[1], row[0]) for row in cga_op)
        shared = gl.PaddedSharedLayout.with_identity_for(
            [[interval, 8]], shape_cga, [1, 0], cga_t
        )
    else:  # B native: [BK, BN], N innermost
        shape_cga = [shape_per_cta[0], shape_per_cta[1] * ctas_per_cga[1]]
        interval = min(shape_per_cta[1], TDM_PAD_INTERVAL_MAX)
        shared = gl.PaddedSharedLayout.with_identity_for(
            [[interval, 16]], shape_cga, [1, 0], cga_op
        )

    return dot, shared
