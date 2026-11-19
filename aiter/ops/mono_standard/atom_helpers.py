# SPDX-License-Identifier: MIT
import torch
import triton
import triton.language as tl

# These MONO kernels target gfx1250, whose FP8 encoding is OCP E4M3FN.
# Bind its compile-time constants directly: the generic AITER dtype falls back
# to uint8 when no GPU is visible, which is not a floating-point finfo input.
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = tl.constexpr(torch.finfo(FP8_DTYPE).max)
FP8_TL_DTYPE = tl.constexpr(tl.float8e4nv)
# The amax floor of every FP8 activation quant: aiter's MoE input quant's, so
# one model's quant sites all agree
AMAX_FLOOR = tl.constexpr(1e-10)


@triton.jit
def ceil_pow2_code(value):
    bits = value.to(tl.uint32, bitcast=True)
    return (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)


@triton.jit
def window_row(
    slot,
    pos,
    ring_start,
    RING_SLOTS: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    RING_STRIDE: tl.constexpr,
    RUN_ROWS: tl.constexpr,
):
    """Row of absolute position `pos` in physical slot `slot`'s window.

    Relative to the calling layer's own view base, and layer-independent
    within a compress class — which is what lets one index buffer serve the
    whole class. See `UnifiedPoolGeometry.window_params`.

    `pos` is the absolute token position; `pos % RING_SLOTS` is the ring
    position. Triton follows C remainder semantics, so a negative `pos` yields
    a negative row here rather than wrapping — callers must not pass the -1
    padding sentinels.
    """
    ring_pos = pos % RING_SLOTS
    return (
        slot * SLOT_ROWS
        + ring_start
        + (ring_pos // RING_STRIDE) * RUN_ROWS
        + (ring_pos % RING_STRIDE)
    )
