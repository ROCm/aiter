# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.extra.hip import libdevice

from aiter.ops.triton._triton_kernels.rope.rope import _get_gptj_rotated_x

# Triton main renamed shared_memory_descriptor._reinterpret to reinterpret
# (triton-lang/triton#11131); older Tritons, such as the ROCm 3.8 wheel, have
# only _reinterpret. Kernels call reinterpret; give older Tritons that name too.
if not hasattr(gl.shared_memory_descriptor, "reinterpret"):
    gl.shared_memory_descriptor.reinterpret = gl.shared_memory_descriptor._reinterpret


@gluon.jit
def exp_scaled(scale, x):
    return gl.exp2((scale * math.log2(math.e)) * x)


@gluon.jit
def softplus(x):
    return gl.where(x < 20.0, gl.log(1.0 + gl.exp(x)), x)


@gluon.jit
def sigmoid(x):
    return 0.5 + 0.5 * libdevice.tanh(0.5 * x)


@gluon.jit
def sigmoid_log2(z):
    """sigmoid(x) given ``z = -x * log2(e)``."""
    return gl.extra.libdevice.fast_dividef(1.0, 1.0 + gl.exp2(z))


@gluon.jit
def fast_sigmoid(x):
    return sigmoid_log2(x.to(gl.float32) * -math.log2(math.e))


@gluon.jit
def sum_combine(a, b):
    return a + b


@gluon.jit
def l2norm(x):
    f = x.to(gl.float32)
    return f * gl.rsqrt(gl.sum(f * f, axis=1) + 1e-6)[:, None]


@gluon.jit
def mx_e8m0_scale(amax, EXP_OFFSET: gl.constexpr):
    """Biased e8m0 scale byte from fp32 block amax; EXP_OFFSET is 2 for e2m1, 8 for e4m3."""
    amax = amax.to(gl.int32, bitcast=True)
    amax = (amax + 0x200000).to(gl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(gl.float32, bitcast=True)
    scale_unbiased = gl.log2(amax).floor() - EXP_OFFSET
    scale_unbiased = gl.maximum(-127, gl.minimum(scale_unbiased, 127))
    return (scale_unbiased.to(gl.int32) + 127).to(gl.uint8)


@gluon.jit
def load_interleaved_cos_sin(
    cos_ptr,
    sin_ptr,
    pos,
    NUM_COLS: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    layout: gl.constexpr,
):
    """cos and sin of interleaved RoPE for a [1, NUM_COLS] row. RoPE is on the last
    ROPE_DIM columns. Each adjacent column pair shares one table value. Other columns
    get 0."""
    gl.static_assert(NUM_COLS % ROPE_DIM == 0)
    col = gl.arange(0, NUM_COLS, layout=gl.SliceLayout(0, layout))[None, :]
    is_rope = col >= NUM_COLS - ROPE_DIM
    # The RoPE part starts at a multiple of ROPE_DIM, so col % ROPE_DIM is the column
    # in the RoPE part. Columns 2i and 2i + 1 use table entry i.
    offsets = pos * (ROPE_DIM // 2) + (col % ROPE_DIM) // 2
    cos = gl.amd.gfx1250.buffer_load(cos_ptr, offsets, is_rope, 0.0).to(gl.float32)
    sin = gl.amd.gfx1250.buffer_load(sin_ptr, offsets, is_rope, 0.0).to(gl.float32)
    return cos, sin


@gluon.jit
def interleaved_rope(x, cos, sin):
    """Interleaved RoPE on a 2D tile: each adjacent pair (x[2i], x[2i+1]) rotates by
    one angle. cos and sin have one value for each column."""
    NUM_ROWS: gl.constexpr = x.shape[0]
    NUM_COLS: gl.constexpr = x.shape[1]
    col = gl.arange(0, NUM_COLS, layout=gl.SliceLayout(0, x.type.layout))[None, :]
    is_even = col % 2 == 0
    # rotated[2i] = -x[2i + 1] and rotated[2i + 1] = x[2i].
    rotated = _get_gptj_rotated_x(x, is_even, NUM_ROWS, NUM_COLS, NUM_COLS // 2)
    # The helper reshapes x. convert_layout gives rotated the layout of x again.
    return x * cos + gl.convert_layout(rotated, x.type.layout) * sin


@gluon.constexpr_function
def cluster_shape(num_ctas, BLOCK_M, BLOCK_N, split="auto"):
    """Split ``num_ctas`` CTAs over a (BLOCK_M, BLOCK_N) cluster tile.

    The split decides which operand the cluster shares, which matters more than
    tile squareness. Splitting M gives the CTAs different rows and the same
    columns, so they share the B/W tile; splitting N shares the A/X tile.

    ``split="auto"`` halves whichever per-CTA dimension is larger, so each CTA
    gets as square a share as the tile allows. Note this picks N whenever
    BLOCK_N >= BLOCK_M, which is the usual case, so B/W sharing is unreachable
    unless it is asked for. ``split="m"`` or ``"n"`` forces the direction.
    """
    if num_ctas < 1 or num_ctas & (num_ctas - 1):
        raise ValueError(f"num_ctas must be a power of 2, got {num_ctas}")
    if split == "m":
        return num_ctas, 1
    if split == "n":
        return 1, num_ctas
    if split != "auto":
        raise ValueError(f"split must be 'auto', 'm' or 'n', got {split!r}")
    ctas_m, ctas_n = 1, 1
    while ctas_m * ctas_n < num_ctas:
        if BLOCK_N // ctas_n > BLOCK_M // ctas_m:
            ctas_n *= 2
        else:
            ctas_m *= 2
    return ctas_m, ctas_n


@gluon.jit
def consume_k_block(
    acc,
    a_slot,
    b_slot,
    DOT_LAYOUT_A: gl.constexpr,
    DOT_LAYOUT_B: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_SUBTILES: gl.constexpr,
):
    """Accumulate one K block on gfx1250, reading LDS one subtile ahead of the dot.

    ``a_slot`` is (BLOCK_M, BLOCK_K) and ``b_slot`` is (BLOCK_K, BLOCK_N), both
    already in the WMMA operand orientation. With NUM_SUBTILES == 1 this is a
    single read then a single wmma, so the dot waits on the whole LDS read.
    Splitting K lets each read overlap the previous subtile's wmma, which is
    what keeps the WMMA pipe fed.
    """
    SUB: gl.constexpr = BLOCK_K // NUM_SUBTILES
    gl.static_assert(
        SUB % 32 == 0, "BLOCK_K // NUM_SUBTILES must be a multiple of the WMMA K step"
    )
    a_cur = a_slot.slice(0, SUB, dim=1).load(layout=DOT_LAYOUT_A)
    b_cur = b_slot.slice(0, SUB, dim=0).load(layout=DOT_LAYOUT_B)
    for j in gl.static_range(NUM_SUBTILES - 1):
        a_nxt = a_slot.slice((j + 1) * SUB, SUB, dim=1).load(layout=DOT_LAYOUT_A)
        b_nxt = b_slot.slice((j + 1) * SUB, SUB, dim=0).load(layout=DOT_LAYOUT_B)
        acc = gl.amd.gfx1250.wmma(a_cur, b_cur, acc)
        a_cur = a_nxt
        b_cur = b_nxt
    return gl.amd.gfx1250.wmma(a_cur, b_cur, acc)
