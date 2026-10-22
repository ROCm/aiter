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
