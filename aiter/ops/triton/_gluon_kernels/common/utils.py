# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.extra.hip import libdevice


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
