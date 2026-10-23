# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.extra.hip import libdevice

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
