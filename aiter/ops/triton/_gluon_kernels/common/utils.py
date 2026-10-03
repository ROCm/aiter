# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import PropagateNan
from triton.language.extra.hip import libdevice

_MAX_PROPAGATE_NAN_ALL = gl.constexpr(PropagateNan.ALL)


@gluon.jit
def elementwise_max_prop_nan(a, b):
    return gl.maximum(a, b, propagate_nan=_MAX_PROPAGATE_NAN_ALL)


@gluon.jit
def reduce_max_prop_nan(input, axis=None, keep_dims=False):
    return gl.reduce(input, axis, elementwise_max_prop_nan, keep_dims=keep_dims)


@gluon.jit
def exp_scaled(scale, x):
    return gl.exp2((scale * math.log2(math.e)) * x)


@gluon.jit
def softplus(x):
    return gl.where(x < 20.0, gl.log(1.0 + gl.exp(x)), x)


@gluon.jit
def sigmoid(x):
    return 0.5 + 0.5 * libdevice.tanh(0.5 * x)
