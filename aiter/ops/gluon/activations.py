# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl


@gluon.jit
def _sigmoid(x):
    # Gluon doesn't export sigmoid directly, so we implement it using exp
    return 1.0 / (1.0 + ttgl.exp(-x))

@gluon.jit
def _silu(x):
    return x * _sigmoid(x)


@gluon.jit
def _silu_exp2(x):
    return x / (1.0 + ttgl.exp2(-(x * 1.44269504089)))

@gluon.jit
def _tanh(x):
    return 2.0 * _sigmoid(2.0 * x) - 1.0

@gluon.jit
def _gelu(x):
    M_SQRT1_2 = 0.70710678118654752440
    ALPHA = M_SQRT1_2
    return 0.5 * x * (1.0 + ttgl.erf(x * ALPHA))


@gluon.jit
def _gelu_tanh(x):
    M_SQRT2 = 1.41421356237309504880
    M_2_SQRTPI = 1.12837916709551257390
    BETA = M_SQRT2 * M_2_SQRTPI * 0.5
    KAPPA = 0.044715
    x_cube = x * x * x
    inner = BETA * (x + KAPPA * x_cube)
    return 0.5 * x * (1.0 + _tanh(inner))


@gluon.jit
def _relu(x):
    return ttgl.maximum(0.0, x)


def _get_activation_from_str(activation: str):
    mapping = {
        "gelu": _gelu,
        "gelu_tanh": _gelu_tanh,
        "silu": _silu,
        "silu_exp2": _silu_exp2,
        "relu": _relu,
    }
    return mapping.get(activation)