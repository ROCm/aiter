# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Activation Functions for Gluon Kernels
======================================
These are applied element-wise to the accumulator before storing results.
Using gluon.jit to ensure compatibility with gluon kernels.

Note: gluon doesn't export sigmoid directly, so we implement it using exp:
    sigmoid(x) = 1 / (1 + exp(-x))
"""

from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl


@gluon.jit
def _silu(x):
    """SiLU (Swish) activation: x * sigmoid(x)"""
    # sigmoid(x) = 1 / (1 + exp(-x))
    sigmoid_x = 1.0 / (1.0 + ttgl.exp(-x))
    return x * sigmoid_x


@gluon.jit
def _silu_exp2(x):
    """SiLU using exp2 for potentially better hardware utilization"""
    return x / (1.0 + ttgl.exp2(-(x * 1.44269504089)))


@gluon.jit
def _gelu(x):
    """GELU activation: 0.5 * x * (1 + erf(x / sqrt(2)))"""
    M_SQRT1_2 = 0.70710678118654752440
    ALPHA = M_SQRT1_2
    return 0.5 * x * (1.0 + ttgl.erf(x * ALPHA))


@gluon.jit
def _gelu_tanh(x):
    """GELU with tanh approximation (faster but less accurate)"""
    M_SQRT2 = 1.41421356237309504880
    M_2_SQRTPI = 1.12837916709551257390
    BETA = M_SQRT2 * M_2_SQRTPI * 0.5
    KAPPA = 0.044715
    x_cube = x * x * x
    inner = BETA * (x + KAPPA * x_cube)
    # tanh(x) = 2 * sigmoid(2x) - 1 = 2 / (1 + exp(-2x)) - 1
    sigmoid_2inner = 1.0 / (1.0 + ttgl.exp(-2.0 * inner))
    tanh_inner = 2.0 * sigmoid_2inner - 1.0
    return 0.5 * x * (1.0 + tanh_inner)


@gluon.jit
def _relu(x):
    """ReLU activation: max(0, x)"""
    return ttgl.maximum(0.0, x)


def _get_activation_from_str(activation: str):
    """Map activation name string to function."""
    mapping = {
        "gelu": _gelu,
        "gelu_tanh": _gelu_tanh,
        "silu": _silu,
        "silu_exp2": _silu_exp2,
        "relu": _relu,
    }
    return mapping.get(activation)