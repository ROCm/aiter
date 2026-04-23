# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
import torch
import torch.nn.functional as F

# Channel padding granularity for prepacked weights/inputs. Must align with the
# BLOCK_K autotune candidates in _triton_kernels/conv/helpers.py — change with care.
BLOCK_K = 64


def dynamic_conv_tolerances(dtype: torch.dtype, K_red: int, ref: torch.Tensor):
    eps = {
        torch.float16: 2**-10,
        torch.bfloat16: 2**-7,
        torch.float32: 2**-23,
    }.get(dtype, 2**-10)
    rtol = 6e-3 if K_red < 1024 else (8e-3 if K_red < 4096 else 1.2e-2)
    # Error model: fp16 inputs multiplied pairwise have eps relative error per product.
    # Accumulated in fp32 over K_red terms, max absolute error grows as ~eps * sqrt(K_red).
    # The 10x multiplier covers worst-case accumulation ordering differences
    # between our Triton kernels and PyTorch reference.
    atol = max(eps * 8, 10.0 * eps * (K_red**0.5))
    return rtol, atol


def flops_conv(N, C, K_out, R, S, P, Q):
    return 2.0 * N * P * Q * K_out * C * R * S


def _out_hw(H, W, R, S, stride, padding, dilation):
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    P = (H + 2 * ph - dh * (R - 1) - 1) // sh + 1
    Q = (W + 2 * pw - dw * (S - 1) - 1) // sw + 1
    return P, Q


def _storage_ptr(t: torch.Tensor) -> int:
    return (
        t.untyped_storage().data_ptr()
        if hasattr(t, "untyped_storage")
        else t.storage().data_ptr()
    )


def _is_1x1_conv(R, S, dilation):
    """Check if this is a 1x1 convolution (no spatial reduction in kernel)."""
    return R == 1 and S == 1 and dilation == (1, 1)


def _is_3x3_conv(R, S):
    """Check if this is a 3x3 convolution."""
    return R == 3 and S == 3


def _is_winograd_eligible(R, S, stride, dilation, C=None):
    if not (R == 3 and S == 3 and stride == (1, 1) and dilation == (1, 1)):
        return False
    # F(4,3) output transform amplifies bf16 rounding by up to 361x (AT row3 L1=19).
    # With very few input channels the tolerance budget is too small to absorb this.
    if C is not None and C < 4:
        return False
    return True


def _winograd_tolerances(dtype, K_red, ref, variant="f4x3"):
    """Return (rtol, atol) for Winograd F(4x4,3x3) correctness checks.
    Winograd transforms amplify fp16 rounding errors:
    - F(4x4,3x3): coefficients up to ±8, significant amplification
    """
    rtol, atol = dynamic_conv_tolerances(dtype, K_red, ref)
    if variant == "f4x3":
        rtol *= 6.0
        atol = max(atol * 6.0, 0.6)
    return rtol, atol


def apply_activation(y: torch.Tensor, activation: str):
    if activation == "relu":
        return F.relu(y)
    if activation == "relu6":
        return torch.clamp(y, 0, 6)
    if activation == "gelu":
        return F.gelu(y, approximate="tanh")
    return y
