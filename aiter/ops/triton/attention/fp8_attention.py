# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public Python wrapper for the FP8 Flash Attention v2 Triton kernels.

Exposes attn_fwd and related utilities at the public aiter.ops.triton.attention
boundary so callers do not need to import internal _triton_kernels symbols.
"""

from aiter.ops.triton._triton_kernels.attention.fp8_attention_kernel import (
    _bwd_kernel_dkdv,
    _bwd_kernel_dq,
    _bwd_preprocess_use_o,
    attn_fwd,
    compute_fp8_scaling_factors,
    get_padded_headsize,
)

__all__ = [
    "_bwd_kernel_dkdv",
    "_bwd_kernel_dq",
    "_bwd_preprocess_use_o",
    "attn_fwd",
    "compute_fp8_scaling_factors",
    "get_padded_headsize",
]
