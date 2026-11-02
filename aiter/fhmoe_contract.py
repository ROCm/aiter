# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Lightweight FHMoE safety helpers shared by runtime, AOT, and tuners.

This module must remain free of imports from the top-level ``aiter`` namespace
so it is safe when ``AITER_AOT_IMPORT=1`` exposes only the lightweight JIT core.
"""

FHMOE_BUFFER_MAX_BYTES = (1 << 32) - 1


def fhmoe_reduce_output_nbytes(
    tokens: int,
    topk: int,
    model_dim: int,
    element_size: int,
) -> int:
    """Return bytes in the reduce-mode ``[M, topk, model_dim]`` temporary."""
    return int(tokens) * int(topk) * int(model_dim) * int(element_size)


def max_fhmoe_reduce_tokens(
    topk: int,
    model_dim: int,
    element_size: int,
) -> int:
    """Largest M addressable by the current 32-bit reduce-output resource."""
    bytes_per_token = fhmoe_reduce_output_nbytes(
        1,
        topk,
        model_dim,
        element_size,
    )
    if bytes_per_token <= 0:
        raise ValueError("FHMoE reduce-output dimensions must be positive")
    return FHMOE_BUFFER_MAX_BYTES // bytes_per_token
