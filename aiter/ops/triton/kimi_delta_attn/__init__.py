# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi Delta Attention (KDA) operations for Kimi-Linear / Kimi-K3.

- ``chunk_kimi_delta_attn``: the chunked prefill KDA linear-attention mixer
  (forward only; mirrors ``fla.ops.kda.chunk_kda``).
- ``kimi_kda_fused_layer_{prefill,decode}``: whole rank-local KDA layer kernels
  for Kimi-K3 on gfx950/TP8 (projections + convolution + gated-delta + gated
  RMSNorm, with the O matmul fused on the decode path). Both report coverage via
  ``*_supported`` so callers fall back to the native KDA path off gfx950 /
  pre-3.8 Triton / unsupported shapes.
"""

from aiter.ops.triton.kimi_delta_attn.chunk_delta_attn import chunk_kimi_delta_attn
from aiter.ops.triton.kimi_delta_attn.fused_layer_decode import (
    kimi_kda_fused_layer_decode,
    kimi_kda_fused_layer_decode_supported,
)
from aiter.ops.triton.kimi_delta_attn.fused_layer_prefill import (
    kimi_kda_fused_layer_prefill,
    kimi_kda_fused_layer_prefill_supported,
)

__all__ = [
    "chunk_kimi_delta_attn",
    "kimi_kda_fused_layer_decode",
    "kimi_kda_fused_layer_decode_supported",
    "kimi_kda_fused_layer_prefill",
    "kimi_kda_fused_layer_prefill_supported",
]
