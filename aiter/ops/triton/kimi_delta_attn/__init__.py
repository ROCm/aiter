# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Kimi Delta Attention Operations (Forward Only).

Public Triton entry points for the KDA linear-attention mixer used by
Kimi-Linear / Kimi-K3. The chunked prefill op mirrors ``fla.ops.kda.chunk_kda``.
"""

from aiter.ops.triton.kimi_delta_attn.chunk_delta_attn import chunk_kimi_delta_attn

__all__ = [
    "chunk_kimi_delta_attn",
]
