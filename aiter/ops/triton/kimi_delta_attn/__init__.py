# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Kimi Delta Attention Operations (Forward Only).

Public Triton entry points for the KDA linear-attention mixer used by
Kimi-Linear / Kimi-K3. The chunked prefill op mirrors ``fla.ops.kda.chunk_kda``.

``chunk_kimi_delta_attn`` runs a call on the FlashKDA path when
``AITER_FDA_ENABLE`` is on and ``flash_kda_supported`` accepts the call, with an
unset ``chunk_size`` passed as ``FLASH_KDA_CHUNK``. Only that path takes
``snapshot_targets``, at offsets that are multiples of ``FLASH_KDA_CHUNK``.
"""

from aiter.ops.triton._triton_kernels.kimi_delta_attn.flash_kda import (
    AITER_FDA_ENABLE,
    FLASH_KDA_CHUNK,
    flash_kda_supported,
)
from aiter.ops.triton.kimi_delta_attn.chunk_delta_attn import chunk_kimi_delta_attn

__all__ = [
    "AITER_FDA_ENABLE",
    "FLASH_KDA_CHUNK",
    "chunk_kimi_delta_attn",
    "flash_kda_supported",
]
