# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL GDN/GDR prefill device kernels.

Host wrappers stay in ``aiter.ops.flydsl.linear_attention_prefill_kernels``,
except the single-kernel prefill's, in ``aiter.ops.flydsl.gdn_fused_prefill``.
"""

from .chunk_gated_delta_h import compile_chunk_gated_delta_h
from .gdn_fused_prefill import compile_gdn_fused_prefill
from .gdn_prepare import compile_gdn_prepare

__all__ = [
    "compile_chunk_gated_delta_h",
    "compile_gdn_fused_prefill",
    "compile_gdn_prepare",
]
