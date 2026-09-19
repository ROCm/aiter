# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL GDN/GDR prefill device kernels.

Host wrappers stay in ``aiter.ops.flydsl.linear_attention_prefill_kernels``.
"""

from .chunk_gated_delta_h import compile_chunk_gated_delta_h
from .chunk_gated_delta_h_gfx942 import (
    compile_chunk_gated_delta_h_gfx942,
    select_variant,
)
from .gdn_prepare import compile_gdn_prepare
from .k5_variants import (
    K5_DEFAULT_VARIANT,
    K5_VARIANTS,
)

__all__ = [
    "K5_DEFAULT_VARIANT",
    "K5_VARIANTS",
    "compile_chunk_gated_delta_h",
    "compile_chunk_gated_delta_h_gfx942",
    "compile_gdn_prepare",
    "select_variant",
]
