# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
aiter.ops.opus — opus kernel Python-side bindings.

Public API:
  * gemm_a16w16_opus       — shape-driven wrapper (CSV lookup + C++
                             heuristic fallback). Typical user entry.
  * opus_gemm_a16w16_tune  — id-based low-level binding (tuner / override).
  * deepgemm_opus          — FP8 grouped binding (unchanged legacy API).
"""

from .gemm_op_a16w16 import opus_gemm_a16w16_tune, gemm_a16w16_opus
from .deepgemm import deepgemm_opus

__all__ = [
    "opus_gemm_a16w16_tune",
    "gemm_a16w16_opus",
    "deepgemm_opus",
]
