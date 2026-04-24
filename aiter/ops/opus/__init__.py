# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
aiter.ops.opus — opus kernel Python-side bindings.

Public API (PR1: id-based tune + FP8 grouped deepgemm):
  * opus_gemm_a16w16_tune (from gemm_op_a16w16)
  * deepgemm_opus         (from deepgemm)

PR2 will add the shape-driven `gemm_a16w16_opus` (CSV lookup + heuristic
fallback + AITER_OPUS_LOG_UNTUNED auto-log).
"""

from .gemm_op_a16w16 import opus_gemm_a16w16_tune
from .deepgemm import deepgemm_opus

__all__ = [
    "opus_gemm_a16w16_tune",
    "deepgemm_opus",
]
