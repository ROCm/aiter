# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
aiter.ops.opus — opus kernel Python user-facing API.

Per-dtype modules. a16w16 lives here today; a8w8 / a8w8_blockscale
arrive in follow-up PRs. Each module owns its own Python surface and
pybind bindings but shares the underlying JIT module
`module_deepgemm_opus` built from csrc/opus_gemm/.

Public API:
  * gemm_a16w16_opus       — shape-driven wrapper (CSV lookup + C++
                             heuristic fallback). Typical user entry.
  * opus_gemm_a16w16_tune  — id-based low-level binding (tuner / override).
"""

from .gemm_op_a16w16 import opus_gemm_a16w16_tune, gemm_a16w16_opus

__all__ = [
    "opus_gemm_a16w16_tune",
    "gemm_a16w16_opus",
]
