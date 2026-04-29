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

from ._arch import _check_arch

# Gate import on gfx950 today: the a16w16 kernels use gfx950-only intrinsics
# (MFMA-32x32x16, ds_read_b64_tr) and the 160 KiB LDS budget. Other archs
# fall through to a clear ImportError instead of either a JIT compile
# failure or a silent runtime mismatch. Future opus subpackages declare
# their own supported sets in their own __init__.
_check_arch(
    {"gfx950"},
    feature="aiter.ops.opus (a16w16)",
    hint=(
        "opus_gemm uses gfx950-only intrinsics (MFMA, ds_read_b64_tr) and "
        "the 160 KiB LDS budget. Set GPU_ARCHS=gfx950 (or run on a gfx950 "
        "device) to use this module."
    ),
)

from .gemm_op_a16w16 import opus_gemm_a16w16_tune, gemm_a16w16_opus  # noqa: E402

__all__ = [
    "opus_gemm_a16w16_tune",
    "gemm_a16w16_opus",
]
