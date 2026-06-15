# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Backward-compatible re-exports. Implementation lives in custom_all_reduce.py."""

from aiter.ops.custom_all_reduce import (  # noqa: F401
    FUSED_AR_MHC_MD_NAME,
    GFX950_LARGE_M_BOUND,
    fused_allreduce_mhc_fused_post_pre_rmsnorm,
    get_mhc_fused_post_pre_config_ar_mhc,
    get_mhc_pre_splitk_ar_mhc,
    get_mhc_pre_splitk_ar_mhc_large_m,
    launch_fused_allreduce_mhc_fused_post_pre_rmsnorm,
)

MD_NAME = FUSED_AR_MHC_MD_NAME
