# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged FP8 MQA logits (decode) -- gfx950 ragged LDS mapping.

The implementation lives in ``fp8_paged_mqa_logits_gfx950``. This module
re-exports the public entry so existing import paths keep working.
"""

from .fp8_paged_mqa_logits_gfx950 import (  # noqa: F401
    flydsl_fp8_paged_mqa_logits,
    flydsl_fp8_paged_mqa_logits_gfx950,
)
