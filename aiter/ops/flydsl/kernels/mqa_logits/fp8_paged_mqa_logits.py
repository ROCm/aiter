# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged FP8 MQA logits (decode), dispatched by architecture.

gfx950 runs the H32/D128/KVB64 ragged LDS kernel in
``fp8_paged_mqa_logits_gfx950``; gfx1250 the WMMA kernel in
``fp8_paged_mqa_logits_gfx1250`` (any 16-multiple head count, 8- to 256-token
pages, the packed cache ``deepgemm_fp8_paged_mqa_logits`` reads).
"""

from aiter.jit.utils.chip_info import get_gfx

from .fp8_paged_mqa_logits_gfx950 import (
    flydsl_fp8_paged_mqa_logits as _flydsl_fp8_paged_mqa_logits_gfx950,
)
from .fp8_paged_mqa_logits_gfx950 import flydsl_fp8_paged_mqa_logits_gfx950
from .fp8_paged_mqa_logits_gfx1250 import flydsl_fp8_paged_mqa_logits_gfx1250

__all__ = [
    "flydsl_fp8_paged_mqa_logits",
    "flydsl_fp8_paged_mqa_logits_gfx950",
    "flydsl_fp8_paged_mqa_logits_gfx1250",
]


def flydsl_fp8_paged_mqa_logits(*args, **kwargs):
    """Paged FP8 MQA logits; see the per-architecture entry points."""
    if get_gfx() == "gfx1250":
        return flydsl_fp8_paged_mqa_logits_gfx1250(*args, **kwargs)
    return _flydsl_fp8_paged_mqa_logits_gfx950(*args, **kwargs)
