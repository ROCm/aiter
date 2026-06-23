# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from aiter.ops.triton.fusions.mhc import mhc, mhc_post
from aiter.ops.triton.fusions.lm_head_argmax import local_argmax_pack

__all__ = [
    "local_argmax_pack",
    "mhc",
    "mhc_post",
]
