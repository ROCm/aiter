#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Post-processors for combining profiling results.

Provides pluggable post-processing for different operators:
- Base processor handles common columns (cfg_idx, kernel_name, execution_time, counters)
- Operator-specific processors add their own columns (e.g., MOE adds token, expert, etc.)
"""

from .base import PostProcessor, ProcessedResult
from .moe_processor import MoePostProcessor

__all__ = [
    'PostProcessor',
    'ProcessedResult',
    'MoePostProcessor',
]
