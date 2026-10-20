# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unified kernel benchmarking. See aiter/bench/README.md."""

from .api import run_bench, time_all, time_fn
from .config import CACHE_POLICIES, DEFAULT_CACHE, METRICS, MODES, MeasureConfig
from .result import SCHEMA_VERSION, BenchResult
from .timers.graph import EmptyGraphCaptureError, GraphCaptureError

__all__ = [
    "CACHE_POLICIES",
    "DEFAULT_CACHE",
    "METRICS",
    "MODES",
    "SCHEMA_VERSION",
    "BenchResult",
    "EmptyGraphCaptureError",
    "GraphCaptureError",
    "MeasureConfig",
    "run_bench",
    "time_all",
    "time_fn",
]
