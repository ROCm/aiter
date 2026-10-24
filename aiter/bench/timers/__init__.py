# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Timing strategies.

Every timer takes ``calls`` -- zero-argument callables, one per input copy;
timed call ``i`` runs ``calls[i % len(calls)]`` -- plus an optional
``before_call`` hook (the cache flush), and returns a ``TimerOutput``.
"""

from dataclasses import dataclass, field


@dataclass
class TimerOutput:
    samples_us: list
    info: dict = field(default_factory=dict)  # mode-specific BenchResult fields
    notes: list = field(default_factory=list)
