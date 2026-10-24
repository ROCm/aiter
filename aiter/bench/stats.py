# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Summary statistics over timing samples."""

import statistics
from dataclasses import dataclass

# Tukey fences: samples outside [Q1 - k*IQR, Q3 + k*IQR] count as outliers.
IQR_K = 1.5


@dataclass(frozen=True)
class Stats:
    median: float
    mean: float
    min: float
    max: float
    std: float
    outliers: int  # reported only; outliers stay in every other statistic


def count_outliers(samples, k=IQR_K) -> int:
    if len(samples) < 4:
        return 0
    q1, _, q3 = statistics.quantiles(samples, n=4)
    lower, upper = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    return sum(1 for x in samples if x < lower or x > upper)


def summarize(samples) -> Stats:
    if not samples:
        raise ValueError("no samples to summarize")
    return Stats(
        median=statistics.median(samples),
        mean=statistics.fmean(samples),
        min=min(samples),
        max=max(samples),
        std=statistics.stdev(samples) if len(samples) > 1 else 0.0,
        outliers=count_outliers(samples),
    )
