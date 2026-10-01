# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tuning thresholds that do not depend on the kernel family.

A value belongs here when its justification never mentions a family: how long
to measure, when a faster candidate is worth publishing over the one already
serving. What a family tunes (key fields, candidates, legality, its error
metric) stays in that family's tuner or policy module, and a family that needs
a different threshold overrides it there with ``dataclasses.replace`` rather
than restating the default.

Standard library only, so the runtime and the tests can import it without
pulling in torch or a device.

``PromotionPolicy.min_improvement_pct`` is the one bar for replacing what a
shape runs today. ``--compare --update_improved`` applies it after tuning, by
timing each shape through the operator serving calls with the old and the new
tuned CSV. A tuner that times the incumbent next to the challengers applies it
during the search, through ``gate_against_incumbent``.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MeasurementPolicy:
    """How each candidate is timed and when it is rejected."""

    warmup: int = 5
    iters: int = 101
    # Fraction of mismatching elements above which a candidate is rejected.
    err_ratio: float = 0.05
    # Per-task watchdog in seconds. A worker killed by a GPU memory-access
    # fault leaves its in-flight task unresolvable, and with no timeout the
    # whole run hangs; this stays well above any legitimate shape group.
    timeout: int = 1800


DEFAULT_MEASUREMENT = MeasurementPolicy()


@dataclass(frozen=True)
class RunPolicy:
    """How a tuning run is split up. Nothing here changes which candidate wins."""

    # Shapes tuned between writes of the tuned CSV, so a crash loses at most
    # one batch.
    batch: int = 100

    def __post_init__(self):
        if self.batch < 1:
            raise ValueError(f"batch must be at least 1, got {self.batch}")


DEFAULT_RUN = RunPolicy()


@dataclass(frozen=True)
class PromotionPolicy:
    """When a measured challenger replaces the configuration already serving."""

    # A challenger must be at least this many percent faster than the
    # incumbent. Smaller gaps are within run-to-run noise on one GPU, and
    # publishing them churns the tuned table without making anything faster.
    min_improvement_pct: float = 3.0

    def __post_init__(self):
        if not 0.0 <= self.min_improvement_pct < 100.0:
            raise ValueError(
                "min_improvement_pct must be in [0, 100), "
                f"got {self.min_improvement_pct}"
            )


DEFAULT_PROMOTION = PromotionPolicy()


@dataclass(frozen=True)
class RacePolicy:
    """How an interleaved elimination race spends its measurements.

    The race's indifference zone is ``PromotionPolicy.min_improvement_pct``
    unless a caller narrows or widens it, so by default the race and the gate
    after it agree on what a tie is.
    """

    # Error budget for eliminating a candidate as slower than the leader.
    alpha: float = 0.05
    # Timed calls per candidate per block.
    block_calls: int = 10
    # Blocks before elimination may start, and at which the race stops
    # without certifying a winner.
    min_blocks: int = 3
    max_blocks: int = 30

    def __post_init__(self):
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.block_calls < 1:
            raise ValueError("block_calls must be at least 1")
        if not 1 <= self.min_blocks <= self.max_blocks:
            raise ValueError("need 1 <= min_blocks <= max_blocks")


DEFAULT_RACE = RacePolicy()

# Seeds candidate sampling and the race's block order, so a run is repeatable.
SAMPLE_SEED = 20240917

PROMOTE = "promote"
RETAIN = "retain"


@dataclass(frozen=True)
class GateDecision:
    outcome: str
    # 100 * (incumbent - challenger) / incumbent; None when there was no
    # incumbent latency to compare against.
    margin_pct: float | None


def _measured(latency_us) -> bool:
    return (
        latency_us is not None
        and math.isfinite(float(latency_us))
        and float(latency_us) > 0
    )


def gate_against_incumbent(
    incumbent_us: float | None,
    challenger_us: float,
    policy: PromotionPolicy = DEFAULT_PROMOTION,
) -> GateDecision:
    """Decide whether a challenger replaces the incumbent for one shape.

    The incumbent is what serving runs for the shape today: the published row,
    or the operator's own choice when there is none. With no usable incumbent
    latency there is nothing to protect, so the challenger is promoted and the
    missing margin tells the caller the improvement is unverified.
    """
    if not _measured(challenger_us):
        raise ValueError(f"challenger latency must be positive, got {challenger_us}")
    if not _measured(incumbent_us):
        return GateDecision(PROMOTE, None)
    incumbent_us = float(incumbent_us)
    margin_pct = (incumbent_us - float(challenger_us)) / incumbent_us * 100.0
    outcome = PROMOTE if margin_pct >= policy.min_improvement_pct else RETAIN
    return GateDecision(outcome, margin_pct)
