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

Promotion bars in use today, which measure different things and are kept
separate on purpose:

- ``COMPARE_MIN_IMPROVEMENT_PCT``: ``--compare --update_improved`` benchmarks
  the public operator before and after tuning and rewrites a shape only if it
  got at least this many percent faster.
- ``PromotionPolicy.indifference_delta``: a tuner that measures the incumbent
  itself (``gate_against_incumbent``) publishes a challenger only if it beats
  the incumbent by more than this fraction.
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

COMPARE_MIN_IMPROVEMENT_PCT = 3.0


@dataclass(frozen=True)
class PromotionPolicy:
    """When a measured challenger replaces the configuration already serving."""

    # A challenger must be faster than the incumbent by more than this
    # fraction. Gaps inside it are run-to-run noise on one GPU, and publishing
    # them churns the tuned table without making anything faster.
    indifference_delta: float = 0.02
    # Screening keeps this many of the fastest candidates per shape, and times
    # them again this many times, before the winner is chosen.
    finalists: int = 8
    finalist_rounds: int = 3

    def __post_init__(self):
        if not 0.0 <= self.indifference_delta < 1.0:
            raise ValueError(
                f"indifference_delta must be in [0, 1), got {self.indifference_delta}"
            )
        if self.finalists < 1 or self.finalist_rounds < 1:
            raise ValueError("finalists and finalist_rounds must be at least 1")


DEFAULT_PROMOTION = PromotionPolicy()

PROMOTE = "promote"
RETAIN = "retain"


@dataclass(frozen=True)
class GateDecision:
    outcome: str
    # (incumbent - challenger) / incumbent; None when there was no incumbent
    # latency to compare against.
    margin: float | None


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
    margin = (incumbent_us - float(challenger_us)) / incumbent_us
    outcome = PROMOTE if margin > policy.indifference_delta else RETAIN
    return GateDecision(outcome, margin)
