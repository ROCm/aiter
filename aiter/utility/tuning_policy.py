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
- ``PromotionPolicy.significance_sigma``: when both candidates were timed over
  repeated rounds, ``standard_error_bar`` turns their scatter into the margin
  to clear instead, so more rounds resolve smaller improvements.
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
    # With repeated rounds of both candidates, the challenger must instead be
    # faster by this many combined standard errors; two is the conventional
    # ~95% two-sample separation.
    significance_sigma: float = 2.0

    def __post_init__(self):
        if not 0.0 <= self.indifference_delta < 1.0:
            raise ValueError(
                f"indifference_delta must be in [0, 1), got {self.indifference_delta}"
            )
        if self.finalists < 1 or self.finalist_rounds < 1:
            raise ValueError("finalists and finalist_rounds must be at least 1")
        if not self.significance_sigma > 0.0:
            raise ValueError(
                f"significance_sigma must be positive, got {self.significance_sigma}"
            )


DEFAULT_PROMOTION = PromotionPolicy()


@dataclass(frozen=True)
class RacePolicy:
    """How an interleaved elimination race spends its measurements.

    The race's indifference zone is ``PromotionPolicy.indifference_delta``, so
    the race and the gate after it agree on what a tie is.
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
    # (incumbent - challenger) / incumbent; None when there was no incumbent
    # latency to compare against.
    margin: float | None


def _measured(latency_us) -> bool:
    return (
        latency_us is not None
        and math.isfinite(float(latency_us))
        and float(latency_us) > 0
    )


def standard_error_bar(
    incumbent_us: float | None,
    combined_standard_error_us: float,
    policy: PromotionPolicy = DEFAULT_PROMOTION,
) -> float:
    """The relative margin that clears ``policy.significance_sigma`` standard
    errors, where the combined standard error is that of the two candidates'
    round means (``math.hypot`` of each)."""
    if not _measured(incumbent_us):
        return 0.0
    return policy.significance_sigma * combined_standard_error_us / float(incumbent_us)


def gate_against_incumbent(
    incumbent_us: float | None,
    challenger_us: float,
    policy: PromotionPolicy = DEFAULT_PROMOTION,
    bar: float | None = None,
) -> GateDecision:
    """Decide whether a challenger replaces the incumbent for one shape.

    The incumbent is what serving runs for the shape today: the published row,
    or the operator's own choice when there is none. With no usable incumbent
    latency there is nothing to protect, so the challenger is promoted and the
    missing margin tells the caller the improvement is unverified.

    ``bar`` is the relative margin to clear, ``policy.indifference_delta`` when
    not given; pass ``standard_error_bar`` when both were timed over rounds.
    """
    if not _measured(challenger_us):
        raise ValueError(f"challenger latency must be positive, got {challenger_us}")
    if not _measured(incumbent_us):
        return GateDecision(PROMOTE, None)
    incumbent_us = float(incumbent_us)
    margin = (incumbent_us - float(challenger_us)) / incumbent_us
    threshold = policy.indifference_delta if bar is None else bar
    outcome = PROMOTE if margin > threshold else RETAIN
    return GateDecision(outcome, margin)
