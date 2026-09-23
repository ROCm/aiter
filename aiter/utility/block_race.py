# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Interleaved randomized-block measurement with delta-based elimination.

The whole field is measured in one process as a randomized complete block
design. Each block visits every surviving candidate once, in a fresh random
order, timing a short run of calls per visit. Blocking pairs the comparison so
drift cancels in the differences; reshuffling each block keeps position from
being confounded with the candidate.

The objective is selection rather than hypothesis testing, so candidates
within ``delta`` are treated as settled. That keeps the block count a function
of delta and the measurement noise, not of the size of the catalogue.

Two constraints on callers:

Correctness is the caller's. Entrants reach :func:`race` only through the
timing function, so nothing here sees kernel output and a candidate computing
the wrong answer races like any other. Check it in the pass that compiles and
warms each candidate, before any timed call. ``_race_one_shape`` in
``op_tests/tuners/tune_mha_fwd.py`` is the reference implementation.

Nothing here imports torch at module scope, so the decision logic is testable
on synthetic latencies. :func:`cuda_event_timer` is the GPU timing function.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BlockJournal",
    "BlockRecord",
    "JsonlBlockJournal",
    "RaceEntrant",
    "RaceResult",
    "Samples",
    "Verdict",
    "critical_t",
    "cuda_event_timer",
    "measure_blocks",
    "race",
    "rank",
    "regularized_incomplete_beta",
    "select_winner",
    "student_t_sf",
]


@dataclass(frozen=True)
class RaceEntrant:
    """One thing to be measured.

    ``payload`` is opaque here: whatever the caller wants back when the race
    names a winner. ``protected`` marks an entrant measured in every block but
    never eliminated, which is how the configuration already in use stays in
    the field.
    """

    label: str
    payload: Any = None
    protected: bool = False


@dataclass
class Samples:
    """Per-call latencies, kept grouped by the block that produced them."""

    blocks: list[list[float]] = field(default_factory=list)

    @property
    def block_medians(self) -> list[float]:
        return [statistics.median(block) for block in self.blocks if block]

    @property
    def estimate(self) -> float:
        medians = self.block_medians
        return statistics.median(medians) if medians else float("inf")

    @property
    def relative_spread(self) -> float:
        """Block-to-block standard deviation as a fraction of the estimate.

        The tie-break statistic: among candidates equally fast in
        expectation, prefer the one whose speed repeats.
        """
        medians = self.block_medians
        estimate = self.estimate
        if len(medians) < 2 or not math.isfinite(estimate) or estimate <= 0:
            return float("inf")
        return statistics.stdev(medians) / estimate


@dataclass
class Verdict:
    label: str
    # "leader", "within_delta", "protected_behind", "eliminated", "undecided",
    # "crashed"
    state: str
    estimate: float
    relative_gap: float
    blocks_used: int
    relative_spread: float = float("inf")
    note: str = ""


@dataclass
class BlockRecord:
    """What one completed block cost, for reporting how a race scales."""

    block: int
    active: int
    calls_spent: int
    wall_seconds: float
    eliminated: int = 0


@dataclass
class RaceResult:
    verdicts: list[Verdict]
    samples: dict[str, Samples]
    calls_spent: int
    history: list[BlockRecord]
    certified: bool
    winner: str
    tie_break: str
    survivors: list[str]
    blocks_run: int
    blocks_replayed: int = 0

    def entrant(self, entrants: Sequence[RaceEntrant]) -> RaceEntrant | None:
        return next((e for e in entrants if e.label == self.winner), None)


class BlockJournal:
    """Append-only record of completed blocks, for resuming an interrupted race.

    The completed block is the unit of durable progress. Elimination is a pure
    function of the accumulated samples, so replaying whole blocks
    reconstructs the state exactly and nothing is re-measured.
    """

    def append(self, record: dict) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def records(self) -> Iterable[dict]:  # pragma: no cover - interface
        raise NotImplementedError


class JsonlBlockJournal(BlockJournal):
    """A journal backed by one JSON object per line.

    A resume picks up from the last block written whole. A process killed
    mid-write leaves an unterminated final line, which is dropped on open so
    that appends cannot land after invalid JSON and hide every block written
    since.
    """

    def __init__(self, path: str, resume: bool = False):
        self.path = path
        if not resume:
            if os.path.exists(path):
                os.remove(path)
        elif os.path.isfile(path):
            with open(path, "rb+") as handle:
                payload = handle.read()
                if payload and not payload.endswith(b"\n"):
                    handle.truncate(payload.rfind(b"\n") + 1)

    def append(self, record: dict) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def records(self) -> Iterable[dict]:
        if not os.path.isfile(self.path):
            return []
        kept = []
        with open(self.path) as handle:
            for line in handle:
                try:
                    kept.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        return kept


def cuda_event_timer(invoke: Callable[[RaceEntrant], Any]):
    """Time ``invoke`` with CUDA events, returning microseconds per call.

    torch is imported lazily so this module stays importable without it.
    """
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def time_calls(entrant: RaceEntrant, count: int) -> list[float]:
        latencies = []
        for _ in range(count):
            start.record()
            invoke(entrant)
            end.record()
            end.synchronize()
            latencies.append(start.elapsed_time(end) * 1000.0)
        return latencies

    return time_calls


def measure_blocks(
    entrants: Sequence[RaceEntrant],
    time_calls,
    block_calls: int,
    blocks: int,
    seed: int,
    discard_per_block: int = 0,
) -> dict[str, Samples]:
    """Run a randomized complete block design over the whole field.

    No elimination: every entrant is measured in every block. This is the
    reference the race is checked against.
    """
    rng = random.Random(seed)
    samples = {entrant.label: Samples() for entrant in entrants}
    for _ in range(blocks):
        order = list(entrants)
        rng.shuffle(order)
        for entrant in order:
            if discard_per_block:
                time_calls(entrant, discard_per_block)
            samples[entrant.label].blocks.append(time_calls(entrant, block_calls))
    return samples


def _usable_latencies(measured: Sequence[float]) -> bool:
    """Reject a block that cannot be compared or divided by.

    A zero or negative reading is a broken measurement, not a fast one, and
    the leader's estimate is a denominator throughout the race.
    """

    return bool(measured) and all(
        math.isfinite(value) and value > 0.0 for value in measured
    )


def _lower_bound(differences: Sequence[float], per_decision: float) -> float:
    """Lower confidence bound on the mean paired difference."""
    spread = statistics.stdev(differences) / math.sqrt(len(differences))
    return (
        statistics.mean(differences)
        - critical_t(per_decision, len(differences) - 1) * spread
    )


def race(
    entrants: Sequence[RaceEntrant],
    time_calls,
    *,
    delta: float,
    alpha: float,
    block_calls: int,
    min_blocks: int,
    max_blocks: int,
    seed: int,
    journal: BlockJournal | None = None,
    resume: bool = False,
    verbose: bool = True,
    report=print,
) -> RaceResult:
    """Eliminate candidates that are worse than the leader by more than delta.

    Elimination and stopping read the same lower bound on the paired
    difference against the leader, but ask opposite questions of it, which is
    where the budget is saved. Dropping a candidate needs proof it is more
    than delta *worse*. Stopping needs proof only that no survivor is more
    than delta *better*, since that is the only way picking the leader could
    be wrong.

    Elimination is permanent and the leader is recomputed every block, so a
    comparison always uses the best evidence so far. Candidates are compared
    on the blocks they both ran, which keeps the pairing even though they
    leave at different times.

    Protected entrants are measured but never removed. One the same test would
    have dropped is reported ``protected_behind``, so being kept is not read
    as being within delta.

    ``survivors`` is the set that may be published: the leader plus whatever
    measured inside the indifference zone. Candidates that merely outlasted
    the budget are reported ``undecided`` and are not eligible.
    """
    rng = random.Random(seed)
    samples = {entrant.label: Samples() for entrant in entrants}
    by_label = {entrant.label: entrant for entrant in entrants}
    active = [entrant.label for entrant in entrants]
    eliminated: dict[str, tuple[int, float]] = {}
    behind: dict[str, float] = {}
    # A candidate that faults or times as zero leaves the field rather than
    # taking the race down with it, and rather than winning on a bad number.
    crashed: dict[str, str] = {}

    replay = list(journal.records()) if (journal is not None and resume) else []
    replayed = 0

    # Union bound over every candidate and every look, since peeking after
    # each block is repeated testing and an uncorrected alpha would drift.
    per_decision = alpha / max(1, len(entrants) * max_blocks)

    calls_spent = 0
    history: list[BlockRecord] = []
    certified = False

    for block_index in range(max_blocks):
        block_started = time.perf_counter()
        order = [by_label[label] for label in active]
        rng.shuffle(order)

        # Replay and live measurement share this loop so a resumed race cannot
        # reach a different verdict than an uninterrupted one.
        record = replay[block_index] if block_index < len(replay) else None
        if record is not None:
            journaled = [
                label
                for label in record["order"]
                if label in by_label and label in record["latencies"]
            ]
            if set(journaled) != set(active):
                # The journal has stopped describing this race: an entrant
                # still active here was not measured in that block, so
                # replaying it would pair blocks that never ran against each
                # other. Keep the blocks that did match and measure the rest.
                if verbose:
                    report(
                        f"  journal diverges at block {block_index + 1}; "
                        f"measuring the remaining blocks live"
                    )
                replay = replay[:block_index]
                record = None
            elif [e.label for e in order] != journaled:
                # The same field in a different order. The journal is the
                # record of what actually ran, so it wins.
                order = [by_label[label] for label in journaled]
        if record is not None:
            for entrant in order:
                samples[entrant.label].blocks.append(record["latencies"][entrant.label])
            calls_spent += sum(len(v) for v in record["latencies"].values())
            replayed += 1
        else:
            latencies = {}
            for entrant in order:
                try:
                    measured = time_calls(entrant, block_calls)
                except Exception as error:  # noqa: BLE001 - a fault is a verdict
                    crashed[entrant.label] = f"{type(error).__name__}: {error}"
                    continue
                if not _usable_latencies(measured):
                    crashed[entrant.label] = (
                        "timer returned a latency that is not positive and finite"
                    )
                    continue
                samples[entrant.label].blocks.append(measured)
                latencies[entrant.label] = measured
                calls_spent += len(measured)
            for label, reason in crashed.items():
                if label in active:
                    active.remove(label)
                    if verbose:
                        report(f"  {label} left the race: {reason}")
            if not active:
                raise RuntimeError(
                    "every entrant failed to time: "
                    + "; ".join(f"{k} ({v})" for k, v in crashed.items())
                )
            if journal is not None:
                journal.append(
                    {
                        "block": block_index + 1,
                        "order": [entrant.label for entrant in order],
                        "latencies": latencies,
                    }
                )

        block_wall = time.perf_counter() - block_started
        history.append(
            BlockRecord(block_index + 1, len(active), calls_spent, block_wall)
        )
        if block_index + 1 < min_blocks:
            continue

        leader = min(active, key=lambda label: samples[label].estimate)
        leader_blocks = samples[leader].block_medians
        leader_estimate = samples[leader].estimate
        tolerance = delta * leader_estimate

        undecided, dropped = [], []
        behind = {}
        for label in active:
            if label == leader:
                continue
            blocks = samples[label].block_medians
            paired = min(len(leader_blocks), len(blocks))
            differences = [blocks[i] - leader_blocks[i] for i in range(paired)]
            if len(differences) < 2:
                undecided.append(label)
                continue
            lower = _lower_bound(differences, per_decision)
            if lower > tolerance:
                if by_label[label].protected:
                    # Kept in the field, but the evidence says it is behind.
                    behind[label] = samples[label].estimate / leader_estimate - 1.0
                else:
                    dropped.append(label)
            elif lower <= -tolerance:
                undecided.append(label)  # could be > delta better than the leader

        for label in dropped:
            eliminated[label] = (
                block_index + 1,
                samples[label].estimate / leader_estimate - 1.0,
            )
            active.remove(label)
        history[-1].eliminated = len(dropped)

        if verbose:
            report(
                f"  block {block_index + 1:>3}: {len(active):>3} active, "
                f"{len(dropped):>2} eliminated, {len(undecided):>2} undecided, "
                f"leader {leader_estimate:8.1f} us"
            )
        if not undecided:
            certified = True
            break

    leader = min(active, key=lambda label: samples[label].estimate)
    leader_estimate = samples[leader].estimate

    verdicts = [
        Verdict(
            leader,
            "leader",
            leader_estimate,
            0.0,
            len(samples[leader].blocks),
            samples[leader].relative_spread,
        )
    ]
    # Surviving elimination is not the same as being tied with the leader.
    # Elimination needs *proof* of being more than delta worse, so a noisy
    # candidate can outlast the race while its point estimate sits well behind.
    # Only candidates whose measured latency is actually inside the
    # indifference zone are eligible to be published, because the tie-break
    # reasons about configurations believed to be equally fast; handing it one
    # that is merely unproven would let steadiness outrank being much slower.
    tolerance = delta * leader_estimate
    notes = {
        "protected_behind": "protected, but measured behind the leader",
        "undecided": "outlasted the budget without being separated from the leader",
        "within_delta": "",
    }
    survivors = [leader]
    for label in active:
        if label == leader:
            continue
        if label in behind:
            state = "protected_behind"
        elif samples[label].estimate - leader_estimate <= tolerance:
            state = "within_delta"
            survivors.append(label)
        else:
            state = "undecided"
        verdicts.append(
            Verdict(
                label,
                state,
                samples[label].estimate,
                samples[label].estimate / leader_estimate - 1.0,
                len(samples[label].blocks),
                samples[label].relative_spread,
                notes[state],
            )
        )
    for label, (block, gap) in eliminated.items():
        verdicts.append(
            Verdict(
                label,
                "eliminated",
                samples[label].estimate,
                gap,
                len(samples[label].blocks),
                samples[label].relative_spread,
                f"dropped after block {block}",
            )
        )
    for label, reason in crashed.items():
        verdicts.append(
            Verdict(
                label,
                "crashed",
                float("inf"),
                float("inf"),
                len(samples[label].blocks),
                samples[label].relative_spread,
                reason,
            )
        )

    winner, tie_break = select_winner(survivors, samples, by_label)
    return RaceResult(
        verdicts=verdicts,
        samples=samples,
        calls_spent=calls_spent,
        history=history,
        certified=certified,
        winner=winner,
        tie_break=tie_break,
        survivors=survivors,
        blocks_run=history[-1].block if history else 0,
        blocks_replayed=replayed,
    )


def select_winner(
    survivors: Sequence[str],
    samples: dict[str, Samples],
    by_label: dict[str, RaceEntrant],
) -> tuple[str, str]:
    """Pick one config from a set the race declared equally fast.

    No survivor is meaningfully faster than another, so this rule is not about
    speed. It exists to make a re-tune return the same answer and leave the
    config file alone when nothing improved.

    The incumbent wins first, then the steadiest survivor, then a stable sort
    on the label so exactly tied spreads still resolve the same way every run.
    """
    if not survivors:
        raise ValueError("a race cannot finish with no survivors")

    protected = sorted(
        (label for label in survivors if by_label[label].protected),
        key=lambda label: (samples[label].estimate, label),
    )
    if protected:
        return protected[0], "incumbent_retained"

    ordered = sorted(
        survivors, key=lambda label: (samples[label].relative_spread, label)
    )
    if len(ordered) == 1:
        return ordered[0], "sole_survivor"
    best, runner_up = ordered[0], ordered[1]
    if samples[best].relative_spread == samples[runner_up].relative_spread:
        return best, "stable_sort"
    return best, "lowest_spread"


def rank(samples: dict[str, Samples]) -> list[tuple[str, float]]:
    return sorted(
        ((label, sample.estimate) for label, sample in samples.items()),
        key=lambda item: item[1],
    )


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Lentz evaluation of the continued fraction for the incomplete beta."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        step = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + step * d
        c = 1.0 + step / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        step = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + step * d
        c = 1.0 + step / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b), the only special function the tests below need."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    # The fraction only converges quickly on one side of this point; past it,
    # evaluate the mirrored parameters and take the complement.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_sf(t: float, degrees: int) -> float:
    """P(T > t) for Student's t.

    Implemented here rather than taken from scipy, which aiter does not
    declare as a dependency. A tuner that only runs where scipy happens to be
    installed is a tuner that silently changes its statistics with the
    environment.
    """
    tail = 0.5 * regularized_incomplete_beta(
        0.5 * degrees, 0.5, degrees / (degrees + t * t)
    )
    return tail if t > 0 else 1.0 - tail


def critical_t(alpha_two_sided: float, degrees: int) -> float:
    """Two-sided critical value: t with P(|T| > t) == alpha_two_sided.

    Bisection rather than a closed form. The Cornish-Fisher expansion from the
    normal quantile is the usual shortcut, but it is worst exactly where this
    is used -- few degrees of freedom and a far tail, where the Bonferroni
    correction puts the per-decision alpha -- so it is not worth the risk.
    """
    degrees = max(1, degrees)
    target = alpha_two_sided / 2.0
    # Grow the bracket instead of assuming a ceiling. With one degree of
    # freedom and a Bonferroni-shrunk alpha the critical value runs into the
    # hundreds of thousands, and a fixed upper bound would silently saturate
    # and hand back a value small enough to eliminate candidates that the
    # evidence does not support.
    low, high = 0.0, 1.0
    while student_t_sf(high, degrees) > target and high < 1e300:
        low, high = high, high * 4.0
    for _ in range(400):
        middle = 0.5 * (low + high)
        if student_t_sf(middle, degrees) > target:
            low = middle
        else:
            high = middle
        if high - low < 1e-12 * max(1.0, high):
            break
    return 0.5 * (low + high)
