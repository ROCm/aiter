# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Interleaved randomized-block measurement with delta-based elimination.

The whole field is measured in one process as a randomized complete block
design. Each block visits every surviving candidate once, in a fresh random
order, timing a short run of calls per visit; the median of those calls is
the visit's sample. Blocking pairs the comparison so drift cancels in the
differences; reshuffling each block keeps position from being confounded with
the candidate.

The objective is selection rather than hypothesis testing, so candidates
within ``delta`` of the leader are treated as settled and the race stops once
no survivor can be more than ``delta`` faster. The error budget is split with
a Bonferroni bound over every candidate and every look, so the critical value
grows slowly (about sqrt(log N)) with the field size N and with
``max_blocks``.

Two constraints on callers:

Correctness is the caller's. Entrants reach :func:`race` only through the
timing function, so nothing here sees kernel output and a candidate computing
the wrong answer races like any other. Check it in the pass that compiles and
warms each candidate, before any timed call.

Timing is the caller's too: ``time_calls(entrant, count)`` returns ``count``
latencies in microseconds, measured however the family measures everything
else.

It pays off where the field is large: Triton MHA tiles (1,440 on gfx950)
tuned inside a cross-backend tuner, or FHMoE's FlyDSL kernel pairs. The
MHA tuner on this branch measures one candidate per backend and does not
use it.

The decision logic only ever sees latencies, so it is tested on synthetic
ones; nothing here touches a GPU.
"""

from __future__ import annotations

import functools
import json
import math
import os
import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "JsonlBlockJournal",
    "RaceEntrant",
    "RacePolicy",
    "RaceResult",
    "Verdict",
    "race",
]


@dataclass(frozen=True)
class RacePolicy:
    """How an interleaved elimination race spends its measurements.

    The race's indifference zone is not set here: ``race`` takes the
    promotion bar (``min_improvement_pct``) and uses it, so the race and the
    gate after it agree on what a tie is.
    """

    # Error budget, split between eliminating the true best and certifying a
    # winner more than the indifference zone behind it.
    alpha: float = 0.05
    # Timed calls per candidate per visit; the visit's median is its sample.
    block_calls: int = 10
    # Untimed calls at the start of each visit. The previous visit ran a
    # different kernel, so the first call is cold.
    warmup_calls: int = 1
    # Blocks before elimination may start, and at which the race stops
    # without certifying a winner.
    min_blocks: int = 3
    max_blocks: int = 30

    def __post_init__(self):
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.block_calls < 1:
            raise ValueError("block_calls must be at least 1")
        if self.warmup_calls < 0:
            raise ValueError("warmup_calls cannot be negative")
        if not 2 <= self.min_blocks <= self.max_blocks:
            # One block gives no spread, so nothing could be eliminated.
            raise ValueError("need 2 <= min_blocks <= max_blocks")


DEFAULT_RACE = RacePolicy()

# Seeds the race's block order, so a run is repeatable.
SAMPLE_SEED = 20240917


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
    """Per-call latencies, kept grouped by the visit that produced them."""

    blocks: list[list[float]] = field(default_factory=list)

    @property
    def block_medians(self) -> list[float]:
        return [statistics.median(block) for block in self.blocks if block]

    @property
    def estimate(self) -> float:
        """Mean of the visit medians: the statistic the paired test is on."""
        medians = self.block_medians
        return statistics.fmean(medians) if medians else float("inf")


@dataclass
class Verdict:
    label: str
    # "leader", "within_delta", "protected_behind", "eliminated", "undecided",
    # "crashed"
    state: str
    estimate: float
    relative_gap: float
    blocks_used: int
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
    """The outcome of :func:`race`.

    ``winner`` is the leader, or the protected incumbent when it passed the
    same certification test. When ``certified`` is false the race ran out of
    blocks with some survivor possibly more than ``delta`` faster than
    ``winner``; ``winner`` is then only the current point-estimate leader, and
    a caller should keep the incumbent or measure the survivors exhaustively.
    """

    verdicts: list[Verdict]
    samples: dict[str, Samples]
    calls_spent: int
    history: list[BlockRecord]
    certified: bool
    winner: str
    # "leader" or "incumbent_retained"
    rule: str
    survivors: list[str]
    blocks_run: int
    delta: float
    blocks_replayed: int = 0

    def entrant(self, entrants: Sequence[RaceEntrant]) -> RaceEntrant | None:
        return next((e for e in entrants if e.label == self.winner), None)


class JsonlBlockJournal:
    """Append-only record of a race, for resuming one that was interrupted.

    The first line fingerprints the race (field, policy, delta, seed and the
    caller's key), and a resume refuses a journal whose fingerprint differs.
    After it come three kinds of line: ``visiting`` before each live visit,
    ``crashed`` when an entrant leaves, and one line per completed block.

    The completed block is the unit of durable progress. Elimination is a pure
    function of the accumulated samples, so replaying whole blocks
    reconstructs the state exactly. A visit journaled as started but in no
    completed block was running when the process died, so a resume treats
    that entrant as crashed rather than run it again.

    A process killed mid-write leaves an unterminated final line, which is
    dropped on open so that appends cannot land after invalid JSON and hide
    every line written since.
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

    def records(self) -> list[dict]:
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


def measure_blocks(
    entrants: Sequence[RaceEntrant],
    time_calls,
    blocks: int,
    seed: int,
    policy: RacePolicy = DEFAULT_RACE,
) -> dict[str, Samples]:
    """Run a randomized complete block design over the whole field.

    No elimination: every entrant is measured in every block, with the same
    per-visit warmup as :func:`race`. This is the reference the race is
    checked against.
    """
    rng = random.Random(seed)
    samples = {entrant.label: Samples() for entrant in entrants}
    for _ in range(blocks):
        order = list(entrants)
        rng.shuffle(order)
        for entrant in order:
            if policy.warmup_calls:
                time_calls(entrant, policy.warmup_calls)
            samples[entrant.label].blocks.append(
                time_calls(entrant, policy.block_calls)
            )
    return samples


def _usable_latencies(measured: Sequence[float]) -> bool:
    """Reject a visit that cannot be compared or divided by.

    A zero or negative reading is a broken measurement, not a fast one, and
    the leader's estimate is a denominator throughout the race.
    """

    return bool(measured) and all(
        math.isfinite(value) and value > 0.0 for value in measured
    )


def _paired(samples: dict[str, Samples], label: str, against: str) -> list[float]:
    """Per-visit differences ``label - against`` over the blocks both ran."""
    mine, theirs = samples[label].block_medians, samples[against].block_medians
    return [mine[i] - theirs[i] for i in range(min(len(mine), len(theirs)))]


def _bounds(differences: Sequence[float], per_decision: float) -> tuple[float, float]:
    """Confidence bounds on the mean paired difference."""
    centre = statistics.fmean(differences)
    spread = statistics.stdev(differences) / math.sqrt(len(differences))
    margin = critical_t(per_decision, len(differences) - 1) * spread
    return centre - margin, centre + margin


def _fingerprint(entrants, policy, delta, seed, key) -> dict:
    return {
        "key": key,
        "labels": sorted(entrant.label for entrant in entrants),
        "protected": sorted(e.label for e in entrants if e.protected),
        "delta": delta,
        "seed": seed,
        "policy": asdict(policy),
    }


def race(
    entrants: Sequence[RaceEntrant],
    time_calls,
    *,
    min_improvement_pct: float,
    policy: RacePolicy = DEFAULT_RACE,
    seed: int = SAMPLE_SEED,
    journal: JsonlBlockJournal | None = None,
    resume: bool = False,
    key: str = "",
    verbose: bool = True,
    report=print,
) -> RaceResult:
    """Eliminate candidates that are worse than the leader by more than delta.

    ``delta`` is ``min_improvement_pct / 100``, the promotion bar, so a
    candidate the race calls tied is one the gate after it would not promote.

    Elimination and stopping read the same paired confidence bound against
    the leader but ask opposite questions of it, which is where the budget is
    saved. Dropping a candidate needs proof it is more than delta *worse*.
    Stopping needs proof only that no survivor is more than delta *better*.

    The guarantee, for near-Gaussian visit medians: with probability at least
    ``1 - alpha``, a certified race returns a winner no more than delta slower
    than the best entrant. Half the budget covers eliminating the true best,
    half covers certifying while it is more than delta faster. The winner is
    the leader, or a protected incumbent that passes the same stopping test
    (no survivor proven more than delta faster than it), so the guarantee
    holds for whichever is returned.

    Elimination is permanent and the leader is recomputed every block, so a
    comparison always uses the best evidence so far. Candidates are compared
    on the blocks they both ran, which keeps the pairing even though they
    leave at different times.

    Protected entrants are measured but never removed. One the same test would
    have dropped is reported ``protected_behind``.

    A visit that raises or returns a reading that is not positive and finite
    is run once more; an entrant leaves the race as ``crashed`` only if the
    retry fails too.

    ``survivors`` lists the leader plus whatever measured inside the
    indifference zone, for reporting. ``key`` names the problem being raced
    (a shape key) in the journal's fingerprint.
    """
    if not 0.0 < min_improvement_pct < 100.0:
        raise ValueError(
            f"min_improvement_pct must be in (0, 100), got {min_improvement_pct}"
        )
    delta = min_improvement_pct / 100.0
    rng = random.Random(seed)
    samples = {entrant.label: Samples() for entrant in entrants}
    by_label = {entrant.label: entrant for entrant in entrants}
    active = [entrant.label for entrant in entrants]
    eliminated: dict[str, tuple[int, float]] = {}
    behind: dict[str, float] = {}
    # A candidate that faults or times as zero leaves the field rather than
    # taking the race down with it, and rather than winning on a bad number.
    crashed: dict[str, str] = {}

    replay: list[dict] = []
    crash_after: dict[int, dict[str, str]] = {}
    if journal is not None:
        fingerprint = _fingerprint(entrants, policy, delta, seed, key)
        records = journal.records() if resume else []
        if not records:
            journal.append({"race": fingerprint})
        elif records[0].get("race") != fingerprint:
            raise ValueError(
                f"journal {journal.path} records a different race "
                f"({records[0].get('race')}); refusing to resume from it"
            )
        replay = [record for record in records if "latencies" in record]
        for record in records:
            if "crashed" in record:
                crash_after.setdefault(record["block"], {})[record["crashed"]] = record[
                    "reason"
                ]
        visits = [record for record in records if "visiting" in record]
        if visits and visits[-1]["block"] > len(replay):
            label = visits[-1]["visiting"]
            if not any(label in reasons for reasons in crash_after.values()):
                reason = "the process died while timing it"
                crash_after.setdefault(len(replay), {})[label] = reason
                journal.append(
                    {"crashed": label, "reason": reason, "block": len(replay)}
                )
    replayed = 0

    def leave(label: str, reason: str) -> None:
        crashed[label] = reason
        if label in active:
            active.remove(label)
            if verbose:
                report(f"  {label} left the race: {reason}")

    def visit(entrant: RaceEntrant) -> tuple[list[float] | None, str, int]:
        """Time one visit, retrying once; returns (latencies, reason, calls)."""
        calls, reason = 0, ""
        for _ in range(2):
            try:
                if policy.warmup_calls:
                    time_calls(entrant, policy.warmup_calls)
                    calls += policy.warmup_calls
                measured = time_calls(entrant, policy.block_calls)
            except Exception as error:  # noqa: BLE001 - a fault is a verdict
                reason = f"{type(error).__name__}: {error}"
                continue
            calls += len(measured)
            if _usable_latencies(measured):
                return measured, "", calls
            reason = "timer returned a latency that is not positive and finite"
        return None, reason, calls

    for label, reason in crash_after.get(0, {}).items():
        leave(label, reason)

    # Union bound over every candidate and every look, since peeking after
    # each block is repeated testing and an uncorrected alpha would drift.
    per_decision = policy.alpha / max(1, len(entrants) * policy.max_blocks)

    calls_spent = 0
    history: list[BlockRecord] = []
    certified = False

    for block_index in range(policy.max_blocks):
        block = block_index + 1
        block_started = time.perf_counter()
        if not active:
            raise RuntimeError(
                "every entrant failed to time: "
                + "; ".join(f"{k} ({v})" for k, v in crashed.items())
            )
        order = [by_label[label] for label in active]
        rng.shuffle(order)

        # Replay and live measurement share this loop so a resumed race cannot
        # reach a different verdict than an uninterrupted one.
        record = replay[block_index] if block_index < len(replay) else None
        if record is not None and set(record["order"]) != set(active):
            # The journal has stopped describing this race; keep the blocks
            # that matched and measure the rest.
            if verbose:
                report(f"  journal diverges at block {block}; measuring live")
            replay = replay[:block_index]
            record = None
        if record is not None:
            order = [by_label[label] for label in record["order"]]
            for entrant in order:
                if entrant.label in record["latencies"]:
                    samples[entrant.label].blocks.append(
                        record["latencies"][entrant.label]
                    )
            calls_spent += record.get("calls", 0)
            replayed += 1
            for label, reason in crash_after.get(block, {}).items():
                leave(label, reason)
        else:
            latencies, failures, block_calls = {}, {}, 0
            for entrant in order:
                if journal is not None:
                    journal.append({"visiting": entrant.label, "block": block})
                measured, reason, calls = visit(entrant)
                block_calls += calls
                if measured is None:
                    failures[entrant.label] = reason
                    continue
                samples[entrant.label].blocks.append(measured)
                latencies[entrant.label] = measured
            calls_spent += block_calls
            for label, reason in failures.items():
                if journal is not None:
                    journal.append({"crashed": label, "reason": reason, "block": block})
                leave(label, reason)
            if not active:
                raise RuntimeError(
                    "every entrant failed to time: "
                    + "; ".join(f"{k} ({v})" for k, v in crashed.items())
                )
            if journal is not None:
                journal.append(
                    {
                        "block": block,
                        "order": [entrant.label for entrant in order],
                        "latencies": latencies,
                        "calls": block_calls,
                    }
                )

        history.append(
            BlockRecord(
                block, len(active), calls_spent, time.perf_counter() - block_started
            )
        )
        if block < policy.min_blocks:
            continue

        leader = min(active, key=lambda label: samples[label].estimate)
        tolerance = delta * samples[leader].estimate

        undecided, dropped = [], []
        behind = {}
        for label in active:
            if label == leader:
                continue
            differences = _paired(samples, label, leader)
            if len(differences) < 2:
                undecided.append(label)
                continue
            lower, _ = _bounds(differences, per_decision)
            if lower > tolerance:
                if by_label[label].protected:
                    # Kept in the field, but the evidence says it is behind.
                    behind[label] = (
                        samples[label].estimate / samples[leader].estimate - 1.0
                    )
                else:
                    dropped.append(label)
            elif lower <= -tolerance:
                undecided.append(label)  # could be > delta better than the leader

        for label in dropped:
            eliminated[label] = (
                block,
                samples[label].estimate / samples[leader].estimate - 1.0,
            )
            active.remove(label)
        history[-1].eliminated = len(dropped)

        if verbose:
            report(
                f"  block {block:>3}: {len(active):>3} active, "
                f"{len(dropped):>2} eliminated, {len(undecided):>2} undecided, "
                f"leader {samples[leader].estimate:8.1f} us"
            )
        if not undecided:
            certified = True
            break

    leader = min(active, key=lambda label: samples[label].estimate)
    leader_estimate = samples[leader].estimate
    tolerance = delta * leader_estimate

    def certifies(label: str) -> bool:
        """No other survivor is proven to be more than delta faster."""
        for other in active:
            if other == label:
                continue
            differences = _paired(samples, other, label)
            if len(differences) < 2:
                return False
            lower, _ = _bounds(differences, per_decision)
            if lower <= -tolerance:
                return False
        return True

    winner, rule = leader, "leader"
    if certified:
        incumbents = sorted(
            (
                label
                for label in active
                if by_label[label].protected and label != leader
            ),
            key=lambda label: (samples[label].estimate, label),
        )
        for label in incumbents:
            if label not in behind and certifies(label):
                winner, rule = label, "incumbent_retained"
                break
    if by_label[leader].protected:
        rule = "incumbent_retained"

    verdicts = [
        Verdict(leader, "leader", leader_estimate, 0.0, len(samples[leader].blocks))
    ]
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
                reason,
            )
        )

    return RaceResult(
        verdicts=verdicts,
        samples=samples,
        calls_spent=calls_spent,
        history=history,
        certified=certified,
        winner=winner,
        rule=rule,
        survivors=survivors,
        blocks_run=history[-1].block if history else 0,
        delta=delta,
        blocks_replayed=replayed,
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


@functools.lru_cache(maxsize=4096)
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
