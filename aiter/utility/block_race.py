# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Interleaved randomized-block measurement with delta-based elimination.

A tuner that measures one candidate at a time, in isolation, pays process and
load cost per candidate and compares candidates that never saw the same
machine conditions. This measures the whole field in one process as a
randomized complete block design: each block visits every surviving candidate
once, in a fresh random order, running a short run of calls per visit.

Blocking is what makes the comparison paired, so drift that moves one
candidate within a block moves them all and cancels in the differences.
Randomizing the order each block is what stops position within a block from
being confounded with the candidate.

Selection, not hypothesis testing, is the objective. Two candidates within
``delta`` of each other are a finished question rather than a harder one,
because whichever is shipped the result is equally fast. That is what keeps
the block count a function of delta and the measurement noise rather than of
the size of the catalogue.

Correctness belongs to the caller. An entrant reaches :func:`race` only
through the timing function, so nothing here ever sees kernel output and a
candidate that computes the wrong answer races exactly like one that does
not. Check it in the same pass that compiles and warms each candidate, before
any timed call: that pass has to happen anyway, and doing the check there
keeps a compile off a measurement as well as a wrong candidate out of the
field. ``_race_one_shape`` in ``op_tests/tuners/tune_mha_fwd.py`` is the
reference implementation.

Nothing here imports torch at module scope. The caller supplies a timing
function, for which :func:`cuda_event_timer` is the GPU implementation, so the
decision logic can be tested on synthetic latencies without a device.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "RaceEntrant",
    "Samples",
    "Verdict",
    "BlockRecord",
    "RaceResult",
    "BlockJournal",
    "JsonlBlockJournal",
    "cuda_event_timer",
    "measure_blocks",
    "race",
    "select_winner",
    "rank",
    "position_effect",
    "indistinguishable_set",
    "wilcoxon_floor",
    "critical_t",
    "student_t_sf",
    "regularized_incomplete_beta",
    "check_t_implementation",
]


@dataclass(frozen=True)
class RaceEntrant:
    """One thing to be measured.

    ``payload`` is opaque to this module: it is whatever the caller needs back
    when the race names a winner. ``protected`` marks an entrant that is
    measured in every block but never eliminated, which is how the
    configuration already in use stays in the field. A run that stops
    measuring the incumbent cannot tell an improvement from a regression.
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

        This is the tie-break statistic. Among candidates that are equally
        fast in expectation, the steadier one is the one more likely to still
        be fast in the next session.
        """
        medians = self.block_medians
        estimate = self.estimate
        if len(medians) < 2 or not math.isfinite(estimate) or estimate <= 0:
            return float("inf")
        return statistics.stdev(medians) / estimate


@dataclass
class Verdict:
    label: str
    # "leader", "within_delta", "protected_behind", "eliminated", "undecided"
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

    The unit of durable progress is the completed block: after block k every
    surviving candidate has been measured k times, and because elimination is
    a pure function of the accumulated samples, replaying those blocks
    reconstructs the state exactly. Nothing needs to be re-measured.
    """

    def append(self, record: dict) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def records(self) -> Iterable[dict]:  # pragma: no cover - interface
        raise NotImplementedError


class JsonlBlockJournal(BlockJournal):
    """A journal backed by one JSON object per line.

    A process killed mid-write leaves one truncated final line. Reading stops
    at the first line that does not parse rather than failing, so a resume
    picks up from the last block that was written whole.
    """

    def __init__(self, path: str, resume: bool = False):
        self.path = path
        if not resume and os.path.exists(path):
            os.remove(path)

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

    Imported lazily so this module stays usable, and testable, without torch.
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
    reference the race is checked against, and what the block-size calibration
    and noise-floor experiments use.
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

    Both decisions read the same lower bound on the paired difference against
    the leader, but they ask opposite questions of it, and the asymmetry is
    where most of the budget is saved. Eliminating a candidate needs proof it
    is more than delta *worse* than the leader. Stopping only needs proof that
    no survivor is more than delta *better* -- that is the only way picking the
    leader could turn out wrong. Certifying the reverse, that a close candidate
    is definitely not slightly worse, costs many blocks and buys nothing,
    because if it were slightly worse we would still be shipping the leader.

    Elimination is permanent and the leader is recomputed every block, so the
    comparison is always against the best evidence so far. Candidates are
    compared on the blocks they both took part in, which keeps every
    comparison paired even though they leave the race at different times.

    Protected entrants are measured like everyone else and are never removed.
    A protected entrant that the same test would have dropped is reported as
    ``protected_behind``, so being kept in the field is not mistaken for being
    within delta of the leader.

    ``survivors`` is the set that may be published: the leader plus whatever
    measured inside the indifference zone. Candidates that merely outlasted
    the budget without being separated are reported ``undecided`` and are not
    eligible, because surviving elimination proves only that the evidence was
    not strong enough to drop them.
    """
    rng = random.Random(seed)
    samples = {entrant.label: Samples() for entrant in entrants}
    by_label = {entrant.label: entrant for entrant in entrants}
    active = [entrant.label for entrant in entrants]
    eliminated: dict[str, tuple[int, float]] = {}
    behind: dict[str, float] = {}

    replay = list(journal.records()) if (journal is not None and resume) else []
    replayed = 0

    # Union bound over every candidate and every look. Peeking after each
    # block is repeated testing, so an uncorrected alpha would drift. This is
    # conservative rather than tight -- an anytime-valid bound such as
    # empirical Bernstein would spend the budget better -- but the eliminations
    # that dominate the cost are decided by factors of ten, where the
    # difference between a tight bound and a loose one is a block at most.
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
            journaled = [label for label in record["order"] if label in by_label]
            if [e.label for e in order] != journaled:
                # A different seed or a changed catalogue. The journal is the
                # record of what actually ran, so it wins.
                order = [by_label[label] for label in journaled]
            for entrant in order:
                samples[entrant.label].blocks.append(record["latencies"][entrant.label])
            calls_spent += sum(len(v) for v in record["latencies"].values())
            replayed += 1
        else:
            latencies = {}
            for entrant in order:
                measured = time_calls(entrant, block_calls)
                samples[entrant.label].blocks.append(measured)
                latencies[entrant.label] = measured
                calls_spent += len(measured)
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
    """Pick one config from a set the race has declared equally fast.

    None of the survivors is meaningfully faster than the others, so this rule
    is not about speed. Its job is to make a re-tune return the same answer and
    leave the config file untouched when nothing genuinely improved.

    The incumbent wins first, because keeping it is the outcome that changes
    nothing. Otherwise the steadiest survivor wins: a variance estimated from a
    dozen blocks is itself noisy, so this is a tie-break and not a ranking
    criterion, but between two configurations that are equally fast in
    expectation it prefers the one whose speed is more repeatable. The final
    fallback is a stable sort on the label, so even exactly tied spreads
    produce the same answer on every run.
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


def position_effect(samples: dict[str, Samples]) -> list[tuple[int, float, int]]:
    """Latency by position within a block, relative to each candidate's median.

    A switching cost that lands inside the timed calls shows up as the first
    positions running slow. Normalizing per candidate lets fast and slow
    candidates be pooled.
    """
    by_position: dict[int, list[float]] = {}
    for sample in samples.values():
        reference = sample.estimate
        if not math.isfinite(reference) or reference <= 0:
            continue
        for block in sample.blocks:
            for position, latency in enumerate(block):
                by_position.setdefault(position, []).append(latency / reference)
    return [
        (position, statistics.median(values), len(values))
        for position, values in sorted(by_position.items())
    ]


def wilcoxon_floor(blocks: int) -> float:
    """Smallest one-sided p a signed-rank test can return with this many pairs.

    Worth printing rather than discovering: at four blocks the floor is 0.0625,
    so no comparison can clear alpha=0.05 and every candidate survives the
    filter no matter how slow it is. That is a powerless test, not a tie.
    """
    return 0.5**blocks if blocks > 0 else 1.0


def indistinguishable_set(samples: dict[str, Samples], alpha: float = 0.05):
    """Candidates that cannot be separated from the fastest.

    Choosing the single fastest point estimate out of many is biased: the
    maximum of noisy estimates is optimistic, and second place is often not
    distinguishable from first. Reporting the set that survives a paired test
    against the leader says what the measurement actually supports, and leaves
    the choice within that set to a policy that can prefer the incumbent.
    """
    ordered = rank(samples)
    best_label = ordered[0][0]
    best_blocks = samples[best_label].block_medians

    raw = []
    for label, _ in ordered[1:]:
        blocks = samples[label].block_medians
        paired = min(len(best_blocks), len(blocks))
        if paired < 3:
            raw.append((label, 1.0))
            continue
        differences = [blocks[i] - best_blocks[i] for i in range(paired)]
        if all(d == 0 for d in differences):
            raw.append((label, 1.0))
            continue
        # A paired t on the block differences rather than a signed-rank test.
        # Rank tests are distribution-free but discard effect size, so their
        # smallest attainable p depends only on the number of blocks: at eight
        # blocks the floor is 1/256, which is above the Holm threshold once
        # there are sixteen comparisons, and a candidate twenty-five times
        # slower than the leader is declared a tie. The block values being
        # compared are already medians of many calls, so approximate normality
        # is a far weaker assumption here than at the level of raw latencies.
        spread = statistics.stdev(differences) / math.sqrt(len(differences))
        if spread <= 0.0:
            raw.append((label, 0.0))
            continue
        statistic = statistics.mean(differences) / spread
        raw.append((label, student_t_sf(statistic, len(differences) - 1)))

    # Holm-Bonferroni: the leader is compared against every other candidate, so
    # without correction the chance of wrongly excluding one grows with the
    # size of the catalogue.
    raw.sort(key=lambda item: item[1])
    total = len(raw)
    survivors = [best_label]
    for index, (label, p_value) in enumerate(raw):
        if p_value > alpha / (total - index):
            # Holm stops at the first failure; everything from here on stays.
            survivors.extend(other for other, _ in raw[index:])
            break
    return survivors


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


def critical_t(confidence: float, degrees: int) -> float:
    """Two-sided critical value: t with P(|T| > t) == confidence.

    Bisection rather than a closed form. The Cornish-Fisher expansion from the
    normal quantile is the usual shortcut, but it is worst exactly where this
    is used -- few degrees of freedom and a far tail, where the Bonferroni
    correction puts the per-decision alpha -- so it is not worth the risk.
    """
    degrees = max(1, degrees)
    target = confidence / 2.0
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


def check_t_implementation() -> None:
    """Pin the hand-rolled t against printed tables before spending GPU time.

    The continued fraction above is the one piece of this file that can be
    wrong without looking wrong: a subtly bad critical value does not raise,
    it just eliminates candidates the evidence does not support. Published
    two-sided critical values are an external check that costs microseconds.
    """
    table = {
        (1, 0.05): 12.706,
        (2, 0.05): 4.303,
        (5, 0.05): 2.571,
        (10, 0.05): 2.228,
        (29, 0.05): 2.045,
        (2, 0.01): 9.925,
        (10, 0.01): 3.169,
        (29, 0.001): 3.659,
    }
    for (degrees, confidence), expected in table.items():
        actual = critical_t(confidence, degrees)
        if abs(actual - expected) > 0.001:
            raise AssertionError(
                f"t_{{{degrees}}}({confidence}) computed {actual:.4f}, "
                f"tables say {expected:.4f}"
            )
