"""Prototype: interleaved randomized-block measurement of an MHA candidate set.

The shipped tuner measures one candidate per task in a fresh worker, which
costs about 2.2 s of process spawn, allocation and kernel load to buy 0.19 s
of timing. It also measures each candidate in isolation, so every candidate
sees a different slice of whatever the machine was doing, and a single
contended measurement during screening drops a candidate permanently.

This measures the whole catalogue in one process as a randomized complete
block design. Each block visits every candidate once in a fresh random order,
running a short run of calls per visit. Blocking is what makes the comparison
paired: drift that moves one candidate moves them all within a block, so it
cancels in the differences. Randomizing the order each block is what stops
position within a block from being confounded with the candidate, which a
fixed order or a strict alternation cannot do once there are more than two.

Nothing here is settled. Two questions have to be answered by measurement
before this becomes a design:

  1. Does interleaving bias a candidate's estimate relative to measuring it
     contiguously? Switching kernels perturbs instruction and data cache, and
     if that cost lands inside the timed calls then short blocks are biased
     rather than merely noisy.
  2. What is the smallest block that does not pay that cost? Shorter blocks
     buy more paired observations for the same number of calls, so the answer
     sets how much statistical power a fixed budget can produce.

Run with PYTHONPATH set to the repository root and HIP_VISIBLE_DEVICES
pinned, e.g.
    HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD python -m \
        op_tests.tuners.prototype_interleaved_bench
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass, field

import torch

from aiter.ops.mha_fwd_policy import enumerate_mha_fwd_candidates
from aiter.ops.triton.attention.mha import flash_attn_varlen_func

# The Kimi hd192/hdv128 varlen shape the storage comparison has been using.
SHAPE = {
    "total_q": 4096,
    "total_k": 42700,
    "max_seqlen_q": 4096,
    "max_seqlen_k": 42700,
    "nhead": 12,
    "hdim_q": 192,
    "hdim_v": 128,
}


@dataclass(frozen=True)
class Candidate:
    backend: str
    config: dict
    origin: str = "catalogue"
    tag: str = ""

    @property
    def label(self) -> str:
        base = f"{self.backend}:{json.dumps(self.config, sort_keys=True)}"
        return f"{base}#{self.tag}" if self.tag else base


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


def build_inputs(device: str = "cuda"):
    dtype = torch.bfloat16
    q = torch.randn(
        SHAPE["total_q"], SHAPE["nhead"], SHAPE["hdim_q"], device=device, dtype=dtype
    )
    k = torch.randn(
        SHAPE["total_k"], SHAPE["nhead"], SHAPE["hdim_q"], device=device, dtype=dtype
    )
    v = torch.randn(
        SHAPE["total_k"], SHAPE["nhead"], SHAPE["hdim_v"], device=device, dtype=dtype
    )
    cu_q = torch.tensor([0, SHAPE["total_q"]], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, SHAPE["total_k"]], device=device, dtype=torch.int32)
    return q, k, v, cu_q, cu_k


def make_invoker(inputs):
    q, k, v, cu_q, cu_k = inputs
    scale = SHAPE["hdim_q"] ** -0.5

    def invoke(candidate: Candidate):
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            SHAPE["max_seqlen_q"],
            SHAPE["max_seqlen_k"],
            dropout_p=0.0,
            softmax_scale=scale,
            causal=False,
            window_size=(-1, -1),
            return_lse=False,
            config=dict(candidate.config),
            backend=candidate.backend,
        )

    return invoke


def collect_candidates(gfx: str) -> list[Candidate]:
    """The restricted catalogue plus whatever each kernel resolves today.

    The incumbents are part of the field by construction rather than a second
    phase: a run that never measures the configuration already in use cannot
    tell an improvement from a regression, and adding them here costs one more
    visit per block.
    """
    candidates = [
        Candidate(c.backend, dict(c.backend_config))
        for c in enumerate_mha_fwd_candidates(gfx, "smoke", ["triton", "gluon"])
    ]
    seen = {c.label for c in candidates}

    from aiter.ops.triton.utils.attention_config_utils import format_mha_shape_key

    shape_key = format_mha_shape_key(
        mode="varlen",
        hdim_q=SHAPE["hdim_q"],
        hdim_v=SHAPE["hdim_v"],
        nhead_q=SHAPE["nhead"],
        nhead_k=SHAPE["nhead"],
        dtype="bfloat16",
        causal=False,
        max_seqlen_q=SHAPE["max_seqlen_q"],
        max_seqlen_k=SHAPE["max_seqlen_k"],
    )
    for backend in ("gluon", "triton"):
        try:
            if backend == "gluon":
                from aiter.ops.triton._gluon_kernels.gfx950.attention.mha import (
                    _get_config as resolve,
                )

                config = resolve(is_fp8=False, has_pe=False, shape_key=shape_key)
            else:
                from aiter.ops.triton._triton_kernels.attention.mha import (
                    _get_config as resolve,
                )

                config = resolve(
                    False,
                    torch.bfloat16,
                    has_pe=False,
                    head_dim_v=SHAPE["hdim_v"],
                    shape_key=shape_key,
                )
        except Exception as error:  # noqa: BLE001 - a missing default is not fatal
            print(f"  no incumbent for {backend}: {error}")
            continue
        incumbent = Candidate(backend, dict(config), origin="incumbent")
        if incumbent.label in seen:
            # Already in the catalogue; relabel so the report can point at it.
            candidates = [
                Candidate(c.backend, c.config, "catalogue+incumbent")
                if c.label == incumbent.label
                else c
                for c in candidates
            ]
        else:
            candidates.append(incumbent)
    return candidates


def screen(candidates: list[Candidate], invoke, warmup: int) -> list[Candidate]:
    """Drop candidates that cannot run this problem, and warm the rest.

    Done once, outside the blocks, so a compile or load cost is never charged
    to a timed call.
    """
    survivors = []
    for candidate in candidates:
        try:
            for _ in range(warmup):
                invoke(candidate)
            torch.cuda.synchronize()
        except Exception as error:  # noqa: BLE001 - unsupported is an outcome
            print(f"  dropped {candidate.label}: {type(error).__name__}")
            continue
        survivors.append(candidate)
    return survivors


def measure_blocks(
    candidates: list[Candidate],
    invoke,
    block_calls: int,
    blocks: int,
    seed: int,
    discard_per_block: int = 0,
) -> dict[str, Samples]:
    """Run a randomized complete block design over the candidate set."""
    rng = random.Random(seed)
    samples = {candidate.label: Samples() for candidate in candidates}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(blocks):
        order = list(candidates)
        rng.shuffle(order)
        for candidate in order:
            for _ in range(discard_per_block):
                invoke(candidate)
            latencies = []
            for _ in range(block_calls):
                start.record()
                invoke(candidate)
                end.record()
                end.synchronize()
                latencies.append(start.elapsed_time(end) * 1000.0)
            samples[candidate.label].blocks.append(latencies)
    return samples


def measure_contiguous(
    candidates: list[Candidate], invoke, calls: int
) -> dict[str, Samples]:
    """The current harness's shape: every call for a candidate back to back.

    This is the reference the interleaved estimates are checked against. If
    they disagree, interleaving is biased and block size is not merely a
    power-versus-cost trade.
    """
    return measure_blocks(candidates, invoke, calls, 1, seed=0)


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


@dataclass
class Verdict:
    label: str
    state: str  # "leader", "within_delta", "eliminated", "undecided"
    estimate: float
    relative_gap: float
    blocks_used: int
    note: str = ""


def race(
    candidates: list[Candidate],
    invoke,
    delta: float,
    alpha: float,
    block_calls: int,
    min_blocks: int,
    max_blocks: int,
    seed: int,
    verbose: bool = True,
):
    """Eliminate candidates that are worse than the leader by more than delta.

    The objective is selection, not hypothesis testing. Family-wise error
    control answers "which candidates are provably different", which is both
    more than we need and unboundedly expensive: separating an arbitrarily
    small difference takes arbitrarily many blocks, so the closer the field is
    packed the more it costs. Discarding a candidate genuinely tied with the
    best costs nothing, because whatever we ship instead is equally fast. An
    indifference zone says so directly -- two candidates within delta are a
    finished question, not a harder one -- which makes the block count a
    function of delta and the measurement noise rather than of the catalogue.

    Delta should be set from reproducibility, not taste. Measuring the same
    configuration in different sessions on this machine moves it by about
    1.8%, so a delta below that would be resolving differences that do not
    survive to the next run.

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
    """
    rng = random.Random(seed)
    samples = {c.label: Samples() for c in candidates}
    by_label = {c.label: c for c in candidates}
    active = [c.label for c in candidates]
    eliminated: dict[str, tuple[int, float]] = {}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Union bound over every candidate and every look. Peeking after each
    # block is repeated testing, so an uncorrected alpha would drift. This is
    # conservative rather than tight -- an anytime-valid bound such as
    # empirical Bernstein would spend the budget better -- but the eliminations
    # that dominate the cost are decided by factors of ten, where the
    # difference between a tight bound and a loose one is a block at most.
    per_decision = alpha / max(1, len(candidates) * max_blocks)

    calls_spent = 0
    history = []
    certified = False

    for block_index in range(max_blocks):
        order = [by_label[label] for label in active]
        rng.shuffle(order)
        for candidate in order:
            latencies = []
            for _ in range(block_calls):
                start.record()
                invoke(candidate)
                end.record()
                end.synchronize()
                latencies.append(start.elapsed_time(end) * 1000.0)
            samples[candidate.label].blocks.append(latencies)
            calls_spent += block_calls

        history.append((block_index + 1, len(active), calls_spent))
        if block_index + 1 < min_blocks:
            continue

        leader = min(active, key=lambda label: samples[label].estimate)
        leader_blocks = samples[leader].block_medians
        leader_estimate = samples[leader].estimate
        tolerance = delta * leader_estimate

        undecided, dropped = [], []
        for label in active:
            if label == leader:
                continue
            blocks = samples[label].block_medians
            paired = min(len(leader_blocks), len(blocks))
            differences = [blocks[i] - leader_blocks[i] for i in range(paired)]
            if len(differences) < 2:
                undecided.append(label)
                continue
            mean = statistics.mean(differences)
            spread = statistics.stdev(differences) / math.sqrt(len(differences))
            lower = mean - critical_t(per_decision, len(differences) - 1) * spread
            if lower > tolerance:
                dropped.append(label)      # worse than the leader by > delta
            elif lower <= -tolerance:
                undecided.append(label)    # could be > delta better than the leader

        for label in dropped:
            eliminated[label] = (
                block_index + 1,
                samples[label].estimate / leader_estimate - 1.0,
            )
            active.remove(label)

        if verbose:
            print(
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
        Verdict(leader, "leader", leader_estimate, 0.0, len(samples[leader].blocks))
    ]
    for label in active:
        if label == leader:
            continue
        verdicts.append(
            Verdict(
                label,
                "within_delta",
                samples[label].estimate,
                samples[label].estimate / leader_estimate - 1.0,
                len(samples[label].blocks),
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
    return verdicts, samples, calls_spent, history, certified


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


def run_race(args, candidates, invoke) -> None:
    print(
        f"\nracing {len(candidates)} candidates, delta={args.delta:.1%}, "
        f"block={args.block_calls} calls, at most {args.max_blocks} blocks"
    )
    verdicts, samples, calls, history, certified = race(
        candidates,
        invoke,
        delta=args.delta,
        alpha=args.alpha,
        block_calls=args.block_calls,
        min_blocks=args.min_blocks,
        max_blocks=args.max_blocks,
        seed=args.seed,
    )

    kernel_seconds = (
        sum(sum(sum(b) for b in s.blocks) for s in samples.values()) / 1e6
    )
    exhaustive = args.max_blocks * args.block_calls * sum(
        s.estimate for s in samples.values() if math.isfinite(s.estimate)
    ) / 1e6
    print(
        f"\nspent {calls} calls / {kernel_seconds:.1f} s of kernel time; "
        f"measuring every candidate for all {args.max_blocks} blocks would "
        f"have cost {exhaustive:.1f} s ({exhaustive / max(kernel_seconds, 1e-9):.1f}x)"
    )

    blocks_run = history[-1][0]
    print(
        f"converged after {blocks_run} blocks: the winner is within "
        f"{args.delta:.0%} of the best candidate in the catalogue"
        if certified
        else f"budget exhausted at {blocks_run} blocks without certifying the "
        f"winner; treat the result as a ranking, not a guarantee"
    )

    survivors = [v for v in verdicts if v.state in ("leader", "within_delta")]
    print(f"\nwithin delta of the best: {len(survivors)} of {len(candidates)}")
    for verdict in sorted(survivors, key=lambda v: v.estimate):
        origin = next(
            (c.origin for c in candidates if c.label == verdict.label), "catalogue"
        )
        mark = "  <- incumbent" if "incumbent" in origin else ""
        print(
            f"  {verdict.estimate:9.1f} us  {verdict.relative_gap:+6.2%}  "
            f"{verdict.label}{mark}"
        )

    incumbent_survives = any(
        "incumbent"
        in next(
            (c.origin for c in candidates if c.label == v.label), "catalogue"
        )
        for v in survivors
    )
    print(
        "\nthe incumbent is inside the indifference zone, so keeping it is the "
        "outcome that changes nothing"
        if incumbent_survives
        else "\nno incumbent survived: the winner is a real improvement, not a tie"
    )

    dropped = sorted(
        (v for v in verdicts if v.state == "eliminated"), key=lambda v: v.blocks_used
    )
    print(f"\neliminated {len(dropped)}, and how early:")
    for verdict in dropped:
        print(
            f"  after {verdict.blocks_used:>3} blocks  {verdict.relative_gap:+8.1%}  "
            f"{verdict.label.split(':')[0]:>7}  {verdict.estimate:9.1f} us"
        )


def run_null(args, candidates, invoke) -> None:
    """Race one configuration against copies of itself.

    Every other experiment here can only compare procedures against each
    other, because the true ranking of two different configurations is never
    known. Replicas make it known: the truth is that all of them are equally
    fast, so any separation the race reports is measurement error and any
    elimination is a false positive with a known correct answer.

    That turns delta from a judgement call into a measurement. A delta below
    the spread seen here is asking the machine for a distinction it cannot
    make twice, and no amount of extra blocks will fix it, because the
    variation is between sessions and not within them.
    """
    seed = next(
        (c for c in candidates if "incumbent" in c.origin), candidates[0]
    )
    replicas = [
        Candidate(seed.backend, seed.config, "replica", f"r{i}")
        for i in range(args.replicas)
    ]
    print(
        f"\nnull experiment: {args.replicas} copies of one configuration\n"
        f"  {seed.backend}:{json.dumps(seed.config, sort_keys=True)}\n"
        f"  ground truth: every gap is zero, so every elimination is an error"
    )

    verdicts, samples, calls, history, certified = race(
        replicas,
        invoke,
        delta=args.delta,
        alpha=args.alpha,
        block_calls=args.block_calls,
        min_blocks=args.min_blocks,
        max_blocks=args.max_blocks,
        seed=args.seed,
    )

    estimates = sorted(s.estimate for s in samples.values())
    spread = estimates[-1] / estimates[0] - 1.0
    false_positives = [v for v in verdicts if v.state == "eliminated"]
    print(
        f"\nspent {calls} calls over {history[-1][0]} blocks\n"
        f"  fastest replica  {estimates[0]:9.1f} us\n"
        f"  slowest replica  {estimates[-1]:9.1f} us\n"
        f"  spread across identical configs: {spread:.2%}\n"
        f"  false eliminations: {len(false_positives)} of {args.replicas - 1}"
    )
    for verdict in false_positives:
        print(f"    wrongly dropped {verdict.label} at {verdict.relative_gap:+.2%}")
    print(
        f"\ndelta={args.delta:.1%} is "
        + (
            "above the observed spread, so the race is not being asked to "
            "resolve differences this machine cannot reproduce"
            if args.delta > spread
            else "BELOW the observed spread: the race is resolving differences "
            "smaller than the noise between identical configurations"
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gfx", default="gfx950")
    parser.add_argument(
        "--mode",
        choices=("race", "sweep", "null"),
        default="race",
        help="race runs delta-based elimination; sweep is the block-size "
        "calibration that chose the default block; null races a configuration "
        "against copies of itself to measure the noise floor that bounds delta "
        "from below",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=8,
        help="null mode only: how many copies of the one configuration to race",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.02,
        help="indifference zone. Differences smaller than this are treated as "
        "settled rather than as a harder question. Default 2%% is just above "
        "the measured session-to-session drift of an unchanged configuration",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--block-calls",
        type=int,
        default=10,
        help="calls per visit. 10 was chosen by --mode sweep: the switching "
        "cost sits in the first call of a block, so a block of 10 carries "
        "0.07%% bias against a contiguous measurement while a block of 1 "
        "carries 0.82%%",
    )
    parser.add_argument("--min-blocks", type=int, default=3)
    parser.add_argument("--max-blocks", type=int, default=30)
    parser.add_argument(
        "--calls-per-candidate",
        type=int,
        default=200,
        help="sweep mode only: held constant across block sizes so the sweep "
        "isolates block size from measurement budget",
    )
    parser.add_argument(
        "--block-sizes",
        default="1,2,5,10,25,50",
        help="sweep mode only: comma-separated calls per visit",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20240917)
    args = parser.parse_args()

    check_t_implementation()
    torch.cuda.init()
    inputs = build_inputs()
    invoke = make_invoker(inputs)

    print(f"building catalogue for {args.gfx}")
    candidates = collect_candidates(args.gfx)
    print(f"  {len(candidates)} candidates before screening")
    candidates = screen(candidates, invoke, args.warmup)
    print(f"  {len(candidates)} survive")
    for candidate in candidates:
        if candidate.origin != "catalogue":
            print(f"  incumbent: {candidate.label}")

    if args.mode == "race":
        run_race(args, candidates, invoke)
        return
    if args.mode == "null":
        run_null(args, candidates, invoke)
        return

    print(f"\nreference pass: {args.calls_per_candidate} contiguous calls each")
    reference = measure_contiguous(candidates, invoke, args.calls_per_candidate)
    reference_estimate = {label: s.estimate for label, s in reference.items()}

    block_sizes = [int(b) for b in args.block_sizes.split(",") if b.strip()]
    print(
        f"\nsweeping block size at a fixed budget of "
        f"{args.calls_per_candidate} calls per candidate"
    )
    print(
        f"{'block':>6} {'blocks':>7} {'wall s':>8} {'kernel s':>9} "
        f"{'median bias':>12} {'max |bias|':>11} {'test floor':>11} {'winner':>8}"
    )

    results = {}
    for block_calls in block_sizes:
        blocks = max(2, args.calls_per_candidate // block_calls)
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        samples = measure_blocks(
            candidates, invoke, block_calls, blocks, seed=args.seed
        )
        finished.record()
        finished.synchronize()
        wall = started.elapsed_time(finished) / 1000.0

        # Signed and unsigned are different questions. A systematic switching
        # cost shifts every candidate the same way and shows in the median;
        # plain noise from short blocks shows only in the maximum.
        signed = [
            (sample.estimate - reference_estimate[label]) / reference_estimate[label]
            for label, sample in samples.items()
            if math.isfinite(reference_estimate.get(label, float("inf")))
        ]
        kernel_seconds = sum(
            sum(sum(block) for block in sample.blocks) for sample in samples.values()
        ) / 1e6
        winner = rank(samples)[0][0].split(":")[0]
        print(
            f"{block_calls:>6} {blocks:>7} {wall:>8.1f} {kernel_seconds:>9.1f} "
            f"{statistics.median(signed) if signed else float('nan'):>11.2%} "
            f"{max(abs(b) for b in signed) if signed else float('nan'):>10.2%} "
            f"{wilcoxon_floor(blocks):>11.4f} {winner:>8}"
        )
        results[block_calls] = samples

    print("\nlatency by position within a block (median, normalized per candidate)")
    print("a switching cost inside the timed calls shows up as slow early positions")
    for block_calls in block_sizes:
        if block_calls < 5:
            continue
        effect = position_effect(results[block_calls])
        head = "  ".join(f"p{p}={v:.3f}" for p, v, _ in effect[:6])
        print(f"  block={block_calls:>3}: {head}")

    # Analyse at the largest block whose block count still leaves the test able
    # to reject anything. Reporting a survivor set from a powerless test would
    # read as "everything ties" when it means "we learned nothing".
    # The bar a comparison has to clear is alpha divided by the number of
    # comparisons, not alpha, so multiplicity belongs in the capability check.
    comparisons = max(1, len(candidates) - 1)
    strictest = 0.05 / comparisons
    capable = [
        b
        for b in block_sizes
        if wilcoxon_floor(max(2, args.calls_per_candidate // b)) <= strictest
    ]
    analysis_block = max(capable) if capable else min(block_sizes)
    analysis_blocks = max(2, args.calls_per_candidate // analysis_block)
    print(f"\nranking at block={analysis_block} ({analysis_blocks} blocks)")
    print(
        f"  {comparisons} comparisons -> Holm's strictest threshold is "
        f"{strictest:.5f}; a rank test needs "
        f"{math.ceil(math.log2(1 / strictest))} blocks to reach it"
    )
    for label, estimate in rank(results[analysis_block])[:6]:
        origin = next(
            (c.origin for c in candidates if c.label == label), "catalogue"
        )
        mark = "  <- incumbent" if "incumbent" in origin else ""
        print(f"  {estimate:9.1f} us  {label}{mark}")

    survivors = indistinguishable_set(results[analysis_block])
    print(
        f"\ncandidates indistinguishable from the fastest: "
        f"{len(survivors)} of {len(candidates)}"
    )
    for label in survivors:
        origin = next(
            (c.origin for c in candidates if c.label == label), "catalogue"
        )
        mark = "  <- incumbent" if "incumbent" in origin else ""
        print(f"  {results[analysis_block][label].estimate:9.1f} us  {label}{mark}")


if __name__ == "__main__":
    main()
