# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Noise-aware comparison of timing samples (pure Python, no SciPy).

Kernel times are compared on the log scale (a ratio is what matters, and
timing noise is roughly multiplicative). ``margin`` is a practical-
equivalence margin: "slower" means slower by more than ``margin``
(relative), so two configs within noise of each other are never declared
different.

Units of replication. Timing noise has two parts: rep-to-rep jitter within
one measurement call, and a per-call part (clock and thermal state,
placement, code layout) that every rep of the call shares and more reps do
not average away. Reps of one call are therefore *not* independent samples
of a config's time. :class:`Race` works on calls: one value per measurement
call (its median), tagged with the interleaved batch it was taken in. Calls
of two configs in the same batch pair up (a paired t test on their log
ratios); otherwise calls are compared unpaired (Welch's t on the logs), and
a config with too few calls for a t test is compared with a z test against
the per-call noise a :class:`NoiseModel` learned from repeated calls, or
not at all.

:class:`Race` drops configs that are statistically slower than some other
config, testing every ordered pair with Holm's correction over all of them
(so picking the leader from the same data cannot inflate the error), at
``alpha / looks`` when the caller races the same configs ``looks`` times
(the rungs of successive halving).

:func:`wilcoxon_greater` (exact signed-rank) is kept for callers with many
independent paired calls.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "Call",
    "NoiseModel",
    "Race",
    "holm",
    "median",
    "normal_sf",
    "paired_t_greater",
    "rel_spread",
    "student_t_sf",
    "welch_greater",
    "wilcoxon_greater",
]

_EXACT_MAX = 60


def median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return math.inf
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def rel_spread(xs: Sequence[float]) -> float:
    """Robust relative spread: 1.4826 * MAD of the logs (about the relative
    standard deviation for small multiplicative noise)."""
    logs = [math.log(x) for x in xs if x > 0 and math.isfinite(x)]
    if len(logs) < 2:
        return 0.0
    m = median(logs)
    return 1.4826 * median([abs(v - m) for v in logs])


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks (1-based) with ties sharing their mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def wilcoxon_greater(diffs: Sequence[float], shift: float = 0.0) -> float:
    """One-sided p-value for "the median of ``diffs`` exceeds ``shift``".

    Zero differences (after the shift) are dropped (Wilcoxon's convention).
    Exact null distribution by dynamic programming over doubled ranks (ties
    included) up to 60 pairs, normal approximation with tie and continuity
    corrections above that. Returns 1.0 with no informative pair.
    """
    d = [x - shift for x in diffs if math.isfinite(x)]
    d = [x for x in d if x != 0.0]
    n = len(d)
    if n == 0:
        return 1.0
    ranks = _ranks([abs(x) for x in d])
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    if n <= _EXACT_MAX:
        weights = [round(2 * r) for r in ranks]
        total = sum(weights)
        counts = [0] * (total + 1)
        counts[0] = 1
        for w in weights:
            for s in range(total, w - 1, -1):
                counts[s] += counts[s - w]
        obs = round(2 * w_plus)
        tail = sum(counts[obs:])
        return tail / float(2**n)
    mean = n * (n + 1) / 4.0
    tie_term = 0.0
    seen: dict[float, int] = {}
    for r in ranks:
        seen[r] = seen.get(r, 0) + 1
    for t in seen.values():
        tie_term += t**3 - t
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return 1.0
    z = (w_plus - mean - 0.5) / math.sqrt(var)
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_sf(t: float, df: float) -> float:
    """P(T > t) for Student's t with ``df`` degrees of freedom."""
    if not math.isfinite(t):
        return 0.0 if t > 0 else 1.0
    x = df / (df + t * t)
    tail = 0.5 * _betainc(0.5 * df, 0.5, x)
    return tail if t > 0 else 1.0 - tail


def welch_greater(a: Sequence[float], b: Sequence[float], shift: float = 0.0) -> float:
    """One-sided Welch p-value for "mean(a) - mean(b) exceeds ``shift``"."""
    a = [x for x in a if math.isfinite(x)]
    b = [x for x in b if math.isfinite(x)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 1.0
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se2 = va / na + vb / nb
    diff = ma - mb - shift
    if se2 <= 0.0:
        return 0.0 if diff > 0 else 1.0
    t = diff / math.sqrt(se2)
    df_den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = se2 * se2 / df_den if df_den > 0 else float(na + nb - 2)
    return student_t_sf(t, df)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


@dataclass
class NoiseModel:
    """Relative timing noise, learned from measurements.

    Two components (log scale): ``rel_sigma``, the rep-to-rep spread within
    one call (pooled from each call's samples), and ``call_sigma``, the
    spread of one call's median across calls of the same config (learned
    from pairs of calls, e.g. multi-rung or confirmation re-measures;
    ``None`` until ``min_pairs`` pairs were seen). ``floor`` is the smallest
    relative noise ever assumed (timer resolution, clock jitter).
    :meth:`median_sigma` is the noise of one call's median, the quantity
    strategies compare; without a learned call component it assumes the
    per-call part is as large as the per-rep part.
    """

    floor: float = 0.003
    k: float = 2.0
    min_pairs: int = 3
    _spreads: list[float] | None = None
    _pairs: list[float] | None = None

    def update(self, samples: Sequence[float]) -> None:
        """One call's samples (its within-call spread)."""
        if len(samples) >= 3:
            if self._spreads is None:
                self._spreads = []
            self._spreads.append(rel_spread(samples))
            if len(self._spreads) > 512:
                self._spreads = self._spreads[-512:]

    def update_call(self, a: float, b: float) -> None:
        """The medians of two calls of one config (same fidelity pool)."""
        if not (a > 0 and b > 0 and math.isfinite(a) and math.isfinite(b)):
            return
        if self._pairs is None:
            self._pairs = []
        self._pairs.append(abs(math.log(a) - math.log(b)))
        if len(self._pairs) > 512:
            self._pairs = self._pairs[-512:]

    @property
    def rel_sigma(self) -> float:
        if not self._spreads:
            return self.floor
        return max(self.floor, median(self._spreads))

    @property
    def call_sigma(self) -> float | None:
        """Learned sd of one call's median (log scale), or ``None``. From
        ``|log a - log b|`` of call pairs: median ``|d|`` is
        ``0.6745 * sqrt(2) * sd``."""
        if not self._pairs or len(self._pairs) < self.min_pairs:
            return None
        return max(self.floor, median(self._pairs) / (0.6745 * math.sqrt(2.0)))

    def median_sigma(self, reps: int) -> float:
        """Noise (sd, log scale) of the median of one call of ``reps`` reps."""
        learned = self.call_sigma
        if learned is not None:
            return learned
        r = self.rel_sigma
        n = max(1, int(reps))
        return r * math.sqrt(1.0 + math.pi / (2.0 * n))

    def indistinct(self, a: float, b: float, reps: int = 1) -> bool:
        """Whether two call medians are within ``k`` noise sds."""
        if not (math.isfinite(a) and math.isfinite(b)) or a <= 0 or b <= 0:
            return False
        sd = math.sqrt(2.0) * self.median_sigma(reps)
        return abs(math.log(a) - math.log(b)) <= self.k * sd

    def to_json(self) -> dict:
        return {
            "floor": self.floor,
            "k": self.k,
            "min_pairs": self.min_pairs,
            "spreads": list(self._spreads or []),
            "pairs": list(self._pairs or []),
        }

    @classmethod
    def from_json(cls, d: Mapping) -> NoiseModel:
        m = cls(
            floor=float(d["floor"]),
            k=float(d["k"]),
            min_pairs=int(d.get("min_pairs", 3)),
        )
        m._spreads = [float(x) for x in d.get("spreads", [])] or None
        m._pairs = [float(x) for x in d.get("pairs", [])] or None
        return m


Call = tuple[int, float]
"""One measurement call of a config: ``(batch, value)``. ``value`` is the
call's median; ``batch`` the interleaved batch it was measured in (``-1``:
measured alone, pairs with nothing)."""


def _mean_sd(xs: Sequence[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def paired_t_greater(diffs: Sequence[float], shift: float = 0.0) -> float:
    """One-sided p-value for "the mean of ``diffs`` exceeds ``shift``"
    (one-sample t test; 1.0 with fewer than 2 differences)."""
    d = [x for x in diffs if math.isfinite(x)]
    if len(d) < 2:
        return 1.0
    m, sd = _mean_sd(d)
    if sd <= 0.0:
        return 0.0 if m > shift else 1.0
    t = (m - shift) / (sd / math.sqrt(len(d)))
    return student_t_sf(t, len(d) - 1)


def normal_sf(z: float) -> float:
    """P(Z > z) for a standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


@dataclass(frozen=True)
class Race:
    """Statistical elimination of configs slower than some other config.

    ``eliminate(calls, sigma=, looks=)`` takes ``{key: [(batch, value),
    ...]}`` -- one entry per measurement call (see :data:`Call`) -- and
    returns the keys that are, with Holm-corrected significance ``alpha /
    looks`` over every ordered pair, more than ``margin`` (relative) slower
    than another key. A pair is compared by:

    - a paired t test on the log ratios of their calls in common batches,
      when there are at least ``min_samples`` of those;
    - else Welch's t on the logs, when both have ``min_samples`` calls;
    - else, when ``sigma`` (a learned per-call sd, see
      :attr:`NoiseModel.call_sigma`) is given, a z test on the mean logs;
    - else not at all.

    The key with the lowest mean log value is never dropped.
    """

    alpha: float = 0.05
    margin: float = 0.01
    min_samples: int = 3

    def __post_init__(self):
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"Race alpha={self.alpha}, expected 0 < alpha < 1")
        if self.margin < 0.0:
            raise ValueError(f"Race margin={self.margin}, expected >= 0")
        if self.min_samples < 2:
            raise ValueError(f"Race min_samples={self.min_samples}, expected >= 2")

    @staticmethod
    def _logs(calls: Sequence[Call]) -> list[tuple[int, float]]:
        return [(int(b), math.log(v)) for b, v in calls if v > 0 and math.isfinite(v)]

    def leader(self, calls: Mapping[str, Sequence[Call]]) -> str | None:
        """The key with the lowest mean log value."""
        logs = {k: self._logs(v) for k, v in calls.items()}
        logs = {k: v for k, v in logs.items() if v}
        if not logs:
            return None
        return min(logs, key=lambda k: (sum(x for _b, x in logs[k]) / len(logs[k]), k))

    def pvalue(
        self, slow: Sequence[Call], fast: Sequence[Call], sigma: float | None = None
    ) -> float | None:
        """One-sided p-value of "``slow`` is more than ``margin`` slower
        than ``fast``", or ``None`` when the calls cannot tell."""
        shift = math.log1p(self.margin)
        a, b = self._logs(slow), self._logs(fast)
        if not a or not b:
            return None
        by_batch: dict[int, list[float]] = {}
        for bt, x in b:
            if bt >= 0:
                by_batch.setdefault(bt, []).append(x)
        diffs = []
        for bt, x in a:
            if bt >= 0 and by_batch.get(bt):
                diffs.append(x - by_batch[bt].pop(0))
        if len(diffs) >= self.min_samples:
            return paired_t_greater(diffs, shift)
        xa, xb = [x for _b, x in a], [x for _b, x in b]
        if len(xa) >= self.min_samples and len(xb) >= self.min_samples:
            return welch_greater(xa, xb, shift)
        if sigma is not None and sigma > 0:
            ma, mb = sum(xa) / len(xa), sum(xb) / len(xb)
            se = sigma * math.sqrt(1.0 / len(xa) + 1.0 / len(xb))
            return normal_sf((ma - mb - shift) / se)
        return None

    def pvalues(
        self, calls: Mapping[str, Sequence[Call]], sigma: float | None = None
    ) -> dict[tuple[str, str], float]:
        """Raw p-values ``{(slow, fast): p}`` of every comparable pair."""
        keys = sorted(k for k, v in calls.items() if self._logs(v))
        out: dict[tuple[str, str], float] = {}
        for k in keys:
            for j in keys:
                if k == j:
                    continue
                p = self.pvalue(calls[k], calls[j], sigma)
                if p is not None:
                    out[(k, j)] = p
        return out

    def eliminate(
        self,
        calls: Mapping[str, Sequence[Call]],
        *,
        sigma: float | None = None,
        looks: int = 1,
    ) -> list[str]:
        """Keys significantly slower than another key (see the class)."""
        raw = self.pvalues(calls, sigma)
        if not raw:
            return []
        pairs = sorted(raw)
        adj = holm([raw[p] for p in pairs])
        level = self.alpha / max(1, int(looks))
        lead = self.leader(calls)
        return sorted({k for (k, _j), p in zip(pairs, adj) if p < level and k != lead})
