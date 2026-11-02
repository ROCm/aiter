# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Model-guided search: priors from a selector, TPE, and a GP with EI.

A :class:`Prior` predicts a config's time before it is measured -- the
LightGBM+Origami selector's score, an analytical roofline, a table of
recorded times from another problem. Lower is better; ``nan`` means "no
opinion". Priors are used three ways:

- to seed: the first trials are the prior's top picks over a candidate pool
  (mixed with a stratified sample, so a wrong prior cannot capture the
  search);
- inside :class:`TPE`, as an extra term in the acquisition score whose
  weight decays as measurements accumulate;
- inside :class:`GPEI`, as the GP's mean function (``a + b * log prior``,
  fitted to the measurements), so the GP models only the prior's residual.

Both model strategies work on the log of the measured time and treat failed
configs as "bad" observations, so they learn to avoid failing regions.

A prior's predictions are put on one scale for the whole search
(:class:`PriorScale`): log when they are times (all positive), linear
otherwise. The choice is made once -- from the prior's ``scale`` attribute
when it declares one, else from the first full candidate pool -- and kept
(it is part of the strategy state), so two sets of predictions of one prior
are never on different scales. Under a log scale a non-positive prediction
is "no opinion".
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from .params import BOOL, CATEGORICAL, ORDINAL, Genome, SearchSpace
from .strategy import Strategy
from .trial import Observation, Trial

__all__ = [
    "GPEI",
    "TPE",
    "CallablePrior",
    "LightGBMPrior",
    "Prior",
    "PriorScale",
    "TablePrior",
    "prior_ranked",
]


@runtime_checkable
class Prior(Protocol):
    """Predicts configs' times (lower is better; ``nan``: no opinion). An
    optional ``scale`` attribute (``"log"`` or ``"linear"``) fixes how the
    predictions are compared; see :class:`PriorScale`."""

    def predict(self, configs: Sequence[Mapping[str, object]]) -> Sequence[float]: ...


class CallablePrior:
    """``fn(config) -> predicted time`` as a :class:`Prior`."""

    def __init__(
        self,
        fn: Callable[[Mapping[str, object]], float],
        *,
        scale: str | None = None,
    ):
        self.fn = fn
        self.scale = scale

    def predict(self, configs: Sequence[Mapping[str, object]]) -> list[float]:
        out = []
        for c in configs:
            try:
                out.append(float(self.fn(c)))
            except (ValueError, TypeError, KeyError, ZeroDivisionError):
                out.append(math.nan)
        return out


class TablePrior:
    """Predictions looked up by a config key function (``nan`` if absent).

    ``table`` maps ``key_fn(config)`` to a predicted time (or a score when
    ``higher_is_better``). Typical source: a config selector's
    per-candidate predictions exported to CSV.
    """

    def __init__(
        self,
        table: Mapping[str, float],
        key_fn: Callable[[Mapping[str, object]], str],
        *,
        higher_is_better: bool = False,
        scale: str | None = None,
    ):
        self.table = dict(table)
        self.key_fn = key_fn
        self.sign = -1.0 if higher_is_better else 1.0
        self.scale = scale

    def predict(self, configs: Sequence[Mapping[str, object]]) -> list[float]:
        out = []
        for c in configs:
            v = self.table.get(self.key_fn(c))
            out.append(math.nan if v is None else self.sign * float(v))
        return out


class LightGBMPrior:
    """A LightGBM text model as a prior (``lightgbm`` imported lazily).

    ``featurize(config) -> feature vector`` must reproduce the features the
    model was trained on (for example a selector's feature-engine columns
    plus an analytic latency estimate). A selector trained to score
    "probability of being best" is ``higher_is_better``.
    """

    def __init__(
        self,
        model_path: str,
        featurize: Callable[[Mapping[str, object]], Sequence[float]],
        *,
        higher_is_better: bool = True,
    ):
        import lightgbm  # optional dependency, imported on use

        self.booster = lightgbm.Booster(model_file=str(model_path))
        self.featurize = featurize
        self.sign = -1.0 if higher_is_better else 1.0
        self.scale = "linear"

    def predict(self, configs: Sequence[Mapping[str, object]]) -> list[float]:
        if not configs:
            return []
        X = np.asarray([list(self.featurize(c)) for c in configs], dtype=float)
        return [self.sign * float(v) for v in self.booster.predict(X)]


_SCALES = ("log", "linear")


class PriorScale:
    """A prior with its comparison scale fixed once.

    ``mode`` is ``"log"`` (predictions are times: compared as log, so a
    ratio is a distance), ``"linear"`` (scores of any sign), or ``None``:
    decided by the first call with any finite prediction (log when every
    finite prediction is positive) and kept from then on.
    """

    def __init__(self, prior: Prior, mode: str | None = None):
        mode = mode if mode is not None else getattr(prior, "scale", None)
        if mode is not None and mode not in _SCALES:
            raise ValueError(f"PriorScale mode={mode!r}, expected one of {_SCALES}")
        self.prior = prior
        self.mode = mode

    def values(self, space: SearchSpace, genomes: Sequence[Genome]) -> list[float]:
        """Predictions on the fixed "lower is better" scale, ``nan`` for
        none."""
        raw = list(self.prior.predict([space.decode(g) for g in genomes]))
        if len(raw) != len(genomes):
            raise ValueError(
                f"prior predictions - len={len(raw)}, expected {len(genomes)} "
                "(one per config)"
            )
        vals = [float(v) if v is not None else math.nan for v in raw]
        if self.mode is None:
            finite = [v for v in vals if math.isfinite(v)]
            if not finite:
                return vals
            self.mode = "log" if min(finite) > 0 else "linear"
        if self.mode == "log":
            return [
                math.log(v) if math.isfinite(v) and v > 0 else math.nan for v in vals
            ]
        return vals


def _prior_values(prior, space: SearchSpace, genomes: Sequence[Genome]) -> list[float]:
    scale = prior if isinstance(prior, PriorScale) else PriorScale(prior)
    return scale.values(space, genomes)


# Validity checks per repaired point of a large candidate pool. A full repair
# (single moves, then pairs) costs up to 256 checks; on a predicate space where
# few genomes are valid (the universal-GEMM space) the GP's 256-point pool per
# proposal then dominates a search's wall time. Past the cap, repair
# falls back to a random valid genome, which is as good a pool point.
POOL_REPAIR_CHECKS = 24


def candidate_pool(space: SearchSpace, rng, size: int) -> list[Genome]:
    """Distinct valid genomes: the whole space when it is small enough,
    else a stratified sample (cheap repairs, see POOL_REPAIR_CHECKS)."""
    listed = space.listed()
    if listed is not None and len(listed) <= size:
        return list(listed)
    if space.size() <= size:
        out, seen = [], set()
        for g in space.enumerate():
            k = space.key(g)
            if k is not None and k not in seen:
                seen.add(k)
                out.append(g)
        return out
    return space.stratified(rng, size, repair_checks=POOL_REPAIR_CHECKS)


def prior_ranked(
    prior: Prior | PriorScale,
    space: SearchSpace,
    rng,
    *,
    pool: int = 2048,
    k: int | None = None,
    polish: int = 4,
    polish_steps: int = 24,
) -> list[Genome]:
    """Valid genomes in prior order (best first), unknowns last.

    The pool is the whole space when it is small, else a stratified sample;
    the ``polish`` best pool points are then improved by greedy descent *on
    the prior* (single moves, ``polish_steps`` at most), so the prior's
    own local optima lead the list even when the sample missed them.
    Predictions are cheap next to measurements, so this costs no budget.
    Pool and polish predictions share one :class:`PriorScale` (pass the
    strategy's own to keep its scale for the rest of the search).
    """
    scale = prior if isinstance(prior, PriorScale) else PriorScale(prior)
    genomes = candidate_pool(space, rng, pool)
    vals = scale.values(space, genomes)
    score = {space.key(g): v for g, v in zip(genomes, vals)}

    def rank_key(i: int):
        v = vals[i]
        return (0 if math.isfinite(v) else 1, v if math.isfinite(v) else 0.0, i)

    order = sorted(range(len(genomes)), key=rank_key)
    ranked = [genomes[i] for i in order]
    polished: list[Genome] = []
    for g in ranked[: max(0, int(polish))]:
        cur, cur_v = g, score[space.key(g)]
        if not math.isfinite(cur_v):
            continue
        for _ in range(int(polish_steps)):
            moves = [m for m in space.moves(cur) if space.valid(m)]
            fresh = [m for m in moves if space.key(m) not in score]
            if fresh:
                for m, v in zip(fresh, scale.values(space, fresh)):
                    score[space.key(m)] = v
            nxt = min(
                moves,
                key=lambda m: (
                    (
                        score[space.key(m)]
                        if math.isfinite(score[space.key(m)])
                        else math.inf
                    ),
                    space.key(m),
                ),
                default=None,
            )
            if nxt is None or not score[space.key(nxt)] < cur_v:
                break
            cur, cur_v = nxt, score[space.key(nxt)]
        polished.append(cur)
    head = sorted(
        {space.key(g): g for g in polished}.values(),
        key=lambda g: (score[space.key(g)], space.key(g)),
    )
    seen = {space.key(g) for g in head}
    out = head + [g for g in ranked if space.key(g) not in seen]
    return out if k is None else out[:k]


def _tri_solve(T: np.ndarray, B: np.ndarray, *, lower: bool) -> np.ndarray:
    """Solve ``T x = B`` for triangular ``T`` by substitution (NumPy has no
    triangular solver, and a general LU solve is needlessly slow here)."""
    n = T.shape[0]
    x = np.array(B, dtype=float, copy=True)
    rows = range(n) if lower else range(n - 1, -1, -1)
    for i in rows:
        if lower:
            if i:
                x[i] -= T[i, :i] @ x[:i]
        elif i < n - 1:
            x[i] -= T[i, i + 1 :] @ x[i + 1 :]
        x[i] /= T[i, i]
    return x


class _ModelStrategy(Strategy):
    """Shared startup (prior top + stratified) and the failed-config log."""

    def __init__(
        self,
        space: SearchSpace,
        seed: int,
        *,
        n_startup: int,
        prior: Prior | None,
        prior_pool: int,
        init: Sequence[Sequence[int]],
    ):
        super().__init__(space, seed)
        if int(n_startup) < 1:
            raise ValueError(f"{self.name} n_startup={n_startup}, expected >= 1")
        self.n_startup = int(n_startup)
        self.prior = prior
        self.prior_scale = None if prior is None else PriorScale(prior)
        self.prior_pool = int(prior_pool)
        self._startup: list[list[int]] | None = None
        self._seeds = [space.check(g) for g in init]
        self.failed: dict[str, list[int]] = {}
        self._exhausted = False
        self._prior_top: list[list[int]] = []

    def _build_startup(self) -> None:
        picks: list[Genome] = []
        keys: set[str] = set(self.history)

        def add(g) -> None:
            k = self.space.key(g)
            if k is not None and k not in keys:
                keys.add(k)
                picks.append(g)

        for g in self._seeds:
            r = self.space.repair(g, self.rng)
            if r is not None:
                add(r)
        if self.prior is not None:
            ranked = prior_ranked(
                self.prior_scale, self.space, self.rng, pool=self.prior_pool
            )
            self._prior_top = [list(g) for g in ranked[:32]]
            for g in ranked:
                if len(picks) >= (self.n_startup + 1) // 2 + len(self._seeds):
                    break
                add(g)
        for g in self.space.stratified(self.rng, self.n_startup + 2):
            if len(picks) >= self.n_startup:
                break
            add(g)
        self._startup = [list(g) for g in picks]

    def observe(self, ob: Observation, own: bool) -> None:
        if ob.full and ob.key is not None and not ob.ok:
            self.failed.setdefault(
                ob.key, list(self.space.canonical_genome(ob.trial.genome))
            )

    def ask(self, n: int = 1) -> list[Trial]:
        if self._startup is None:
            self._build_startup()
        out: list[Trial] = []
        taken = self.pending_keys()
        while self._startup and len(out) < int(n):
            g = self._startup.pop(0)
            k = self.space.key(g)
            if k in self.history or k in taken:
                continue
            taken.add(k)
            out.append(self.trial(g))
        ok = [k for k, v in self.history.items() if math.isfinite(v)]
        while len(out) < int(n):
            if len(ok) < 2:
                g = self._random_unseen(taken)
            else:
                g = self._propose(taken, len(out))
            if g is None:
                self._exhausted = True
                break
            taken.add(self.space.key(g))
            out.append(self.trial(g))
        return out

    def _random_unseen(self, taken: set[str]) -> Genome | None:
        for _ in range(256):
            g = self.space.random_valid(self.rng, tries=8)
            if g is None:
                continue
            k = self.space.key(g)
            if k not in self.history and k not in taken:
                return g
        return None

    def _propose(self, taken: set[str], batch_pos: int) -> Genome | None:
        raise NotImplementedError

    @property
    def done(self) -> bool:
        return self._exhausted

    def _state(self) -> dict:
        return {
            "startup": self._startup,
            "failed": self.failed,
            "exhausted": self._exhausted,
            "prior_top": self._prior_top,
            "prior_scale": None if self.prior_scale is None else self.prior_scale.mode,
            **self._extra_state(),
        }

    def _load(self, d: Mapping) -> None:
        self._startup = (
            None if d["startup"] is None else [list(g) for g in d["startup"]]
        )
        self.failed = {k: list(g) for k, g in d["failed"].items()}
        self._exhausted = bool(d["exhausted"])
        self._prior_top = [list(g) for g in d.get("prior_top", [])]
        if self.prior_scale is not None:
            self.prior_scale.mode = d.get("prior_scale", self.prior_scale.mode)
        self._load_extra(d)

    def _extra_state(self) -> dict:
        return {}

    def _load_extra(self, d: Mapping) -> None:
        pass


class TPE(_ModelStrategy):
    """Tree-structured Parzen estimator over mixed, gated parameters.

    Observations are split at the ``gamma`` quantile of log time into good
    and bad; each parameter gets a smoothed density per side (frequencies for
    categorical/bool, a discrete Gaussian kernel on level indices for
    ordinal), counted only where the parameter is active. Candidates drawn
    from the good density (plus single moves of the best configs) are scored
    by ``sum log l(x)/g(x)`` plus the decaying prior term; the best unseen
    one is asked. Failed configs count as bad. Batched asks treat pending
    configs as bad ("constant liar") so a batch spreads out.
    """

    name = "tpe"

    def __init__(
        self,
        space: SearchSpace,
        seed: int = 0,
        *,
        n_startup: int = 10,
        gamma: float = 0.25,
        n_candidates: int = 48,
        prior: Prior | None = None,
        prior_weight: float = 1.0,
        prior_pool: int = 2048,
        init: Sequence[Sequence[int]] = (),
    ):
        super().__init__(
            space,
            seed,
            n_startup=n_startup,
            prior=prior,
            prior_pool=prior_pool,
            init=init,
        )
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"TPE gamma={gamma}, expected 0 < gamma < 1")
        self.gamma = float(gamma)
        self.n_candidates = int(n_candidates)
        self.prior_weight = float(prior_weight)

    def _density(self, genomes: Sequence[Sequence[int]]) -> list[list[float]]:
        dens: list[list[float]] = []
        for i, p in enumerate(self.space.params):
            vals = [g[i] for g in genomes if self.space.active(g)[i]]
            n = p.size
            if p.kind == ORDINAL:
                bw = max(0.5, (n - 1) * 0.5 * max(1, len(vals)) ** (-0.2))
                w = [1.0 / n] * n  # uniform pseudo-observation
                for v in vals:
                    ker = [math.exp(-0.5 * ((j - v) / bw) ** 2) for j in range(n)]
                    z = sum(ker)
                    w = [a + b / z for a, b in zip(w, ker)]
            else:
                w = [1.0 / n] * n
                for v in vals:
                    w[v] += 1.0
            z = sum(w)
            dens.append([x / z for x in w])
        return dens

    def _score(self, g: Sequence[int], lden, gden) -> float:
        act = self.space.active(g)
        s = 0.0
        for i, a in enumerate(act):
            if a:
                s += math.log(lden[i][g[i]]) - math.log(gden[i][g[i]])
        return s

    def _sample(self, lden) -> Genome:
        g = []
        for dist in lden:
            r, acc = self.rng.random(), 0.0
            pick = len(dist) - 1
            for j, p in enumerate(dist):
                acc += p
                if r < acc:
                    pick = j
                    break
            g.append(pick)
        return tuple(g)

    def _propose(self, taken: set[str], batch_pos: int) -> Genome | None:
        ranked = sorted(
            ((v, k) for k, v in self.history.items() if math.isfinite(v)),
        )
        n_good = max(1, math.ceil(self.gamma * len(ranked)))
        good = [self.genomes[k] for _v, k in ranked[:n_good]]
        bad = [self.genomes[k] for _v, k in ranked[n_good:]]
        bad += list(self.failed.values())
        bad += [self.genomes.get(k) or self._genome_of(k) for k in taken]
        bad = [g for g in bad if g is not None]
        lden, gden = self._density(good), self._density(bad)
        cands: list[Genome] = []
        for _ in range(self.n_candidates):
            g = self.space.repair(self._sample(lden), self.rng, archive=good)
            if g is not None:
                cands.append(g)
        for _v, k in ranked[:2]:
            cands.extend(self.space.moves(self.genomes[k]))
        cands.extend(tuple(g) for g in self._prior_top[:8])
        seen: set[str] = set()
        uniq: list[Genome] = []
        for g in cands:
            k = self.space.key(g)
            if k is None or k in seen or k in self.history or k in taken:
                continue
            seen.add(k)
            uniq.append(g)
        if not uniq:
            return self._random_unseen(taken)
        scores = [self._score(g, lden, gden) for g in uniq]
        if self.prior is not None and self.prior_weight > 0:
            pv = self.prior_scale.values(self.space, uniq)
            fin = [v for v in pv if math.isfinite(v)]
            if len(fin) >= 2:
                mu = sum(fin) / len(fin)
                sd = math.sqrt(sum((v - mu) ** 2 for v in fin) / len(fin)) or 1.0
                w = self.prior_weight * self.n_startup / (self.n_startup + len(ranked))
                scores = [
                    s - w * ((v - mu) / sd if math.isfinite(v) else 0.0)
                    for s, v in zip(scores, pv)
                ]
        best = max(range(len(uniq)), key=lambda i: (scores[i], -i))
        return uniq[best]

    def _genome_of(self, key: str):
        for g in self.pending.values():
            if self.space.key(g) == key:
                return g
        return None


class GPEI(_ModelStrategy):
    """Gaussian process on log time with expected improvement.

    Features: ordinal position in [0, 1], one-hot categorical/bool, inactive
    gated parameters at 0 with their gate's feature already telling them
    apart. Kernel: Matern 5/2, lengthscale picked by marginal likelihood
    from a small grid. Mean: the prior's fitted ``a + b * log prior`` when
    a prior is given, else the observed mean. Failed configs enter as
    observations at the worst time plus a margin. Candidates: a stratified
    pool, single moves of the best configs, and the prior's top picks.
    """

    name = "gp"

    def __init__(
        self,
        space: SearchSpace,
        seed: int = 0,
        *,
        n_startup: int = 8,
        n_pool: int = 256,
        xi: float = 0.01,
        max_points: int = 160,
        refit_every: int = 8,
        prior: Prior | None = None,
        prior_pool: int = 2048,
        init: Sequence[Sequence[int]] = (),
    ):
        super().__init__(
            space,
            seed,
            n_startup=n_startup,
            prior=prior,
            prior_pool=prior_pool,
            init=init,
        )
        self.n_pool = int(n_pool)
        self.xi = float(xi)
        self.max_points = int(max_points)
        self.refit_every = int(refit_every)
        self._ell: float | None = None
        self._fits = 0

    def _features(self, g: Sequence[int]) -> list[float]:
        act = self.space.active(g)
        f: list[float] = []
        for i, p in enumerate(self.space.params):
            if p.kind == ORDINAL:
                f.append(g[i] / (p.size - 1) if act[i] else 0.0)
            elif p.kind in (CATEGORICAL, BOOL):
                one = [0.0] * p.size
                if act[i]:
                    one[g[i]] = 1.0 / math.sqrt(2.0)
                f.extend(one)
        return f

    @staticmethod
    def _dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
        return np.sqrt(np.maximum(sq, 0.0))

    @staticmethod
    def _matern_d(D: np.ndarray, ell: float) -> np.ndarray:
        d = D / ell
        s5 = math.sqrt(5.0)
        return (1.0 + s5 * d + 5.0 / 3.0 * d * d) * np.exp(-s5 * d)

    def _propose(self, taken: set[str], batch_pos: int) -> Genome | None:
        ok = sorted((v, k) for k, v in self.history.items() if math.isfinite(v))
        if len(ok) > self.max_points:
            # The best half plus the most recent rest: the model must be
            # accurate where the optimum is and know where it has been.
            half = self.max_points // 2
            recent = [k for k in self.history if math.isfinite(self.history[k])]
            keep = {k for _v, k in ok[:half]} | set(recent[-(self.max_points - half) :])
            ok = [(v, k) for v, k in ok if k in keep]
        failed = sorted(self.failed)[-max(1, self.max_points // 4) :]
        keys = [k for _v, k in ok] + failed
        genomes = [self.genomes.get(k) or self.failed[k] for k in keys]
        y = [math.log(v) for v, _k in ok]
        worst = max(y)
        y += [worst + math.log(1.5)] * len(failed)
        # Constant liar for the rest of the batch: pending configs at the best.
        for k in sorted(taken):
            g = next((g for g in self.pending.values() if self.space.key(g) == k), None)
            if g is not None:
                genomes.append(list(g))
                y.append(min(y))
        X = np.asarray([self._features(g) for g in genomes], dtype=float)
        Y = np.asarray(y, dtype=float)
        mean_fn = self._mean_fn(genomes, Y)
        M = np.asarray(mean_fn(genomes), dtype=float)
        R = Y - M
        var = float(np.var(R)) or 1e-4
        D = self._dist(X, X)
        grid = (0.25, 0.5, 1.0, 2.0)
        if self._ell is not None and self._fits % max(1, self.refit_every):
            grid = (self._ell,)
        self._fits += 1
        best_ell, best_ll, best_fac = None, -math.inf, None
        for ell in grid:
            K = var * self._matern_d(D, ell) + (1e-6 + 1e-2 * var) * np.eye(len(X))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                continue
            alpha = _tri_solve(L.T, _tri_solve(L, R, lower=True), lower=False)
            ll = -0.5 * float(R @ alpha) - float(np.log(np.diag(L)).sum())
            if ll > best_ll:
                best_ell, best_ll, best_fac = ell, ll, (L, alpha)
        if best_ell is None:
            return self._random_unseen(taken)
        self._ell = best_ell
        L, alpha = best_fac
        pool: list[Genome] = list(
            self.space.stratified(
                self.rng, self.n_pool, repair_checks=POOL_REPAIR_CHECKS
            )
        )
        for _v, k in ok[:3]:
            pool.extend(self.space.moves(self.genomes[k]))
        pool.extend(tuple(g) for g in self._prior_top[:16])
        seen: set[str] = set()
        cands: list[Genome] = []
        for g in pool:
            k = self.space.key(g)
            if k is None or k in seen or k in self.history or k in taken:
                continue
            seen.add(k)
            cands.append(g)
        if not cands:
            return self._random_unseen(taken)
        Xc = np.asarray([self._features(g) for g in cands], dtype=float)
        Ks = var * self._matern_d(self._dist(Xc, X), best_ell)
        mu = np.asarray(mean_fn(cands), dtype=float) + Ks @ alpha
        v = _tri_solve(L, Ks.T, lower=True)
        sd = np.sqrt(np.maximum(var - (v * v).sum(0), 1e-12))
        best = float(min(y[: len(ok)]))
        imp = best - mu - self.xi
        z = imp / sd
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
        pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        ei = imp * cdf + sd * pdf
        i = int(np.argmax(ei))
        return cands[i]

    def _extra_state(self) -> dict:
        return {"ell": self._ell, "fits": self._fits}

    def _load_extra(self, d: Mapping) -> None:
        self._ell = d.get("ell")
        self._fits = int(d.get("fits", 0))

    def _mean_fn(self, genomes, Y: np.ndarray):
        const = float(Y.mean())
        if self.prior is None:
            return lambda gs: [const] * len(gs)
        scale = self.prior_scale
        pv = np.asarray(scale.values(self.space, genomes), dtype=float)
        ok = np.isfinite(pv)
        if ok.sum() < 3 or float(np.var(pv[ok])) <= 0:
            return lambda gs: [const] * len(gs)
        b = float(np.cov(pv[ok], Y[ok], bias=True)[0, 1] / np.var(pv[ok]))
        b = min(2.0, max(0.0, b))
        a = float(Y[ok].mean() - b * pv[ok].mean())

        def mean(gs):
            vals = scale.values(self.space, gs)
            return [a + b * v if math.isfinite(v) else const for v in vals]

        return mean
