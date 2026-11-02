# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Offline replay: score strategies against a recorded sweep table.

A recorded sweep (one row per measured config and problem) is an oracle:
for one problem, the configs it lists form a :meth:`SearchSpace.from_configs`
space and the recorded times a :class:`~.measure.TableBackend`. A strategy
then runs as it would live, but each "measurement" reads the table, and
only for configs the strategy asked for: no *value* leaks from unevaluated
configs. The anytime curve (best found after ``n`` evaluations, as a
fraction of the table's best) is what tells strategies apart.

What does leak is *membership*. The space is exactly the recorded rows, so
the member sampler draws only recorded configs and repair snaps to the
nearest recorded one; rows the recording dropped (configs the sweep never
measured, ``unsupported_*`` rows) are invisible and free, where a live
space would pay for its invalid configs (most genomes of the universal-GEMM
space are invalid). Tables that were themselves pruned by an earlier search (transfer
pruning, a stage-2 shortlist of stage-1 winners) carry the oracle's
information in their geometry: a random sample of such a shortlist is
already near the optimum. Compare strategies only on full grids, and read
absolute fractions from replay as optimistic.

Values may be times (lower is better) or rates such as TF/s
(``higher_is_better``; converted to 1/rate so every strategy minimizes).

Recorded failures (compile errors, numeric failures) stay in the space and
cost an evaluation when asked for, exactly as live; configs whose recorded
time is only a lower bound (a screen that stopped early because the config
was clearly slow) are measurable but never the oracle, and their true value
is unknown: :meth:`ReplayProblem.truth` returns ``nan`` for them, the
anytime curves do not credit them, and a run that returns one has
``final=None`` (counted as ``unknown`` by :func:`replay_summary`). With noise
(``noise`` per rep, ``call_noise`` per measurement) a run reports two
anytime curves: ``curve`` (the best *true* value among the configs measured
so far: what a perfect final re-measure could reach) and
``incumbent_curve`` (the true value of the config the strategy itself
ranks best so far); ``final`` is the true value of the config the search
returns, after its confirmation round.
"""

from __future__ import annotations

import csv
import gzip
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .api import make_strategy
from .driver import run_search
from .measure import Evaluator, TableBackend
from .params import SearchSpace
from .portfolio import WarmStart
from .surrogate import Prior
from .trial import Budget, Fidelity, Status

__all__ = [
    "STATUS_MAP",
    "ReplayProblem",
    "load_table",
    "replay_problem",
    "replay_summary",
]


def _open(path: str | Path):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", newline="")
    return p.open(newline="")


def _parse(v: str):
    s = v.strip()
    if s in ("True", "true"):
        return True
    if s in ("False", "false"):
        return False
    if s == "":
        return None
    try:
        i = int(s)
        return i
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


_FAILURE_STATUSES = {
    Status.COMPILE_ERROR,
    Status.NUMERIC_FAIL,
    Status.SPILL,
    Status.TIMEOUT,
    Status.RUN_ERROR,
}

# Status words of recorded sweep tables (rocKE's sweep format) -> engine
# statuses; ``early_stop`` (a lower bound) and ``slow_unchecked`` (measured,
# slower than the checked leaders) stay as written.
STATUS_MAP = {
    "compile_fail": Status.COMPILE_ERROR,
    "compile_or_run_error": Status.RUN_ERROR,
    "crashed_previous_run": Status.RUN_ERROR,
    "numeric_fail": Status.NUMERIC_FAIL,
    "aborted_early": "early_stop",
}


class ReplayProblem:
    """One problem of a recorded table: its space, backend and oracle.

    ``statuses`` (parallel to ``values``, optional) marks rows: a failure
    status (``compile_error``, ``numeric_fail``, ...) keeps the config in
    the space as a recorded failure; a status in ``bound_statuses`` (default
    ``early_stop``) marks a lower-bound time that is measurable but never
    the oracle; anything else is a plain measurement. Rows without a finite
    positive value and without a failure status are dropped.
    """

    def __init__(
        self,
        name: str,
        configs: Sequence[Mapping[str, object]],
        values: Sequence[float | None],
        *,
        noise: float = 0.0,
        seed: int = 0,
        statuses: Sequence[str] | None = None,
        bound_statuses: Iterable[str] = ("early_stop",),
        call_noise: float = 0.0,
        member_index: bool = True,
    ):
        if len(configs) != len(values):
            raise ValueError("ReplayProblem configs and values differ in length")
        if statuses is not None and len(statuses) != len(values):
            raise ValueError("ReplayProblem statuses and values differ in length")
        self.name = name
        bounds_st = set(bound_statuses)
        keep_c, kinds = [], []
        for i, (c, v) in enumerate(zip(configs, values)):
            st = statuses[i] if statuses is not None else "ok"
            if st in _FAILURE_STATUSES:
                keep_c.append(c)
                kinds.append((st, None))
                continue
            if v is None or not math.isfinite(v) or v <= 0:
                continue
            keep_c.append(c)
            kinds.append(("bound" if st in bounds_st else "ok", float(v)))
        if not keep_c:
            raise ValueError(f"ReplayProblem {name!r}: no usable rows")
        self.space = SearchSpace.from_configs(
            keep_c,
            name=f"replay:{name}",
            member_index=member_index,
            allow_duplicates=True,
        )
        table: dict[str, list[float]] = {}
        bound: dict[str, bool] = {}
        failures: dict[str, str] = {}
        for c, (kind, v) in zip(keep_c, kinds):
            key = self.space.key(self.space.encode(c))
            if key is None:
                continue
            if v is None:
                failures.setdefault(key, kind)
                continue
            table.setdefault(key, []).append(v)
            bound[key] = bound.get(key, True) and kind == "bound"
        # A config measured at least once is not a failure.
        failures = {k: st for k, st in failures.items() if k not in table}
        self.table = {k: sorted(v)[len(v) // 2] for k, v in table.items()}
        self.bounds = {k for k, b in bound.items() if b}
        self.failures = failures
        self.noise, self.call_noise = float(noise), float(call_noise)
        self.backend = TableBackend(
            self.table,
            noise=noise,
            seed=seed,
            failures=failures,
            call_noise=call_noise,
        )
        exact = [k for k in self.table if k not in self.bounds]
        if not exact:
            raise ValueError(f"ReplayProblem {name!r}: no exact measurement")
        self.oracle_key = min(exact, key=lambda k: (self.table[k], k))
        self.oracle = self.table[self.oracle_key]
        # Lower bounds below the oracle make the oracle uncertain; recorded
        # sweeps should have none (count them before trusting a table).
        self.ambiguous_bounds = sum(
            1 for k in self.bounds if self.table[k] < self.oracle
        )

    def truth(self, key: str | None) -> float:
        """The recorded value of ``key``: ``inf`` for failures and keys
        outside the table, ``nan`` (unknown) for a lower bound -- its true
        time is somewhere above the bound, so it is never credited."""
        if key is None:
            return math.inf
        if key in self.bounds:
            return math.nan
        return self.table.get(key, math.inf)

    def with_seed(self, seed: int) -> ReplayProblem:
        """The same problem (space, table, oracle shared) with a fresh noise
        seed: one replay problem serves every seed of a run."""
        twin = object.__new__(ReplayProblem)
        twin.__dict__.update(self.__dict__)
        twin.backend = TableBackend(
            self.table,
            noise=self.noise,
            seed=seed,
            failures=self.failures,
            call_noise=self.call_noise,
        )
        return twin

    def __len__(self) -> int:
        return len(self.table) + len(self.failures)


def load_table(
    path: str | Path,
    *,
    config_cols: Sequence[str],
    value_col: str,
    problem_cols: Sequence[str],
    higher_is_better: bool = False,
    where: Mapping[str, str] | None = None,
    status_col: str | None = None,
    status_map: Mapping[str, str] | None = None,
) -> dict[str, tuple]:
    """``{problem: (configs, times)}`` from a CSV (``.gz`` ok).

    With ``status_col`` each entry is ``(configs, times, statuses)``: rows
    whose status maps (``status_map``, default :data:`STATUS_MAP`, unknown
    statuses kept as written) to a failure are kept with time ``None`` so
    :class:`ReplayProblem` charges them as failures, and ``early_stop`` rows
    are lower bounds.
    """
    smap = dict(STATUS_MAP if status_map is None else status_map)
    out: dict[str, tuple] = {}
    with _open(path) as f:
        for row in csv.DictReader(f):
            if where and any(row.get(k) != v for k, v in where.items()):
                continue
            st = None
            if status_col is not None:
                raw_st = row.get(status_col, "")
                st = smap.get(raw_st, raw_st)
            raw = row.get(value_col, "")
            try:
                val = float(raw)
            except ValueError:
                val = math.nan
            if st in _FAILURE_STATUSES:
                t = None
            elif not math.isfinite(val) or val <= 0:
                continue
            else:
                t = 1.0 / val if higher_is_better else val
            prob = "|".join(row.get(c, "") for c in problem_cols)
            cfg = {c: _parse(row.get(c, "")) for c in config_cols}
            entry = out.setdefault(
                prob, ([], []) if status_col is None else ([], [], [])
            )
            entry[0].append(cfg)
            entry[1].append(t)
            if status_col is not None:
                entry[2].append(st)
    return out


def replay_problem(
    problem: ReplayProblem,
    strategy: str,
    *,
    budget: int,
    seed: int = 0,
    prior: Prior | None = None,
    warm: Sequence[WarmStart] = (),
    batch: int = 1,
    ladder: Sequence[Fidelity] = (Fidelity(reps=1),),
    options: Mapping | None = None,
    members: Sequence[str] | None = None,
    confirm: int = 0,
    confirm_rounds: int = 2,
    stall_rounds: int = 32,
) -> dict:
    """One replay run: anytime curves as fractions of the oracle.

    Each entry of ``curve`` / ``incumbent_curve`` / ``cost`` belongs to one
    fresh evaluation (cache hits and refused configs are free). ``cost`` is
    the running sum of the measured fidelities' costs. ``final`` is the
    returned config's recorded value as a fraction of the oracle (after
    ``confirm`` finalists were re-measured, when set).
    """
    evaluator = Evaluator(problem.space, problem.backend, ladder=ladder)
    kw = {} if members is None else {"members": tuple(members)}
    strat = make_strategy(
        strategy,
        problem.space,
        seed,
        prior=prior,
        warm=warm,
        ladder=ladder,
        options=options,
        **kw,
    )
    bud = budget if isinstance(budget, Budget) else Budget(evaluations=int(budget))
    res = run_search(
        problem.space,
        strat,
        evaluator,
        bud,
        batch=batch,
        stall_rounds=stall_rounds,
        confirm=confirm,
        confirm_rounds=confirm_rounds,
    )
    curve: list[float] = []
    inc_curve: list[float] = []
    cost: list[float] = []
    best = math.inf
    inc_val, inc_key = math.inf, None
    spent = 0.0
    failures_hit = 0
    for ob in res.observations:
        if ob.cached or ob.status == Status.INVALID:
            continue
        if ob.status != Status.OK:
            failures_hit += 1
        else:
            spent += evaluator.ladder[ob.fidelity].cost
        if ob.full and ob.ok:
            t = problem.truth(ob.key)
            if not math.isnan(t):
                best = min(best, t)
            if ob.value < inc_val:
                inc_val, inc_key = ob.value, ob.key
        curve.append(problem.oracle / best if math.isfinite(best) else 0.0)
        inc_t = problem.truth(inc_key)
        inc_curve.append(problem.oracle / inc_t if math.isfinite(inc_t) else 0.0)
        cost.append(spent)
    final_truth = problem.truth(res.best_key)
    if math.isnan(final_truth):
        final = None  # the search returned a lower bound: true value unknown
    else:
        final = problem.oracle / final_truth if math.isfinite(final_truth) else 0.0
    return {
        "problem": problem.name,
        "strategy": strategy,
        "seed": seed,
        "budget": bud.to_json() if isinstance(budget, Budget) else budget,
        "space": len(problem),
        "evaluations": res.evaluations,
        "curve": curve,
        "incumbent_curve": inc_curve,
        "cost": cost,
        "final": final,
        "found": final,
        "best_seen": curve[-1] if curve else 0.0,
        "failures_hit": failures_hit,
        "confirmed": bool(res.confirmed.get("winner")),
        "stopped": res.stopped,
    }


def replay_summary(runs: Iterable[Mapping], thresholds=(0.95, 0.98, 0.99)) -> dict:
    """Per strategy: mean / p10 / worst of the *final* fraction of oracle
    (the true value of the config each search returned; runs whose return
    is a lower bound are counted in ``unknown`` and left out), the mean of
    ``best_seen`` (the oracle-informed ideal: the best true value among the
    configs measured, what a perfect final re-measure could reach), and the
    median evaluations for the ideal curve to reach each threshold
    (``None`` when not reached)."""
    by: dict[str, list[Mapping]] = {}
    for r in runs:
        by.setdefault(r["strategy"], []).append(r)
    out = {}
    for s, rs in sorted(by.items()):
        finals = sorted(r["final"] for r in rs if r.get("final") is not None)
        ideal = [r["best_seen"] for r in rs]
        row = {
            "runs": len(rs),
            "unknown": len(rs) - len(finals),
            "mean": sum(finals) / len(finals) if finals else None,
            "p10": finals[int(0.1 * (len(finals) - 1))] if finals else None,
            "worst": finals[0] if finals else None,
            "ideal_mean": sum(ideal) / len(ideal),
        }
        for th in thresholds:
            hits = []
            for r in rs:
                n = next((i + 1 for i, v in enumerate(r["curve"]) if v >= th), None)
                hits.append(n)
            reached = sorted(h for h in hits if h is not None)
            row[f"reach{int(th * 100)}"] = len(reached) / len(hits)
            row[f"evals{int(th * 100)}"] = (
                reached[len(reached) // 2] if reached else None
            )
        out[s] = row
    return out
