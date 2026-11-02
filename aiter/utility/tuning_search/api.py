# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The entry points: build a strategy by name, tune a space, record the
winner, search a flat config list.

These are what kernel tuning scripts and the CLI call. The data they
exchange -- knob dicts, ``tuning_id``, :class:`~.db.TunedRecord` JSON -- is
language-neutral.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .db import TraceWriter, TunedRecord, TuningDB
from .de import JDE, DERand, DifferentialEvolution
from .driver import SearchResult, run_search
from .halving import Hyperband, SuccessiveHalving
from .local import HillClimb
from .measure import Backend, CallableBackend, CompileOutcome, Evaluator
from .params import SearchSpace, flatten_spec
from .portfolio import Portfolio, WarmStart
from .stats import Race, rel_spread
from .strategy import Exhaustive, RandomSearch, Strategy
from .surrogate import GPEI, TPE, Prior
from .trial import Budget, Fidelity

__all__ = [
    "DEFAULT_PORTFOLIO",
    "FAMILY_DEFAULTS",
    "STRATEGY_NAMES",
    "default_ladder",
    "make_strategy",
    "recommend",
    "record_result",
    "search_configs",
    "tune",
]

STRATEGY_NAMES = (
    "portfolio",
    "de",
    "jde",
    "de_rand",
    "tpe",
    "gp",
    "hill",
    "sh",
    "hyperband",
    "random",
    "exhaustive",
)
DEFAULT_PORTFOLIO = ("hill", "gp", "de")

# Defaults per kernel family, chosen by replaying every strategy offline
# (:mod:`.replay`) against recorded rocKE sweeps of each family: GEMM and
# conv-wgrad tile grids, permute / elementwise / reduction variants, and CK
# contraction instance lists, several seeds per problem. ``warm`` lists the
# warm-start sources worth asking for, most useful first ("dtype": the same
# problem's winners in another dtype on this arch; "neighbor": winners of
# the nearest shapes; "selector": the selector's top-k; "arch": winners on
# another arch). Families not listed use "default".
FAMILY_DEFAULTS: dict[str, dict[str, Any]] = {
    "default": {"strategy": "portfolio", "members": DEFAULT_PORTFOLIO},
    # Large dense tile grids (hundreds to tens of thousands of configs): the
    # default portfolio tracks the best single strategy closely at every
    # budget replayed.
    "gemm": {
        "strategy": "portfolio",
        "members": DEFAULT_PORTFOLIO,
        "warm": ("dtype", "neighbor", "selector"),
    },
    # Sparse conv-wgrad grids: a solo climber started from its stratified
    # probe led cold and under noise.
    "conv": {"strategy": "hill", "warm": ("neighbor", "selector", "dtype")},
    # Instance lists of a few hundred configs: GP leads at small budgets.
    "contraction": {
        "strategy": "portfolio",
        "members": ("gp", "hill"),
        "warm": ("neighbor", "dtype", "arch"),
    },
    # permute variants x knobs (about a hundred configs): GP converges
    # fastest.
    "permute": {"strategy": "gp", "warm": ("neighbor", "arch")},
    # elementwise / reduction spaces are a few dozen configs at most:
    # enumerate them.
    "elementwise": {"strategy": "exhaustive"},
    "reduce": {"strategy": "exhaustive"},
}


def recommend(
    family: str | None = None,
    space_size: int | None = None,
    budget: int | None = None,
) -> dict[str, Any]:
    """The default strategy for a family (see :data:`FAMILY_DEFAULTS`).

    A space no larger than the evaluation budget is enumerated: exhaustive
    is then the only strategy that guarantees the optimum. Returns a copy of
    the family's entry (``strategy``, optional ``members`` and ``warm``).
    """
    if space_size is not None and budget is not None and 0 < space_size <= budget:
        return {"strategy": "exhaustive"}
    entry = FAMILY_DEFAULTS.get(family or "default", FAMILY_DEFAULTS["default"])
    return dict(entry)


def _space_size(space: SearchSpace) -> int:
    listed = space.listed()
    return len(listed) if listed is not None else space.size()


def default_ladder(reps: int = 20, *, cold: bool = False) -> tuple[Fidelity, ...]:
    """A three-rung rep ladder for the multi-fidelity strategies: 3, 8 and
    ``reps`` reps of the same (pooled) measurement."""
    return (
        Fidelity("r3", reps=3, warmup=1, cold=cold, cost=max(0.05, 3 / reps)),
        Fidelity("r8", reps=8, warmup=2, cold=cold, cost=max(0.1, 8 / reps)),
        Fidelity("full", reps=reps, warmup=3, cold=cold, cost=1.0),
    )


def _init_genomes(space: SearchSpace, warm: Sequence[WarmStart], rng_seed: int):
    import random

    rng = random.Random(f"warm:{rng_seed}")
    out = []
    for w in warm:
        r = space.repair(space.encode(w.config), rng)
        if r is not None:
            out.append(r)
    return out


def make_strategy(
    name: str,
    space: SearchSpace,
    seed: int = 0,
    *,
    prior: Prior | None = None,
    warm: Sequence[WarmStart] = (),
    ladder: Sequence[Fidelity] = (Fidelity(),),
    members: Sequence[str] = DEFAULT_PORTFOLIO,
    options: Mapping[str, Any] | None = None,
) -> Strategy:
    """A strategy by name (:data:`STRATEGY_NAMES`).

    ``warm`` starts become the portfolio's first evaluations, or the
    initial population / start point of a single strategy. ``ladder`` is
    the evaluator's fidelity ladder (the multi-fidelity strategies use all
    of it; the others measure at the top). ``options`` are passed to the
    strategy's constructor.
    """
    opts = dict(options or {})
    init = _init_genomes(space, warm, seed) if name != "portfolio" else []
    fids = tuple(range(len(ladder) - 1)) + (-1,)
    pooled = len({f.pool_id for f in ladder}) == 1
    if name == "portfolio":
        if not members:
            raise ValueError("make_strategy - members=(), expected at least one member")
        if "portfolio" in members:
            raise ValueError(
                "make_strategy - members contains 'portfolio', expected no nested "
                "portfolio"
            )
        # A climber inside a portfolio starts from the others' finds, so it
        # skips its own stratified probe (offline replay: the probe helps a
        # solo climber and costs a portfolio at small budgets).
        built = [
            make_strategy(
                m,
                space,
                seed * 101 + i + 1,
                prior=prior,
                ladder=ladder,
                options={"n_init": 0} if m == "hill" else None,
            )
            for i, m in enumerate(members)
        ]
        return Portfolio(space, seed, members=built, warm=warm, **opts)
    if name == "de":
        opts = {"variant": "current_to_pbest1bin", "adapt": "shade", **opts}
        return DifferentialEvolution(space, seed, init=init, **opts)
    if name == "jde":
        return JDE(space, seed, init=init, **opts)
    if name == "de_rand":
        return DERand(space, seed, init=init, **opts)
    if name == "tpe":
        return TPE(space, seed, prior=prior, init=init, **opts)
    if name == "gp":
        return GPEI(space, seed, prior=prior, init=init, **opts)
    if name == "hill":
        if "start" not in opts:
            opts["start"] = init[0] if init else None
            opts.setdefault("init", init[1:])
        return HillClimb(space, seed, **opts)
    if name == "sh":
        return SuccessiveHalving(
            space, seed, fidelities=fids, pooled=pooled, init=init, **opts
        )
    if name == "hyperband":
        return Hyperband(space, seed, fidelities=fids, pooled=pooled, init=init, **opts)
    if name == "random":
        return RandomSearch(space, seed, **opts)
    if name == "exhaustive":
        return Exhaustive(space, seed)
    raise ValueError(f"strategy={name!r}, expected one of {STRATEGY_NAMES}")


def tune(
    space: SearchSpace,
    backend: Backend,
    *,
    strategy: str | Strategy = "portfolio",
    budget: Budget | None = None,
    seed: int = 0,
    ladder: Sequence[Fidelity] | None = None,
    prior: Prior | None = None,
    warm: Sequence[WarmStart] = (),
    batch: int = 1,
    trace: str | os.PathLike | TraceWriter | None = None,
    checkpoint: str | os.PathLike | None = None,
    resume: bool = False,
    confirm: int = 3,
    confirm_rounds: int = 2,
    cache_path: str | os.PathLike | None = None,
    compile_timeout: float | None = None,
    measure_timeout: float | None = None,
    reject_spill: bool = True,
    strategy_options: Mapping[str, Any] | None = None,
    members: Sequence[str] = DEFAULT_PORTFOLIO,
    race: Race | None = None,
    baseline: Sequence[int] | None = None,
    environment: Mapping[str, object] | None = None,
) -> SearchResult:
    """Search ``space`` with ``backend`` and return the result.

    The default budget is 64 evaluations; ``confirm`` finalists (plus the
    ``baseline`` genome, when given) are re-measured together at the end,
    ``confirm_rounds`` interleaved rounds, out of that budget. Pass a name
    or a ready :class:`Strategy` (built over this ``space``). ``"auto"``
    picks by :func:`recommend` (no family: the space size against the
    budget, else the default portfolio); ``"auto:<family>"`` uses that
    family's entry of :data:`FAMILY_DEFAULTS`. Without a ``ladder`` the
    multi-fidelity strategies (by name, or an instance with
    ``fidelities``) get :func:`default_ladder`, the others one full
    fidelity.
    """
    budget = budget or Budget(evaluations=64)
    if isinstance(strategy, str) and (
        strategy == "auto" or strategy.startswith("auto:")
    ):
        family = strategy.split(":", 1)[1] if ":" in strategy else None
        rec = recommend(family, _space_size(space), budget.evaluations)
        strategy = rec["strategy"]
        members = tuple(rec.get("members", members))
    fids = getattr(strategy, "fidelities", None)
    if ladder is None:
        multi = strategy in ("sh", "hyperband") or (
            fids is not None and any(int(f) >= 0 for f in fids)
        )
        ladder = default_ladder() if multi else (Fidelity(),)
    if fids is not None:
        bad = [f for f in fids if int(f) >= len(ladder)]
        if bad:
            raise ValueError(
                f"tune: strategy fidelities={tuple(fids)} - index {bad[0]}, expected "
                f"-1 or < len(ladder)={len(ladder)}"
            )
    evaluator = Evaluator(
        space,
        backend,
        ladder=ladder,
        reject_spill=reject_spill,
        compile_timeout=compile_timeout,
        measure_timeout=measure_timeout,
        cache_path=cache_path,
        environment=environment,
    )
    if isinstance(strategy, Strategy):
        strat = strategy
    else:
        strat = make_strategy(
            strategy,
            space,
            seed,
            prior=prior,
            warm=warm,
            ladder=ladder,
            members=members,
            options=strategy_options,
        )
    writer = (
        trace
        if isinstance(trace, TraceWriter) or trace is None
        else TraceWriter(trace, append=resume)
    )
    return run_search(
        space,
        strat,
        evaluator,
        budget,
        batch=batch,
        trace=writer,
        checkpoint=checkpoint,
        resume=resume,
        confirm=confirm,
        confirm_rounds=confirm_rounds,
        race=race,
        baseline=baseline,
    )


def record_result(
    result: SearchResult,
    *,
    family: str,
    candidate: str,
    arch: str,
    problem: str,
    tuning_id: str,
    config_key: str,
    knobs: Mapping[str, object],
    db: TuningDB | None = None,
    algorithm: str = "",
    spec_id: str = "",
    features: Mapping[str, float] | None = None,
    unit: str = "ms",
    seed: int = 0,
    trace: str = "",
    stale: bool = False,
) -> TunedRecord | None:
    """A :class:`TunedRecord` for ``result``'s winner (stored in ``db``).

    With a confirmation round, ``value``, ``samples`` and ``spread`` all
    describe the confirmation samples, and the paired speedup over the
    baseline and its p-value come along; without one they describe the
    search's own measurements of the winner (and the record carries no
    speedup, so it is never served)."""
    if result.best_key is None:
        return None
    conf = result.confirmed or {}
    if conf.get("winner") == result.best_key:
        n_samples = int(conf.get("samples", {}).get(result.best_key, 0))
        spread = float(conf.get("spread", {}).get(result.best_key, 0.0))
    else:
        samples = []
        for ob in result.observations:
            if ob.key == result.best_key and ob.full and ob.ok:
                samples.extend(ob.samples)
        n_samples, spread = len(samples), rel_spread(samples)
    fp = result.fingerprint or {}
    rec = TunedRecord(
        family=family,
        candidate=candidate,
        arch=arch,
        problem=problem,
        tuning_id=tuning_id,
        config_key=config_key,
        knobs=dict(knobs),
        value=float(result.best_value),
        unit=unit,
        algorithm=algorithm,
        spec_id=spec_id,
        features=dict(features or {}),
        samples=n_samples,
        spread=spread,
        strategy=result.strategy,
        evaluations=result.evaluations,
        seed=seed,
        trace=trace,
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        speedup=conf.get("speedup"),
        p_value=conf.get("p_value"),
        baseline_key=str(conf.get("baseline") or ""),
        rounds=int(conf.get("speedup_rounds", 0)),
        toolchain=str(fp.get("tool", "")),
        machine=str(fp.get("machine", "")),
    )
    if db is not None:
        db.put(rec, stale=stale)
    return rec


def search_configs(
    configs: Sequence[Any],
    *,
    measure: Callable[[Any, Fidelity], float | Sequence[float]],
    check: Callable[[Any], bool | tuple[bool, str]],
    flatten: Callable[[Any], Mapping[str, object]] = flatten_spec,
    **tune_kwargs,
) -> tuple[int | None, SearchResult]:
    """Search a flat list of configs (specs, AutotuneConfigs...) instead of
    sweeping all of them; returns ``(winner index, result)``.

    Each config is flattened to scalar fields (:func:`.params.flatten_spec`)
    and the list becomes a :meth:`SearchSpace.from_configs` space, so the
    strategies move through field space and land on listed configs.
    ``measure(config, fidelity)`` returns a time or per-rep times;
    ``check(config)`` is the mandatory numeric check.
    """
    rows = [dict(flatten(c)) for c in configs]
    space = SearchSpace.from_configs(rows, name="configs")
    members = space.members

    def index(key: str) -> int:
        return members[key]

    def do_check(config, key, artifact):
        res = check(configs[artifact])
        if isinstance(res, tuple):
            return bool(res[0]), str(res[1])
        return bool(res), "" if res else "numeric check failed"

    def do_measure(config, key, artifact, fidelity):
        res = measure(configs[artifact], fidelity)
        if isinstance(res, (int, float)):
            return [float(res)]
        return [float(x) for x in res]

    backend = CallableBackend(
        compile=lambda config, key: CompileOutcome(True, artifact=index(key)),
        check=do_check,
        measure=do_measure,
    )
    result = tune(space, backend, **tune_kwargs)
    result.index = dict(members)
    winner = None if result.best_key is None else index(result.best_key)
    return winner, result
