# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The ask/tell loop: budget, trace, checkpoint/resume, final confirmation.

:func:`run_search` is deliberately small -- everything a C++ host would
need to reimplement is here::

    while not budget exhausted:
        trials = strategy.ask(batch)          # [] -> converged
        obs, left = evaluator.evaluate(trials, max_fresh=remaining)
        strategy.tell(obs)
        trace / incumbent / checkpoint

Trials cut by the budget (``left``) are saved as outstanding; a resumed run
evaluates them first, so a run split into pieces makes the same decisions
as one run of the total budget (with an evaluation budget and a
deterministic backend).

The loop stops when the budget is spent, the strategy has nothing to
propose, ``stall_rounds`` rounds in a row spent nothing new (every ask
was a cache hit: the strategy is circling known configs), or the evaluator
is dirty (a timed-out call may still be running on the device).

Confirmation (``confirm=k``): the ``k`` best distinct configs -- and the
``baseline`` (the dispatcher's default pick), always -- are re-measured
together, interleaved, for ``confirm_rounds`` rounds; each round is one
interleaved batch and the unit of replication. The winner is the best
geometric mean of the per-round medians. This removes the winner's curse
(the best of many noisy measurements is biased low), reports via
:class:`~.stats.Race` which finalists are statistically indistinguishable
from the winner, and gives the winner's paired speedup over the baseline
with a one-sided p-value (a paired t test on the per-round log ratios).

The confirmation is paid from the budget: ``(k + baseline) * rounds``
evaluations are reserved under an evaluation budget, their cost under a
cost budget, and their estimated time under a seconds budget. A budget too
small to leave ``k`` search evaluations skips the confirmation (reported as
``confirmed["skipped"]``).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .db import TraceWriter
from .measure import Evaluator
from .params import SearchSpace
from .stats import Race, median, paired_t_greater, rel_spread
from .strategy import Strategy
from .trial import Budget, BudgetMeter, Observation, Trial

__all__ = ["CHECKPOINT_VERSION", "SearchResult", "run_search"]

CHECKPOINT_VERSION = 1


@dataclass
class SearchResult:
    """What a search found and what it spent."""

    best_key: str | None
    best_config: dict
    best_value: float
    best_genome: tuple[int, ...] | None
    stopped: str
    meter: BudgetMeter
    observations: list[Observation]
    strategy: str
    seconds: float
    confirmed: dict = field(default_factory=dict)
    origins: dict = field(default_factory=dict)
    index: dict = field(default_factory=dict)
    """``{key: position}`` in the searched list (set by ``search_configs``)."""
    fingerprint: dict = field(default_factory=dict)
    """The evaluator's ``{"tool": ..., "machine": ...}`` fingerprint."""

    @property
    def evaluations(self) -> int:
        return self.meter.evaluations

    def summary(self) -> dict:
        return {
            "best_key": self.best_key,
            "best_config": self.best_config,
            "best_value": self.best_value if math.isfinite(self.best_value) else None,
            "stopped": self.stopped,
            "strategy": self.strategy,
            "spent": self.meter.to_json(),
            "seconds": self.seconds,
            "confirmed": self.confirmed,
            "origins": self.origins,
        }


def _atomic_write(path: Path, data: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True))
    tmp.replace(path)


def run_search(
    space: SearchSpace,
    strategy: Strategy,
    evaluator: Evaluator,
    budget: Budget,
    *,
    batch: int = 1,
    trace: TraceWriter | None = None,
    checkpoint: str | os.PathLike | None = None,
    checkpoint_every: int = 1,
    resume: bool = False,
    stall_rounds: int = 64,
    confirm: int = 0,
    confirm_rounds: int = 2,
    race: Race | None = None,
    clock: Callable[[], float] = time.perf_counter,
    on_observation: Callable[[Observation], None] | None = None,
    baseline: Sequence[int] | None = None,
) -> SearchResult:
    """Run ``strategy`` against ``evaluator`` until ``budget`` is spent.

    ``checkpoint`` (a JSON path) is written every ``checkpoint_every``
    rounds and at the end; with ``resume`` an existing checkpoint is loaded
    first (strategy state, evaluator cache, spent budget, outstanding
    trials). The checkpoint is refused if the space fingerprint or strategy
    name differs. ``baseline`` is the genome of the reference config (the
    dispatcher's default) that the confirmation always includes.
    """
    if strategy.space is not space or evaluator.space is not space:
        raise ValueError(
            "run_search - space differs, expected the strategy, evaluator and "
            "run_search to share one space"
        )
    if int(batch) < 1:
        raise ValueError(f"run_search batch={batch}, expected >= 1")
    meter = BudgetMeter(budget)
    outstanding: list[Trial] = []
    observations: list[Observation] = []
    origins: dict[str, dict] = {}
    best: Observation | None = None
    ckpt = Path(checkpoint) if checkpoint else None
    rounds = 0
    if resume and ckpt is not None and ckpt.exists():
        data = json.loads(ckpt.read_text())
        if data.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"checkpoint {ckpt} - version={data.get('version')!r}, "
                f"expected {CHECKPOINT_VERSION}"
            )
        if data["space"] != space.fingerprint():
            raise ValueError(
                f"checkpoint {ckpt} - space fingerprint differs, expected this search "
                "space"
            )
        if data["strategy"]["name"] != strategy.name:
            raise ValueError(
                f"checkpoint {ckpt} - strategy={data['strategy']['name']!r}, "
                f"expected {strategy.name!r}"
            )
        strategy.load_state_dict(data["strategy"])
        evaluator.load_state_dict(data["evaluator"])
        meter.load(data["meter"])
        outstanding = [Trial.from_json(t) for t in data["outstanding"]]
        observations = [Observation.from_json(o) for o in data["observations"]]
        origins = data["origins"]
        rounds = int(data["rounds"])
        if data["best"] is not None:
            best = Observation.from_json(data["best"])
    strategy.attach_noise(evaluator.noise)
    if trace is not None and not resume:
        trace.header(
            space=space.describe(),
            strategy=strategy.name,
            seed=strategy.seed,
            budget=budget.to_json(),
        )

    def save() -> None:
        if ckpt is None:
            return
        _atomic_write(
            ckpt,
            {
                "version": CHECKPOINT_VERSION,
                "space": space.fingerprint(),
                "strategy": strategy.state_dict(),
                "evaluator": evaluator.state_dict(),
                "meter": meter.to_json(),
                "outstanding": [t.to_json() for t in outstanding],
                "observations": [o.to_json() for o in observations],
                "origins": origins,
                "rounds": rounds,
                "best": None if best is None else best.to_json(),
            },
        )

    t_start = clock() - meter.seconds
    stopped = ""
    stall = 0
    base_key = None if baseline is None else space.key(space.check(baseline))
    confirm = max(0, int(confirm))
    rounds_c = max(1, int(confirm_rounds))
    n_confirm = confirm + (1 if base_key is not None and confirm else 0)
    skipped = ""
    if (
        confirm
        and budget.evaluations is not None
        and budget.evaluations < n_confirm * rounds_c + confirm
    ):
        skipped = (
            f"budget evaluations={budget.evaluations} < "
            f"{n_confirm * rounds_c + confirm} (confirmation plus {confirm} "
            "search evaluations)"
        )
        confirm = n_confirm = 0
    reserve = n_confirm * rounds_c if budget.evaluations is not None else 0
    full_cost = evaluator.ladder[-1].cost
    reserve_cost = n_confirm * rounds_c * full_cost
    if confirm and budget.cost is not None and budget.cost <= reserve_cost:
        skipped = f"budget cost={budget.cost} <= confirmation cost {reserve_cost:g}"
        confirm = n_confirm = 0
        reserve_cost = 0.0

    def reserve_seconds() -> float:
        if not n_confirm or budget.seconds is None:
            return 0.0
        spent = [o.measure_s for o in observations if o.full and o.ok and not o.cached]
        per = median(spent) if spent else 0.0
        return n_confirm * rounds_c * per

    while True:
        meter.seconds = clock() - t_start
        why = meter.exhausted()
        if not why and reserve and meter.evaluations >= budget.evaluations - reserve:
            why = "evaluations"
        if (
            not why
            and reserve_cost
            and budget.cost is not None
            and meter.cost >= budget.cost - reserve_cost
        ):
            why = "cost"
        if (
            not why
            and budget.seconds is not None
            and meter.seconds >= budget.seconds - reserve_seconds()
        ):
            why = "seconds"
        if not why and evaluator.dirty:
            why = "evaluator_dirty"
        if why:
            stopped = why
            break
        if outstanding:
            trials, outstanding = outstanding, []
        else:
            trials = strategy.ask(int(batch))
        if not trials:
            stopped = "strategy_done" if strategy.done else "no_proposals"
            break
        remaining = meter.remaining_evaluations()
        if remaining is not None and reserve:
            remaining = max(0, remaining - reserve)
        deadline = None
        if budget.seconds is not None:
            deadline = t_start + budget.seconds - reserve_seconds()
        obs, left = evaluator.evaluate(trials, max_fresh=remaining, deadline=deadline)
        outstanding = list(left)
        before = meter.evaluations
        for ob in obs:
            fid = evaluator.ladder[ob.fidelity]
            meter.charge(ob, fid.cost)
            observations.append(ob)
            o = origins.setdefault(
                ob.trial.origin or "?", {"asks": 0, "fresh": 0, "best": 0}
            )
            o["asks"] += 1
            if not ob.cached and ob.status != "invalid":
                o["fresh"] += 1
            if ob.full and ob.ok and (best is None or ob.value < best.value):
                best = ob
                o["best"] += 1
            if trace is not None:
                trace.observation(
                    ob,
                    config=space.decode(ob.trial.genome),
                    elapsed=clock() - t_start,
                    incumbent=best.value if best else math.inf,
                    n=meter.evaluations,
                )
            if on_observation is not None:
                on_observation(ob)
        strategy.tell(obs)
        rounds += 1
        stall = 0 if meter.evaluations > before else stall + 1
        if ckpt is not None and rounds % max(1, int(checkpoint_every)) == 0:
            meter.seconds = clock() - t_start
            save()
        if stall >= int(stall_rounds):
            stopped = "stalled"
            break
    confirmed: dict = {"skipped": skipped} if skipped else {}
    if confirm and best is not None and not evaluator.dirty:
        confirmed = _confirm(
            evaluator,
            observations,
            confirm,
            rounds_c,
            race or Race(),
            meter,
            baseline=None if base_key is None else (base_key, tuple(baseline)),
        )
        if confirmed.get("winner") and confirmed["winner"] != best.key:
            winner = confirmed["winner"]
            best = next(
                (o for o in observations if o.key == winner and o.full and o.ok),
                None,
            ) or Observation(
                trial=Trial(genome=tuple(baseline), uid="confirm:baseline"),
                key=winner,
                status="ok",
                value=confirmed["values"][winner],
            )
    meter.seconds = clock() - t_start
    save()
    if trace is not None:
        trace.write(
            {
                "type": "end",
                "stopped": stopped,
                "spent": meter.to_json(),
                "best": None if best is None else best.key,
                "confirmed": confirmed,
            }
        )
    best_value = best.value if best is not None else math.inf
    if confirmed.get("winner"):
        best_value = confirmed["values"][confirmed["winner"]]
    return SearchResult(
        best_key=None if best is None else best.key,
        best_config={} if best is None else space.decode(best.trial.genome),
        best_value=best_value,
        best_genome=None if best is None else space.canonical_genome(best.trial.genome),
        stopped=stopped,
        meter=meter,
        observations=observations,
        strategy=strategy.name,
        seconds=meter.seconds,
        confirmed=confirmed,
        origins=origins,
        fingerprint=evaluator.fingerprint(),
    )


def _confirm(evaluator, observations, k, rounds, race, meter, *, baseline=None) -> dict:
    """Re-measure the ``k`` best distinct full-fidelity configs (and the
    ``baseline`` ``(key, genome)``) together, ``rounds`` interleaved
    rounds; a round is the unit of replication."""
    best_by_key: dict[str, Observation] = {}
    for ob in observations:
        if (
            ob.full
            and ob.ok
            and (ob.key not in best_by_key or ob.value < best_by_key[ob.key].value)
        ):
            best_by_key[ob.key] = ob
    finalists = sorted(best_by_key, key=lambda kk: (best_by_key[kk].value, kk))[:k]
    genomes = {kk: best_by_key[kk].trial.genome for kk in finalists}
    keys = list(finalists)
    base_key = None
    if baseline is not None:
        base_key, base_genome = baseline
        genomes.setdefault(base_key, base_genome)
        if base_key not in keys:
            keys.append(base_key)
    if len(keys) < 2:
        return {}
    per_round = evaluator.remeasure(keys, genomes, rounds=rounds)
    calls = {
        kk: [(r, median(v)) for r, v in enumerate(rs) if v]
        for kk, rs in per_round.items()
    }
    calls = {kk: c for kk, c in calls.items() if c}
    if not calls:
        return {}
    # Confirmation spends what run_search reserved: one evaluation per key
    # and round, at the full fidelity's cost.
    measured = sum(len(c) for c in calls.values())
    meter.evaluations += measured
    meter.cost += evaluator.ladder[-1].cost * measured

    def gmean(c) -> float:
        return math.exp(sum(math.log(v) for _r, v in c) / len(c))

    values = {kk: gmean(c) for kk, c in calls.items()}
    winner = min(values, key=lambda kk: (values[kk], kk))
    dominated = race.eliminate(calls)
    flat = {kk: [x for rs in per_round[kk] for x in rs] for kk in calls}
    out = {
        "finalists": finalists,
        "winner": winner,
        "values": values,
        "rounds": {kk: len(c) for kk, c in calls.items()},
        "samples": {kk: len(v) for kk, v in flat.items()},
        "spread": {kk: rel_spread(v) for kk, v in flat.items()},
        "round_spread": {kk: rel_spread([v for _r, v in c]) for kk, c in calls.items()},
        "dominated": dominated,
        "tied_with_winner": [
            kk for kk in calls if kk != winner and kk not in dominated
        ],
        "search_rank_winner": finalists[0] if finalists else None,
    }
    if base_key is not None:
        out["baseline"] = base_key
        if base_key in calls and winner in calls:
            wr = dict(calls[winner])
            logs = [
                math.log(v) - math.log(wr[r]) for r, v in calls[base_key] if r in wr
            ]
            if logs:
                out["speedup"] = math.exp(sum(logs) / len(logs))
                out["speedup_rounds"] = len(logs)
                # one-sided: the winner is faster than the baseline
                out["p_value"] = (
                    1.0 if winner == base_key else paired_t_greater(logs, 0.0)
                )
    return out
