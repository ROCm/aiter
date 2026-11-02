# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Value types of the ask/tell loop: trials, observations, fidelities, budgets.

Everything here round-trips through JSON (``to_json`` / ``from_json``), which
is what makes a search resumable and its trace replayable. Values are
"lower is better" (a time); a config that failed anything -- compile, the
numeric check, a resource gate, a timeout -- has value ``inf`` and a status
that says why, so it can never look fast.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .params import Genome

__all__ = [
    "FAILED_STATUSES",
    "STATUSES",
    "STRUCTURAL_STATUSES",
    "TRANSIENT_STATUSES",
    "Budget",
    "BudgetMeter",
    "Fidelity",
    "Observation",
    "Status",
    "Trial",
    "rng_from_state",
    "rng_state",
]


class Status:
    """Observation statuses. Only ``OK`` carries a finite value."""

    OK = "ok"
    INVALID = "invalid"  # the space refused the config; nothing was spent
    COMPILE_ERROR = "compile_error"
    NUMERIC_FAIL = "numeric_fail"
    SPILL = "spill"
    TIMEOUT = "timeout"
    RUN_ERROR = "run_error"


STATUSES = (
    Status.OK,
    Status.INVALID,
    Status.COMPILE_ERROR,
    Status.NUMERIC_FAIL,
    Status.SPILL,
    Status.TIMEOUT,
    Status.RUN_ERROR,
)
FAILED_STATUSES = tuple(s for s in STATUSES if s != Status.OK)
# Failures that may not repeat (a hung launch, a driver error): they are
# remembered for one process and one fidelity only, never persisted.
TRANSIENT_STATUSES = (Status.TIMEOUT, Status.RUN_ERROR)
# Failures of the config itself: they hold at every fidelity of the same
# problem (proxy), and across processes while the toolchain is the same.
STRUCTURAL_STATUSES = (Status.COMPILE_ERROR, Status.NUMERIC_FAIL, Status.SPILL)


@dataclass(frozen=True)
class Fidelity:
    """One measurement fidelity: how a backend times a config.

    ``reps`` timed repetitions after ``warmup``; ``cold`` flushes caches
    before each rep; ``proxy`` names a cheaper stand-in problem the backend
    understands ("" is the real problem). ``cost`` is the relative budget
    charge of one measurement. Samples at fidelities with the same
    ``proxy`` and ``cold`` are comparable and may be pooled
    (:attr:`pool_id`).
    """

    name: str = "full"
    reps: int = 20
    warmup: int = 3
    cold: bool = False
    proxy: str = ""
    cost: float = 1.0

    def __post_init__(self):
        if int(self.reps) < 1:
            raise ValueError(f"Fidelity reps={self.reps}, expected >= 1")
        if int(self.warmup) < 0:
            raise ValueError(f"Fidelity warmup={self.warmup}, expected >= 0")
        if not (self.cost > 0 and math.isfinite(self.cost)):
            raise ValueError(f"Fidelity cost={self.cost}, expected a finite value > 0")

    @property
    def pool_id(self) -> str:
        return f"{self.proxy}|{'cold' if self.cold else 'hot'}"

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "reps": self.reps,
            "warmup": self.warmup,
            "cold": self.cold,
            "proxy": self.proxy,
            "cost": self.cost,
        }

    @classmethod
    def from_json(cls, d: Mapping) -> Fidelity:
        return cls(**dict(d))


@dataclass(frozen=True)
class Trial:
    """A point a strategy asks to have measured.

    ``fidelity`` indexes the evaluator's ladder; ``-1`` is the top (full)
    fidelity. ``uid`` is unique per strategy instance and is how a strategy
    recognizes the observation of its own ask; ``origin`` names who asked
    (a strategy, or ``warm:<source>``).
    """

    genome: Genome
    uid: str
    origin: str = ""
    fidelity: int = -1

    def to_json(self) -> dict:
        return {
            "genome": list(self.genome),
            "uid": self.uid,
            "origin": self.origin,
            "fidelity": self.fidelity,
        }

    @classmethod
    def from_json(cls, d: Mapping) -> Trial:
        return cls(
            genome=tuple(int(x) for x in d["genome"]),
            uid=str(d["uid"]),
            origin=str(d.get("origin", "")),
            fidelity=int(d.get("fidelity", -1)),
        )


@dataclass(frozen=True)
class Observation:
    """The outcome of one trial.

    ``value`` aggregates ``samples`` (the median by default); ``inf`` unless
    ``status`` is ``ok``. ``full`` is true when measured at the top fidelity,
    the only values strategies compare across configs. ``cached`` marks a
    result served from the evaluator's cache (no budget spent). ``batch``
    groups measurements interleaved in one round, whose samples are paired
    rep by rep.
    """

    trial: Trial
    key: str | None
    status: str
    value: float = math.inf
    samples: tuple[float, ...] = ()
    fidelity: int = -1
    full: bool = True
    compile_s: float = 0.0
    measure_s: float = 0.0
    cached: bool = False
    batch: int = -1
    detail: str = ""
    resources: Mapping[str, float] = field(default_factory=dict)
    pool: str = ""

    @property
    def ok(self) -> bool:
        return self.status == Status.OK and math.isfinite(self.value)

    def to_json(self) -> dict:
        return {
            "trial": self.trial.to_json(),
            "key": self.key,
            "status": self.status,
            "value": self.value if math.isfinite(self.value) else None,
            "samples": list(self.samples),
            "fidelity": self.fidelity,
            "full": self.full,
            "compile_s": self.compile_s,
            "measure_s": self.measure_s,
            "cached": self.cached,
            "batch": self.batch,
            "detail": self.detail,
            "resources": dict(self.resources),
            "pool": self.pool,
        }

    @classmethod
    def from_json(cls, d: Mapping) -> Observation:
        value = d.get("value")
        return cls(
            trial=Trial.from_json(d["trial"]),
            key=d.get("key"),
            status=str(d["status"]),
            value=math.inf if value is None else float(value),
            samples=tuple(float(x) for x in d.get("samples", ())),
            fidelity=int(d.get("fidelity", -1)),
            full=bool(d.get("full", True)),
            compile_s=float(d.get("compile_s", 0.0)),
            measure_s=float(d.get("measure_s", 0.0)),
            cached=bool(d.get("cached", False)),
            batch=int(d.get("batch", -1)),
            detail=str(d.get("detail", "")),
            resources=dict(d.get("resources", {})),
            pool=str(d.get("pool", "")),
        )

    def replace_trial(
        self,
        trial: Trial,
        *,
        cached: bool,
        fidelity: int | None = None,
        full: bool | None = None,
    ) -> Observation:
        """This outcome reported for another trial (a cache hit).

        ``fidelity`` / ``full`` re-label it at the asked fidelity: a failure
        cached at one rung answers an ask at another (a config that does not
        build or computes wrong values fails at every fidelity), and a
        strategy waiting for a full-fidelity answer must get one.
        """
        return Observation(
            trial=trial,
            key=self.key,
            status=self.status,
            value=self.value,
            samples=self.samples,
            fidelity=self.fidelity if fidelity is None else int(fidelity),
            full=self.full if full is None else bool(full),
            compile_s=0.0 if cached else self.compile_s,
            measure_s=0.0 if cached else self.measure_s,
            cached=cached,
            batch=self.batch,
            detail=self.detail,
            resources=self.resources,
            pool=self.pool,
        )


@dataclass(frozen=True)
class Budget:
    """Search limits; any that is set stops the search.

    ``evaluations`` counts fresh measurements (a config reaching the
    backend; cache hits and configs the space refuses are free).
    ``seconds`` is wall time of the whole loop, compiles included (not
    deterministic: prefer ``evaluations`` for reproducible runs). ``cost``
    sums :attr:`Fidelity.cost` of fresh measurements, so a
    multi-fidelity strategy can spend cheap rungs freely.
    """

    evaluations: int | None = None
    seconds: float | None = None
    cost: float | None = None

    def __post_init__(self):
        if self.evaluations is not None and int(self.evaluations) < 1:
            raise ValueError(f"Budget evaluations={self.evaluations}, expected >= 1")
        if self.seconds is not None and not self.seconds > 0:
            raise ValueError(f"Budget seconds={self.seconds}, expected > 0")
        if self.cost is not None and not self.cost > 0:
            raise ValueError(f"Budget cost={self.cost}, expected > 0")
        if self.evaluations is None and self.seconds is None and self.cost is None:
            raise ValueError(
                "Budget limits - evaluations=None, seconds=None, cost=None, "
                "expected at least one set"
            )

    def to_json(self) -> dict:
        return {
            "evaluations": self.evaluations,
            "seconds": self.seconds,
            "cost": self.cost,
        }

    @classmethod
    def from_json(cls, d: Mapping) -> Budget:
        return cls(
            evaluations=d.get("evaluations"),
            seconds=d.get("seconds"),
            cost=d.get("cost"),
        )


@dataclass
class BudgetMeter:
    """What a search has spent against its :class:`Budget`."""

    budget: Budget
    evaluations: int = 0
    seconds: float = 0.0
    cost: float = 0.0
    cache_hits: int = 0
    refused: int = 0
    compile_s: float = 0.0

    def charge(self, ob: Observation, cost: float) -> None:
        if ob.status == Status.INVALID:
            self.refused += 1
            return
        if ob.cached:
            self.cache_hits += 1
            return
        self.evaluations += 1
        self.compile_s += ob.compile_s
        if ob.status == Status.OK or ob.measure_s > 0:
            self.cost += cost

    def remaining_evaluations(self) -> int | None:
        if self.budget.evaluations is None:
            return None
        return max(0, int(self.budget.evaluations) - self.evaluations)

    def exhausted(self) -> str:
        """Why the budget is spent, or ``""``."""
        b = self.budget
        if b.evaluations is not None and self.evaluations >= b.evaluations:
            return "evaluations"
        if b.seconds is not None and self.seconds >= b.seconds:
            return "seconds"
        if b.cost is not None and self.cost >= b.cost:
            return "cost"
        return ""

    def to_json(self) -> dict:
        return {
            "budget": self.budget.to_json(),
            "evaluations": self.evaluations,
            "seconds": self.seconds,
            "cost": self.cost,
            "cache_hits": self.cache_hits,
            "refused": self.refused,
            "compile_s": self.compile_s,
        }

    def load(self, d: Mapping) -> None:
        self.evaluations = int(d["evaluations"])
        self.seconds = float(d["seconds"])
        self.cost = float(d["cost"])
        self.cache_hits = int(d.get("cache_hits", 0))
        self.refused = int(d.get("refused", 0))
        self.compile_s = float(d.get("compile_s", 0.0))


def rng_state(rng: random.Random) -> list:
    """``rng``'s state as JSON data."""
    version, internal, gauss = rng.getstate()
    return [version, list(internal), gauss]


def rng_from_state(state: Sequence) -> random.Random:
    rng = random.Random()
    version, internal, gauss = state
    rng.setstate((int(version), tuple(int(x) for x in internal), gauss))
    return rng
