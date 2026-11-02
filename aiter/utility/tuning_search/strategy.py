# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The strategy interface and the two reference strategies.

A strategy proposes trials (:meth:`Strategy.ask`) and learns from
observations (:meth:`Strategy.tell`). It never measures anything itself, so
the same object runs against a GPU backend, a recorded table, or a synthetic
objective, and a C++ host can drive the same protocol.

Contract:

- ``ask(n)`` returns up to ``n`` trials; ``[]`` means the strategy has
  nothing left to propose (exhausted or converged).
- ``tell(observations)`` takes the observations of the strategy's own trials
  *and* of trials other strategies asked (a portfolio shares everything).
  Own trials are recognized by ``uid``. Only full-fidelity observations
  feed :attr:`Strategy.history`.
- Every random draw goes through ``self.rng`` (seeded), and every
  container is iterated in insertion or sorted order, so a run is a pure
  function of the seed and the observation sequence.
- ``state_dict()`` is JSON data; ``load_state_dict`` restores it on a
  freshly constructed strategy with the same space and hyperparameters.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

from .params import Genome, SearchSpace
from .stats import NoiseModel
from .trial import Observation, Trial, rng_from_state, rng_state

__all__ = ["Exhaustive", "RandomSearch", "Strategy"]


class Strategy:
    """Base class: history bookkeeping, uids, RNG and state plumbing."""

    name = "strategy"

    def __init__(self, space: SearchSpace, seed: int = 0):
        self.space = space
        self.seed = int(seed)
        self.rng = random.Random(f"{self.name}:{self.seed}")
        self.history: dict[str, float] = {}
        self.genomes: dict[str, Genome] = {}
        self.pending: dict[str, Genome] = {}
        self._uid = 0
        self.noise: NoiseModel | None = None
        self.uid_prefix = f"{self.name}:{self.seed}"

    def attach_noise(self, noise: NoiseModel | None) -> None:
        """Share the evaluator's learned timing noise (``run_search`` does
        this), so "better" can mean "better beyond noise"."""
        self.noise = noise

    def gain_threshold(self, floor: float, samples: int) -> float:
        """The relative improvement that counts: ``floor``, or twice the
        noise of one call's median of ``samples`` reps when that is larger
        (:meth:`~.stats.NoiseModel.median_sigma`: the per-call part of the
        noise, which more reps do not average away, included)."""
        if self.noise is None or samples < 1:
            return floor
        return max(floor, 2.0 * self.noise.median_sigma(samples))

    # -------------------------------------------------------------- protocol
    def ask(self, n: int = 1) -> list[Trial]:
        raise NotImplementedError

    def tell(self, observations: Sequence[Observation]) -> None:
        for ob in observations:
            own = self.pending.pop(ob.trial.uid, None) is not None
            if ob.full and ob.key is not None:
                old = self.history.get(ob.key)
                if old is None or ob.value < old:
                    self.history[ob.key] = ob.value
                self.genomes.setdefault(
                    ob.key, self.space.canonical_genome(ob.trial.genome)
                )
            self.observe(ob, own)

    def observe(self, ob: Observation, own: bool) -> None:
        """Hook: one observation after the base bookkeeping."""

    @property
    def done(self) -> bool:
        return False

    # --------------------------------------------------------------- helpers
    def trial(self, genome: Sequence[int], *, fidelity: int = -1) -> Trial:
        self._uid += 1
        g = tuple(int(x) for x in genome)
        t = Trial(
            genome=g,
            uid=f"{self.uid_prefix}:{self._uid}",
            origin=self.name,
            fidelity=fidelity,
        )
        self.pending[t.uid] = g
        return t

    def best(self) -> tuple[str | None, float]:
        """``(key, value)`` of the best full-fidelity observation seen."""
        if not self.history:
            return None, math.inf
        key = min(self.history, key=lambda k: (self.history[k], k))
        return key, self.history[key]

    def seen(self, genome: Sequence[int]) -> bool:
        """Whether ``genome``'s config was observed or is pending here."""
        key = self.space.key(genome)
        if key is None:
            return True
        if key in self.history:
            return True
        return any(self.space.key(g) == key for g in self.pending.values())

    def pending_keys(self) -> set[str]:
        return {k for k in (self.space.key(g) for g in self.pending.values()) if k}

    # ----------------------------------------------------------------- state
    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "rng": rng_state(self.rng),
            "history": dict(self.history),
            "genomes": {k: list(g) for k, g in self.genomes.items()},
            "pending": {u: list(g) for u, g in self.pending.items()},
            "uid": self._uid,
            "own": self._state(),
        }

    def load_state_dict(self, d: Mapping) -> None:
        if d.get("name") != self.name:
            raise ValueError(
                f"state for strategy {d.get('name')!r}, expected {self.name!r}"
            )
        self.seed = int(d["seed"])
        self.rng = rng_from_state(d["rng"])
        self.history = {k: float(v) for k, v in d["history"].items()}
        self.genomes = {k: tuple(g) for k, g in d["genomes"].items()}
        self.pending = {u: tuple(g) for u, g in d["pending"].items()}
        self._uid = int(d["uid"])
        self._load(d["own"])

    def _state(self) -> dict:
        return {}

    def _load(self, d: Mapping) -> None:
        pass


def _f(x: float) -> float | None:
    """A float as JSON (``inf`` -> None)."""
    return x if math.isfinite(x) else None


def _uf(x) -> float:
    return math.inf if x is None else float(x)


class RandomSearch(Strategy):
    """Uniform random valid configs, never repeating a key. The baseline
    every other strategy must beat at equal budget."""

    name = "random"

    def __init__(self, space: SearchSpace, seed: int = 0, *, tries: int = 256):
        super().__init__(space, seed)
        self.tries = int(tries)
        self._exhausted = False

    def ask(self, n: int = 1) -> list[Trial]:
        out: list[Trial] = []
        taken = self.pending_keys()
        for _ in range(int(n)):
            for _attempt in range(self.tries):
                g = self.space.random_valid(self.rng, tries=8)
                if g is None:
                    continue
                k = self.space.key(g)
                if k in self.history or k in taken:
                    continue
                taken.add(k)
                out.append(self.trial(g))
                break
            else:
                self._exhausted = True
                break
        return out

    @property
    def done(self) -> bool:
        return self._exhausted

    def _state(self) -> dict:
        return {"exhausted": self._exhausted}

    def _load(self, d: Mapping) -> None:
        self._exhausted = bool(d["exhausted"])


class Exhaustive(Strategy):
    """Every valid config in lexicographic genome order: the oracle for
    small spaces and for offline validation."""

    name = "exhaustive"

    def __init__(self, space: SearchSpace, seed: int = 0):
        super().__init__(space, seed)
        self._pos = 0
        self._done = False
        self._it = None
        self._it_pos = 0

    def ask(self, n: int = 1) -> list[Trial]:
        out: list[Trial] = []
        taken = self.pending_keys()
        if self._it is None or self._it_pos != self._pos:
            self._it = self.space.enumerate()
            for _ in range(self._pos):
                next(self._it, None)
        it = self._it
        while len(out) < int(n):
            g = next(it, None)
            if g is None:
                self._done = True
                break
            self._pos += 1
            self._it_pos = self._pos
            k = self.space.key(g)
            if k is None or k in self.history or k in taken:
                continue
            taken.add(k)
            out.append(self.trial(g))
        return out

    @property
    def done(self) -> bool:
        return self._done

    def _state(self) -> dict:
        return {"pos": self._pos, "done": self._done}

    def _load(self, d: Mapping) -> None:
        self._pos = int(d["pos"])
        self._done = bool(d["done"])
