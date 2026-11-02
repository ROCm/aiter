# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Multi-fidelity search: successive halving and Hyperband, with racing.

Fidelity is how a config is measured -- few or many reps, a proxy problem,
hot or cold caches -- as a ladder the evaluator owns (cheapest first, the
last entry being the real measurement). These strategies spend the cheap
rungs on many configs and the expensive ones on few:

- :class:`SuccessiveHalving` measures ``n_configs`` configs at the first
  rung, keeps the best ``1/eta`` (after :class:`~.stats.Race` has dropped
  every config statistically dominated by the leader) and promotes them to
  the next rung, until the last rung picks the winner.
- :class:`Hyperband` cycles successive-halving brackets that trade the
  number of configs against the starting rung, hedging against a cheap
  rung that ranks configs badly.

Racing works on measurement *calls* (one value per call: its median),
never on the reps of one call as if they were independent (see
:mod:`.stats`). A config's calls are those of the current rung, plus -- with
``pooled`` -- its earlier calls under the same fidelity pool (same proxy,
same cache state; each observation carries its pool). Calls taken in one
interleaved batch pair up across configs. With one call per config the
race can only use a learned per-call noise (the evaluator learns it from
configs measured more than once); without one it drops nothing, and the
rung keeps the best ``1/eta`` by value. Each rung races at ``alpha /
rungs`` so repeated elimination keeps the family-wise error near ``alpha``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .params import Genome, SearchSpace
from .stats import Race
from .strategy import Strategy, _f, _uf
from .trial import Observation, Trial

__all__ = ["Hyperband", "SuccessiveHalving"]

_DEFAULT_RACE = Race()


class SuccessiveHalving(Strategy):
    """One successive-halving bracket over trial fidelities ``fidelities``
    (evaluator ladder indices, cheapest first; ``-1`` is the full one)."""

    name = "sh"

    def __init__(
        self,
        space: SearchSpace,
        seed: int = 0,
        *,
        fidelities: Sequence[int] = (-1,),
        n_configs: int = 27,
        eta: float = 3.0,
        race: Race | None = _DEFAULT_RACE,
        pooled: bool = True,
        init: Sequence[Sequence[int]] = (),
        exclude: Sequence[str] = (),
    ):
        super().__init__(space, seed)
        if not fidelities:
            raise ValueError(
                "SuccessiveHalving - fidelities=(), expected at least one fidelity"
            )
        if int(n_configs) < 1:
            raise ValueError(f"SuccessiveHalving n_configs={n_configs}, expected >= 1")
        if not eta > 1.0:
            raise ValueError(f"SuccessiveHalving eta={eta}, expected > 1")
        self.fidelities = tuple(int(f) for f in fidelities)
        self.n_configs = int(n_configs)
        self.eta = float(eta)
        self.race = race
        self.pooled = bool(pooled)
        self._seeds = [space.check(g) for g in init]
        self._exclude = set(exclude)
        self.rung = 0
        self.members: list[str] = []
        self.member_genomes: dict[str, list[int]] = {}
        self.rung_values: dict[str, float] = {}
        self.rung_batch: dict[str, int] = {}
        # Per key: every call as [batch, median value, pool, reps].
        self.calls: dict[str, list[list]] = {}
        self._pool: str | None = None
        self.asked: dict[str, str] = {}
        self.dropped: dict[str, str] = {}
        self._started = False
        self._done = False
        self.winner: str | None = None

    # -------------------------------------------------------------- start
    def _start(self) -> None:
        self._started = True
        keys: list[str] = []
        genomes: dict[str, list[int]] = {}

        def add(g) -> None:
            k = self.space.key(g)
            if k is None or k in genomes or k in self._exclude:
                return
            keys.append(k)
            genomes[k] = list(self.space.canonical_genome(g))

        for g in self._seeds:
            r = self.space.repair(g, self.rng)
            if r is not None:
                add(r)
        for g in self.space.stratified(self.rng, self.n_configs + 4):
            if len(keys) >= self.n_configs:
                break
            add(g)
        for _ in range(4 * self.n_configs):
            if len(keys) >= self.n_configs:
                break
            g = self.space.random_valid(self.rng)
            if g is not None:
                add(g)
        self.members = keys[: self.n_configs]
        self.member_genomes = {k: genomes[k] for k in self.members}
        if not self.members:
            self._done = True

    # ----------------------------------------------------------- protocol
    def ask(self, n: int = 1) -> list[Trial]:
        if not self._started:
            self._start()
        out: list[Trial] = []
        if self._done:
            return out
        fid = self.fidelities[self.rung]
        for k in self.members:
            if len(out) >= int(n):
                break
            if k in self.asked or k in self.rung_values:
                continue
            t = self.trial(self.member_genomes[k], fidelity=fid)
            self.asked[k] = t.uid
            out.append(t)
        return out

    def observe(self, ob: Observation, own: bool) -> None:
        if not own or self._done or ob.key not in self.asked:
            return
        if self.asked[ob.key] != ob.trial.uid:
            return
        del self.asked[ob.key]
        if not ob.ok:
            self.dropped[ob.key] = ob.status
            self.members.remove(ob.key)
        else:
            self.rung_values[ob.key] = ob.value
            self.rung_batch[ob.key] = ob.batch
            self._pool = ob.pool
            self.calls.setdefault(ob.key, []).append(
                [int(ob.batch), float(ob.value), ob.pool, len(ob.samples)]
            )
        if all(k in self.rung_values for k in self.members):
            self._advance()

    def _race_calls(self, k: str) -> list[tuple[int, float]]:
        """``k``'s calls the race may use: this rung's, plus (``pooled``)
        earlier ones in the same fidelity pool."""
        out = []
        calls = self.calls.get(k, [])
        for i, (batch, value, pool, _reps) in enumerate(calls):
            last = i == len(calls) - 1
            if last or (self.pooled and pool == self._pool):
                out.append((int(batch), float(value)))
        return out

    def _advance(self) -> None:
        if not self.members:
            self._done = True
            return
        last = self.rung == len(self.fidelities) - 1
        survivors = list(self.members)
        if self.race is not None and len(survivors) > 1:
            calls = {k: self._race_calls(k) for k in survivors}
            sigma = self.noise.call_sigma if self.noise is not None else None
            looks = len(self.fidelities)
            for k in self.race.eliminate(calls, sigma=sigma, looks=looks):
                self.dropped[k] = "raced"
                survivors.remove(k)

        def score(k: str) -> tuple[float, str]:
            # reps-weighted mean log of the calls the race used
            if not self.pooled:
                return self.rung_values[k], k
            used = [
                (math.log(v), max(1, reps))
                for _b, v, pool, reps in self.calls[k]
                if pool == self._pool and v > 0
            ] or [(math.log(self.rung_values[k]), 1)]
            w = sum(r for _x, r in used)
            return sum(x * r for x, r in used) / w, k

        survivors.sort(key=score)
        if last:
            self.winner = survivors[0]
            self.members = survivors
            self._done = True
            return
        keep = max(1, math.ceil(len(self.members) / self.eta))
        self.members = survivors[:keep]
        self.rung += 1
        self.rung_values = {}
        self.rung_batch = {}

    @property
    def done(self) -> bool:
        return self._done

    # -------------------------------------------------------------- state
    def _state(self) -> dict:
        return {
            "rung": self.rung,
            "members": self.members,
            "member_genomes": self.member_genomes,
            "rung_values": {k: _f(v) for k, v in self.rung_values.items()},
            "rung_batch": self.rung_batch,
            "calls": self.calls,
            "pool": self._pool,
            "asked": self.asked,
            "dropped": self.dropped,
            "started": self._started,
            "done": self._done,
            "winner": self.winner,
        }

    def _load(self, d: Mapping) -> None:
        self.rung = int(d["rung"])
        self.members = list(d["members"])
        self.member_genomes = {k: list(g) for k, g in d["member_genomes"].items()}
        self.rung_values = {k: _uf(v) for k, v in d["rung_values"].items()}
        self.rung_batch = {k: int(v) for k, v in d["rung_batch"].items()}
        self.calls = {k: [list(c) for c in v] for k, v in d["calls"].items()}
        self._pool = d.get("pool")
        self.asked = dict(d["asked"])
        self.dropped = dict(d["dropped"])
        self._started = bool(d["started"])
        self._done = bool(d["done"])
        self.winner = d["winner"]


class Hyperband(Strategy):
    """Hyperband: successive-halving brackets from aggressive (many configs,
    cheapest rung) to conservative (few configs, full fidelity), repeated
    with fresh configs. Each new bracket is seeded with the incumbent so it
    must beat it on equal footing."""

    name = "hyperband"

    def __init__(
        self,
        space: SearchSpace,
        seed: int = 0,
        *,
        fidelities: Sequence[int] = (-1,),
        eta: float = 3.0,
        race: Race | None = _DEFAULT_RACE,
        pooled: bool = True,
        max_configs: int = 81,
        init: Sequence[Sequence[int]] = (),
    ):
        super().__init__(space, seed)
        if not eta > 1.0:
            raise ValueError(f"Hyperband eta={eta}, expected > 1")
        self.fidelities = tuple(int(f) for f in fidelities)
        self.eta = float(eta)
        self.race = race
        self.pooled = bool(pooled)
        self.max_configs = int(max_configs)
        self._seeds = [space.check(g) for g in init]
        self.bracket = 0
        self.inner: SuccessiveHalving | None = None
        self._done = False
        self._empty = 0

    def _bracket_shape(self, b: int) -> tuple[int, tuple[int, ...]]:
        s_max = len(self.fidelities) - 1
        s = s_max - (b % (s_max + 1))
        n = math.ceil((s_max + 1) / (s + 1) * self.eta**s)
        return min(n, self.max_configs), self.fidelities[s_max - s :]

    def _new_bracket(self) -> None:
        n, fids = self._bracket_shape(self.bracket)
        seeds: list[Genome] = list(self._seeds) if self.bracket == 0 else []
        best_key, _v = self.best()
        if best_key is not None:
            seeds.insert(0, self.genomes[best_key])
        self.inner = SuccessiveHalving(
            self.space,
            seed=self.seed * 1009 + self.bracket,
            fidelities=fids,
            n_configs=n,
            eta=self.eta,
            race=self.race,
            pooled=self.pooled,
            init=seeds,
            exclude=[k for k in self.history if k != best_key],
        )
        self.inner.uid_prefix = self._inner_prefix(self.bracket)
        self.inner.attach_noise(self.noise)
        self.bracket += 1

    def _inner_prefix(self, bracket: int) -> str:
        # Inner trials are this strategy's: their uids carry its prefix, so
        # they never collide with another member's (a standalone "sh").
        return f"{self.uid_prefix}:b{bracket}"

    def ask(self, n: int = 1) -> list[Trial]:
        out: list[Trial] = []
        while len(out) < int(n) and not self._done:
            if self.inner is None or self.inner.done:
                self._new_bracket()
                if self._stale_bracket():
                    # nothing new left to sample: only known configs (the
                    # incumbent) would be re-raced, from the cache
                    self.inner._done = True
                    self._empty += 1
                    if self._empty > len(self.fidelities) + 1:
                        self._done = True
                    continue
            got = self.inner.ask(int(n) - len(out))
            if got:
                self._empty = 0
                for t in got:
                    self.pending[t.uid] = t.genome
                out.extend(got)
                continue
            if self.inner.done:
                self._empty += 1
                if self._empty > len(self.fidelities) + 1:
                    self._done = True
                continue
            break
        return out

    def _stale_bracket(self) -> bool:
        """Whether the new bracket holds only configs already measured at
        full fidelity (an exhausted space)."""
        if not self.inner._started:
            self.inner._start()
        return all(k in self.history for k in self.inner.members)

    def observe(self, ob: Observation, own: bool) -> None:
        if self.inner is not None:
            self.inner.tell([ob])

    def attach_noise(self, noise) -> None:
        super().attach_noise(noise)
        if self.inner is not None:
            self.inner.attach_noise(noise)

    @property
    def done(self) -> bool:
        return self._done

    def _state(self) -> dict:
        return {
            "bracket": self.bracket,
            "inner": None if self.inner is None else self.inner.state_dict(),
            "inner_shape": (
                None
                if self.inner is None
                else {
                    "fidelities": list(self.inner.fidelities),
                    "n_configs": self.inner.n_configs,
                    "seed": self.inner.seed,
                }
            ),
            "done": self._done,
            "empty": self._empty,
        }

    def _load(self, d: Mapping) -> None:
        self.bracket = int(d["bracket"])
        self._done = bool(d["done"])
        self._empty = int(d["empty"])
        if d["inner"] is None:
            self.inner = None
            return
        shape = d["inner_shape"]
        self.inner = SuccessiveHalving(
            self.space,
            seed=int(shape["seed"]),
            fidelities=shape["fidelities"],
            n_configs=int(shape["n_configs"]),
            eta=self.eta,
            race=self.race,
            pooled=self.pooled,
        )
        self.inner.load_state_dict(d["inner"])
        self.inner.uid_prefix = self._inner_prefix(self.bracket - 1)
        self.inner.attach_noise(self.noise)
