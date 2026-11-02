# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Local search: hill climbing with lookback and knob-combination re-sweeps.

From a center config the climber runs three stages, each a batch of
neighbors:

0. every single-knob move (ordinal +-1, every other categorical level, gate
   flips), the center itself first when it is new;
1. a *combination re-sweep*: the single moves that came within
   ``combo_tol`` of the center (or beat it) -- the best level per knob, at
   most ``combo_knobs`` knobs -- are combined exhaustively (every subset of
   two or more), because knob effects interact and a pair of individually
   neutral moves often wins;
2. ordinal moves of exactly two steps (escape small plateaus).

The first neighbor that beats the center by more than ``min_gain``
(relative) becomes the new center (first improvement), and the old center
goes on the lookback stack. When the center's stages are exhausted the
climber looks back: it reopens the last ``lookback`` centers whose stages a
move interrupted, then restarts from the best *foreign* config that was
never a center (a warm start, another strategy's find), then from a random
unseen one. Its own rejected neighbors are never restart points: they are
worse than a local optimum already climbed, and restarting from them kept a
climber in a poor basin for hundreds of evaluations in offline replay.

``init`` genomes (warm starts beyond ``start``) are measured first and count
as foreign finds. Without a start, ``n_init`` stratified configs are
measured first and the climb starts at the best of them (a single random
start in a space whose times span orders of magnitude can spend the whole
budget walking out of a slow region).

Any observation better than the best seen -- including other strategies'
in a portfolio -- moves the center there.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence

from .params import ORDINAL, Genome, SearchSpace
from .strategy import Strategy
from .trial import Observation, Trial

__all__ = ["HillClimb"]

_STAGES = 3


class HillClimb(Strategy):
    name = "hill"

    def __init__(
        self,
        space: SearchSpace,
        seed: int = 0,
        *,
        start: Sequence[int] | None = None,
        lookback: int = 3,
        combo_knobs: int = 4,
        combo_tol: float = 0.01,
        min_gain: float = 0.002,
        init: Sequence[Sequence[int]] = (),
        n_init: int = 8,
    ):
        super().__init__(space, seed)
        if int(n_init) < 0:
            raise ValueError(f"HillClimb n_init={n_init}, expected >= 0")
        if int(lookback) < 0:
            raise ValueError(f"HillClimb lookback={lookback}, expected >= 0")
        if int(combo_knobs) < 0:
            raise ValueError(f"HillClimb combo_knobs={combo_knobs}, expected >= 0")
        if combo_tol < 0 or min_gain < 0:
            raise ValueError(
                f"HillClimb - combo_tol={combo_tol}, min_gain={min_gain}, "
                "expected both >= 0"
            )
        seeds = [list(space.check(g)) for g in init]
        if start is None and seeds:
            start = seeds.pop(0)
        self.start = None if start is None else space.check(start)
        self.seed_queue: list[list[int]] = seeds
        self.foreign: set[str] = set()
        self.n_init = int(n_init)
        self.probe_queue: list[list[int]] | None = None
        self._probed = self.start is not None or self.n_init == 0
        self.lookback = int(lookback)
        self.combo_knobs = int(combo_knobs)
        self.combo_tol = float(combo_tol)
        self.min_gain = float(min_gain)
        self.center: str | None = None
        self.center_genome: list[int] | None = None
        self.stage_done: dict[str, int] = {}
        self.stage: list | None = None  # [center key, stage index, [keys]]
        self.queue: list[list[int]] = []
        self.stack: list[str] = []
        self.stack_genomes: dict[str, list[int]] = {}
        self.restarts = 0
        self.moves_made = 0
        self._done = False

    # -------------------------------------------------------------- stages
    def _stage_genomes(self, c: list[int], s: int) -> list[Genome]:
        cg = tuple(c)
        if s == 0:
            return [cg] + self.space.moves(cg, ordinal_radius=1)
        if s == 1:
            return self._combos(cg)
        out = []
        for m in self.space.moves(cg, ordinal_radius=2):
            i = next(i for i in range(len(cg)) if m[i] != cg[i])
            if self.space.params[i].kind == ORDINAL and abs(m[i] - cg[i]) == 2:
                out.append(m)
        return out

    def _combos(self, cg: Genome) -> list[Genome]:
        ckey = self.space.key(cg)
        cval = self.history.get(ckey, math.inf)
        if not math.isfinite(cval) or self.combo_knobs < 2:
            return []
        best_per_param: dict[int, tuple[float, int]] = {}
        for m in self.space.moves(cg, ordinal_radius=1):
            k = self.space.key(m)
            v = self.history.get(k) if k is not None else None
            if v is None or not math.isfinite(v) or v > cval * (1.0 + self.combo_tol):
                continue
            i = next(i for i in range(len(cg)) if m[i] != cg[i])
            if i not in best_per_param or (v, m[i]) < best_per_param[i]:
                best_per_param[i] = (v, m[i])
        top = sorted(best_per_param.items(), key=lambda t: (t[1][0], t[0]))
        top = top[: self.combo_knobs]
        out: list[Genome] = []
        for r in range(2, len(top) + 1):
            for subset in itertools.combinations(top, r):
                g = list(cg)
                for i, (_v, level) in subset:
                    g[i] = level
                if self.space.valid(g):
                    out.append(tuple(g))
        return out

    def _choose_center(self) -> bool:
        """Point ``center`` at the next config with stages left."""
        if self.center is not None and self.stage_done.get(self.center, 0) < _STAGES:
            return True
        for k in reversed(self.stack[-self.lookback :] if self.lookback else []):
            if self.stage_done.get(k, 0) < _STAGES:
                self._set_center(k, self.stack_genomes[k], push=False)
                return True
        fresh = sorted(
            (v, k)
            for k, v in self.history.items()
            if math.isfinite(v) and k not in self.stage_done and k in self.foreign
        )
        if fresh:
            k = fresh[0][1]
            self.restarts += 1
            self._set_center(k, list(self.genomes[k]), push=False)
            return True
        g = self._random_unseen()
        if g is None:
            return False
        self.restarts += 1
        self._set_center(self.space.key(g), list(g), push=False)
        return True

    def _random_unseen(self) -> Genome | None:
        taken = self.pending_keys()
        for _ in range(256):
            g = self.space.random_valid(self.rng, tries=8)
            if g is None:
                continue
            k = self.space.key(g)
            if k not in self.history and k not in taken:
                return g
        return None

    def _set_center(self, key: str, genome: list[int], *, push: bool) -> None:
        if push and self.center is not None and self.center != key:
            self.stack.append(self.center)
            self.stack_genomes[self.center] = list(self.center_genome)
            del self.stack[:-32]
            self.stack_genomes = {k: self.stack_genomes[k] for k in self.stack}
        self.center = key
        self.center_genome = list(self.space.canonical_genome(genome))
        self.stage = None
        self.queue = []

    def _next_stage(self) -> bool:
        while True:
            if not self._choose_center():
                return False
            c = self.center
            s = self.stage_done.get(c, 0)
            if s >= _STAGES:
                continue
            gs = self._stage_genomes(self.center_genome, s)
            keys = []
            todo = []
            for g in gs:
                k = self.space.key(g)
                if k is None or k in keys:
                    continue
                keys.append(k)
                if k not in self.history:
                    todo.append(list(g))
            self.stage = [c, s, keys]
            self.stage_done.setdefault(c, 0)
            if todo:
                self.queue = todo
                return True
            self.stage_done[c] = s + 1
            self.stage = None

    def _stage_complete(self) -> bool:
        return self.stage is not None and all(k in self.history for k in self.stage[2])

    # ------------------------------------------------------------ protocol
    def ask(self, n: int = 1) -> list[Trial]:
        out: list[Trial] = []
        if self._done:
            return out
        if not self._probed:
            # Measure the stratified probe first; climb from its best.
            if self.probe_queue is None:
                self.probe_queue = [
                    list(g) for g in self.space.stratified(self.rng, self.n_init)
                ]
                self.n_init = len(self.probe_queue)
            taken = self.pending_keys()
            while self.probe_queue and len(out) < int(n):
                g = self.probe_queue.pop(0)
                k = self.space.key(g)
                if k is None or k in self.history or k in taken:
                    continue
                self.foreign.add(k)
                taken.add(k)
                out.append(self.trial(g))
            if out or self.pending:
                return out
            self._probed = True
        if self.center is None:
            start = self.start
            if start is None:
                key, _v = self.best()
                start = self.genomes[key] if key is not None else None
            if start is None:
                start = self.space.random_valid(self.rng)
            if start is not None:
                start = self.space.repair(start, self.rng)
            if start is None:
                self._done = True
                return out
            self._set_center(self.space.key(start), list(start), push=False)
        taken = self.pending_keys()
        # Warm starts beyond ``start`` follow the start's own measurement.
        while self.seed_queue and len(out) < int(n) and self.center in self.history:
            g = self.space.repair(self.seed_queue.pop(0), self.rng)
            k = None if g is None else self.space.key(g)
            if k is None or k in self.history or k in taken:
                continue
            self.foreign.add(k)
            taken.add(k)
            out.append(self.trial(g))
        while len(out) < int(n):
            if self.queue:
                g = self.queue.pop(0)
                k = self.space.key(g)
                if k in self.history or k in taken:
                    continue
                taken.add(k)
                out.append(self.trial(g))
                continue
            if self.stage is not None and not self._stage_complete():
                break  # wait for this stage's results
            if self.stage is not None:
                c, s, _keys = self.stage
                self.stage_done[c] = max(self.stage_done.get(c, 0), s + 1)
                self.stage = None
            if not self._next_stage():
                if not self.pending:
                    self._done = True
                break
        return out

    def observe(self, ob: Observation, own: bool) -> None:
        if not ob.full or not ob.ok or ob.key is None:
            return
        if not own:
            self.foreign.add(ob.key)
        if self.center is None:
            return
        cval = self.history.get(self.center, math.inf)
        best_key, _best_val = self.best()
        in_stage = self.stage is not None and ob.key in self.stage[2]
        gain = self.gain_threshold(self.min_gain, len(ob.samples))
        beats_center = ob.value < cval * (1.0 - gain)
        if ob.key == self.center:
            return
        if (in_stage and beats_center) or (best_key == ob.key and beats_center):
            self.moves_made += 1
            self._set_center(
                ob.key, list(self.space.canonical_genome(ob.trial.genome)), push=True
            )

    @property
    def done(self) -> bool:
        return self._done

    # --------------------------------------------------------------- state
    def _state(self) -> dict:
        return {
            "center": self.center,
            "center_genome": self.center_genome,
            "stage_done": self.stage_done,
            "stage": self.stage,
            "queue": self.queue,
            "stack": self.stack,
            "stack_genomes": self.stack_genomes,
            "restarts": self.restarts,
            "moves_made": self.moves_made,
            "done": self._done,
            "foreign": sorted(self.foreign),
            "seed_queue": self.seed_queue,
            "n_init": self.n_init,
            "probe_queue": self.probe_queue,
            "probed": self._probed,
        }

    def _load(self, d: Mapping) -> None:
        self.center = d["center"]
        self.center_genome = (
            None if d["center_genome"] is None else list(d["center_genome"])
        )
        self.stage_done = {k: int(v) for k, v in d["stage_done"].items()}
        self.stage = (
            None
            if d["stage"] is None
            else [d["stage"][0], int(d["stage"][1]), list(d["stage"][2])]
        )
        self.queue = [list(g) for g in d["queue"]]
        self.stack = list(d["stack"])
        self.stack_genomes = {k: list(g) for k, g in d["stack_genomes"].items()}
        self.restarts = int(d["restarts"])
        self.moves_made = int(d["moves_made"])
        self._done = bool(d["done"])
        self.foreign = set(d.get("foreign", ()))
        self.seed_queue = [list(g) for g in d.get("seed_queue", ())]
        self.n_init = int(d.get("n_init", 0))
        pq = d.get("probe_queue")
        self.probe_queue = None if pq is None else [list(g) for g in pq]
        self._probed = bool(d.get("probed", True))
