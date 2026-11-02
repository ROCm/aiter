# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Mixed-variable differential evolution with self-adaptive F and CR.

Encoding: an individual holds one number per parameter. Ordinal parameters
are positions on the level axis, so the classic arithmetic operators apply
(``v = a + F * (b - c)``); the result is rounded *stochastically* (``2.3``
becomes 2 with probability 0.7, 3 with 0.3) so small differences still move
on axes with few levels. Categorical and bool parameters use the discrete
analogue: the mutant starts at the base vector's level and takes
a difference donor's level with probability ``min(F, 0.5)`` where its two
difference vectors disagree -- the first term that fires wins, so in
current-to-pbest the pull towards the elite is not overwritten by the
random term, and a large adapted ``F`` does not turn categorical genes into
uniform recombination -- plus a small uniform reset rate.

Variants: ``rand1bin`` (DE/rand/1/bin), ``current_to_best1bin`` and
``current_to_pbest1bin`` (JADE/SHADE's DE/current-to-pbest/1 with an
archive of replaced parents). Adaptation: ``none`` (fixed F, CR), ``jde``
(Brest et al.: per-individual F, CR resampled with probability 0.1 and kept
when the trial wins) or ``shade`` (Tanabe & Fukunaga: success-history
memories, Cauchy F and normal CR around a random memory slot, updated by the
improvement-weighted Lehmer / arithmetic means).

Gated parameters (sub-knobs read only while their gate is on) cross over as
a block with their gate whenever the gate's state differs between target and
mutant, so a trial never inherits a sub-knob setting that was tuned under
the other gate state; when both agree, sub-knobs cross gene by gene.

Every trial vector is repaired to the nearest valid config before it is
asked (:meth:`SearchSpace.repair`, with the population as the archive of
known-valid points; ties between equally near repairs are broken at random,
so repair does not drift the population towards low parameter indices),
and the repaired levels are written back into the trial (Lamarckian
repair). A trial whose config is already known is resolved from the
history without spending budget.

Selection is noise-aware: a trial replaces its parent when it is not worse
beyond noise (:meth:`Strategy.gain_threshold`), so a parent whose single
measurement was lucky does not block every comparable trial, and only a
trial better beyond noise counts as a success for the F/CR adaptation.
Parents are not re-measured: the evaluator answers a known config from its
cache. Use a confirmation round (``run_search(confirm=...)``) for the final
pick.

The strategy reports ``done`` once ``_MAX_EMPTY_GENERATIONS`` generations in
a row produced only known configs (the population has converged).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .params import ORDINAL, Genome, SearchSpace
from .strategy import Strategy, _f, _uf
from .trial import Observation, Trial

__all__ = ["DE_ADAPT", "DE_VARIANTS", "JDE", "DERand", "DifferentialEvolution"]

DE_VARIANTS = ("rand1bin", "current_to_best1bin", "current_to_pbest1bin")
DE_ADAPT = ("none", "jde", "shade")

_INIT, _EVOLVE = "init", "evolve"
# Generations built in one ask() whose every trial was already known before
# the strategy reports it has nothing new to propose.
_MAX_EMPTY_GENERATIONS = 50


class DifferentialEvolution(Strategy):
    """Differential evolution over a :class:`SearchSpace`; see the module
    docstring. ``init`` genomes seed the first population (warm starts);
    the rest is a stratified sample."""

    name = "de"

    def __init__(
        self,
        space: SearchSpace,
        seed: int = 0,
        *,
        pop_size: int | None = None,
        variant: str = "current_to_pbest1bin",
        adapt: str = "shade",
        F: float = 0.5,
        CR: float = 0.9,
        memory: int = 6,
        p_best: float = 0.2,
        reset_prob: float | None = None,
        init: Sequence[Sequence[int]] = (),
    ):
        super().__init__(space, seed)
        if variant not in DE_VARIANTS:
            raise ValueError(f"DE variant={variant!r}, expected one of {DE_VARIANTS}")
        if adapt not in DE_ADAPT:
            raise ValueError(f"DE adapt={adapt!r}, expected one of {DE_ADAPT}")
        if not 0.0 < F <= 2.0:
            raise ValueError(f"DE F={F}, expected 0 < F <= 2")
        if not 0.0 <= CR <= 1.0:
            raise ValueError(f"DE CR={CR}, expected 0 <= CR <= 1")
        if int(memory) < 1:
            raise ValueError(f"DE memory={memory}, expected >= 1")
        if not 0.0 < p_best <= 1.0:
            raise ValueError(f"DE p_best={p_best}, expected 0 < p_best <= 1")
        dims = space.dims
        if pop_size is None:
            pop_size = max(6, min(24, 2 + 2 * dims))
            listed = space.listed()
            n_valid = len(listed) if listed is not None else space.size()
            pop_size = min(pop_size, max(4, n_valid // 2))
        if int(pop_size) < 3:
            raise ValueError(f"DE pop_size={pop_size}, expected >= 3")
        self.pop_size = int(pop_size)
        self.variant = variant
        self.adapt = adapt
        self.F0 = float(F)
        self.CR0 = float(CR)
        self.H = int(memory)
        self.p_best = float(p_best)
        self.reset_prob = (
            float(reset_prob) if reset_prob is not None else 0.5 / max(1, dims)
        )
        self._seeds = [space.check(g) for g in init]
        self._blocks = self._gate_blocks()
        # Population.
        self.phase = _INIT
        self.pop: list[list[int]] = []
        self.fit: list[float] = []
        self.Fi: list[float] = []
        self.CRi: list[float] = []
        self.M_F = [0.5] * self.H
        self.M_CR = [0.5] * self.H
        self.k = 0
        self.archive: list[list[int]] = []
        self.generation = 0
        self._init_slots: list[dict] = []
        self._gen: list[dict] = []
        self._immigrants: list[tuple[str, list[int], float]] = []
        self._started = False
        self._done = False
        self.min_gain = 0.0

    # ----------------------------------------------------------- structure
    def _gate_blocks(self) -> dict[int, list[int]]:
        """``{gate index: [gated param indices, transitively]}``."""
        blocks: dict[int, list[int]] = {}
        for i in range(self.space.dims):
            g = self.space.gate_of(i)
            while g is not None:
                blocks.setdefault(g, []).append(i)
                g = self.space.gate_of(g)
        return blocks

    # --------------------------------------------------------------- init
    def _start(self) -> None:
        self._started = True
        chosen: list[Genome] = []
        keys: set[str] = set()
        for g in self._seeds:
            r = self.space.repair(g, self.rng, archive=chosen)
            if r is None:
                continue
            k = self.space.key(r)
            if k not in keys:
                keys.add(k)
                chosen.append(r)
        # Best known configs (warm starts told before the first ask).
        for k in sorted(self.history, key=lambda k: (self.history[k], k)):
            if len(chosen) >= (self.pop_size + 1) // 2:
                break
            if k not in keys and math.isfinite(self.history[k]):
                keys.add(k)
                chosen.append(self.genomes[k])
        need = self.pop_size - len(chosen)
        for g in self.space.stratified(self.rng, need + 2):
            if len(chosen) >= self.pop_size:
                break
            k = self.space.key(g)
            if k not in keys:
                keys.add(k)
                chosen.append(g)
        chosen = chosen[: self.pop_size]
        self._init_slots = [
            {"g": list(g), "key": self.space.key(g), "value": None, "uid": None}
            for g in chosen
        ]
        for slot in self._init_slots:
            if slot["key"] in self.history:
                slot["value"] = self.history[slot["key"]]
        self._maybe_finish_init()

    def _maybe_finish_init(self) -> None:
        if not self._init_slots or any(s["value"] is None for s in self._init_slots):
            return
        if self.phase == _INIT and not self.pop:
            self.pop = [list(s["g"]) for s in self._init_slots]
            self.fit = [float(s["value"]) for s in self._init_slots]
            self.Fi = [self.F0] * len(self.pop)
            self.CRi = [self.CR0] * len(self.pop)
        else:
            # A restart refilled every slot but the elite one.
            for s in self._init_slots:
                self.pop[s["slot"]] = list(s["g"])
                self.fit[s["slot"]] = float(s["value"])
        self._init_slots = []
        self.phase = _EVOLVE
        self._absorb_immigrants()

    # ---------------------------------------------------------- variation
    def _pick(self, n: int, exclude: set[int], pool: int) -> list[int]:
        idx = [i for i in range(pool) if i not in exclude]
        if len(idx) >= n:
            return self.rng.sample(idx, n)
        return [self.rng.randrange(pool) for _ in range(n)]

    def _params_for(self, i: int) -> tuple[float, float]:
        if self.adapt == "none":
            return self.F0, self.CR0
        if self.adapt == "jde":
            F = 0.1 + 0.9 * self.rng.random() if self.rng.random() < 0.1 else self.Fi[i]
            CR = self.rng.random() if self.rng.random() < 0.1 else self.CRi[i]
            return F, CR
        r = self.rng.randrange(self.H)
        CR = min(1.0, max(0.0, self.rng.gauss(self.M_CR[r], 0.1)))
        F = 0.0
        for _ in range(16):
            F = self.M_F[r] + 0.1 * math.tan(math.pi * (self.rng.random() - 0.5))
            if F > 0.0:
                break
        F = min(1.0, max(F, 0.05))
        return F, CR

    def _mutant(self, i: int, F: float) -> list[float]:
        n = len(self.pop)
        x = self.pop[i]
        order = sorted(range(n), key=lambda j: (self.fit[j], j))
        if self.variant == "rand1bin":
            r1, r2, r3 = self._pick(3, {i}, n)
            base, terms = self.pop[r1], [(self.pop[r2], self.pop[r3])]
        else:
            if self.variant == "current_to_best1bin":
                lead = self.pop[order[0]]
            else:
                top = max(2, round(self.p_best * n))
                lead = self.pop[self.rng.choice(order[:top])]
            (r1,) = self._pick(1, {i}, n)
            union = self.pop + self.archive
            (r2,) = self._pick(1, {i, r1}, len(union))
            base, terms = x, [(lead, x), (self.pop[r1], union[r2])]
        v: list[float] = []
        for d, p in enumerate(self.space.params):
            if p.kind == ORDINAL:
                val = float(base[d]) + sum(F * (a[d] - b[d]) for a, b in terms)
                hi = p.size - 1
                if val < 0:
                    val = 0.5 * x[d]
                elif val > hi:
                    val = 0.5 * (x[d] + hi)
                v.append(val)
                continue
            level = base[d]
            p_take = min(F, 0.5)
            for a, b in terms:
                if a[d] != b[d] and self.rng.random() < p_take:
                    level = a[d]
                    break
            if self.rng.random() < self.reset_prob:
                level = self.rng.randrange(p.size)
            v.append(float(level))
        return v

    def _round(self, v: list[float]) -> list[int]:
        out = []
        for val, p in zip(v, self.space.params):
            if p.kind == ORDINAL:
                lo = math.floor(val)
                level = lo + (1 if self.rng.random() < val - lo else 0)
            else:
                level = int(val)
            out.append(min(p.size - 1, max(0, level)))
        return out

    def _crossover(self, x: list[int], v: list[int], CR: float) -> list[int]:
        dims = len(x)
        u = list(x)
        j_rand = self.rng.randrange(dims)
        taken: set[int] = set()
        for d in range(dims):
            if d in taken:
                continue
            from_v = d == j_rand or self.rng.random() < CR
            gate = self.space.gate_of(d)
            if gate is not None and gate in taken:
                continue
            u[d] = v[d] if from_v else x[d]
            block = self._blocks.get(d)
            if block and (x[d] != 0) != (v[d] != 0):
                # Gate state differs: the gated knobs follow the gate's source.
                for s in block:
                    u[s] = v[s] if from_v else x[s]
                    taken.add(s)
        return u

    def _trial_vector(self, i: int) -> dict:
        F, CR = self._params_for(i)
        parent_key = self.space.key(self.pop[i])
        g: Genome | None = None
        for _attempt in range(4):
            v = self._round(self._mutant(i, F))
            u = self._crossover(self.pop[i], v, CR)
            g = self.space.repair(u, self.rng, archive=self.pop)
            if g is not None and self.space.key(g) != parent_key:
                break
        if g is None or self.space.key(g) == parent_key:
            moves = self.space.moves(self.pop[i])
            self.rng.shuffle(moves)
            g = next((m for m in moves if self.space.valid(m)), None) or tuple(
                self.pop[i]
            )
        key = self.space.key(g)
        entry = {
            "i": i,
            "g": list(g),
            "key": key,
            "F": F,
            "CR": CR,
            "value": None,
            "uid": None,
        }
        if key in self.history:
            entry["value"] = self.history[key]
        return entry

    # ------------------------------------------------------------ protocol
    def ask(self, n: int = 1) -> list[Trial]:
        if not self._started:
            self._start()
        out: list[Trial] = []
        empty = 0
        while len(out) < int(n):
            if self.phase == _INIT:
                slots = self._init_slots
            else:
                if not self._gen:
                    self._gen = [self._trial_vector(i) for i in range(len(self.pop))]
                    self._resolve_known()
                    if self._gen_complete():
                        empty += 1
                        self._select()
                        if empty >= _MAX_EMPTY_GENERATIONS:
                            self._done = not self.pending
                            break
                        continue
                slots = self._gen
            asked_keys = {
                s["key"] for s in slots if s["uid"] is not None and s["value"] is None
            }
            added = False
            for s in slots:
                if len(out) >= int(n):
                    break
                if (
                    s["value"] is not None
                    or s["uid"] is not None
                    or s["key"] in asked_keys
                ):
                    continue
                t = self.trial(s["g"])
                s["uid"] = t.uid
                asked_keys.add(s["key"])
                out.append(t)
                added = True
            if not added:
                break
        return out

    def observe(self, ob: Observation, own: bool) -> None:
        if not ob.full or ob.key is None:
            return
        value = ob.value if ob.ok else math.inf
        slots = self._init_slots if self.phase == _INIT else self._gen
        for s in slots:
            if s["value"] is None and s["key"] == ob.key:
                s["value"] = value
        if not own and ob.ok and self.pop:
            pop_keys = {self.space.key(g) for g in self.pop}
            worst = max(self.fit)
            if ob.key not in pop_keys and ob.value < worst:
                g = list(self.space.canonical_genome(ob.trial.genome))
                self._immigrants.append((ob.key, g, ob.value))
                self._immigrants.sort(key=lambda t: (t[2], t[0]))
                del self._immigrants[max(1, len(self.pop) // 4) :]
        if self.phase == _INIT:
            self._maybe_finish_init()
        elif self._gen and self._gen_complete():
            self._select()

    def _resolve_known(self) -> None:
        for s in self._gen:
            if s["value"] is None and s["key"] in self.history:
                s["value"] = self.history[s["key"]]

    def _gen_complete(self) -> bool:
        return all(s["value"] is not None for s in self._gen)

    # ----------------------------------------------------------- selection
    def _select(self) -> None:
        S_F: list[float] = []
        S_CR: list[float] = []
        W: list[float] = []
        gain = self._noise_gain()
        for s in self._gen:
            i = s["i"]
            trial_val, parent_val = float(s["value"]), self.fit[i]
            if trial_val <= parent_val * (1.0 + gain):
                if trial_val < parent_val * (1.0 - gain) or (
                    not math.isfinite(parent_val) and math.isfinite(trial_val)
                ):
                    S_F.append(s["F"])
                    S_CR.append(s["CR"])
                    if math.isfinite(parent_val) and trial_val > 0:
                        W.append(max(1e-12, math.log(parent_val / trial_val)))
                    else:
                        W.append(1.0)
                    if self.adapt == "jde":
                        self.Fi[i], self.CRi[i] = s["F"], s["CR"]
                    if self.variant == "current_to_pbest1bin":
                        self.archive.append(list(self.pop[i]))
                self.pop[i] = list(s["g"])
                self.fit[i] = trial_val
        while len(self.archive) > len(self.pop):
            self.archive.pop(self.rng.randrange(len(self.archive)))
        if self.adapt == "shade" and S_F:
            wsum = sum(W)
            w = [x / wsum for x in W]
            num = sum(wi * f * f for wi, f in zip(w, S_F))
            den = sum(wi * f for wi, f in zip(w, S_F))
            self.M_F[self.k] = num / den if den > 0 else self.M_F[self.k]
            self.M_CR[self.k] = sum(wi * c for wi, c in zip(w, S_CR))
            self.k = (self.k + 1) % self.H
        self._gen = []
        self.generation += 1
        self._absorb_immigrants()
        self._maybe_restart()

    def _absorb_immigrants(self) -> None:
        if not self.pop:
            return
        for key, g, value in self._immigrants:
            pop_keys = [self.space.key(x) for x in self.pop]
            if key in pop_keys:
                continue
            worst = max(range(len(self.pop)), key=lambda j: (self.fit[j], j))
            if value < self.fit[worst]:
                self.pop[worst] = list(g)
                self.fit[worst] = value
        self._immigrants = []

    def _maybe_restart(self) -> None:
        """A population collapsed onto one config restarts around it: the
        elite stays, the other slots are refilled by a stratified sample."""
        keys = {self.space.key(g) for g in self.pop}
        if len(keys) > 1:
            return
        elite = min(range(len(self.pop)), key=lambda j: (self.fit[j], j))
        fresh = [
            g
            for g in self.space.stratified(self.rng, len(self.pop) + 2)
            if self.space.key(g) not in keys
        ]
        slots = []
        for j, g in zip([j for j in range(len(self.pop)) if j != elite], fresh):
            k = self.space.key(g)
            slots.append(
                {
                    "slot": j,
                    "g": list(g),
                    "key": k,
                    "value": self.history.get(k),
                    "uid": None,
                }
            )
        if not slots:
            return
        self.phase = _INIT
        self._init_slots = slots
        self._maybe_finish_init()

    def _noise_gain(self) -> float:
        """The selection tolerance: 0 until the noise model has seen noise
        above its floor (a noise-free table must not let trials drift
        uphill), then twice the noise of one call's median."""
        n = self.noise
        if n is None or (n.call_sigma is None and n.rel_sigma <= n.floor):
            return 0.0
        return self.gain_threshold(self.min_gain, 1)

    @property
    def done(self) -> bool:
        return self._done

    # --------------------------------------------------------------- state
    def _state(self) -> dict:
        def slot(s: Mapping) -> dict:
            d = dict(s)
            if d.get("value") is not None:
                d["value"] = _f(float(d["value"]))
                d["resolved"] = True
            return d

        return {
            "phase": self.phase,
            "started": self._started,
            "pop": self.pop,
            "fit": [_f(x) for x in self.fit],
            "Fi": self.Fi,
            "CRi": self.CRi,
            "M_F": self.M_F,
            "M_CR": self.M_CR,
            "k": self.k,
            "archive": self.archive,
            "generation": self.generation,
            "init_slots": [slot(s) for s in self._init_slots],
            "gen": [slot(s) for s in self._gen],
            "immigrants": [[k, g, _f(v)] for k, g, v in self._immigrants],
            "done": self._done,
        }

    def _load(self, d: Mapping) -> None:
        def slot(s: Mapping) -> dict:
            out = dict(s)
            resolved = out.pop("resolved", False)
            out["value"] = _uf(out["value"]) if resolved else None
            return out

        self.phase = d["phase"]
        self._started = bool(d["started"])
        self.pop = [list(g) for g in d["pop"]]
        self.fit = [_uf(x) for x in d["fit"]]
        self.Fi = [float(x) for x in d["Fi"]]
        self.CRi = [float(x) for x in d["CRi"]]
        self.M_F = [float(x) for x in d["M_F"]]
        self.M_CR = [float(x) for x in d["M_CR"]]
        self.k = int(d["k"])
        self.archive = [list(g) for g in d["archive"]]
        self.generation = int(d["generation"])
        self._init_slots = [slot(s) for s in d["init_slots"]]
        self._gen = [slot(s) for s in d["gen"]]
        self._immigrants = [(k, list(g), _uf(v)) for k, g, v in d["immigrants"]]
        self._done = bool(d.get("done", False))


class JDE(DifferentialEvolution):
    """DE/rand/1/bin with jDE self-adaptation (Brest et al.)."""

    name = "jde"

    def __init__(self, space: SearchSpace, seed: int = 0, **kw):
        kw.setdefault("variant", "rand1bin")
        kw.setdefault("adapt", "jde")
        super().__init__(space, seed, **kw)


class DERand(DifferentialEvolution):
    """Classic DE/rand/1/bin with fixed F and CR."""

    name = "de_rand"

    def __init__(self, space: SearchSpace, seed: int = 0, **kw):
        kw.setdefault("variant", "rand1bin")
        kw.setdefault("adapt", "none")
        super().__init__(space, seed, **kw)
