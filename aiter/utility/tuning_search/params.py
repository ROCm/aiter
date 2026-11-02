# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The search space every strategy works on: mixed parameters over knobs.

A :class:`SearchSpace` is a tuple of :class:`Param`. A point is a *genome*:
one level index per parameter. A level is a knob payload (sorted
``(name, value)`` pairs); decoding a genome merges the payloads of its active
parameters into one knob dict -- the config a backend compiles.

Parameters come in three kinds:

- ``ordinal``: numeric levels in increasing order. Distance is the level
  difference over the level count, so a tile of 128 is nearer 64 than 256.
- ``categorical``: unordered levels; any change has distance 1.
- ``bool``: two levels, off then on.

A parameter with a ``gate`` is read only while the gate parameter is off its
level 0 (the knob it gates is set). Its gene stays in the genome while the
gate is off -- strategies such as differential evolution keep it as latent
material -- but it contributes nothing to the decoded config or the key.

Validity is a predicate over decoded configs. :meth:`SearchSpace.repair` maps
an invalid genome to the nearest valid one (single moves first, then pairs,
then the nearest valid genome a strategy already holds), so a strategy never
spends a measurement on a config the space refuses.

Identity: :meth:`SearchSpace.key` is the canonical key of a genome, or
``None`` when it is invalid. Two genomes with the same key are one config and
are measured once.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import random
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from ._hash import stable_json_hash

__all__ = [
    "BOOL",
    "CATEGORICAL",
    "ORDINAL",
    "PARAM_KINDS",
    "Genome",
    "Param",
    "SearchSpace",
    "flatten_spec",
]

ORDINAL = "ordinal"
CATEGORICAL = "categorical"
BOOL = "bool"
PARAM_KINDS = (ORDINAL, CATEGORICAL, BOOL)

Genome = tuple[int, ...]
# One level of a parameter: sorted ``(knob, value)`` pairs, ``()`` meaning
# "leave the base value".
Knobs = tuple[tuple[str, object], ...]

# Bound on cached validity verdicts; a verdict may cost a build, so
# the cache matters, but an unbounded one would grow with a long search.
_CACHE_LIMIT = 200_000


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class Param:
    """One search dimension: ``levels`` are knob payloads, ``()`` meaning
    "leave the base value"."""

    name: str
    levels: tuple[Knobs, ...]
    kind: str = CATEGORICAL
    gate: str | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Param - name='', expected a non-empty name")
        if self.kind not in PARAM_KINDS:
            raise ValueError(
                f"Param {self.name!r}: kind={self.kind!r}, expected one of "
                f"{PARAM_KINDS}"
            )
        levels = tuple(tuple(tuple(p) for p in level) for level in self.levels)
        object.__setattr__(self, "levels", levels)
        if not levels:
            raise ValueError(f"Param {self.name!r} - levels=(), expected at least one")
        if len(set(levels)) != len(levels):
            raise ValueError(
                f"Param {self.name!r} - levels={levels!r}, expected no level to repeat"
            )
        if self.kind == BOOL and len(levels) != 2:
            raise ValueError(
                f"Param {self.name!r} - bool levels={len(levels)}, expected 2"
            )
        if self.gate == self.name:
            raise ValueError(
                f"Param {self.name!r} - gate={self.gate!r}, expected another param "
                "(it cannot gate itself)"
            )

    @classmethod
    def ordinal(cls, name: str, values: Sequence[object], *, gate=None) -> Param:
        """Numeric levels sorted ascending; each sets ``name``."""
        vals = list(dict.fromkeys(values))
        if not all(_is_number(v) for v in vals):
            raise ValueError(
                f"Param {name!r} - values={vals!r}, expected numeric ordinal values"
            )
        vals.sort()
        return cls(name, tuple(((name, v),) for v in vals), ORDINAL, gate)

    @classmethod
    def categorical(cls, name: str, values: Sequence[object], *, gate=None) -> Param:
        vals = list(dict.fromkeys(values))
        return cls(name, tuple(((name, v),) for v in vals), CATEGORICAL, gate)

    @classmethod
    def boolean(cls, name: str, *, gate=None) -> Param:
        """``name`` False, then True."""
        return cls(name, (((name, False),), ((name, True),)), BOOL, gate)

    @property
    def size(self) -> int:
        return len(self.levels)

    def label(self, level: int) -> str:
        payload = self.levels[level]
        if not payload:
            return "base"
        if len(payload) == 1:
            return repr(payload[0][1])
        return ",".join(f"{k}={v!r}" for k, v in payload)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "gate": self.gate,
            "levels": [[list(p) for p in level] for level in self.levels],
        }


class SearchSpace:
    """A mixed discrete space with validity and a canonical key.

    ``is_valid(config) -> bool`` refuses configs; ``canonical(config) -> str |
    None`` names a config (None refuses it) and replaces ``is_valid`` when
    given. ``fixed`` knobs are merged into every decoded config.
    """

    def __init__(
        self,
        params: Sequence[Param],
        *,
        is_valid: Callable[[dict], bool] | None = None,
        canonical: Callable[[dict], str | None] | None = None,
        fixed: Mapping[str, object] | None = None,
        name: str = "space",
        fingerprint_extra: object = "",
        prefix_valid: Callable[[dict], bool] | None = None,
        sampler: Callable[[random.Random], Mapping[str, object] | None] | None = None,
        nearest: Callable[[Genome, random.Random], Genome | None] | None = None,
        listed: Callable[[], list[Genome]] | None = None,
        nearest_first: bool = True,
    ):
        params = tuple(params)
        if not params:
            raise ValueError("SearchSpace - params=(), expected at least one Param")
        names = [p.name for p in params]
        if len(set(names)) != len(names):
            raise ValueError(
                f"SearchSpace - names={names!r}, expected no name to repeat"
            )
        index = {p.name: i for i, p in enumerate(params)}
        gates: list[int | None] = []
        for i, p in enumerate(params):
            if p.gate is None:
                gates.append(None)
                continue
            g = index.get(p.gate)
            if g is None or g >= i:
                raise ValueError(
                    f"Param {p.name!r} gate={p.gate!r}, expected an earlier param "
                    f"of {names!r}"
                )
            gates.append(g)
        self.params = params
        self.name = name
        self._index = index
        self._gates = tuple(gates)
        self._is_valid = is_valid
        self._canonical = canonical
        self._fixed = dict(fixed or {})
        self._fp_extra = fingerprint_extra
        self._keys: dict[Genome, str | None] = {}
        self._prefix_valid = prefix_valid
        self._prefix_cache: dict[str, bool] = {}
        self._sampler = sampler
        self._nearest = nearest
        self._nearest_first = bool(nearest_first)
        self._listed = listed
        self._base_levels = tuple(
            p.levels.index(()) if () in p.levels else 0 for p in params
        )
        self._members: dict[str, int] | None = None
        self.checks = 0

    # ------------------------------------------------------------ structure
    @property
    def members(self) -> dict[str, int] | None:
        """``{key: row index}`` of a :meth:`from_configs` space, else None."""
        return None if self._members is None else dict(self._members)

    @property
    def dims(self) -> int:
        return len(self.params)

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(p.size for p in self.params)

    def size(self) -> int:
        """Product of level counts: an upper bound on distinct configs."""
        return math.prod(self.sizes)

    def index(self, name: str) -> int:
        return self._index[name]

    def gate_of(self, i: int) -> int | None:
        return self._gates[i]

    def fingerprint(self) -> str:
        """Stable hash of the parameters; checkpoints and caches are refused
        when it changes."""
        return stable_json_hash(
            {
                "name": self.name,
                "params": [p.describe() for p in self.params],
                "fixed": sorted((k, repr(v)) for k, v in self._fixed.items()),
                "extra": repr(self._fp_extra),
            }
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "fingerprint": self.fingerprint(),
            "params": [p.describe() for p in self.params],
            "size_bound": self.size(),
        }

    # ------------------------------------------------------------- genomes
    def check(self, genome: Sequence[int]) -> Genome:
        """``genome`` as a tuple; raises ``ValueError`` when malformed."""
        g = tuple(int(x) for x in genome)
        if len(g) != self.dims:
            raise ValueError(
                f"genome length={len(g)}, expected {self.dims} for space {self.name!r}"
            )
        for i, (x, p) in enumerate(zip(g, self.params)):
            if not 0 <= x < p.size:
                raise ValueError(
                    f"genome[{i}] ({p.name})={x}, expected 0 <= level < {p.size}"
                )
        return g

    def active(self, genome: Sequence[int]) -> tuple[bool, ...]:
        act: list[bool] = []
        for i, g in enumerate(self._gates):
            act.append(g is None or (act[g] and genome[g] != 0))
        return tuple(act)

    def canonical_genome(self, genome: Sequence[int]) -> Genome:
        """``genome`` with inactive genes at level 0."""
        act = self.active(genome)
        return tuple(int(x) if a else 0 for x, a in zip(genome, act))

    def decode(self, genome: Sequence[int]) -> dict:
        """The knob dict of ``genome`` (fixed knobs included)."""
        config = dict(self._fixed)
        for p, x, a in zip(self.params, genome, self.active(genome)):
            if a:
                config.update(p.levels[x])
        return config

    def describe_genome(self, genome: Sequence[int]) -> dict:
        """``{param: level label}`` over active params, for traces."""
        return {
            p.name: p.label(x)
            for p, x, a in zip(self.params, genome, self.active(genome))
            if a
        }

    def encode(self, config: Mapping[str, object]) -> Genome:
        """The genome whose decode best matches ``config``.

        Each parameter takes the longest level whose every pair ``config``
        sets to the same value; with none, level 0. A warm start recorded as
        knobs (a dispatcher pick, a neighbor problem's winner) re-enters the
        space this way; repair it when the space refuses it.
        """
        genome = []
        for p in self.params:
            best, best_len = 0, -1
            for j, level in enumerate(p.levels):
                matches = all(k in config and _same(config[k], v) for k, v in level)
                if matches and len(level) > best_len:
                    best, best_len = j, len(level)
            genome.append(best)
        return tuple(genome)

    # ------------------------------------------------------------ identity
    def key(self, genome: Sequence[int]) -> str | None:
        """Canonical key of ``genome``; ``None`` when the space refuses it."""
        g = tuple(genome)
        if g in self._keys:
            return self._keys[g]
        cg = self.canonical_genome(g)
        if cg in self._keys:
            key = self._keys[cg]
        else:
            key = self._compute_key(self.decode(cg))
            self._remember(cg, key)
        self._remember(g, key)
        return key

    def _remember(self, genome: Genome, key: str | None) -> None:
        if len(self._keys) >= _CACHE_LIMIT:
            self._keys.clear()
        self._keys[genome] = key

    def _compute_key(self, config: dict) -> str | None:
        self.checks += 1
        if self._canonical is not None:
            return self._canonical(config)
        if self._is_valid is not None and not self._is_valid(config):
            return None
        return config_items_key(config)

    def valid(self, genome: Sequence[int]) -> bool:
        return self.key(genome) is not None

    # ------------------------------------------------------------ geometry
    def distance(self, a: Sequence[int], b: Sequence[int]) -> float:
        """Sum over parameters: ordinal level gap over (levels - 1),
        categorical/bool 0 or 1, 1 where only one side is active."""
        aa, ab = self.active(a), self.active(b)
        d = 0.0
        for p, x, y, ia, ib in zip(self.params, a, b, aa, ab):
            if ia and ib:
                if x != y:
                    d += abs(x - y) / (p.size - 1) if p.kind == ORDINAL else 1.0
            elif ia or ib:
                d += 1.0
        return d

    def moves(
        self,
        genome: Sequence[int],
        *,
        ordinal_radius: int | None = 1,
        rng: random.Random | None = None,
    ) -> list[Genome]:
        """Single-parameter moves from ``genome`` over its active params,
        nearest first. ``ordinal_radius`` bounds ordinal steps (None: any).

        Moves at equal distance are ordered by ``rng`` when given (repair
        must not prefer low parameter indices or low levels: on a space
        symmetric in two knobs that skews every repaired population), else
        by parameter index and level (a stable order for neighborhoods that
        are evaluated whole)."""
        g = tuple(genome)
        out: list[tuple[float, float, int, int, Genome]] = []
        for i, (p, a) in enumerate(zip(self.params, self.active(g))):
            if not a:
                continue
            for j in range(p.size):
                if j == g[i]:
                    continue
                if (
                    p.kind == ORDINAL
                    and ordinal_radius is not None
                    and abs(j - g[i]) > ordinal_radius
                ):
                    continue
                child = g[:i] + (j,) + g[i + 1 :]
                tie = rng.random() if rng is not None else 0.0
                out.append((round(self.distance(g, child), 9), tie, i, j, child))
        out.sort(key=lambda t: t[:4])
        return [t[4] for t in out]

    def random_genome(self, rng: random.Random) -> Genome:
        return tuple(rng.randrange(p.size) for p in self.params)

    def random_valid(self, rng: random.Random, tries: int = 64) -> Genome | None:
        """A random valid genome. With a ``sampler`` (a structural walk such
        as the dispatcher's pruned random walk) its draws come first; uniform
        genomes are the fallback (and the only source otherwise)."""
        if self._sampler is not None:
            for _ in range(int(tries)):
                config = self._sampler(rng)
                if config is None:
                    continue
                g = self.encode(config)
                if self.valid(g):
                    return g
        for _ in range(int(tries)):
            g = self.random_genome(rng)
            if self.valid(g):
                return g
        return None

    def stratified(
        self,
        rng: random.Random,
        n: int,
        tries: int = 8,
        *,
        repair_checks: int | None = None,
    ) -> list[Genome]:
        """``n`` genomes Latin-hypercube style: each parameter's levels are
        spread evenly over the draws, then shuffled per parameter. Invalid
        draws are repaired; the result has distinct keys (possibly fewer than
        ``n`` in a small space). ``repair_checks`` caps the validity checks of
        each repair (default: :meth:`repair`'s own cap); large candidate pools
        over a space where few genomes are valid pass a small cap, because a
        full repair per pool point there costs hundreds of checks."""
        n = int(n)
        if n <= 0:
            return []
        cols = []
        for p in self.params:
            col = [min(p.size - 1, (k * p.size) // n) for k in range(n)]
            rng.shuffle(col)
            cols.append(col)
        out: list[Genome] = []
        seen: set[str] = set()
        for k in range(n):
            g = tuple(col[k] for col in cols)
            if repair_checks is None:
                fixed = self.repair(g, rng, archive=out)
            else:
                # no archive fallback either: its nearest member is already in
                # the pool, so it would only be rejected as a duplicate
                fixed = self.repair(g, rng, max_checks=repair_checks)
            attempts = 0
            while (fixed is None or self.key(fixed) in seen) and attempts < tries:
                fixed = self.random_valid(rng)
                attempts += 1
            if fixed is None:
                continue
            key = self.key(fixed)
            if key in seen:
                continue
            seen.add(key)
            out.append(fixed)
        return out

    def repair(
        self,
        genome: Sequence[int],
        rng: random.Random,
        *,
        archive: Iterable[Sequence[int]] = (),
        max_checks: int = 256,
    ) -> Genome | None:
        """The nearest valid genome to ``genome``.

        Tries every single move (nearest first, any ordinal step), then
        pairs of the 24 nearest single moves, within ``max_checks`` validity
        checks; then the nearest valid genome in ``archive``; then a random
        valid genome. ``None`` only when none of those is valid. Candidates
        at equal distance are tried in an order drawn from ``rng``.
        """
        g = self.check(genome)
        if self.valid(g):
            return g
        if self._nearest is not None and self._nearest_first:
            # A space that knows its members (a sparse config list) finds the
            # exact nearest one directly; ties are broken by ``rng``.
            near = self._nearest(g, rng)
            if near is not None:
                return near
        checks = 0
        singles = self.moves(g, ordinal_radius=None, rng=rng)
        for child in singles:
            if checks >= max_checks:
                break
            checks += 1
            if self.valid(child):
                return child
        if self._prefix_valid is not None:
            fixed = self._prefix_repair(g)
            if fixed is not None and self.valid(fixed):
                return fixed
        if self._nearest is not None:
            near = self._nearest(g, rng)
            if near is not None:
                return near
        if checks < max_checks:
            near = singles[:24]
            pairs: list[tuple[float, float, Genome]] = []
            for a, b in itertools.combinations(near, 2):
                ia = next(i for i in range(self.dims) if a[i] != g[i])
                ib = next(i for i in range(self.dims) if b[i] != g[i])
                if ia == ib:
                    continue
                child = list(g)
                child[ia], child[ib] = a[ia], b[ib]
                child_t = tuple(child)
                pairs.append(
                    (round(self.distance(g, child_t), 9), rng.random(), child_t)
                )
            pairs.sort(key=lambda t: t[:2])
            for _d, _tie, child in pairs:
                if checks >= max_checks:
                    break
                checks += 1
                if self.valid(child):
                    return child
        best: list[Genome] = []
        best_d = math.inf
        for a in archive:
            a = tuple(a)
            if not self.valid(a):
                continue
            d = round(self.distance(g, a), 9)
            if d < best_d:
                best, best_d = [a], d
            elif d == best_d and a not in best:
                best.append(a)
        if best:
            return best[rng.randrange(len(best))] if len(best) > 1 else best[0]
        return self.random_valid(rng)

    def _prefix_ok(self, genome: Sequence[int]) -> bool:
        config = self.decode(genome)
        k = config_items_key(config)
        if k not in self._prefix_cache:
            if len(self._prefix_cache) >= _CACHE_LIMIT:
                self._prefix_cache.clear()
            self.checks += 1
            self._prefix_cache[k] = bool(self._prefix_valid(config))
        return self._prefix_cache[k]

    def _prefix_repair(self, genome: Genome) -> Genome | None:
        """Greedy repair along the parameter order, for spaces whose
        constraints only point backwards (a knob requires an earlier one, as
        in the dispatcher's axis order): each parameter keeps the target
        level when the prefix stays valid, else takes the nearest level that
        does; later parameters sit at their base level meanwhile."""
        cur = list(self._base_levels)
        for i, p in enumerate(self.params):
            if not self.active(cur)[i]:
                cur[i] = genome[i]
                continue
            target = genome[i]

            def order(j: int, target=target, p=p, base=self._base_levels[i]) -> tuple:
                gap = abs(j - target) if p.kind == ORDINAL else (j != target)
                return (j != target, gap, j != base, j)

            for j in sorted(range(p.size), key=order):
                cur[i] = j
                if self._prefix_ok(cur):
                    break
            else:
                return None
        return tuple(cur)

    def listed(self) -> list[Genome] | None:
        """Every valid genome, when the space knows them without enumerating
        its product (a config list); else ``None``."""
        return None if self._listed is None else list(self._listed())

    def enumerate(self, limit: int | None = None) -> Iterator[Genome]:
        """Every canonical genome (inactive genes at 0), valid or not, in
        lexicographic order; at most ``limit``."""
        count = 0

        def walk(i: int, prefix: list[int]):
            nonlocal count
            if limit is not None and count >= limit:
                return
            if i == self.dims:
                count += 1
                yield tuple(prefix)
                return
            g = self._gates[i]
            act = True
            if g is not None:
                act = self.active(prefix + [0] * (self.dims - len(prefix)))[g] and (
                    prefix[g] != 0
                )
            for j in range(self.params[i].size if act else 1):
                yield from walk(i + 1, prefix + [j])

        yield from walk(0, [])

    # -------------------------------------------------------- constructors
    @classmethod
    def from_configs(
        cls,
        configs: Sequence[Mapping[str, object]],
        *,
        name: str = "configs",
        member_index: bool = True,
        allow_duplicates: bool = False,
    ) -> SearchSpace:
        """A space whose valid points are exactly ``configs``.

        One parameter per field that varies (numeric with 3+ values ->
        ordinal, ``{False, True}`` -> bool, else categorical); constant fields
        are fixed. Strategies then move through field space and repair to the
        nearest listed config, which is how a flat config list (an Autotuner
        list, a recorded sweep table) gets a geometry.

        The list is known, so (``member_index``, the default) random valid
        points are drawn from it, :meth:`listed` returns every member, and
        repair falls back to the exact nearest member (random among ties)
        instead of pairs / archive / random draws. In a sparse list (members
        under a quarter of the field product, as in recorded conv-wgrad
        grids) the nearest member is tried first, because single moves mostly
        miss there; in a dense list single moves come first, because they
        are cheaper than a scan of every member and almost always succeed.
        ``member_index=False`` keeps the generic behaviour (an ablation for
        offline replay).

        Two configs with the same fields are one point: a ``ValueError``
        unless ``allow_duplicates`` (repeated measurements of one config in
        a recorded table), when the first row wins.
        """
        if not configs:
            raise ValueError("from_configs - configs=[], expected at least one config")
        fields = list(dict.fromkeys(k for c in configs for k in c))
        rows = [{f: _hashable(c.get(f)) for f in fields} for c in configs]
        params: list[Param] = []
        fixed: dict[str, object] = {}
        for f in fields:
            vals = list(dict.fromkeys(r[f] for r in rows))
            if len(vals) == 1:
                fixed[f] = rows[0][f]
                continue
            present = [v for v in vals if v is not None]
            if len(present) == len(vals) and all(_is_number(v) for v in vals):
                if len(vals) >= 3:
                    params.append(Param.ordinal(f, vals))
                    continue
                params.append(Param.categorical(f, sorted(vals)))
                continue
            if set(vals) == {False, True} and all(isinstance(v, bool) for v in vals):
                params.append(Param.boolean(f))
                continue
            params.append(Param.categorical(f, vals))
        if not params:
            # One config (or all identical): a single dummy dimension.
            params.append(Param("_config", ((),), CATEGORICAL))
        members: dict[str, int] = {}
        dups: list[tuple[int, int]] = []
        for i, r in enumerate(rows):
            k = config_items_key(r)
            if k in members:
                dups.append((members[k], i))
                continue
            members[k] = i
        if dups and not allow_duplicates:
            first, again = dups[0]
            raise ValueError(
                f"from_configs: configs[{again}] - fields equal configs[{first}] "
                f"({len(dups)} duplicate(s)), expected distinct configs (configs "
                "that flatten to the same fields cannot be told apart; add the "
                "field that distinguishes them)"
            )
        member_rows = [rows[i] for i in members.values()]
        index = _MemberIndex(params, member_rows) if member_index else None
        density = len(member_rows) / max(1, math.prod(p.size for p in params))
        space = cls(
            params,
            canonical=lambda c: (
                config_items_key(c) if config_items_key(c) in members else None
            ),
            fixed=fixed,
            name=name,
            fingerprint_extra=sorted(members),
            sampler=(
                (lambda rng: member_rows[rng.randrange(len(member_rows))])
                if member_index
                else None
            ),
            nearest=index.nearest if index is not None else None,
            listed=index.genomes if index is not None else None,
            nearest_first=density < 0.25,
        )
        space._members = members
        if index is not None:
            index.bind(space)
        return space


class _MemberIndex:
    """Nearest-member lookup for a config-list space (NumPy, vectorized).

    Distance matches :meth:`SearchSpace.distance` for ungated params:
    ordinal level gap over (levels - 1), categorical/bool 0 or 1.
    """

    def __init__(self, params: Sequence[Param], rows: Sequence[Mapping[str, object]]):
        self._params = tuple(params)
        self._rows = list(rows)
        self._space: SearchSpace | None = None
        self._G = None
        self._genomes: list[Genome] | None = None

    def bind(self, space: SearchSpace) -> None:
        import numpy as np

        self._space = space
        self._G = np.array(
            [space.encode(r) for r in self._rows], dtype=np.int32
        ).reshape(len(self._rows), space.dims)
        # Per parameter: its member column and a (levels x levels) table of
        # level distances, so a scan is one gather + add per parameter.
        self._cols = [np.ascontiguousarray(self._G[:, j]) for j in range(space.dims)]
        self._tables = []
        for p in self._params:
            lv = np.arange(p.size)
            if p.kind == ORDINAL and p.size > 1:
                t = np.abs(lv[:, None] - lv[None, :]) / (p.size - 1)
            else:
                t = (lv[:, None] != lv[None, :]).astype(float)
            self._tables.append(t)

    def genomes(self) -> list[Genome]:
        if self._genomes is None:
            self._genomes = [tuple(int(x) for x in row) for row in self._G]
        return self._genomes

    def nearest(self, genome: Genome, rng: random.Random) -> Genome | None:
        import numpy as np

        if self._G is None or not len(self._G):
            return None
        dist = np.zeros(len(self._G))
        for j, (col, table) in enumerate(zip(self._cols, self._tables)):
            dist += table[int(genome[j])][col]
        best = float(dist.min())
        ties = np.flatnonzero(dist <= best + 1e-12)
        pick = int(ties[rng.randrange(len(ties))]) if len(ties) > 1 else int(ties[0])
        return tuple(int(x) for x in self._G[pick])


def config_items_key(config: Mapping[str, object]) -> str:
    """Canonical JSON of a knob dict: the key of a space without its own."""
    return json.dumps(
        [[k, _jsonable(v)] for k, v in sorted(config.items())],
        separators=(",", ":"),
    )


def _jsonable(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    return repr(v)


def _hashable(v):
    try:
        hash(v)
        return v
    except TypeError:
        return repr(v)


def _same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return a == b


def flatten_spec(spec: object, prefix: str = "") -> dict:
    """A spec dataclass (nested dataclasses included) as a flat
    ``{dotted.field: scalar}`` dict; non-scalar leaves become their repr."""
    out: dict[str, object] = {}
    if dataclasses.is_dataclass(spec) and not isinstance(spec, type):
        for f in dataclasses.fields(spec):
            out.update(flatten_spec(getattr(spec, f.name), f"{prefix}{f.name}."))
        return out
    if isinstance(spec, Mapping):
        for k, v in spec.items():
            out.update(flatten_spec(v, f"{prefix}{k}."))
        return out
    name = prefix[:-1] if prefix else "value"
    out[name] = (
        spec
        if isinstance(spec, (bool, int, float, str)) or spec is None
        else repr(spec)
    )
    return out
