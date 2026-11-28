# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Default ranking: the structure used when no tuned row applies (§3.6 of the plan).

An `Entry` covers one class of problems, given by `match(...)`: set membership on named
fields. Its `Region`s split the class by `when(...)`: integer thresholds and set
membership on named scalars. `otherwise` is the part of the class no region covers.
Each region lists structures best first; the first feasible one wins.

The result never depends on the order of entries or regions: classes are disjoint, and so
are the regions of one entry. `check_table` verifies that statically, together with dead
regions, unknown structures and an always-feasible last structure in every list;
`check_problems` verifies, over a set of problems (the dispatch snapshot key grid in CI),
that every supported problem lands in exactly one entry and one region whose last
structure is feasible, and that every region is reached.

Each entry is a branch of the legacy `get_2stage_cfgs` default path, translated 1:1 with
its full condition.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

import itertools
from collections import Counter
from dataclasses import dataclass

import torch

from aiter import QuantType

from ..problem import FrozenMap
from ..segments.base import Segment
from .ops import Op

# Fields `match(...)` tests by set membership.
MATCH_FIELDS = {
    "arch": lambda p: p.arch,
    "quant_type": lambda p: p.quant_type,
    "act_dtype": lambda p: p.act_in.dtype,
    "w1_dtype": lambda p: p.w1.dtype,
    "w2_dtype": lambda p: p.w2.dtype,
    "out_dtype": lambda p: p.out_dtype,
    "activation": lambda p: p.activation.kind,
    "gate_mode": lambda p: p.layout.gate_mode,
    "shuffle": lambda p: p.layout.shuffle,
    "g1u1": lambda p: p.layout.g1u1,
    "shared": lambda p: None if p.shared is None else p.shared.placement,
}

# Integer scalars `when(...)` (and `match(...)`) can test.
SCALARS = {
    "m": lambda p: p.shape.m,
    "rows_per_expert": lambda p: p.rows_per_expert,
    "model_dim": lambda p: p.shape.model_dim,
    "inter_dim": lambda p: p.shape.inter_dim,
    "experts": lambda p: p.shape.experts,
    "topk": lambda p: p.shape.topk,
    "cu_num": lambda p: p.cu_num,
}

_GETTERS = {**MATCH_FIELDS, **SCALARS}
_OPS = {"le": "<=", "lt": "<", "ge": ">=", "gt": ">", "in": " in "}


def _as_set(name, value):
    values = (
        frozenset(value)
        if isinstance(value, (set, frozenset, list, tuple))
        else frozenset({value})
    )
    if not values:
        raise ValueError(f"empty value set for {name!r}")
    return values


def _show(value):
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, QuantType):
        return value.name
    return str(value)


def _show_set(values):
    shown = sorted(map(_show, values))
    return shown[0] if len(shown) == 1 else "{" + ", ".join(shown) + "}"


@dataclass(frozen=True)
class Match:
    """A problem class: every listed field takes one of its listed values."""

    sets: FrozenMap

    def holds(self, p) -> bool:
        return all(_GETTERS[name](p) in values for name, values in self.sets.items())

    def disjoint(self, other) -> bool:
        """True when some field both constrain has no common value."""
        return any(
            name in other.sets and not values & other.sets[name]
            for name, values in self.sets.items()
        )

    def __str__(self):
        return ", ".join(f"{k}={_show_set(v)}" for k, v in self.sets.items()) or "any"


def match(**fields) -> Match:
    unknown = sorted(set(fields) - set(MATCH_FIELDS) - set(SCALARS))
    if unknown:
        raise ValueError(
            f"unknown match fields {unknown}; known: {sorted(MATCH_FIELDS)}"
        )
    return Match(FrozenMap({k: _as_set(k, v) for k, v in fields.items()}))


@dataclass(frozen=True)
class Domain:
    """The integers `lo <= x < hi` (None: unbounded), restricted to `values` if set."""

    lo: int | None = None
    hi: int | None = None
    values: frozenset | None = None

    def _in_bounds(self, x) -> bool:
        return (self.lo is None or x >= self.lo) and (self.hi is None or x < self.hi)

    def __contains__(self, x) -> bool:
        return self._in_bounds(x) and (self.values is None or x in self.values)

    def __and__(self, other) -> "Domain":
        lo = max((b for b in (self.lo, other.lo) if b is not None), default=None)
        hi = min((b for b in (self.hi, other.hi) if b is not None), default=None)
        if self.values is None:
            values = other.values
        elif other.values is None:
            values = self.values
        else:
            values = self.values & other.values
        return Domain(lo, hi, values)

    @property
    def empty(self) -> bool:
        if self.lo is not None and self.hi is not None and self.lo >= self.hi:
            return True
        return self.values is not None and not any(map(self._in_bounds, self.values))


def _condition(op, threshold) -> Domain:
    if op == "in":
        values = _as_set("in", threshold)
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            raise TypeError(f"`in` takes integers, got {threshold!r}")
        return Domain(values=values)
    if not isinstance(threshold, int) or isinstance(threshold, bool):
        raise TypeError(f"thresholds are integers, got {threshold!r}")
    return {
        "le": Domain(hi=threshold + 1),
        "lt": Domain(hi=threshold),
        "ge": Domain(lo=threshold),
        "gt": Domain(lo=threshold + 1),
    }[op]


@dataclass(frozen=True)
class When:
    """Conditions on named scalars, all of which must hold."""

    domains: FrozenMap  # scalar name -> Domain
    text: str

    def holds(self, p) -> bool:
        return all(SCALARS[name](p) in d for name, d in self.domains.items())

    def disjoint(self, other) -> bool:
        return any(
            name in other.domains and (d & other.domains[name]).empty
            for name, d in self.domains.items()
        )

    @property
    def empty(self) -> bool:
        return any(d.empty for d in self.domains.values())

    def __str__(self):
        return self.text


def when(**conditions) -> When:
    """`when(rows_per_expert__le=8, model_dim__gt=2048, topk__in={6, 8})`."""
    if not conditions:
        raise ValueError("when() needs at least one condition; use `otherwise`")
    domains, shown = {}, []
    for key, threshold in conditions.items():
        name, sep, op = key.rpartition("__")
        if not sep or name not in SCALARS or op not in _OPS:
            raise ValueError(
                f"bad condition {key!r}: use <scalar>__<op> with a scalar in "
                f"{sorted(SCALARS)} and an op in {sorted(_OPS)}"
            )
        d = _condition(op, threshold)
        domains[name] = domains[name] & d if name in domains else d
        value = _show_set(_as_set(key, threshold)) if op == "in" else threshold
        shown.append(f"{name}{_OPS[op]}{value}")
    return When(FrozenMap(domains), ", ".join(shown))


def _structure(text) -> tuple:
    """`"sort | quant.a1 | ck.gemm1"` -> `("sort", "quant.a1", "ck.gemm1")`."""
    names = tuple(n.strip() for n in text.split("|"))
    if not all(names):
        raise ValueError(f"empty segment name in structure {text!r}")
    return names


def _structures(texts) -> tuple:
    if isinstance(texts, str):
        raise TypeError(f"a structure list, not a string: {texts!r}")
    return tuple(_structure(t) for t in texts)


@dataclass(frozen=True)
class Region:
    when: When
    prefer: tuple  # structures (segment-name tuples), best first

    def __post_init__(self):
        object.__setattr__(self, "prefer", _structures(self.prefer))


@dataclass(frozen=True)
class Entry:
    name: str
    match: Match
    regions: tuple = ()
    otherwise: tuple = ()  # structures for the rest of the class, best first

    def __post_init__(self):
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(self, "otherwise", _structures(self.otherwise))

    def lists(self):
        """`(label, structures)` for every region, then `otherwise` if it has any."""
        for r in self.regions:
            yield f"{self.name}[{r.when}]", r.prefer
        if self.otherwise:
            yield f"{self.name}[otherwise]", self.otherwise

    def region_for(self, p):
        """`(label, structures)` of the region `p` falls in; empty structures if none."""
        hits = [r for r in self.regions if r.when.holds(p)]
        if len(hits) > 1:
            raise RuntimeError(
                f"{self.name}: overlapping regions {[str(r.when) for r in hits]}"
            )
        if hits:
            return f"{self.name}[{hits[0].when}]", hits[0].prefer
        return f"{self.name}[otherwise]", self.otherwise


FP8 = frozenset({torch.float8_e4m3fn, torch.float8_e4m3fnuz})
CK_2STAGE = "sort | quant.a1 | ck.gemm1 | quant.inter | ck.gemm2"

RANKING = (
    # get_2stage_cfgs without a tuned row: fused_moe_1stage_dict has no per-tensor
    # entry (run_1stage stays False), and fp8 weights with an empty kernelName1 take
    # the CK branch (`not kernelName1 and q_dtype_w in [..., fp8]`).
    Entry(
        "fp8_per_tensor",
        match(quant_type=QuantType.per_Tensor, w1_dtype=FP8),
        otherwise=[CK_2STAGE],
    ),
)


def entry_for(p, table=RANKING):
    hits = [e for e in table if e.match.holds(p)]
    if len(hits) > 1:
        raise RuntimeError(f"overlapping ranking entries {[e.name for e in hits]}")
    return hits[0] if hits else None


def best(problem, plans, table=RANKING):
    """`(label, plan)`: the first feasible structure of `problem`'s region, or None.

    `plans` are the feasible segment tuples; `label` names the entry and region.
    """
    entry = entry_for(problem, table)
    if entry is None:
        return None
    label, prefer = entry.region_for(problem)
    by_names = {tuple(s.name for s in plan): plan for plan in plans}
    for names in prefer:
        if names in by_names:
            return label, by_names[names]
    return None


def rankable(plans):
    """The plans the ranking chooses among: whole-graph plans are tuned-only."""
    return [plan for plan in plans if len(plan) > 1]


def _shape_free(segment) -> bool:
    """Feasibility comes only from `Cap`s, which have no shape condition."""
    cls = type(segment)
    return cls.reject_reason is Segment.reject_reason and cls.accepts is Segment.accepts


def _structure_errors(label, names, segments):
    unknown = [n for n in names if n not in segments]
    if unknown:
        return [f"{label}: unknown segments {unknown} in {' | '.join(names)}"]
    chain = [segments[n] for n in names]
    errors = []
    if chain[0].covers[0] != Op.SORT or chain[-1].covers[1] != Op.REDUCE:
        errors.append(f"{label}: {' | '.join(names)} does not cover SORT..REDUCE")
    for a, b in itertools.pairwise(chain):
        if b.covers[0] != a.covers[1] + 1:
            errors.append(
                f"{label}: {a.name} and {b.name} are not adjacent in the chain"
            )
    return errors


def check_table(table, segments) -> list:
    """Static problems with `table` given the known `segments`; empty when it passes."""
    errors = []
    by_name = {}
    for s in segments:
        if s.name in by_name and type(by_name[s.name]) is not type(s):
            errors.append(f"two different segments are named {s.name!r}")
        by_name[s.name] = s
    for name, n in Counter(e.name for e in table).items():
        if n > 1:
            errors.append(f"{n} entries are named {name!r}")
    for a, b in itertools.combinations(table, 2):
        if not a.match.disjoint(b.match):
            errors.append(
                f"entries {a.name} and {b.name} overlap: no field separates "
                f"({a.match}) from ({b.match})"
            )
    for e in table:
        if not e.regions and not e.otherwise:
            errors.append(f"{e.name}: no regions and no otherwise")
        for r in e.regions:
            if r.when.empty:
                errors.append(
                    f"{e.name}[{r.when}]: dead region, its conditions contradict"
                )
        for a, b in itertools.combinations(e.regions, 2):
            if not a.when.disjoint(b.when):
                errors.append(f"{e.name}: regions [{a.when}] and [{b.when}] overlap")
        for label, prefer in e.lists():
            if not prefer:
                errors.append(f"{label}: empty structure list")
                continue
            for names, n in Counter(prefer).items():
                if n > 1:
                    errors.append(f"{label}: {' | '.join(names)} listed {n} times")
            for names in prefer:
                errors += _structure_errors(label, names, by_name)
            for n in prefer[-1]:
                if n in by_name and not _shape_free(by_name[n]):
                    errors.append(
                        f"{label}: the last structure is not always feasible: "
                        f"{n} decides feasibility beyond its capabilities"
                    )
    return errors


def check_problems(problems, segment_sets, table=RANKING):
    """`(errors, hits)` over `(label, problem)` pairs.

    A problem is supported by a segment set when it has a rankable feasible plan there; it
    must land in exactly one entry and one region (or `otherwise`) whose last structure is
    feasible. `hits` counts problems per region label, for `dead_lists`.
    """
    from .resolver import feasible_plans

    errors, hits = [], Counter()
    for label, p in problems:
        for segments in segment_sets:
            plans = rankable(feasible_plans(p, segments)[0])
            if not plans:
                continue
            entries = [e for e in table if e.match.holds(p)]
            if len(entries) != 1:
                names = [e.name for e in entries]
                errors.append(f"{label}: lands in {len(entries)} entries {names}")
                continue
            entry = entries[0]
            regions = [r for r in entry.regions if r.when.holds(p)]
            if len(regions) > 1:
                shown = [str(r.when) for r in regions]
                errors.append(f"{label}: lands in regions {shown} of {entry.name}")
                continue
            region, prefer = entry.region_for(p)
            if not prefer:
                errors.append(f"{label}: no region of {entry.name} and no otherwise")
                continue
            hits[region] += 1
            feasible = {tuple(s.name for s in plan) for plan in plans}
            if prefer[-1] not in feasible:
                errors.append(
                    f"{label}: the last structure of {region} is not feasible "
                    f"({' | '.join(prefer[-1])})"
                )
    return errors, hits


def dead_lists(hits, table=RANKING) -> list:
    """Regions (and non-empty `otherwise` lists) no supported problem reached."""
    return [
        f"{label}: dead, no supported problem lands here"
        for e in table
        for label, _ in e.lists()
        if not hits[label]
    ]
