# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Immutable profile loader. JSON: exact rows plus optional range rules.

Format:
  {"profile": str, "generated_by": {...},
   "rows":  [{"sig","arch","cu","world","m","vid","params":{...}}, ...],
   "rules": [{"sig","arch","cu","world","m_min","m_max","m_multiple","vid","params":{...},"name"?}, ...]}
Row key = (sig, arch, cu, world, m). Duplicate keys are rejected. Path comes from an
explicit argument or env AITER_GEMM_FUSION_PROFILE; a bare name (no path
separator, no .json) resolves to aiter/configs/gemm_fusion/<name>.json.

Range rules (P1): a rule covers every m with m_min <= m <= m_max and m % m_multiple == 0 for its (sig, arch, cu, world); it is consulted ONLY when no exact
row exists for that m, so an exact row always wins (a tuned cell overrides the rule that spans it). Matching is deterministic and rank-uniform by construction:
  * the profile is immutable and the lookup depends only on (sig, arch, cu, world, m): no rank, env, device or call-order input;
  * when several rules cover the same m the MOST SPECIFIC wins: the one with the largest m_multiple (it covers the sparsest set of m). That is what lets a broad
    safe rule (m_multiple 256) be refined by narrower ones (512, 1024) for the m values where a tuned variant is valid; it never depends on the order of the file;
  * two rules of the same (sig, arch, cu, world) with the SAME m_multiple must not overlap on any m (rejected when the profile is parsed): equal specificity has no
    deterministic winner, so it is an error, never a file-order accident.
The variant's own contract (e.g. M % 256 == 0, M <= 8192) is still enforced at match time; a rule whose m_multiple is not a multiple of the variant's is a profile
error reported at match time (ContractError), never silently narrowed.
"""

from __future__ import annotations

import json
import math
import os
import types
from dataclasses import dataclass

ENV_PROFILE = "AITER_GEMM_FUSION_PROFILE"
_REQ = ("sig", "arch", "cu", "world", "m", "vid")
_REQ_RULE = ("sig", "arch", "cu", "world", "m_min", "m_max", "m_multiple", "vid")


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class Row:
    sig: str
    arch: str
    cu: int
    world: int
    m: int
    vid: str
    params: types.MappingProxyType
    # R16 pad-M: the smallest REAL local m padded into this row as a bucket (None = the variant's / profile's default floor): the pad range of the bucket is [pad_min, m]; a real m in
    # (previous bucket, pad_min) is native. Lets a measurement tighten one bucket's range without code changes.
    pad_min: int | None = None


@dataclass(frozen=True)
class Rule:
    """A range rule (see the module doc). ``name`` is an optional label used in the dispatch log."""

    sig: str
    arch: str
    cu: int
    world: int
    m_min: int
    m_max: int
    m_multiple: int
    vid: str
    params: types.MappingProxyType
    name: str = ""

    def covers(self, m: int) -> bool:
        return self.m_min <= m <= self.m_max and m % self.m_multiple == 0

    def label(self) -> str:
        return self.name or f"m{self.m_min}-{self.m_max}/{self.m_multiple}"


def _rules_overlap(a: Rule, b: Rule) -> int | None:
    """The smallest m covered by BOTH rules (same key), or None."""
    lo, hi = max(a.m_min, b.m_min), min(a.m_max, b.m_max)
    if lo > hi:
        return None
    step = math.lcm(a.m_multiple, b.m_multiple)
    first = -(-lo // step) * step
    return first if first <= hi else None


class Profile:
    def __init__(
        self,
        name: str,
        rows: dict[tuple, Row],
        meta: dict,
        path: str,
        rules: tuple[Rule, ...] = (),
    ):
        self.name = name
        self._rows = types.MappingProxyType(dict(rows))
        self.meta = types.MappingProxyType(dict(meta))
        self.path = path
        self._rules = tuple(rules)
        by_key: dict[tuple, list[Rule]] = {}
        for r in self._rules:
            by_key.setdefault((r.sig, r.arch, r.cu, r.world), []).append(r)
        self._rules_by_key = types.MappingProxyType(
            {k: tuple(v) for k, v in by_key.items()}
        )
        rows_by_key: dict[tuple, list[Row]] = {}
        for key, row in self._rows.items():
            rows_by_key.setdefault(key[:4], []).append(row)
        self._rows_by_key = types.MappingProxyType(
            {k: tuple(sorted(v, key=lambda r: r.m)) for k, v in rows_by_key.items()}
        )  # R16 pad-M: the exact rows of one (sig, arch, cu, world) band, ascending in m (the candidate buckets)

    def lookup(self, sig, arch, cu, world, m) -> Row | None:
        """The EXACT row for (sig, arch, cu, world, m), or None (rules are not consulted: see resolve)."""
        return self._rows.get((sig, arch, int(cu), int(world), int(m)))

    def lookup_rule(self, sig, arch, cu, world, m) -> Rule | None:
        """The range rule covering m for (sig, arch, cu, world): the most specific one (largest m_multiple), or None."""
        hits = [
            r
            for r in self._rules_by_key.get((sig, arch, int(cu), int(world)), ())
            if r.covers(int(m))
        ]
        if not hits:
            return None
        top = max(r.m_multiple for r in hits)
        best = [r for r in hits if r.m_multiple == top]
        if (
            len(best) > 1
        ):  # unreachable for a parsed profile (equal-specificity overlap is rejected); guards a hand-built one
            raise ProfileError(
                f"{self.path}: m={m} covered by {len(best)} equally specific range rules ({[h.label() for h in best]})"
            )
        return best[0]

    def resolve(self, sig, arch, cu, world, m) -> tuple[Row | Rule | None, str]:
        """(entry, source): the exact row ("row"), else the covering range rule ("rule"), else (None, "none"). Exact rows always win."""
        row = self.lookup(sig, arch, cu, world, m)
        if row is not None:
            return row, "row"
        rule = self.lookup_rule(sig, arch, cu, world, m)
        if rule is not None:
            return rule, "rule"
        return None, "none"

    def bucket_rows(self, sig, arch, cu, world) -> tuple[Row, ...]:
        """The exact rows of (sig, arch, cu, world) ascending in m: the prepared BUCKETS a real local m can be padded into (see match.bucket_for_m)."""
        return self._rows_by_key.get((sig, arch, int(cu), int(world)), ())

    def rows(self):
        return tuple(self._rows.values())

    def rules(self) -> tuple[Rule, ...]:
        return self._rules

    def entries(self):
        """Every row and every rule (what a build enumeration has to cover)."""
        return self.rows() + self._rules

    def __len__(self):
        return len(self._rows)


def _config_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "configs", "gemm_fusion"
    )


def _resolve(path_or_name: str) -> str:
    if (
        os.path.sep in path_or_name
        or "/" in path_or_name
        or path_or_name.endswith(".json")
    ):
        return path_or_name
    return os.path.join(_config_dir(), path_or_name + ".json")


def parse_profile(doc: dict, path: str = "<memory>") -> Profile:
    if not isinstance(doc, dict) or "rows" not in doc:
        raise ProfileError(f"{path}: missing 'rows'")
    rows: dict[tuple, Row] = {}
    for i, r in enumerate(doc["rows"]):
        miss = [k for k in _REQ if k not in r]
        if miss:
            raise ProfileError(f"{path}: row {i} missing {miss}")
        for k in ("cu", "world", "m"):
            if isinstance(r[k], bool) or not isinstance(r[k], int):
                raise ProfileError(
                    f"{path}: row {i} {k} must be an int (exact rows only)"
                )
        key = (r["sig"], r["arch"], r["cu"], r["world"], r["m"])
        if key in rows:
            raise ProfileError(f"{path}: duplicate row key {key} (row {i})")
        pad_min = r.get("pad_min")
        if pad_min is not None and (
            isinstance(pad_min, bool)
            or not isinstance(pad_min, int)
            or not 1 <= pad_min <= r["m"]
        ):
            raise ProfileError(
                f"{path}: row {i} pad_min must be an int in 1..m ({r['m']}), got {pad_min!r}"
            )
        rows[key] = Row(
            *key,
            r["vid"],
            types.MappingProxyType(dict(r.get("params", {}))),
            pad_min,
        )
    rules = _parse_rules(doc.get("rules", []), path)
    meta = {k: v for k, v in doc.items() if k not in ("rows", "rules")}
    mp = meta.get(
        "m_pad_min_large"
    )  # R16: the smallest real local m a LARGE plan is padded into a bucket for (below it: exact rows only, i.e. native between the small and large paths)
    if mp is not None and (isinstance(mp, bool) or not isinstance(mp, int) or mp < 1):
        raise ProfileError(
            f"{path}: m_pad_min_large must be a positive int, got {mp!r}"
        )
    return Profile(str(doc.get("profile", "")), rows, meta, path, rules)


def _parse_rules(raw, path: str) -> tuple[Rule, ...]:
    if not isinstance(raw, list):
        raise ProfileError(f"{path}: 'rules' must be a list")
    rules: list[Rule] = []
    for i, r in enumerate(raw):
        if not isinstance(r, dict):
            raise ProfileError(f"{path}: rule {i} must be an object")
        miss = [k for k in _REQ_RULE if k not in r]
        if miss:
            raise ProfileError(f"{path}: rule {i} missing {miss}")
        for k in ("cu", "world", "m_min", "m_max", "m_multiple"):
            if isinstance(r[k], bool) or not isinstance(r[k], int):
                raise ProfileError(f"{path}: rule {i} {k} must be an int")
        if r["m_multiple"] < 1 or r["m_min"] < 1 or r["m_max"] < r["m_min"]:
            raise ProfileError(
                f"{path}: rule {i} needs 1 <= m_min <= m_max and m_multiple >= 1 (got {r['m_min']}..{r['m_max']} step {r['m_multiple']})"
            )
        first = -(-r["m_min"] // r["m_multiple"]) * r["m_multiple"]
        if first > r["m_max"]:
            raise ProfileError(
                f"{path}: rule {i} covers no m ({r['m_min']}..{r['m_max']} step {r['m_multiple']})"
            )
        params = r.get("params", {})
        if not isinstance(params, dict):
            raise ProfileError(f"{path}: rule {i} params must be an object")
        rules.append(
            Rule(
                r["sig"],
                r["arch"],
                r["cu"],
                r["world"],
                r["m_min"],
                r["m_max"],
                r["m_multiple"],
                r["vid"],
                types.MappingProxyType(dict(params)),
                str(r.get("name", "")),
            )
        )
    for i, a in enumerate(rules):
        for j in range(i):
            b = rules[j]
            if (a.sig, a.arch, a.cu, a.world) == (
                b.sig,
                b.arch,
                b.cu,
                b.world,
            ) and a.m_multiple == b.m_multiple:
                m = _rules_overlap(a, b)
                if m is not None:
                    raise ProfileError(
                        f"{path}: range rules {j} ({b.label()}) and {i} ({a.label()}) are equally specific (m_multiple {a.m_multiple}) and both cover m={m}"
                    )
    return tuple(rules)


_CACHE: dict[str, Profile] = {}


def window_budget_of(profile: Profile) -> dict:
    """The profile's uniform-window budget ({window_bytes, max_windows, window_sizes, ...}); {} when the profile has none."""
    return dict(profile.meta.get("window_budget") or {})


def resolve_profile(profile) -> Profile:
    """Explicit-only resolution: a Profile object is returned as is; a str is a profile name/path (NO environment lookup)."""
    if isinstance(profile, Profile):
        return profile
    if isinstance(profile, str) and profile:
        return load_profile(profile)
    raise ProfileError(
        f"resolve_profile needs a Profile or a profile name/path, got {profile!r}"
    )


def load_profile(path: str | None = None) -> Profile:
    """Load (and cache) a profile. ``path=None`` is the documented top-level CONVENIENCE: it falls back to the env var
    AITER_GEMM_FUSION_PROFILE. Runtime components (FusionResources, plans) must never rely on that fallback: they take the profile explicitly.
    """
    p = path or os.environ.get(ENV_PROFILE)
    if not p:
        raise ProfileError(f"no profile given and {ENV_PROFILE} is unset")
    full = os.path.abspath(_resolve(p))
    prof = _CACHE.get(full)
    if prof is None:
        try:
            with open(full, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except OSError as e:
            raise ProfileError(f"cannot read profile {full}: {e}") from e
        prof = parse_profile(doc, full)
        _CACHE[full] = prof
    return prof
