# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Immutable profile loader. JSON, exact rows only (first release).

Format:
  {"profile": str, "generated_by": {...},
   "rows": [{"sig","arch","cu","world","m","vid","params":{...}}, ...]}
Key = (sig, arch, cu, world, m). Duplicate keys are rejected. Path comes from an
explicit argument or env AITER_GEMM_FUSION_PROFILE; a bare name (no path
separator, no .json) resolves to aiter/configs/gemm_fusion/<name>.json.
"""

from __future__ import annotations

import json
import os
import types
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

ENV_PROFILE = "AITER_GEMM_FUSION_PROFILE"
_REQ = ("sig", "arch", "cu", "world", "m", "vid")


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


class Profile:
    def __init__(self, name: str, rows: Dict[Tuple, Row], meta: dict, path: str):
        self.name = name
        self._rows = types.MappingProxyType(dict(rows))
        self.meta = types.MappingProxyType(dict(meta))
        self.path = path

    def lookup(self, sig, arch, cu, world, m) -> Optional[Row]:
        return self._rows.get((sig, arch, int(cu), int(world), int(m)))

    def rows(self):
        return tuple(self._rows.values())

    def __len__(self):
        return len(self._rows)


def _config_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "gemm_fusion")


def _resolve(path_or_name: str) -> str:
    if os.path.sep in path_or_name or "/" in path_or_name or path_or_name.endswith(".json"):
        return path_or_name
    return os.path.join(_config_dir(), path_or_name + ".json")


def parse_profile(doc: dict, path: str = "<memory>") -> Profile:
    if not isinstance(doc, dict) or "rows" not in doc:
        raise ProfileError(f"{path}: missing 'rows'")
    rows: Dict[Tuple, Row] = {}
    for i, r in enumerate(doc["rows"]):
        miss = [k for k in _REQ if k not in r]
        if miss:
            raise ProfileError(f"{path}: row {i} missing {miss}")
        for k in ("cu", "world", "m"):
            if isinstance(r[k], bool) or not isinstance(r[k], int):
                raise ProfileError(f"{path}: row {i} {k} must be an int (exact rows only)")
        key = (r["sig"], r["arch"], r["cu"], r["world"], r["m"])
        if key in rows:
            raise ProfileError(f"{path}: duplicate row key {key} (row {i})")
        rows[key] = Row(*key, r["vid"], types.MappingProxyType(dict(r.get("params", {}))))
    meta = {k: v for k, v in doc.items() if k != "rows"}
    return Profile(str(doc.get("profile", "")), rows, meta, path)


_CACHE: Dict[str, Profile] = {}


def window_budget_of(profile: "Profile") -> dict:
    """The profile's uniform-window budget ({window_bytes, max_windows, window_sizes, ...}); {} when the profile has none."""
    return dict(profile.meta.get("window_budget") or {})


def resolve_profile(profile) -> "Profile":
    """Explicit-only resolution: a Profile object is returned as is; a str is a profile name/path (NO environment lookup)."""
    if isinstance(profile, Profile):
        return profile
    if isinstance(profile, str) and profile:
        return load_profile(profile)
    raise ProfileError(f"resolve_profile needs a Profile or a profile name/path, got {profile!r}")


def load_profile(path: Optional[str] = None) -> Profile:
    """Load (and cache) a profile. ``path=None`` is the documented top-level CONVENIENCE: it falls back to the env var
    AITER_GEMM_FUSION_PROFILE. Runtime components (FusionResources, plans) must never rely on that fallback: they take the profile explicitly."""
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


