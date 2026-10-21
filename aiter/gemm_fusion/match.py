# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Longest-prefix matching of a FusionSpec against an immutable profile.

Semantics (Artemis §3.4): no profile row (or range rule) for any prefix -> (None, 0) = native.
Row (or rule) found but its variant contract is not satisfied -> ContractError.

Per prefix the entry is the EXACT row, else the range rule covering m (Profile.resolve); the longest prefix with an entry wins, whatever its source. The result
depends only on (spec, arch, cu, world, m, available capabilities, graph_mode) and the immutable profile, so every rank resolves identically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from . import catalog
from .profile import Profile, load_profile
from .spec import FusionSpec

log = logging.getLogger("aiter.gemm_fusion")


class ContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class MatchCtx:
    arch: str
    cu: int
    world: int
    m: int  # local M (tokens per rank, padded)
    available: frozenset[str] = frozenset()  # {"mori_cco","sdma","gfx950_sc_bypass"}
    graph_mode: str = "eager"  # "eager" | "capture"


def _check_contract(info, sub: FusionSpec, ctx: MatchCtx, row):
    c = info.contract
    bad = []
    m_multiple = getattr(row, "m_multiple", None)  # a range rule
    if m_multiple is not None and m_multiple % c.m_multiple:
        bad.append(
            f"range rule {row.label()} m_multiple {m_multiple} is not a multiple of the variant's {c.m_multiple}"
        )
    if ctx.arch not in c.arch:
        bad.append(f"arch {ctx.arch}")
    if ctx.world != c.world:
        bad.append(f"world {ctx.world}")
    if ctx.m % c.m_multiple:
        bad.append(f"m {ctx.m} % {c.m_multiple}")
    missing = set(c.requires) - set(ctx.available)
    if missing:
        bad.append(f"missing {sorted(missing)}")
    if tuple(s.kind for s in sub.stages) != info.stage_kinds:
        bad.append("stage kinds")
    if bad:
        raise ContractError(
            f"gemm_fusion {row.vid} covers sig {sub.signature()} but contract fails: {bad}"
        )


_BUILTIN = {"sp_qkv.": "aiter.ops.gemm_fusion.sp_qkv.variants"}


def _load_builtin(vid: str):
    """Import the module that registers a built-in variant family (lazy; only when a profile row names it)."""
    import importlib

    for prefix, mod in _BUILTIN.items():
        if vid.startswith(prefix):
            try:
                importlib.import_module(mod)
            except (
                ImportError
            ) as e:  # surfaces as an unregistered variant (ContractError)
                log.warning("event=variant_load_failed vid=%s err=%s", vid, e)
            return


def match_prefix(
    spec: FusionSpec, ctx: MatchCtx, profile: str | Profile | None = None
) -> tuple[object | None, int]:
    """Return (plan, n_stages) for the longest covered prefix, or (None, 0). ``profile``: a name/path, an already loaded Profile (used as is: a rules-only profile has no
    rows, so it must never be tested for truthiness), or None (the env var)."""
    prof = profile if isinstance(profile, Profile) else load_profile(profile)
    for n in range(len(spec.stages), 0, -1):
        sub = spec.prefix(n)
        row, source = prof.resolve(sub.signature(), ctx.arch, ctx.cu, ctx.world, ctx.m)
        if row is None:
            continue
        if row.vid not in catalog._CATALOG:
            _load_builtin(row.vid)
        info = catalog.get_variant(row.vid) if row.vid in catalog._CATALOG else None
        if info is None:
            raise ContractError(
                f"profile row references unregistered variant {row.vid!r}"
            )
        _check_contract(info, sub, ctx, row)
        plan = info.factory(sub, dict(row.params), ctx)
        plan.vid = row.vid
        plan.n_stages = n
        plan.profile_source = source  # "row" | "rule"
        log.info(
            "event=dispatch sig=%s m=%d vid=%s n_stages=%d source=%s%s params=%s",
            sub.signature(),
            ctx.m,
            row.vid,
            n,
            source,
            f" rule={row.label()}" if source == "rule" else "",
            dict(row.params),
        )
        return plan, n
    log.info(
        "event=dispatch sig=%s m=%d vid=native n_stages=0 source=none",
        spec.signature(),
        ctx.m,
    )
    return None, 0
