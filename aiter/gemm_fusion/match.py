# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Longest-prefix matching of a FusionSpec against an immutable profile.

Semantics (Artemis §3.4): no profile row for any prefix -> (None, 0) = native.
Row found but its variant contract is not satisfied -> ContractError.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import FrozenSet, Optional, Tuple

from . import catalog
from .profile import load_profile
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
    available: FrozenSet[str] = frozenset()  # {"mori_cco","sdma","gfx950_sc_bypass"}
    graph_mode: str = "eager"  # "eager" | "capture"


def _check_contract(info, sub: FusionSpec, ctx: MatchCtx, row):
    c = info.contract
    bad = []
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
            except ImportError as e:  # surfaces as an unregistered variant (ContractError)
                log.warning("event=variant_load_failed vid=%s err=%s", vid, e)
            return


def match_prefix(
    spec: FusionSpec, ctx: MatchCtx, profile: Optional[str] = None
) -> Tuple[Optional[object], int]:
    """Return (plan, n_stages) for the longest covered prefix, or (None, 0)."""
    prof = load_profile(profile)
    for n in range(len(spec.stages), 0, -1):
        sub = spec.prefix(n)
        row = prof.lookup(sub.signature(), ctx.arch, ctx.cu, ctx.world, ctx.m)
        if row is None:
            continue
        if row.vid not in catalog._CATALOG:
            _load_builtin(row.vid)
        info = catalog.get_variant(row.vid) if row.vid in catalog._CATALOG else None
        if info is None:
            raise ContractError(f"profile row references unregistered variant {row.vid!r}")
        _check_contract(info, sub, ctx, row)
        plan = info.factory(sub, dict(row.params), ctx)
        plan.vid = row.vid
        plan.n_stages = n
        log.info(
            "event=dispatch sig=%s m=%d vid=%s n_stages=%d params=%s",
            sub.signature(), ctx.m, row.vid, n, dict(row.params),
        )
        return plan, n
    log.info("event=dispatch sig=%s m=%d vid=native n_stages=0", spec.signature(), ctx.m)
    return None, 0
