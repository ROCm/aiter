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
from dataclasses import dataclass, replace

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


def _variant_info(vid: str):
    if vid not in catalog._CATALOG:
        _load_builtin(vid)
    info = catalog.get_variant(vid) if vid in catalog._CATALOG else None
    if info is None:
        raise ContractError(f"profile row references unregistered variant {vid!r}")
    return info


def bucket_for_m(
    spec: FusionSpec, ctx: MatchCtx, profile: str | Profile | None = None
) -> int | None:
    """R16 pad-M, bucket selection ("the padded m only selects the kernel / plan; the kernel only gets the valid m"): the prepared bucket M a real local m = ctx.m runs in, or None (native).

    Per prefix (longest first, like match_prefix) the candidates are the profile's EXACT rows of (sig, arch, cu, world), ascending in m: the smallest row with m_row >= m whose variant
    accepts the launch (``Plan.accepts_launch_m(params, m_row, m)``: m == m_row always; below it only pad-M sources, within their padded rows and shape switches) wins, so m == an
    exact row is that row, and e.g. m 129..256 lands on the MP256 row, m 65..128 on the MP128 row, m 896..1023 on the 1024 row, m 1025..1280 on the 1280 row. A variant that declares ``pad_m_min_field``
    (the large plans: profile field ``m_pad_min_large``, default 896 = where the pad path starts to beat native at the 1024 bucket) is padded only from that m on: below it only an exact row is used. When no row can take m, a range rule covering
    m resolves it exactly (the bucket is m itself); otherwise the next shorter prefix, else None. A pure function of (spec, arch, cu, world, m) and the immutable profile, so every
    rank chooses the same bucket. Capabilities / graph mode are not consulted here (they are checked when the plan is built for the bucket).
    """
    prof = profile if isinstance(profile, Profile) else load_profile(profile)
    for n in range(len(spec.stages), 0, -1):
        got = bucket_in_band(
            prof, spec.prefix(n).signature(), ctx.arch, ctx.cu, ctx.world, ctx.m
        )
        if got is not None:
            return got[0]
    return None


def bucket_in_band(
    prof: Profile, sig: str, arch: str, cu: int, world: int, m: int
) -> tuple[int, str] | None:
    """The bucket of ONE (sig, arch, cu, world) band for the real local m: (bucket_m, vid), or None (see bucket_for_m for the rule). Shared with the call_id universe."""
    m = int(m)
    for row in prof.bucket_rows(sig, arch, cu, world):
        if row.m < m:
            continue
        if row.m == m:
            return m, row.vid
        cls = _variant_info(row.vid).factory
        field = getattr(cls, "pad_m_min_field", None)
        if field is not None and m < int(
            prof.meta.get(field, getattr(cls, "pad_m_min_default", 1))
        ):
            continue
        accepts = getattr(cls, "accepts_launch_m", None)
        if accepts is not None and accepts(dict(row.params), row.m, m):
            return row.m, row.vid
    rule = prof.lookup_rule(sig, arch, cu, world, m)
    return (m, rule.vid) if rule is not None else None


def match_padded(
    spec: FusionSpec, ctx: MatchCtx, profile: str | Profile | None = None
) -> tuple[object | None, int, int | None]:
    """R16 pad-M: (plan, n_stages, bucket_m) for the REAL local m = ctx.m: the plan is the one prepared for the bucket (bucket_for_m); run it with x holding exactly the real
    m rows (the plan's launch gets m_real = x.shape[0]). (None, 0, None) = native. m == an exact row gives that row's plan, bucket == m.
    """
    prof = profile if isinstance(profile, Profile) else load_profile(profile)
    bucket = bucket_for_m(spec, ctx, prof)
    if bucket is None:
        log.info(
            "event=dispatch_padded sig=%s m=%d vid=native source=none",
            spec.signature(),
            ctx.m,
        )
        return None, 0, None
    plan, n = match_prefix(spec, replace(ctx, m=bucket), prof)
    if plan is not None:
        log.info(
            "event=dispatch_padded sig=%s m=%d bucket=%d vid=%s n_stages=%d",
            spec.signature(),
            ctx.m,
            bucket,
            plan.vid,
            n,
        )
    return plan, n, bucket
