# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch core of the neutral op ``aiter.ops.qsa.qsa_paged_mqa_logits`` (private).

The public surface is :mod:`aiter.ops.qsa` (op + ``configure`` / ``prewarm`` /
``explain`` / ``status``); this module holds the policy state and the per-call-class
decision cache so it can be tested without a GPU (torch only, no FlyDSL, no Triton).

Order of implementations for one call class (decided once, on the host):

1. ``fused`` -- the FlyDSL gfx1250 kernels of :mod:`.api` when a profile is active
   (``configure(profile=...)`` or ``$AITER_QSA_PROFILE``, not killed by
   ``$AITER_QSA_FUSED=0``) and a verified profile row / rule / bucket covers the class;
2. ``native-triton`` -- aiter's existing native dispatch (Triton, or the FlyDSL page-16
   prefill scorer when ``allow_flydsl``), unchanged; the caller of :mod:`aiter.ops.qsa`
   passes it in as ``existing`` and says which calls it takes (``existing_route``);
3. ``native-torch`` -- :mod:`.native` (pure torch, any device) for valid calls the
   existing dispatch cannot take (CPU tensors, an fp8 dtype the arch's Triton path does
   not use, ...).

A call the contract rejects is ``invalid`` and raises ``ValueError`` (the existing
dispatch's own error where it has one). Default: no profile -> the existing dispatch
serves every call it served before, with no decision cache and no FlyDSL import.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from . import native as N

logger = logging.getLogger("aiter")

#: op contract (AITER-owned): name + version; bump the version on any semantic change
OP_NAME = "qsa_paged_mqa_logits"
OP_CONTRACT_VERSION = 1
#: version of the policy API (configure / prewarm / explain / status)
API_VERSION = 1
#: operator override of the profile (wins over ``configure``); ``off`` disables
PROFILE_ENV = "AITER_QSA_PROFILE"
#: kill switch: ``0`` / ``off`` / ``false`` / ``no`` disables the fused path entirely
FUSED_ENV = "AITER_QSA_FUSED"
#: explicit FlyDSL cache directory for ``prewarm`` (default: FlyDSL's own, where the
#: AOT build ``aiter.aot.flydsl.qsa`` writes)
CACHE_DIR_ENV = "AITER_QSA_INDEX_SCORER_CACHE_DIR"

PATH_FUSED = "fused"
PATH_NATIVE_TRITON = "native-triton"
PATH_NATIVE_TORCH = "native-torch"
PATH_INVALID = "invalid"
PATHS = (PATH_FUSED, PATH_NATIVE_TRITON, PATH_NATIVE_TORCH)

VID_TRITON = "native.triton.v1"
VID_FLYDSL_PAGE16 = "native.flydsl_page16.v1"
VID_TORCH = N.NATIVE_VID

#: decided call classes kept (FIFO beyond this; a class is a few hundred bytes plus,
#: for ``native-torch`` only, its bounded workspace)
MAX_CLASSES = 4096
_OFF = ("0", "off", "none", "false", "no")


@dataclass(frozen=True)
class Decision:
    """Which implementation serves a call class, and why (``explain``)."""

    path: str  # "fused" | "native-triton" | "native-torch" | "invalid"
    reason: str  # "ok" (fused), else why the fused path did not take the class
    vid: str | None = None  # variant id (fused) / native implementation id
    detail: str = ""
    profile: str | None = None  # active profile when decided (None = fused off)
    cell: str | None = None  # profile row / rule / bucket that covers a fused class


class _Entry:
    __slots__ = ("decision", "first", "fused", "run")

    def __init__(self, decision: Decision, run: Callable, fused=None):
        self.decision = decision
        self.run = run
        self.fused = fused  # api.Decision of a fused class
        self.first = fused is not None


# ---------------------------------------------------------------------------
# Policy state (process-global; environment snapshotted at import / configure)
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()
_CONFIGURED: str | None = None
_SOURCE = "default"
_ENV: tuple = (None, None)  # (AITER_QSA_PROFILE, AITER_QSA_FUSED) snapshot
_ACTIVE: str | None = None  # effective profile (None = fused off)
_ORIGIN = "default"
_PROFILE_ERROR: str | None = None
_CLASSES: dict = {}
#: fall back to native when a fused class fails on its first launch (compile/load error,
#: before any device write); the AOT replay turns it off so a failure fails the build
FUSED_FALLBACK_ON_ERROR = True


def _read_env() -> tuple:
    return os.environ.get(PROFILE_ENV), os.environ.get(FUSED_ENV)


def _effective(env: tuple, configured: str | None, source: str) -> tuple:
    prof_env, fused_env = env
    if fused_env is not None and fused_env.strip().lower() in _OFF:
        return None, f"kill-switch:{FUSED_ENV}={fused_env.strip()}"
    if prof_env is not None and prof_env.strip():
        v = prof_env.strip()
        return (None if v.lower() in _OFF else v), f"env:{PROFILE_ENV}"
    if configured:
        return configured, source
    return None, "default"


def _apply(configured: str | None, source: str, env: tuple) -> None:
    """Install a policy (caller holds ``_LOCK``); drops decided classes on any change."""
    global _CONFIGURED, _SOURCE, _ENV, _ACTIVE, _ORIGIN, _PROFILE_ERROR
    active, origin = _effective(env, configured, source)
    changed = (active, env) != (_ACTIVE, _ENV)
    _CONFIGURED, _SOURCE, _ENV = configured, source, env
    _ACTIVE, _ORIGIN = active, origin
    if changed:
        _CLASSES.clear()
        _PROFILE_ERROR = _check_profile(active)
        if _PROFILE_ERROR:
            logger.warning(
                "[aiter] qsa_paged_mqa_logits: profile %r unusable (%s); every call "
                "is served natively",
                active,
                _PROFILE_ERROR,
            )


def _check_profile(name: str | None) -> str | None:
    if name is None:
        return None
    try:
        _api().P.load_profile(name)
    except Exception as exc:  # noqa: BLE001 -- reported, never raised
        return f"{type(exc).__name__}: {exc}"
    return None


def configure(*, profile: str | None, source: str) -> None:
    """Enable the fused path with a named profile (``None`` = off); idempotent.

    ``source`` names who decided (e.g. ``"atom:<config>"``) and is reported by
    :func:`status`. ``$AITER_QSA_PROFILE`` (operator) overrides ``profile``;
    ``$AITER_QSA_FUSED=0`` disables the fused path whatever is configured. The
    environment is re-read here (and at import). An unusable profile name is reported
    (``status()["profile_error"]``, a warning) and every call is served natively.
    Reconfiguring to a different effective profile drops the decided call classes.
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError("configure(source=...) must name the caller")
    if profile is not None and (not isinstance(profile, str) or not profile.strip()):
        raise ValueError("configure(profile=...) must be a profile name/path or None")
    with _LOCK:
        _apply(profile.strip() if profile else None, source.strip(), _read_env())


def _reset_for_testing() -> None:
    """Back to the import-time state (environment re-read). Tests only."""
    with _LOCK:
        global _CONFIGURED, _SOURCE, _ACTIVE, _ENV
        _CONFIGURED, _SOURCE = None, "default"
        _ENV, _ACTIVE = (None, None), None
        _CLASSES.clear()
        _apply(None, "default", _read_env())


def active_profile() -> str | None:
    return _ACTIVE


# ---------------------------------------------------------------------------
# Lazy fused implementation
# ---------------------------------------------------------------------------
_API = None


def _api():
    """The fused implementation (:mod:`.api`); imported on first need (no FlyDSL)."""
    global _API
    if _API is None:
        from . import api

        _API = api
    return _API


# ---------------------------------------------------------------------------
# Call classes
# ---------------------------------------------------------------------------
def _tk(t):
    if t is None:
        return None
    return t.shape, t.dtype, t.is_contiguous(), t.is_cuda


def class_key(
    q,
    compressed_k_cache,
    page_table,
    token_to_request,
    query_positions,
    context_lens,
    compress_ratio,
    score_divisor,
    max_columns,
    row_starts,
    allow_flydsl,
    compressed_k_scale,
    stacked_decode,
    logits,
    visible_groups,
):
    """Everything a decision depends on, from tensor metadata only (no tensor reads, no
    sync): shapes, dtypes, contiguity (every implementation's choice depends on
    contiguity, not on the strides themselves), GPU placement and q's device, the
    cache base address (stable for a KV cache; the fused router keys on its
    alignment), q / scale alignment classes, the scalar arguments and output-buffer
    layouts. Raises for non-tensor arguments."""
    c, s = compressed_k_cache, compressed_k_scale
    return (
        q.shape,
        q.dtype,
        q.is_contiguous(),
        q.device,
        q.data_ptr() & 15,
        c.shape,
        c.dtype,
        c.is_contiguous(),
        c.is_cuda,
        c.data_ptr(),
        None if s is None else (_tk(s), s.data_ptr() & 3),
        _tk(page_table),
        _tk(token_to_request),
        _tk(query_positions),
        _tk(context_lens),
        _tk(row_starts),
        _tk(logits),
        _tk(visible_groups),
        compress_ratio,
        score_divisor,
        max_columns,
        allow_flydsl,
        stacked_decode,
    )


def _decide(existing, existing_route, args: tuple) -> _Entry:
    """Host-side decision of one call class (never raises for any argument values)."""
    (
        q,
        cache,
        pt,
        t2r,
        pos,
        ctx,
        ratio,
        divisor,
        maxc,
        rs,
        allow,
        scale,
        _stacked,
        logits,
        vg,
    ) = args
    prof = _ACTIVE
    reason, detail, cell = ("disabled", "", None)
    if prof is None:
        if _ORIGIN.startswith("kill-switch"):
            reason, detail = "kill_switch", _ORIGIN
        elif _ORIGIN.startswith("env:"):
            reason, detail = "disabled", _ORIGIN
    elif _PROFILE_ERROR:
        reason, detail = "profile_unusable", _PROFILE_ERROR
    else:
        try:
            d = _api().decide(
                q,
                cache,
                pt,
                t2r,
                pos,
                ctx,
                ratio,
                divisor,
                maxc,
                rs,
                allow,
                scale,
                logits=logits,
                visible_groups=vg,
                profile=prof,
                native=False,
            )
        except Exception as exc:  # noqa: BLE001 -- the native path serves the call
            reason, detail = "fused_decide_error", f"{type(exc).__name__}: {exc}"
        else:
            if d.ok:
                dec = Decision(PATH_FUSED, "ok", d.vid, d.detail, prof, d.cell)
                return _Entry(dec, _fused_runner(d), fused=d)
            reason, detail, cell = d.reason.value, d.detail, d.cell
    return _native_entry(existing, existing_route, args, reason, detail, prof, cell)


def _native_entry(existing, existing_route, args, reason, detail, prof, cell):
    q, cache, pt, t2r, pos, ctx, ratio, divisor, maxc, rs, allow, scale = args[:12]
    logits, vg = args[13:]
    try:
        vid = existing_route(q, cache, pt, allow, scale)
    except Exception:  # noqa: BLE001 -- not a call the existing dispatch can take
        vid = None
    if vid is not None:
        dec = Decision(PATH_NATIVE_TRITON, reason, vid, detail, prof, cell)
        return _Entry(dec, existing)
    try:
        geo = N.validate(
            q, cache, pt, t2r, pos, ctx, ratio, divisor, maxc, rs, scale, logits, vg
        )
    except Exception as exc:  # noqa: BLE001 -- contract violation (or non-tensors)
        msg = f"{detail}; contract: {exc}" if detail else f"contract: {exc}"
        dec = Decision(PATH_INVALID, reason, None, msg, prof, cell)
        return _Entry(dec, _invalid_runner(existing, exc))
    plan = N.NativePlan(geo)
    dec = Decision(PATH_NATIVE_TORCH, reason, VID_TORCH, detail, prof, cell)
    return _Entry(dec, _torch_runner(plan))


def _fused_runner(d):
    launch = _api().launch

    def run(
        q,
        cache,
        pt,
        t2r,
        pos,
        ctx,
        ratio,
        divisor,
        maxc,
        rs,
        allow,
        scale,
        stacked,
        logits,
        vg,
    ):
        return launch(
            d,
            q,
            cache,
            pt,
            t2r,
            pos,
            ctx,
            ratio,
            divisor,
            maxc,
            rs,
            allow,
            scale,
            logits=logits,
            visible_groups=vg,
        )

    return run


def _torch_runner(plan):
    def run(
        q,
        cache,
        pt,
        t2r,
        pos,
        ctx,
        ratio,
        divisor,
        maxc,
        rs,
        allow,
        scale,
        stacked,
        logits,
        vg,
    ):
        return plan.run(q, cache, pt, t2r, pos, ctx, rs, scale, logits, vg)

    return run


def _invalid_runner(existing, exc):
    def run(*args):
        if isinstance(args[0], torch.Tensor) and args[0].is_cuda:
            existing(*args)  # the existing dispatch's own error for this call
        raise N.NativeContractError(str(exc))

    return run


def _entry(existing, existing_route, args) -> _Entry:
    try:
        key = class_key(*args)
    except Exception:  # noqa: BLE001 -- non-tensor arguments: decide, do not cache
        return _decide(existing, existing_route, args)
    e = _CLASSES.get(key)
    if e is None:
        e = _decide(existing, existing_route, args)
        if len(_CLASSES) >= MAX_CLASSES:
            _CLASSES.pop(next(iter(_CLASSES)), None)
        _CLASSES[key] = e
    return e


def call(existing, existing_route, *args):
    """Serve one call: ``args`` = the 12 contract arguments, ``stacked_decode``,
    ``logits``, ``visible_groups`` (positional). First call of a class decides on the
    host; later calls are a dict lookup plus the implementation's launch."""
    e = _entry(existing, existing_route, args)
    if e.first:
        return _first_fused(e, existing, existing_route, args)
    return e.run(*args)


def _first_fused(e: _Entry, existing, existing_route, args):
    """First launch of a fused class: compile/load errors (before any device write)
    re-decide the class natively (logged); afterwards errors propagate."""
    try:
        out = e.run(*args)
    except Exception as exc:
        if not FUSED_FALLBACK_ON_ERROR:
            raise
        logger.warning(
            "[aiter] qsa_paged_mqa_logits: fused %s failed on first launch (%s: %s); "
            "class served natively",
            e.decision.vid,
            type(exc).__name__,
            exc,
        )
        ne = _native_entry(
            existing,
            existing_route,
            args,
            "fused_launch_error",
            f"{e.decision.vid}: {type(exc).__name__}: {exc}",
            e.decision.profile,
            e.decision.cell,
        )
        with contextlib.suppress(Exception):
            _CLASSES[class_key(*args)] = ne
        return ne.run(*args)
    e.first = False
    return out


def explain(existing, existing_route, *args) -> Decision:
    """The :class:`Decision` for this call's class (decided and cached if new)."""
    return _entry(existing, existing_route, args).decision


def prewarm(
    *,
    page_sizes,
    cache_dtypes,
    pos_dtypes=(torch.int64,),
    compressed_k_cache=None,
    compressed_k_scale=None,
    device=None,
) -> list[str]:
    """Load / compile every fused variant the active profile can select for these
    caches (an AOT-built FlyDSL cache makes this a disk hit). ``[]`` when the fused path
    is off, the profile is unusable or the device is not one the profile covers."""
    prof = _ACTIVE
    if prof is None or _PROFILE_ERROR:
        return []
    tags = tuple(_dtype_tag(d) for d in cache_dtypes)
    pages = None if page_sizes is None else tuple(int(p) for p in page_sizes)
    return _api().prewarm(
        prof,
        cache_dtypes=tags,
        pos_dtypes=tuple(pos_dtypes),
        device=device,
        page_sizes=pages,
        compressed_k_cache=compressed_k_cache,
        compressed_k_scale=compressed_k_scale,
        cache_dir=os.environ.get(CACHE_DIR_ENV) or "",
    )


def _dtype_tag(d) -> str:
    if isinstance(d, str):
        if d not in ("bf16", "fp8"):
            raise ValueError(f"cache dtype {d!r}: 'bf16' or 'fp8'")
        return d
    if d == torch.bfloat16:
        return "bf16"
    if d in tuple(
        getattr(torch, n)
        for n in ("float8_e4m3fn", "float8_e4m3fnuz")
        if hasattr(torch, n)
    ):
        return "fp8"
    raise ValueError(f"cache dtype {d}: bf16 or fp8 E4M3")


def status() -> dict[str, Any]:
    """Observability snapshot (cheap; no kernel import, no device query)."""
    paths: dict[str, int] = {}
    for e in list(_CLASSES.values()):
        paths[e.decision.path] = paths.get(e.decision.path, 0) + 1
    out: dict[str, Any] = {
        "op": OP_NAME,
        "op_version": OP_CONTRACT_VERSION,
        "api_version": API_VERSION,
        "enabled": _ACTIVE is not None and not _PROFILE_ERROR,
        "profile": _ACTIVE,
        "origin": _ORIGIN,
        "configured_profile": _CONFIGURED,
        "configured_source": _SOURCE,
        "env": {PROFILE_ENV: _ENV[0], FUSED_ENV: _ENV[1]},
        "profile_error": _PROFILE_ERROR,
        "classes": len(_CLASSES),
        "paths": paths,
        "kernel_sha256": None,
        "profile_kernel_sha256": None,
        "impl_api_version": None,
    }
    if _ACTIVE is not None and not _PROFILE_ERROR:
        api = _api()
        try:
            out["kernel_sha256"] = api._kernel_sha256()
            prof = api.P.load_profile(_ACTIVE)
            out["profile_kernel_sha256"] = prof.kernel_sha256
            out["profile_path"] = prof.path
            out["impl_api_version"] = api.P.API_VERSION
            out["kernel_loaded"] = api._kernel_loaded()
        except Exception as exc:  # noqa: BLE001
            out["profile_error"] = f"{type(exc).__name__}: {exc}"
    return out


with _LOCK:
    _apply(None, "default", _read_env())
