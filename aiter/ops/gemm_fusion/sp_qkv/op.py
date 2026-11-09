# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Neutral public op for the MiniMax-M3 SP4 QKV fusion (``sp_qkv_fused`` v1).

This is the AITER-owned contract surface extracted from the ``gemm_fusion``/``sp_qkv``
framework.  It is intentionally backend-free: the public signature speaks in the
semantic tensors of one SP4 QKV step (no ``res``/``profile``/``bucket``/``call_id``/
``variant`` policy arguments), and the native implementation is delegated to
``aiter.ops.gemm_fusion.sp_qkv.runtime.sp_qkv_native(payload)``.

Provider-route guard
--------------------
``aiter.plugin`` (the interface stack aiter#7465/#7466/#7484) is **not on main** at the
PR head this module was written against (``582d19d0``).  The ``route`` decorator is
therefore imported behind a guard: when the plugin package is present the op is
registered as ``sp_qkv_fused`` v1 (and carries ``__aiter_plugin_op__``); when it is
absent the decorator is an identity shim, so importing this module and calling the op
still works on every checkout.  The guard is deliberately local to this op; it must be
removed (replaced by the direct import) once the interface stack lands.

The payload passed to the native runtime uses the contract argument names as keys and
the signature order as its order.  The runtime owns profile/variant/resource selection
and the fused-vs-unfused policy; see ``CONTRACT.md`` for the full contract sheet.

Coverage predicate
------------------
``sp_qkv_covers(...)`` takes the same flat arguments and asks the runtime
``sp_qkv_covers(payload)`` whether the fused-native path covers the call.  It is a
cheap, no-allocation/no-launch gate for consumers (ATOM's mode-B binding must gate on
it before selecting a call path); it returns ``False`` when the runtime or the
predicate is unavailable.  It does not change ``sp_qkv_native``'s behaviour:
uncovered calls still raise ``ContractError`` when executed.

**Coverage today:** the fused-native coverage is the prologue cells the runtime can
serve (2-stage plans: the profile's prologue buckets + pad hits, i.e. local ``m``
``1..256`` for the default profile, all three modes).  The one-stage large / r19 band
(``m`` 512..16384) needs a native post-exchange prologue stage first; it is Phase-2
coverage, not served today (see CONTRACT.md §8).
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

__all__ = ["sp_qkv_covers", "sp_qkv_fused"]

MODES = ("dense", "indexer", "reuse")

# ---------------------------------------------------------------------------
# Guarded provider-route decorator.
#
# At the pinned PR head `aiter.plugin` does not exist.  `try/except ImportError` (not
# ModuleNotFoundError) also tolerates a partial/older plugin package without `route`.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - branch selected by the installed aiter
    from aiter.plugin import route as _plugin_route

    _PLUGIN_ROUTE_PRESENT = True
except ImportError:  # pragma: no cover - the current main / PR head
    _PLUGIN_ROUTE_PRESENT = False

    def _plugin_route(
        *, op: str, version: int = 1, supersedes: Mapping[int, str] | None = None
    ):
        """Identity shim used while the provider interface is not merged."""

        def _decorate(fn):
            return fn

        return _decorate


# `runtime.py` (owned by the runtime implementor) is resolved lazily on the first call:
# importing the op / querying coverage must not import the framework eagerly or create
# a session.  Tests and the provider integration can replace `runtime` with a fake
# module before the first call.
runtime = None


def _native_runtime() -> Any:
    """The native runtime module, imported on first use when it was absent at import."""

    global runtime
    if runtime is None:
        try:
            from . import runtime as runtime_module
        except ImportError as exc:  # pragma: no cover - defensive
            raise ImportError(
                "sp_qkv_fused needs aiter.ops.gemm_fusion.sp_qkv.runtime "
                "(the native implementation of this op)"
            ) from exc
        runtime = runtime_module
    return runtime


def _validate_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return mode


def _validate_mode_tensors(
    mode: str,
    *,
    k: Any,
    v: Any,
    index_q: Any,
    index_q_norm_w: Any,
    index_k_norm_w: Any,
    index_cache: Any,
) -> None:
    """Reject an internally inconsistent mode/tensor set before the native path runs.

    These are the mode-specific output/weight tensors of the contract; the native path
    still performs the full dtype/shape/layout checks (``inputs.py``).  Keeping this
    cheap check here makes an invalid *call shape* a public-op error rather than a
    provider-selection surprise.
    """

    if mode == "indexer":
        missing = [
            name
            for name, value in (
                ("index_q", index_q),
                ("index_q_norm_w", index_q_norm_w),
                ("index_k_norm_w", index_k_norm_w),
                ("index_cache", index_cache),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"mode='indexer' requires {', '.join(missing)} (the indexer outputs/weights)"
            )
        return
    missing = [name for name, value in (("k", k), ("v", v)) if value is None]
    if missing:
        raise ValueError(
            f"mode={mode!r} requires {', '.join(missing)} (the dense/reuse KV outputs)"
        )


@_plugin_route(op="sp_qkv_fused", version=1)
def sp_qkv_fused(
    q,
    x,
    x_scale,
    w,
    w_scale,
    q_norm_w,
    k_norm_w,
    cos_cache,
    sin_cache,
    positions,
    slot_mapping,
    k_cache,
    v_cache,
    k_scale,
    v_scale,
    *,
    mode,
    k=None,
    v=None,
    index_q=None,
    index_q_norm_w=None,
    index_k_norm_w=None,
    index_cache=None,
    eps=1e-6,
    group=None,
    stream=None,
) -> None:
    """One MiniMax-M3 SP4 QKV step: GEMM + SP exchange + head norm + RoPE + cache writes.

    The semantic contract (shapes/dtypes/layouts, output mutation, tolerance,
    workspace/lifetime, error behaviour, coverage and non-goals) is
    ``aiter/ops/gemm_fusion/sp_qkv/CONTRACT.md``.  ``mode`` changes the GEMM N and which
    caches are written (``dense``/``indexer``/``reuse``); it is a contract argument,
    unlike tuner/profile/bucket policy.  Returns ``None``; every output and cache tensor
    is written in place.

    The native path is **coverage-limited today** (skill §5 option B): the fused-native
    coverage is the prologue cells the runtime serves (default profile: local m 1..256,
    all modes); the one-stage large / r19 band is Phase-2 coverage that needs the
    post-exchange prologue to become an aiter-owned native stage/op first.  Calls
    outside coverage raise ``ContractError`` instead of silently taking a different
    implementation; option A (an unfused fallback owned by native) is the named
    follow-up.  A provider may decline any call and native serves it; once a provider's
    ``run`` starts, an exception propagates and is never retried natively.

    Use :func:`sp_qkv_covers` (same flat arguments) for a cheap, non-executing coverage
    check before selecting a call path.
    """

    _validate_mode(mode)
    _validate_mode_tensors(
        mode,
        k=k,
        v=v,
        index_q=index_q,
        index_q_norm_w=index_q_norm_w,
        index_k_norm_w=index_k_norm_w,
        index_cache=index_cache,
    )

    _native_runtime().sp_qkv_native(
        _build_payload(
            q=q,
            x=x,
            x_scale=x_scale,
            w=w,
            w_scale=w_scale,
            q_norm_w=q_norm_w,
            k_norm_w=k_norm_w,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            positions=positions,
            slot_mapping=slot_mapping,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=k_scale,
            v_scale=v_scale,
            mode=mode,
            k=k,
            v=v,
            index_q=index_q,
            index_q_norm_w=index_q_norm_w,
            index_k_norm_w=index_k_norm_w,
            index_cache=index_cache,
            eps=eps,
            group=group,
            stream=stream,
        )
    )


def sp_qkv_covers(
    q,
    x,
    x_scale,
    w,
    w_scale,
    q_norm_w,
    k_norm_w,
    cos_cache,
    sin_cache,
    positions,
    slot_mapping,
    k_cache,
    v_cache,
    k_scale,
    v_scale,
    *,
    mode,
    k=None,
    v=None,
    index_q=None,
    index_q_norm_w=None,
    index_k_norm_w=None,
    index_cache=None,
    eps=1e-6,
    group=None,
    stream=None,
) -> bool:
    """Cheap, non-executing fused-native coverage predicate for the same call.

    Builds the same flat payload as :func:`sp_qkv_fused` and delegates to
    ``runtime.sp_qkv_covers(payload)``.  No allocation, no launch, no prepare, no
    provider selection.  Returns ``False`` for an unknown ``mode``, a missing runtime,
    or a runtime without the predicate; an invalid mode never raises (it is simply not
    covered).  This is the gate ATOM's mode-B binding uses before choosing a path.
    """

    if mode not in MODES:
        return False
    try:
        runtime_module = _native_runtime()
    except ImportError:
        return False
    predicate = getattr(runtime_module, "sp_qkv_covers", None)
    if predicate is None:
        return False
    payload = _build_payload(
        q=q,
        x=x,
        x_scale=x_scale,
        w=w,
        w_scale=w_scale,
        q_norm_w=q_norm_w,
        k_norm_w=k_norm_w,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        positions=positions,
        slot_mapping=slot_mapping,
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
        mode=mode,
        k=k,
        v=v,
        index_q=index_q,
        index_q_norm_w=index_q_norm_w,
        index_k_norm_w=index_k_norm_w,
        index_cache=index_cache,
        eps=eps,
        group=group,
        stream=stream,
    )
    return bool(predicate(payload))


def _build_payload(
    *,
    q,
    x,
    x_scale,
    w,
    w_scale,
    q_norm_w,
    k_norm_w,
    cos_cache,
    sin_cache,
    positions,
    slot_mapping,
    k_cache,
    v_cache,
    k_scale,
    v_scale,
    mode,
    k,
    v,
    index_q,
    index_q_norm_w,
    index_k_norm_w,
    index_cache,
    eps,
    group,
    stream,
):
    """The flat native payload: contract names, signature order, optional values None."""

    return {
        "q": q,
        "x": x,
        "x_scale": x_scale,
        "w": w,
        "w_scale": w_scale,
        "q_norm_w": q_norm_w,
        "k_norm_w": k_norm_w,
        "cos_cache": cos_cache,
        "sin_cache": sin_cache,
        "positions": positions,
        "slot_mapping": slot_mapping,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "mode": mode,
        "k": k,
        "v": v,
        "index_q": index_q,
        "index_q_norm_w": index_q_norm_w,
        "index_k_norm_w": index_k_norm_w,
        "index_cache": index_cache,
        "eps": eps,
        "group": group,
        "stream": stream,
    }


# The real route() decorator already stamps __aiter_plugin_op__ on its wrapper; make the
# marker explicit here as well so consumers that import the op directly can see it.
if _PLUGIN_ROUTE_PRESENT:  # pragma: no branch - exercised only with the interface stack
    try:
        from aiter.plugin import Op as _PluginOp

        sp_qkv_fused.__aiter_plugin_op__ = _PluginOp("sp_qkv_fused", 1)
    except (ImportError, AttributeError):  # pragma: no cover - partial plugin package
        pass


# Keep the native contract visibly introspectable even when the identity shim is used:
# the real route() wrapper preserves this through functools.wraps / __wrapped__.
assert (
    inspect.signature(sp_qkv_fused).parameters["mode"].kind
    is inspect.Parameter.KEYWORD_ONLY
)
assert list(inspect.signature(sp_qkv_covers).parameters.values()) == list(
    inspect.signature(sp_qkv_fused).parameters.values()
)
