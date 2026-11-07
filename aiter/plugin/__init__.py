# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Experimental interface for out-of-tree AITER kernel providers.

Ownership boundary:
- AITER owns the op contract (signatures, versions), the execution rules
  (isolation, fallback, no-fallback-after-run), and observable provenance.
- Providers own their kernels, their qualification evidence, and their
  lifecycle; ``KernelCandidate.qualification`` is self-reported provenance,
  not an AITER certification.
- Frameworks and users own enablement: installing a provider never makes it
  the default. A provider serves only when explicitly enabled
  (``AITER_PLUGIN_ENABLE``) or forced (``AITER_PLUGIN_FORCE``).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Collection, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from importlib import metadata
from inspect import Signature, signature
from threading import Lock
from types import MappingProxyType
from typing import Any, ParamSpec, Protocol, TypeVar, cast

API_VERSION = 1
ENTRY_POINT_GROUP = "aiter.kernel_providers"

_P = ParamSpec("_P")
_R = TypeVar("_R")

logger = logging.getLogger("aiter.plugin")


@dataclass(frozen=True, slots=True)
class Op:
    """A versioned operation owned by AITER."""

    name: str
    version: int

    def __post_init__(self) -> None:
        if not self.name or self.version < 1:
            raise ValueError("op name must be non-empty and version must be positive")


@dataclass(frozen=True, slots=True)
class KernelCandidate:
    """An immutable identity and entry point for one provider kernel."""

    provider: str
    kernel_id: str
    artifact_digest: str
    qualification: str
    run: Callable[..., Any] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        identity = (
            self.provider,
            self.kernel_id,
            self.artifact_digest,
            self.qualification,
        )
        if not all(identity) or not callable(self.run):
            raise ValueError(
                "candidate identity must be complete and run must be callable"
            )


class KernelProvider(Protocol):
    """Contract implemented by an out-of-tree provider entry point.

    Entry points must target a provider instance or a zero-argument
    factory returning a provider. Anything with a ``select`` attribute is
    treated as an instance, so a class object is never called.
    """

    name: str
    api_version: int
    op_versions: Mapping[str, Collection[int]]

    def select(
        self, op: Op, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> KernelCandidate | None: ...


@dataclass(frozen=True, slots=True)
class Decision:
    """The observable routing result for the current execution context."""

    op: Op
    route: str
    reason: str
    provider: str | None = None
    kernel_id: str | None = None
    artifact_digest: str | None = None
    qualification: str | None = None


@dataclass(frozen=True, slots=True)
class _ProviderRecord:
    entry_point: str
    provider: KernelProvider


_provider_records: tuple[_ProviderRecord, ...] | None = None
_provider_lock = Lock()
_last_decision: ContextVar[Decision | None] = ContextVar(
    "aiter_plugin_last_decision", default=None
)


def last_decision() -> Decision | None:
    """Return the last routing decision in the current execution context."""

    return _last_decision.get()


def route(*, op: str, version: int) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Route an AITER op through a compatible provider before its native path."""

    operation = Op(op, version)

    def decorate(native: Callable[_P, _R]) -> Callable[_P, _R]:
        contract = signature(native)

        @wraps(native)
        def routed(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return _dispatch(operation, contract, native, args, kwargs)

        routed.__aiter_plugin_op__ = operation
        return routed

    return decorate


def _dispatch(
    op: Op,
    contract: Signature,
    native: Callable[_P, _R],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> _R:
    disabled_reason = _disabled_reason(op)
    if disabled_reason:
        return _run_native(op, native, args, kwargs, disabled_reason)

    forced_provider = os.getenv("AITER_PLUGIN_FORCE")
    enabled = _enabled_providers()
    eligible = [
        record
        for record in _providers()
        if (forced_provider or record.provider.name in enabled)
        and (not forced_provider or record.provider.name == forced_provider)
        and _supports(record.provider, op)
    ]
    if not eligible:
        if forced_provider:
            reason = "forced-provider-unavailable"
        elif _providers() and not enabled:
            reason = "provider-not-enabled"
        else:
            reason = "no-provider"
        return _run_native(op, native, args, kwargs, reason)

    bound = contract.bind(*args, **kwargs)
    bound.apply_defaults()
    provider_args = bound.args
    provider_kwargs = bound.kwargs
    readonly_kwargs = MappingProxyType(provider_kwargs)
    declined = False
    for record in eligible:
        provider = record.provider
        try:
            candidate = provider.select(op, provider_args, readonly_kwargs)
        except Exception as error:  # noqa: BLE001 - providers are an isolation boundary
            logger.warning(
                "Provider %s selection failed for %s: %s", provider.name, op, error
            )
            return _run_native(
                op, native, args, kwargs, "provider-select-error", provider.name
            )

        if candidate is None:
            declined = True
            continue
        if (
            not isinstance(candidate, KernelCandidate)
            or candidate.provider != provider.name
        ):
            logger.warning(
                "Provider %s returned an invalid candidate (owned by %s)",
                provider.name,
                getattr(candidate, "provider", type(candidate).__name__),
            )
            return _run_native(
                op, native, args, kwargs, "invalid-candidate", provider.name
            )

        _last_decision.set(
            Decision(
                op=op,
                route="provider",
                reason="selected",
                provider=candidate.provider,
                kernel_id=candidate.kernel_id,
                artifact_digest=candidate.artifact_digest,
                qualification=candidate.qualification,
            )
        )
        # Once provider execution starts, errors propagate. Falling back here could
        # repeat side effects or hide a broken kernel behind a successful result.
        return cast(_R, candidate.run(*provider_args, **provider_kwargs))

    reason = "provider-declined" if declined else "no-provider"
    return _run_native(op, native, args, kwargs, reason)


def _run_native(
    op: Op,
    native: Callable[_P, _R],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    reason: str,
    provider: str | None = None,
) -> _R:
    _last_decision.set(
        Decision(op=op, route="native", reason=reason, provider=provider)
    )
    return native(*args, **kwargs)


def _disabled_reason(op: Op) -> str | None:
    if os.getenv("AITER_PLUGINS", "1").lower() in {"0", "false", "no", "off"}:
        return "plugins-disabled"
    disabled_ops = {
        name.strip() for name in os.getenv("AITER_PLUGIN_DISABLE", "").split(",")
    }
    if op.name in disabled_ops:
        return "op-disabled"
    return None


def _enabled_providers() -> frozenset[str]:
    """Providers allowed to serve; installing a provider never enables it."""
    return frozenset(
        name.strip()
        for name in os.getenv("AITER_PLUGIN_ENABLE", "").split(",")
        if name.strip()
    )


def _supports(provider: KernelProvider, op: Op) -> bool:
    try:
        return op.version in provider.op_versions.get(op.name, ())
    except Exception as error:  # noqa: BLE001 - providers are an isolation boundary
        logger.warning("Provider %s has invalid op_versions: %s", provider.name, error)
        return False


def _providers() -> Sequence[_ProviderRecord]:
    global _provider_records
    if _provider_records is None:
        with _provider_lock:
            if _provider_records is None:
                _provider_records = _discover_providers()
    return _provider_records


def _discover_providers() -> tuple[_ProviderRecord, ...]:
    records = []
    for entry_point in sorted(_entry_points(), key=lambda item: item.name):
        try:
            loaded = entry_point.load()
            provider = loaded if hasattr(loaded, "select") else loaded()
            name = provider.name
            api_version = provider.api_version
            _ = provider.op_versions
            if api_version != API_VERSION:
                logger.warning(
                    "Skipping provider %s: API version %s is incompatible with %s",
                    name,
                    api_version,
                    API_VERSION,
                )
                continue
            records.append(_ProviderRecord(entry_point.name, provider))
        except Exception as error:  # noqa: BLE001 - one plugin must not break AITER
            logger.warning(
                "Skipping provider entry point %s: %s", entry_point.name, error
            )
    return tuple(
        sorted(records, key=lambda item: (item.provider.name, item.entry_point))
    )


def _entry_points() -> Sequence[metadata.EntryPoint]:
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        return tuple(entry_points.select(group=ENTRY_POINT_GROUP))
    return tuple(entry_points.get(ENTRY_POINT_GROUP, ()))  # type: ignore[union-attr]


def _reset_for_testing() -> None:
    global _provider_records
    with _provider_lock:
        _provider_records = None
    _last_decision.set(None)


__all__ = [
    "API_VERSION",
    "Decision",
    "KernelCandidate",
    "KernelProvider",
    "Op",
    "last_decision",
    "route",
]
