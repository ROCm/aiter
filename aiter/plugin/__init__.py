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
from dataclasses import dataclass, field, replace
from functools import wraps
from importlib import metadata
from inspect import Parameter, Signature, signature
from threading import Lock
from types import MappingProxyType
from typing import Any, ParamSpec, Protocol, TypeVar, cast

API_VERSION = 1
ENTRY_POINT_GROUP = "aiter.kernel_providers"

_Canonicalizer = Callable[..., tuple[tuple[Any, ...], dict[str, Any]]]
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
    config_source: str | None = None


@dataclass(frozen=True, slots=True)
class _ProviderRecord:
    entry_point: str
    provider: KernelProvider


@dataclass(frozen=True, slots=True)
class _FrameworkDefault:
    """What ``configure()`` last set: the config layer below the environment."""

    enable: tuple[str, ...]
    disable: frozenset[str]
    source: str


@dataclass(frozen=True, slots=True)
class _Config:
    """An immutable configuration snapshot.

    Dispatch reads one reference per call, so a call always runs against exactly
    one generation. ``enabled`` is ordered: the enable list order is the
    preference order, and it is the only order source.
    """

    kill_switch: bool
    # op name -> the config layer that disabled it, so an op-disabled route
    # names the layer that produced it rather than the enable layer.
    disabled_ops: Mapping[str, str]
    forced_provider: str | None
    enabled: tuple[str, ...]
    source: str
    generation: int

    def _settings(self) -> tuple[Any, ...]:
        """Everything that defines this configuration except its generation."""

        return (
            self.kill_switch,
            self.disabled_ops,
            self.forced_provider,
            self.enabled,
            self.source,
        )


_provider_records: tuple[_ProviderRecord, ...] | None = None
_provider_lock = Lock()
_config_snapshot: _Config | None = None
_config_lock = Lock()
_framework_default: _FrameworkDefault | None = None
_eligible_cache: dict[tuple[str, int, int], tuple[_ProviderRecord, ...]] = {}
_FALSY = {"0", "false", "no", "off"}
_CANONICALIZER = "_canonicalize"
_DEFAULT_PREFIX = "__aiter_default_"
_last_decision: ContextVar[Decision | None] = ContextVar(
    "aiter_plugin_last_decision", default=None
)


def last_decision() -> Decision | None:
    """Return the last routing decision in the current execution context."""

    return _last_decision.get()


def configure(
    *,
    source: str,
    enable: Sequence[str] = (),
    disable: Sequence[str] = (),
) -> None:
    """Set the framework default enablement.

    ``enable`` is a provider list in preference order and **replaces** any
    previous ``configure()`` state; ``disable`` is a per-op opt-out that
    **unions** with the user's. ``source`` is required provenance ("vllm rocm
    defaults 2026.11") and is recorded so every route names an owner.

    Init-time by convention. A call after the first dispatch takes effect on the
    next dispatch and bumps the config generation; it is never silent. All
    parameters are keyword-only: a required parameter cannot follow a defaulted
    one, and keyword-only keeps call sites self-describing.
    """

    global _framework_default
    if not source.strip():
        raise ValueError("configure() requires a non-empty source")
    with _config_lock:
        _framework_default = _FrameworkDefault(
            # dict.fromkeys de-duplicates while preserving the preference order.
            enable=tuple(dict.fromkeys(n.strip() for n in enable if n.strip())),
            disable=frozenset(n.strip() for n in disable if n.strip()),
            source=source.strip(),
        )
        _swap_config_locked()


def reload_config() -> None:
    """Re-read the environment into a new snapshot.

    Mid-process environment changes are not observed otherwise. A reload that
    produces an identical snapshot does not bump the generation.
    """

    with _config_lock:
        _swap_config_locked()


def _env_list(name: str) -> tuple[str, ...] | None:
    """Parse an env list, or ``None`` when unset. An empty value counts as unset."""

    raw = os.getenv(name)
    if raw is None:
        return None
    items = tuple(item.strip() for item in raw.split(",") if item.strip())
    return items or None


def _build_config(default: _FrameworkDefault | None, generation: int) -> _Config:
    env_enable = _env_list("AITER_PLUGIN_ENABLE")
    env_disable = _env_list("AITER_PLUGIN_DISABLE") or ()
    forced = os.getenv("AITER_PLUGIN_FORCE") or None
    kill_switch = os.getenv("AITER_PLUGINS", "1").strip().lower() in _FALSY

    # Enable layers replace; disable layers union.
    enabled = (default.enable if default else ()) if env_enable is None else env_enable

    if default:
        framework_source = f"configure:{default.source}"
        disabled = dict.fromkeys(default.disable, framework_source)
    else:
        framework_source = None
        disabled = {}
    # Union, and the user's act is the one named when both layers disable an op.
    disabled.update(dict.fromkeys(env_disable, "env"))

    if kill_switch or env_enable is not None or forced:
        source = "env"
    elif framework_source:
        source = framework_source
    else:
        source = "default-deny"

    return _Config(
        kill_switch, MappingProxyType(disabled), forced, enabled, source, generation
    )


def _swap_config_locked() -> _Config:
    """Build a new snapshot and swap the reference. Caller holds the config lock."""

    global _config_snapshot, _eligible_cache
    current = _config_snapshot
    generation = 1 if current is None else current.generation
    candidate = _build_config(_framework_default, generation)
    if current is not None:
        if candidate._settings() == current._settings():
            # Nothing changed: bumping would flush the eligible cache, spend a
            # route-change scan per op, and churn provider resources for nothing.
            return current
        candidate = replace(candidate, generation=current.generation + 1)

    _config_snapshot = candidate
    _eligible_cache = {}
    return candidate


def _config() -> _Config:
    """The current snapshot, built on first dispatch. Lock-free on the hot path."""

    snapshot = _config_snapshot
    if snapshot is None:
        with _config_lock:
            snapshot = _swap_config_locked()
    return snapshot


def route(*, op: str, version: int) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Route an AITER op through a compatible provider before its native path."""

    operation = Op(op, version)

    def decorate(native: Callable[_P, _R]) -> Callable[_P, _R]:
        canonicalize = _build_canonicalizer(signature(native))

        @wraps(native)
        def routed(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return _dispatch(operation, canonicalize, native, args, kwargs)

        routed.__aiter_plugin_op__ = operation
        return routed

    return decorate


def _dispatch(
    op: Op,
    canonicalize: _Canonicalizer,
    native: Callable[_P, _R],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> _R:
    config = _config()
    if config.kill_switch:
        # The feature is off, so there is nothing to observe and no Decision is
        # recorded. A per-op disable below still records one: that op was routed
        # and deliberately turned off, which is what an operator is watching.
        return native(*args, **kwargs)
    disabled_by = config.disabled_ops.get(op.name)
    if disabled_by is not None:
        return _run_native(op, native, args, kwargs, "op-disabled", disabled_by)

    eligible = _eligible(op, config)
    if not eligible:
        if config.forced_provider:
            reason = "forced-provider-unavailable"
        elif _providers() and not config.enabled:
            reason = "provider-not-enabled"
        else:
            reason = "no-provider"
        return _run_native(op, native, args, kwargs, reason, config.source)

    provider_args, provider_kwargs = canonicalize(*args, **kwargs)
    readonly_kwargs = MappingProxyType(provider_kwargs)
    declined = False
    failure: tuple[str, str] | None = None
    for record in eligible:
        provider = record.provider
        candidate, rejection = _ask_provider(
            provider, op, provider_args, readonly_kwargs
        )
        if rejection is not None:
            # Isolation: this provider is unusable for this call, but it may not
            # deny service to a well-behaved one. The first error is the recorded
            # one; either kind outranks a decline.
            failure = failure or (rejection, provider.name)
            continue
        if candidate is None:
            declined = True
            continue

        _last_decision.set(
            Decision(
                op=op,
                route="provider",
                reason="selected",
                provider=candidate.provider,
                kernel_id=candidate.kernel_id,
                artifact_digest=candidate.artifact_digest,
                qualification=candidate.qualification,
                config_source=config.source,
            )
        )
        # Once provider execution starts, errors propagate. Falling back here could
        # repeat side effects or hide a broken kernel behind a successful result.
        return cast(_R, candidate.run(*provider_args, **provider_kwargs))

    if failure is not None:
        reason, provider_name = failure
        return _run_native(
            op, native, args, kwargs, reason, config.source, provider_name
        )
    reason = "provider-declined" if declined else "no-provider"
    return _run_native(op, native, args, kwargs, reason, config.source)


def _ask_provider(
    provider: KernelProvider,
    op: Op,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[KernelCandidate | None, str | None]:
    """Ask one provider whether it serves this call.

    Returns the candidate, ``(None, None)`` for a decline, or ``(None, reason)``
    when the provider misbehaved before execution. Both kinds of misbehaviour --
    an exception and a candidate that is not one, or is owned by somebody else --
    mean the same thing to the caller, and are never silent.
    """

    try:
        candidate = provider.select(op, args, kwargs)
    except Exception as error:  # noqa: BLE001 - providers are an isolation boundary
        logger.warning(
            "Provider %s selection failed for %s: %s", provider.name, op, error
        )
        return None, "provider-select-error"

    if candidate is None:
        return None, None
    if (
        not isinstance(candidate, KernelCandidate)
        or candidate.provider != provider.name
    ):
        logger.warning(
            "Provider %s returned an invalid candidate (owned by %s)",
            provider.name,
            getattr(candidate, "provider", type(candidate).__name__),
        )
        return None, "invalid-candidate"
    return candidate, None


def _build_canonicalizer(contract: Signature) -> _Canonicalizer:
    """Build this op's argument canonicalizer once, at decoration time.

    ``Signature.bind`` is generic: it re-derives the parameter mapping on every
    call, which measures 4.6-6.0us for the signatures AITER routes. The
    signature is already known here, so the mapping is generated once instead.
    The generated function produces exactly what ``bind()`` + ``apply_defaults()``
    produce, and raises the same ``TypeError`` on a malformed call.
    """

    source = _canonicalizer_source(contract)
    if source is None:
        return _binding_canonicalizer(contract)

    # Defaults travel through the namespace, never through source text: a
    # default like ``dtypes.fp32`` has no parseable repr.
    namespace: dict[str, Any] = {
        f"{_DEFAULT_PREFIX}{name}": parameter.default
        for name, parameter in contract.parameters.items()
        if parameter.default is not parameter.empty
    }
    exec(source, namespace)  # noqa: S102 - source is generated from a known signature
    return cast(_Canonicalizer, namespace[_CANONICALIZER])


def _canonicalizer_source(contract: Signature) -> str | None:
    """Source for this signature, or ``None`` when it must use ``Signature.bind``."""

    positional_only, positional, keyword_only = [], [], []
    for parameter in contract.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            return None  # *args/**kwargs: nothing stable to generate against
        if parameter.name.startswith(_DEFAULT_PREFIX):
            return None  # would collide with an injected default
        if parameter.kind is parameter.POSITIONAL_ONLY:
            positional_only.append(parameter)
        elif parameter.kind is parameter.KEYWORD_ONLY:
            keyword_only.append(parameter)
        else:
            positional.append(parameter)

    declarations = [_declaration(p) for p in positional_only]
    if positional_only:
        declarations.append("/")
    declarations.extend(_declaration(p) for p in positional)
    if keyword_only:
        declarations.append("*")
        declarations.extend(_declaration(p) for p in keyword_only)

    # Mirrors BoundArguments: positional parameters become args, keyword-only
    # parameters become kwargs.
    by_position = positional_only + positional
    args = ", ".join(p.name for p in by_position) + (
        "," if len(by_position) == 1 else ""
    )
    kwargs = ", ".join(f"{p.name!r}: {p.name}" for p in keyword_only)
    return (
        f"def {_CANONICALIZER}({', '.join(declarations)}):\n"
        f"    return ({args}), {{{kwargs}}}\n"
    )


def _declaration(parameter: Parameter) -> str:
    if parameter.default is parameter.empty:
        return parameter.name
    return f"{parameter.name}={_DEFAULT_PREFIX}{parameter.name}"


def _binding_canonicalizer(contract: Signature) -> _Canonicalizer:
    """Fallback for signatures a generated binder cannot describe."""

    def canonicalize(
        *args: Any, **kwargs: Any
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        bound = contract.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound.args, bound.kwargs

    return canonicalize


def _run_native(
    op: Op,
    native: Callable[_P, _R],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    reason: str,
    config_source: str,
    provider: str | None = None,
) -> _R:
    _last_decision.set(
        Decision(
            op=op,
            route="native",
            reason=reason,
            provider=provider,
            config_source=config_source,
        )
    )
    return native(*args, **kwargs)


def _eligible(op: Op, config: _Config) -> tuple[_ProviderRecord, ...]:
    """Cached per ``(op, config generation)``.

    A build race is benign: both racers compute the same value from the same
    immutable snapshot.
    """

    key = (op.name, op.version, config.generation)
    cached = _eligible_cache.get(key)
    if cached is None:
        cached = _eligible_cache[key] = _build_eligible(op, config)
    return cached


def _build_eligible(op: Op, config: _Config) -> tuple[_ProviderRecord, ...]:
    """Providers that may serve this op, in preference order."""

    if config.forced_provider:
        return tuple(
            record
            for record in _providers()
            if record.provider.name == config.forced_provider
            and _supports(record.provider, op)
        )

    # Walk the enable list, not the provider names: the enable list order is the
    # preference order, and entry-point order is environment-dependent.
    by_name = _records_by_provider_name()
    return tuple(
        record
        for name in config.enabled
        for record in by_name.get(name, ())
        if _supports(record.provider, op)
    )


def _records_by_provider_name() -> dict[str, list[_ProviderRecord]]:
    grouped: dict[str, list[_ProviderRecord]] = {}
    for record in _providers():
        grouped.setdefault(record.provider.name, []).append(record)
    return grouped


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
    global _provider_records, _config_snapshot, _eligible_cache, _framework_default
    with _provider_lock:
        _provider_records = None
    with _config_lock:
        _config_snapshot = None
        _eligible_cache = {}
        _framework_default = None
    _last_decision.set(None)


__all__ = [
    "API_VERSION",
    "Decision",
    "KernelCandidate",
    "KernelProvider",
    "Op",
    "configure",
    "last_decision",
    "reload_config",
    "route",
]
