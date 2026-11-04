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
from collections import Counter
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
# Provider API versions still accepted after a bump, each naming the release that
# removes it. The window exists so adapting is never an emergency; the runtime
# warning always names the removal point, so nobody learns it from a crash.
SUPERSEDED_API_VERSIONS: Mapping[int, str] = MappingProxyType({})
# The single-version ``api_version`` spelling, and the release that stops
# reading it. Providers declare ``api_versions``.
SUPERSEDED_API_SPELLING: str | None = "a later aiter release"
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
    op_versions: Mapping[str, Collection[int]]

    # Declare ``api_versions`` -- every provider API version this provider
    # supports -- so one wheel can cover several AITER releases. ``api_version``
    # is the single-version spelling the first prototype used; it is read as a
    # one-element set while it is still accepted (``SUPERSEDED_API_SPELLING``).
    api_versions: Collection[int]
    api_version: int

    # The interface also describes an architecture, dtype, layout and shape
    # coverage declaration. It is deliberately not implemented here: no form is
    # defined for it yet, and a declaration surface is public contract -- harder
    # to undo than to add. Coverage is answered per call by select() until the
    # form is settled.

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
    # the provider API version this provider and this AITER agreed on
    api_version: int


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


@dataclass(frozen=True, slots=True)
class _RoutedOp:
    """What ``route()`` registered, so routed ops are enumerable by name."""

    op: Op
    contract: Signature
    native: Callable[..., Any]
    canonicalize: _Canonicalizer
    # older op_version -> the release that stops accepting it
    superseded: Mapping[int, str] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class _SkippedProvider:
    entry_point: str
    reason: str


@dataclass(frozen=True, slots=True)
class Explanation:
    """What ``explain()`` found without executing anything."""

    decision: Decision
    declines: Mapping[str, str]


_routed_ops: dict[str, _RoutedOp] = {}
_decision_counts: Counter[tuple[str, str, str, str | None]] = Counter()
_decision_callbacks: list[Callable[[Decision], None]] = []
_silenced_callbacks: set[int] = set()
_warned: set[tuple[Any, ...]] = set()
_callback_errors = 0
_route_log: dict[str, tuple[int, tuple[str, str | None, str | None]]] = {}
_skipped_providers: tuple[_SkippedProvider, ...] = ()
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


def route(
    *,
    op: str,
    version: int,
    supersedes: Mapping[int, str] = MappingProxyType({}),
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Route an AITER op through a compatible provider before its native path.

    ``supersedes`` maps an older ``op_version`` still accepted for this contract
    to the release that stops accepting it. AITER owns the contract, so only
    AITER can say that a provider written against the older version is still
    acceptable for calls on this one; declaring it here makes that assertion
    explicit and dated rather than implied.
    """

    operation = Op(op, version)
    accepted = MappingProxyType(dict(supersedes))
    if version in accepted:
        raise ValueError(f"op {op!r} cannot supersede its own version {version}")

    def decorate(native: Callable[_P, _R]) -> Callable[_P, _R]:
        contract = signature(native)
        canonicalize = _build_canonicalizer(contract)
        # Two routes claiming one op name make the contract ambiguous and would
        # make routing depend on import order, which is exactly what the enable
        # list exists to avoid.
        if op in _routed_ops:
            raise ValueError(f"op {op!r} is already routed")
        _routed_ops[op] = _RoutedOp(operation, contract, native, canonicalize, accepted)

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
        return _run_native(
            op, native, args, kwargs, "op-disabled", disabled_by, config.generation
        )

    eligible = _eligible(op, config)
    if not eligible:
        if config.forced_provider:
            reason = "forced-provider-unavailable"
        elif _providers() and not config.enabled:
            reason = "provider-not-enabled"
        else:
            reason = "no-provider"
        return _run_native(
            op, native, args, kwargs, reason, config.source, config.generation
        )

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

        _record(
            Decision(
                op=op,
                route="provider",
                reason="selected",
                provider=candidate.provider,
                kernel_id=candidate.kernel_id,
                artifact_digest=candidate.artifact_digest,
                qualification=candidate.qualification,
                config_source=config.source,
            ),
            config.generation,
            candidate.kernel_id,
        )
        # Once provider execution starts, errors propagate. Falling back here could
        # repeat side effects or hide a broken kernel behind a successful result.
        return cast(_R, candidate.run(*provider_args, **provider_kwargs))

    if failure is not None:
        reason, provider_name = failure
        return _run_native(
            op,
            native,
            args,
            kwargs,
            reason,
            config.source,
            config.generation,
            provider_name,
        )
    reason = "provider-declined" if declined else "no-provider"
    return _run_native(
        op, native, args, kwargs, reason, config.source, config.generation
    )


def _record(
    decision: Decision,
    generation: int,
    kernel_id: str | None = None,
) -> None:
    """The one place a Decision is published. Four readers, one record."""

    _last_decision.set(decision)
    _decision_counts[
        (decision.op.name, decision.route, decision.reason, decision.provider)
    ] += 1
    _log_route_change(decision, generation, kernel_id)
    for callback in _decision_callbacks:
        try:
            callback(decision)
        except Exception as error:  # noqa: BLE001 - telemetry must never break a launch
            _note_callback_error(callback, error)


def _log_route_change(
    decision: Decision, generation: int, kernel_id: str | None
) -> None:
    """Log a route change once per generation, not once per call.

    Within one generation the route is a deterministic function of the call
    arguments; call-to-call variation is provider behaviour, visible in
    ``decision_counts()``, and is not an audit event.
    """

    previous = _route_log.get(decision.op.name)
    if previous is not None and previous[0] == generation:
        return
    current = (decision.route, decision.provider, kernel_id)
    if previous is not None and previous[1] != current:
        logger.info(
            "Route for %s changed at config generation %s: %s -> %s",
            decision.op.name,
            generation,
            previous[1],
            current,
        )
    _route_log[decision.op.name] = (generation, current)


def _note_callback_error(callback: Callable[[Decision], None], error: object) -> None:
    global _callback_errors
    _callback_errors += 1
    if id(callback) not in _silenced_callbacks:
        _silenced_callbacks.add(id(callback))
        logger.warning(
            "Decision callback %r failed and will not be reported again: %s",
            getattr(callback, "__name__", callback),
            error,
        )


def on_decision(callback: Callable[[Decision], None]) -> None:
    """Call ``callback`` inline for every decision, native routes included.

    Callback exceptions are swallowed, logged once per callback, and counted in
    ``status()``: a telemetry bug must never break a kernel launch. The callback
    budget is microseconds; do your own queueing for I/O.
    """

    _decision_callbacks.append(callback)


def decision_counts() -> Counter[tuple[str, str, str, str | None]]:
    """Decisions so far, keyed ``(op, route, reason, provider)``.

    Decisions, not launches: a CUDA graph replay runs no Python, so a captured
    op counts once for the life of the graph however often it replays. Do not
    read these as a call mix.
    """

    return Counter(_decision_counts)


def status() -> Mapping[str, Any]:
    """The configuration snapshot as data: what is installed, enabled, routable."""

    config = _config()
    return MappingProxyType(
        {
            "api_version": API_VERSION,
            "generation": config.generation,
            "config_source": config.source,
            "kill_switch": config.kill_switch,
            "enabled": config.enabled,
            "forced_provider": config.forced_provider,
            "disabled_ops": dict(config.disabled_ops),
            "providers": tuple(
                {
                    "name": record.provider.name,
                    "entry_point": record.entry_point,
                    "op_versions": dict(record.provider.op_versions),
                }
                for record in _providers()
            ),
            "skipped_providers": tuple(
                {"entry_point": skipped.entry_point, "reason": skipped.reason}
                for skipped in _skipped_providers
            ),
            "routed_ops": {
                name: {
                    "version": routed.op.version,
                    "eligible": tuple(
                        record.provider.name for record in _eligible(routed.op, config)
                    ),
                }
                for name, routed in _routed_ops.items()
            },
            "api_versions_accepted": (API_VERSION, *sorted(SUPERSEDED_API_VERSIONS)),
            "deprecations": tuple(
                sorted(
                    [
                        {
                            "kind": "provider-api",
                            "subject": record.provider.name,
                            "version": record.api_version,
                            "removed_in": SUPERSEDED_API_VERSIONS[record.api_version],
                        }
                        for record in _providers()
                        if record.api_version in SUPERSEDED_API_VERSIONS
                    ]
                    + [
                        {
                            "kind": "op-version",
                            "subject": name,
                            "version": version,
                            "removed_in": removal,
                        }
                        for name, routed in _routed_ops.items()
                        for version, removal in routed.superseded.items()
                    ],
                    key=lambda item: (item["kind"], item["subject"], item["version"]),
                )
            ),
            "decision_callback_errors": _callback_errors,
        }
    )


def explain(op: str, *args: Any, **kwargs: Any) -> Explanation:
    """Dry-run dispatch: gates, eligibility, selection -- but never ``run()``.

    Returns the Decision this call would produce and why each provider declined.
    Relies on ``select()`` being pure, which is already a provider obligation.
    """

    routed = _routed_ops.get(op)
    if routed is None:
        raise KeyError(f"op {op!r} is not routed")

    config = _config()
    operation = routed.op
    if config.kill_switch:
        return _explained(operation, "plugins-disabled", "env")
    disabled_by = config.disabled_ops.get(op)
    if disabled_by is not None:
        return _explained(operation, "op-disabled", disabled_by)

    eligible = _eligible(operation, config)
    if not eligible:
        if config.forced_provider:
            reason = "forced-provider-unavailable"
        elif _providers() and not config.enabled:
            reason = "provider-not-enabled"
        else:
            reason = "no-provider"
        return _explained(operation, reason, config.source)

    canonical_args, canonical_kwargs = routed.canonicalize(*args, **kwargs)
    readonly = MappingProxyType(canonical_kwargs)
    declines: dict[str, str] = {}
    for record in eligible:
        provider = record.provider
        candidate, rejection = _ask_provider(
            provider, operation, canonical_args, readonly
        )
        if rejection is not None:
            declines[provider.name] = rejection
            continue
        if candidate is None:
            declines[provider.name] = "declined"
            continue
        return Explanation(
            Decision(
                op=operation,
                route="provider",
                reason="selected",
                provider=candidate.provider,
                kernel_id=candidate.kernel_id,
                artifact_digest=candidate.artifact_digest,
                qualification=candidate.qualification,
                config_source=config.source,
            ),
            MappingProxyType(declines),
        )

    errored = next((r for r in declines.values() if r != "declined"), None)
    reason = errored or "provider-declined"
    return _explained(operation, reason, config.source, declines)


def _explained(
    op: Op,
    reason: str,
    config_source: str,
    declines: Mapping[str, str] | None = None,
) -> Explanation:
    return Explanation(
        Decision(op=op, route="native", reason=reason, config_source=config_source),
        MappingProxyType(dict(declines or {})),
    )


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
        # Roughly sixty times the per-call cost, paid silently otherwise.
        logger.info(
            "Routed op with signature %s cannot use a generated binder and will "
            "canonicalize through Signature.bind on every call; give it explicit "
            "parameters to avoid that.",
            contract,
        )
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
    generation: int,
    provider: str | None = None,
) -> _R:
    _record(
        Decision(
            op=op,
            route="native",
            reason=reason,
            provider=provider,
            config_source=config_source,
        ),
        generation,
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
    return _matched_version(provider, op) is not None


def _matched_version(provider: KernelProvider, op: Op) -> int | None:
    """The op version this provider serves ``op`` on, or ``None``.

    The contract version wins. An older version is matched only while this
    contract still accepts it, and matching one warns once, naming the release
    that stops accepting it.
    """

    try:
        declared = frozenset(provider.op_versions.get(op.name, ()))
    except Exception as error:  # noqa: BLE001 - providers are an isolation boundary
        logger.warning("Provider %s has invalid op_versions: %s", provider.name, error)
        return None

    if op.version in declared:
        return op.version

    routed = _routed_ops.get(op.name)
    accepted = routed.superseded if routed is not None else {}
    for version, removal in accepted.items():
        if version in declared:
            _warn_once(
                ("op", provider.name, op.name, version),
                "Provider %s serves %s on superseded op version %s; %s stops "
                "accepting it. Re-validate against version %s.",
                provider.name,
                op.name,
                version,
                removal,
                op.version,
            )
            return version
    return None


def _warn_once(key: tuple[Any, ...], message: str, *args: Any) -> None:
    """Warn about a deprecation once per subject, not once per call."""

    if key in _warned:
        return
    _warned.add(key)
    logger.warning(message, *args)


def _providers() -> Sequence[_ProviderRecord]:
    global _provider_records
    if _provider_records is None:
        with _provider_lock:
            if _provider_records is None:
                _provider_records = _discover_providers()
    return _provider_records


def _accepted_api_versions() -> frozenset[int]:
    return frozenset({API_VERSION, *SUPERSEDED_API_VERSIONS})


def _declared_api_versions(provider: KernelProvider) -> frozenset[int]:
    """Every provider API version this provider says it supports.

    ``api_versions`` is the spelling the interface asks for. A provider that
    only has the prototype's single ``api_version`` is read as declaring that
    one version, and warned once.
    """

    declared = getattr(provider, "api_versions", None)
    if declared is not None:
        return frozenset(declared)
    single = provider.api_version
    if SUPERSEDED_API_SPELLING is not None:
        _warn_once(
            ("spelling", provider.name),
            "Provider %s declares api_version=%s; declare api_versions instead, "
            "so one release can support several. %s stops reading the single "
            "spelling.",
            provider.name,
            single,
            SUPERSEDED_API_SPELLING,
        )
    return frozenset({single})


def _discover_providers() -> tuple[_ProviderRecord, ...]:
    global _skipped_providers
    records: list[_ProviderRecord] = []
    skipped: list[_SkippedProvider] = []
    for entry_point in sorted(_entry_points(), key=lambda item: item.name):
        try:
            loaded = entry_point.load()
            provider = loaded if hasattr(loaded, "select") else loaded()
            name = provider.name
            _ = provider.op_versions
            declared = _declared_api_versions(provider)
            accepted = _accepted_api_versions()
            usable = declared & accepted
            if not usable:
                reason = (
                    f"declares API versions {sorted(declared)}; "
                    f"this aiter accepts {sorted(accepted)}"
                )
                logger.warning("Skipping provider %s: %s", name, reason)
                skipped.append(_SkippedProvider(entry_point.name, reason))
                continue

            # Both sides may support several; the newest shared one is used.
            matched = max(usable)
            removal = SUPERSEDED_API_VERSIONS.get(matched)
            if removal is not None:
                # Inside the window: it loads, but nobody should learn the
                # removal release from a future crash.
                _warn_once(
                    ("api", name, matched),
                    "Provider %s is served on superseded provider API version "
                    "%s; %s removes it. Support version %s.",
                    name,
                    matched,
                    removal,
                    API_VERSION,
                )
            records.append(_ProviderRecord(entry_point.name, provider, matched))
        except Exception as error:  # noqa: BLE001 - one plugin must not break AITER
            logger.warning(
                "Skipping provider entry point %s: %s", entry_point.name, error
            )
            # status() reports what was skipped and why: an installed provider
            # that never serves is the question an operator asks first.
            skipped.append(_SkippedProvider(entry_point.name, str(error)))
    ordered = sorted(records, key=lambda item: (item.provider.name, item.entry_point))
    kept: list[_ProviderRecord] = []
    claimed: dict[str, str] = {}
    for record in ordered:
        owner = claimed.get(record.provider.name)
        if owner is not None:
            # A provider name is an identity: AITER_PLUGIN_ENABLE and
            # AITER_PLUGIN_FORCE address it, and a Decision names it. Two
            # providers answering to one name would make every one of those
            # ambiguous and would let install order decide which serves.
            # Skipped rather than raised: a bad install must never break
            # ``import aiter``.
            reason = f"name {record.provider.name!r} is already used by entry point {owner!r}"
            logger.warning(
                "Skipping provider entry point %s: %s", record.entry_point, reason
            )
            skipped.append(_SkippedProvider(record.entry_point, reason))
            continue
        claimed[record.provider.name] = record.entry_point
        kept.append(record)

    _skipped_providers = tuple(skipped)
    return tuple(kept)


def _entry_points() -> Sequence[metadata.EntryPoint]:
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        return tuple(entry_points.select(group=ENTRY_POINT_GROUP))
    return tuple(entry_points.get(ENTRY_POINT_GROUP, ()))  # type: ignore[union-attr]


def _reset_for_testing() -> None:
    global _provider_records, _config_snapshot, _eligible_cache, _framework_default
    global _skipped_providers, _callback_errors
    _routed_ops.clear()
    _decision_counts.clear()
    _decision_callbacks.clear()
    _silenced_callbacks.clear()
    _warned.clear()
    _route_log.clear()
    _callback_errors = 0
    with _provider_lock:
        _provider_records = None
        _skipped_providers = ()
    with _config_lock:
        _config_snapshot = None
        _eligible_cache = {}
        _framework_default = None
    _last_decision.set(None)


__all__ = [
    "API_VERSION",
    "SUPERSEDED_API_SPELLING",
    "SUPERSEDED_API_VERSIONS",
    "Decision",
    "Explanation",
    "KernelCandidate",
    "KernelProvider",
    "Op",
    "configure",
    "decision_counts",
    "explain",
    "last_decision",
    "on_decision",
    "reload_config",
    "route",
    "status",
]
