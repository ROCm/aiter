# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Experimental interface for out-of-tree AITER kernel providers.

Ownership boundary:
- AITER owns the op contract (signatures, versions), the execution rules
  (isolation, fallback, no-fallback-after-run), and observable provenance.
- Providers own their kernels, their qualification evidence, and their
  lifecycle; ``KernelCandidate.qualification`` is self-reported provenance,
  not an AITER certification. A kernel plugin is the unit that ships and
  registers; ``KernelProvider`` is the interface it implements, the way a
  pytest plugin implements hooks the host calls.
- Frameworks own enablement: installing a provider makes it available, never
  active. A provider serves only when a framework names it in
  ``configure(enable=...)``, which records the source of that decision.
  ``AITER_PLUGINS=0`` is the one environment control, and it only turns
  everything off: an operator can stop routing without editing framework code,
  and nothing can start it from the environment.
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
from threading import RLock
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
# Every kernel plugin registers here, and many are expected to: AITER is a
# kernel library, so a plugin to it is a kernel plugin, and the group is not
# qualified by a kind that has no second member. The plugin is the unit that
# ships; what it registers is an object implementing KernelProvider, which is
# the role this module asks, the way a pytest plugin implements hooks.
ENTRY_POINT_GROUP = "aiter.plugins"

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
    # Optional: the interface says AITER *may* record a provider-supplied
    # evidence reference, not that a provider must have one to serve.
    qualification: str | None = None
    run: Callable[..., Any] = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        identity = (self.provider, self.kernel_id, self.artifact_digest)
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
    # Either a mapping of op name to the contract versions served, or just the
    # op names, which means whichever version is current. Most providers have
    # nothing to say about versions and should not have to say it; the ones that
    # span a contract change say so explicitly.
    op_versions: Mapping[str, Collection[int]] | Collection[str]

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
    # Read once, at discovery. A provider may be a proxy whose attributes start
    # throwing when its backend goes away; reading the name again inside an
    # except block would let that exception escape the isolation boundary.
    name: str


@dataclass(frozen=True, slots=True)
class _FrameworkDefault:
    """What ``configure()`` last set: the config layer below the environment."""

    enable: tuple[str, ...]
    source: str


@dataclass(frozen=True, slots=True)
class _Config:
    """An immutable configuration snapshot.

    Dispatch reads one reference per call, so a call always runs against exactly
    one generation. ``enabled`` is ordered: the enable list order is the
    preference order, and it is the only order source.
    """

    kill_switch: bool
    enabled: tuple[str, ...]
    source: str
    generation: int

    def _settings(self) -> tuple[Any, ...]:
        """Everything that defines this configuration except its generation."""

        return (self.kill_switch, self.enabled, self.source)


@dataclass(frozen=True, slots=True)
class _Eligibility:
    """Who may serve one op at one config generation, and why the rest may not.

    ``reason`` is what a native route records when ``providers`` is empty; it
    is computed here because everything it depends on is already in hand.
    """

    providers: tuple[_ProviderRecord, ...]
    # provider name -> why it cannot serve this op, for enabled providers only
    ineligible: Mapping[str, str]
    reason: str


@dataclass(frozen=True, slots=True)
class _RoutedOp:
    """What ``route()`` registered, so routed ops are enumerable by name."""

    op: Op
    contract: Signature
    canonicalize: _Canonicalizer
    # older op_version -> the release that stops accepting it.
    #
    # default_factory, not a plain default: a mappingproxy is unhashable before
    # 3.12, and 3.11's dataclasses reject any unhashable default outright, so
    # the plain spelling makes `import aiter` raise ValueError on 3.11 -- which
    # setup.py supports and no CI leg runs. 3.10 accepts it (it only rejects
    # list/dict/set by type) and 3.12 accepts it (mappingproxy gained __hash__),
    # so the two Pythons CI does run are exactly the two that hide it.
    superseded: Mapping[int, str] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class _SkippedProvider:
    entry_point: str
    reason: str


@dataclass(frozen=True, slots=True)
class Explanation:
    """What ``explain()`` found without executing anything.

    ``decision`` is ``None`` when the call would not produce one at all, which
    today means the kill switch is on: dispatch goes straight to native and
    records nothing, so there is no would-be Decision to report.
    """

    decision: Decision | None
    declines: Mapping[str, str]


_routed_ops: dict[str, _RoutedOp] = {}
# Ops routed by importing aiter belong to the process, not to whichever test
# runs first; see _reset_for_testing.
_imported_routes: dict[str, _RoutedOp] | None = None
_decision_counts: Counter[tuple[str, str, str, str | None]] = Counter()
_decision_callbacks: list[Callable[[Decision], None]] = []
_silenced_callbacks: set[int] = set()
_warned: set[tuple[Any, ...]] = set()
# (id(run), op name) -> the run itself when it takes the contract, False when
# it does not. Holding the callable keeps its id from being reused by another
# object after a collection. See _accepts_contract.
_checked_runs: dict[tuple[int, str], Any] = {}
_PROBE = object()
_callback_errors = 0
_route_log: dict[str, tuple[int, tuple[str, str | None, str | None]]] = {}
# What an op was doing before anything was recorded for it. See _log_route_change.
_UNROUTED = ("native", None, None)
_skipped_providers: tuple[_SkippedProvider, ...] = ()
_provider_records: tuple[_ProviderRecord, ...] | None = None
# Reentrant, and discovery runs outside it: both guard against a provider
# that touches aiter while importing.
_provider_lock = RLock()
_config_snapshot: _Config | None = None
_config_lock = RLock()
_framework_default: _FrameworkDefault | None = None
_eligible_cache: dict[tuple[str, int, int], _Eligibility] = {}
_FALSY = {"0", "false", "no", "off"}
_CANONICALIZER = "_canonicalize"
_DEFAULT_PREFIX = "__aiter_default_"
_last_decision: ContextVar[Decision | None] = ContextVar(
    "aiter_plugin_last_decision", default=None
)


def last_decision() -> Decision | None:
    """The last routing decision in this execution context, or ``None``.

    ``None`` whenever no provider can serve -- under ``AITER_PLUGINS=0`` and
    while no provider is enabled. Dispatch records nothing in either state, and
    whatever was recorded before it was entered describes a world that no longer
    holds. Answering with it would be worse than answering with nothing, because
    nothing about it says it is stale: a framework that calls
    ``configure(enable=())`` to turn routing off would still read back the
    provider route from before the call.

    The check is here rather than on the dispatch path on purpose. Both states
    exist to cost nothing, and clearing the record on every call would tax a few
    million of them to make a handful of reads correct; this way the read pays,
    and reads are rare.
    """

    snapshot = _config_snapshot
    if snapshot is not None and (snapshot.kill_switch or not snapshot.enabled):
        return None
    return _last_decision.get()


def configure(*, source: str, enable: Sequence[str] = ()) -> None:
    """Set which providers may serve. This is the only way to enable one.

    ``enable`` is a provider list in preference order and **replaces** any
    previous ``configure()`` state. ``source`` is required provenance ("vllm
    rocm defaults 2026.11") and is recorded so every route names an owner --
    the reason enablement lives here rather than in the environment, which
    cannot say who set it.

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


def _build_config(default: _FrameworkDefault | None, generation: int) -> _Config:
    """The snapshot dispatch reads: who may serve, and who decided that.

    Enablement has one source, ``configure()``, so the framework that turned a
    provider on is always named. The environment can only take enablement
    away, never grant it: a provider wheel arriving as somebody's transitive
    dependency must not be able to change which kernels run.
    """

    kill_switch = os.getenv("AITER_PLUGINS", "1").strip().lower() in _FALSY
    enabled = default.enable if default else ()
    if kill_switch:
        source = "env"
    elif default:
        source = f"configure:{default.source}"
    else:
        source = "default-deny"
    return _Config(kill_switch, enabled, source, generation)


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
    version: int = 1,
    supersedes: Mapping[int, str] = MappingProxyType({}),
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Route an AITER op through a compatible provider before its native path.

    ``version`` defaults to 1 and is worth writing only once it has moved: an
    op whose contract has never changed says nothing about versions, and the
    number appears exactly where it carries information.

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
        _routed_ops[op] = _RoutedOp(operation, contract, canonicalize, accepted)

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
    if config.kill_switch or not config.enabled:
        # Two ways of saying the same thing: no provider can possibly serve this
        # call. The kill switch says so for the process; an empty enable list
        # says so because no framework has called configure(). Neither can
        # produce a provider route, so dispatch spends nothing deciding it --
        # and nothing recording the decision, which is the expensive half: a
        # frozen Decision costs more to build than the rest of the path.
        #
        # The cost of this shortcut is that these calls record no Decision, so
        # last_decision() and decision_counts() stay empty. That is the same
        # trade the kill switch already makes, and it is the right one here:
        # on the day this merges every caller is in this state, so the whole
        # cost would be paid by people who get nothing back for it. What an
        # operator actually asks -- is anything enabled, who is installed, what
        # would this call do -- is still answered on demand by status() and
        # explain().
        return native(*args, **kwargs)

    eligible = _eligible(op, config)
    if not eligible.providers:
        return _run_native(
            op, native, args, kwargs, eligible.reason, config.source, config.generation
        )

    provider_args, provider_kwargs = canonicalize(*args, **kwargs)
    readonly_kwargs = MappingProxyType(provider_kwargs)
    declined = False
    failure: tuple[str, str] | None = None
    for record in eligible.providers:
        candidate, rejection = _ask_provider(record, op, provider_args, readonly_kwargs)
        if rejection is not None:
            # Isolation: this provider is unusable for this call, but it may not
            # deny service to a well-behaved one. The first error is the recorded
            # one; either kind outranks a decline.
            failure = failure or (rejection, record.name)
            continue
        if candidate is None:
            declined = True
            continue

        _record(
            _selected(op, candidate, config), config.generation, candidate.kernel_id
        )
        # Once provider execution starts, errors propagate. Falling back here could
        # repeat side effects or hide a broken kernel behind a successful result.
        return cast(_R, candidate.run(*provider_args, **provider_kwargs))

    reason, provider_name = _native_outcome(failure, declined)
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


def _selected(op: Op, candidate: KernelCandidate, config: _Config) -> Decision:
    """The Decision a selected candidate produces.

    Dispatch records it and explain returns it without running anything. They
    must not drift: explain exists to say what dispatch would do, so a field
    added on one side and not the other makes it lie.
    """

    return Decision(
        op=op,
        route="provider",
        reason="selected",
        provider=candidate.provider,
        kernel_id=candidate.kernel_id,
        artifact_digest=candidate.artifact_digest,
        qualification=candidate.qualification,
        config_source=config.source,
    )


def _native_outcome(
    failure: tuple[str, str] | None, declined: bool
) -> tuple[str, str | None]:
    """Why the call went native once every eligible provider was asked.

    An error outranks a decline, and the first error is the one named: one
    misbehaving provider is more actionable than the fact that others said no.
    Both callers keep the same bookkeeping so the rule is shared rather than
    mirrored -- it used to be written twice, once as a tuple and once derived
    back out of a dictionary.
    """

    if failure is not None:
        return failure
    return ("provider-declined" if declined else "no-provider"), None


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

    An op with no entry yet was running native: dispatch records nothing until a
    provider is enabled, so ``_UNROUTED`` is what it was doing before, not an
    unknown. Saying so turns the one change an operator most wants to see -- the
    moment a framework takes an op over -- into an audit line instead of a
    silent first baseline. A first decision that is native with nobody named
    matches ``_UNROUTED`` and stays silent; one that names a provider does not,
    and should not -- "aklr was asked and did not serve" is the line an operator
    goes looking for when an enabled provider seems to do nothing.
    """

    previous = _route_log.get(decision.op.name)
    if previous is not None and previous[0] == generation:
        return
    current = (decision.route, decision.provider, kernel_id)
    before = previous[1] if previous is not None else _UNROUTED
    if before != current:
        logger.info(
            "Route for %s changed at config generation %s: %s -> %s",
            decision.op.name,
            generation,
            before,
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
            "providers": tuple(_described(record) for record in _providers()),
            "skipped_providers": tuple(
                {"entry_point": skipped.entry_point, "reason": skipped.reason}
                for skipped in _skipped_providers
            ),
            "routed_ops": {
                name: _described_op(routed, config)
                for name, routed in _routed_ops.items()
            },
            "api_versions_accepted": (API_VERSION, *sorted(SUPERSEDED_API_VERSIONS)),
            "deprecations": tuple(
                sorted(
                    [
                        {
                            "kind": "provider-api",
                            "subject": record.name,
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
        # Mirrors dispatch exactly: it records nothing, so neither do we.
        return Explanation(None, MappingProxyType({}))
    eligible = _eligible(operation, config, warn=False)
    if not eligible.providers:
        return _explained(
            operation, eligible.reason, config.source, eligible.ineligible
        )

    canonical_args, canonical_kwargs = routed.canonicalize(*args, **kwargs)
    readonly = MappingProxyType(canonical_kwargs)
    declines: dict[str, str] = dict(eligible.ineligible)
    failure: tuple[str, str] | None = None
    declined = False
    for record in eligible.providers:
        candidate, rejection = _ask_provider(
            record, operation, canonical_args, readonly
        )
        if rejection is not None:
            declines[record.name] = rejection
            failure = failure or (rejection, record.name)
            continue
        if candidate is None:
            declines[record.name] = "declined"
            declined = True
            continue
        return Explanation(
            _selected(operation, candidate, config), MappingProxyType(declines)
        )

    reason, provider_name = _native_outcome(failure, declined)
    return _explained(operation, reason, config.source, declines, provider_name)


def _described(record: _ProviderRecord) -> Mapping[str, Any]:
    """One provider as data, without trusting it to answer.

    A provider may be a proxy whose attributes start throwing when its backend
    goes away. Dispatch already degrades to native in that case; status() is
    where an operator goes to find out why, so it must not be the one call that
    fails.
    """

    try:
        op_versions: Any = {
            name: sorted(versions)
            for name, versions in record.provider.op_versions.items()
        }
    except Exception as error:  # noqa: BLE001 - providers are an isolation boundary
        op_versions = f"<unavailable: {error}>"
    return MappingProxyType(
        {
            "name": record.name,
            "entry_point": record.entry_point,
            "api_version": record.api_version,
            "op_versions": op_versions,
        }
    )


def _explained(
    op: Op,
    reason: str,
    config_source: str,
    declines: Mapping[str, str] | None = None,
    provider: str | None = None,
) -> Explanation:
    return Explanation(
        Decision(
            op=op,
            route="native",
            reason=reason,
            config_source=config_source,
            provider=provider,
        ),
        MappingProxyType(dict(declines or {})),
    )


def _accepts_contract(run: Callable[..., Any], routed: _RoutedOp) -> bool:
    """Whether ``run`` can take the shape dispatch is about to hand it.

    A declared op version says a provider believes it implements this contract.
    This checks it, which is the cheaper half of the same guarantee: a provider
    built against an older shape is caught here even if it declared the current
    version, or declared no version at all.

    It catches shape, not meaning. A contract whose parameters are unchanged but
    whose semantics moved is exactly what the version number is for, and nothing
    here can see it.

    Checked once per callable and remembered, so the hot path pays a dict
    lookup. A ``*args, **kwargs`` forwarder binds anything and passes, which is
    correct: it genuinely accepts the call.
    """

    positional, keyword_only = [], []
    for parameter in routed.contract.parameters.values():
        if parameter.kind is parameter.KEYWORD_ONLY:
            keyword_only.append(parameter.name)
        elif parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            positional.append(parameter.name)
    try:
        signature(run).bind(
            *([_PROBE] * len(positional)), **{n: _PROBE for n in keyword_only}
        )
        return True
    except TypeError:
        return False
    except Exception:  # noqa: BLE001 - a builtin has no signature to read
        return True  # cannot be checked is not the same as wrong


def _ask_provider(
    record: _ProviderRecord,
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
        candidate = record.provider.select(op, args, kwargs)
    except Exception as error:  # noqa: BLE001 - providers are an isolation boundary
        logger.warning(
            "Provider %s selection failed for %s: %s", record.name, op, error
        )
        return None, "provider-select-error"

    if candidate is None:
        return None, None
    if not isinstance(candidate, KernelCandidate) or candidate.provider != record.name:
        logger.warning(
            "Provider %s returned an invalid candidate (owned by %s)",
            record.name,
            getattr(candidate, "provider", type(candidate).__name__),
        )
        return None, "invalid-candidate"

    # Checked once per (run, op) and remembered; the hot path is one lookup.
    run = candidate.run
    key = (id(run), op.name)
    checked = _checked_runs.get(key)
    if checked is None:
        routed = _routed_ops.get(op.name)
        checked = run if routed is None or _accepts_contract(run, routed) else False
        _checked_runs[key] = checked
    if checked is False:
        # Handing it the call anyway is the failure worth preventing: the
        # argument counts can line up by accident and an input is written as
        # though it were an output, silently.
        logger.warning(
            "Provider %s offered %s for %s, which does not take this contract",
            record.name,
            candidate.kernel_id,
            op.name,
        )
        return None, "contract-mismatch"
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


def _eligible(op: Op, config: _Config, *, warn: bool = True) -> _Eligibility:
    """Cached per ``(op, config generation)``.

    A build race is benign: both racers compute the same value from the same
    immutable snapshot.
    """

    key = (op.name, op.version, config.generation)
    cached = _eligible_cache.get(key)
    if cached is not None:
        return cached
    built = _build_eligible(op, config, warn=warn)
    if warn:
        _eligible_cache[key] = built
    # A quiet build is not cached: caching it would swallow the deprecation
    # warning for every later call, which is worse than spending it here.
    return built


def _build_eligible(op: Op, config: _Config, *, warn: bool = True) -> _Eligibility:
    """Who may serve this op, in preference order, and why the rest may not.

    The exclusions are kept rather than filtered away. A provider that is
    installed and enabled and simply declares the wrong op version used to be
    indistinguishable from nothing being installed at all: both answered
    ``no-provider``, and ``explain()`` -- the read whose whole job is saying
    why -- reported no declines. Built once per ``(op, generation)``, so the
    detail costs the hot path nothing.
    """

    # Walk the enable list, not the provider names: the enable list order is the
    # preference order, and entry-point order is environment-dependent.
    by_name = _records_by_provider_name()
    providers: list[_ProviderRecord] = []
    ineligible: dict[str, str] = {}
    installed_enabled = False
    for name in config.enabled:
        records = by_name.get(name, ())
        if not records:
            ineligible[name] = "enabled but not installed"
            continue
        installed_enabled = True
        for record in records:
            if _supports(record, op, warn=warn):
                providers.append(record)
            else:
                ineligible[record.name] = _unsupported_reason(record, op)

    if providers:
        reason = "selected"
    elif not config.enabled:
        reason = "provider-not-enabled" if _providers() else "no-provider"
    elif installed_enabled:
        reason = "provider-op-unsupported"
    else:
        reason = "no-provider"
    return _Eligibility(tuple(providers), MappingProxyType(ineligible), reason)


def _unsupported_reason(record: _ProviderRecord, op: Op) -> str:
    """Why this installed, enabled provider cannot serve ``op``."""

    try:
        served = record.provider.op_versions
        declared = (
            sorted(served.get(op.name, ()))
            if isinstance(served, Mapping)
            else (op.name in served)
        )
    except Exception as error:  # noqa: BLE001 - providers are an isolation boundary
        return f"op_versions could not be read: {error}"
    if declared is False:
        return f"does not serve {op.name}"
    if declared is True or not declared:
        return f"does not serve {op.name} version {op.version}"
    return (
        f"serves {op.name} version{'s' if len(declared) > 1 else ''} "
        f"{', '.join(map(str, declared))}, not {op.version}"
    )


def _described_op(routed: _RoutedOp, config: _Config) -> dict[str, Any]:
    """One routed op as status() reports it: who may serve, and why the rest may not.

    The ineligible side is the half that is otherwise invisible -- without it an
    enabled provider missing from the list can only be explained by reading its
    op_versions against the contract by eye.
    """

    eligible = _eligible(routed.op, config)
    return {
        "version": routed.op.version,
        "eligible": tuple(record.name for record in eligible.providers),
        "ineligible": dict(eligible.ineligible),
    }


def _records_by_provider_name() -> dict[str, list[_ProviderRecord]]:
    grouped: dict[str, list[_ProviderRecord]] = {}
    for record in _providers():
        grouped.setdefault(record.name, []).append(record)
    return grouped


def _supports(record: _ProviderRecord, op: Op, *, warn: bool = True) -> bool:
    return _matched_version(record, op, warn=warn) is not None


def _matched_version(
    record: _ProviderRecord, op: Op, *, warn: bool = True
) -> int | None:
    """The op version this provider serves ``op`` on, or ``None``.

    The contract version wins. An older version is matched only while this
    contract still accepts it; the newest acceptable one is chosen, the same
    rule discovery uses for API versions. Matching one warns once, naming the
    release that stops accepting it -- unless ``warn`` is off, which is how a
    dry run avoids spending the one-shot warning on an interactive query.
    """

    try:
        served = record.provider.op_versions
        if isinstance(served, Mapping):
            declared = frozenset(served.get(op.name, ()))
        elif op.name in served:
            declared = frozenset({op.version})  # named without a version
        else:
            return None
    except Exception as error:  # noqa: BLE001 - providers are an isolation boundary
        logger.warning("Provider %s has invalid op_versions: %s", record.name, error)
        return None

    if op.version in declared:
        return op.version

    routed = _routed_ops.get(op.name)
    accepted = routed.superseded if routed is not None else {}
    for version in sorted(declared & accepted.keys(), reverse=True):
        if warn:
            _warn_once(
                ("op", record.name, op.name, version),
                "Provider %s serves %s on superseded op version %s; %s stops "
                "accepting it. Re-validate against version %s.",
                record.name,
                op.name,
                version,
                accepted[version],
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
    global _provider_records, _skipped_providers
    records = _provider_records
    if records is not None:
        return records

    # Discovery imports third-party code. It runs outside the lock so that a
    # provider which touches aiter while importing re-enters cleanly instead of
    # deadlocking on a lock this thread already holds. Two threads racing only
    # means the same set is discovered twice; the first to publish wins.
    discovered, skipped = _discover_providers()
    with _provider_lock:
        if _provider_records is None:
            _provider_records = discovered
            _skipped_providers = skipped
        return _provider_records


def _accepted_api_versions() -> frozenset[int]:
    return frozenset({API_VERSION, *SUPERSEDED_API_VERSIONS})


def _declared_api_versions(provider: KernelProvider, name: str) -> frozenset[int]:
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
            ("spelling", name),
            "Provider %s declares api_version=%s; declare api_versions instead, "
            "so one release can support several. %s stops reading the single "
            "spelling.",
            name,
            single,
            SUPERSEDED_API_SPELLING,
        )
    return frozenset({single})


def _discover_providers() -> (
    tuple[tuple[_ProviderRecord, ...], tuple[_SkippedProvider, ...]]
):
    records: list[_ProviderRecord] = []
    skipped: list[_SkippedProvider] = []
    try:
        entry_points = sorted(_entry_points(), key=lambda item: item.name)
    except Exception as error:  # noqa: BLE001 - a broken wheel is not aiter's bug
        # Enumeration reads every installed distribution's metadata, including
        # packages with nothing to do with aiter. One malformed entry_points.txt
        # must not raise into a kernel call -- and must not leave discovery
        # unfinished, or every later call would raise again.
        logger.warning("Provider discovery failed, no provider is available: %s", error)
        return (), (_SkippedProvider("<entry point scan>", str(error)),)

    for entry_point in entry_points:
        try:
            loaded = entry_point.load()
            provider = loaded if hasattr(loaded, "select") else loaded()
            name = provider.name
            _ = provider.op_versions
            declared = _declared_api_versions(provider, name)
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
            records.append(_ProviderRecord(entry_point.name, provider, matched, name))
        except Exception as error:  # noqa: BLE001 - one plugin must not break AITER
            logger.warning(
                "Skipping provider entry point %s: %s", entry_point.name, error
            )
            # status() reports what was skipped and why: an installed provider
            # that never serves is the question an operator asks first.
            skipped.append(_SkippedProvider(entry_point.name, str(error)))
    ordered = sorted(records, key=lambda item: (item.name, item.entry_point))
    kept: list[_ProviderRecord] = []
    claimed: dict[str, str] = {}
    for record in ordered:
        owner = claimed.get(record.name)
        if owner is not None:
            # A provider name is an identity: configure(enable=...) addresses
            # it, and a Decision names it. Two
            # providers answering to one name would make every one of those
            # ambiguous and would let install order decide which serves.
            # Skipped rather than raised: a bad install must never break
            # ``import aiter``.
            reason = f"name {record.name!r} is already used by entry point {owner!r}"
            logger.warning(
                "Skipping provider entry point %s: %s", record.entry_point, reason
            )
            skipped.append(_SkippedProvider(record.entry_point, reason))
            continue
        claimed[record.name] = record.entry_point
        kept.append(record)

    return tuple(kept), tuple(skipped)


def _entry_points() -> Sequence[metadata.EntryPoint]:
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        return tuple(entry_points.select(group=ENTRY_POINT_GROUP))
    return tuple(entry_points.get(ENTRY_POINT_GROUP, ()))  # type: ignore[union-attr]


def _reset_for_testing() -> None:
    global _provider_records, _config_snapshot, _eligible_cache, _framework_default
    global _skipped_providers, _callback_errors, _imported_routes
    if _imported_routes is None:
        # Captured the first time this runs, which is before any test has
        # routed an op of its own: everything here was registered by importing
        # aiter. Clearing those would make status() and explain() report the
        # production ops as unrouted for every test after the first.
        _imported_routes = dict(_routed_ops)
    _routed_ops.clear()
    _routed_ops.update(_imported_routes)
    _decision_counts.clear()
    _decision_callbacks.clear()
    _silenced_callbacks.clear()
    _warned.clear()
    _checked_runs.clear()
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
    "ENTRY_POINT_GROUP",
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
