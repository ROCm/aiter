# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

from aiter import plugin


@dataclass
class _EntryPoint:
    name: str
    target: Any

    def load(self) -> Any:
        if isinstance(self.target, Exception):
            raise self.target
        return self.target


class _Provider:
    def __init__(self, name: str, selected: plugin.KernelCandidate | None = None):
        self.name = name
        self.api_versions = {plugin.API_VERSION}
        self.op_versions = {"test.add": {1}}
        self.selected = selected
        self.select_calls = 0

    def select(self, op, args, kwargs):
        self.select_calls += 1
        return self.selected


@pytest.fixture(autouse=True)
def _clean_plugin_state(monkeypatch):
    for name in (
        "AITER_PLUGINS",
        "AITER_PLUGIN_DISABLE",
        "AITER_PLUGIN_FORCE",
        "AITER_PLUGIN_ENABLE",
    ):
        monkeypatch.delenv(name, raising=False)
    plugin._reset_for_testing()
    yield
    plugin._reset_for_testing()


def _route_add(native_calls: list[tuple[int, int]]):
    @plugin.route(op="test.add", version=1)
    def add(left: int, right: int = 1) -> int:
        native_calls.append((left, right))
        return left + right

    return add


def _candidate(provider: str, result: int = 100) -> plugin.KernelCandidate:
    return plugin.KernelCandidate(
        provider=provider,
        kernel_id="kernel-1",
        artifact_digest="sha256:abc",
        qualification="experimental",
        run=lambda *args, **kwargs: result,
    )


def test_no_provider_uses_native_path_and_discovers_lazily(monkeypatch):
    discovery_calls = 0

    def entry_points():
        nonlocal discovery_calls
        discovery_calls += 1
        return ()

    monkeypatch.setattr(plugin, "_entry_points", entry_points)
    native_calls = []
    add = _route_add(native_calls)

    assert discovery_calls == 0
    assert add(2, right=3) == 5
    assert add(4) == 5
    assert discovery_calls == 1
    assert native_calls == [(2, 3), (4, 1)]
    assert plugin.last_decision() == plugin.Decision(
        op=plugin.Op("test.add", 1),
        route="native",
        reason="no-provider",
        config_source="default-deny",
    )


def test_disabled_plugins_do_not_trigger_discovery(monkeypatch):
    monkeypatch.setenv("AITER_PLUGINS", "0")
    monkeypatch.setattr(
        plugin,
        "_entry_points",
        lambda: pytest.fail("disabled plugins must not be discovered"),
    )
    add = _route_add([])

    assert add(2, 3) == 5
    # The kill switch is a free path: one flag check, straight to native. The
    # feature is off, so there is nothing to observe and nothing is recorded.
    assert plugin.last_decision() is None


@pytest.mark.parametrize("outcome", [None, RuntimeError("selection failed")])
def test_decline_or_select_error_uses_visible_native_fallback(monkeypatch, outcome):
    class Provider(_Provider):
        def select(self, op, args, kwargs):
            if isinstance(outcome, Exception):
                raise outcome

    provider = Provider("aklr")
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )
    native_calls = []
    add = _route_add(native_calls)

    assert add(2, 3) == 5
    assert native_calls == [(2, 3)]
    assert plugin.last_decision().reason == (
        "provider-select-error"
        if isinstance(outcome, Exception)
        else "provider-declined"
    )


def test_selected_candidate_records_provenance(monkeypatch):
    provider = _Provider("aklr", _candidate("aklr"))
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )
    native_calls = []

    assert _route_add(native_calls)(2, 3) == 100
    assert native_calls == []
    assert plugin.last_decision() == plugin.Decision(
        op=plugin.Op("test.add", 1),
        route="provider",
        reason="selected",
        provider="aklr",
        kernel_id="kernel-1",
        artifact_digest="sha256:abc",
        qualification="experimental",
        config_source="env",
    )


def test_aiter_signature_is_enforced_before_provider_selection(monkeypatch):
    provider = _Provider("aklr", _candidate("aklr"))
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _route_add([])(2, unknown=3)

    assert provider.select_calls == 0
    assert plugin.last_decision() is None


def test_run_error_propagates_without_native_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("kernel failed")

    candidate = plugin.KernelCandidate(
        provider="aklr",
        kernel_id="broken-kernel",
        artifact_digest="sha256:def",
        qualification="experimental",
        run=fail,
    )
    provider = _Provider("aklr", candidate)
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )
    native_calls = []

    with pytest.raises(RuntimeError, match="kernel failed"):
        _route_add(native_calls)(2, 3)

    assert native_calls == []
    assert plugin.last_decision().provider == "aklr"
    assert plugin.last_decision().route == "provider"


def test_enable_list_order_is_the_preference_order(monkeypatch):
    """The enable order must beat alphabetical order, or it is not covered."""

    zebra = _Provider("zebra", _candidate("zebra", 200))
    alpha = _Provider("alpha", _candidate("alpha", 100))
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "zebra, alpha")
    monkeypatch.setattr(
        plugin,
        "_entry_points",
        lambda: (_EntryPoint("z-entry", zebra), _EntryPoint("a-entry", alpha)),
    )

    assert _route_add([])(2, 3) == 200
    assert zebra.select_calls == 1
    assert alpha.select_calls == 0


def test_enable_list_order_is_honoured_through_configure(monkeypatch):
    zebra = _Provider("zebra", _candidate("zebra", 200))
    alpha = _Provider("alpha", _candidate("alpha", 100))
    monkeypatch.setattr(
        plugin,
        "_entry_points",
        lambda: (_EntryPoint("z-entry", zebra), _EntryPoint("a-entry", alpha)),
    )
    plugin.configure(enable=["zebra", "alpha"], source="test")

    assert _route_add([])(2, 3) == 200
    assert zebra.select_calls == 1


def test_broken_and_incompatible_providers_are_isolated(monkeypatch):
    warnings = []
    incompatible = _Provider("old")
    incompatible.api_versions = {plugin.API_VERSION + 1}
    good = _Provider("good", _candidate("good"))
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "good")
    monkeypatch.setattr(plugin.logger, "warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(
        plugin,
        "_entry_points",
        lambda: (
            _EntryPoint("broken", ImportError("missing dependency")),
            _EntryPoint("old", incompatible),
            _EntryPoint("good", good),
        ),
    )

    assert _route_add([])(2, 3) == 100
    assert plugin.last_decision().provider == "good"
    assert len(warnings) == 2
    assert {args[0] for args in warnings} == {
        "Skipping provider entry point %s: %s",
        "Skipping provider %s: %s",
    }
    # An installed provider that never serves is the first thing an operator
    # asks about, so status() has to say what was skipped and why.
    skipped = {
        entry["entry_point"]: entry["reason"]
        for entry in plugin.status()["skipped_providers"]
    }
    assert set(skipped) == {"broken", "old"}
    assert "missing dependency" in skipped["broken"]
    assert "this aiter accepts" in skipped["old"]


def test_force_and_per_op_disable_are_explicit(monkeypatch):
    alpha = _Provider("alpha", _candidate("alpha", 100))
    zebra = _Provider("zebra", _candidate("zebra", 200))
    monkeypatch.setattr(
        plugin,
        "_entry_points",
        lambda: (_EntryPoint("alpha", alpha), _EntryPoint("zebra", zebra)),
    )
    add = _route_add([])

    monkeypatch.setenv("AITER_PLUGIN_FORCE", "zebra")
    assert add(2, 3) == 200

    monkeypatch.setenv("AITER_PLUGIN_DISABLE", "other.op, test.add")
    plugin.reload_config()
    assert add(2, 3) == 5
    assert plugin.last_decision().reason == "op-disabled"


def test_installed_provider_is_not_enabled_by_default(monkeypatch):
    provider = _Provider("aklr", _candidate("aklr"))
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )
    native_calls = []
    add = _route_add(native_calls)

    assert add(2, 3) == 5
    assert native_calls == [(2, 3)]
    assert provider.select_calls == 0
    assert plugin.last_decision().reason == "provider-not-enabled"

    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    plugin.reload_config()  # env is snapshotted; a mid-process change needs a reload
    assert add(2, 3) == 100
    assert provider.select_calls == 1
    assert plugin.last_decision().route == "provider"


def test_non_candidate_return_falls_back_to_native(monkeypatch):
    class Garbage(_Provider):
        def select(self, op, args, kwargs):
            return 42

    provider = Garbage("aklr")
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )
    native_calls = []
    add = _route_add(native_calls)

    assert add(2, 3) == 5
    assert native_calls == [(2, 3)]
    assert plugin.last_decision().reason == "invalid-candidate"
    assert plugin.last_decision().provider == "aklr"


def _enabled_pair(monkeypatch):
    alpha = _Provider("alpha", _candidate("alpha", 100))
    zebra = _Provider("zebra", _candidate("zebra", 200))
    monkeypatch.setattr(
        plugin,
        "_entry_points",
        lambda: (_EntryPoint("alpha", alpha), _EntryPoint("zebra", zebra)),
    )
    return alpha, zebra


def test_env_change_is_not_observed_without_reload(monkeypatch):
    provider = _Provider("aklr", _candidate("aklr"))
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    add = _route_add([])
    assert add(2, 3) == 100

    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "")
    assert add(2, 3) == 100, "the snapshot is taken at first dispatch"
    plugin.reload_config()
    assert add(2, 3) == 5


def test_empty_enable_env_falls_through_to_the_framework_default(monkeypatch):
    """Empty env is unset, so it must not wipe out configure()'s enable list."""

    provider = _Provider("aklr", _candidate("aklr"))
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )
    plugin.configure(enable=["aklr"], source="framework")
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "")
    plugin.reload_config()

    assert _route_add([])(2, 3) == 100
    assert plugin._config().source == "configure:framework"


def test_env_enable_replaces_the_framework_default(monkeypatch):
    alpha, _ = _enabled_pair(monkeypatch)
    plugin.configure(enable=["alpha"], source="framework")
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "zebra")
    plugin.reload_config()

    assert _route_add([])(2, 3) == 200
    assert alpha.select_calls == 0
    assert plugin._config().source == "env"


def test_disable_layers_union(monkeypatch):
    _enabled_pair(monkeypatch)
    plugin.configure(enable=["alpha"], disable=["framework.op"], source="framework")
    monkeypatch.setenv("AITER_PLUGIN_DISABLE", "test.add")
    plugin.reload_config()

    assert _route_add([])(2, 3) == 5
    assert plugin.last_decision().reason == "op-disabled"
    # Union, and each op names the layer that disabled it.
    assert dict(plugin._config().disabled_ops) == {
        "framework.op": "configure:framework",
        "test.add": "env",
    }


def test_generation_bumps_on_change_but_not_on_an_identical_reload(monkeypatch):
    plugin.configure(enable=["alpha"], source="framework")
    first = plugin._config().generation

    plugin.reload_config()
    assert plugin._config().generation == first, "identical snapshot must not bump"

    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "zebra")
    plugin.reload_config()
    assert plugin._config().generation == first + 1


def test_eligible_list_is_cached_per_generation(monkeypatch):
    _enabled_pair(monkeypatch)
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "alpha")
    add = _route_add([])
    assert add(2, 3) == 100
    cached = dict(plugin._eligible_cache)
    assert cached, "the eligible list must be cached"

    assert add(2, 3) == 100
    assert dict(plugin._eligible_cache) == cached, "a second call must reuse it"

    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "zebra")
    plugin.reload_config()
    assert add(2, 3) == 200
    assert dict(plugin._eligible_cache) != cached, "a bump must invalidate it"


def test_configure_requires_a_source():
    with pytest.raises(ValueError):
        plugin.configure(enable=["alpha"], source="   ")
    with pytest.raises(TypeError):
        plugin.configure(["alpha"], source="positional-is-rejected")  # type: ignore[misc]


class _OpaqueDefault:
    """A default whose repr cannot be parsed back, like ``dtypes.fp32``."""

    def __repr__(self) -> str:
        return "<dtype object at 0x...>"


_OPAQUE = _OpaqueDefault()


def _positional(out, input, weight, epsilon, gemma_norm=False): ...
def _many_defaults(x, scale=None, scale_dtype=_OPAQUE, quant_dtype=_OPAQUE): ...
def _keyword_only(x, *, eps=1e-6, out=None): ...
def _positional_only(x, y, /, z=3): ...
def _single(x): ...
def _empty(): ...
def _var_positional(x, *rest, **options): ...


@pytest.mark.parametrize(
    ("native", "args", "kwargs"),
    [
        (_positional, ("o", "i", "w", 1e-6), {}),
        (_positional, ("o", "i"), {"weight": "w", "epsilon": 1e-6}),
        (_positional, ("o", "i", "w", 1e-6, True), {}),
        (_many_defaults, ("x",), {}),
        (_many_defaults, ("x",), {"quant_dtype": "fp8"}),
        (_keyword_only, ("x",), {}),
        (_keyword_only, ("x",), {"eps": 1.0, "out": "o"}),
        (_positional_only, (1, 2), {}),
        (_positional_only, (1, 2, 9), {}),
        (_single, (1,), {}),
        (_empty, (), {}),
        (_var_positional, (1, 2, 3), {"flag": True}),
    ],
)
def test_canonicalizer_matches_signature_bind(native, args, kwargs):
    """The generated binder must be a drop-in for bind() + apply_defaults()."""

    contract = inspect.signature(native)
    bound = contract.bind(*args, **kwargs)
    bound.apply_defaults()

    canonicalize = plugin._build_canonicalizer(contract)

    assert canonicalize(*args, **kwargs) == (bound.args, bound.kwargs)


@pytest.mark.parametrize(
    ("native", "args", "kwargs"),
    [
        (_positional, ("o",), {}),
        (_positional, ("o", "i", "w", 1e-6), {"unknown": 1}),
        (_keyword_only, (), {}),
        (_positional_only, (1, 2), {"x": 3}),
        (_empty, (1,), {}),
    ],
)
def test_canonicalizer_rejects_what_signature_bind_rejects(native, args, kwargs):
    contract = inspect.signature(native)
    canonicalize = plugin._build_canonicalizer(contract)

    with pytest.raises(TypeError):
        contract.bind(*args, **kwargs)
    with pytest.raises(TypeError):
        canonicalize(*args, **kwargs)


def test_var_signatures_fall_back_to_signature_bind():
    canonicalize = plugin._build_canonicalizer(inspect.signature(_var_positional))

    assert canonicalize.__name__ == "canonicalize"  # the fallback, not generated
    assert plugin._canonicalizer_source(inspect.signature(_var_positional)) is None


def test_a_parameter_named_like_an_injected_default_falls_back():
    """A parameter name that would collide with an injected default is unsafe."""

    namespace: dict = {}
    exec(  # noqa: S102 - generating a colliding signature is the point
        f"def collide({plugin._DEFAULT_PREFIX}x=1): ...", namespace
    )
    contract = inspect.signature(namespace["collide"])

    assert plugin._canonicalizer_source(contract) is None
    assert plugin._build_canonicalizer(contract)(2) == ((2,), {})


def test_opaque_defaults_survive_generation():
    """A default with no parseable repr must still be applied, by identity."""

    canonicalize = plugin._build_canonicalizer(inspect.signature(_many_defaults))
    args, _ = canonicalize("x")

    assert args[2] is _OPAQUE


class _Broken(_Provider):
    """A provider that misbehaves before execution."""

    def __init__(self, name, failure):
        super().__init__(name)
        self.failure = failure

    def select(self, op, args, kwargs):
        self.select_calls += 1
        if isinstance(self.failure, Exception):
            raise self.failure
        return self.failure


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (RuntimeError("select exploded"), "provider-select-error"),
        (42, "invalid-candidate"),
    ],
)
def test_a_misbehaving_provider_does_not_deny_service_to_the_next(
    monkeypatch, failure, reason
):
    broken = _Broken("broken", failure)
    good = _Provider("good", _candidate("good", 100))
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "broken, good")
    monkeypatch.setattr(
        plugin,
        "_entry_points",
        lambda: (_EntryPoint("broken", broken), _EntryPoint("good", good)),
    )

    assert _route_add([])(2, 3) == 100
    assert broken.select_calls == 1
    assert good.select_calls == 1
    assert plugin.last_decision().provider == "good"
    # The two parameter sets must not collapse into the same assertion: a
    # misbehaving provider is not a declining one, and the record says which.
    assert plugin.explain("test.add", 2, 3).declines["broken"] == reason


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (RuntimeError("select exploded"), "provider-select-error"),
        (42, "invalid-candidate"),
    ],
)
def test_native_fallback_records_the_first_provider_error(monkeypatch, failure, reason):
    broken = _Broken("broken", failure)
    quiet = _Provider("quiet")  # declines
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "broken, quiet")
    monkeypatch.setattr(
        plugin,
        "_entry_points",
        lambda: (_EntryPoint("broken", broken), _EntryPoint("quiet", quiet)),
    )

    assert _route_add([])(2, 3) == 5
    decision = plugin.last_decision()
    # An error outranks a decline: a decline is normal operation, an error is
    # something someone must fix.
    assert (decision.reason, decision.provider) == (reason, "broken")


def test_every_provider_declining_is_not_reported_as_an_error(monkeypatch):
    first, second = _Provider("first"), _Provider("second")
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "first, second")
    monkeypatch.setattr(
        plugin,
        "_entry_points",
        lambda: (_EntryPoint("first", first), _EntryPoint("second", second)),
    )

    assert _route_add([])(2, 3) == 5
    assert plugin.last_decision().reason == "provider-declined"


def test_per_op_disable_is_still_recorded(monkeypatch):
    """Unlike the kill switch: the op was routed and deliberately turned off."""

    monkeypatch.setenv("AITER_PLUGIN_DISABLE", "test.add")
    assert _route_add([])(2, 3) == 5

    decision = plugin.last_decision()
    assert (decision.reason, decision.config_source) == ("op-disabled", "env")


def test_config_source_names_the_layer_that_produced_the_route(monkeypatch):
    provider = _Provider("aklr", _candidate("aklr"))
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )
    add = _route_add([])

    assert add(2, 3) == 5
    assert plugin.last_decision().config_source == "default-deny"

    plugin.configure(enable=["aklr"], source="vllm rocm 2026.11")
    assert add(2, 3) == 100
    assert plugin.last_decision().config_source == "configure:vllm rocm 2026.11"

    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    plugin.reload_config()
    assert add(2, 3) == 100
    assert plugin.last_decision().config_source == "env"


def test_routing_one_op_name_twice_is_an_error():
    _route_add([])
    with pytest.raises(ValueError, match="already routed"):
        _route_add([])


def test_status_reports_the_stack_and_what_is_routable(monkeypatch):
    provider = _Provider("aklr", _candidate("aklr"))
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )
    _route_add([])
    plugin.configure(enable=["aklr"], disable=["other.op"], source="vllm 2026.11")

    status = plugin.status()

    assert status["config_source"] == "configure:vllm 2026.11"
    assert status["enabled"] == ("aklr",)
    assert status["disabled_ops"] == {"other.op": "configure:vllm 2026.11"}
    assert status["routed_ops"]["test.add"] == {"version": 1, "eligible": ("aklr",)}
    assert status["providers"][0]["name"] == "aklr"


def test_explain_answers_without_executing(monkeypatch):
    provider = _Provider("aklr", _candidate("aklr"))
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )
    native_calls = []
    _route_add(native_calls)

    explanation = plugin.explain("test.add", 2, 3)

    assert explanation.decision.route == "provider"
    assert explanation.decision.provider == "aklr"
    assert native_calls == [], "explain() must not run anything"
    assert plugin.last_decision() is None, "a dry run is not a decision"
    assert plugin.decision_counts() == {}


def test_explain_names_why_each_provider_declined(monkeypatch):
    broken = _Broken("broken", RuntimeError("boom"))
    quiet = _Provider("quiet")
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "broken, quiet")
    monkeypatch.setattr(
        plugin,
        "_entry_points",
        lambda: (_EntryPoint("broken", broken), _EntryPoint("quiet", quiet)),
    )
    _route_add([])

    explanation = plugin.explain("test.add", 2, 3)

    assert explanation.decision.route == "native"
    assert explanation.decision.reason == "provider-select-error"
    assert dict(explanation.declines) == {
        "broken": "provider-select-error",
        "quiet": "declined",
    }


def test_explain_rejects_an_unrouted_op():
    with pytest.raises(KeyError):
        plugin.explain("nobody.routes.this")


def test_decision_counts_include_native_routes(monkeypatch):
    add = _route_add([])

    add(2, 3)
    add(4, 5)

    assert plugin.decision_counts() == {("test.add", "native", "no-provider", None): 2}


def test_on_decision_is_called_inline_for_every_route():
    seen = []
    plugin.on_decision(seen.append)
    add = _route_add([])

    add(2, 3)

    assert [decision.reason for decision in seen] == ["no-provider"]


def test_a_broken_callback_is_swallowed_logged_once_and_counted(monkeypatch):
    warnings = []
    monkeypatch.setattr(plugin.logger, "warning", lambda *args: warnings.append(args))

    def broken(decision):
        raise RuntimeError("telemetry is broken")

    plugin.on_decision(broken)
    add = _route_add([])

    assert add(2, 3) == 5, "a telemetry bug must never break a kernel launch"
    assert add(4, 5) == 9

    assert len(warnings) == 1, "logged once per callback, not once per call"
    assert plugin.status()["decision_callback_errors"] == 2


def test_route_changes_are_logged_once_per_generation(monkeypatch):
    infos = []
    monkeypatch.setattr(
        plugin.logger, "info", lambda fmt, *rest: infos.append(fmt % rest)
    )
    provider = _Provider("aklr", _candidate("aklr"))
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )
    add = _route_add([])

    add(2, 3)
    add(2, 3)
    assert infos == [], "the first generation has nothing to compare against"

    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    plugin.reload_config()
    add(2, 3)
    add(2, 3)

    assert len(infos) == 1, "one line per generation, not per call"
    assert "test.add" in infos[0]


def _provider_at(api_version: int, name: str = "aklr"):
    provider = _Provider(name, _candidate(name))
    provider.api_versions = {api_version}
    return provider


def test_a_superseded_api_version_still_loads_and_names_its_removal(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        plugin.logger, "warning", lambda fmt, *rest: warnings.append(fmt % rest)
    )
    monkeypatch.setattr(
        plugin, "SUPERSEDED_API_VERSIONS", {plugin.API_VERSION - 1: "aiter 0.5"}
    )
    provider = _provider_at(plugin.API_VERSION - 1)
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )
    add = _route_add([])

    assert add(2, 3) == 100, "inside the window it serves"
    assert len(warnings) == 1
    assert "aiter 0.5" in warnings[0], "the warning must name the removal release"

    add(4, 5)
    assert len(warnings) == 1, "warned once per provider, not once per call"


def test_an_api_version_outside_the_window_is_skipped(monkeypatch):
    monkeypatch.setattr(plugin, "SUPERSEDED_API_VERSIONS", {})
    provider = _provider_at(plugin.API_VERSION - 1)
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )

    assert _route_add([])(2, 3) == 5
    assert plugin.status()["skipped_providers"][0]["entry_point"] == "aklr"


def test_a_superseded_op_version_is_routable_while_the_contract_accepts_it(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        plugin.logger, "warning", lambda fmt, *rest: warnings.append(fmt % rest)
    )
    provider = _Provider("aklr", _candidate("aklr"))
    provider.op_versions = {"test.add": {1}}  # still on the old contract
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )

    @plugin.route(op="test.add", version=2, supersedes={1: "aiter 0.6"})
    def add(left: int, right: int = 1) -> int:
        return left + right

    assert add(2, 3) == 100
    assert len(warnings) == 1
    assert "aiter 0.6" in warnings[0]
    # The Decision names the contract that was routed, which is the new one.
    assert plugin.last_decision().op == plugin.Op("test.add", 2)


def test_an_op_version_the_contract_does_not_accept_stays_native(monkeypatch):
    provider = _Provider("aklr", _candidate("aklr"))
    provider.op_versions = {"test.add": {1}}
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )

    @plugin.route(op="test.add", version=2)
    def add(left: int, right: int = 1) -> int:
        return left + right

    assert add(2, 3) == 5
    # "no-provider" means no *eligible* provider, which is literally true here.
    # It does not distinguish "nothing installed" from "installed and enabled but
    # on an op version this contract no longer accepts" -- a distinction the
    # window makes newly common. Not invented here: the design enumerates no
    # reason for it.
    assert plugin.last_decision().reason == "no-provider"


def test_a_contract_cannot_supersede_its_own_version():
    with pytest.raises(ValueError, match="own version"):
        plugin.route(op="test.add", version=2, supersedes={2: "aiter 0.6"})


def test_status_lists_what_is_deprecated_and_when_it_goes(monkeypatch):
    monkeypatch.setattr(
        plugin, "SUPERSEDED_API_VERSIONS", {plugin.API_VERSION - 1: "aiter 0.5"}
    )
    provider = _provider_at(plugin.API_VERSION - 1)
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )

    @plugin.route(op="test.add", version=2, supersedes={1: "aiter 0.6"})
    def add(left: int, right: int = 1) -> int:
        return left + right

    assert plugin.status()["deprecations"] == (
        {
            "kind": "op-version",
            "subject": "test.add",
            "version": 1,
            "removed_in": "aiter 0.6",
        },
        {
            "kind": "provider-api",
            "subject": "aklr",
            "version": plugin.API_VERSION - 1,
            "removed_in": "aiter 0.5",
        },
    )


def test_a_provider_supporting_several_api_versions_is_served_on_the_newest():
    """One release covering several AITER versions is the point of the set."""

    provider = _Provider("aklr", _candidate("aklr"))
    provider.api_versions = {plugin.API_VERSION - 1, plugin.API_VERSION}

    assert plugin._declared_api_versions(provider, "aklr") == frozenset(
        {plugin.API_VERSION - 1, plugin.API_VERSION}
    )


def test_the_single_api_version_spelling_still_works_and_says_so(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        plugin.logger, "warning", lambda fmt, *rest: warnings.append(fmt % rest)
    )
    legacy = _Provider("aklr", _candidate("aklr"))
    del legacy.api_versions
    legacy.api_version = plugin.API_VERSION
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    monkeypatch.setattr(plugin, "_entry_points", lambda: (_EntryPoint("aklr", legacy),))

    assert _route_add([])(2, 3) == 100, "the old spelling must keep working"
    assert len(warnings) == 1
    assert "declare api_versions instead" in warnings[0]


def test_two_providers_cannot_answer_to_one_name(monkeypatch):
    """A name is an identity: enable, force and every Decision address it."""

    first = _Provider("aklr", _candidate("aklr", 100))
    second = _Provider("aklr", _candidate("aklr", 200))
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    monkeypatch.setattr(
        plugin,
        "_entry_points",
        lambda: (_EntryPoint("a-entry", first), _EntryPoint("b-entry", second)),
    )

    assert _route_add([])(2, 3) == 100, "the first entry point keeps the name"
    assert second.select_calls == 0
    # Skipped, never raised: a bad install must not break ``import aiter``.
    skipped = {
        e["entry_point"]: e["reason"] for e in plugin.status()["skipped_providers"]
    }
    assert "already used by entry point 'a-entry'" in skipped["b-entry"]


def test_a_var_signature_warns_that_it_pays_the_generic_binder(monkeypatch):
    messages = []
    monkeypatch.setattr(
        plugin.logger, "info", lambda fmt, *rest: messages.append(fmt % rest)
    )

    @plugin.route(op="test.varargs", version=1)
    def anything(*args, **kwargs):
        return "native"

    assert len(messages) == 1
    assert "Signature.bind on every call" in messages[0]


def test_a_candidate_without_evidence_can_still_serve(monkeypatch):
    """The interface says AITER *may* record a qualification, not that one exists."""

    bare = plugin.KernelCandidate(
        provider="aklr",
        kernel_id="k1",
        artifact_digest="sha256:abc",
        run=lambda left, right: 100,
    )
    provider = _Provider("aklr", bare)
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )

    assert _route_add([])(2, 3) == 100
    assert plugin.last_decision().qualification is None


def test_the_newest_acceptable_op_version_is_matched(monkeypatch):
    """Same rule as the API version handshake, which takes max(usable)."""

    warnings = []
    monkeypatch.setattr(
        plugin.logger, "warning", lambda fmt, *rest: warnings.append(fmt % rest)
    )
    provider = _Provider("aklr", _candidate("aklr"))
    provider.op_versions = {"test.add": {1, 2}}
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )

    # Written oldest-first on purpose: the answer must not depend on this order.
    @plugin.route(op="test.add", version=3, supersedes={1: "aiter 0.5", 2: "aiter 0.6"})
    def add(left: int, right: int = 1) -> int:
        return left + right

    assert add(2, 3) == 100
    assert "op version 2" in warnings[0], "the newest acceptable version serves"
    assert "aiter 0.6" in warnings[0], "and its own removal release is named"


def test_explain_names_the_provider_it_blames(monkeypatch):
    """explain() claims to return the Decision the call would produce."""

    broken = _Broken("broken", RuntimeError("boom"))
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "broken")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("broken", broken),)
    )
    add = _route_add([])

    predicted = plugin.explain("test.add", 2, 3).decision
    add(2, 3)
    actual = plugin.last_decision()

    assert (predicted.reason, predicted.provider) == (actual.reason, actual.provider)


def test_explain_reports_no_decision_under_the_kill_switch(monkeypatch):
    monkeypatch.setenv("AITER_PLUGINS", "0")
    _route_add([])

    # Dispatch records nothing there, so there is no would-be Decision either.
    assert plugin.explain("test.add", 2, 3).decision is None


def test_explain_does_not_spend_the_one_shot_deprecation_warning(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        plugin.logger, "warning", lambda fmt, *rest: warnings.append(fmt % rest)
    )
    provider = _Provider("aklr", _candidate("aklr"))
    provider.op_versions = {"test.add": {1}}
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )

    @plugin.route(op="test.add", version=2, supersedes={1: "aiter 0.6"})
    def add(left: int, right: int = 1) -> int:
        return left + right

    plugin.explain("test.add", 2, 3)
    assert warnings == [], "a dry run must not consume the warning"

    add(2, 3)
    assert len(warnings) == 1, "real traffic still gets told"


def test_a_forced_provider_name_is_stripped(monkeypatch):
    provider = _Provider("aklr", _candidate("aklr"))
    monkeypatch.setenv("AITER_PLUGIN_FORCE", "  aklr ")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )

    assert _route_add([])(2, 3) == 100, "whitespace is a typo, not a different name"


def test_status_survives_a_provider_that_stops_answering(monkeypatch):
    """status() is where an operator goes to find out why routing stopped."""

    class Rotting:
        broken = False
        name = "aklr"
        api_versions: ClassVar[set] = {plugin.API_VERSION}

        @property
        def op_versions(self):
            if Rotting.broken:
                raise RuntimeError("backend went away")
            return {"test.add": {1}}

        def select(self, op, args, kwargs):
            return _candidate("aklr")

    provider = Rotting()
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )
    add = _route_add([])
    assert add(2, 3) == 100

    Rotting.broken = True
    try:
        # Eligibility is cached per (op, generation), so a provider that rots
        # mid-generation keeps serving until the config changes -- the hot path
        # does not re-interrogate it.
        assert add(2, 3) == 100

        # status() does interrogate it, and must survive the answer.
        entry = plugin.status()["providers"][0]
        assert entry["name"] == "aklr"
        assert "unavailable" in entry["op_versions"]
        assert "backend went away" in entry["op_versions"]
    finally:
        Rotting.broken = False
