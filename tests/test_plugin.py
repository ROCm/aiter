# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
        self.api_version = plugin.API_VERSION
        self.op_versions = {"test.add": {1}}
        self.selected = selected
        self.select_calls = 0

    def select(self, op, args, kwargs):
        self.select_calls += 1
        return self.selected


@pytest.fixture(autouse=True)
def _clean_plugin_state(monkeypatch):
    for name in ("AITER_PLUGINS", "AITER_PLUGIN_DISABLE", "AITER_PLUGIN_FORCE"):
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
        op=plugin.Op("test.add", 1), route="native", reason="no-provider"
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
    assert plugin.last_decision().reason == "plugins-disabled"


@pytest.mark.parametrize("outcome", [None, RuntimeError("selection failed")])
def test_decline_or_select_error_uses_visible_native_fallback(monkeypatch, outcome):
    class Provider(_Provider):
        def select(self, op, args, kwargs):
            if isinstance(outcome, Exception):
                raise outcome

    provider = Provider("aklr")
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
    )


def test_aiter_signature_is_enforced_before_provider_selection(monkeypatch):
    provider = _Provider("aklr", _candidate("aklr"))
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
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )
    native_calls = []

    with pytest.raises(RuntimeError, match="kernel failed"):
        _route_add(native_calls)(2, 3)

    assert native_calls == []
    assert plugin.last_decision().provider == "aklr"
    assert plugin.last_decision().route == "provider"


def test_provider_arbitration_is_deterministic(monkeypatch):
    zebra = _Provider("zebra", _candidate("zebra", 200))
    alpha = _Provider("alpha", _candidate("alpha", 100))
    monkeypatch.setattr(
        plugin,
        "_entry_points",
        lambda: (_EntryPoint("z-entry", zebra), _EntryPoint("a-entry", alpha)),
    )

    assert _route_add([])(2, 3) == 100
    assert alpha.select_calls == 1
    assert zebra.select_calls == 0


def test_broken_and_incompatible_providers_are_isolated(monkeypatch):
    warnings = []
    incompatible = _Provider("old")
    incompatible.api_version = plugin.API_VERSION + 1
    good = _Provider("good", _candidate("good"))
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
        "Skipping provider %s: API version %s is incompatible with %s",
    }


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
    assert add(2, 3) == 5
    assert plugin.last_decision().reason == "op-disabled"
