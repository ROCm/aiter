# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Any

import pytest
import torch

import aiter
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


def test_provider_arbitration_is_deterministic(monkeypatch):
    zebra = _Provider("zebra", _candidate("zebra", 200))
    alpha = _Provider("alpha", _candidate("alpha", 100))
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "alpha, zebra")
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


def test_routed_ops_register_the_public_contract():
    assert aiter.rmsnorm.__aiter_plugin_op__ == plugin.Op("rmsnorm", 1)
    assert aiter.pertoken_quant.__aiter_plugin_op__ == plugin.Op("pertoken_quant", 1)
    assert list(signature(aiter.rmsnorm).parameters) == [
        "out",
        "input",
        "weight",
        "epsilon",
        "gemma_norm",
    ]
    assert list(signature(aiter.pertoken_quant).parameters) == [
        "x",
        "scale",
        "x_scale",
        "scale_dtype",
        "quant_dtype",
        "dtypeMax",
    ]


def test_pertoken_quant_stays_native_by_default(monkeypatch):
    monkeypatch.setattr(plugin, "_entry_points", lambda: ())
    x = torch.tensor([[-1.0, 2.0], [0.5, -3.0]])

    y, y_scale = aiter.pertoken_quant(x, quant_dtype=torch.int8)

    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = amax / 127
    assert torch.equal(y, (x / scale).to(torch.int8))
    assert torch.equal(y_scale, scale.to(torch.float32))
    decision = plugin.last_decision()
    assert decision.route == "native"
    assert decision.reason == "no-provider"


def test_pertoken_quant_provider_serves_when_enabled(monkeypatch):
    seen = []

    class Serving(_Provider):
        def select(self, op, args, kwargs):
            seen.append((op, args, dict(kwargs)))
            return self.selected

    provider = Serving("aklr", _candidate("aklr", result=(7, 0.5)))
    provider.op_versions = {"pertoken_quant": {1}}
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )
    x = torch.tensor([[1.0, -2.0]])

    y, y_scale = aiter.pertoken_quant(x, quant_dtype=torch.int8)

    assert (y, y_scale) == (7, 0.5)
    assert seen == [
        (
            plugin.Op("pertoken_quant", 1),
            (x, None, None, torch.float32, torch.int8, None),
            {},
        )
    ]
    decision = plugin.last_decision()
    assert decision.route == "provider"
    assert decision.provider == "aklr"
    assert decision.kernel_id == "kernel-1"


def test_rmsnorm_provider_serves_without_gpu(monkeypatch):
    # A served call never reaches the native ROCm kernel, so CPU tensors are
    # enough to prove the routed path: if native ran, it would fail on CPU.
    provider = _Provider("aklr", _candidate("aklr"))
    provider.op_versions = {"rmsnorm": {1}}
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", "aklr")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )
    out = torch.zeros(2, 4)

    aiter.rmsnorm(out, torch.rand(2, 4), torch.ones(4), 1e-6)

    assert provider.select_calls == 1
    decision = plugin.last_decision()
    assert decision.route == "provider"
    assert decision.provider == "aklr"
