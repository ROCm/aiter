# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The two ops this milestone routes, exercised end to end.

Kept apart from tests/test_plugin.py because these need a built AITER: they
import torch and call the real ops. The interface's own tests stay buildless so
they can gate every PR without a GPU runner (see tests/conftest.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Any, ClassVar

import pytest
from conftest import requires_built_aiter

requires_built_aiter()

import torch

import aiter
from aiter import plugin


@dataclass
class _EntryPoint:
    name: str
    target: Any

    def load(self) -> Any:
        return self.target


class _Provider:
    api_versions: ClassVar[set] = {plugin.API_VERSION}

    def __init__(self, name: str, selected: plugin.KernelCandidate | None = None):
        self.name = name
        self.op_versions: dict = {}
        self.selected = selected
        self.select_calls = 0

    def select(self, op, args, kwargs):
        self.select_calls += 1
        return self.selected


def _candidate(provider: str, result: Any = None) -> plugin.KernelCandidate:
    return plugin.KernelCandidate(
        provider=provider,
        kernel_id="kernel-1",
        artifact_digest="sha256:abc",
        qualification="experimental",
        run=lambda *args, **kwargs: result,
    )


@pytest.fixture(autouse=True)
def _clean_plugin_state(monkeypatch):
    monkeypatch.delenv("AITER_PLUGINS", raising=False)
    plugin._reset_for_testing()
    yield
    plugin._reset_for_testing()


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
    plugin.configure(enable=("aklr",), source="test")
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
    plugin.configure(enable=("aklr",), source="test")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )
    out = torch.zeros(2, 4)

    aiter.rmsnorm(out, torch.rand(2, 4), torch.ones(4), 1e-6)

    assert provider.select_calls == 1
    decision = plugin.last_decision()
    assert decision.route == "provider"
    assert decision.provider == "aklr"
