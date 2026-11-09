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
    # rmsnorm is routed where rms_norm and rmsnorm2d_fwd converge, so neither
    # public spelling carries the marker and the low-level binding no longer
    # does either -- the op is the contract, not any one function name.
    assert "rmsnorm" in plugin.status()["routed_ops"]
    assert not hasattr(aiter.rmsnorm, "__aiter_plugin_op__")
    assert aiter.pertoken_quant.__aiter_plugin_op__ == plugin.Op("pertoken_quant", 1)
    # Both public spellings still report their own signatures, which is what
    # makes either of them routable at all.
    for spelling in (aiter.rms_norm, aiter.rmsnorm2d_fwd):
        assert list(signature(spelling).parameters) == [
            "input",
            "weight",
            "epsilon",
            "use_model_sensitive_rmsnorm",
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
    assert plugin.last_decision() is None, "nothing enabled, nothing recorded"
    decision = plugin.explain("pertoken_quant", x, quant_dtype=torch.int8).decision
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


def test_rmsnorm_provider_serves_through_either_public_spelling(monkeypatch):
    """Routing the convergence point covers both names, not one of them."""

    provider = _Provider("aklr", _candidate("aklr", torch.full((2, 4), 7.0)))
    provider.op_versions = {"rmsnorm": {2}}
    plugin.configure(enable=("aklr",), source="test")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )

    for spelling in (aiter.rms_norm, aiter.rmsnorm2d_fwd):
        result = spelling(torch.rand(2, 4), torch.ones(4), 1e-6)
        decision = plugin.last_decision()
        assert (decision.route, decision.provider) == ("provider", "aklr")
        assert torch.equal(result, torch.full((2, 4), 7.0))

    assert provider.select_calls == 2


_REFERENCE_PROVIDER = '''
"""A reference provider for the two routed contracts.

Deliberately minimal: it declares what the interface asks for, serves one
contract, declines the other, and reports evidence it does not ask AITER to
interpret. Everything here is what an out-of-tree provider would ship.
"""

import torch

from aiter import plugin


def _rms_norm(input, weight, epsilon, use_model_sensitive_rmsnorm=0):
    """Functional, because that is the shape of the routed contract.

    The op is routed where both public spellings converge, above the backend
    choice, and that layer allocates and returns rather than writing into a
    caller's buffer.
    """
    scale = torch.rsqrt(input.float().pow(2).mean(-1, keepdim=True) + epsilon)
    return (input.float() * scale * weight.float()).to(input.dtype)


def _rms_norm_with_add(
    out,
    input,
    residual_in,
    residual_out,
    weight,
    epsilon,
    gemma_norm=False,
    use_model_sensitive_rmsnorm=0,
):
    """The residual-add path: both outputs are the caller's to keep."""
    total = input.float() + residual_in.float()
    residual_out.copy_(total.to(residual_out.dtype))
    scale = torch.rsqrt(total.pow(2).mean(-1, keepdim=True) + epsilon)
    out.copy_((total * scale * weight.float()).to(out.dtype))


class Provider:
    name = "refprov"
    api_versions = {plugin.API_VERSION}
    # One release, two contracts. A provider need not cover an op family to
    # serve part of it.
    op_versions = {"rmsnorm": {2}, "rmsnorm_with_add": {1}, "pertoken_quant": {1}}

    def __init__(self):
        self.kernels = {
            "rmsnorm": plugin.KernelCandidate(
                provider="refprov",
                kernel_id="rms_norm_reference",
                artifact_digest="sha256:0000",
                qualification="https://example.invalid/evidence/rms_norm",
                run=_rms_norm,
            ),
            "rmsnorm_with_add": plugin.KernelCandidate(
                provider="refprov",
                kernel_id="add_rms_norm_reference",
                artifact_digest="sha256:1111",
                qualification="https://example.invalid/evidence/add_rms_norm",
                run=_rms_norm_with_add,
            ),
        }

    # Which argument carries the hidden size, per contract. Reading args[0] for
    # both would work today by coincidence -- the functional contract takes the
    # input first and the write-out one takes the output -- and would break
    # silently the first time a contract's parameter order changed. A provider
    # reads each contract on its own terms.
    _normalized_over = {"rmsnorm": 0, "rmsnorm_with_add": 1}

    def select(self, op, args, kwargs):
        candidate = self.kernels.get(op.name)
        if candidate is None:
            return None  # declining is normal, and must be observable

        hidden = args[self._normalized_over[op.name]].shape[-1]
        if hidden % 2:
            return None  # a per-call coverage decision, the shape it is for
        return candidate


instance = Provider()
'''


def _install_reference_provider(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    (site / "refprov_pkg.py").write_text(_REFERENCE_PROVIDER)
    dist = site / "refprov-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: refprov\nVersion: 1.0\n"
    )
    (dist / "entry_points.txt").write_text(
        f"[{plugin.ENTRY_POINT_GROUP}]\nrefprov = refprov_pkg:instance\n"
    )
    monkeypatch.syspath_prepend(str(site))


def test_an_installed_provider_serves_a_routed_contract(tmp_path, monkeypatch):
    """The milestone's end state: a wheel, discovered, enabled, serving.

    Nothing is monkeypatched into discovery -- the provider arrives through real
    packaging metadata, the way one actually would.
    """

    _install_reference_provider(tmp_path, monkeypatch)
    plugin.configure(enable=("refprov",), source="test")

    weight = torch.rand(8)
    value = torch.rand(2, 8)

    out = aiter.rmsnorm2d_fwd(value, weight, 1e-6)

    decision = plugin.last_decision()
    assert (decision.route, decision.provider) == ("provider", "refprov")
    assert decision.kernel_id == "rms_norm_reference"
    assert decision.artifact_digest == "sha256:0000"
    assert decision.config_source == "configure:test"

    # It computed the op, not merely claimed the route. CPU tensors are proof
    # enough on their own: the native path here is a ROCm kernel that would not
    # run on them at all.
    scale = torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    expect = (value.float() * scale * weight.float()).to(out.dtype)
    torch.testing.assert_close(out, expect)

    del tmp_path
    plugin.configure(enable=(), source="test")


def test_the_same_provider_declines_the_other_contract(tmp_path, monkeypatch):
    """Declining is ordinary operation, and it has to be visible as such."""

    _install_reference_provider(tmp_path, monkeypatch)
    plugin.configure(enable=("refprov",), source="test")

    aiter.pertoken_quant(torch.rand(4, 8))

    decision = plugin.last_decision()
    assert (decision.route, decision.reason) == ("native", "provider-declined")


def test_a_per_call_decline_is_not_an_error(tmp_path, monkeypatch):
    """Coverage is the provider's to judge, per call, and costs native only."""

    _install_reference_provider(tmp_path, monkeypatch)
    plugin.configure(enable=("refprov",), source="test")

    explanation = plugin.explain(
        "rmsnorm", torch.zeros(2, 7), torch.rand(2, 7), torch.rand(7), 1e-6
    )

    assert explanation.decision.reason == "provider-declined"
    assert dict(explanation.declines) == {"refprov": "declined"}


def test_the_residual_add_path_is_servable_too(tmp_path, monkeypatch):
    """The path vLLM-ROCm and SGLang actually call for per-layer rmsnorm.

    It dispatches to a different kernel than ``rmsnorm`` and has a second
    output, so serving it is a separate contract rather than a shape of the
    first one.
    """

    _install_reference_provider(tmp_path, monkeypatch)
    plugin.configure(enable=("refprov",), source="test")

    value, residual, weight = torch.rand(2, 8), torch.rand(2, 8), torch.rand(8)
    out, residual_out = torch.zeros(2, 8), torch.zeros(2, 8)

    aiter.rmsnorm2d_fwd_with_add(out, value, residual, residual_out, weight, 1e-6)

    decision = plugin.last_decision()
    assert (decision.route, decision.provider) == ("provider", "refprov")
    assert decision.kernel_id == "add_rms_norm_reference"
    assert decision.op == plugin.Op("rmsnorm_with_add", 1)

    total = value.float() + residual.float()
    scale = torch.rsqrt(total.pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(residual_out, total.to(residual_out.dtype))
    torch.testing.assert_close(out, (total * scale * weight.float()).to(out.dtype))


def test_one_provider_serving_two_contracts_is_recorded_separately(
    tmp_path, monkeypatch
):
    """Each contract is answered on its own terms, and counted on its own."""

    _install_reference_provider(tmp_path, monkeypatch)
    plugin.configure(enable=("refprov",), source="test")

    aiter.rmsnorm2d_fwd(torch.rand(2, 8), torch.rand(8), 1e-6)
    aiter.rmsnorm2d_fwd_with_add(
        torch.zeros(2, 8),
        torch.rand(2, 8),
        torch.rand(2, 8),
        torch.zeros(2, 8),
        torch.rand(8),
        1e-6,
    )
    aiter.pertoken_quant(torch.rand(4, 8))

    counts = plugin.decision_counts()
    assert counts[("rmsnorm", "provider", "selected", "refprov")] == 1
    assert counts[("rmsnorm_with_add", "provider", "selected", "refprov")] == 1
    assert counts[("pertoken_quant", "native", "provider-declined", None)] == 1

    routed = plugin.status()["routed_ops"]
    assert routed["rmsnorm_with_add"]["eligible"] == ("refprov",)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="graph capture needs a device"
)


def _capture(fn):
    """Capture ``fn`` into a graph, after one warm-up call outside it."""

    fn()  # the RFC's advice: warm the op before capturing, so first-call work
    torch.cuda.synchronize()  # does not land inside the graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


@requires_cuda
def test_the_capture_time_decision_is_what_every_replay_runs(tmp_path, monkeypatch):
    """Replay runs no Python, so dispatch cannot revisit a routing decision.

    This is the quietest failure mode in the design: whatever was decided on
    the capturing call is what the graph does for its whole life, and because
    replay records no Decision, nothing afterwards says which call decided it.
    Pinning it is the difference between a documented property and a belief.
    """

    _install_reference_provider(tmp_path, monkeypatch)
    plugin.configure(enable=("refprov",), source="test")

    weight = torch.rand(8, device="cuda")
    value = torch.rand(2, 8, device="cuda")
    graph, _ = _capture(lambda: aiter.rmsnorm2d_fwd(value, weight, 1e-6))

    captured = plugin.last_decision()
    assert captured.route == "provider" and captured.provider == "refprov"

    before = plugin.decision_counts().copy()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    assert plugin.decision_counts() == before, (
        "replay must not reach dispatch: a Decision per replay would mean "
        "Python ran, and a graph that runs Python is not a graph"
    )
    assert plugin.last_decision() == captured, (
        "the reading still describes the capturing call, which is why an "
        "operator cannot tell replays apart from it"
    )


@requires_cuda
def test_turning_a_provider_off_does_not_reach_an_already_captured_graph(
    tmp_path, monkeypatch
):
    """The consequence worth stating: the kill switch does not un-bake a graph.

    AITER_PLUGINS=0 stops future dispatch. It cannot touch kernels already
    captured, because there is no dispatch left in them to stop. An operator
    reaching for the switch during an incident needs to know the graphs have
    to be rebuilt.
    """

    _install_reference_provider(tmp_path, monkeypatch)
    plugin.configure(enable=("refprov",), source="test")

    weight = torch.rand(8, device="cuda")
    value = torch.rand(2, 8, device="cuda")
    graph, out = _capture(lambda: aiter.rmsnorm2d_fwd(value, weight, 1e-6))
    assert plugin.last_decision().provider == "refprov"

    monkeypatch.setenv("AITER_PLUGINS", "0")
    plugin.reload_config()
    graph.replay()
    torch.cuda.synchronize()

    scale = torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    expect = (value.float() * scale * weight.float()).to(out.dtype)
    torch.testing.assert_close(out, expect)
    assert plugin.last_decision() is None, "the switch is off for anything new"
