# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The two ops this milestone routes, exercised end to end.

Kept apart from tests/test_plugin.py because these need a built AITER: they
import torch and call the real ops. The interface's own tests stay buildless so
they can gate every PR without a GPU runner (see tests/conftest.py).
"""

from __future__ import annotations

import gc
import subprocess
import sys
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


def _sparse_prefill_args(tokens=2, heads=1, dim=512):
    """Well-formed enough to reach the routed layer, which dispatches at once.

    Shapes are not exercised here: a provider that serves means the native
    never runs, and the point of this test is which callers reach the
    provider at all.
    """
    return {
        "q": torch.rand(tokens, heads, dim),
        "unified_kv": torch.rand(4, dim),
        "kv_indices_prefix": torch.zeros(4, dtype=torch.int32),
        "kv_indptr_prefix": torch.tensor([0, 2, 4], dtype=torch.int32),
        "kv": torch.rand(4, dim),
        "kv_indices_extend": torch.zeros(4, dtype=torch.int32),
        "kv_indptr_extend": torch.tensor([0, 2, 4], dtype=torch.int32),
        "attn_sink": None,
        "softmax_scale": 1.0,
    }


def test_sparse_prefill_serves_through_either_public_spelling(monkeypatch):
    """`mla_sparse_prefill` and `pa_sparse_prefill_opus` are one op.

    They take the same arguments and forward them unchanged to the same
    implementation; the second name differs only by pinning the backend the
    first one would have chosen anyway. Routing their convergence point is
    what makes a provider reach the callers who use the backend-named
    spelling -- which, on the DSV4.1 prefill path, is all of them.
    """

    from aiter.ops.mla_sparse_prefill import mla_sparse_prefill
    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_opus

    served = torch.full((2, 1, 512), 7.0)
    provider = _Provider("aklr", _candidate("aklr", served))
    provider.op_versions = {"mla_sparse_prefill": {1}}
    plugin.configure(enable=("aklr",), source="test")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )

    for spelling in (mla_sparse_prefill, pa_sparse_prefill_opus):
        result = spelling(**_sparse_prefill_args())
        decision = plugin.last_decision()
        assert (decision.route, decision.provider) == ("provider", "aklr")
        assert torch.equal(result, served)

    assert provider.select_calls == 2


def test_the_backend_pin_is_not_part_of_the_sparse_prefill_contract(monkeypatch):
    """`backend=` is answered above the routed layer, so it never reaches a provider.

    A provider serving this contract has one kernel, not a choice of AITER's;
    being handed a pin it cannot satisfy would leave it declining every call
    from the spelling that pins. The dispatcher settles the native backend
    first and the op begins below that.
    """

    from aiter.ops.mla_sparse_prefill import mla_sparse_prefill

    served = torch.full((2, 1, 512), 7.0)
    provider = _Provider("aklr", _candidate("aklr", served))
    provider.op_versions = {"mla_sparse_prefill": {1}}
    plugin.configure(enable=("aklr",), source="test")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )

    result = mla_sparse_prefill(**_sparse_prefill_args(), backend="opus")

    assert plugin.last_decision().route == "provider"
    assert torch.equal(result, served)


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
def _capture_cold(fn):
    """Capture ``fn`` with no warm-up, so its first call lands inside.

    The opposite of ``_capture``. What a framework does to an op its warmup
    pass does not reach -- a decode-only op under a prefill-only dummy batch is
    the shape ATOM already hit, which is why ``capture_cudagraph()`` calls
    ``warmup_exchange()`` before recording.
    """

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


@requires_cuda
def test_a_provider_allocating_on_its_first_call_inside_a_capture_keeps_it(
    tmp_path, monkeypatch
):
    """Lazy allocation under capture is safe, and the memory is the provider's.

    Worth pinning because the intuition runs the other way: capture-time
    allocations are widely assumed to land in the graph's private pool and die
    with it. They do not when the provider holds the reference -- measured on
    gfx950 at 16 MiB, same address after the graph is gone.

    The consequence is what this says to a provider author: you do not have to
    pre-allocate to be correct. What you must not do is read a result back,
    which the next test covers.
    """

    _install_reference_provider(tmp_path, monkeypatch)
    plugin.configure(enable=("refprov",), source="test")
    weight = torch.rand(8, device="cuda")
    value = torch.rand(2, 8, device="cuda")

    scratch = {}

    def allocate_then_call():
        if "t" not in scratch:
            # The provider's lazy init, standing in for a tuning table or a
            # workspace: it happens on the first call, which is this one.
            scratch["t"] = torch.empty(1 << 20, dtype=torch.float32, device="cuda")
            scratch["ptr"] = scratch["t"].data_ptr()
        return aiter.rmsnorm2d_fwd(value, weight, 1e-6)

    graph, _ = _capture_cold(allocate_then_call)
    assert plugin.last_decision().route == "provider"

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    del graph
    gc.collect()
    torch.cuda.synchronize()

    assert scratch["t"].data_ptr() == scratch["ptr"], (
        "the allocation moved or was freed with the graph; a provider that "
        "allocates lazily under capture would be left holding a dangling buffer"
    )
    scratch["t"][:4].fill_(1.0)
    torch.cuda.synchronize()
    assert scratch["t"][:4].tolist() == [1.0, 1.0, 1.0, 1.0]


_CAPTURE_READBACK_PROBE = """
import torch

x = torch.rand(16, device="cuda")
graph = torch.cuda.CUDAGraph()
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    x.mul(2)
torch.cuda.current_stream().wait_stream(side)
torch.cuda.synchronize()

try:
    with torch.cuda.graph(graph):
        x[0].item()          # the readback a provider's logging or assert would do
    print("NOT_REFUSED")
except Exception as error:
    print("REFUSED", type(error).__name__)

# Whatever the error said, is anything still usable?
try:
    torch.rand(8, device="cuda")
    print("CONTEXT_SURVIVED")
except Exception as error:
    print("CONTEXT_POISONED", type(error).__name__)
"""


@requires_cuda
def test_a_readback_under_capture_is_refused_and_takes_the_context_with_it():
    """Run in a subprocess, because proving this ruins the process that proves it.

    A provider's first call can land inside a capture (see the test above), and
    the ordinary contents of a lazy init -- a log line with `.item()`, an assert
    on a checksum, a `synchronize()` after setup -- read a result back. A capture
    records instructions without executing them, so there is nothing to read.

    What makes this worth a test rather than a line of documentation is the
    second assertion. The refusal is not recoverable: the CUDA context is left
    unusable, and the next unrelated allocation fails with a message naming
    neither capture nor the provider. A provider author sees the symptom
    somewhere else entirely.

    Subprocess isolation is not fastidiousness. Run this inline and every CUDA
    test after it in this file fails, including ones that have nothing to do
    with plugins -- which is exactly how the behaviour was found.

    Measured on gfx950 / torch 2.10.0+rocm7.2.4. If a future ROCm or PyTorch
    makes readback under capture legal, or recoverable, this goes red and the
    obligation stated in #7240 should be revisited.

    Both assertions were mutation-checked by swapping the `.item()` in the
    probe for a `torch.empty` -- an operation capture permits. That reports
    NOT_REFUSED and CONTEXT_SURVIVED, failing both, so neither assertion is
    satisfied by something other than the behaviour it names.
    """

    result = subprocess.run(
        [sys.executable, "-c", _CAPTURE_READBACK_PROBE],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,  # the probe reports through stdout; a non-zero exit is
    )  # itself a result, not a reason to raise here
    out = result.stdout

    assert "REFUSED" in out, (
        f"reading a result back inside a capture was accepted; "
        f"#7240's provider obligation rests on it being refused. stdout={out!r} "
        f"stderr={result.stderr[-400:]!r}"
    )
    assert "CONTEXT_POISONED" in out, (
        "the capture failure was recoverable -- the context survived. That is "
        "better than what #7240 documents, and the RFC should be corrected "
        f"rather than this test relaxed. stdout={out!r}"
    )


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


# ---------------------------------------------------------------------------
# fused_dual_rmsnorm_mxfp8_quant: the out-buffer bypass
#
# This op's public name has two paths -- naming any of `yq`, `sq`, `yk`
# bypasses routing, because the buffered form RETURNS the buffers it was
# given while a custom op's schema declares its results do not alias its
# inputs. The split is deliberate and documented; what it needs is proof
# that each side is the side it claims to be.
#
# The risk is one-directional. Landing on the buffered path by naming a
# buffer gets AITER's own kernel, which is what naming it asked for. The
# failure that matters is the reverse: a call that looks routed quietly
# not being routed. So every test below pins WHICH path ran, not just that
# the call returned something.
# ---------------------------------------------------------------------------


def _dual_rmsnorm_args(device="cpu", dtype=torch.bfloat16):
    """The seven parameters of the routed contract -- no output buffers."""

    m, kq, kk = 4, 64, 32
    return {
        "q": torch.randn(m, kq, device=device, dtype=dtype),
        "k": torch.randn(m, kk, device=device, dtype=dtype),
        "q_weight": torch.randn(kq, device=device, dtype=dtype),
        "k_weight": torch.randn(kk, device=device, dtype=dtype),
        "eps_q": 1e-6,
    }


def _serve_dual_rmsnorm(monkeypatch, served):
    """Enable a provider for this contract and hand back its spy object."""

    provider = _Provider("aklr", _candidate("aklr", served))
    provider.op_versions = {"fused_dual_rmsnorm_mxfp8_quant": {1}}
    plugin.configure(enable=("aklr",), source="test")
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider),)
    )
    return provider


def test_the_allocating_form_is_what_a_provider_serves(monkeypatch):
    """No buffers named, so the call is routed and a provider can take it."""

    from aiter.ops.triton.quant.fused_mxfp8_quant import (
        fused_dual_rmsnorm_mxfp8_quant,
    )

    served = (torch.zeros(4, 64), torch.zeros(4, 2), torch.zeros(4, 32))
    provider = _serve_dual_rmsnorm(monkeypatch, served)

    result = fused_dual_rmsnorm_mxfp8_quant(**_dual_rmsnorm_args())

    decision = plugin.last_decision()
    assert (decision.route, decision.provider) == ("provider", "aklr")
    assert provider.select_calls == 1
    assert all(torch.equal(a, b) for a, b in zip(result, served))


@pytest.mark.parametrize("buffer_name", ["yq", "sq", "yk"])
def test_naming_any_one_output_buffer_bypasses_routing(monkeypatch, buffer_name):
    """Each buffer on its own is enough to leave the routed path.

    Parametrised per buffer rather than testing them together: the branch
    reads all three, so a test that passes them at once would still pass if
    two of the three conditions were dropped.

    The native body is replaced here so this runs without a device -- what is
    under test is which branch the public name picks, not what the kernel
    computes. The provider is enabled and must come out untouched.
    """

    from aiter.ops.triton.quant import fused_mxfp8_quant as module

    provider = _serve_dual_rmsnorm(monkeypatch, (None, None, None))
    sentinel = ("native-ran", None, None)
    monkeypatch.setattr(
        module,
        "_fused_dual_rmsnorm_mxfp8_quant_native",
        lambda *args, **kwargs: sentinel,
    )

    args = _dual_rmsnorm_args()
    args[buffer_name] = torch.zeros(4, 64)
    result = module.fused_dual_rmsnorm_mxfp8_quant(**args)

    assert result is sentinel, f"naming {buffer_name} did not reach the native body"
    assert provider.select_calls == 0, (
        f"naming {buffer_name} still asked a provider to select; the buffered "
        f"form must not reach routing at all"
    )


@requires_cuda
def test_the_buffered_form_writes_and_returns_the_caller_s_buffers():
    """Buffer identity, not just equal values.

    A fake that allocates compiles fine and returns the right value while
    leaving the caller's buffer unwritten -- correct return, stale buffer,
    no error. Comparing values alone cannot see that. These assertions are
    on `data_ptr`, so they fail if the returned tensor is a different
    allocation that merely holds the right numbers.
    """

    from aiter.ops.triton.quant.fused_mxfp8_quant import (
        fused_dual_rmsnorm_mxfp8_quant,
    )

    args = _dual_rmsnorm_args(device="cuda")
    expected = fused_dual_rmsnorm_mxfp8_quant(**args)

    buffers = {
        "yq": torch.empty_like(expected[0]),
        "sq": torch.empty_like(expected[1]),
        "yk": torch.empty_like(expected[2]),
    }
    got = fused_dual_rmsnorm_mxfp8_quant(**args, **buffers)

    for name, returned, want in zip(buffers, got, expected):
        assert returned.data_ptr() == buffers[name].data_ptr(), (
            f"{name}: the buffered form returned a different allocation, so a "
            f"caller reading its own buffer afterwards would read stale data"
        )
        assert torch.equal(
            returned, want
        ), f"{name}: buffered value differs from routed"


@requires_cuda
def test_the_routed_form_runs_inside_a_compiled_region():
    """The allocating form is the one a compiled caller reaches.

    Registration exists so the dispatch stays out of the caller's graph; a
    compiled call that returns something other than the eager one would mean
    the fake's metadata disagrees with the kernel.
    """

    from aiter.ops.triton.quant.fused_mxfp8_quant import (
        fused_dual_rmsnorm_mxfp8_quant,
    )

    args = _dual_rmsnorm_args(device="cuda")
    eager = fused_dual_rmsnorm_mxfp8_quant(**args)

    compiled = torch.compile(fused_dual_rmsnorm_mxfp8_quant, dynamic=False)
    got = compiled(**args)

    for i, (a, b) in enumerate(zip(got, eager)):
        assert a.shape == b.shape and a.dtype == b.dtype, (
            f"output {i}: compiled metadata {tuple(a.shape)}/{a.dtype} differs "
            f"from eager {tuple(b.shape)}/{b.dtype} -- the fake is wrong"
        )
        assert torch.equal(a, b), f"output {i}: compiled value differs from eager"
