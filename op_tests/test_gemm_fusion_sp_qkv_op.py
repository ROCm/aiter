# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only contract tests for the neutral ``sp_qkv_fused`` v1 op.

No GPU, no built kernels, no profile/resources: the op is exercised through

* its real public signature (``inspect.signature``) — no dicts, no policy arguments;
* the guarded ``aiter.plugin`` route: inert when the interface package is absent and
  registered (with ``__aiter_plugin_op__``) when a fake interface is injected;
* native delegation through a fake ``runtime.sp_qkv_native`` that records the payload;
* mode/default/error validation; and
* the fake-provider dispatch path from the provider skill (§5): a provider candidate
  serves the canonicalized call, a decline falls back to native, and the native
  payload is never built when a provider serves.

The tests deliberately use sentinel objects for tensors: this contract layer is tensor
metadata-free; dtype/shape/layout enforcement belongs to the native path
(``inputs.py``) and the GPU evidence.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from typing import ClassVar

import pytest

OP_MODULE = "aiter.ops.gemm_fusion.sp_qkv.op"
PLUGIN_MODULE = "aiter.plugin"

POSITIONAL = [
    "q",
    "x",
    "x_scale",
    "w",
    "w_scale",
    "q_norm_w",
    "k_norm_w",
    "cos_cache",
    "sin_cache",
    "positions",
    "slot_mapping",
    "k_cache",
    "v_cache",
    "k_scale",
    "v_scale",
]
KEYWORD_ONLY = [
    "mode",
    "k",
    "v",
    "index_q",
    "index_q_norm_w",
    "index_k_norm_w",
    "index_cache",
    "eps",
    "group",
    "stream",
]
DEFAULTS = {
    "k": None,
    "v": None,
    "index_q": None,
    "index_q_norm_w": None,
    "index_k_norm_w": None,
    "index_cache": None,
    "eps": 1e-6,
    "group": None,
    "stream": None,
}

# The 15 required tensors, plus the mode-specific outputs used by the default tests.
SENTINELS = {name: object() for name in POSITIONAL}
DENSE = {**SENTINELS, "k": object(), "v": object()}
INDEXER = {
    **SENTINELS,
    "index_q": object(),
    "index_q_norm_w": object(),
    "index_k_norm_w": object(),
    "index_cache": object(),
}


# ---------------------------------------------------------------------------
# fake runtime / fake provider interface
# ---------------------------------------------------------------------------
class FakeRuntime:
    def __init__(self):
        self.payloads = []
        self.cover_payloads = []
        self.cover_result = True

    def sp_qkv_native(self, payload):
        self.payloads.append(payload)
        return "native-ok"

    def sp_qkv_covers(self, payload):
        self.cover_payloads.append(payload)
        return self.cover_result

    def sp_qkv_available(self):
        return True

    def profile_name(self):
        return "fake-profile"


def _install_fake_plugin(monkeypatch):
    """Install a minimal ``aiter.plugin`` with route/select/configure semantics.

    This is the CPU-only stand-in for the unmerged #7465/#7466/#7484 interface.  It
    canonicalizes exactly like the real one: positional-or-keyword params become a
    positional tuple (defaults applied), keyword-only params stay in a read-only mapping.
    """

    module = types.ModuleType(PLUGIN_MODULE)
    module.API_VERSION = 1
    module.routes = {}
    module.enabled = []
    module.providers = []

    class Op:
        def __init__(self, name, version):
            self.name, self.version = name, version

        def __eq__(self, other):
            return isinstance(other, Op) and (self.name, self.version) == (
                other.name,
                other.version,
            )

        def __hash__(self):
            return hash((self.name, self.version))

        def __repr__(self):
            return f"Op({self.name!r}, {self.version!r})"

    class KernelCandidate:
        def __init__(
            self,
            *,
            provider,
            kernel_id,
            artifact_digest,
            run,
            qualification=None,
        ):
            self.provider = provider
            self.kernel_id = kernel_id
            self.artifact_digest = artifact_digest
            self.qualification = qualification
            self.run = run

    class FakeProvider:
        name: ClassVar[str] = "fake-provider"
        api_versions: ClassVar[set[int]] = {1}
        op_versions: ClassVar[dict[str, set[int]]] = {"sp_qkv_fused": {1}}

        def __init__(self, *, serve=True):
            self.serve = serve
            self.launches = []

        def select(self, op, args, kwargs):
            if op.name != "sp_qkv_fused" or op.version != 1 or not self.serve:
                return None
            return KernelCandidate(
                provider=self.name,
                kernel_id="fake.sp_qkv.v1",
                artifact_digest="sha256:fake",
                qualification="cpu-contract-test",
                run=self.run,
            )

        def run(self, *args, **kwargs):
            self.launches.append((args, dict(kwargs)))

        def decline(self):
            self.serve = False
            return self

        def offer(self):
            self.serve = True
            return self

    def _canonicalize(fn, args, kwargs):
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        pos, kw = [], {}
        for p in sig.parameters.values():
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                pos.append(bound.arguments[p.name])
            elif p.kind is inspect.Parameter.KEYWORD_ONLY:
                kw[p.name] = bound.arguments[p.name]
        return tuple(pos), types.MappingProxyType(kw)

    def route(*, op, version=1, supersedes=None):
        del supersedes

        def decorate(fn):
            import functools

            routed = Op(op, version)
            module.routes[op] = routed

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                pos, kw = _canonicalize(fn, args, kwargs)
                for provider in module.enabled:
                    candidate = provider.select(routed, pos, kw)
                    if candidate is not None:
                        return candidate.run(*pos, **kw)
                return fn(*pos, **kw)

            wrapper.__aiter_plugin_op__ = routed
            return wrapper

        return decorate

    def configure(*, source, enable=()):
        assert source
        module.enabled = [
            p
            for p in module.providers
            if p.name in {str(name).strip() for name in enable}
        ]

    module.Op = Op
    module.KernelCandidate = KernelCandidate
    module.FakeProvider = FakeProvider
    module.route = route
    module.configure = configure
    module.status = lambda: {
        "enabled": tuple(p.name for p in module.enabled),
        "routed_ops": {
            name: {"version": routed.version} for name, routed in module.routes.items()
        },
    }
    module.last_decision = lambda: None
    module.decision_counts = dict

    monkeypatch.setitem(sys.modules, PLUGIN_MODULE, module)
    monkeypatch.setattr(sys.modules["aiter"], "plugin", module, raising=False)
    return module


def _fresh_op(monkeypatch):
    """Import the op module with the current sys.modules (no cached previous import)."""

    monkeypatch.delitem(sys.modules, OP_MODULE, raising=False)
    return importlib.import_module(OP_MODULE)


@pytest.fixture
def op(monkeypatch):
    """The real op module, imported with ``aiter.plugin`` absent (the PR-head state)."""

    monkeypatch.delitem(sys.modules, PLUGIN_MODULE, raising=False)
    aiter = sys.modules.get("aiter")
    if aiter is not None and hasattr(aiter, "plugin"):
        monkeypatch.delattr(aiter, "plugin", raising=False)
    return _fresh_op(monkeypatch)


@pytest.fixture
def fake_runtime(op, monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(op, "runtime", runtime)
    return runtime


# ---------------------------------------------------------------------------
# 1. the real flat contract
# ---------------------------------------------------------------------------
def test_public_signature_is_flat_and_exact(op):
    sig = inspect.signature(op.sp_qkv_fused)
    params = list(sig.parameters.values())

    assert [p.name for p in params] == POSITIONAL + KEYWORD_ONLY
    assert not any(
        p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in params
    ), "no *args/**kwargs: the contract must be introspectable"
    for p in params[: len(POSITIONAL)]:
        assert p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert p.default is inspect.Parameter.empty
    assert params[len(POSITIONAL)].name == "mode"
    assert params[len(POSITIONAL)].kind is inspect.Parameter.KEYWORD_ONLY
    assert params[len(POSITIONAL)].default is inspect.Parameter.empty
    for p in params[len(POSITIONAL) + 1 :]:
        assert p.kind is inspect.Parameter.KEYWORD_ONLY
        assert p.default == DEFAULTS[p.name]
    assert sig.return_annotation in (None, "None", type(None))
    # no policy objects leaked into the public signature
    for forbidden in (
        "res",
        "profile",
        "bucket",
        "variant",
        "call_id",
        "args",
        "kwargs",
    ):
        assert forbidden not in sig.parameters


def test_contract_sheet_exists_and_names_the_coverage_limitation():
    from pathlib import Path

    sheet = (
        Path(__file__).resolve().parents[1]
        / "aiter"
        / "ops"
        / "gemm_fusion"
        / "sp_qkv"
        / "CONTRACT.md"
    )
    text = sheet.read_text(encoding="utf-8")
    for needle in (
        "sp_qkv_fused",
        "contract version",
        "coverage-limited",
        "option B",
        "option A",
        "unfused fallback",
        "Phase-2",
        "sp_qkv_covers",
        "prologue",
        "prepare",
        "graph capture",
        "provider",
    ):
        assert needle in text, needle


# ---------------------------------------------------------------------------
# 2. guarded route: inert without the interface, registered with it
# ---------------------------------------------------------------------------
def test_route_guard_is_inert_without_aiter_plugin(op):
    assert not hasattr(op.sp_qkv_fused, "__aiter_plugin_op__")
    # and the op still works: native delegation is available without the interface
    runtime = FakeRuntime()
    op.runtime = runtime
    assert op.sp_qkv_fused(**DENSE, mode="dense") is None
    assert len(runtime.payloads) == 1


def test_route_guard_registers_with_an_injected_fake_plugin(monkeypatch):
    fake = _install_fake_plugin(monkeypatch)
    op = _fresh_op(monkeypatch)

    assert op.sp_qkv_fused.__aiter_plugin_op__ == fake.Op("sp_qkv_fused", 1)
    assert fake.routes["sp_qkv_fused"] == fake.Op("sp_qkv_fused", 1)
    # registration happened exactly once (import-time decorator, not per call)
    assert sum(1 for name in fake.routes if name == "sp_qkv_fused") == 1


# ---------------------------------------------------------------------------
# 3. native delegation: payload keys = contract names, in signature order
# ---------------------------------------------------------------------------
def test_native_delegation_carries_the_contract_payload(op, fake_runtime):
    result = op.sp_qkv_fused(**DENSE, mode="dense")
    assert result is None, "the op returns None even if a native stub returns a value"
    assert len(fake_runtime.payloads) == 1
    payload = fake_runtime.payloads[0]
    assert (
        list(payload) == POSITIONAL + KEYWORD_ONLY
    ), "payload keys follow the contract"
    assert [payload[name] for name in POSITIONAL] == [
        DENSE[name] for name in POSITIONAL
    ]
    assert payload["mode"] == "dense"
    assert payload["k"] is DENSE["k"] and payload["v"] is DENSE["v"]
    for name in (
        "index_q",
        "index_q_norm_w",
        "index_k_norm_w",
        "index_cache",
    ):
        assert payload[name] is None
    assert payload["eps"] == 1e-6
    assert payload["group"] is None and payload["stream"] is None
    assert payload["eps"] == 1e-6
    assert payload["group"] is None and payload["stream"] is None


def test_native_delegation_indexer_optionals_are_passed_through(op, fake_runtime):
    assert op.sp_qkv_fused(**INDEXER, mode="indexer") is None
    payload = fake_runtime.payloads[0]
    for name in (
        "index_q",
        "index_q_norm_w",
        "index_k_norm_w",
        "index_cache",
    ):
        assert payload[name] is INDEXER[name]
    assert payload["k"] is None and payload["v"] is None


def test_defaults_and_keyword_only_call_style(op, fake_runtime):
    # required tensors are positional-or-keyword; the optional/mode part is keyword-only
    args = [DENSE[name] for name in POSITIONAL]
    assert (
        op.sp_qkv_fused(*args, mode="dense", k=DENSE["k"], v=DENSE["v"], eps=2e-6)
        is None
    )
    payload = fake_runtime.payloads[0]
    assert payload["eps"] == 2e-6
    with pytest.raises(TypeError):
        op.sp_qkv_fused(*args, "dense")  # mode is keyword-only


# ---------------------------------------------------------------------------
# 4. mode validation and mode-specific tensors
# ---------------------------------------------------------------------------
def test_invalid_mode_is_rejected_before_any_runtime_call(op, fake_runtime):
    with pytest.raises(ValueError, match="mode must be one of"):
        op.sp_qkv_fused(**DENSE, mode="bogus")
    assert fake_runtime.payloads == []


def test_missing_mode_specific_tensors_are_rejected_before_any_runtime_call(
    op, fake_runtime
):
    with pytest.raises(ValueError, match="requires k, v"):
        op.sp_qkv_fused(**SENTINELS, mode="dense")
    with pytest.raises(ValueError, match="requires index_q"):
        op.sp_qkv_fused(**SENTINELS, mode="indexer")
    assert fake_runtime.payloads == []


@pytest.mark.parametrize("mode", ["dense", "reuse"])
def test_dense_and_reuse_require_the_kv_outputs(op, fake_runtime, mode):
    assert op.sp_qkv_fused(**DENSE, mode=mode) is None
    assert fake_runtime.payloads[0]["mode"] == mode


# ---------------------------------------------------------------------------
# 5. provider dispatch path (skill §5): provider serves, decline -> native
# ---------------------------------------------------------------------------
def test_fake_provider_serves_the_canonicalized_call_and_native_is_not_run(
    monkeypatch, fake_runtime
):
    fake = _install_fake_plugin(monkeypatch)
    op = _fresh_op(monkeypatch)
    monkeypatch.setattr(op, "runtime", fake_runtime)
    provider = fake.FakeProvider()
    fake.providers.append(provider)
    fake.configure(source="cpu-test", enable=[provider.name])

    # Mixed call style: required tensors by keyword, mode/outputs keyword-only.
    assert op.sp_qkv_fused(**DENSE, mode="dense") is None
    assert fake_runtime.payloads == [], "provider served: native must not be reached"
    assert len(provider.launches) == 1
    args, kwargs = provider.launches[0]
    assert args == tuple(
        DENSE[name] for name in POSITIONAL
    ), "canonicalized positional prefix follows the contract order"
    assert kwargs["mode"] == "dense"
    assert kwargs["k"] is DENSE["k"] and kwargs["v"] is DENSE["v"]
    assert kwargs["eps"] == 1e-6
    assert kwargs["group"] is None and kwargs["stream"] is None
    assert op.sp_qkv_fused.__aiter_plugin_op__ == fake.Op("sp_qkv_fused", 1)


def test_fake_provider_decline_falls_back_to_native(monkeypatch, fake_runtime):
    fake = _install_fake_plugin(monkeypatch)
    op = _fresh_op(monkeypatch)
    monkeypatch.setattr(op, "runtime", fake_runtime)
    provider = fake.FakeProvider(serve=False)
    fake.providers.append(provider)
    fake.configure(source="cpu-test", enable=[provider.name])

    assert op.sp_qkv_fused(**DENSE, mode="dense") is None
    assert provider.launches == []
    assert len(fake_runtime.payloads) == 1


# ---------------------------------------------------------------------------
# 6. coverage predicate: cheap, non-executing, same flat args
# ---------------------------------------------------------------------------
def test_coverage_predicate_has_the_same_flat_signature(op):
    fused = inspect.signature(op.sp_qkv_fused)
    covers = inspect.signature(op.sp_qkv_covers)
    assert list(covers.parameters.values()) == list(fused.parameters.values())
    assert covers.return_annotation in (bool, "bool")


def test_coverage_predicate_delegates_the_payload_without_executing(op, fake_runtime):
    fake_runtime.cover_result = True
    assert op.sp_qkv_covers(**DENSE, mode="dense") is True
    assert fake_runtime.payloads == [], "covers() must not call sp_qkv_native"
    assert len(fake_runtime.cover_payloads) == 1
    payload = fake_runtime.cover_payloads[0]
    assert list(payload) == POSITIONAL + KEYWORD_ONLY
    assert [payload[name] for name in POSITIONAL] == [
        DENSE[name] for name in POSITIONAL
    ]
    assert payload["mode"] == "dense"

    fake_runtime.cover_result = False
    assert op.sp_qkv_covers(**DENSE, mode="dense") is False


def test_coverage_predicate_is_false_when_the_runtime_lacks_it(op, monkeypatch):
    monkeypatch.setattr(op, "runtime", types.SimpleNamespace())
    assert op.sp_qkv_covers(**DENSE, mode="dense") is False


def test_coverage_predicate_says_false_for_an_unknown_mode_without_a_runtime_call(
    op, fake_runtime
):
    assert op.sp_qkv_covers(**DENSE, mode="bogus") is False
    assert fake_runtime.cover_payloads == []


def test_coverage_predicate_is_exported_lazily(monkeypatch):
    monkeypatch.delitem(sys.modules, OP_MODULE, raising=False)
    import aiter

    fn = aiter.sp_qkv_covers
    assert fn.__name__ == "sp_qkv_covers"
    assert fn is importlib.import_module(OP_MODULE).sp_qkv_covers


# ---------------------------------------------------------------------------
# 7. the op is exported lazily from aiter
# ---------------------------------------------------------------------------
def test_aiter_exports_sp_qkv_fused_lazily(monkeypatch):
    monkeypatch.delitem(sys.modules, OP_MODULE, raising=False)
    import aiter

    fn = aiter.sp_qkv_fused
    assert fn.__name__ == "sp_qkv_fused"
    assert fn is importlib.import_module(OP_MODULE).sp_qkv_fused
