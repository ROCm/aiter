# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER_MOEV2 switch and the legacy fused_moe hook."""

import inspect
import logging
import pathlib
import sys

import pytest

import aiter.fused_moe as legacy
from aiter.moev2 import api, envs, paths

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """Each test starts with no path registered (the built-in ones are restored after)."""
    monkeypatch.delenv(envs.AITER_MOEV2, raising=False)
    saved = paths.registered()
    paths._REGISTRY.clear()
    envs.reload()
    paths.reload()
    yield
    paths._REGISTRY.clear()
    paths._REGISTRY.update(saved)
    envs.reload()
    paths.reload()


def _set(monkeypatch, value):
    monkeypatch.setenv(envs.AITER_MOEV2, value)
    envs.reload()


def _fake_path(name, graduated=False, claims=lambda call: True, entry="api", out=None):
    calls = []

    def run(call):
        calls.append(call)
        return ("v2", name) if out is None else out(call)

    paths.register(paths.Path(name, claims, run, graduated, entry=entry))
    return calls


def _cpu_moe_args(tokens=2, model_dim=8, inter_dim=4, experts=3, topk=2):
    import torch

    hidden = torch.randn(tokens, model_dim, dtype=torch.bfloat16)
    w1 = torch.randn(experts, 2 * inter_dim, model_dim, dtype=torch.bfloat16)
    w2 = torch.randn(experts, model_dim, inter_dim, dtype=torch.bfloat16)
    topk_weight = torch.rand(tokens, topk, dtype=torch.float32)
    topk_ids = torch.randint(0, experts, (tokens, topk), dtype=torch.int32)
    return hidden, w1, w2, topk_weight, topk_ids


@pytest.mark.parametrize(
    "value,expected",
    [
        ("", "default"),
        ("off", "off"),
        ("0", "off"),
        ("ON", "on"),
        ("all", "on"),
        ("a4w4, fhmoe", frozenset({"a4w4", "fhmoe"})),
    ],
)
def test_selection_parsing(monkeypatch, value, expected):
    _set(monkeypatch, value)
    assert envs.moev2_selection() == expected


def test_nothing_registered_routes_nothing(monkeypatch):
    for value in ("", "off", "on"):
        _set(monkeypatch, value)
        assert not paths.any_active()
        assert paths.route_to_v2({}) is None


def test_unknown_path_name_is_an_error(monkeypatch):
    _set(monkeypatch, "a4w4,a4w44")
    with pytest.raises(ValueError, match="a4w44"):
        paths.any_active()


def test_planned_but_unimplemented_path_warns(monkeypatch, caplog):
    _set(monkeypatch, "a4w4")
    # caplog listens on the root logger; aiter's logger does not propagate.
    monkeypatch.setattr(logging.getLogger("aiter"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="aiter"):
        assert not paths.any_active()
    assert "not implemented yet" in caplog.text


def test_register_rejects_unplanned_and_duplicate_names():
    with pytest.raises(ValueError, match="unknown"):
        _fake_path("not_a_path")
    _fake_path("a4w4")
    with pytest.raises(ValueError, match="already"):
        _fake_path("a4w4")


def test_default_runs_only_graduated_paths(monkeypatch):
    _fake_path("a4w4", graduated=False)
    _fake_path("fhmoe", graduated=True)
    assert [p.name for p in paths.active()] == ["fhmoe"]
    _set(monkeypatch, "on")
    assert [p.name for p in paths.active()] == ["a4w4", "fhmoe"]
    _set(monkeypatch, "a4w4")
    assert [p.name for p in paths.active()] == ["a4w4"]
    _set(monkeypatch, "off")
    assert paths.active() == ()


def test_legacy_fused_moe_routes_claimed_calls(monkeypatch):
    calls = _fake_path("a16w16", claims=lambda call: call["hidden_states"] == "claimed")
    _set(monkeypatch, "a16w16")
    out = legacy.fused_moe("claimed", "w1", "w2", "tw", "ti", splitk=3)
    assert out == ("v2", "a16w16")
    (call,) = calls
    assert call["splitk"] == 3 and call["w1"] == "w1"
    assert set(call) == set(inspect.signature(legacy.fused_moe).parameters)


def test_op_entry_sees_the_custom_op_arguments(monkeypatch):
    import torch

    from aiter import ActivationType, QuantType

    marker = torch.full((2, 8), 7.0, dtype=torch.bfloat16)
    calls = _fake_path("a16w16", entry="op", out=lambda call: marker.clone())
    _set(monkeypatch, "a16w16")
    out = legacy.fused_moe(*_cpu_moe_args(), activation=ActivationType.Gelu)
    assert torch.equal(out, marker)
    (call,) = calls
    assert set(call) == set(inspect.signature(legacy.fused_moe_fake).parameters)
    assert call["activation"] == ActivationType.Gelu.value
    assert call["quant_type"] == QuantType.No.value
    assert call["block_size_M"] == -1 and call["ep_source_token_map"] is None


def test_declined_calls_fall_back_to_legacy(monkeypatch):
    import torch

    _fake_path("a16w16", out=lambda call: paths.DECLINED)
    op_calls = _fake_path("a8w8_pt", entry="op", out=lambda call: paths.DECLINED)
    _set(monkeypatch, "on")
    legacy_out = torch.zeros(2, 8, dtype=torch.bfloat16)
    legacy_calls = []

    def fake_impl(*args, **kwargs):
        legacy_calls.append(args)
        return legacy_out

    monkeypatch.setattr(legacy, "_fused_moe_impl", fake_impl)
    assert legacy.fused_moe(*_cpu_moe_args()) is legacy_out
    assert len(op_calls) == 1 and len(legacy_calls) == 1


def test_op_entry_path_records_into_the_legacy_kernel_bench(monkeypatch):
    import torch

    from aiter.moev2 import hooks

    def run(call):
        hooks.record_kernel("stage1", run)
        return torch.zeros(2, 8, dtype=torch.bfloat16)

    paths.register(paths.Path("a8w8_pt", lambda call: True, run, entry="op"))
    _set(monkeypatch, "a8w8_pt")
    bench = []
    monkeypatch.setattr(legacy, "kernel_bench_callable", bench)
    legacy.fused_moe(*_cpu_moe_args())
    assert bench == [("stage1", run)]


def test_torch_compile_traces_the_hooks_with_a_path_active(monkeypatch):
    import torch

    marker = torch.full((2, 8), 7.0, dtype=torch.bfloat16)
    _fake_path("a16w16", entry="op", out=lambda call: marker.clone())
    _set(monkeypatch, "a16w16")
    # The first call also imports the built-in paths, which dynamo cannot trace.
    monkeypatch.setattr(paths, "_builtin_loaded", False)
    paths.reload()
    compiled = torch.compile(legacy.fused_moe, backend="eager", fullgraph=True)
    assert torch.equal(compiled(*_cpu_moe_args()), marker)


def test_graduation_can_be_scoped_to_archs(monkeypatch):
    _fake_path("a8w8_pt", graduated=frozenset({"gfx950"}))
    for arch, expected in (("gfx950", ["a8w8_pt"]), ("gfx942", [])):
        monkeypatch.setattr(paths, "get_gfx_runtime", lambda arch=arch: arch)
        paths.reload()
        assert [p.name for p in paths.active()] == expected

    def no_gpu():
        raise RuntimeError("no GPU")

    monkeypatch.setattr(paths, "get_gfx_runtime", no_gpu)
    paths.reload()
    assert paths.active() == ()


def test_unknown_entry_is_an_error():
    with pytest.raises(ValueError, match="entry"):
        paths.Path("a4w4", lambda call: True, lambda call: None, entry="fused_moe")


def test_builtin_paths():
    from aiter.moev2.migration import a8w8_pt

    assert paths._BUILTIN_MODULES == ("aiter.moev2.migration.a8w8_pt",)
    assert a8w8_pt.PATH.name == "a8w8_pt" and a8w8_pt.PATH.entry == "op"
    assert a8w8_pt.PATH.dispatch_record is not None
    # No path graduates: each runs only when enabled with AITER_MOEV2.
    assert not a8w8_pt.PATH.graduated


def test_gpu_suites_cover_exactly_the_planned_paths():
    from moe_dispatch import gpu_suites

    assert set(gpu_suites.PATHS) == set(paths.PLANNED)


def test_gpu_suite_runner_enables_the_selected_path(monkeypatch):
    from moe_dispatch import gpu_suites

    seen = []

    def run(cmd, *, check, env):
        seen.append((cmd, check, env[envs.AITER_MOEV2]))
        return type("Result", (), {"returncode": 0})()

    suite = gpu_suites.Suite(("python3", "selected_path_test.py"))
    monkeypatch.setattr(gpu_suites, "PATHS", {"a8w8_pt": (suite,)})
    monkeypatch.setattr(gpu_suites, "SHARED", ())
    monkeypatch.setattr(gpu_suites.subprocess, "run", run)
    assert gpu_suites.main(["--path", "a8w8_pt", "--run"]) == 0
    assert seen == [(suite.cmd, False, "a8w8_pt")]


def test_legacy_fused_moe_is_a_plain_function():
    assert inspect.isfunction(legacy.fused_moe)
    assert not hasattr(legacy.fused_moe, "cache_info")


def test_v2_api_without_a_path_raises():
    with pytest.raises(NotImplementedError):
        api.fused_moe(None, None, None, None, None)


def test_v2_api_drops_a_registered_presort_when_it_returns_or_raises(monkeypatch):
    from aiter import presorted_moe

    monkeypatch.setattr(presorted_moe._PRESORTED_MOE, "aux", {"M": 1}, raising=False)
    with pytest.raises(NotImplementedError):
        api.fused_moe(None, None, None, None, None)
    assert presorted_moe._PRESORTED_MOE.aux is None

    _fake_path("a16w16", entry="op")
    monkeypatch.setattr(presorted_moe._PRESORTED_MOE, "aux", {"M": 1}, raising=False)
    assert api.fused_moe(*_cpu_moe_args()) == ("v2", "a16w16")
    assert presorted_moe._PRESORTED_MOE.aux is None


def test_v2_api_matches_legacy_signature():
    old = inspect.signature(legacy.fused_moe).parameters
    new = inspect.signature(api.fused_moe).parameters
    assert list(old) == list(new)
    for name, p in old.items():
        q = new[name]
        assert p.kind == q.kind, name
        default = (
            getattr(p.default, "value", p.default) if name == "gate_mode" else p.default
        )
        assert default == q.default, name
