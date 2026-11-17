# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER_MOEV2 switch and the legacy fused_moe hook (steps doc 1.2)."""

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
    monkeypatch.delenv(envs.AITER_MOEV2, raising=False)
    saved = paths.registered()
    envs.reload()
    yield
    paths._REGISTRY.clear()
    paths._REGISTRY.update(saved)
    envs.reload()


def _set(monkeypatch, value):
    monkeypatch.setenv(envs.AITER_MOEV2, value)
    envs.reload()


def _fake_path(name, graduated=False, claims=lambda call: True):
    calls = []

    def run(call):
        calls.append(call)
        return ("v2", name)

    paths.register(paths.Path(name, claims, run, graduated))
    return calls


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


def test_gpu_suites_cover_exactly_the_planned_paths():
    from moe_dispatch import gpu_suites

    assert set(gpu_suites.PATHS) == set(paths.PLANNED)


def test_legacy_fused_moe_is_a_plain_function():
    assert inspect.isfunction(legacy.fused_moe)
    assert not hasattr(legacy.fused_moe, "cache_info")


def test_v2_api_without_a_path_raises():
    with pytest.raises(NotImplementedError):
        api.fused_moe(None, None, None, None, None)


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
