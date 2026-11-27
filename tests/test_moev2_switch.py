# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER_MOEV2 switch and the legacy fused_moe hook.

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

import importlib
import inspect
import logging
import os
import pathlib
import sys
import threading
import types

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


def test_the_selection_is_logged_once_when_set(monkeypatch, caplog):
    _fake_path("a4w4")
    # caplog listens on the root logger; aiter's logger does not propagate.
    monkeypatch.setattr(logging.getLogger("aiter"), "propagate", True)
    with caplog.at_level(logging.INFO, logger="aiter"):
        paths.active()
        assert caplog.text == ""
        _set(monkeypatch, "a4w4")
        paths.active()
        paths.active()
    assert caplog.text.count("'a4w4': a4w4 on aiter.moev2") == 1
    assert "AITER_MOEV2=off runs every call on the legacy code" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="aiter"):
        _set(monkeypatch, "off")
        paths.active()
    assert "AITER_MOEV2='off': every call on the legacy code" in caplog.text


def test_the_differential_harness_leaves_out_the_selection_line(monkeypatch):
    from moe_dispatch.differential import captured_logs

    _fake_path("a4w4")
    _set(monkeypatch, "a4w4")
    with captured_logs() as logs:
        paths.active()
        logging.getLogger("aiter").info("a dispatch line")
    assert logs == [(logging.INFO, "a dispatch line")]


def test_explain_logs_why_an_active_path_does_not_claim_a_call(monkeypatch, caplog):
    paths.register(
        paths.Path(
            "a8w8_pt",
            lambda call: call["ok"],
            lambda call: None,
            why_not=lambda call: None if call["ok"] else "quant_type No",
        )
    )
    _set(monkeypatch, "a8w8_pt")
    paths.active()
    monkeypatch.setattr(logging.getLogger("aiter"), "propagate", True)
    line = "path a8w8_pt does not claim a call (quant_type No)"
    with caplog.at_level(logging.INFO, logger="aiter"):
        assert paths.route_to_v2({"ok": False}) is None
        assert line not in caplog.text
        monkeypatch.setenv("AITER_MOE_EXPLAIN", "1")
        for _ in range(2):
            assert paths.route_to_v2({"ok": False}) is None
        assert paths.route_to_v2({"ok": True}) is not None
    assert caplog.text.count(line) == 1


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


@pytest.fixture
def gated_builtin(monkeypatch, tmp_path):
    """A built-in path module (a4w4) that pauses until `release` once it starts, and
    until `register` between the two steps of its `register()` call."""
    gate = types.SimpleNamespace(
        **{n: threading.Event() for n in ("entered", "release", "loaded", "register")}
    )
    monkeypatch.setitem(sys.modules, "moev2_test_gate", gate)
    (tmp_path / "moev2_gated_builtin.py").write_text(
        "import moev2_test_gate as gate\n"
        "from aiter.moev2 import paths\n"
        "gate.entered.set()\n"
        "gate.release.wait(10)\n"
        "PATH = paths.Path('a4w4', lambda call: True, lambda call: None)\n"
        "paths._load_builtin()\n"
        "gate.loaded.set()\n"
        "gate.register.wait(10)\n"
        "paths.register(PATH)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(paths, "_BUILTIN_MODULES", ("moev2_gated_builtin",))
    monkeypatch.setattr(paths, "_builtin_loaded", False)
    paths.reload()
    yield gate
    sys.modules.pop("moev2_gated_builtin", None)


def _in_thread(fn):
    out = []
    thread = threading.Thread(target=lambda: out.append(fn()), daemon=True)
    thread.start()
    return thread, out


def _names(selected):
    return [p.name for p in selected]


def test_a_call_during_the_first_load_waits_for_the_builtin_paths(
    monkeypatch, gated_builtin
):
    _set(monkeypatch, "on")
    first, first_out = _in_thread(paths.active)
    assert gated_builtin.entered.wait(10)
    second, second_out = _in_thread(paths.active)
    second.join(0.2)
    assert second.is_alive()
    gated_builtin.release.set()
    gated_builtin.register.set()
    first.join(10)
    second.join(10)
    assert [_names(out) for out in first_out + second_out] == [["a4w4"], ["a4w4"]]
    assert _names(paths.active()) == ["a4w4"]


@pytest.mark.parametrize("pause", ["entered", "loaded"])
def test_importing_a_builtin_module_directly_during_a_first_call(
    monkeypatch, gated_builtin, pause
):
    """The call waits for the module, whether it comes before the module's register()
    runs the loader ("entered") or between that and the registration ("loaded"); no
    deadlock either way."""
    _set(monkeypatch, "on")
    importer, _ = _in_thread(lambda: importlib.import_module("moev2_gated_builtin"))
    if pause == "loaded":
        gated_builtin.release.set()
    assert getattr(gated_builtin, pause).wait(10)
    caller, caller_out = _in_thread(paths.active)
    caller.join(0.2)
    assert caller.is_alive()
    gated_builtin.release.set()
    gated_builtin.register.set()
    importer.join(10)
    caller.join(10)
    assert not importer.is_alive() and not caller.is_alive()
    assert _names(caller_out[0]) == ["a4w4"]
    assert _names(paths.active()) == ["a4w4"]


def test_a_failed_builtin_import_is_retried_by_the_next_call(monkeypatch, tmp_path):
    """The failure reaches the caller, and loading stays undone: a caller that catches
    it and calls again gets the path, not a selection from an empty registry."""
    failures = types.SimpleNamespace(left=1)
    monkeypatch.setitem(sys.modules, "moev2_test_failures", failures)
    (tmp_path / "moev2_flaky_builtin.py").write_text(
        "import moev2_test_failures as failures\n"
        "from aiter.moev2 import paths\n"
        "if failures.left:\n"
        "    failures.left -= 1\n"
        "    raise ImportError('the first import fails')\n"
        "PATH = paths.Path('a4w4', lambda call: True, lambda call: None)\n"
        "paths.register(PATH)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(paths, "_BUILTIN_MODULES", ("moev2_flaky_builtin",))
    monkeypatch.setattr(paths, "_builtin_loaded", False)
    paths.reload()
    _set(monkeypatch, "on")
    try:
        with pytest.raises(ImportError, match="the first import fails"):
            paths.active()
        assert not paths._builtin_loaded
        assert _names(paths.active()) == ["a4w4"]
        assert paths._builtin_loaded
    finally:
        sys.modules.pop("moev2_flaky_builtin", None)


def test_off_selects_nothing_without_importing_the_path_modules(monkeypatch):
    """The rollback works even when a path module cannot be imported."""
    monkeypatch.setattr(paths, "_BUILTIN_MODULES", ("moev2_no_such_module",))
    monkeypatch.setattr(paths, "_builtin_loaded", False)
    _set(monkeypatch, "off")
    paths.reload()
    assert paths.active() == ()
    _set(monkeypatch, "on")
    paths.reload()
    with pytest.raises(ModuleNotFoundError, match="moev2_no_such_module"):
        paths.active()


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

    def run(cmd, *, check, env, cwd):
        first = env["PYTHONPATH"].split(os.pathsep)[0]
        seen.append((cmd, check, env[envs.AITER_MOEV2], cwd, first))
        return type("Result", (), {"returncode": 0})()

    suite = gpu_suites.Suite(("python3", "selected_path_test.py"))
    monkeypatch.setattr(gpu_suites, "PATHS", {"a8w8_pt": (suite,)})
    monkeypatch.setattr(gpu_suites, "SHARED", ())
    monkeypatch.setattr(gpu_suites.subprocess, "run", run)
    monkeypatch.setattr(gpu_suites, "_device", lambda: ("gfx942", 1))
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    assert gpu_suites.main(["--path", "a8w8_pt", "--run"]) == 0
    repo = gpu_suites._REPO
    assert seen == [(suite.cmd, False, "a8w8_pt", repo, str(repo))]


def test_gpu_suite_runner_children_import_this_checkout(monkeypatch, tmp_path):
    """An installed aiter (here a fake on PYTHONPATH) must not stand in for the checkout."""
    from moe_dispatch import gpu_suites

    (tmp_path / "installed" / "aiter").mkdir(parents=True)
    (tmp_path / "installed" / "aiter" / "__init__.py").write_text("")
    script = tmp_path / "suite.py"
    script.write_text(
        "import importlib.util, pathlib, sys\n"
        "origin = pathlib.Path(importlib.util.find_spec('aiter').origin)\n"
        f"sys.exit(0 if origin.parents[1] == pathlib.Path({str(gpu_suites._REPO)!r})"
        " and pathlib.Path.cwd() == origin.parents[1] else 1)\n"
    )
    suite = gpu_suites.Suite((sys.executable, str(script)))
    monkeypatch.setattr(gpu_suites, "PATHS", {"a8w8_pt": (suite,)})
    monkeypatch.setattr(gpu_suites, "SHARED", ())
    monkeypatch.setattr(gpu_suites, "_device", lambda: ("gfx942", 1))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "installed"))
    monkeypatch.chdir(tmp_path)
    assert gpu_suites.main(["--path", "a8w8_pt", "--run"]) == 0


def test_gpu_suite_runner_runs_the_shared_suites_for_every_path(monkeypatch):
    from moe_dispatch import gpu_suites

    seen = []
    own = {name: gpu_suites.Suite(("python3", f"{name}.py")) for name in ("a", "b")}
    shared = gpu_suites.Suite(("python3", "shared.py"))

    def run(cmd, *, check, env, cwd):
        seen.append((cmd, env[envs.AITER_MOEV2]))
        failed = cmd == shared.cmd and env[envs.AITER_MOEV2] == "b"
        return type("Result", (), {"returncode": 1 if failed else 0})()

    monkeypatch.setattr(gpu_suites, "PATHS", {n: (s,) for n, s in own.items()})
    monkeypatch.setattr(gpu_suites, "SHARED", (shared,))
    monkeypatch.setattr(gpu_suites.subprocess, "run", run)
    monkeypatch.setattr(gpu_suites, "_device", lambda: ("gfx942", 1))
    assert gpu_suites.main(["--run"]) == 1
    assert seen == [
        (own["a"].cmd, "a"),
        (shared.cmd, "a"),
        (own["b"].cmd, "b"),
        (shared.cmd, "b"),
    ]


def test_gpu_suite_runner_skips_suites_the_machine_cannot_run(monkeypatch, capsys):
    from moe_dispatch import gpu_suites

    seen = []
    fits = gpu_suites.Suite(("python3", "fits.py"))
    other_arch = gpu_suites.Suite(("python3", "gfx950_only.py"), archs=("gfx950",))
    eight_gpus = gpu_suites.Suite(("torchrun", "multi.py"), gpus=8)

    def run(cmd, *, check, env, cwd):
        seen.append(cmd)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(
        gpu_suites, "PATHS", {"a": (fits, other_arch), "b": (eight_gpus,)}
    )
    monkeypatch.setattr(gpu_suites, "SHARED", ())
    monkeypatch.setattr(gpu_suites.subprocess, "run", run)
    monkeypatch.setattr(gpu_suites, "_device", lambda: ("gfx942", 1))
    assert gpu_suites.main(["--run"]) == 0
    assert seen == [fits.cmd]
    out = capsys.readouterr().out
    assert "SKIPPED on gfx942 with 1 GPU(s): [a] python3 gfx950_only.py" in out
    assert "SKIPPED on gfx942 with 1 GPU(s): [b] torchrun multi.py" in out
    # A run where nothing applies fails instead of passing empty.
    seen.clear()
    assert gpu_suites.main(["--path", "b", "--run"]) == 1
    assert seen == []


def test_gpu_suite_compare_fails_only_where_v2_fails_and_legacy_passes(
    monkeypatch, tmp_path, capsys
):
    from moe_dispatch import gpu_suites

    check = "[checkAllclose atol=0.01 rtol=0.01 "
    # (exit status, checkAllclose lines) per (suite, AITER_MOEV2).
    results = {
        ("both_pass.py", "off"): (0, [f"m 1.5 us, 3 tflops{check}\x1b[32mpassed~"]),
        ("both_pass.py", "a8w8_pt"): (
            0,
            [f"m 9.9 us, 7 tflops{check}\x1b[33mwarning!"],
        ),
        ("v2_fails.py", "off"): (0, []),
        ("v2_fails.py", "a8w8_pt"): (1, []),
        ("legacy_fails.py", "off"): (1, [f"m{check}\x1b[32mpassed~"]),
        ("legacy_fails.py", "a8w8_pt"): (1, []),
    }
    seen = []

    def run(cmd, *, check, env, cwd, stdout, stderr):
        mode = env[envs.AITER_MOEV2]
        seen.append((cmd[1], mode, env["PYTHONPATH"].split(os.pathsep)[0], cwd))
        status, lines = results[(cmd[1], mode)]
        stdout.write("".join(f"[aiter] {ln}\x1b[0m]\n" for ln in lines))
        return type("Result", (), {"returncode": status})()

    suites = tuple(
        gpu_suites.Suite(("python3", name))
        for name in ("both_pass.py", "v2_fails.py", "legacy_fails.py")
    )
    monkeypatch.setattr(gpu_suites, "PATHS", {**gpu_suites.PATHS, "a8w8_pt": suites})
    monkeypatch.setattr(gpu_suites, "SHARED", ())
    monkeypatch.setattr(gpu_suites.subprocess, "run", run)
    monkeypatch.setattr(gpu_suites, "_device", lambda: ("gfx942", 1))
    monkeypatch.setattr(paths, "registered", lambda: {"a8w8_pt": None})
    monkeypatch.setenv(envs.AITER_MOEV2, "off")
    argv = ["--path", "a8w8_pt", "--compare", "--log-dir", str(tmp_path)]
    assert gpu_suites.main(argv) == 1
    repo = str(gpu_suites._REPO)
    assert seen == [
        (name, mode, repo, gpu_suites._REPO)
        for name in ("both_pass.py", "v2_fails.py", "legacy_fails.py")
        for mode in ("off", "a8w8_pt")
    ]
    out = capsys.readouterr().out
    assert "FAILED: [a8w8_pt] python3 v2_fails.py  # passes with off" in out
    assert out.count("FAILED:") == 1
    # The timings differ between the runs; only the outcome counts as a difference.
    assert "checkAllclose: off 1 passed; v2 1 warning" in out
    assert "off passed, v2 warning: m [atol=0.01 rtol=0.01]" in out
    assert "the runs made 1 and 0 checks; compare the logs" in out
    assert (tmp_path / "a8w8_pt.0.off.log").exists()
    assert (tmp_path / "a8w8_pt.0.a8w8_pt.log").exists()


def test_gpu_suite_compare_needs_an_implemented_path(monkeypatch, capsys):
    from moe_dispatch import gpu_suites

    def run(*args, **kwargs):
        raise AssertionError("nothing may run")

    monkeypatch.setattr(gpu_suites.subprocess, "run", run)
    monkeypatch.setattr(gpu_suites, "_device", lambda: ("gfx942", 1))
    missing = sorted(set(paths.PLANNED) - set(paths.registered()))
    assert gpu_suites.main(["--path", missing[0], "--compare"]) == 1
    assert "is not implemented: nothing to compare" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        gpu_suites.main(["--compare"])


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
