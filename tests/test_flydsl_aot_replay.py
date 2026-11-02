"""FlyDSL AOT replay: how it hooks into FlyDSL's launches, which errors it
reports instead of failing (the runtime refusing a size inside a row's range,
another backend taking the call), and the command line its per-arch processes
get.

These run on CPU: each launch happens inside ``dry_run`` and compiles nothing.
"""

import os
import pathlib
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.jit_function import JitFunction

from aiter.aot.flydsl import replay
from aiter.aot.flydsl.dry_run import TritonLaunchReached, dry_run
from aiter.jit.utils.asm_guard import AsmUnsupportedError

_JOB = {"kernel_name": "launch", "gfx": "gfx950", "cu_num": 256, "m": 64}


@pytest.fixture
def run_only(tmp_path, monkeypatch):
    """An empty cache that every launch must hit: a real launch derives its
    key, then fails on the miss."""
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "1")


def _launcher():
    # A JitFunction binds its cache directory on its first launch.
    @flyc.jit
    def launch(a: fx.Tensor, n: fx.Constexpr[int]):
        pass

    return launch


def _module(replay_job, refusals=()):
    return SimpleNamespace(
        replay=replay_job, describe=lambda job: f"M={job['m']}", REFUSALS=refusals
    )


def _raising(error):
    def replay_job(job):
        raise error

    return _module(replay_job)


def _keyless(self, args, kwargs, **_options):
    """A FlyDSL launch that no longer derives its key in _build_full_cache_key."""


def test_identified_kernels_are_the_keys_launches_look_up(run_only):
    launch = _launcher()
    with dry_run("gfx950", 256):
        a = torch.empty(8, device="cuda")
        with (
            replay.requested_kernels() as looked_up,
            pytest.raises(RuntimeError, match="no usable AOT cache"),
        ):
            launch(a, 1)
        with replay.identified_kernels() as identified:
            assert launch(a, 1) is None
    assert len(looked_up) == 1
    assert identified == looked_up


def test_a_launch_that_derives_no_key_fails_the_replay(run_only):
    launch = _launcher()
    module = _module(lambda job: launch(torch.empty(8, device="cuda"), job["m"]))
    with mock.patch.object(JitFunction, "_compile_or_execute", _keyless):
        with pytest.raises(replay.ReplayHookBroken):
            replay._kernel_sets(module, _JOB, range(1, 4))
        result, kernels = replay.replay_job(module, _JOB, probe=True)
    assert result["compile_time"] is None
    assert kernels is None


def test_a_probe_error_is_a_refusal_only_where_the_module_declares_it():
    def refuse(job):
        raise RuntimeError(f"no config for M={job['m']}")

    declared = _module(refuse, refusals=(r"^RuntimeError: no config for M=",))
    result, kernels = replay.replay_job(declared, _JOB, probe=True)
    assert result["refused"] == "RuntimeError: no config for M=64"
    assert kernels == frozenset()

    for module, probe in ((declared, False), (_module(refuse), True)):
        result, kernels = replay.replay_job(module, _JOB, probe=probe)
        assert result["compile_time"] is None
        assert kernels is None


def test_the_dry_run_stops_a_triton_launch():
    pytest.importorskip("triton")
    from triton.runtime.driver import driver

    with dry_run("gfx950", 256), pytest.raises(TritonLaunchReached):
        driver.active.get_current_stream(0)


def test_only_a_call_another_backend_takes_passes_without_a_kernel():
    for error in (TritonLaunchReached("launch"), AsmUnsupportedError("asm")):
        result, kernels = replay.replay_job(_raising(error), _JOB)
        assert result["other_backend"]
        assert kernels == frozenset()
    result, kernels = replay.replay_job(_raising(RuntimeError("launch")), _JOB)
    assert result["compile_time"] is None
    assert kernels is None


def test_per_arch_replays_get_the_command_line(monkeypatch):
    argv = ["--shape", "12:12:192:128", "--check"]
    forwarded = []

    def run(module_name, jobs, argv, env=None):
        forwarded.append(argv)
        return 0

    monkeypatch.setattr(sys, "argv", ["fmha_fp8", *argv])
    monkeypatch.setattr(replay, "run", run)
    # main() sets FLYDSL_RUNTIME_RUN_ONLY for --check.
    with mock.patch.dict(os.environ), pytest.raises(SystemExit) as exited:
        replay.main("aiter.aot.flydsl.fmha_fp8", "FP8 FMHA AOT")
    assert exited.value.code == 0
    assert forwarded == [argv]
