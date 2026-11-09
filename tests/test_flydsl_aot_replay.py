"""FlyDSL AOT replay: how it hooks into FlyDSL's launches, which errors at a
size inside a row's range it may report as the runtime refusing that size, and
the command line its per-arch processes get.

These run on CPU: each launch happens inside ``dry_run`` and compiles nothing.
The last test needs a gfx950 GPU: it serves a real run-only process from an
AOT-built cache.
"""

import csv
import os
import pathlib
import signal
import subprocess
import sys
import time
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
from aiter.aot.flydsl.dry_run import dry_run

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


def test_the_dry_run_answers_a_launch_s_raw_stream_query(monkeypatch):
    from aiter.ops.flydsl.kernels import tensor_shim

    def no_gpu(_index):
        raise RuntimeError("HIP error: no ROCm-capable device is detected")

    monkeypatch.setattr(tensor_shim, "_get_current_raw_stream", no_gpu)
    with dry_run("gfx950", 256):
        assert isinstance(tensor_shim.current_stream_arg(), fx.Stream)


def test_the_dry_run_builds_capture_scoped_scratch_without_asking_hip(monkeypatch):
    """A split-K GEMM's cached semaphores ask HIP which capture the stream is in;
    with the GPUs hidden that fails, and a replay never captures."""
    from aiter.utility import graph_alloc

    def no_gpu(_stream):
        raise RuntimeError("hipStreamGetCaptureInfo failed with error 100")

    monkeypatch.setattr(graph_alloc, "_capture_id", no_gpu)
    calls = []

    @graph_alloc.capture_scoped_cache
    def zeros(device):
        calls.append(device)
        return torch.zeros(4, dtype=torch.int32, device=device)

    with dry_run("gfx950", 256):
        device = torch.device("cuda", 0)
        assert zeros(device) is zeros(device)
    assert calls == [device]


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


def test_a4w4_csv_rows_reach_flydsl_through_public_replay(monkeypatch):
    from aiter.aot.flydsl import gemm
    from aiter.ops.gemm_op_a4w4 import get_GEMM_config

    path = (
        _REPO_ROOT
        / "aiter/configs/model_configs/narrow_n_a4w4_blockscale_tuned_gemm.csv"
    )
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    expected = {
        (int(row["M"]), int(row["N"]), int(row["K"]), row["kernelName"])
        for row in rows
        if row["kernelName"].startswith("flydsl_")
    }
    jobs = gemm.parse_csv(str(path))
    assert len(jobs) == len(expected) == 21
    assert {(j["m"], j["n"], j["k"], j["kernel_name"]) for j in jobs} == expected

    monkeypatch.setenv("AITER_CONFIG_GEMM_A4W4", str(path))
    monkeypatch.delattr(get_GEMM_config, "gemm_dict", raising=False)
    monkeypatch.delattr(get_GEMM_config, "has_gfx", raising=False)
    get_GEMM_config.cache_clear()
    try:
        with dry_run("gfx950", 256, python_ops=True):
            for job in jobs:
                with replay.identified_kernels() as kernels:
                    gemm.replay(job)
                assert kernels, job
    finally:
        get_GEMM_config.cache_clear()


# One split-K A16W16 row (dsv3_bf16_tuned_gemm.csv, M=2 N=256 K=7168, ks7); the
# CU count is filled in from the live device.
_HGEMM_CSV = (
    "gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,"
    "splitK,us,kernelName,err_ratio,tflops,bw\n"
    "gfx950,{cu_num},2,256,7168,False,torch.bfloat16,torch.bfloat16,False,False,"
    "flydsl,312,7,5.1366,"
    "flydsl_hgemm_abf16_wbf16_bf16_t16x32x128x7_ks7_w1x2x1_bias0_ktail0_gm0_pft_gfx950,"
    "0.0137,1.43,720.26\n"
)
_DEVICE_PROBE = """
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    arch = str(props.gcnArchName).split(":", 1)[0].lower()
    print("device", arch, props.multi_processor_count)
"""
_HGEMM_RUN_ONLY = """
import torch
import aiter.tuned_gemm as tuned_gemm
# gemm_a16w16 picks its backend here, from the arch and CU count it reads itself. A
# missed row falls back to another one and never reads the cache, so count FlyDSL calls.
flydsl_gemm, flydsl_calls = tuned_gemm.solMap["flydsl"], []
def counted(*args, **kwargs):
    flydsl_calls.append(args[0].shape)
    return flydsl_gemm(*args, **kwargs)
tuned_gemm.solMap["flydsl"] = counted
m, n, k = 2, 256, 7168
# seeded values in {-1, 0, 1}: sums stay exact and within bf16's exact integer range
# (|out| <= 161 for this seed), and a dropped K-slice or swapped row/column changes the result
g = torch.Generator().manual_seed(0)
a = torch.randint(-1, 2, (m, k), generator=g).to("cuda", torch.bfloat16)
w = torch.randint(-1, 2, (n, k), generator=g).to("cuda", torch.bfloat16)
ref = (a.float() @ w.float().t()).to(torch.bfloat16)
out = tuned_gemm.gemm_a16w16(a, w, otype=torch.bfloat16)
assert flydsl_calls == [a.shape], f"the tuned row did not reach FlyDSL: {flydsl_calls}"
torch.testing.assert_close(out, ref, rtol=0, atol=0)
"""
# One budget for every child of the test, well inside CI's 10-minute tests/ step.
_BUDGET_S = 240


def _run(name, cmd, env, deadline):
    """Run cmd in its own process group; kill the group (replay workers too) at deadline."""
    proc = subprocess.Popen(
        [sys.executable, *cmd],
        cwd=_REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=max(deadline - time.monotonic(), 1))
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        pytest.fail(
            f"{name} step ran past the test's {_BUDGET_S}s budget:\n{out[-4000:]}"
        )
    assert proc.returncode == 0, f"{name}:\n{out[-4000:]}"
    return out


def test_a16w16_aot_cache_serves_a_run_only_process(tmp_path):
    """A fresh FLYDSL_RUNTIME_RUN_ONLY process must load the kernel the AOT
    replay built for a tuned A16W16 row; a cache miss raises there."""
    deadline = time.monotonic() + _BUDGET_S
    env = dict(
        os.environ,
        FLYDSL_RUNTIME_CACHE_DIR=str(tmp_path / "cache"),
        AITER_CONFIG_GEMM_BF16=str(tmp_path / "bf16_tuned_gemm.csv"),
    )
    for name in ("FLYDSL_RUNTIME_RUN_ONLY", "GPU_ARCHS"):
        env.pop(name, None)
    # Device 0 is read in a child, as gemm_a16w16 reads it, so that this pytest
    # process starts no HIP runtime for the tests that follow.
    probe = _run("device probe", ["-c", _DEVICE_PROBE], env, deadline)
    lines = [line.split() for line in probe.splitlines() if line.startswith("device ")]
    _, arch, cu_num = lines[-1] if lines else (None, None, None)
    if arch != "gfx950":
        pytest.skip(f"needs a gfx950 GPU as device 0, found {arch}")
    # The row takes the CU count the runtime looks rows up with.
    pathlib.Path(env["AITER_CONFIG_GEMM_BF16"]).write_text(
        _HGEMM_CSV.format(cu_num=cu_num)
    )
    # GPU_ARCHS picks the arch the replay builds for. An inference process has
    # none, so the run-only child detects its arch as inference does.
    aot = ["-m", "aiter.aot.flydsl.gemm", "--csv", env["AITER_CONFIG_GEMM_BF16"]]
    _run("AOT", aot, {**env, "GPU_ARCHS": "gfx950"}, deadline)
    run_only = ["-c", _HGEMM_RUN_ONLY]
    _run("RUN_ONLY", run_only, {**env, "FLYDSL_RUNTIME_RUN_ONLY": "1"}, deadline)
