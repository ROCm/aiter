"""FlyDSL AOT replay: how it hooks into FlyDSL's launches, which errors at a
size inside a row's range it may report as the runtime refusing that size, and
the command line its per-arch processes get.

These run on CPU: each launch happens inside ``dry_run`` and compiles nothing.
"""

import csv
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


def test_a_scan_identifies_only_the_sizes_the_module_names():
    seen = []
    module = _module(lambda job: seen.append(job["m"]))
    module.scan_sizes = lambda job, low: [low, 7, job["m"] - 1]
    with mock.patch.object(replay, "replay_job") as replayed:
        replay._scan(module, {**_JOB, "m": 16}, 3, frozenset())
    assert seen == [3, 7, 15]
    # No size requested a kernel the row's replay lacks: none is replayed.
    replayed.assert_not_called()


def _padded_m(m, gl, n=7168):
    """getPaddedM (csrc/py_itfs_cu/gemm_common.cu)."""
    pow2 = 1 << (m - 1).bit_length()
    if gl == 1:
        return 8192 if m > 8192 and n > 4096 else pow2
    if m < 16:
        return pow2
    limits = ((256, 16), (1024, 32), (4096, 64))
    grain = next((g for limit, g in limits if m <= limit), 128)
    return -(-m // grain) * grain


def test_gemm_scan_splits_a_range_where_the_padded_lookup_changes():
    from aiter.aot.flydsl import gemm

    # Above 8192 (N > 4096) gl=1 pads to 8192; only the top 128-row block pads
    # (gl=0) to the row's own M.
    assert gemm._lookup_pieces(_padded_m, 8193, 16384) == [
        (8193, 16257),
        (16257, 16384),
    ]
    # gl=1 changes at each power of two; 100 pads to 112 (gl=0), past the range.
    assert gemm._lookup_pieces(_padded_m, 5, 100) == [
        (5, 9),
        (9, 17),
        (17, 33),
        (33, 65),
        (65, 97),
        (97, 100),
    ]


def test_gemm_scan_takes_each_tile_count_ragged_at_both_parities_and_full():
    from aiter.aot.flydsl import gemm

    assert sorted(gemm._tile_sizes(1, 1000, 256, 2)) == [
        *(1, 2, 256),  # count 1
        *(257, 258, 512),  # count 2
        *(769, 770, 999),  # the last count, its tile not filled below 1000
    ]


def test_gemm_scan_covers_the_heuristics_where_a_sparse_table_has_no_row(
    monkeypatch,
):
    from aiter.aot.flydsl import gemm
    from aiter.ops import gemm_op_common

    monkeypatch.setattr(
        gemm_op_common, "get_padded_m", lambda m, n, k, gl: _padded_m(m, gl, n)
    )
    job = {
        "table": "a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm",
        "kernel_name": "flydsl_mxfp8_128_bpreshuffle_wmma_t256x256x128_cm1_cn1",
        "m": 32768,
        "n": 7168,
        "k": 1024,
    }
    blocks = range(33, 127)  # the 256-row blocks in (8448, 32512]
    # Above 8192, gl=1 pads to 8192: with no row there the heuristics pick,
    # and they tell every 256-row block apart.
    sparse = gemm.scan_sizes({**job, "tuned": ((1, 16, 1), (32768, 256, 1))}, 2)
    assert all(any(b * 256 < m <= (b + 1) * 256 for m in sparse) for b in blocks)
    # With a row at 8192 the lookup finds it there, and only that row's
    # tile counts are scanned.
    dense = gemm.scan_sizes(
        {**job, "tuned": ((1, 16, 1), (8192, 256, 1), (32768, 256, 1))}, 8193
    )
    assert not all(any(b * 256 < m <= (b + 1) * 256 for m in dense) for b in blocks)
