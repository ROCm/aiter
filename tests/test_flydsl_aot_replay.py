"""FlyDSL AOT replay: how it hooks into FlyDSL's launches, which errors at a
size inside a row's range it may report as the runtime refusing that size, and
the command line its per-arch processes get.

These run on CPU: each launch happens inside ``dry_run`` and compiles nothing.
"""

import csv
import itertools
import os
import pathlib
import subprocess
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


def test_the_dry_run_pads_an_mxfp4_scale_with_the_gpus_hidden():
    """``shuffle_scale_a16w4`` pads a scale whose K/32 is not a multiple of 8
    with ``padded[:, :k] = scale``. On torch 2.9.1 the bindings of
    ``__setitem__`` and ``copy_`` take a device guard, which with the GPUs
    hidden, as a replay process hides them, raises hipErrorNoDevice."""
    from aiter.aot.flydsl.common import replay_env

    code = (
        "import torch\n"
        "from aiter import dtypes\n"
        "from aiter.aot.flydsl.dry_run import dry_run\n"
        "from aiter.ops.shuffle import shuffle_scale_a16w4\n"
        "with dry_run('gfx950', 256):\n"
        "    scale = torch.empty(2 * 3584, 12, dtype=dtypes.fp8_e8m0, device='cuda')\n"
        "    assert shuffle_scale_a16w4(scale, 2, False).shape == (2 * 3584, 16)\n"
        "    scale[0] = 1.0\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        env=replay_env("gfx950"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]


def test_the_dry_run_answers_an_assignment_through_an_index(monkeypatch):
    """torch 2.9.1's guarded ``__setitem__`` binding, simulated so that a torch
    without the guard runs the stand-in too. The scale shuffles' K padding, the
    int4 weight packing and MegaMoE's peer pointer table assign this way."""

    def no_gpu(*_args):
        raise RuntimeError("HIP error: no ROCm-capable device is detected")

    monkeypatch.setattr(torch.Tensor, "__setitem__", no_gpu)
    with dry_run("gfx950", 256):
        scale = torch.empty((4, 16), device="cuda", dtype=torch.uint8)
        scale[:, :12] = torch.empty((4, 12), device="cuda", dtype=torch.uint8)
        scale[:, 0] = scale[:, 1] | (scale[:, 2] << 4)
        table = torch.empty(8, device="cuda", dtype=torch.int64)
        table[3] = 1 << 40
        assert scale.shape == (4, 16)
        assert table.shape == (8,)


def test_the_dry_run_gives_each_storage_its_own_address_range():
    """Host code compares pointers and the byte ranges they span, so fake
    storages must be disjoint, aligned allocations, as on a GPU."""
    with dry_run("gfx1250", 256):
        tensors = [
            torch.empty(shape, dtype=dtype, device="cuda")
            for shape, dtype in (
                ((1, 4096), torch.bfloat16),
                ((128, 4096), torch.bfloat16),
                ((1, 128), torch.bfloat16),
                ((3,), torch.uint8),
                ((8, 4096), torch.float32),
            )
        ]
        spans = [
            (t.data_ptr(), t.data_ptr() + t.untyped_storage().nbytes()) for t in tensors
        ]
    assert all(start % 4096 == 0 for start, _ in spans)
    for (start_a, end_a), (start_b, end_b) in itertools.combinations(spans, 2):
        assert end_a <= start_b or end_b <= start_a


def test_a_view_keeps_its_storage_s_address_under_the_dry_run():
    with dry_run("gfx1250", 256):
        base = torch.empty((8, 128), dtype=torch.bfloat16, device="cuda")
        assert base.data_ptr() == base.data_ptr()
        assert base[2:].data_ptr() == base.data_ptr() + 2 * 128 * 2
        assert base[:, 64:].data_ptr() == base.data_ptr() + 64 * 2
        assert base.t().data_ptr() == base.data_ptr()
        assert torch.empty(0, device="cuda").data_ptr() == 0


@pytest.mark.parametrize(
    ("m", "block_k", "stages"), [(1, 128, 5), (2, 256, 3), (4, 512, 2), (8, 512, 2)]
)
def test_a_split_k_tdm_gemm_replay_reaches_flydsl(m, block_k, stages):
    """The gfx1250 TDM GEMM refuses an output that overlaps its inputs. The
    fresh output of a replay must pass that check, as on a GPU (#8134: the
    split-K bf16_tuned_gemm rows of Qwen3.5-397B, N=128 K=4096)."""
    from aiter.ops.flydsl.gemm_a16w16_tdm import gemm_a16w16_tdm

    with dry_run("gfx1250", 256):
        a = torch.empty((m, 4096), dtype=torch.bfloat16, device="cuda")
        b = torch.empty((128, 4096), dtype=torch.bfloat16, device="cuda")
        with replay.identified_kernels() as kernels:
            gemm_a16w16_tdm(
                a,
                b.t(),
                block_m=16,
                block_n=16,
                block_k=block_k,
                stages=stages,
                m_waves=1,
                n_waves=1,
                split_k=8,
            )
    assert len(kernels) == 1


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("with_out", [False, True])
def test_mm_fallback_with_fp32_output_under_dry_run(dtype, with_out):
    # The smallest M probed for kimik3_bf16_tuned_gemm has no tuned config
    # and reaches torch.mm with out_dtype, whose torch 2.9.1 meta is broken.
    with dry_run("gfx950", 256):
        a = torch.empty((1, 7168), device="cuda", dtype=dtype)
        w = torch.empty((896, 7168), device="cuda", dtype=dtype)
        out = torch.empty((1, 896), device="cuda", dtype=torch.float32)
        result = torch.mm(
            a, w.t(), out_dtype=torch.float32, **({"out": out} if with_out else {})
        )
        assert result.shape == (1, 896)
        assert result.dtype == torch.float32
        assert result.device == a.device
        if with_out:
            assert result is out


def test_mm_fallback_still_checks_shape_and_output_dtype():
    with dry_run("gfx950", 256):
        a = torch.empty((1, 16), device="cuda", dtype=torch.bfloat16)
        w = torch.empty((8, 16), device="cuda", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="same reduction dim"):
            torch.mm(a, w, out_dtype=torch.float32)
        with pytest.raises(RuntimeError, match="out_dtype"):
            torch.mm(a, w.t(), out_dtype=torch.float64)


@pytest.mark.parametrize("raise_in_body", [False, True])
def test_legacy_mm_meta_is_restored_after_dry_run(raise_in_body):
    import inspect

    from torch._meta_registrations import meta_mm
    from torch._subclasses.fake_tensor import FakeTensorMode

    if (
        not hasattr(torch.ops.aten.mm, "dtype")
        or "out_dtype" in inspect.signature(meta_mm).parameters
    ):
        pytest.skip("torch already has out_dtype meta support")

    def mm():
        a = torch.empty((1, 16), device="cuda", dtype=torch.bfloat16)
        w = torch.empty((8, 16), device="cuda", dtype=torch.bfloat16)
        return torch.mm(a, w.t(), out_dtype=torch.float32)

    with FakeTensorMode(), pytest.raises(TypeError, match="meta_mm"):
        mm()
    from contextlib import nullcontext

    expected = (
        pytest.raises(RuntimeError, match="end replay")
        if raise_in_body
        else nullcontext()
    )
    with expected, dry_run("gfx950", 256):
        assert mm().dtype == torch.float32
        if raise_in_body:
            raise RuntimeError("end replay")
    with FakeTensorMode(), pytest.raises(TypeError, match="meta_mm"):
        mm()


@pytest.mark.parametrize("with_out", [False, True])
def test_mm_dtype_backport_runs_against_a_simulated_legacy_meta(monkeypatch, with_out):
    # On a torch whose meta_mm takes out_dtype (CI's), the backport would never
    # run. Simulate torch 2.9.1's two-argument meta_mm and its failure, so the
    # backport is exercised, and restored, on every torch.
    import torch._meta_registrations as meta_registrations
    from torch._subclasses import fake_impls
    from torch._subclasses.fake_tensor import FakeTensorMode

    def meta_mm(a, b):
        raise TypeError("meta_mm() takes 2 positional arguments but 3 were given")

    def legacy(fake_mode, func, *args, **kwargs):
        return meta_mm(*args, **kwargs)

    monkeypatch.setattr(meta_registrations, "meta_mm", meta_mm)
    for overload in (torch.ops.aten.mm.dtype, torch.ops.aten.mm.dtype_out):
        monkeypatch.setitem(fake_impls.op_implementations_dict, overload, legacy)
    FakeTensorMode.cache_clear()

    def mm():
        a = torch.empty((1, 16), device="cuda", dtype=torch.bfloat16)
        w = torch.empty((8, 16), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((1, 8), device="cuda", dtype=torch.float32)
        kwargs = {"out": out} if with_out else {}
        return torch.mm(a, w.t(), out_dtype=torch.float32, **kwargs), out

    with FakeTensorMode(), pytest.raises(TypeError, match="meta_mm"):
        mm()
    with dry_run("gfx950", 256):
        result, out = mm()
        assert (result.shape, result.dtype) == ((1, 8), torch.float32)
        if with_out:
            assert result is out
    with FakeTensorMode(), pytest.raises(TypeError, match="meta_mm"):
        mm()


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


def test_int8_csv_rows_reach_the_4wave_kernel_through_public_replay(
    tmp_path, monkeypatch
):
    from aiter.aot.flydsl import gemm
    from aiter.ops import gemm_op_a8w8
    from aiter.ops.flydsl import gemm_a8w8_bpreshuffle_4wave as four_wave

    source = _REPO_ROOT / "aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv"
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [
            row
            for row in reader
            if row["q_dtype_w"] == "torch.int8" and row["libtype"] == "flydsl"
        ]
        fieldnames = reader.fieldnames
    assert len(rows) == 3
    path = tmp_path / "a8w8_bpreshuffle_tuned_gemm.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    jobs = gemm.parse_csv(str(path))
    assert {(j["m"], j["n"], j["k"], j["kernel_name"]) for j in jobs} == {
        (int(r["M"]), int(r["N"]), int(r["K"]), r["kernelName"]) for r in rows
    }

    launched = []
    run_4wave = four_wave.run_gemm_a8w8_bpreshuffle_4wave

    def record(*args, **kwargs):
        launched.append(args[5])
        return run_4wave(*args, **kwargs)

    monkeypatch.setattr(four_wave, "run_gemm_a8w8_bpreshuffle_4wave", record)
    monkeypatch.setenv("AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE", str(path))
    gemm_op_a8w8.get_GEMM_config_with_quant_type.cache_clear()
    try:
        with dry_run("gfx950", 256, python_ops=True):
            for job in jobs:
                with replay.identified_kernels() as kernels:
                    gemm.replay(job)
                assert kernels, job
    finally:
        gemm_op_a8w8.get_GEMM_config_with_quant_type.cache_clear()
    assert launched == [job["kernel_name"] for job in jobs]


def test_4wave_mxfp8_csv_rows_reach_flydsl_through_public_replay(tmp_path):
    # A 4-wave row used to fail its replay: the scale shuffle's Triton launch is another
    # backend's, and the dry run's Triton driver rejects it from under aiter/ops/flydsl.
    # Run as a wheel build's replay does, in a process with the GPUs hidden.
    from aiter.aot.flydsl.common import replay_env

    source = (
        _REPO_ROOT
        / "aiter/configs/model_configs/minimax_m3_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv"
    )
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [
            row for row in reader if row["kernelName"].startswith("flydsl_mxfp8_4w_")
        ]
        fieldnames = reader.fieldnames
    assert rows
    path = tmp_path / "a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    code = (
        "import sys\n"
        "from aiter.aot.flydsl import gemm, replay\n"
        "from aiter.aot.flydsl.dry_run import dry_run\n"
        "from aiter.ops.flydsl import gemm_a8w8_mxfp8_4wave as four_wave\n"
        "jobs = gemm.parse_csv(sys.argv[1])\n"
        f"assert len(jobs) == {len(rows)}, len(jobs)\n"
        "launched = []\n"
        "run_4wave = four_wave.run_gemm_a8w8_mxfp8_4wave\n"
        "def record(*args, **kwargs):\n"
        "    launched.append(args[5])\n"
        "    return run_4wave(*args, **kwargs)\n"
        "four_wave.run_gemm_a8w8_mxfp8_4wave = record\n"
        "with dry_run('gfx950', 256, python_ops=True):\n"
        "    for job in jobs:\n"
        "        with replay.identified_kernels() as kernels:\n"
        "            gemm.replay(job)\n"
        "        assert kernels, job\n"
        "assert launched == [job['kernel_name'] for job in jobs], launched\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        cwd=_REPO_ROOT,
        env={
            **replay_env("gfx950"),
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE": str(path),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]


def test_gemm_jobs_of_a_shape_share_its_tuned_sizes():
    """A table of n rows per shape must not hold n**2 tuned sizes: every
    replay process holds the jobs, and every worker forked from it."""
    from aiter.aot.flydsl import gemm

    path = (
        _REPO_ROOT
        / "aiter/configs/model_configs/narrow_n_a4w4_blockscale_tuned_gemm.csv"
    )
    jobs = gemm.parse_csv(str(path))
    by_shape = {}
    for job in jobs:
        shape = (job["n"], job["k"], job["gfx"], job["cu_num"])
        by_shape.setdefault(shape, []).append(job)
    assert any(len(shape_jobs) > 1 for shape_jobs in by_shape.values())
    for shape_jobs in by_shape.values():
        tuned = shape_jobs[0]["tuned"]
        assert {job["m"] for job in shape_jobs} <= {m for m, _, _ in tuned}
        assert all(job["tuned"] is tuned for job in shape_jobs)


def _collects_and_reports(**_job):
    import gc

    gc.collect()
    return {"kernel_name": "k", "compile_time": 0.0, "frozen": gc.get_freeze_count()}


def test_fork_pool_workers_inherit_a_frozen_heap_and_the_parent_gets_it_back():
    import gc

    from aiter.aot.flydsl import common

    assert gc.get_freeze_count() == 0
    with mock.patch.object(common, "get_max_workers", return_value=2):
        results = common.run_jobs_parallel(
            _collects_and_reports, [{"kernel_name": "k"}] * 3
        )
    assert [r["compile_time"] for r in results] == [0.0] * 3
    assert all(r["frozen"] > 0 for r in results)
    assert gc.get_freeze_count() == 0


def test_fork_pool_leaves_a_caller_s_frozen_heap_frozen():
    import gc

    from aiter.aot.flydsl import common

    gc.freeze()
    try:
        frozen = gc.get_freeze_count()
        made_after = [[] for _ in range(1000)]
        with common._lean_fork_parent("fork"):
            # Nothing more is frozen: it could not be unfrozen on its own.
            assert gc.get_freeze_count() == frozen
        assert gc.get_freeze_count() == frozen
        del made_after
    finally:
        gc.unfreeze()


def test_only_a_fork_pool_freezes_or_takes_the_fork_budget():
    import gc

    from aiter.aot.flydsl import common

    with common._lean_fork_parent("forkserver"):
        assert gc.get_freeze_count() == 0
    with (
        mock.patch.object(common, "get_fork_worker_count_for", return_value=9),
        mock.patch.object(common, "get_worker_count_for", return_value=4),
    ):
        assert common.get_max_workers(100) == 9
        assert common.get_max_workers(100, "fork") == 9
        assert common.get_max_workers(100, "forkserver") == 4
        assert common.get_max_workers(100, "spawn") == 4
