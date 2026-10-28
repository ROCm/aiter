# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU tests of the FlyDSL AOT modules that replay the runtime (replay.py).

The end-to-end test builds one tuned GEMM row into an empty cache, then
checks that ``--check`` passes on that cache and fails on an empty one. No GPU
is used: the replay processes hide every device.
"""

import csv
import os
import subprocess
import sys
import types

import flydsl.compiler as flyc
import flydsl.expr as fx
import pytest
from flydsl.expr.typing import Constexpr

from aiter.aot.flydsl import gemm, replay
from aiter.aot.flydsl.dry_run import GpuRuntimeReached, dry_run


@pytest.mark.parametrize("path", gemm.DEFAULT_CSVS)
def test_every_default_gemm_table_has_a_caller(path):
    assert gemm.table_of(path) is not None, path


def test_table_of_takes_the_longest_name():
    assert gemm.table_of("/x/bf16_tuned_gemm.csv").name == "bf16_tuned_gemm"
    assert (
        gemm.table_of("/x/dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv").name
        == "a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm"
    )
    assert gemm.table_of("/x/a8w8_tuned_batched_gemm.csv") is None


def _write_table(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bf16_row(m, kernel_name, libtype="flydsl"):
    return {
        "gfx": "gfx950",
        "cu_num": "256",
        "M": str(m),
        "N": "6144",
        "K": "2048",
        "bias": "False",
        "dtype": "torch.bfloat16",
        "outdtype": "torch.float32",
        "scaleAB": "False",
        "bpreshuffle": "False",
        "libtype": libtype,
        "solidx": "0",
        "splitK": "1",
        "us": "1.0",
        "kernelName": kernel_name,
    }


_HGEMM = (
    "flydsl_hgemm_abf16_wbf16_fp32_t16x16x128x5_ks1_w1x1x1_bias0_ktail0_gm4_pft_gfx950"
)


def test_gemm_rows_serve_down_to_the_previous_tuned_m(tmp_path):
    path = tmp_path / "bf16_tuned_gemm.csv"
    _write_table(
        path,
        [_bf16_row(1, _HGEMM), _bf16_row(16, "", "hipblaslt"), _bf16_row(64, _HGEMM)],
    )
    jobs = gemm.parse_csv(str(path))
    # Only FlyDSL rows replay, but every row bounds its neighbours' ranges.
    assert [(job["m_low"], job["m"]) for job in jobs] == [(1, 1), (17, 64)]


def test_mxscale_x_scale_block_pairs_with_the_weights():
    table = next(t for t in gemm.TABLES if t.name.startswith("a8w8_blockscale_mx"))
    job = {"gfx": "gfx950", "cu_num": 256, "w_scale_block": "128x128"}
    assert gemm._variants(table, job) == [{"x_block": 128}]
    for block in ("32x32", "1x32"):
        assert gemm._variants(table, {**job, "w_scale_block": block}) == [
            {"x_block": 32}
        ]
    # A blank cell, or the 0 a config merge fills a missing column with, holds
    # 128x128 rows.
    for blank in ("", "0"):
        assert gemm._variants(table, {**job, "w_scale_block": blank}) == [
            {"x_block": 128}
        ]


def _thrower(error):
    raise error


def _raise_in(filename, error):
    """``error`` as raised from a frame of ``filename``."""
    code = _thrower.__code__.replace(co_filename=filename)
    try:
        types.FunctionType(code, globals())(error)
    except BaseException as raised:  # noqa: BLE001
        return raised
    raise AssertionError("not raised")


def test_errors_outside_flydsl_are_another_backends():
    flydsl_dir = os.path.join(os.sep, "site-packages", "flydsl", "compiler", "x.py")
    triton_dir = os.path.join(os.sep, "site-packages", "triton", "runtime", "x.py")
    assert replay._in_flydsl(_raise_in(flydsl_dir, RuntimeError("x")))
    assert not replay._in_flydsl(_raise_in(triton_dir, RuntimeError("x")))
    # This package's own frames do not count as FlyDSL code.
    aot_file = os.path.join(replay._AOT_DIR, "x.py")
    assert not replay._in_flydsl(_raise_in(aot_file, RuntimeError("x")))
    assert issubclass(GpuRuntimeReached, RuntimeError)


def test_python_ops_run_their_body_under_fake_tensors():
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    from aiter.jit.utils import torch_guard

    calls = []

    def fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    @torch_guard.torch_compile_guard(gen_fake=fake)
    def aot_replay_probe_op(x: torch.Tensor) -> torch.Tensor:
        calls.append(x.shape)
        return x + 1

    with FakeTensorMode():
        x = torch.empty(3)
        aot_replay_probe_op(x)
        assert calls == []
        previous = torch_guard.run_python_ops_for_real(True)
        try:
            aot_replay_probe_op(x)
        finally:
            torch_guard.run_python_ops_for_real(previous)
    assert calls == [torch.Size([3])]


@flyc.jit
def _launch_probe(m: fx.Int32, cluster_m: Constexpr[int]):
    raise AssertionError("an identified launch is never traced")


def _fitted_cluster_m(m):
    """The largest cluster_m up to 4 that divides M's 256-row tile count, as
    the gfx1250 GEMMs fit it."""
    tiles = -(-m // 256)
    return next(cm for cm in (4, 3, 2, 1) if tiles % cm == 0)


def test_identified_kernels_compile_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    with dry_run("gfx1250", 256), replay.identified_kernels() as kernels:
        assert _launch_probe(1, 2) is None
        _launch_probe(7, 2)
        _launch_probe(1, 4)
    # A runtime value is no part of a kernel's identity; a constexpr is.
    assert len(kernels) == 2
    assert not any(path.suffix == ".pkl" for path in tmp_path.rglob("*"))


def test_scan_builds_every_kernel_between_equal_ends(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    module = types.ModuleType("aot_replay_scan_probe")
    module.LABEL = "PROBE"
    module.AOT_ARCH_DEFAULT = "gfx1250"
    module.SCAN_ARCHS = ("gfx1250",)
    module.describe = lambda job: f"M={job['m']}"

    def replay_probe(job):
        if job["m"] % 2:
            raise RuntimeError(f"no kernel for an odd M={job['m']}")
        _launch_probe(job["m"], _fitted_cluster_m(job["m"]))

    module.replay = replay_probe
    monkeypatch.setitem(sys.modules, module.__name__, module)

    built = set()

    def replay_job(module, job, probe=False):
        with replay.identified_kernels() as kernels:
            try:
                with dry_run("gfx1250", 256):
                    module.replay(job)
            except RuntimeError as error:
                return {"compile_time": 0.0, "refused": str(error)}, frozenset()
        built.add(_fitted_cluster_m(job["m"]))
        return {"compile_time": 0.0}, frozenset(kernels)

    monkeypatch.setattr(replay, "replay_job", replay_job)
    # M=5120 and M=6144 both fit cluster_m 4, M=5632 fits 2, and the row's
    # smallest M, 4097, is refused.
    job = {
        "kernel_name": "probe",
        "gfx": "gfx1250",
        "cu_num": 256,
        "m": 6144,
        "m_low": 4097,
    }
    outcome = replay._replay_batch(module.__name__, [job], "probe")
    assert built == {1, 2, 3, 4}
    assert any(probe.get("refused") for probe in outcome["probes"])


def _gemm_aot(table, cache_dir, *extra):
    env = dict(os.environ, FLYDSL_RUNTIME_CACHE_DIR=str(cache_dir), AITER_MAX_JOBS="1")
    env.pop("GPU_ARCHS", None)
    return subprocess.run(
        [sys.executable, "-m", "aiter.aot.flydsl.gemm", "--csv", str(table), *extra],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )


def test_gemm_replay_builds_what_check_then_finds(tmp_path):
    table = tmp_path / "bf16_tuned_gemm.csv"
    _write_table(table, [_bf16_row(1, _HGEMM)])
    cache = tmp_path / "cache"

    built = _gemm_aot(table, cache)
    assert built.returncode == 0, built.stdout + built.stderr
    assert any(
        name.endswith(".pkl") for _, _, names in os.walk(cache) for name in names
    )

    checked = _gemm_aot(table, cache, "--check")
    assert checked.returncode == 0, checked.stdout + checked.stderr

    missing = _gemm_aot(table, tmp_path / "empty", "--check")
    assert missing.returncode != 0
    assert "FLYDSL_RUNTIME_RUN_ONLY=1" in missing.stdout + missing.stderr
