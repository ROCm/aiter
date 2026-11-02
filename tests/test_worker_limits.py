"""Worker budget policy: CPU/memory budgets, cgroup traversal, and diagnostics.

``aiter_worker_limits`` imports without ROCm, so these tests run on any host.
ROCm-dependent AOT drivers are covered in ``test_aot_drivers.py``.
"""

import contextlib
import ctypes
import importlib.util
import os
import pathlib
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import aiter_worker_limits as limits

GiB = 1024**3


@contextlib.contextmanager
def _proc_membership(tmp_path, membership, mountinfo):
    """Point the policy at fake ``/proc/self`` files under ``tmp_path``."""
    (tmp_path / "membership").write_text(membership)
    (tmp_path / "mountinfo").write_text(mountinfo.format(root=tmp_path))
    with (
        patch.object(limits, "_PROC_SELF_CGROUP_PATH", str(tmp_path / "membership")),
        patch.object(limits, "_PROC_SELF_MOUNTINFO_PATH", str(tmp_path / "mountinfo")),
    ):
        yield


# --- CPU budget ------------------------------------------------------------


@pytest.mark.parametrize(("cpu_count", "expected"), [(24, 19), (4, 3), (1, 1), (0, 1)])
def test_cpu_budget_applies_eighty_percent_with_a_one_worker_floor(cpu_count, expected):
    with patch.object(limits, "_cgroup_cpu_quota", return_value=None):
        assert limits.get_cpu_worker_budget(cpu_count=cpu_count) == expected


def test_cpu_budget_uses_process_available_cpus():
    with (
        patch.object(limits, "_cgroup_cpu_quota", return_value=None),
        patch.object(limits, "_process_cpu_count", return_value=24),
    ):
        assert limits.get_cpu_worker_budget() == 19
    with (
        patch.object(limits, "_cgroup_cpu_quota", return_value=None),
        patch.object(limits, "_process_cpu_count", return_value=1),
    ):
        assert limits.get_cpu_worker_budget() == 1


def test_process_cpu_count_falls_back_to_affinity():
    with (
        patch.object(os, "process_cpu_count", return_value=None, create=True),
        patch.object(os, "sched_getaffinity", return_value={0, 1, 2, 3}),
    ):
        assert limits._process_cpu_count() == 4


# --- cgroup CPU quota ------------------------------------------------------


@pytest.fixture
def cpu_hierarchy(tmp_path, monkeypatch):
    """A v2 cgroup with a nested job directory and fake /proc metadata."""
    mount = tmp_path / "cgroup"
    child = mount / "job"
    child.mkdir(parents=True)
    membership = tmp_path / "membership"
    mountinfo = tmp_path / "mountinfo"
    membership.write_text("0::/job\n")
    mountinfo.write_text(f"36 25 0:32 / {mount} rw - cgroup2 none rw\n")
    monkeypatch.setattr(limits, "_PROC_SELF_CGROUP_PATH", str(membership))
    monkeypatch.setattr(limits, "_PROC_SELF_MOUNTINFO_PATH", str(mountinfo))
    monkeypatch.setattr(limits, "_process_cpu_count", lambda: 192)
    monkeypatch.setattr(limits, "_host_available_memory_bytes", lambda: 1024**4)
    monkeypatch.delenv("AITER_MAX_JOBS", raising=False)
    monkeypatch.delenv("MAX_JOBS", raising=False)
    return mount, child, membership, mountinfo


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("800000 100000", 6),
        ("250000 100000", 2),
        ("50000 100000", 1),
        ("100000 100000", 1),
        ("max 100000", 153),
        ("0 100000", 153),
        ("100000 0", 153),
        ("-1 100000", 153),
        ("garbage", 153),
        ("100000 -1", 153),
        ("100000 x", 153),
        ("", 153),
        ("800000 100000 extra", 153),
    ],
)
def test_v2_quota_and_invalid_fallback(cpu_hierarchy, raw, expected):
    _, child, _, _ = cpu_hierarchy
    (child / "cpu.max").write_text(raw)
    assert limits.get_cpu_worker_budget() == expected
    assert limits.get_worker_count() == expected


@pytest.mark.parametrize(
    ("quota", "period", "expected"),
    [
        ("800000", "100000", 6),
        ("250000", "100000", 2),
        ("-1", "100000", 153),
        ("100000", "0", 153),
        ("bad", "100000", 153),
    ],
)
def test_v1_cpu_controller_combined_mount(cpu_hierarchy, quota, period, expected):
    mount, child, membership, mountinfo = cpu_hierarchy
    membership.write_text("4:cpu,cpuacct:/container/job\n5:memory:/different\n")
    mountinfo.write_text(
        f"29 23 0:26 /container {mount} rw - cgroup cgroup rw,cpu,cpuacct\n"
    )
    (child / "cpu.cfs_quota_us").write_text(quota)
    (child / "cpu.cfs_period_us").write_text(period)
    assert limits.get_cpu_worker_budget() == expected


def test_ancestor_quota_survives_bad_or_unreadable_child(cpu_hierarchy):
    mount, child, _, _ = cpu_hierarchy
    (mount / "cpu.max").write_text("400000 100000")
    for value in ("max 100000", "bad", "800000 100000"):
        (child / "cpu.max").write_text(value)
        assert limits.get_cpu_worker_budget() == 3
    (child / "cpu.max").unlink()
    assert limits.get_cpu_worker_budget() == 3
    # A tighter child still wins, even with a different period.
    (child / "cpu.max").write_text("125000 50000")
    assert limits._cgroup_cpu_quota() == Fraction(5, 2)
    assert limits.get_cpu_worker_budget() == 2
    # An unreadable child must not hide the finite ancestor.
    original = open

    def read(path, *args, **kwargs):
        if str(path) == str(child / "cpu.max"):
            raise PermissionError("masked")
        return original(path, *args, **kwargs)

    with patch("builtins.open", side_effect=read):
        assert limits.get_cpu_worker_budget() == 3


def test_hybrid_scans_both_mounts_and_decodes_spaces(cpu_hierarchy, tmp_path):
    mount, child, membership, mountinfo = cpu_hierarchy
    v1 = tmp_path / "cpu mount"
    v1.mkdir()
    membership.write_text("0::/job\n4:cpu,cpuacct:/\n")
    escaped = str(v1).replace(" ", r"\040")
    mountinfo.write_text(
        f"36 25 0:32 / {mount} rw - cgroup2 none rw\n"
        f"29 23 0:26 / {escaped} rw - cgroup cgroup rw,cpu,cpuacct\n"
    )
    (child / "cpu.max").write_text("max 100000")
    (v1 / "cpu.cfs_quota_us").write_text("800000")
    (v1 / "cpu.cfs_period_us").write_text("100000")
    assert limits.get_cpu_worker_budget() == 6


def test_affinity_memory_and_user_ceilings_still_apply(cpu_hierarchy, monkeypatch):
    _, child, _, _ = cpu_hierarchy
    (child / "cpu.max").write_text("800000 100000")
    monkeypatch.setattr(limits, "_process_cpu_count", lambda: 4)
    assert limits.get_worker_count() == 3
    monkeypatch.setattr(limits, "_process_cpu_count", lambda: 192)
    monkeypatch.setenv("AITER_MAX_JOBS", "999")
    assert limits.get_worker_count() == 6
    monkeypatch.setenv("AITER_MAX_JOBS", "2")
    assert limits.get_worker_count() == 2
    monkeypatch.delenv("AITER_MAX_JOBS")
    monkeypatch.setenv("MAX_JOBS", "2")
    before = dict(os.environ)
    assert limits.get_compile_worker_count() == 2
    assert limits.get_worker_count() == 6
    assert dict(os.environ) == before
    monkeypatch.setattr(
        limits, "_host_available_memory_bytes", lambda: limits.EST_WORKER_RSS_BYTES
    )
    assert limits.get_worker_count() == 1


def test_quota_is_reread_and_not_exported(cpu_hierarchy):
    _, child, _, _ = cpu_hierarchy
    before = dict(os.environ)
    for raw, expected in [
        ("800000 100000", 6),
        ("250000 100000", 2),
        ("max 100000", 153),
    ]:
        (child / "cpu.max").write_text(raw)
        assert limits.get_worker_count() == expected
    assert dict(os.environ) == before


def test_no_proc_metadata_falls_back_to_affinity(cpu_hierarchy):
    _, _, membership, mountinfo = cpu_hierarchy
    membership.unlink()
    mountinfo.unlink()
    assert limits.get_cpu_worker_budget() == 153


# --- cgroup memory traversal ----------------------------------------------


@pytest.mark.parametrize(
    ("membership", "mountinfo", "expected"),
    [
        (
            "0::/jobs/worker\n",
            "36 25 0:32 / {root}/cgroup rw - cgroup2 none rw\n",
            [("v2", "cgroup/jobs/worker"), ("v2", "cgroup/jobs"), ("v2", "cgroup")],
        ),
        (
            "5:cpu,memory:/docker/worker\n",
            "29 23 0:26 /docker {root}/memory rw - cgroup cgroup rw,memory\n",
            [("v1", "memory/worker"), ("v1", "memory")],
        ),
    ],
    ids=["v2-membership", "v1-mount-root"],
)
def test_cgroup_directories_follow_membership_and_mount_root(
    tmp_path, membership, mountinfo, expected
):
    with _proc_membership(tmp_path, membership, mountinfo):
        assert limits._cgroup_directories("memory") == [
            (kind, str(tmp_path / relative)) for kind, relative in expected
        ]


def test_hybrid_mounts_all_contribute_memory_directories(tmp_path):
    membership = "0::/container\n5:memory:/container\n"
    mountinfo = (
        "36 25 0:32 / {root}/unified rw - cgroup2 none rw\n"
        "29 23 0:26 / {root}/memory rw - cgroup cgroup rw,memory\n"
    )
    for mount, limit_file, usage_file, limit, usage in (
        ("unified/container", "memory.max", "memory.current", 16 * GiB, 1 * GiB),
        (
            "memory/container",
            "memory.limit_in_bytes",
            "memory.usage_in_bytes",
            8 * GiB,
            7 * GiB,
        ),
    ):
        directory = tmp_path / mount
        directory.mkdir(parents=True)
        (directory / limit_file).write_text(str(limit))
        (directory / usage_file).write_text(str(usage))
    with _proc_membership(tmp_path, membership, mountinfo):
        assert limits._cgroup_directories("memory") == [
            ("v2", str(tmp_path / "unified/container")),
            ("v2", str(tmp_path / "unified")),
            ("v1", str(tmp_path / "memory/container")),
            ("v1", str(tmp_path / "memory")),
        ]
        assert limits._cgroup_memory_bound()[0] == 1 * GiB


def test_memory_bound_uses_tightest_finite_ancestor_or_v1_files(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    (child / "memory.max").write_text("max\n")
    (child / "memory.current").write_text(str(GiB))
    (tmp_path / "memory.max").write_text(str(8 * GiB))
    (tmp_path / "memory.current").write_text(str(2 * GiB))
    with patch.object(
        limits,
        "_cgroup_directories",
        return_value=[("v2", str(child)), ("v2", str(tmp_path))],
    ):
        assert limits._cgroup_memory_bound()[0] == 6 * GiB
    (tmp_path / "memory.limit_in_bytes").write_text(str(8 * GiB))
    (tmp_path / "memory.usage_in_bytes").write_text(str(3 * GiB))
    with patch.object(
        limits, "_cgroup_directories", return_value=[("v1", str(tmp_path))]
    ):
        assert limits._cgroup_memory_bound()[0] == 5 * GiB


# --- automatic budget ------------------------------------------------------


@pytest.mark.parametrize(
    ("cpu_budget", "host_available", "cgroup_bound", "memory_budget", "expected"),
    [
        (6, 3 * limits.EST_WORKER_RSS_BYTES, (None, None), 3, 3),
        (3, 10 * GiB, (None, None), 7, 3),
        (51, 4 * limits.EST_WORKER_RSS_BYTES, (None, None), 4, 4),
        (1, 0, (None, None), 1, 1),
        (102, 256 * GiB, (8 * GiB, None), 5, 5),
    ],
    ids=[
        "cpu-and-memory-only",
        "eighty-percent-cpu",
        "available-memory",
        "zero-memory",
        "container-cap",
    ],
)
def test_automatic_budget_is_capped_by_cpu_and_memory(
    cpu_budget, host_available, cgroup_bound, memory_budget, expected
):
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(limits, "get_cpu_worker_budget", return_value=cpu_budget),
        patch.object(
            limits, "_host_available_memory_bytes", return_value=host_available
        ),
        patch.object(limits, "_cgroup_memory_bound", return_value=cgroup_bound),
    ):
        assert limits._automatic_worker_snapshot() == (
            cpu_budget,
            memory_budget,
            None,
        )
        assert limits.get_worker_count() == expected
        assert "AITER_MAX_JOBS" not in os.environ


# --- ceilings --------------------------------------------------------------


@pytest.mark.parametrize(
    ("canonical", "legacy", "generic_expected", "compile_expected"),
    [
        (None, None, 6, 6),
        (None, "2", 6, 2),
        (None, "99", 6, 6),
        (None, "not-an-integer", 6, 6),
        (None, "0", 6, 6),
        (None, "-7", 6, 6),
        ("5", "2", 5, 5),
        ("99", None, 6, 6),
        ("99", "2", 6, 6),
        ("1", None, 1, 1),
        ("0", "2", 1, 1),
        ("-7", "2", 1, 1),
        ("", "2", 6, 6),
    ],
)
def test_worker_ceiling_precedence_without_environment_mutation(
    canonical, legacy, generic_expected, compile_expected
):
    environment = {
        name: value
        for name, value in (("AITER_MAX_JOBS", canonical), ("MAX_JOBS", legacy))
        if value is not None
    }
    with (
        patch.dict(os.environ, environment, clear=True),
        patch.object(
            limits, "_automatic_worker_snapshot", return_value=(8, 6, None)
        ) as snapshot,
    ):
        assert limits.get_worker_count() == generic_expected
        assert limits.get_compile_worker_count() == compile_expected
        assert snapshot.call_count == 2
        assert dict(os.environ) == environment


def test_automatic_worker_budget_is_recomputed_on_every_call():
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(
            limits,
            "_automatic_worker_snapshot",
            side_effect=((102, 180, None), (1, 1, None)),
        ) as automatic_budgets,
    ):
        assert limits.get_worker_count() == 102
        assert limits.get_worker_count() == 1
        assert automatic_budgets.call_count == 2
        assert "AITER_MAX_JOBS" not in os.environ


@pytest.mark.parametrize(
    ("work_count", "explicit", "expected"),
    [
        (0, None, 1),
        (3, None, 3),
        (100, None, 19),
        (100, 4, 4),
        (100, 99, 19),
        (2, 99, 2),
        (100, 0, 1),
        (100, -7, 1),
    ],
)
def test_work_capped_worker_count_never_returns_zero(work_count, explicit, expected):
    with (
        patch.dict(os.environ, {"AITER_MAX_JOBS": "19"}, clear=True),
        patch.object(limits, "_automatic_worker_snapshot", return_value=(32, 32, None)),
    ):
        assert limits.get_worker_count_for(work_count, explicit) == expected


def test_split_worker_budget_keeps_the_global_total_and_a_one_worker_floor():
    assert limits.split_worker_budget(32, 40) == (5, 6)
    assert limits.split_worker_budget(32, 3) == (3, 10)
    # The *global* total is divided, so a short module list still uses the
    # whole machine instead of collapsing to one inner job per module.
    assert limits.split_worker_budget(24, 40) == (5, 4)
    assert limits.split_worker_budget(24, 2) == (2, 12)
    assert limits.split_worker_budget(24, module_count=1) == (1, 24)
    assert limits.split_worker_budget(0, 0) == (1, 1)
    assert limits.split_worker_budget(1, 100) == (1, 1)


# --- legacy MAX_JOBS adoption, ceilings, and GPU worker count ------------


class WorkerCeilingAndGpuPolicyTest(unittest.TestCase):
    def test_standalone_entrypoint_adopts_valid_legacy_max_jobs(self):
        with (
            patch.dict(os.environ, {"MAX_JOBS": "7"}, clear=True),
            self.assertWarns(FutureWarning) as warning,
        ):
            limits.adopt_legacy_max_jobs()
            self.assertEqual(os.environ["MAX_JOBS"], "7")
            self.assertEqual(os.environ["AITER_MAX_JOBS"], "7")
            self.assertIn(
                "use AITER_MAX_JOBS instead", str(warning.warnings[0].message)
            )

    def test_explicit_aiter_max_jobs_prevents_legacy_adoption(self):
        with (
            patch.dict(
                os.environ,
                {"AITER_MAX_JOBS": "3", "MAX_JOBS": "7"},
                clear=True,
            ),
            patch.object(limits.warnings, "warn") as warn,
        ):
            limits.adopt_legacy_max_jobs()

            self.assertEqual(os.environ["AITER_MAX_JOBS"], "3")
            self.assertEqual(os.environ["MAX_JOBS"], "7")
            warn.assert_not_called()

    def test_invalid_legacy_max_jobs_is_not_adopted(self):
        for raw_value in ("", "auto", "0", "-7"):
            with (
                self.subTest(raw_value=raw_value),
                patch.dict(os.environ, {"MAX_JOBS": raw_value}, clear=True),
                patch.object(limits.warnings, "warn") as warn,
            ):
                limits.adopt_legacy_max_jobs()

                self.assertEqual(os.environ["MAX_JOBS"], raw_value)
                self.assertNotIn("AITER_MAX_JOBS", os.environ)
                warn.assert_not_called()

    def test_adopted_legacy_ceiling_remains_clamped_to_live_limits(self):
        with (
            patch.dict(os.environ, {"MAX_JOBS": "99"}, clear=True),
            patch.object(
                limits, "_automatic_worker_snapshot", return_value=(4, 3, None)
            ),
        ):
            with self.assertWarns(FutureWarning):
                limits.adopt_legacy_max_jobs()
            self.assertEqual(limits.get_worker_count(), 3)

    def test_worker_ceiling_scopes_nested_compiler_budget(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                limits, "_automatic_worker_snapshot", return_value=(8, 8, None)
            ),
        ):
            with limits.worker_ceiling(3):
                self.assertEqual(os.environ["AITER_MAX_JOBS"], "3")
                self.assertEqual(limits.get_compile_worker_count(), 3)
            self.assertNotIn("AITER_MAX_JOBS", os.environ)

    def test_worker_ceiling_restores_previous_ceiling_on_error(self):
        with patch.dict(os.environ, {"AITER_MAX_JOBS": "5"}, clear=True):
            with self.assertRaises(RuntimeError), limits.worker_ceiling(2):
                self.assertEqual(os.environ["AITER_MAX_JOBS"], "2")
                raise RuntimeError("boom")
            self.assertEqual(os.environ["AITER_MAX_JOBS"], "5")

    def test_worker_configures_supported_build_controls(self):
        with patch.dict(
            os.environ,
            {
                "AITER_MAX_JOBS": "23",
                "MAX_JOBS": "64",
                "NINJAFLAGS": "-j99",
            },
            clear=True,
        ):
            limits.configure_worker_subprocesses()
            self.assertEqual(os.environ["AITER_MAX_JOBS"], "1")
            self.assertEqual(os.environ["MAX_JOBS"], "64")
            self.assertEqual(os.environ["CMAKE_BUILD_PARALLEL_LEVEL"], "1")
            self.assertEqual(os.environ["MAKEFLAGS"], "-j1")
            self.assertEqual(os.environ["NINJAFLAGS"], "-j99")

    def test_gpu_worker_count_scales_with_visible_devices(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                limits.get_gpu_worker_count(1000, 4), 4 * limits.EST_WORKERS_PER_GPU
            )

    def test_gpu_worker_count_honors_per_device_override(self):
        with patch.dict(os.environ, {"AITER_GPU_WORKERS_PER_DEVICE": "2"}, clear=True):
            self.assertEqual(limits.get_gpu_worker_count(1000, 3), 6)

    def test_gpu_worker_count_is_clamped_by_explicit_ceiling(self):
        with patch.dict(os.environ, {"AITER_MAX_JOBS": "5"}, clear=True):
            self.assertEqual(limits.get_gpu_worker_count(1000, 4), 5)

    def test_gpu_worker_count_is_capped_by_submitted_work(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(limits.get_gpu_worker_count(3, 4), 3)
            self.assertEqual(limits.get_gpu_worker_count(0, 4), 1)

    def test_gpu_worker_count_requires_a_visible_gpu(self):
        for device_count in (0, -1):
            with (
                self.subTest(device_count=device_count),
                self.assertRaisesRegex(
                    RuntimeError, "requires at least one visible GPU"
                ),
            ):
                limits.get_gpu_worker_count(3, device_count)

    def test_gpu_worker_count_uses_runtime_count_without_reparsing_masks(self):
        with patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "0,99"}, clear=True):
            self.assertEqual(
                limits.get_gpu_worker_count(1000, 1), limits.EST_WORKERS_PER_GPU
            )

    def test_worker_does_not_set_unsupported_ninja_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            limits.configure_worker_subprocesses()
            self.assertEqual(os.environ["AITER_MAX_JOBS"], "1")
            self.assertEqual(os.environ["CMAKE_BUILD_PARALLEL_LEVEL"], "1")
            self.assertEqual(os.environ["MAKEFLAGS"], "-j1")
            self.assertNotIn("NINJAFLAGS", os.environ)
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")


# --- shared AOT runner ----------------------------------------------------


def _compile_pointer(config):
    if os.environ.get("AITER_MAX_JOBS") != "1":
        raise RuntimeError("nested compilation was not bounded")
    return ctypes.pointer(ctypes.c_int(config))


def _compile_failure(config):
    raise RuntimeError(f"compile failed: {config}")


class AotRunnerTest(unittest.TestCase):
    def test_real_pool_keeps_ctypes_results_in_children(self):
        with (
            patch.dict(os.environ, {"AITER_MAX_JOBS": "2"}),
            patch.object(limits, "get_worker_count_for", return_value=2),
        ):
            limits.run_configs([1, 2, 3], _compile_pointer)
            self.assertEqual(os.environ["AITER_MAX_JOBS"], "2")

    def test_real_pool_propagates_compile_failures(self):
        with (
            patch.object(limits, "get_worker_count_for", return_value=1),
            self.assertRaisesRegex(RuntimeError, "compile failed: 7"),
        ):
            limits.run_configs([7], _compile_failure)


# --- legacy OPUS builders -------------------------------------------------


@pytest.fixture(
    params=["csrc/opus_gemm/gen_co/build_co.py", "op_tests/opus/device/setup.py"]
)
def builder(request):
    path = _REPO_ROOT / request.param
    spec = importlib.util.spec_from_file_location("opus_jobs_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PoolReached(Exception):
    pass


def _bounded_worker_count(count, explicit=None, cap=6):
    """Stand-in for the shared policy: cap the work-bound budget, then clamp."""
    budget = min(cap, max(1, count))
    if explicit is None:
        return budget
    return min(budget, max(1, explicit))


@pytest.mark.parametrize(
    "flags,expected",
    [([], 6), (["-j2"], 2), (["-j999"], 6), (["-j0"], 1)],
)
def test_cli_reaches_bounded_compilation(
    builder, flags, expected, monkeypatch, tmp_path
):
    monkeypatch.setattr(sys, "argv", [builder.__file__, *flags])
    monkeypatch.setattr(builder, "get_worker_count_for", _bounded_worker_count)
    observed = []

    def pool(*args, **kwargs):
        observed.append(kwargs["max_workers"])
        raise PoolReached

    if hasattr(builder, "build_one"):
        tag = next(iter(builder._PIPELINE_HEADERS))
        monkeypatch.setattr(
            builder,
            "gfx1250_4wave_co_kernels_declared",
            {i: SimpleNamespace(kernel_tag=tag) for i in range(10)},
        )
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--out-dir", str(tmp_path)])

        def serial(*args):
            observed.append(1)
            raise PoolReached

        monkeypatch.setattr(builder, "build_one", serial)
    else:
        monkeypatch.setattr(builder, "_find_hipcc", lambda: "unused")
        monkeypatch.setattr(builder, "_detect_arch", lambda: "gfx1250")
    with (
        patch("concurrent.futures.ThreadPoolExecutor", side_effect=pool),
        patch("concurrent.futures.ProcessPoolExecutor", side_effect=pool),
        pytest.raises(PoolReached),
    ):
        builder.main()
    assert observed == [expected]


def test_device_python_api_and_work_count_ceiling(monkeypatch):
    path = _REPO_ROOT / "op_tests/opus/device/setup.py"
    spec = importlib.util.spec_from_file_location("opus_device_api", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_find_hipcc", lambda: "unused")
    monkeypatch.setattr(module, "_detect_arch", lambda: "gfx1250")
    monkeypatch.setattr(module, "_CU_SOURCES", ["test_mfma_f16.cu"])
    monkeypatch.setattr(
        module,
        "get_worker_count_for",
        lambda count, explicit=None: _bounded_worker_count(count, explicit, cap=8),
    )
    with (
        patch(
            "concurrent.futures.ProcessPoolExecutor", side_effect=PoolReached
        ) as pool,
        pytest.raises(PoolReached),
    ):
        module.build(jobs=99)
    assert pool.call_args.kwargs["max_workers"] == 1


# --- cgroup memory diagnostics --------------------------------------------


@pytest.fixture(autouse=True)
def isolated_diagnostics():
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(limits, "_memory_diagnostic_last_time", None),
        patch.object(limits, "_memory_diagnostic_last_budget", None),
    ):
        yield


@pytest.fixture
def cgroup(tmp_path):
    (tmp_path / "memory.max").write_text(str(64 * 1024**3))
    (tmp_path / "memory.current").write_text(str(52 * 1024**3))
    (tmp_path / "memory.stat").write_text("anon 1073741824\ninactive_file 2147483648\n")
    with (
        patch.object(
            limits, "_cgroup_directories", return_value=[("v2", str(tmp_path))]
        ),
        patch.object(limits, "_host_available_memory_bytes", return_value=1024**4),
        patch.object(limits, "get_cpu_worker_budget", return_value=204),
    ):
        yield tmp_path


def test_real_cgroup_reports_eight_workers_and_cache(cgroup, caplog):
    assert limits.get_worker_count() == 8
    assert limits.get_worker_count() == 8
    assert len(caplog.records) == 1
    message = caplog.text
    for expected in (
        "requested=204",
        "selected=8",
        "limit=64.00",
        "current=52.00",
        "inactive_file",
        "configured_ceiling=unset",
    ):
        assert expected in message


@pytest.mark.parametrize(
    "cpu,memory,ceiling,reported",
    [
        (204, 8, None, True),
        (32, 8, None, True),
        (31, 8, None, False),
        (204, 8, "4", False),
        (4, 2, None, True),
        (1, 1, None, False),
        (204, 1, "1", False),
        (204, 1, "0", False),
        (204, 1, "-3", False),
        (204, 8, "auto", True),
    ],
)
def test_threshold_and_ceiling(cgroup, caplog, cpu, memory, ceiling, reported):
    if ceiling is not None:
        os.environ["AITER_MAX_JOBS"] = ceiling
    _, observation = limits._cgroup_memory_bound()
    with patch.object(
        limits, "_automatic_worker_snapshot", return_value=(cpu, memory, observation)
    ):
        selected = limits.get_worker_count()
    normalized = max(1, int(ceiling)) if ceiling and ceiling != "auto" else cpu
    assert selected == max(1, min(cpu, memory, normalized))
    assert bool(caplog.records) == reported


def test_runtime_legacy_ceiling_and_precedence(cgroup, caplog):
    os.environ["MAX_JOBS"] = "4"
    before = dict(os.environ)
    assert limits.get_compile_worker_count() == 4
    assert not caplog.records
    assert dict(os.environ) == before
    # Generic policy ignores legacy; explicit AITER setting also supersedes it.
    assert limits.get_worker_count() == 8
    assert "requested=204" in caplog.text
    os.environ["AITER_MAX_JOBS"] = "6"
    assert limits.get_compile_worker_count() == 6


def test_host_limited_not_attributed_to_cgroup(cgroup, caplog):
    with patch.object(
        limits, "_host_available_memory_bytes", return_value=limits.EST_WORKER_RSS_BYTES
    ):
        assert limits.get_worker_count() == 1
    assert not caplog.records


def test_unreadable_usage_remains_fail_closed(cgroup, caplog):
    (cgroup / "memory.current").unlink()
    assert limits.get_worker_count() == 1
    assert "usage was unreadable" in caplog.text
    assert "current=unavailable" in caplog.text


def test_worsening_cooldown_and_recovery(cgroup, caplog):
    _, observation = limits._cgroup_memory_bound()
    with (
        patch.object(limits.time, "monotonic", return_value=0) as clock,
        patch.object(
            limits, "_automatic_worker_snapshot", return_value=(204, 8, observation)
        ) as snapshot,
    ):
        assert limits.get_worker_count() == 8
        snapshot.return_value = (204, 4, observation)
        clock.return_value = 59
        assert limits.get_worker_count() == 4
        assert len(caplog.records) == 1
        clock.return_value = 60
        assert limits.get_worker_count() == 4
        assert len(caplog.records) == 2
        clock.return_value = 120
        assert limits.get_worker_count() == 4
        assert len(caplog.records) == 2
        snapshot.return_value = (204, 1, observation)
        assert limits.get_worker_count() == 1
        assert len(caplog.records) == 3
        # Recovery clears the baseline, but preserves the cooldown.
        snapshot.return_value = (204, 300, observation)
        assert limits.get_worker_count() == 204
        snapshot.return_value = (204, 8, observation)
        assert limits.get_worker_count() == 8
        assert len(caplog.records) == 3
        clock.return_value = 180
        assert limits.get_worker_count() == 8
        assert len(caplog.records) == 4


def test_reaching_one_from_three_reports(cgroup, caplog):
    _, observation = limits._cgroup_memory_bound()
    with (
        patch.object(limits.time, "monotonic", return_value=0) as clock,
        patch.object(
            limits, "_automatic_worker_snapshot", return_value=(4, 3, observation)
        ) as snapshot,
    ):
        limits.get_worker_count()
        snapshot.return_value = (4, 1, observation)
        clock.return_value = 60
        limits.get_worker_count()
    assert len(caplog.records) == 2


def test_concurrent_calls_keep_observations_local(tmp_path, caplog):
    barrier = Barrier(8)
    for index in range(8):
        folder = tmp_path / str(index)
        folder.mkdir()
        (folder / "memory.max").write_text(str((index + 10) * 1024**3))
        (folder / "memory.current").write_text(str((index + 9) * 1024**3))
    from threading import local

    thread = local()

    def directories(controller):
        assert controller == "memory"
        return [("v2", str(tmp_path / str(thread.index)))]

    original = limits._cgroup_memory_bound

    def synchronized_bound():
        result = original()
        barrier.wait(timeout=10)
        return result

    def run(index):
        thread.index = index
        return limits.get_worker_count()

    with (
        patch.object(limits, "_cgroup_directories", side_effect=directories),
        patch.object(limits, "_cgroup_memory_bound", side_effect=synchronized_bound),
        patch.object(limits, "_host_available_memory_bytes", return_value=1024**4),
        patch.object(limits, "get_cpu_worker_budget", return_value=204),
        patch.object(
            limits,
            "_read_cgroup_memory_stat",
            side_effect=lambda observation: {
                "anon": int(str(observation["directory"]).rsplit("/", 1)[1]) * 1024**3
            },
        ),
        ThreadPoolExecutor(max_workers=8) as pool,
    ):
        assert list(pool.map(run, range(8))) == [1] * 8
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    import re

    limit = float(re.search(r"limit=([0-9.]+)", message)[1])
    anon = float(re.search(r"anon=([0-9.]+)", message)[1])
    assert limit == anon + 10


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_child_does_not_inherit_locked_diagnostic_state():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
import aiter_worker_limits as w
w._memory_diagnostic_lock.acquire()
pid = os.fork()
if pid == 0:
    acquired = w._memory_diagnostic_lock.acquire(blocking=False)
    os._exit(0 if acquired else 1)
w._memory_diagnostic_lock.release()
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0
""",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
