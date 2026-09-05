import ast
import inspect
import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import aiter_worker_limits as worker_limits

configure_worker_subprocesses = worker_limits.configure_worker_subprocesses
adopt_legacy_max_jobs = worker_limits.adopt_legacy_max_jobs
get_automatic_worker_budgets = worker_limits.get_automatic_worker_budgets
get_compile_worker_count = worker_limits.get_compile_worker_count
get_cpu_worker_budget = worker_limits.get_cpu_worker_budget
get_worker_count = worker_limits.get_worker_count
get_worker_count_for = worker_limits.get_worker_count_for


class WorkerAwarenessTest(unittest.TestCase):
    def test_worker_count_accepts_no_per_caller_default(self):
        self.assertEqual(tuple(inspect.signature(get_worker_count).parameters), ())

    def test_cpu_budget_uses_at_most_eighty_percent(self):
        self.assertEqual(get_cpu_worker_budget(cpu_count=24), 19)
        self.assertEqual(get_cpu_worker_budget(cpu_count=4), 3)

    def test_cpu_budget_uses_process_available_cpus(self):
        with patch.object(worker_limits, "_process_cpu_count", return_value=24):
            self.assertEqual(get_cpu_worker_budget(), 19)

    def test_process_cpu_count_falls_back_to_affinity(self):
        with patch.object(
            os, "process_cpu_count", return_value=None, create=True
        ), patch.object(os, "sched_getaffinity", return_value={0, 1, 2, 3}):
            self.assertEqual(worker_limits._process_cpu_count(), 4)

    def test_cpu_budget_always_returns_at_least_one(self):
        for logical_cpus in (0, 1):
            with self.subTest(logical_cpus=logical_cpus):
                self.assertEqual(get_cpu_worker_budget(cpu_count=logical_cpus), 1)
        with patch.object(worker_limits, "_process_cpu_count", return_value=1):
            self.assertEqual(get_cpu_worker_budget(), 1)

    def test_automatic_budgets_contain_only_cpu_and_memory(self):
        with patch.object(
            worker_limits, "get_cpu_worker_budget", return_value=6
        ), patch.object(
            worker_limits,
            "_available_memory_bounds",
            return_value=(3 * worker_limits.EST_WORKER_RSS_BYTES, None),
        ):
            self.assertEqual(get_automatic_worker_budgets(), (6, 3))

    def test_v2_cgroup_membership_maps_to_current_and_parent_directories(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = pathlib.Path(tempdir)
            mount_point = root / "cgroup"
            mount_point.mkdir()
            cgroup_file = root / "cgroup.txt"
            mountinfo_file = root / "mountinfo.txt"
            cgroup_file.write_text("0::/jobs/worker\n")
            mountinfo_file.write_text(
                f"36 25 0:32 / {mount_point} rw - cgroup2 none rw\n"
            )
            with patch.object(
                worker_limits, "_PROC_SELF_CGROUP_PATH", str(cgroup_file)
            ), patch.object(
                worker_limits, "_PROC_SELF_MOUNTINFO_PATH", str(mountinfo_file)
            ):
                directories = worker_limits._cgroup_memory_directories()

        self.assertEqual(
            directories,
            [
                ("v2", str(mount_point / "jobs/worker")),
                ("v2", str(mount_point / "jobs")),
                ("v2", str(mount_point)),
            ],
        )

    def test_v1_memory_controller_membership_respects_mount_root(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = pathlib.Path(tempdir)
            mount_point = root / "memory"
            mount_point.mkdir()
            cgroup_file = root / "cgroup.txt"
            mountinfo_file = root / "mountinfo.txt"
            cgroup_file.write_text("5:cpu,memory:/docker/worker\n")
            mountinfo_file.write_text(
                f"29 23 0:26 /docker {mount_point} rw - cgroup cgroup rw,memory\n"
            )
            with patch.object(
                worker_limits, "_PROC_SELF_CGROUP_PATH", str(cgroup_file)
            ), patch.object(
                worker_limits, "_PROC_SELF_MOUNTINFO_PATH", str(mountinfo_file)
            ):
                directories = worker_limits._cgroup_memory_directories()

        self.assertEqual(
            directories,
            [
                ("v1", str(mount_point / "worker")),
                ("v1", str(mount_point)),
            ],
        )

    def test_hybrid_cgroup_mounts_all_contribute_memory_directories(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = pathlib.Path(tempdir)
            v2_mount = root / "unified"
            v1_mount = root / "memory"
            v2_mount.mkdir()
            v1_mount.mkdir()
            v2_current = v2_mount / "container"
            v1_current = v1_mount / "container"
            v2_current.mkdir()
            v1_current.mkdir()
            (v2_current / "memory.max").write_text(str(16 * 1024**3))
            (v2_current / "memory.current").write_text(str(1 * 1024**3))
            (v1_current / "memory.limit_in_bytes").write_text(str(8 * 1024**3))
            (v1_current / "memory.usage_in_bytes").write_text(str(7 * 1024**3))
            cgroup_file = root / "cgroup.txt"
            mountinfo_file = root / "mountinfo.txt"
            cgroup_file.write_text("0::/container\n5:memory:/container\n")
            mountinfo_file.write_text(
                f"36 25 0:32 / {v2_mount} rw - cgroup2 none rw\n"
                f"29 23 0:26 / {v1_mount} rw - cgroup cgroup rw,memory\n"
            )
            with patch.object(
                worker_limits, "_PROC_SELF_CGROUP_PATH", str(cgroup_file)
            ), patch.object(
                worker_limits, "_PROC_SELF_MOUNTINFO_PATH", str(mountinfo_file)
            ):
                directories = worker_limits._cgroup_memory_directories()
                remaining = worker_limits._cgroup_memory_remaining_bytes()

        self.assertEqual(
            directories,
            [
                ("v2", str(v2_mount / "container")),
                ("v2", str(v2_mount)),
                ("v1", str(v1_mount / "container")),
                ("v1", str(v1_mount)),
            ],
        )
        self.assertEqual(remaining, 1 * 1024**3)

    def test_cgroup_remaining_memory_uses_tightest_finite_ancestor(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = pathlib.Path(tempdir)
            child = root / "child"
            child.mkdir()
            (child / "memory.max").write_text("max\n")
            (child / "memory.current").write_text("1073741824\n")
            (root / "memory.max").write_text(str(8 * 1024**3))
            (root / "memory.current").write_text(str(2 * 1024**3))
            with patch.object(
                worker_limits,
                "_cgroup_memory_directories",
                return_value=[("v2", str(child)), ("v2", str(root))],
            ):
                remaining = worker_limits._cgroup_memory_remaining_bytes()

        self.assertEqual(remaining, 6 * 1024**3)

    def test_v1_cgroup_remaining_memory_uses_limit_and_usage_files(self):
        with tempfile.TemporaryDirectory() as tempdir:
            directory = pathlib.Path(tempdir)
            (directory / "memory.limit_in_bytes").write_text(str(8 * 1024**3))
            (directory / "memory.usage_in_bytes").write_text(str(3 * 1024**3))
            with patch.object(
                worker_limits,
                "_cgroup_memory_directories",
                return_value=[("v1", str(directory))],
            ):
                remaining = worker_limits._cgroup_memory_remaining_bytes()

        self.assertEqual(remaining, 5 * 1024**3)

    def test_finite_cgroup_with_unreadable_usage_fails_closed(self):
        with tempfile.TemporaryDirectory() as tempdir:
            directory = pathlib.Path(tempdir)
            (directory / "memory.max").write_text(str(8 * 1024**3))
            with patch.object(
                worker_limits,
                "_cgroup_memory_directories",
                return_value=[("v2", str(directory))],
            ):
                remaining = worker_limits._cgroup_memory_remaining_bytes()

        self.assertEqual(remaining, 0)

    def test_cgroup_memory_diagnostic_reports_page_cache_once(self):
        with tempfile.TemporaryDirectory() as tempdir:
            directory = pathlib.Path(tempdir)
            (directory / "memory.max").write_text(str(4 * 1024**3))
            (directory / "memory.current").write_text(str(3 * 1024**3))
            (directory / "memory.stat").write_text(
                "anon 1073741824\n"
                "file 2147483648\n"
                "active_file 536870912\n"
                "inactive_file 1610612736\n"
                "slab_reclaimable 268435456\n"
            )
            previous_diagnostic_state = (
                worker_limits._cgroup_memory_diagnostic_emitted
            )
            worker_limits._cgroup_memory_diagnostic_emitted = False
            try:
                with patch.object(
                    worker_limits,
                    "_cgroup_memory_directories",
                    return_value=[("v2", str(directory))],
                ), patch.object(
                    worker_limits,
                    "_host_available_memory_bytes",
                    return_value=256 * 1024**3,
                ), patch.object(
                    worker_limits, "_process_cpu_count", return_value=64
                ), self.assertLogs(
                    worker_limits._logger, level="WARNING"
                ) as logs:
                    self.assertEqual(
                        worker_limits.get_automatic_worker_budgets(), (51, 1)
                    )
                    self.assertEqual(
                        worker_limits.get_automatic_worker_budgets(), (51, 1)
                    )
                self.assertEqual(len(logs.output), 1)
                self.assertIn("page cache", logs.output[0])
                self.assertIn("inactive_file", logs.output[0])
            finally:
                worker_limits._cgroup_memory_diagnostic_emitted = (
                    previous_diagnostic_state
                )

    def test_cgroup_diagnostic_skips_host_limited_budget(self):
        with tempfile.TemporaryDirectory() as tempdir:
            directory = pathlib.Path(tempdir)
            (directory / "memory.max").write_text(str(32 * 1024**3))
            (directory / "memory.current").write_text(str(1 * 1024**3))
            previous_diagnostic_state = (
                worker_limits._cgroup_memory_diagnostic_emitted
            )
            worker_limits._cgroup_memory_diagnostic_emitted = False
            try:
                with patch.object(
                    worker_limits,
                    "_available_memory_bounds",
                    return_value=(1 * 1024**3, 31 * 1024**3),
                ), patch.object(
                    worker_limits, "_process_cpu_count", return_value=64
                ), patch.object(
                    worker_limits._logger, "warning"
                ) as warning, patch.object(worker_limits._logger, "info") as info:
                    self.assertEqual(
                        worker_limits.get_automatic_worker_budgets(), (51, 1)
                    )
                warning.assert_not_called()
                info.assert_not_called()
            finally:
                worker_limits._cgroup_memory_diagnostic_emitted = (
                    previous_diagnostic_state
                )

    def test_container_memory_caps_large_host_worker_budget(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            worker_limits,
            "_host_available_memory_bytes",
            return_value=256 * 1024**3,
        ), patch.object(
            worker_limits,
            "_cgroup_memory_remaining_bytes",
            return_value=8 * 1024**3,
        ), patch.object(
            worker_limits, "_process_cpu_count", return_value=128
        ):
            self.assertEqual(get_automatic_worker_budgets(), (102, 5))
            self.assertEqual(get_worker_count(), 5)

    def test_four_cpu_worker_count_uses_eighty_percent(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            worker_limits,
            "_available_memory_bounds",
            return_value=(10 * 1024**3, None),
        ), patch.object(worker_limits, "_process_cpu_count", return_value=4):
            self.assertEqual(get_worker_count(), 3)
            self.assertNotIn("AITER_MAX_JOBS", os.environ)

    def test_explicit_aiter_max_jobs_is_clamped_to_automatic_caps(self):
        with patch.dict(os.environ, {"AITER_MAX_JOBS": "99"}, clear=True), patch.object(
            worker_limits, "get_automatic_worker_budgets", return_value=(4, 3)
        ) as automatic_budgets:
            self.assertEqual(get_worker_count(), 3)
            automatic_budgets.assert_called_once_with()

    def test_automatic_worker_budget_is_recomputed_on_every_call(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            worker_limits,
            "get_automatic_worker_budgets",
            side_effect=((102, 180), (1, 1)),
        ) as automatic_budgets:
            self.assertEqual(get_worker_count(), 102)
            self.assertEqual(get_worker_count(), 1)
            self.assertEqual(automatic_budgets.call_count, 2)
            self.assertNotIn("AITER_MAX_JOBS", os.environ)

    def test_framework_max_jobs_is_ignored(self):
        with patch.dict(os.environ, {"MAX_JOBS": "99"}, clear=True), patch.object(
            worker_limits,
            "_available_memory_bounds",
            return_value=(10 * 1024**3, None),
        ), patch.object(worker_limits, "_process_cpu_count", return_value=4):
            self.assertEqual(get_worker_count(), 3)
            self.assertEqual(os.environ["MAX_JOBS"], "99")
            self.assertNotIn("AITER_MAX_JOBS", os.environ)

    def test_runtime_compile_honors_legacy_max_jobs_without_mutating_environment(
        self,
    ):
        with patch.dict(os.environ, {"MAX_JOBS": "2"}, clear=True), patch.object(
            worker_limits,
            "get_automatic_worker_budgets",
            return_value=(8, 8),
        ):
            self.assertEqual(get_compile_worker_count(), 2)
            self.assertEqual(os.environ["MAX_JOBS"], "2")
            self.assertNotIn("AITER_MAX_JOBS", os.environ)

    def test_runtime_compile_prefers_aiter_max_jobs_over_legacy(self):
        with patch.dict(
            os.environ,
            {"AITER_MAX_JOBS": "5", "MAX_JOBS": "2"},
            clear=True,
        ), patch.object(
            worker_limits, "get_automatic_worker_budgets", return_value=(8, 8)
        ):
            self.assertEqual(get_compile_worker_count(), 5)

    def test_runtime_compile_invalid_legacy_max_jobs_falls_back_to_automatic(
        self,
    ):
        for raw_value in ("not-an-integer", "0", "-7"):
            with self.subTest(raw_value=raw_value), patch.dict(
                os.environ, {"MAX_JOBS": raw_value}, clear=True
            ), patch.object(
                worker_limits,
                "get_automatic_worker_budgets",
                return_value=(8, 3),
            ):
                self.assertEqual(get_compile_worker_count(), 3)
                self.assertEqual(os.environ["MAX_JOBS"], raw_value)
                self.assertNotIn("AITER_MAX_JOBS", os.environ)

    def test_runtime_jit_uses_legacy_aware_compile_helper(self):
        tree = ast.parse(
            (_REPO_ROOT / "aiter/jit/utils/cpp_extension.py").read_text()
        )
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "aiter_worker_limits"
            for alias in node.names
        }
        self.assertIn("get_compile_worker_count", imported_names)
        calls = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "get_compile_worker_count"
        ]
        self.assertEqual(len(calls), 1)

    def test_standalone_entrypoint_adopts_valid_legacy_max_jobs(self):
        with patch.dict(os.environ, {"MAX_JOBS": "7"}, clear=True), self.assertWarns(
            FutureWarning
        ) as warning:
            adopt_legacy_max_jobs()
            self.assertEqual(os.environ["MAX_JOBS"], "7")
            self.assertEqual(os.environ["AITER_MAX_JOBS"], "7")
            self.assertIn(
                "use AITER_MAX_JOBS instead", str(warning.warnings[0].message)
            )

    def test_explicit_aiter_max_jobs_prevents_legacy_adoption(self):
        with patch.dict(
            os.environ,
            {"AITER_MAX_JOBS": "3", "MAX_JOBS": "7"},
            clear=True,
        ), patch.object(worker_limits.warnings, "warn") as warn:
            adopt_legacy_max_jobs()

            self.assertEqual(os.environ["AITER_MAX_JOBS"], "3")
            self.assertEqual(os.environ["MAX_JOBS"], "7")
            warn.assert_not_called()

    def test_invalid_legacy_max_jobs_is_not_adopted(self):
        for raw_value in ("", "auto", "0", "-7"):
            with self.subTest(raw_value=raw_value), patch.dict(
                os.environ, {"MAX_JOBS": raw_value}, clear=True
            ), patch.object(worker_limits.warnings, "warn") as warn:
                adopt_legacy_max_jobs()

                self.assertEqual(os.environ["MAX_JOBS"], raw_value)
                self.assertNotIn("AITER_MAX_JOBS", os.environ)
                warn.assert_not_called()

    def test_adopted_legacy_ceiling_remains_clamped_to_live_limits(self):
        with patch.dict(os.environ, {"MAX_JOBS": "99"}, clear=True), patch.object(
            worker_limits, "get_automatic_worker_budgets", return_value=(4, 3)
        ):
            with self.assertWarns(FutureWarning):
                adopt_legacy_max_jobs()
            self.assertEqual(get_worker_count(), 3)

    def test_legacy_adoption_is_wired_only_at_owned_entrypoints(self):
        guarded_entrypoints = (
            "aiter/aot/asm_mla_decode_fwd.py",
            "aiter/aot/pa.py",
            "aiter/aot/pa_ragged.py",
            "aiter/aot/pa_v1.py",
            "aiter/aot/sampling.py",
            "aiter/aot/flydsl/chunk_gdn_h.py",
            "aiter/aot/flydsl/gemm.py",
            "aiter/aot/flydsl/grouped_moe.py",
            "aiter/aot/flydsl/moe.py",
            "csrc/cpp_itfs/pa_gluon_aot/pa_decode_gluon_aot_prebuild.py",
            "csrc/opus_gemm/gen_co/build_co.py",
            "op_tests/opus/device/setup.py",
        )
        for relative_path in guarded_entrypoints:
            with self.subTest(relative_path=relative_path):
                tree = ast.parse((_REPO_ROOT / relative_path).read_text())
                main_guards = [
                    node
                    for node in tree.body
                    if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
                ]
                self.assertEqual(len(main_guards), 1)
                calls = [
                    node.func.id
                    for node in ast.walk(main_guards[0])
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                ]
                self.assertIn("adopt_legacy_max_jobs", calls)

        setup_tree = ast.parse((_REPO_ROOT / "setup.py").read_text())
        top_level_calls = [
            node.value.func.id
            for node in setup_tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ]
        self.assertIn("adopt_legacy_max_jobs", top_level_calls)

    def test_explicit_lower_aiter_max_jobs_is_honored(self):
        with patch.dict(os.environ, {"AITER_MAX_JOBS": "1"}, clear=True):
            self.assertEqual(get_worker_count(), 1)

    def test_nonpositive_aiter_max_jobs_is_clamped_without_mutating_environment(self):
        for raw_value in ("0", "-7"):
            with self.subTest(raw_value=raw_value), patch.dict(
                os.environ, {"AITER_MAX_JOBS": raw_value}, clear=True
            ):
                self.assertEqual(get_worker_count(), 1)
                self.assertEqual(os.environ["AITER_MAX_JOBS"], raw_value)

    def test_invalid_aiter_max_jobs_falls_back_to_automatic_sizing(self):
        for raw_value in ("", "auto", "not-an-integer"):
            with self.subTest(raw_value=raw_value), patch.dict(
                os.environ, {"AITER_MAX_JOBS": raw_value}, clear=True
            ), patch.object(
                worker_limits, "get_automatic_worker_budgets", return_value=(6, 4)
            ):
                self.assertEqual(get_worker_count(), 4)
                self.assertEqual(os.environ["AITER_MAX_JOBS"], raw_value)

    def test_zero_memory_capacity_still_returns_one_worker(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            worker_limits,
            "_available_memory_bounds",
            return_value=(0, None),
        ), patch.object(worker_limits, "_process_cpu_count", return_value=1):
            self.assertEqual(get_worker_count(), 1)
            self.assertNotIn("AITER_MAX_JOBS", os.environ)

    def test_available_memory_caps_default_workers(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            worker_limits,
            "_available_memory_bounds",
            return_value=(4 * worker_limits.EST_WORKER_RSS_BYTES, None),
        ), patch.object(worker_limits, "_process_cpu_count", return_value=64):
            self.assertEqual(get_worker_count(), 4)
            self.assertNotIn("AITER_MAX_JOBS", os.environ)

    def test_worker_descendants_are_forced_to_one_job(self):
        with patch.dict(
            os.environ,
            {
                "AITER_MAX_JOBS": "23",
                "MAX_JOBS": "64",
            },
            clear=True,
        ):
            configure_worker_subprocesses()
            self.assertEqual(os.environ["AITER_MAX_JOBS"], "1")
            self.assertEqual(os.environ["MAX_JOBS"], "64")
            self.assertEqual(os.environ["CMAKE_BUILD_PARALLEL_LEVEL"], "1")
            self.assertEqual(os.environ["MAKEFLAGS"], "-j1")
            self.assertEqual(os.environ["NINJAFLAGS"], "-j1")

    def test_work_capped_worker_count_never_returns_zero(self):
        with patch.dict(os.environ, {"AITER_MAX_JOBS": "19"}, clear=True), patch.object(
            worker_limits, "get_automatic_worker_budgets", return_value=(32, 32)
        ):
            self.assertEqual(get_worker_count_for(0), 1)
            self.assertEqual(get_worker_count_for(3), 3)

    def test_one_job_reaches_all_descendant_controls(self):
        with patch.dict(os.environ, {}, clear=True):
            configure_worker_subprocesses()
            self.assertEqual(os.environ["AITER_MAX_JOBS"], "1")
            self.assertEqual(os.environ["CMAKE_BUILD_PARALLEL_LEVEL"], "1")
            self.assertEqual(os.environ["MAKEFLAGS"], "-j1")
            self.assertEqual(os.environ["NINJAFLAGS"], "-j1")
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
