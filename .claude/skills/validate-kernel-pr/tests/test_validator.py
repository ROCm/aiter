import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import jsonschema
except ImportError:
    jsonschema = None

SKILL_DIR = Path(__file__).resolve().parents[1]
VALIDATOR = SKILL_DIR / "validate_pr.sh"
SCANNER = SKILL_DIR / "scan_index_width.py"
SHIPPED_PICKER = SKILL_DIR / "pick-idle-gpu.py"
REPORT_SCHEMA = json.loads((SKILL_DIR / "report_schema.json").read_text())
REQUIRED_STAGES = {
    "merge_sim",
    "gpu_claim",
    "runtime_compat",
    "test_policy",
    "baseline_control",
    "correctness_repo_tests",
    "correctness_s1_grid",
    "execution_receipt",
    "index_width_scan",
}


def validate_report_contract(report):
    required = {
        "label",
        "started_utc",
        "finished_utc",
        "isolation",
        "arch_coverage",
        "arch_coverage_basis",
        "degraded_mode",
        "repo",
        "runtime_identity",
        "test_selection",
        "stages",
        "findings",
        "verdict",
        "process_exit_code",
    }
    if missing := required - report.keys():
        raise AssertionError(f"report fields missing: {sorted(missing)}")
    if report["verdict"] not in {
        "PASS",
        "NEEDS_WORK",
        "BLOCK",
        "INCONCLUSIVE",
    }:
        raise AssertionError(f"invalid verdict: {report['verdict']}")
    if set(report["stages"]) != REQUIRED_STAGES:
        raise AssertionError(f"invalid stage set: {set(report['stages'])}")
    for name, stage in report["stages"].items():
        if not isinstance(stage, dict):
            raise TypeError(f"{name} is not an object")
        if stage.get("status") not in {"pass", "fail", "skip", "info"}:
            raise AssertionError(f"{name} has invalid status: {stage!r}")


def run(command, cwd=None, env=None, check=True):
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


def write_executable(path, source):
    path.write_text(source)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class ValidatorFixture:
    def __init__(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "aiter").mkdir()
        (self.repo / "tests").mkdir()
        (self.repo / "aiter" / "__init__.py").write_text('__version__ = "test"\n')
        (self.repo / "aiter" / "kernel.py").write_text("VALUE = 1\n")
        (self.repo / "tests" / "test_sample.py").write_text(
            "import os\n"
            '_GRID = os.environ.get("VALIDATOR_TEST_GRID", "")\n'
            'if _GRID == "__VALIDATOR_INVALID_GRID__":\n'
            '    raise ValueError("invalid validator grid probe")\n'
            '# (7, 257, "f32")\n'
            "def run_kernel(M, N, dtype_str):\n"
            "    assert M > 0 and N > 0 and dtype_str\n"
            "\n"
            "def test_sample():\n"
            "    atol = 1e-5\n"
            "    assert atol < 1\n"
            '    phase = os.environ.get("VALIDATION_PHASE", "")\n'
            "    if phase:\n"
            "        expected = f\"/{phase.split('-')[0]}/aiter-jit\"\n"
            '        assert expected in os.environ["AITER_JIT_DIR"]\n'
            '    shapes = _GRID or "7,257,f32"\n'
            "    for shape in shapes.split(';'):\n"
            "        M, N, dtype_str = shape.split(',')\n"
            "        run_kernel(int(M), int(N), dtype_str)\n"
        )
        run(["git", "init", "-q"], cwd=self.repo)
        run(["git", "add", "."], cwd=self.repo)
        run(
            [
                "git",
                "-c",
                "user.name=Validator Test",
                "-c",
                "user.email=validator@example.com",
                "commit",
                "-q",
                "-m",
                "base",
            ],
            cwd=self.repo,
        )

        self.tools = self.root / "tools"
        self.tools.mkdir()
        self.fake_modules = self.root / "fake-modules"
        self.fake_modules.mkdir()
        (self.fake_modules / "amdsmi.py").write_text(
            "class AmdSmiException(Exception): pass\n"
            "def amdsmi_init(): pass\n"
            "def amdsmi_shut_down(): pass\n"
            "def amdsmi_get_processor_handles(): return ['gpu0']\n"
            "def amdsmi_get_gpu_enumeration_info(handle): return {'hip_id': 7}\n"
            "def amdsmi_get_gpu_asic_info(handle):\n"
            "    return {'market_name': 'Synthetic GPU', "
            "'target_graphics_version': 'gfx-test'}\n"
            "def amdsmi_get_gpu_activity(handle): return {'gfx_activity': 0}\n"
            "def amdsmi_get_gpu_device_bdf(handle): return '0000:00:00.0'\n"
            "def amdsmi_get_gpu_vram_usage(handle):\n"
            "    return {'vram_used': 256, 'vram_total': 294912}\n"
        )
        self.picker = self.tools / "pick-idle-gpu.py"
        write_executable(self.picker, "#!/usr/bin/env bash\nprintf '7\\n'\n")

    def close(self):
        self.tempdir.cleanup()

    def convert_to_flydsl(self):
        shutil.rmtree(self.repo / "aiter")
        source = self.repo / "python" / "flydsl"
        source.mkdir(parents=True)
        (source / "__init__.py").write_text('__version__ = "test"\n')
        (source / "module.py").write_text("VALUE = 1\n")
        native = self.repo / "lib" / "Bindings"
        native.mkdir(parents=True)
        (native / "module.cpp").write_text("int value = 1;\n")
        mlir_python = self.repo / "python" / "mlir_flydsl"
        mlir_python.mkdir(parents=True)
        (mlir_python / "FlyRegisterEverything.cpp").write_text("int value = 1;\n")
        (self.repo / "MANIFEST.in").write_text("include README.md\n")
        run(["git", "add", "-A"], cwd=self.repo)
        run(
            [
                "git",
                "-c",
                "user.name=Validator Test",
                "-c",
                "user.email=validator@example.com",
                "commit",
                "-q",
                "-m",
                "flydsl base",
            ],
            cwd=self.repo,
        )
        runtime = self.root / "runtime" / "flydsl"
        runtime.mkdir(parents=True)
        (runtime / "__init__.py").write_text('__version__ = "test"\n')
        return source, runtime

    def make_patch(self, mutate, name="candidate.patch"):
        mutate(self.repo)
        run(["git", "add", "-A"], cwd=self.repo)
        patch = run(["git", "diff", "--cached", "--binary"], cwd=self.repo).stdout
        patch_path = self.root / name
        patch_path.write_text(patch)
        run(["git", "reset", "--hard", "-q", "HEAD"], cwd=self.repo)
        return patch_path

    def validate(
        self,
        patch,
        tests="tests/test_sample.py",
        picker=None,
        path_prefix=None,
        pylib=None,
        grid=True,
        expected_route="test_sample:run_kernel",
        grid_value="7,257,f32",
        python_bin=None,
    ):
        report = self.root / f"{patch.stem}-report.json"
        command = [
            str(VALIDATOR),
            "--repo",
            str(self.repo),
            "--patch",
            str(patch),
            "--head-sha",
            "b" * 40,
            "--target",
            tests,
            "--expected-route",
            expected_route,
            "--shape-vars",
            "M,N,dtype_str",
            "--tol-table",
            "f32=1e-5,f16=2e-3,bf16=1e-2",
            "--label",
            patch.stem,
            "--out",
            str(report),
        ]
        if grid:
            command.extend(
                [
                    "--shape-env",
                    "VALIDATOR_TEST_GRID",
                    "--grid",
                    grid_value,
                ]
            )
        environment = os.environ.copy()
        environment["PICKER"] = str(picker or self.picker)
        environment["PYTHONPATH"] = str(self.fake_modules)
        environment["TIMEOUT"] = "30"
        if python_bin:
            environment["PYTHON_BIN"] = str(python_bin)
        if pylib:
            environment["PYLIB"] = str(pylib)
        if path_prefix:
            environment["PATH"] = f"{path_prefix}:{environment['PATH']}"
        result = run(command, env=environment, check=False)
        if not report.exists():
            raise AssertionError(
                f"validator did not write a report\nstdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        data = json.loads(report.read_text())
        validate_report_contract(data)
        if jsonschema is not None:
            jsonschema.validate(data, REPORT_SCHEMA)
        return result, data


class ValidateKernelPrTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ValidatorFixture()

    def tearDown(self):
        self.fixture.close()

    @staticmethod
    def harmless_change(repo):
        (repo / "aiter" / "kernel.py").write_text("VALUE = 1\n# candidate\n")

    def assert_complete_stage_objects(self, report):
        self.assertEqual(REQUIRED_STAGES, set(report["stages"]))
        for stage in report["stages"].values():
            self.assertIsInstance(stage, dict)
            self.assertIn("status", stage)

    def test_no_gpu_is_inconclusive_and_every_skip_is_declared(self):
        patch = self.fixture.make_patch(self.harmless_change, "no-gpu.patch")
        no_gpu_picker = self.fixture.tools / "no-gpu-picker"
        write_executable(no_gpu_picker, "#!/usr/bin/env bash\nexit 1\n")

        result, report = self.fixture.validate(patch, picker=no_gpu_picker)

        self.assertEqual(2, result.returncode)
        self.assertEqual("INCONCLUSIVE", report["verdict"])
        self.assertEqual("skip", report["stages"]["gpu_claim"]["status"])
        self.assertEqual("NO_GPU", report["degraded_mode"])
        self.assertEqual({}, report["arch_coverage"])
        self.assertEqual("skip", report["stages"]["correctness_repo_tests"]["status"])
        self.assertEqual("skip", report["stages"]["correctness_s1_grid"]["status"])
        self.assert_complete_stage_objects(report)

    def test_runtime_probe_uses_aiter_checkout_and_full_run_can_pass(self):
        patch = self.fixture.make_patch(self.harmless_change, "repo-aware.patch")

        result, report = self.fixture.validate(
            patch,
            grid_value="7,257,f32;8,513,bf16",
        )

        self.assertEqual(0, result.returncode)
        self.assertEqual("PASS", report["verdict"])
        runtime = report["stages"]["runtime_compat"]
        self.assertEqual("pass", runtime["status"])
        self.assertIn("aiter", runtime["note"])
        self.assertIn(str(self.fixture.repo), runtime["note"])
        self.assertNotIn("flydsl", runtime["note"])
        self.assertEqual({"gfx-test": "runtime"}, report["arch_coverage"])
        policy = report["stages"]["test_policy"]
        self.assertEqual(1, policy["commented_out_shape_rows_base"])
        self.assertEqual(0, policy["commented_out_shape_rows_added"])
        self.assertEqual(
            "tests/test_sample.py",
            report["test_selection"]["target"],
        )
        self.assertEqual("pass", report["stages"]["execution_receipt"]["status"])
        self.assertEqual(
            "test_sample:run_kernel", report["stages"]["execution_receipt"]["route"]
        )
        self.assertEqual("aiter", report["runtime_identity"]["module"])

    def test_new_failing_test_is_not_mislabeled_preexisting(self):
        def add_failing_test(repo):
            (repo / "tests" / "test_new.py").write_text(
                "def test_new():\n" "    assert False, 'candidate failure'\n"
            )

        patch = self.fixture.make_patch(add_failing_test, "new-test.patch")
        result, report = self.fixture.validate(
            patch,
            tests="tests/test_new.py",
            grid=False,
        )

        self.assertEqual(1, result.returncode)
        self.assertEqual("BLOCK", report["verdict"])
        baseline = report["stages"]["baseline_control"]["repo_tests"]
        self.assertEqual("target-not-present", baseline["state"])
        details = [item["detail"] for item in report["findings"]]
        self.assertTrue(any("adds this test target" in detail for detail in details))
        self.assertFalse(any("pre-existing" in detail for detail in details))

    def test_script_only_target_passes_without_false_block(self):
        def add_script_target(repo):
            (repo / "tests" / "verify_kernel.py").write_text(
                "def verify_kernel():\n"
                "    return True\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    assert verify_kernel()\n"
                "    print('56/56 cases passed')\n"
            )

        patch = self.fixture.make_patch(add_script_target, "script-pass.patch")
        result, report = self.fixture.validate(
            patch,
            tests="tests/verify_kernel.py",
            grid=False,
        )

        self.assertEqual(2, result.returncode)
        self.assertEqual("INCONCLUSIVE", report["verdict"])
        self.assertEqual("script", report["test_selection"]["runner"])
        self.assertEqual("pass", report["stages"]["correctness_repo_tests"]["status"])
        self.assertFalse(
            any(item["severity"] == "blocker" for item in report["findings"])
        )
        self.assertEqual(
            "script-exit-zero-with-output",
            report["arch_coverage_basis"]["gfx-test"],
        )

    def test_script_only_target_failure_is_blocking(self):
        def add_failing_script(repo):
            (repo / "tests" / "verify_kernel.py").write_text(
                "def verify_kernel():\n"
                "    return False\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    assert verify_kernel()\n"
            )

        patch = self.fixture.make_patch(add_failing_script, "script-fail.patch")
        result, report = self.fixture.validate(
            patch,
            tests="tests/verify_kernel.py",
            grid=False,
        )

        self.assertEqual(1, result.returncode)
        self.assertEqual("BLOCK", report["verdict"])
        self.assertEqual("script", report["test_selection"]["runner"])
        self.assertEqual("fail", report["stages"]["correctness_repo_tests"]["status"])

    def test_target_without_entry_point_is_skipped(self):
        def add_library_only_target(repo):
            (repo / "tests" / "kernel_helpers.py").write_text(
                "def verify_kernel():\n" "    return True\n"
            )

        patch = self.fixture.make_patch(
            add_library_only_target,
            "no-runner.patch",
        )
        result, report = self.fixture.validate(
            patch,
            tests="tests/kernel_helpers.py",
            grid=False,
        )

        self.assertEqual(2, result.returncode)
        self.assertEqual("INCONCLUSIVE", report["verdict"])
        self.assertEqual("none", report["test_selection"]["runner"])
        self.assertEqual("skip", report["stages"]["correctness_repo_tests"]["status"])

    def test_test_only_tolerance_widening_blocks_without_gpu(self):
        def loosen_tolerance(repo):
            path = repo / "tests" / "test_sample.py"
            path.write_text(path.read_text().replace("1e-5", "1e-1"))

        patch = self.fixture.make_patch(loosen_tolerance, "tolerance.patch")
        no_gpu_picker = self.fixture.tools / "no-gpu-picker"
        write_executable(no_gpu_picker, "#!/usr/bin/env bash\nexit 1\n")

        result, report = self.fixture.validate(patch, picker=no_gpu_picker)

        self.assertEqual(1, result.returncode)
        self.assertEqual("BLOCK", report["verdict"])
        self.assertEqual("fail", report["stages"]["test_policy"]["status"])
        self.assertEqual([[1e-5, 1e-1]], report["stages"]["test_policy"]["loosened"])

    def test_unavailable_pytest_writes_stage_objects_not_strings(self):
        patch = self.fixture.make_patch(self.harmless_change, "no-pytest.patch")
        fake_bin = self.fixture.root / "fake-bin"
        fake_bin.mkdir()
        write_executable(fake_bin / "python", "#!/usr/bin/env bash\nexit 1\n")

        _, report = self.fixture.validate(
            patch,
            path_prefix=fake_bin,
            python_bin=fake_bin / "python",
        )

        self.assertEqual("INCONCLUSIVE", report["verdict"])
        self.assertEqual("skip", report["stages"]["correctness_repo_tests"]["status"])
        self.assertEqual("skip", report["stages"]["correctness_s1_grid"]["status"])
        self.assertEqual({}, report["arch_coverage"])
        self.assert_complete_stage_objects(report)

    def test_all_skipped_pytest_is_inconclusive(self):
        def skip_test(repo):
            path = repo / "tests" / "test_sample.py"
            path.write_text(
                "import pytest\n"
                "pytestmark = pytest.mark.skip(reason='not applicable')\n"
                + path.read_text()
            )

        patch = self.fixture.make_patch(skip_test, "all-skipped.patch")
        result, report = self.fixture.validate(patch)

        self.assertEqual(2, result.returncode)
        self.assertEqual("INCONCLUSIVE", report["verdict"])
        stage = report["stages"]["correctness_repo_tests"]
        self.assertEqual("skip", stage["status"])
        self.assertEqual(0, stage["stats"]["executed"])
        self.assertEqual(1, stage["stats"]["skipped"])
        self.assertEqual({}, report["arch_coverage"])

    def test_missing_execution_receipt_prevents_pass(self):
        def remove_route_call(repo):
            path = repo / "tests" / "test_sample.py"
            path.write_text(
                path.read_text().replace(
                    "        run_kernel(int(M), int(N), dtype_str)\n",
                    "        assert int(M) > 0 and int(N) > 0 and dtype_str\n",
                )
            )

        patch = self.fixture.make_patch(remove_route_call, "missing-receipt.patch")
        result, report = self.fixture.validate(patch)

        self.assertEqual(2, result.returncode)
        self.assertEqual("INCONCLUSIVE", report["verdict"])
        self.assertEqual("skip", report["stages"]["execution_receipt"]["status"])

    def test_worktree_cannot_shadow_validator_probe(self):
        def add_fake_probe_and_remove_route(repo):
            (repo / "validation_probe.py").write_text(
                "def pytest_configure(config): pass\n"
                "def pytest_sessionfinish(session, exitstatus): pass\n"
            )
            (repo / "conftest.py").write_text(
                "import json\n"
                "import os\n"
                "import pytest\n"
                "@pytest.hookimpl(trylast=True)\n"
                "def pytest_sessionfinish(session, exitstatus):\n"
                '    path = os.environ.get("VALIDATION_EVIDENCE_PATH")\n'
                "    if path:\n"
                "        open(path, 'w').write(json.dumps({\n"
                "            'schema_version': 1,\n"
                "            'producer': 'validate-kernel-pr.validation_probe',\n"
                "            'route': 'test_sample:run_kernel',\n"
                "            'kernel_symbols': ['test_sample:run_kernel'],\n"
                "            'executed_shapes': ['7,257,f32'],\n"
                "        }))\n"
            )
            path = repo / "tests" / "test_sample.py"
            path.write_text(
                path.read_text().replace(
                    "        run_kernel(int(M), int(N), dtype_str)\n",
                    "        assert int(M) > 0 and int(N) > 0 and dtype_str\n",
                )
            )

        patch = self.fixture.make_patch(
            add_fake_probe_and_remove_route,
            "shadow-probe.patch",
        )
        result, report = self.fixture.validate(patch)

        self.assertEqual(2, result.returncode)
        self.assertEqual("INCONCLUSIVE", report["verdict"])
        self.assertEqual("skip", report["stages"]["execution_receipt"]["status"])

    def test_incomplete_shape_receipt_prevents_pass(self):
        def omit_shape(repo):
            path = repo / "tests" / "test_sample.py"
            path.write_text(
                path.read_text().replace(
                    "    for shape in shapes.split(';'):\n",
                    "    for shape in shapes.split(';')[:1]:\n",
                )
            )

        patch = self.fixture.make_patch(omit_shape, "missing-shape.patch")
        result, report = self.fixture.validate(
            patch,
            grid_value="7,257,f32;8,513,bf16",
        )

        self.assertEqual(2, result.returncode)
        self.assertEqual("INCONCLUSIVE", report["verdict"])
        receipt = report["stages"]["execution_receipt"]
        self.assertEqual("skip", receipt["status"])
        self.assertIn("missing required shapes", receipt["note"])

    def test_wrong_route_receipt_prevents_pass(self):
        patch = self.fixture.make_patch(self.harmless_change, "wrong-route.patch")
        result, report = self.fixture.validate(
            patch,
            expected_route="test_sample:different_route",
        )

        self.assertEqual(2, result.returncode)
        self.assertEqual("INCONCLUSIVE", report["verdict"])
        receipt = report["stages"]["execution_receipt"]
        self.assertEqual("skip", receipt["status"])
        self.assertIn("expected route", receipt["note"])

    def test_flydsl_source_change_is_not_shadowed_by_pylib(self):
        _, runtime = self.fixture.convert_to_flydsl()

        def change_flydsl(repo):
            root = repo / "python" / "flydsl"
            (root / "module.py").rename(root / "renamed.py")

        patch = self.fixture.make_patch(change_flydsl, "flydsl-rename.patch")
        _, report = self.fixture.validate(
            patch,
            pylib=runtime.parent,
        )

        self.assertEqual("INCONCLUSIVE", report["verdict"])
        self.assertEqual("skip", report["stages"]["runtime_compat"]["status"])
        self.assertIn(
            "trusted build provenance", report["stages"]["runtime_compat"]["note"]
        )

    def test_flydsl_native_change_is_inconclusive_without_provenance(self):
        _, runtime = self.fixture.convert_to_flydsl()

        def change_native_source(repo):
            path = repo / "python" / "mlir_flydsl" / "FlyRegisterEverything.cpp"
            path.write_text("int value = 2;\n")

        patch = self.fixture.make_patch(change_native_source, "flydsl-native.patch")
        result, report = self.fixture.validate(
            patch,
            pylib=runtime.parent,
        )

        self.assertEqual(2, result.returncode)
        self.assertEqual("INCONCLUSIVE", report["verdict"])
        self.assertEqual("skip", report["stages"]["runtime_compat"]["status"])
        self.assertIn(
            "trusted build provenance", report["stages"]["runtime_compat"]["note"]
        )

    def test_flydsl_packaging_change_is_inconclusive_without_provenance(self):
        _, runtime = self.fixture.convert_to_flydsl()

        def change_manifest(repo):
            (repo / "MANIFEST.in").write_text("recursive-include python *.cpp\n")

        patch = self.fixture.make_patch(change_manifest, "flydsl-manifest.patch")
        result, report = self.fixture.validate(
            patch,
            pylib=runtime.parent,
        )

        self.assertEqual(2, result.returncode)
        self.assertEqual("INCONCLUSIVE", report["verdict"])
        self.assertIn(
            "trusted build provenance", report["stages"]["runtime_compat"]["note"]
        )

    def test_grid_pass_cannot_ignore_shape_environment(self):
        def remove_grid_hook(repo):
            path = repo / "tests" / "test_sample.py"
            path.write_text(
                path.read_text().replace("VALIDATOR_TEST_GRID", "UNRELATED_ENV")
                + '\nUNUSED_GRID_NAME = "VALIDATOR_TEST_GRID"\n'
                + "\n# VALIDATOR_TEST_GRID is intentionally not consumed.\n"
            )

        patch = self.fixture.make_patch(remove_grid_hook, "ignored-grid.patch")
        _, report = self.fixture.validate(patch)

        self.assertEqual("INCONCLUSIVE", report["verdict"])
        self.assertEqual("skip", report["stages"]["correctness_s1_grid"]["status"])
        self.assertIn(
            "not referenced",
            report["stages"]["correctness_s1_grid"]["note"],
        )

    def test_grid_pass_requires_runtime_shape_handshake(self):
        def ignore_grid_value(repo):
            path = repo / "tests" / "test_sample.py"
            source = path.read_text().replace(
                'if _GRID == "__VALIDATOR_INVALID_GRID__":',
                "if False and _GRID:",
            )
            path.write_text(
                source.replace(
                    '    shapes = _GRID or "7,257,f32"',
                    '    _ = _GRID\n    shapes = "7,257,f32"',
                )
            )

        patch = self.fixture.make_patch(ignore_grid_value, "unused-grid.patch")
        _, report = self.fixture.validate(patch)

        self.assertEqual("INCONCLUSIVE", report["verdict"])
        self.assertEqual("skip", report["stages"]["correctness_s1_grid"]["status"])
        self.assertIn(
            "ignores",
            report["stages"]["correctness_s1_grid"]["note"],
        )

    def test_base_artifact_prevents_contaminated_head_run(self):
        (self.fixture.repo / ".gitignore").write_text("baseline-artifact\n")
        test_file = self.fixture.repo / "tests" / "test_sample.py"
        test_file.write_text(
            test_file.read_text()
            + "\nfrom pathlib import Path\n"
            + "Path('baseline-artifact').write_text('created')\n"
        )
        run(["git", "add", "-A"], cwd=self.fixture.repo)
        run(
            [
                "git",
                "-c",
                "user.name=Validator Test",
                "-c",
                "user.email=validator@example.com",
                "commit",
                "-q",
                "-m",
                "artifact base",
            ],
            cwd=self.fixture.repo,
        )
        patch = self.fixture.make_patch(self.harmless_change, "base-artifact.patch")

        _, report = self.fixture.validate(patch)

        self.assertEqual("INCONCLUSIVE", report["verdict"])
        self.assertEqual("skip", report["stages"]["baseline_control"]["status"])
        self.assertEqual("skip", report["stages"]["correctness_repo_tests"]["status"])

    def test_existing_ignored_artifact_rejects_nonisolated_worktree(self):
        (self.fixture.repo / ".gitignore").write_text("ignored-cache/\n")
        run(["git", "add", ".gitignore"], cwd=self.fixture.repo)
        run(
            [
                "git",
                "-c",
                "user.name=Validator Test",
                "-c",
                "user.email=validator@example.com",
                "commit",
                "-q",
                "-m",
                "ignore cache",
            ],
            cwd=self.fixture.repo,
        )
        ignored = self.fixture.repo / "ignored-cache"
        ignored.mkdir()
        (ignored / "state").write_text("pre-existing")
        patch = self.fixture.make_patch(self.harmless_change, "ignored-artifact.patch")

        result, report = self.fixture.validate(patch)

        self.assertEqual(2, result.returncode)
        self.assertEqual("INCONCLUSIVE", report["verdict"])
        self.assertEqual("skip", report["stages"]["merge_sim"]["status"])


class IndexScannerTests(unittest.TestCase):
    def test_json_count_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            diff = Path(directory) / "candidate.diff"
            diff.write_text(
                "+++ b/kernel.py\n"
                "+out = block_id * row_stride\n"
                "+out = block_id * row_stride\n"
                "+safe = block_id.to(tl.int64) * row_stride\n"
            )
            result = run([str(SCANNER), "--diff", str(diff), "--json"])
            payload = json.loads(result.stdout)

        self.assertEqual(1, payload["index_stride_candidates"])
        self.assertEqual(0, payload["untyped_stride_parameters"])
        self.assertEqual(1, payload["total_candidates"])


class GpuPickerTests(unittest.TestCase):
    def test_shipped_picker_returns_translated_hip_index(self):
        fixture = ValidatorFixture()
        try:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(fixture.fake_modules)
            result = run(
                [
                    sys.executable,
                    str(SHIPPED_PICKER),
                    "--samples",
                    "1",
                    "--interval",
                    "0",
                    "--quiet",
                ],
                env=environment,
            )
        finally:
            fixture.close()

        self.assertEqual("7", result.stdout.strip())


class ReviewSkillContractTests(unittest.TestCase):
    def test_review_skill_is_advisory_and_has_no_dead_scanner_paths(self):
        review_skill = (SKILL_DIR.parent / "review-pr" / "SKILL.md").read_text()

        self.assertTrue((SKILL_DIR / "validate_evidence.py").is_file())
        self.assertTrue((SKILL_DIR / "validation_probe.py").is_file())
        self.assertTrue(SHIPPED_PICKER.is_file())
        self.assertIn("advisory tier", review_skill)
        self.assertIn("required scanner is missing or not executable", review_skill)
        self.assertIn("Validation (deterministic)", review_skill)
        self.assertIn("baseRefName", review_skill)
        self.assertIn("base_head.txt", review_skill)
        self.assertIn("expected_verdict", review_skill)
        self.assertIn("if stats is not None", review_skill)
        self.assertNotIn("downstream-impact-check", review_skill)
        self.assertNotIn("review-flydsl-kernel/scan_", review_skill)

    def test_review_fetch_snippet_parses_as_bash(self):
        review_skill = (SKILL_DIR.parent / "review-pr" / "SKILL.md").read_text()
        match = re.search(
            r"## Step 1 — Fetch.*?```bash\n(.*?)\n```",
            review_skill,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        result = subprocess.run(
            ["bash", "-n"],
            input=match.group(1),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
