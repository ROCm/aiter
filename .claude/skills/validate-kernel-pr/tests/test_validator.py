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
# Stages that may be absent without making a report incomplete. `perf` runs only when the
# target exposes a benchmark harness and both phases completed, so its presence is a
# property of the PR under test rather than of the validator. Asserting an exact stage set
# would turn every optional stage into a failure in all 25 tests; asserting a subset plus
# this allowlist keeps the real intent -- every required stage present and well-formed, and
# nothing unrecognised sitting alongside them.
OPTIONAL_STAGES = {"perf", "claims"}


def assert_stage_set(stages):
    if missing := REQUIRED_STAGES - set(stages):
        raise AssertionError(f"required stages missing: {sorted(missing)}")
    if unknown := set(stages) - REQUIRED_STAGES - OPTIONAL_STAGES:
        raise AssertionError(f"unrecognised stages present: {sorted(unknown)}")


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
    assert_stage_set(report["stages"])
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

    BENCH_TARGET = "tests/test_bench.py"

    def add_bench_target(self, scale="1.0"):
        """Commit a timeable target, so a later patch can change what it costs.

        The perf stage compares base against head, so the target has to exist on BOTH
        sides. A target the patch *adds* is a different case with its own test below.
        """
        (self.repo / self.BENCH_TARGET).write_text(
            "import argparse\n"
            "\n"
            f"SCALE = {scale}\n"
            "\n"
            "\n"
            "def run_kernel(dim):\n"
            "    return dim * SCALE / 100.0\n"
            "\n"
            "\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--scenario', default='test',\n"
            "                        choices=['test', 'bench'])\n"
            "    parser.parse_args()\n"
            "    print('| dim | kernel us | reference us |')\n"
            "    print('|---|---|---|')\n"
            "    for dim in (1024, 2048, 4096, 8192):\n"
            "        print(f'| {dim} | {run_kernel(dim)} | {dim / 50.0} |')\n"
            "    print('4/4 cases passed')\n"
            "\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
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
                "add bench target",
            ],
            cwd=self.repo,
        )

    def rewrite_bench(self, body):
        """Return a mutate() that replaces the bench target wholesale."""

        def mutate(repo):
            (repo / self.BENCH_TARGET).write_text(body)

        return mutate

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
        perf=True,
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
        if not perf:
            command.append("--no-perf")
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

    @staticmethod
    def gpu_requiring_change(repo):
        (repo / "aiter" / "kernel.py").write_text("VALUE = 1\n# candidate\n")
        (repo / "tests" / "test_needs_device.py").write_text(
            "import os\n"
            "\n"
            "def run_kernel(M, N, dtype_str):\n"
            "    assert M > 0 and N > 0 and dtype_str\n"
            "\n"
            "def test_needs_device():\n"
            '    assert os.environ.get("HIP_VISIBLE_DEVICES"), "target needs a device"\n'
            "    run_kernel(7, 257, 'f32')\n"
        )

    def assert_complete_stage_objects(self, report):
        assert_stage_set(report["stages"])
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
        self.assertEqual({}, report["arch_coverage_basis"])
        # This target was observed to need no device, so its correctness stages run
        # rather than abstain. Everything except gpu_claim can therefore pass, which is
        # exactly why PASS must still be withheld: nothing here exercised an
        # architecture, so a clearance would be a claim no stage established.
        self.assertEqual("not-required", report["test_selection"]["gpu_requirement"])
        self.assertEqual("pass", report["stages"]["correctness_repo_tests"]["status"])
        self.assertEqual("pass", report["stages"]["correctness_s1_grid"]["status"])
        self.assert_complete_stage_objects(report)

    def test_no_gpu_withholds_correctness_from_a_target_that_needs_a_device(self):
        patch = self.fixture.make_patch(self.gpu_requiring_change, "needs-device.patch")
        no_gpu_picker = self.fixture.tools / "no-gpu-picker"
        write_executable(no_gpu_picker, "#!/usr/bin/env bash\nexit 1\n")

        result, report = self.fixture.validate(
            patch,
            tests="tests/test_needs_device.py",
            picker=no_gpu_picker,
            grid=False,
            expected_route="test_needs_device:run_kernel",
        )

        self.assertEqual(2, result.returncode)
        self.assertEqual("INCONCLUSIVE", report["verdict"])
        self.assertEqual("required", report["test_selection"]["gpu_requirement"])
        self.assertEqual("skip", report["stages"]["correctness_repo_tests"]["status"])
        self.assertEqual({}, report["arch_coverage"])
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
                "def test_new():\n    assert False, 'candidate failure'\n"
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
                "def verify_kernel():\n    return True\n"
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

    # ---- perf stage -------------------------------------------------------------
    # A measured regression is the only thing here allowed to change a verdict, so the
    # tests are weighted toward the false-positive side: one case must fire, four cases
    # must not. A perf stage that blocks a good PR gets switched off within a week, and
    # then it catches nothing at all.

    def bench_body(self, scale, trailer=""):
        self.fixture.add_bench_target()
        source = (self.fixture.repo / self.fixture.BENCH_TARGET).read_text()
        return source.replace("SCALE = 1.0", f"SCALE = {scale}") + trailer

    def validate_bench(self, scale, name, trailer=""):
        body = self.bench_body(scale, trailer)
        patch = self.fixture.make_patch(self.fixture.rewrite_bench(body), name)
        return self.fixture.validate(
            patch,
            tests=self.fixture.BENCH_TARGET,
            grid=False,
            expected_route="test_bench:run_kernel",
        )

    def test_perf_regression_is_should_fix_and_flips_the_verdict(self):
        result, report = self.fixture.validate(
            self.fixture.make_patch(
                self.fixture.rewrite_bench(self.bench_body("1.25")), "perf-slow.patch"
            ),
            tests=self.fixture.BENCH_TARGET,
            grid=False,
            expected_route="test_bench:run_kernel",
        )
        perf = report["stages"]["perf"]
        self.assertEqual("fail", perf["status"])
        self.assertLess(perf["median_ratio"], 0.95)
        self.assertEqual("NEEDS_WORK", report["verdict"])
        self.assertEqual(1, result.returncode)
        self.assertTrue(
            any(
                item["stage"] == "perf" and item["severity"] == "should-fix"
                for item in report["findings"]
            )
        )
        # The untouched reference column must sit at 1.0 and must NOT be what the
        # verdict was drawn from -- that is the whole reason the gate takes the minimum
        # across columns rather than the mean.
        self.assertEqual("kernel us", perf["worst_column"])
        self.assertAlmostEqual(1.0, perf["columns"]["reference us"]["median_ratio"], 2)
        # A regression finding has to ship its reproducer, or nobody can check it.
        self.assertTrue(perf["regressed_rows"])
        self.assertIn("--scenario bench", perf["command"])

    def test_perf_improvement_does_not_gate(self):
        result, report = self.validate_bench("0.5", "perf-fast.patch")
        perf = report["stages"]["perf"]
        self.assertEqual("pass", perf["status"])
        self.assertNotEqual("NEEDS_WORK", report["verdict"])
        self.assertNotEqual(1, result.returncode)
        self.assertFalse(
            any(
                item["stage"] == "perf" and item["severity"] == "should-fix"
                for item in report["findings"]
            )
        )

    def test_perf_ignores_movement_inside_the_threshold(self):
        # 2% slower is under the 5% bar; firing here would make the stage untrustworthy.
        _, report = self.validate_bench("1.02", "perf-noise.patch")
        self.assertEqual("pass", report["stages"]["perf"]["status"])

    def test_perf_never_fires_on_a_nonzero_exit(self):
        # Head prints a table that looks like a 4x regression and then dies. The table is
        # truncated at an unknown point, so any ratio drawn from it is meaningless; the
        # stage must report skip, never fail.
        _, report = self.validate_bench(
            "4.0", "perf-crash.patch", trailer="\nraise SystemExit(1)\n"
        )
        perf = report["stages"]["perf"]
        self.assertEqual("skip", perf["status"])
        self.assertNotIn("median_ratio", perf)
        self.assertFalse(
            any(
                item["stage"] == "perf" and item["severity"] == "should-fix"
                for item in report["findings"]
            )
        )

    def test_perf_is_skipped_when_the_target_has_no_benchmark_harness(self):
        patch = self.fixture.make_patch(self.harmless_change, "perf-noharness.patch")
        _, report = self.fixture.validate(patch)
        perf = report["stages"]["perf"]
        self.assertEqual("skip", perf["status"])
        self.assertIn("no benchmark entry point", perf["note"])
        self.assertNotIn("median_ratio", perf)

    def test_perf_can_be_disabled(self):
        patch = self.fixture.make_patch(
            self.fixture.rewrite_bench(self.bench_body("1.25")), "perf-off.patch"
        )
        result, report = self.fixture.validate(
            patch,
            tests=self.fixture.BENCH_TARGET,
            grid=False,
            expected_route="test_bench:run_kernel",
            perf=False,
        )
        self.assertEqual("skip", report["stages"]["perf"]["status"])
        self.assertIn("--no-perf", report["stages"]["perf"]["note"])
        self.assertNotEqual("NEEDS_WORK", report["verdict"])
        self.assertNotEqual(1, result.returncode)

    def run_review_gate(self, report, patch):
        """Feed a report through review-pr's real identity gate.

        The gate is the seam between the two skills, and it is the only place that can
        catch a report whose perf fields do not hang together. Extracting the block from
        SKILL.md rather than restating it means the two cannot drift apart silently.
        """
        skill = (SKILL_DIR.parent / "review-pr" / "SKILL.md").read_text()
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY\n", skill, re.DOTALL)
        gate = self.fixture.root / "gate.py"
        gate.write_text(blocks[1])
        meta = self.fixture.root / "gate-meta.json"
        meta.write_text(json.dumps({"headRefOid": report["repo"]["head"]}))
        base = self.fixture.root / "gate-base.txt"
        base.write_text(report["repo"]["base"] + "\n")
        target = self.fixture.root / "gate-report.json"
        target.write_text(json.dumps(report))
        return run(
            [
                sys.executable,
                str(gate),
                str(meta),
                str(base),
                str(patch),
                str(SKILL_DIR / "report_schema.json"),
                str(target),
                str(self.fixture.root / "gate-out.json"),
            ],
            check=False,
        )

    def test_perf_report_survives_the_review_identity_gate(self):
        patch = self.fixture.make_patch(
            self.fixture.rewrite_bench(self.bench_body("1.25")), "perf-gate.patch"
        )
        _, report = self.fixture.validate(
            patch,
            tests=self.fixture.BENCH_TARGET,
            grid=False,
            expected_route="test_bench:run_kernel",
        )
        self.assertEqual("fail", report["stages"]["perf"]["status"])

        accepted = self.run_review_gate(report, patch)
        self.assertEqual(0, accepted.returncode, accepted.stdout + accepted.stderr)
        self.assertIn("perf stage: fail", accepted.stdout)
        self.assertIn("median_ratio", accepted.stdout)

    def test_review_gate_rejects_a_perf_result_that_contradicts_itself(self):
        patch = self.fixture.make_patch(
            self.fixture.rewrite_bench(self.bench_body("1.25")), "perf-launder.patch"
        )
        _, report = self.fixture.validate(
            patch,
            tests=self.fixture.BENCH_TARGET,
            grid=False,
            expected_route="test_bench:run_kernel",
        )

        # Every individual field stays well-formed; only the status is flipped. Without a
        # cross-check the card would print "NO REGRESSION" over a measured 20% regression.
        laundered = json.loads(json.dumps(report))
        laundered["stages"]["perf"]["status"] = "pass"
        laundered["findings"] = [
            item for item in laundered["findings"] if item["stage"] != "perf"
        ]
        laundered["verdict"] = "INCONCLUSIVE"
        laundered["process_exit_code"] = 2
        rejected = self.run_review_gate(laundered, patch)
        self.assertNotEqual(0, rejected.returncode)
        self.assertIn("contradicts its own numbers", rejected.stdout + rejected.stderr)

        # A perf failure must also keep its finding: drop it and the report is refused.
        stripped = json.loads(json.dumps(report))
        stripped["findings"] = [
            item for item in stripped["findings"] if item["stage"] != "perf"
        ]
        stripped["verdict"] = "INCONCLUSIVE"
        stripped["process_exit_code"] = 2
        refused = self.run_review_gate(stripped, patch)
        self.assertNotEqual(0, refused.returncode)
        self.assertIn("no should-fix finding", refused.stdout + refused.stderr)

    def test_timing_run_does_not_dirty_the_worktree(self):
        """A bench harness that writes results must not disable correctness validation.

        Real aiter targets are pytest modules whose `__main__` bench drops a results file
        (tuned_op_bench.csv) in the repo root. The baseline phase asserts a clean worktree
        after the base runs, so an artifact left behind by the timing run sets BASE_READY=0
        and the entire head correctness phase is skipped. Caught by measurement, not review:
        the same target went PASS with --no-perf and INCONCLUSIVE with perf enabled, with
        head correctness never executed. A perf stage that silently switches off correctness
        validation is worse than no perf stage at all.
        """
        source = (self.fixture.repo / "tests" / "test_sample.py").read_text()
        source = (
            "def perftest(fn):\n    return fn\n\n" + source + "\ndef _bench():\n"
            "    import os\n"
            "    print('| dim | k us |')\n"
            "    print('|---|---|')\n"
            "    for d in (1024, 2048, 4096, 8192):\n"
            "        print(f'| {d} | {d / 100.0} |')\n"
            "    open('tuned_op_bench.csv', 'w').write('done\\n')\n"
            "    os.makedirs('bench_out', exist_ok=True)\n"
            "    open('bench_out/x.json', 'w').write('{}')\n"
            "\n@perftest\ndef _timed():\n    return None\n"
            "\nif __name__ == '__main__':\n    _bench()\n"
        )
        (self.fixture.repo / "tests" / "test_sample.py").write_text(source)
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
                "bench harness that writes results",
            ],
            cwd=self.fixture.repo,
        )

        patch = self.fixture.make_patch(self.harmless_change, "perf-artifacts.patch")
        result, report = self.fixture.validate(patch, grid_value="7,257,f32;8,513,bf16")

        self.assertEqual("PASS", report["verdict"])
        self.assertEqual(0, result.returncode)
        self.assertEqual("pass", report["stages"]["baseline_control"]["status"])
        self.assertEqual("pass", report["stages"]["correctness_repo_tests"]["status"])
        # It must not merely avoid breaking things -- it must actually have measured.
        self.assertEqual("pass", report["stages"]["perf"]["status"])
        # Both the stray file and the stray directory are gone.
        leftover = run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=self.fixture.repo,
        ).stdout.strip()
        self.assertEqual("", leftover)

    PERF_LINE_TARGET = "tests/test_perfline.py"

    def add_perf_line_target(self):
        """Commit a target using aiter's OTHER timing convention.

        The `@benchmark`/`run_perftest` pair prints a markdown table, but the older bare
        `perftest` decorator's callers hand-roll an f-string per test. Eleven real kernel
        targets print only this shape -- test_moe.py, test_pa_v1.py, test_rope.py,
        test_layernorm2d.py among them -- so a scraper that reads tables alone would detect
        a harness, spend a full base+head run, and then measure nothing.
        """
        body = (
            "def perftest(fn):\n    return fn\n"
            "SCALE = 1.0\n"
            "def main():\n"
            "    for d in ((128, 8192), (256, 4096), (512, 2048), (1024, 1024)):\n"
            "        c = 13.1 * SCALE\n"
            "        print(f'[perf] dim: {d!s:<20}, dtype: torch.bfloat16, "
            "torch avg: {14.2:<8.2f} us, ck avg: {c:<8.2f} us')\n"
            "    print('4/4 cases passed')\n"
            "if __name__ == '__main__':\n    main()\n"
        )
        (self.fixture.repo / self.PERF_LINE_TARGET).write_text(body)
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
                "perf-line target",
            ],
            cwd=self.fixture.repo,
        )
        return body

    def validate_perf_line(self, body, scale, name):
        patch = self.fixture.make_patch(
            lambda repo: (repo / self.PERF_LINE_TARGET).write_text(
                body.replace("SCALE = 1.0", f"SCALE = {scale}")
            ),
            name,
        )
        return self.fixture.validate(
            patch,
            tests=self.PERF_LINE_TARGET,
            grid=False,
            expected_route="test_perfline:main",
        )

    def test_perf_reads_aiters_perf_line_format(self):
        body = self.add_perf_line_target()
        _, report = self.validate_perf_line(body, "1.3", "perfline-slow.patch")
        perf = report["stages"]["perf"]
        self.assertEqual("fail", perf["status"])
        self.assertEqual("ck us", perf["worst_column"])
        self.assertEqual(4, perf["matched_rows"])
        # The untouched reference column must not be what the gate fired on.
        self.assertAlmostEqual(1.0, perf["columns"]["torch us"]["median_ratio"], 2)
        self.assertEqual("NEEDS_WORK", report["verdict"])

    def test_perf_line_format_does_not_false_positive(self):
        body = self.add_perf_line_target()
        _, report = self.validate_perf_line(
            body, "1.0  # unchanged", "perfline-same.patch"
        )
        perf = report["stages"]["perf"]
        self.assertEqual("pass", perf["status"])
        self.assertNotEqual("NEEDS_WORK", report["verdict"])

    def test_each_side_is_measured_more_than_once(self):
        """The 0.95 threshold is only defensible as a best-of-N comparison.

        Measured on this box, five warm repeat runs of an unchanged
        op_tests/test_layernorm2d.py gave `ck avg` of 13.10 20.98 20.70 13.28 13.17 us --
        bimodal, 1.60x spread, on code that did not change, while the untouched `torch avg`
        reference column held to 1.03x. One run per side would land the ratio anywhere in
        [0.62, 1.60] and fire a false regression roughly half the time. If the repeat count
        ever silently drops to 1, the threshold stops being defensible.
        """
        body = self.add_perf_line_target()
        _, report = self.validate_perf_line(body, "1.3", "perfline-repeats.patch")
        repeats = report["stages"]["perf"]["repeats"]
        self.assertGreaterEqual(repeats["base"], 3)
        self.assertGreaterEqual(repeats["head"], 3)
        self.assertIn("min", repeats["reduction"])

    def test_perf_skip_does_not_prevent_a_pass(self):
        # perf is not in finish_report's required-stage set. A target with no benchmark
        # harness is the common case -- 26 of the 123 targets in aiter's op_tests/ -- and
        # it must still be able to reach PASS on correctness alone. If a skipped perf
        # stage could hold a verdict at INCONCLUSIVE, the stage would be unshippable.
        patch = self.fixture.make_patch(self.harmless_change, "perf-skip-pass.patch")
        result, report = self.fixture.validate(patch, grid_value="7,257,f32;8,513,bf16")
        self.assertEqual("skip", report["stages"]["perf"]["status"])
        self.assertEqual("PASS", report["verdict"])
        self.assertEqual(0, result.returncode)
        self.assertFalse(
            any(
                item["stage"] == "perf" and item["severity"] != "note"
                for item in report["findings"]
            )
        )


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

    def test_perf_harness_detection_agrees_across_both_skills(self):
        """review-pr and the validator must classify a target identically.

        Each file carries a comment telling the next reader to keep the two in step. A
        comment cannot enforce that. If they drift, review-pr prints a manual recipe for a
        harness the validator declined to use -- or, worse, prints "no benchmark entry
        point" for a target the validator happily timed.
        """
        review_skill = (SKILL_DIR.parent / "review-pr" / "SKILL.md").read_text()
        validator = VALIDATOR.read_text()

        review_body = re.search(
            r"def perf_command\(path\):(.*?)\n\n", review_skill, re.DOTALL
        )
        validator_body = re.search(
            r"perf_detect\(\).*?<<'PY'\n(.*?)\nPY", validator, re.DOTALL
        )
        self.assertIsNotNone(review_body)
        self.assertIsNotNone(validator_body)

        for name, body in (
            ("review-pr", review_body.group(1)),
            ("validate_pr.sh", validator_body.group(1)),
        ):
            self.assertIn('"--scenario" in text', body, name)
            self.assertIn('"bench" in text', body, name)
            self.assertIn('"perftest" in text', body, name)
            # `run_perftest` is a strict subset of `perftest`, so testing the longer name
            # only narrows coverage. It missed 12 of aiter's 123 op_tests/ targets, every
            # one of which does have a timing harness.
            self.assertNotIn('"run_perftest" in text', body, name)

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
