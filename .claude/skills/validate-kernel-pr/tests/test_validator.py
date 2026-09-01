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
# would turn every optional stage into a failure in every test in this file; asserting a
# subset plus this allowlist keeps the real intent -- every required stage present and
# well-formed, and nothing unrecognised sitting alongside them.
OPTIONAL_STAGES = {"perf"}


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
        shape_env="VALIDATOR_TEST_GRID",
        shape_arg=None,
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
            command.extend(["--grid", grid_value])
            if shape_env:
                command.extend(["--shape-env", shape_env])
        if shape_arg:
            command.extend(["--shape-arg", shape_arg])
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

    def test_unfound_shape_arg_reports_a_missing_hook_not_an_absent_grid(self):
        # A --shape-arg naming a flag the target does not accept used to reach the branch that
        # says "no shape grid was configured" -- a fact about the caller, when what happened is
        # a fact about the target. Both skip, so only the reason distinguishes a validator that
        # could not find the hook from a caller that never asked for one, and that reason is
        # the whole point of a stage that reports its own limits.
        def add_script_target(repo):
            (repo / "tests" / "verify_kernel.py").write_text(
                "def verify_kernel():\n"
                "    return True\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    assert verify_kernel()\n"
                "    print('56/56 cases passed')\n"
            )

        patch = self.fixture.make_patch(add_script_target, "unfound-shape-arg.patch")
        _, report = self.fixture.validate(
            patch,
            tests="tests/verify_kernel.py",
            shape_env=None,
            shape_arg="--shapes",
        )

        # Asserted before the grid_channel field below, so that this test fails on the reason
        # the skip gives rather than on the field that was added to carry it.
        self.assertEqual("skip", report["stages"]["correctness_s1_grid"]["status"])
        note = report["stages"]["correctness_s1_grid"]["note"]
        self.assertNotIn("no configured shape override", note)
        self.assertIn("not referenced", note)
        self.assertIn("--shapes", note)
        self.assertEqual(
            "hook-not-found",
            report["stages"]["baseline_control"]["s1_grid"]["state"],
        )
        self.assertEqual("cli-flag", report["test_selection"]["grid_channel"])

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


def new_file_diff(path, source):
    """A diff that CREATES `path` with `source`.

    Every line is an addition and the post image is fully contained in the diff, so the
    scanner needs neither --source-root nor a git object store to recover it. Line
    numbers are 1-based, matching the post image, which is what the scanner filters
    candidates on.
    """
    lines = source.splitlines()
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "index 0000000..1111111\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
    ) + "".join(f"+{line}\n" for line in lines)


class IndexScannerTests(unittest.TestCase):
    def scan(self, diff_text, directory=None):
        with tempfile.TemporaryDirectory() as scratch:
            diff = Path(scratch) / "candidate.diff"
            diff.write_text(diff_text)
            result = run(
                [str(SCANNER), "--diff", str(diff), "--json"],
                cwd=directory or scratch,
            )
            payload = json.loads(result.stdout)
        return payload

    def scan_source(self, source, path="kernel.py"):
        return self.scan(new_file_diff(path, source))

    def test_broadcast_subscript_operand_is_found(self):
        r"""The regression the reviewer's blocker was really about.

        `(offs_token // top_k)[:, None] * stride_gm` is the standard Triton pointer
        idiom and the exact shape of the ROCm/aiter#4978 overflow. The old regex operand
        class was `[\w\.\[\]]+`, which spans neither the comma nor the space inside
        `[:, None]`, so it matched nothing on either of these two lines and the scanner
        reported the PR clean. Structurally both are candidates: a multiply feeding an
        addition chain, with a plain (non-constexpr) kernel parameter as an operand.
        """
        payload = self.scan_source(
            "import triton\n"
            "import triton.language as tl\n"
            "\n"
            "@triton.jit\n"
            "def moe_wgrad_kernel(a_ptr, offs_token, top_k, stride_gm, stride_gn,\n"
            "                     BLOCK_N: tl.constexpr):\n"
            "    offs_n = tl.arange(0, BLOCK_N)\n"
            "    a_ptrs = (\n"
            "        a_ptr\n"
            "        + (offs_token // top_k)[:, None] * stride_gm\n"
            "        + offs_n[None, :] * stride_gn\n"
            "    )\n"
            "    return a_ptrs\n"
        )

        expressions = sorted(row["expression"] for row in payload["candidates"])
        self.assertEqual(
            [
                "(offs_token // top_k)[:, None] * stride_gm",
                "offs_n[None, :] * stride_gn",
            ],
            expressions,
        )
        self.assertEqual(2, payload["index_stride_candidates"])
        # Each row names the runtime parameters that made it a candidate, so the
        # reviewer can judge production scale without re-deriving provenance.
        provenance = {
            row["expression"]: row["runtime_params"] for row in payload["candidates"]
        }
        self.assertEqual(
            ["offs_token", "stride_gm", "top_k"],
            provenance["(offs_token // top_k)[:, None] * stride_gm"],
        )
        # The unannotated-parameter list is now scoped to parameters this function
        # actually multiplies in pointer arithmetic -- `a_ptr` is a parameter too and is
        # correctly absent -- rather than to every parameter whose name looked stride-
        # ish.
        self.assertEqual(
            {"offs_token", "stride_gm", "stride_gn", "top_k"},
            {row["name"] for row in payload["parameters"]},
        )

    def test_widening_hoisted_to_an_earlier_line_suppresses_the_candidate(self):
        """This is the shape of the FIX (aiter#5132), not the defect.

        The widening is applied on its own statement and carried into the multiply
        through a local name. A scanner that only looks at the multiply's own line fires
        here, which would make it report every fixed kernel as still defective.
        """
        payload = self.scan_source(
            "import triton\n"
            "import triton.language as tl\n"
            "\n"
            "@triton.jit\n"
            "def moe_wgrad_kernel(a_ptr, offs_token, top_k, stride_gm):\n"
            "    token_row = (offs_token // top_k).to(tl.int64)\n"
            "    a_ptrs = a_ptr + token_row[:, None] * stride_gm\n"
            "    return a_ptrs\n"
        )

        self.assertEqual([], payload["candidates"])
        self.assertEqual(0, payload["index_stride_candidates"])

    def test_flydsl_capitalised_widening_spelling_is_recognised(self):
        """FlyDSL writes `fx.Int64(...)`; Triton writes `tl.int64`.

        WIDEN_ATTRS was matched case-sensitively, so a kernel that had been explicitly widened
        in FlyDSL's own spelling was reported as an unwidened overflow candidate -- the
        scanner publishing its own vocabulary gap as a defect in the code. The list is a list
        of widening FORMS, and how a form is capitalised is not part of what it does.
        """
        payload = self.scan_source(
            "import flydsl as fx\n"
            "\n"
            "def kernel(base_ptr, offs, stride):\n"
            "    return base_ptr + fx.Int64(offs) * stride\n"
        )

        self.assertEqual([], payload["candidates"])
        self.assertEqual(0, payload["index_stride_candidates"])

    def test_constexpr_only_operand_is_not_a_candidate(self):
        """A tile-constant multiplicand bounds the product at compile time.

        `k_ptrs += BLOCK_N * stride_kn` advances a pointer by one tile and cannot
        overflow. It nonetheless matches "runtime parameter inside a multiply inside
        pointer arithmetic" exactly, so provenance -- not the name -- has to exclude it.
        """
        payload = self.scan_source(
            "import triton\n"
            "import triton.language as tl\n"
            "\n"
            "@triton.jit\n"
            "def gemm_kernel(k_ptr, stride_kn, BLOCK_N: tl.constexpr):\n"
            "    k_ptrs = k_ptr + BLOCK_N * stride_kn\n"
            "    return k_ptrs\n"
        )

        self.assertEqual([], payload["candidates"])

    def test_json_count_is_deduplicated(self):
        """Identical expressions collapse into ONE row that carries its occurrences.

        The old contract counted every textual site. One reasoning step clears one
        distinct expression however many times a generated kernel family repeats it, so
        the count is now of distinct expressions and the repetition is carried in
        `occurrences`/`lines` rather than discarded.
        """
        payload = self.scan_source(
            "import triton\n"
            "import triton.language as tl\n"
            "\n"
            "@triton.jit\n"
            "def twin_kernel(a_ptr, b_ptr, offs_m, stride_am):\n"
            "    p1 = a_ptr + offs_m[:, None] * stride_am\n"
            "    p2 = b_ptr + offs_m[:, None] * stride_am\n"
            "    return p1, p2\n"
        )

        self.assertEqual(1, payload["index_stride_candidates"])
        row = payload["candidates"][0]
        self.assertEqual("offs_m[:, None] * stride_am", row["expression"])
        self.assertEqual(2, row["occurrences"])
        self.assertEqual([6, 7], row["lines"])
        self.assertEqual(6, row["line"])
        # Renamed key: the old `untyped_stride_parameters` was a name-list verdict; this
        # one counts runtime parameters the diff added with no width annotation.
        self.assertEqual(2, payload["unannotated_runtime_parameters"])
        self.assertEqual(3, payload["total_candidates"])
        self.assertEqual([], payload["unscanned"])

    def test_candidate_is_found_with_names_from_no_list(self):
        """Proof that the INDEXY/STRIDEY name lists are actually gone.

        Not one identifier here matches the deleted patterns (idx/_id/block/row/token/
        offset ... times stride/pitch/hidden_dim). The old scanner was silent on this
        file; the structure is identical to the defect, so the new one must not be.
        """
        payload = self.scan_source(
            "import triton\n"
            "import triton.language as tl\n"
            "\n"
            "@triton.jit\n"
            "def frobnicate(base_ptr, quux, WIDTH: tl.constexpr):\n"
            "    zork = tl.arange(0, WIDTH)\n"
            "    return base_ptr + zork * quux\n"
        )

        self.assertEqual(1, payload["index_stride_candidates"])
        self.assertEqual("zork * quux", payload["candidates"][0]["expression"])
        self.assertEqual(["quux"], payload["candidates"][0]["runtime_params"])

    def test_unrecoverable_post_image_is_reported_not_counted_clean(self):
        """A file the scanner could not read is not a file with no defects.

        The diff modifies an existing file, so the post image lives in a blob this
        object store does not have. Dropping it silently would make an incomplete scan
        indistinguishable from a clean one -- the exact failure this skill argues
        against.
        """
        missing_blob = "b" * 40
        diff = (
            "diff --git a/kernel.py b/kernel.py\n"
            f"index {'a' * 40}..{missing_blob} 100644\n"
            "--- a/kernel.py\n"
            "+++ b/kernel.py\n"
            "@@ -1,1 +1,2 @@\n"
            " import triton\n"
            "+x = 1\n"
        )
        with tempfile.TemporaryDirectory() as outside_git:
            payload = self.scan(diff, directory=outside_git)
            plain = run(
                [str(SCANNER), "--diff", str(self._write(outside_git, diff))],
                cwd=outside_git,
            )

        self.assertEqual([], payload["candidates"])
        self.assertEqual(1, len(payload["unscanned"]))
        self.assertEqual("kernel.py", payload["unscanned"][0]["path"])
        self.assertTrue(payload["unscanned"][0]["reason"])
        self.assertIn(missing_blob, payload["unscanned"][0]["reason"])
        # And it is loud in the human-readable output: 0 candidates here must not read
        # as a clearance.
        self.assertIn("NOT SCANNED", plain.stdout)
        self.assertIn("D9 CANNOT be cleared", plain.stdout)

    @staticmethod
    def _write(directory, diff_text):
        path = Path(directory) / "plain.diff"
        path.write_text(diff_text)
        return path


class ScannerScopeTests(unittest.TestCase):
    def _scan(self, source):
        with tempfile.TemporaryDirectory() as directory:
            diff = Path(directory) / "d.diff"
            body = "".join(f"+{line}\n" for line in source.splitlines())
            diff.write_text(
                "diff --git a/k.py b/k.py\n--- /dev/null\n+++ b/k.py\n"
                f"@@ -0,0 +1,{len(source.splitlines())} @@\n" + body
            )
            return json.loads(run([str(SCANNER), "--diff", str(diff), "--json"]).stdout)

    def test_a_nested_helper_inherits_its_kernels_device_scope(self):
        # _scan_body used ast.walk, which descends through nested defs, so every expression in
        # a nested helper was scanned in the ENCLOSING function's scope -- with the outer
        # function's parameters and its host/device verdict. Four real candidates inside a
        # @flyc.kernel body were classified host-side that way, which is a miss.
        payload = self._scan(
            "def build(stride_a):\n"
            "    @flyc.kernel(name='k')\n"
            "    def kernel_gemm(stride_b):\n"
            "        def helper(row):\n"
            "            return base + row * stride_b\n"
            "        return helper\n"
        )
        self.assertEqual(1, payload["index_stride_candidates"], payload)
        self.assertEqual(0, payload["host_scope_candidates"], payload)
        self.assertEqual("device", payload["candidates"][0]["scope"])

    def test_host_side_arithmetic_is_listed_apart_from_device_candidates(self):
        # int32 overflow is a device concern; host-side FLOP accounting in the same shape is
        # not D9. It is listed rather than dropped, because the device test is a spelling.
        payload = self._scan(
            "def _flops_bytes(rows, stride_x):\n    return total + rows * stride_x\n"
        )
        self.assertEqual(0, payload["index_stride_candidates"], payload)
        self.assertEqual(1, payload["host_scope_candidates"], payload)


class ProbeReceiptTests(unittest.TestCase):
    """Unit-level checks on the probe and the evidence checker it feeds."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def run_probe(self, target_source, shape_vars, route_suffix="run_kernel"):
        """Generate the probe exactly as validate_pr.sh does and run pytest under it."""
        directory = Path(self.tempdir.name)
        target = directory / "test_route.py"
        target.write_text(target_source)
        receipt = directory / "execution-receipt.json"
        route = f"test_route:{route_suffix}"
        probe = directory / "validation_probe_under_test.py"
        probe.write_text(
            (SKILL_DIR / "validation_probe.py").read_text()
            + f"\n_VALIDATION_EXPECTED_ROUTE = {route!r}\n"
            + f"_VALIDATION_SHAPE_VARS = {shape_vars!r}\n"
            + f"_VALIDATION_RECEIPT_PATH = {str(receipt)!r}\n"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(directory)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "validation_probe_under_test",
                str(target),
                "-q",
                "-o",
                f"cache_dir={directory}/pytest-cache",
            ],
            cwd=directory,
            env=environment,
        )
        return route, receipt, json.loads(receipt.read_text())

    def test_two_calls_to_one_route_produce_two_shape_rows(self):
        """Each call is its own row.

        The pending-capture table is keyed by the frame OBJECT. Keying it by id() would
        be wrong for exactly this case: a freed frame's address is reused, so the second
        call's frame can present the identity of the first and be treated as already
        recorded.
        """
        _, _, receipt = self.run_probe(
            "def run_kernel(spec):\n"
            "    M, N, dtype_str = spec.split(',')\n"
            "    M = int(M)\n"
            "    N = int(N)\n"
            "    assert M > 0\n"
            "\n"
            "def test_two_calls():\n"
            "    run_kernel('7,257,f32')\n"
            "    run_kernel('8,513,bf16')\n",
            "M,N,dtype_str",
        )

        self.assertEqual(["7,257,f32", "8,513,bf16"], receipt["executed_shapes"])

    def test_requested_but_empty_capture_is_declared_in_the_result(self):
        """An empty `executed_shapes`, from a route that bound none of the requested
        names, is an absence of evidence. The result has to say so rather than let a
        consumer read the empty list as "no shapes were needed"."""
        route, receipt_path, receipt = self.run_probe(
            "def run_kernel(payload):\n"
            "    assert payload\n"
            "\n"
            "def test_one_call():\n"
            "    run_kernel({'shape': (7, 257)})\n",
            "M,N,dtype_str",
        )
        self.assertEqual([], receipt["executed_shapes"])
        # Carried in the receipt so the checker can tell "no shapes were asked for" from
        # "shapes were asked for and none arrived".
        self.assertEqual(["M", "N", "dtype_str"], receipt["shape_vars"])

        result = json.loads(
            run(
                [
                    sys.executable,
                    str(SKILL_DIR / "validate_evidence.py"),
                    "receipt",
                    str(receipt_path),
                    "--expected-route",
                    route,
                    "--grid",
                    "",
                ]
            ).stdout
        )

        self.assertEqual("pass", result["status"])
        self.assertEqual([], result["executed_shapes"])
        self.assertEqual(["M", "N", "dtype_str"], result["shape_capture"]["requested"])
        self.assertEqual(0, result["shape_capture"]["observed"])
        self.assertIn("makes no claim", result["shape_capture"]["note"])
