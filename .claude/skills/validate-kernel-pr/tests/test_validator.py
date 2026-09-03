import fcntl
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
# target exposes a benchmark harness and both phases completed, so asserting an exact stage
# set would turn an optional stage into a failure in every test in this file.
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
            "def amdsmi_get_gpu_enumeration_info(handle): return {'hip_id': 57}\n"
            "def amdsmi_get_gpu_asic_info(handle):\n"
            "    return {'market_name': 'Synthetic GPU', "
            "'target_graphics_version': 'gfx-test'}\n"
            "def amdsmi_get_gpu_activity(handle): return {'gfx_activity': 0}\n"
            "def amdsmi_get_gpu_device_bdf(handle): return '0000:00:00.0'\n"
            "def amdsmi_get_gpu_vram_usage(handle):\n"
            "    return {'vram_used': 256, 'vram_total': 294912}\n"
        )
        self.picker = self.tools / "pick-idle-gpu.py"
        # 57 is deliberately NOT a real device index on any host this suite runs on. The
        # validator locks /tmp/gpu-<index>.lock for whatever index it claims, so a fixture
        # that named a real GPU would contend with anything genuinely using that device --
        # and report NO_GPU for a reason that has nothing to do with the code under test.
        write_executable(self.picker, "#!/usr/bin/env bash\nprintf '57\\n'\n")

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
        perf=True,
        shape_env="VALIDATOR_TEST_GRID",
        shape_arg=None,
        shape_argnames=None,
        shape_vars="M,N,dtype_str",
        tol_table="f32=1e-5,f16=2e-3,bf16=1e-2",
        use_picker_env=True,
        cwd=None,
        axes=(),
        perf_control_column=None,
        runner=None,
    ):
        report = self.root / f"{patch.stem}-report.json"
        # `cwd` exists for one reason: the validator has to accept RELATIVE --patch/--out from
        # whatever directory the caller happens to be in. Passing them absolute, as every other
        # test does, cannot see a path resolved against the wrong base.
        patch_arg = os.path.relpath(patch, cwd) if cwd else str(patch)
        report_arg = os.path.relpath(report, cwd) if cwd else str(report)
        command = [
            str(VALIDATOR),
            "--repo",
            str(self.repo),
            "--patch",
            patch_arg,
            "--head-sha",
            "b" * 40,
            "--target",
            tests,
            "--expected-route",
            expected_route,
            "--shape-vars",
            shape_vars,
            "--tol-table",
            tol_table,
            "--label",
            patch.stem,
            "--out",
            report_arg,
        ]
        if grid:
            if shape_env:
                command.extend(["--shape-env", shape_env])
            command.extend(["--grid", grid_value])
        if shape_arg:
            command.extend(["--shape-arg", shape_arg])
        if shape_argnames:
            command.extend(["--shape-argnames", shape_argnames])
        for axis in axes:
            command.extend(["--axis", axis])
        if perf_control_column:
            command.extend(["--perf-control-column", perf_control_column])
        if runner:
            command.extend(["--runner", runner])
        if not perf:
            command.append("--no-perf")
        environment = os.environ.copy()
        # Some tests exercise the validator's own PICKER resolution order, which only
        # runs when the caller has not pinned one.
        if use_picker_env:
            environment["PICKER"] = str(picker or self.picker)
        else:
            environment.pop("PICKER", None)
        environment["PYTHONPATH"] = str(self.fake_modules)
        environment["TIMEOUT"] = "30"
        if python_bin:
            environment["PYTHON_BIN"] = str(python_bin)
        if pylib:
            environment["PYLIB"] = str(pylib)
        if path_prefix:
            environment["PATH"] = f"{path_prefix}:{environment['PATH']}"
        result = run(command, env=environment, cwd=cwd, check=False)
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
        patch = self.fixture.make_patch(
            self.gpu_requiring_change, "needs-device.patch"
        )
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
        # The target runs and passes, so correctness stays `pass` and nothing is blamed on
        # the author. What it does NOT do is earn architecture coverage: this fixture names a
        # route the script never calls, so the run has positive evidence that no watched work
        # reached the device. The old contract credited `gfx-test: runtime` here on the basis
        # `script-exit-zero-with-output`, which is a statement about the process, not the
        # kernel -- and is exactly what a silently-returning target produces.
        self.assertNotIn("gfx-test", report["arch_coverage_basis"])
        self.assertNotIn("gfx-test", report["arch_coverage"])
        self.assertEqual(
            0, report["stages"]["correctness_repo_tests"]["stats"]["observed_work"]
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

    def test_relative_patch_path_is_not_a_merge_failure(self):
        """A relative --patch is an invocation detail, not a defect in the PR.

        The path was read from the caller's cwd for the readability check and then handed to
        `git -C "$REPO_WT" apply`, which resolves it against the WORKTREE. The mismatch
        surfaced as `merge_sim: patch does not apply to the recorded base` -- a BLOCK verdict
        published against an author whose patch applies perfectly.
        """
        patch = self.fixture.make_patch(self.harmless_change, "relative-patch.patch")
        # Run from a directory that is neither the repo nor the worktree, with --patch and
        # --out given relative to it. Nested TWO levels deep on purpose: from one level the
        # relative path happens to resolve to the same file against the worktree as against
        # the caller, and the bug is invisible.
        elsewhere = self.fixture.root / "caller" / "cwd"
        elsewhere.mkdir(parents=True)

        _, report = self.fixture.validate(patch, cwd=elsewhere)

        merge = report["stages"]["merge_sim"]
        self.assertEqual("pass", merge["status"])
        self.assertFalse(
            [
                item
                for item in report["findings"]
                if item["stage"] == "merge_sim" and item["severity"] == "blocker"
            ]
        )

    def test_tol_table_reports_a_tolerance_above_the_loosest_reference(self):
        """--tol-table was parsed, validated, published -- and compared against nothing.

        The added `rtol = 5e-2` loosens nothing (it is a new assertion, not a widened one), so
        the `loosened` check stays silent. It is nonetheless above every reference tolerance
        the caller declared, which is the fact the flag exists to surface: a kernel defect
        smaller than that gap cannot turn this suite red.
        """

        def add_loose_tolerance(repo):
            path = repo / "tests" / "test_sample.py"
            path.write_text(
                path.read_text().replace(
                    "    atol = 1e-5\n",
                    "    atol = 1e-5\n    rtol = 5e-2\n",
                )
            )

        patch = self.fixture.make_patch(add_loose_tolerance, "tol-exceeds.patch")

        _, report = self.fixture.validate(patch)

        policy = report["stages"]["test_policy"]
        self.assertEqual([0.05], policy["exceeds_reference"])
        # No tolerance was WIDENED, so the older check must stay quiet -- the two findings are
        # about different things and one must not stand in for the other.
        self.assertNotIn("loosened", policy)
        findings = [
            item
            for item in report["findings"]
            if item["stage"] == "test_policy" and item["severity"] == "should-fix"
        ]
        self.assertEqual(1, len(findings))
        self.assertIn("loosest reference tolerance", findings[0]["detail"])

    def test_tol_table_is_silent_when_every_tolerance_is_within_reference(self):
        """The other half of the contract: the field is recorded even when empty.

        An absent key and an empty list read the same to a consumer that uses `.get`, so the
        comparison has to be visible as having been made and found nothing.
        """
        patch = self.fixture.make_patch(self.harmless_change, "tol-within.patch")

        _, report = self.fixture.validate(patch)

        policy = report["stages"]["test_policy"]
        self.assertEqual([], policy["exceeds_reference"])
        self.assertFalse(
            [
                item
                for item in report["findings"]
                if item["stage"] == "test_policy" and item["severity"] == "should-fix"
            ]
        )

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
        self.assertIn("is not passed to add_argument", note)
        self.assertIn("tests/verify_kernel.py", note)
        self.assertIn("--shapes", note)
        self.assertEqual(
            "hook-not-found",
            report["stages"]["baseline_control"]["s1_grid"]["state"],
        )
        # `grid_channel` names the channel that actually CARRIED the grid, so a hook
        # that was requested and not found leaves it empty; the reason field is where
        # the request survives, and it distinguishes a validator limit from a target
        # property.
        self.assertEqual("", report["test_selection"]["grid_channel"])
        self.assertIn(
            "--shapes", report["test_selection"]["grid_channel_reason"]
        )

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

    # ---- last-mile regressions found by running this skill against ROCm/aiter#4538 ----
    #
    # Every test below fails against the pre-fix validator, and each names the invariant it
    # protects rather than the PR that exposed it.

    #: A target shaped like aiter#4538's: a script with its own shape flag, its own
    #: head-count flag, and a route whose legality depends on the head count. The head-count
    #: constraint is the point -- it is a configuration the shape flag alone cannot reach.
    #: The kernel lives in its own module, as aiter#4538's does: a script target run under
    #: runpy sees its own functions as ``__main__:...``, so a route naming the module is only
    #: stable for code the target imports.
    AXIS_KERNEL = (
        "def run_kernel(M, N, dtype_str, num_heads):\n"
        "    # The kernel's tile is 32 rows, so a head count below that has no legal\n"
        "    # mapping. This is the shape of aiter#4538's MFMA_M assertion.\n"
        "    assert num_heads % 32 == 0, 'heads must be a multiple of 32'\n"
        "    return M * N\n"
    )

    AXIS_TARGET = (
        "import argparse\n"
        "import os\n"
        "import sys\n"
        "\n"
        "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
        "\n"
        "from axis_kernel import run_kernel\n"
        "\n"
        "\n"
        "def _shape(text):\n"
        "    M, N, dtype_str = text.split(',')\n"
        "    return int(M), int(N), dtype_str\n"
        "\n"
        "\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('-s', '--shapes', type=_shape, nargs='*',\n"
        "                        default=[(7, 257, 'f32'), (8, 64, 'f32')])\n"
        "    parser.add_argument('--num-heads', type=int, nargs='*',\n"
        "                        default=[64, 128])\n"
        "    args = parser.parse_args()\n"
        "    for M, N, dtype_str in args.shapes:\n"
        "        for num_heads in args.num_heads:\n"
        "            run_kernel(M, N, dtype_str, num_heads)\n"
        "    print('%d cases passed' % (len(args.shapes) * len(args.num_heads)))\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    AXIS_TARGET_PATH = "tests/axis_target.py"

    def _axis_patch(self, name, body=None):
        def mutate(repo):
            (repo / "tests" / "axis_kernel.py").write_text(self.AXIS_KERNEL)
            (repo / self.AXIS_TARGET_PATH).write_text(body or self.AXIS_TARGET)

        return self.fixture.make_patch(mutate, name)

    def _validate_axis_target(self, patch, **kwargs):
        return self.fixture.validate(
            patch,
            tests=self.AXIS_TARGET_PATH,
            expected_route="axis_kernel:run_kernel",
            shape_env=None,
            shape_arg="--shapes",
            perf=False,
            **kwargs,
        )

    def test_a_grid_that_duplicates_the_targets_own_defaults_is_not_a_control(self):
        # SKILL.md calls the S1 grid "a positive control against reporting the same default
        # test run twice under different stage names". On aiter#4538 all three requested
        # shapes were already in the target's own --shapes default list, so the stage
        # reported `pass` for re-running a strict subset of correctness_repo_tests, and the
        # verdict was PASS. A duplicate grid proves nothing the repository run did not
        # already prove, so it cannot be credited.
        patch = self._axis_patch("grid-duplicate.patch")
        result, report = self._validate_axis_target(patch, grid_value="7,257,f32")

        selection = report["test_selection"]
        self.assertEqual("duplicates-target-defaults", selection["grid_independence"])
        self.assertIn("--shapes", selection["grid_independence_reason"])
        grid_stage = report["stages"]["correctness_s1_grid"]
        self.assertEqual("skip", grid_stage["status"])
        self.assertEqual(0, grid_stage["exit"])
        self.assertEqual("INCONCLUSIVE", report["verdict"])
        self.assertEqual(2, result.returncode)

    def test_a_duplicate_grid_rescued_by_an_axis_says_so_in_one_place(self):
        # A duplicate shape grid is rescued when a PROVEN axis asks for values the target
        # does not run by default -- the configuration reaching the kernel is genuinely new.
        # But test_selection carries the same two fields and is written before the axes are
        # proven, so it kept the pre-override answer and the report contradicted itself:
        # test_selection said "duplicates-target-defaults" while the stage said
        # "adds-coverage". Observed on ROCm/aiter#5081. One question, one answer.
        patch = self._axis_patch("duplicate-rescued.patch")
        _, report = self._validate_axis_target(
            patch,
            grid_value="7,257,f32",
            axes=("num_heads=--num-heads:32;64",),
        )

        stage = report["stages"]["correctness_s1_grid"]
        selection = report["test_selection"]
        self.assertEqual("proven", selection["axis_state"])
        self.assertEqual("adds-coverage", stage["independence"])
        self.assertEqual(
            stage["independence"], selection["grid_independence"]
        )
        self.assertEqual(
            stage["independence_reason"], selection["grid_independence_reason"]
        )
        self.assertEqual("pass", stage["status"])

    def test_a_grid_outside_the_targets_defaults_still_counts_as_coverage(self):
        # The control case for the test above: the fix must not turn every grid into a skip.
        patch = self._axis_patch("grid-novel.patch")
        _, report = self._validate_axis_target(patch, grid_value="9,1023,f32")

        self.assertEqual(
            "adds-coverage", report["test_selection"]["grid_independence"]
        )
        self.assertEqual("pass", report["stages"]["correctness_s1_grid"]["status"])

    def test_each_head_run_keeps_its_own_execution_receipt(self):
        # head-repo and head-grid both ran inside the head phase and shared one receipt
        # path, so the second run erased the first. With the grid shapes a subset of the
        # target's defaults -- aiter#4538's case -- a receipt written by EITHER run satisfies
        # --grid, which makes the grid's own evidence unfalsifiable.
        patch = self._axis_patch("receipt-split.patch")
        _, report = self._validate_axis_target(patch, grid_value="9,1023,f32")

        work = Path(report["stages"]["correctness_repo_tests"]["log"]).parent
        repo_receipt = work / "head" / "execution-receipt-head-repo.json"
        grid_receipt = work / "head" / "execution-receipt-head-grid.json"
        self.assertTrue(repo_receipt.exists(), f"missing {repo_receipt}")
        self.assertTrue(grid_receipt.exists(), f"missing {grid_receipt}")

        repo_shapes = set(json.loads(repo_receipt.read_text())["executed_shapes"])
        grid_shapes = set(json.loads(grid_receipt.read_text())["executed_shapes"])
        # The repo run executes the target's defaults; the grid run executes the grid.
        # Neither may contain the other's shapes, which is only checkable once they are
        # separate files.
        self.assertIn("7,257,f32", repo_shapes)
        self.assertNotIn("9,1023,f32", repo_shapes)
        self.assertEqual({"9,1023,f32"}, grid_shapes)
        self.assertIn("head-grid", report["stages"]["execution_receipt"]["receipt_scope"])

    def test_an_extra_axis_reaches_a_configuration_the_shape_grid_cannot_express(self):
        # The shape channel is one ordered tuple bound to --shape-vars, so on aiter#4538 it
        # could only ever vary (seq_len, seq_len_kv). num_heads is a separate flag whose
        # default is [64, 128], and the kernel asserts at num_heads=16 -- a real blocker the
        # validator had no way to request. --axis is that way.
        patch = self._axis_patch("axis-blocker.patch")
        result, report = self._validate_axis_target(
            patch,
            grid_value="9,1023,f32",
            axes=("num_heads=--num-heads:16;32",),
        )

        selection = report["test_selection"]
        self.assertEqual("proven", selection["axis_state"])
        axis = selection["axes"][0]
        self.assertEqual("num_heads", axis["name"])
        self.assertEqual("flag-declared-in-add_argument", axis["hook_proof"])
        self.assertEqual(["16", "32"], axis["values"])
        self.assertEqual("adds-coverage", axis["independence"])
        # The configuration the grid alone could never request now fails, loudly, and is
        # attributed to the PR that adds the target.
        self.assertEqual("fail", report["stages"]["correctness_s1_grid"]["status"])
        self.assertEqual(1, result.returncode)
        self.assertEqual("BLOCK", report["verdict"])
        self.assertTrue(
            any(
                item["severity"] == "blocker" and "shape grid" in item["detail"]
                for item in report["findings"]
            ),
            report["findings"],
        )

    def test_an_axis_the_target_ignores_is_named_not_silently_dropped(self):
        # A flag the target declares but does not constrain would let the report claim
        # coverage of head counts that never reached the kernel. The runtime refusal probe
        # is what separates "declared" from "consumed", exactly as for --shape-arg, and a
        # dropped axis has to be visible or the test space narrowed silently.
        permissive = self.AXIS_TARGET.replace(
            "parser.add_argument('--num-heads', type=int, nargs='*',\n"
            "                        default=[64, 128])\n",
            "parser.add_argument('--num-heads', type=str, nargs='*',\n"
            "                        default=['64', '128'])\n",
        ).replace(
            "            run_kernel(M, N, dtype_str, num_heads)\n",
            "            run_kernel(M, N, dtype_str, 64)\n",
        )
        self.assertNotEqual(permissive, self.AXIS_TARGET)
        patch = self._axis_patch("axis-ignored.patch", body=permissive)
        _, report = self._validate_axis_target(
            patch,
            grid_value="9,1023,f32",
            axes=("num_heads=--num-heads:16;32",),
        )

        selection = report["test_selection"]
        self.assertEqual("hook-not-consumed", selection["axis_state"])
        self.assertIn("--num-heads", selection["axis_state_reason"])
        self.assertTrue(
            any(
                "requested test axes were dropped" in item["detail"]
                for item in report["findings"]
            ),
            report["findings"],
        )

    def test_a_script_that_returns_without_working_earns_no_architecture_credit(self):
        # aiter#4538's target returns with exit 0 and a log line when the arch is
        # unsupported or an optional package is missing. That produced
        # `arch_coverage: {gfx: runtime}` on basis `script-exit-zero-with-output` and
        # `executed: 1` -- indistinguishable from the run that graded 56 cases.
        silent = (
            "import argparse\n"
            "\n"
            "\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('-s', '--shapes', nargs='*', default=['7,257,f32'])\n"
            "    parser.parse_args()\n"
            "    print('unsupported arch; skipping')\n"
            "    return\n"
            "\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        patch = self._axis_patch("silent-skip.patch", body=silent)
        result, report = self.fixture.validate(
            patch,
            tests=self.AXIS_TARGET_PATH,
            expected_route="axis_kernel:run_kernel",
            shape_env=None,
            shape_arg="--shapes",
            perf=False,
            grid=False,
        )

        stats = report["stages"]["correctness_repo_tests"]["stats"]
        self.assertEqual(0, stats["observed_work"])
        self.assertIn("execution receipt", stats["basis"])
        self.assertEqual({}, report["arch_coverage"])
        self.assertEqual("INCONCLUSIVE", report["verdict"])
        self.assertEqual(2, result.returncode)

    #: A target the patch ADDS, which times a kernel that already exists on base against an
    #: unchanged reference. aiter#4538 is exactly this shape: a new op_tests target whose
    #: whole point is that the rewritten kernel is faster than what it replaces.
    NEW_BENCH_TARGET = "tests/bench_added.py"

    def _new_bench_patch(self, name, scale):
        # `VALUE` already exists on base (the fixture commits `VALUE = 1`), so the
        # transplanted target imports cleanly there and times the PRE-patch cost. That is
        # the whole premise: a new target file driving an entry point that is not new.
        def mutate(repo):
            (repo / "aiter" / "kernel.py").write_text(f"VALUE = {scale}\n")
            (repo / self.NEW_BENCH_TARGET).write_text(
                "import argparse\n"
                "import os\n"
                "import sys\n"
                "\n"
                "sys.path.insert(0, os.path.dirname(os.path.dirname(\n"
                "    os.path.abspath(__file__))))\n"
                "\n"
                "from aiter.kernel import VALUE\n"
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
                "        print(f'| {dim} | {dim * VALUE / 100.0} | {dim / 50.0} |')\n"
                "    print('4/4 cases passed')\n"
                "\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            )

        return self.fixture.make_patch(mutate, name)

    def test_an_added_target_can_still_time_the_code_it_replaces(self):
        # "The PR adds this target, so base has nothing to time against" ended the perf
        # stage on aiter#4538 -- a PR whose entire motivation is being faster than the kernel
        # it replaces. The file being new does not make the code it drives new: copying the
        # target into the base tree times the pre-PR implementation through the same harness.
        patch = self._new_bench_patch("added-bench-faster.patch", scale="0.5")
        _, report = self.fixture.validate(
            patch,
            tests=self.NEW_BENCH_TARGET,
            expected_route="aiter.kernel:main",
            grid=False,
            perf=True,
            perf_control_column="reference us",
        )

        perf = report["stages"]["perf"]
        self.assertEqual("target-transplant", perf["baseline_method"])
        self.assertEqual("pass", perf["status"])
        # The unchanged reference column has to reproduce across the two trees, or the
        # cross-tree comparison means nothing.
        self.assertIn("reproduced within", perf["control_note"])
        self.assertAlmostEqual(1.0, perf["control_ratio"], places=6)
        # base VALUE = 1, head VALUE = 0.5 -> head is twice as fast on the kernel column,
        # while the reference column sits at exactly 1.0. median_ratio is the WORST column
        # by design, so the improvement is read off the kernel column itself.
        kernel_column = next(
            stats
            for name, stats in perf["columns"].items()
            if "kernel" in name.lower()
        )
        self.assertGreater(kernel_column["median_ratio"], 1.5)
        self.assertGreaterEqual(perf["median_ratio"], 0.95)
        self.assertEqual(
            "target-not-present",
            report["stages"]["baseline_control"]["repo_tests"]["state"],
        )

    def test_an_added_target_regression_is_caught_not_skipped(self):
        # The direction that matters: a NEW target whose kernel got slower must still fail.
        patch = self._new_bench_patch("added-bench-slower.patch", scale="2.0")
        result, report = self.fixture.validate(
            patch,
            tests=self.NEW_BENCH_TARGET,
            expected_route="aiter.kernel:main",
            grid=False,
            perf=True,
            perf_control_column="reference us",
        )

        perf = report["stages"]["perf"]
        self.assertEqual("target-transplant", perf["baseline_method"])
        self.assertEqual("fail", perf["status"])
        self.assertLess(perf["median_ratio"], 0.95)
        self.assertEqual(1, result.returncode)
        self.assertTrue(
            any(
                item["stage"] == "perf" and item["severity"] == "should-fix"
                for item in report["findings"]
            ),
            report["findings"],
        )

    def test_a_transplanted_baseline_whose_control_moved_is_not_believed(self):
        # The gate that makes a cross-tree comparison reportable at all. If a column the
        # patch does not touch moves between the two trees, the two runs are not comparable
        # and the kernel ratio means nothing - the stage must skip and say so rather than
        # publish a confident number drawn from two different machines-in-effect.
        def mutate(repo):
            (repo / "aiter" / "kernel.py").write_text("VALUE = 0.5\nREFERENCE = 3.0\n")
            (repo / self.NEW_BENCH_TARGET).write_text(
                "import argparse\n"
                "import os\n"
                "import sys\n"
                "\n"
                "sys.path.insert(0, os.path.dirname(os.path.dirname(\n"
                "    os.path.abspath(__file__))))\n"
                "\n"
                "try:\n"
                "    from aiter.kernel import REFERENCE\n"
                "except ImportError:\n"
                # On base REFERENCE does not exist, so the transplanted target falls back to
                # a different reference cost -- which is exactly the situation the control
                # column exists to detect.
                "    REFERENCE = 1.0\n"
                "from aiter.kernel import VALUE\n"
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
                "        print(f'| {dim} | {dim * VALUE / 100.0} "
                "| {dim * REFERENCE / 50.0} |')\n"
                "    print('4/4 cases passed')\n"
                "\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            )

        patch = self.fixture.make_patch(mutate, "added-bench-control-moved.patch")
        _, report = self.fixture.validate(
            patch,
            tests=self.NEW_BENCH_TARGET,
            expected_route="aiter.kernel:main",
            grid=False,
            perf=True,
            perf_control_column="reference us",
        )

        perf = report["stages"]["perf"]
        self.assertEqual("target-transplant", perf["baseline_method"])
        self.assertEqual("skip", perf["status"])
        self.assertIn("not comparable", perf["control_note"])
        self.assertNotIn("median_ratio", perf)

    def test_a_transplanted_baseline_without_a_control_column_reports_why(self):
        # The transplant is a cross-tree comparison. With nothing unchanged to check it
        # against, the honest outcome is a skip whose reason names the missing evidence --
        # not a confident ratio, and not the old claim that base had nothing to time.
        patch = self._new_bench_patch("added-bench-nocontrol.patch", scale="0.5")
        _, report = self.fixture.validate(
            patch,
            tests=self.NEW_BENCH_TARGET,
            expected_route="aiter.kernel:main",
            grid=False,
            perf=True,
        )

        perf = report["stages"]["perf"]
        self.assertEqual("skip", perf["status"])
        self.assertIn("--perf-control-column", perf["note"])
        self.assertNotIn("nothing to time against", perf["note"])

    #: A pytest-named file that ALSO parses argv in its module body. Found on
    #: ROCm/aiter#5172: pytest wins the runner selection, imports the module at collection
    #: with its own argv, and argparse exits the process. The file is green as a script.
    ARGV_AT_IMPORT_TARGET = (
        "import argparse\n"
        "import os\n"
        "import sys\n"
        "\n"
        "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
        "\n"
        "from axis_kernel import run_kernel\n"
        "\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('-s', '--shapes', nargs='*', default=['7,257,f32'])\n"
        "parser.add_argument('--num-heads', type=int, nargs='*', default=[64, 128])\n"
        "args = parser.parse_args()\n"
        "\n"
        "\n"
        "def test_top_level():\n"
        "    run_kernel(7, 257, 'f32', 64)\n"
    )

    #: aiter's dominant op_tests convention: a SCRIPT whose worker happens to be named
    #: `test_<op>(...)` and is called from main() with real arguments. pytest collects it,
    #: cannot supply the parameters, and errors. Observed on ROCm/aiter#5081, where the
    #: validator published "the PR adds this test target and it fails on head" against the
    #: author for a target that is green as a script with the very shapes the run requested.
    UNCOLLECTABLE_WORKER_TARGET = (
        "import argparse\n"
        "import os\n"
        "import sys\n"
        "\n"
        "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
        "\n"
        "from axis_kernel import run_kernel\n"
        "\n"
        "\n"
        "def test_worker(M, N, dtype_str, num_heads):\n"
        "    run_kernel(M, N, dtype_str, num_heads)\n"
        "    return M * N\n"
        "\n"
        "\n"
        "def _shape(text):\n"
        "    M, N, dtype_str = text.split(',')\n"
        "    return int(M), int(N), dtype_str\n"
        "\n"
        "\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('-s', '--shapes', type=_shape, nargs='*',\n"
        "                        default=[(7, 257, 'f32'), (8, 64, 'f32')])\n"
        "    parser.add_argument('--num-heads', type=int, nargs='*',\n"
        "                        default=[64, 128])\n"
        "    args = parser.parse_args()\n"
        "    for M, N, dtype_str in args.shapes:\n"
        "        for num_heads in args.num_heads:\n"
        "            test_worker(M, N, dtype_str, num_heads)\n"
        "    print('%d cases passed' % (len(args.shapes) * len(args.num_heads)))\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    def test_a_test_named_worker_pytest_cannot_collect_runs_as_a_script(self):
        patch = self._axis_patch(
            "uncollectable-worker.patch", body=self.UNCOLLECTABLE_WORKER_TARGET
        )
        result, report = self._validate_axis_target(
            patch,
            grid_value="9,1023,f32",
            axes=("num_heads=--num-heads:32;64",),
        )

        selection = report["test_selection"]
        # "Defines a test* function" is not the same as "pytest can collect it".
        self.assertEqual("script", selection["runner"])
        self.assertIn("pytest cannot collect them", selection["runner_reason"])
        # And because it is a script, the shape grid and the axis both reach it.
        self.assertEqual("cli", selection["grid_channel"])
        self.assertEqual("proven", selection["axis_state"])
        self.assertEqual("pass", report["stages"]["correctness_s1_grid"]["status"])
        self.assertGreater(
            report["stages"]["correctness_repo_tests"]["stats"]["observed_work"], 0
        )
        # Above all: no blocker charged to an author whose target works.
        self.assertFalse(
            any(item["severity"] == "blocker" for item in report["findings"]),
            report["findings"],
        )
        self.assertNotEqual(1, result.returncode)

    def test_a_route_behind_a_wraps_decorator_is_still_observed(self):
        # A route is written the way a caller reads the source, `module:function`. The
        # profiler sees frames, and a frame's identity is
        # f_globals["__name__"] + ":" + f_code.co_name -- not the same thing once the
        # function is wrapped, because functools.wraps copies __name__ onto the wrapper
        # object and leaves co_name alone. aiter's entire @compile_ops family therefore
        # executes as `aiter.jit.core:wrapper`, so naming the op a reviewer cares about
        # matched nothing and the receipt reported an empty route. Observed on
        # ROCm/aiter#5081.
        wrapped_kernel = (
            "import functools\n"
            "\n"
            "\n"
            "def _dispatch(func):\n"
            "    @functools.wraps(func)\n"
            "    def wrapper(*args, **kwargs):\n"
            "        return func(*args, **kwargs)\n"
            "\n"
            "    return wrapper\n"
            "\n"
            "\n"
            "@_dispatch\n"
            "def run_kernel(M, N, dtype_str, num_heads):\n"
            "    assert num_heads % 32 == 0, 'heads must be a multiple of 32'\n"
            "    return M * N\n"
        )

        def mutate(repo):
            (repo / "tests" / "axis_kernel.py").write_text(wrapped_kernel)
            (repo / self.AXIS_TARGET_PATH).write_text(self.AXIS_TARGET)

        patch = self.fixture.make_patch(mutate, "wrapped-route.patch")
        _, report = self._validate_axis_target(patch, grid_value="9,1023,f32")

        receipt = report["stages"]["execution_receipt"]
        self.assertEqual("pass", receipt["status"])
        self.assertEqual("axis_kernel:run_kernel", receipt["route"])
        self.assertEqual(["9,1023,f32"], sorted(set(receipt["executed_shapes"])))
        self.assertGreater(
            report["stages"]["correctness_repo_tests"]["stats"]["observed_work"], 0
        )

    def test_the_caller_can_force_a_runner_the_classifier_got_wrong(self):
        patch = self._axis_patch(
            "forced-runner.patch", body=self.UNCOLLECTABLE_WORKER_TARGET
        )
        _, report = self.fixture.validate(
            patch,
            tests=self.AXIS_TARGET_PATH,
            expected_route="axis_kernel:run_kernel",
            shape_env=None,
            shape_arg="--shapes",
            perf=False,
            grid=False,
            runner="pytest",
        )

        selection = report["test_selection"]
        self.assertEqual("pytest", selection["runner"])
        self.assertIn("caller forced --runner pytest", selection["runner_reason"])
        self.assertIn("structural selection said script", selection["runner_reason"])

    def test_a_runner_that_cannot_run_the_target_is_named_as_such(self):
        # "Red on both sides" is an attribution, not an explanation. When the target carries
        # a structural reason the SELECTED runner cannot run it, a reader who is not told so
        # concludes the code is broken when the runner choice is.
        patch = self._axis_patch("argv-at-import.patch", body=self.ARGV_AT_IMPORT_TARGET)
        _, report = self.fixture.validate(
            patch,
            tests=self.AXIS_TARGET_PATH,
            expected_route="axis_kernel:run_kernel",
            shape_env=None,
            shape_arg="--shapes",
            grid_value="9,1023,f32",
            axes=("num_heads=--num-heads:16;32",),
            perf=False,
        )

        selection = report["test_selection"]
        self.assertEqual("pytest", selection["runner"])
        self.assertIn("parses argv in its module body", selection["runner_risk"])
        self.assertTrue(
            any(
                "under the selected pytest runner" in item["detail"]
                for item in report["findings"]
            ),
            report["findings"],
        )

        # A requested axis that could not be honoured must still appear. Publishing an empty
        # `axes` beside a non-`none` axis_state loses the request itself, which is exactly
        # the silently narrowed test space these fields exist to make visible.
        self.assertEqual("unusable", selection["axis_state"])
        self.assertEqual(1, len(selection["axes"]))
        self.assertEqual("num_heads", selection["axes"][0]["name"])
        self.assertEqual(["16", "32"], selection["axes"][0]["values"])
        self.assertEqual("not-evaluated", selection["axes"][0]["hook_proof"])

        # And the grid-independence reason must describe THIS run. The old default claimed
        # "the channel exposes no declared defaults to compare against" whenever the
        # comparison did not happen - a statement about the target that this run never
        # established, and false here: the target declares a default for --shapes.
        self.assertEqual("unknown", selection["grid_independence"])
        self.assertNotIn(
            "no declared defaults", selection["grid_independence_reason"]
        )
        # The channel this run established, named -- rather than a claim about the target.
        self.assertIn(
            "independence is only computed for the CLI-flag channel",
            selection["grid_independence_reason"],
        )

    def test_a_killed_run_leaves_no_stale_verdict_at_the_output_path(self):
        # The process exit code used to be read back out of `--out` AFTER finish_report, so
        # `--out` was a fallback source of truth. A run that died before finish_report copied
        # its own report over the file left the PREVIOUS run's report sitting there, and the
        # caller -- `review-pr`'s identity gate included -- read a stale `PASS` as this PR's
        # result. A run owns its output path.
        patch = self._axis_patch("stale-verdict.patch")
        report_path = self.fixture.root / "stale-report.json"
        report_path.write_text(
            json.dumps(
                {
                    "label": "an-earlier-run",
                    "verdict": "PASS",
                    "process_exit_code": 0,
                    "stages": {},
                    "findings": [],
                }
            )
        )
        slow_picker = self.fixture.tools / "slow-picker"
        write_executable(slow_picker, "#!/usr/bin/env bash\nsleep 30\nprintf '97\\n'\n")

        environ = os.environ.copy()
        environ["PICKER"] = str(slow_picker)
        environ["PYTHONPATH"] = str(self.fixture.fake_modules)
        environ["TIMEOUT"] = "30"
        process = subprocess.Popen(
            [
                str(VALIDATOR),
                "--repo",
                str(self.fixture.repo),
                "--patch",
                str(patch),
                "--target",
                self.AXIS_TARGET_PATH,
                "--expected-route",
                "axis_kernel:run_kernel",
                "--no-perf",
                "--label",
                "stale-verdict",
                "--out",
                str(report_path),
            ],
            env=environ,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        process.kill()
        process.wait(timeout=30)

        self.assertFalse(
            report_path.exists(),
            "a killed run left a previous run's report at --out: "
            + (report_path.read_text() if report_path.exists() else ""),
        )

    def test_the_target_cannot_read_the_reviewers_credentials(self):
        # `env VAR=... cmd` ADDS to the inherited environment. The target is arbitrary code
        # from an unmerged PR, so every token in the reviewer's shell was readable from
        # os.environ inside it -- and lands in a stage log the moment the target prints its
        # environment, which is a perfectly ordinary thing for a test to do on failure.
        leaky = (
            "import os\n"
            "\n"
            "\n"
            "def main():\n"
            "    for name, value in sorted(os.environ.items()):\n"
            "        print(f'{name}={value}')\n"
            "\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        patch = self._axis_patch("env-leak.patch", body=leaky)
        environ = os.environ.copy()
        environ["VALIDATOR_TEST_GITHUB_TOKEN"] = "leakcanary-token"
        environ["MY_API_KEY"] = "leakcanary-key"
        environ["UNRELATED_HOME_DECOR"] = "leakcanary-unrelated"

        report_path = self.fixture.root / "env-leak-report.json"
        command = [
            str(VALIDATOR),
            "--repo",
            str(self.fixture.repo),
            "--patch",
            str(patch),
            "--target",
            self.AXIS_TARGET_PATH,
            "--expected-route",
            "axis_kernel:run_kernel",
            "--shape-vars",
            "M,N,dtype_str",
            "--no-perf",
            "--label",
            "env-leak",
            "--out",
            str(report_path),
        ]
        environ["PICKER"] = str(self.fixture.picker)
        environ["PYTHONPATH"] = str(self.fixture.fake_modules)
        environ["TIMEOUT"] = "30"
        run(command, env=environ, check=False)

        report = json.loads(report_path.read_text())
        log = Path(report["stages"]["correctness_repo_tests"]["log"]).read_text()
        for canary in ("leakcanary-token", "leakcanary-key", "leakcanary-unrelated"):
            self.assertNotIn(canary, log)
        for name in ("VALIDATOR_TEST_GITHUB_TOKEN", "MY_API_KEY", "UNRELATED_HOME_DECOR"):
            self.assertNotIn(name, log)
        # The policy is a reported fact, not an implicit one.
        policy = report["isolation"]["target_environment"]
        self.assertIn("env -i", policy["policy"])
        self.assertIn("PATH", policy["passed_through"])

    def test_an_unlabeled_measurement_column_does_not_destroy_row_identity(self):
        # aiter bench tables print columns whose headers carry no unit -- `flydsl rel`,
        # `triton err`, `speedup`. They are measurements, but `classify()` calls anything
        # without a unit a KEY column, so they became part of the row's identity. They
        # differ between base and head by construction, so every row got a unique name on
        # each side and `matched_rows` was 0: `insufficient` on 56 perfectly comparable
        # rows. Observed on aiter#4538's real bench logs. The strict key is still tried
        # first; only when it matches nothing is the relaxed key used, and the result says
        # which was used.
        table = (
            "| s_q | s_k | kernel us | rel err | speedup |\n"
            "|---|---|---|---|---|\n"
            "| 1024 | 1024 | {a} | {r1} | {s1} |\n"
            "| 2048 | 2048 | {b} | {r2} | {s2} |\n"
            "| 4096 | 4096 | {c} | {r3} | {s3} |\n"
        )
        base_log = self.fixture.root / "rowkey-base.log"
        head_log = self.fixture.root / "rowkey-head.log"
        base_log.write_text(
            table.format(
                a=10.0, b=20.0, c=40.0,
                r1=1.11e-5, r2=2.22e-5, r3=3.33e-5,
                s1=1.0101, s2=1.0202, s3=1.0303,
            )
        )
        head_log.write_text(
            table.format(
                a=5.0, b=10.0, c=20.0,
                r1=1.19e-5, r2=2.28e-5, r3=3.37e-5,
                s1=2.0404, s2=2.0505, s3=2.0606,
            )
        )

        result = json.loads(
            run(
                [
                    sys.executable,
                    str(SKILL_DIR / "scrape_perf.py"),
                    "--base",
                    str(base_log),
                    "--head",
                    str(head_log),
                ]
            ).stdout
        )

        self.assertEqual(3, result["matched_rows"])
        self.assertIn("non-integral", result["row_key_basis"])
        # The shape columns are integral, so they stay in the key and keep the three rows
        # distinct -- the relaxed key must not collapse them.
        self.assertEqual(3, result["base_rows"])
        self.assertEqual("ok", result["status"])
        self.assertAlmostEqual(2.0, result["columns"]["kernel us"]["median_ratio"], 3)


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


class GridChannelTests(unittest.TestCase):
    """The S1 grid needs a delivery channel; there are three, probed independently."""

    def setUp(self):
        self.fixture = ValidatorFixture()

    def tearDown(self):
        self.fixture.close()

    @staticmethod
    def add_cli_shape_script(repo):
        """A script target whose shapes arrive on its own CLI flag, and which
        never reads an environment variable."""
        (repo / "tests" / "run_shapes.py").write_text(
            "import argparse\n"
            "\n"
            "def run_kernel(M, N, dtype_str):\n"
            "    assert M > 0 and N > 0 and dtype_str\n"
            "\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            '    parser.add_argument("--shape", nargs="*", default=["7,257,f32"])\n'
            "    args = parser.parse_args()\n"
            "    for shape in args.shape:\n"
            "        M, N, dtype_str = shape.split(',')\n"
            "        run_kernel(int(M), int(N), dtype_str)\n"
            "    print(f'{len(args.shape)}/{len(args.shape)} shapes passed')\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )

    @staticmethod
    def add_parametrized_target(repo):
        """A pytest target whose shapes are literals inside its own parametrize mark --
        the dominant shape in the real repository, and the case neither of the older two
        channels can reach."""
        (repo / "tests" / "test_parametrized.py").write_text(
            "import pytest\n"
            "\n"
            "def run_kernel(M, N, dtype_str):\n"
            "    assert M > 0 and N > 0 and dtype_str\n"
            "\n"
            '@pytest.mark.parametrize("M,N,dtype_str", [(3, 5, "f32")])\n'
            "def test_shapes(M, N, dtype_str):\n"
            "    run_kernel(M, N, dtype_str)\n"
        )

    @staticmethod
    def add_single_name_parametrized_target(repo):
        """One shape parameter -- the dominant shape in the targets this channel exists for.

        `run_kernel` requires an int so that the invalid-grid probe fails INSIDE the target:
        the sentinel arrives as a string and the assertion is what rejects it.
        """
        (repo / "tests" / "test_one_name.py").write_text(
            "import pytest\n"
            "\n"
            "def run_kernel(m):\n"
            "    assert isinstance(m, int) and m > 0\n"
            "\n"
            '@pytest.mark.parametrize("m", [3])\n'
            "def test_one_shape(m):\n"
            "    run_kernel(m)\n"
        )

    @staticmethod
    def add_target_with_an_unrelated_parametrize(repo):
        """Two tests in one file: one the grid replaces, one it must not touch.

        `test_unrelated` binds `m` together with `other`, so the grid cannot be substituted
        into it without leaving `other` unfilled. Its assertion on its OWN values is what
        proves the plugin left it alone.
        """
        (repo / "tests" / "test_two_marks.py").write_text(
            "import pytest\n"
            "\n"
            "def run_kernel(m, n):\n"
            "    assert isinstance(m, int) and isinstance(n, int)\n"
            "    assert m > 0 and n > 0\n"
            "\n"
            '@pytest.mark.parametrize("m,n", [(3, 5)])\n'
            "def test_shapes(m, n):\n"
            "    run_kernel(m, n)\n"
            "\n"
            '@pytest.mark.parametrize("m,other", [(11, "keep")])\n'
            "def test_unrelated(m, other):\n"
            "    assert (m, other) == (11, 'keep')\n"
        )

    @staticmethod
    def add_target_that_rejects_the_grid_after_the_route_ran(repo):
        """The repository run reaches the route; the grid run dies before it.

        The guard is on the TEST, ahead of the call, so the grid phase produces a receipt
        that observed nothing while the repository phase produced one that proved the route.
        """
        (repo / "tests" / "test_late_grid.py").write_text(
            "import pytest\n"
            "\n"
            "def run_kernel(m):\n"
            "    assert m > 0\n"
            "\n"
            '@pytest.mark.parametrize("m", [3])\n'
            "def test_shape(m):\n"
            '    assert m < 100, "shape unsupported by this target"\n'
            "    run_kernel(m)\n"
        )

    def test_single_shape_argname_is_delivered_and_not_published_as_a_defect(self):
        """The most consequential regression of the batch.

        The plugin unwrapped one-name rows to scalars but passed `argnames` as a LIST, and
        pytest sets force_tuple only for a `str` argnames. Collection died with "object of
        type 'int' has no len()", the grid run exited non-zero, and the executor published
        that crash as `[blocker] the PR adds this target and its independent shape grid
        fails` -- a BLOCK verdict against three real authors for a fault in the injector.
        """
        patch = self.fixture.make_patch(
            self.add_single_name_parametrized_target, "one-name.patch"
        )

        _, report = self.fixture.validate(
            patch,
            tests="tests/test_one_name.py",
            expected_route="test_one_name:run_kernel",
            shape_env=None,
            shape_argnames="m",
            shape_vars="m",
            grid_value="128;256",
        )

        self.assertEqual("pytest", report["test_selection"]["grid_channel"])
        self.assertNotEqual("BLOCK", report["verdict"])
        self.assertEqual(
            [], [item for item in report["findings"] if item["severity"] == "blocker"]
        )
        grid_stage = report["stages"]["correctness_s1_grid"]
        self.assertEqual("pass", grid_stage["status"])
        # Both grid rows collected and ran -- not one collection error counted as a test.
        self.assertEqual(2, grid_stage["stats"]["executed"])
        # And the injected values, not the target's own literal 3, are what reached the route.
        receipt = report["stages"]["execution_receipt"]
        self.assertEqual("pass", receipt["status"])
        self.assertEqual(["128", "256"], sorted(receipt["executed_shapes"]))

    def test_an_unrelated_parametrize_does_not_disable_the_channel(self):
        """The partial-overlap guard belongs to a test function, not to a file.

        Evaluated file-wide, `test_unrelated`'s `(m, other)` mark -- which overlaps the
        requested names without being contained in them -- switched the channel off for every
        test in the file, and the skip text then blamed the target for taking parameters it
        demonstrably takes. The plugin has always decided per metafunc; only the executor's
        reachability probe was file-scoped.
        """
        patch = self.fixture.make_patch(
            self.add_target_with_an_unrelated_parametrize, "two-marks.patch"
        )

        _, report = self.fixture.validate(
            patch,
            tests="tests/test_two_marks.py",
            expected_route="test_two_marks:run_kernel",
            shape_env=None,
            shape_argnames="m,n",
            shape_vars="m,n",
            grid_value="128,7;256,9",
        )

        self.assertEqual("pytest", report["test_selection"]["grid_channel"])
        self.assertEqual("", report["test_selection"]["grid_channel_reason"])
        grid_stage = report["stages"]["correctness_s1_grid"]
        self.assertEqual("pass", grid_stage["status"])
        # Two grid rows for test_shapes plus the one unrelated case, which kept its own
        # parametrization: it asserts (11, 'keep') and would have failed had the grid been
        # substituted into it.
        self.assertEqual(3, grid_stage["stats"]["executed"])
        receipt = report["stages"]["execution_receipt"]
        self.assertEqual(["128,7", "256,9"], sorted(receipt["executed_shapes"]))

    def test_a_failed_grid_run_does_not_erase_the_repository_run_receipt(self):
        """Receipts are per label, because evidence already collected must not be deleted.

        `head-repo` and `head-grid` shared `$WORK/head/execution-receipt.json`. The grid run
        starts by removing that path, so a grid phase that observed nothing overwrote a
        receipt that had already proved the route, and the report then said the route never
        executed -- an erasure reported as an absence.
        """
        patch = self.fixture.make_patch(
            self.add_target_that_rejects_the_grid_after_the_route_ran,
            "receipt-erasure.patch",
        )

        _, report = self.fixture.validate(
            patch,
            tests="tests/test_late_grid.py",
            expected_route="test_late_grid:run_kernel",
            shape_env=None,
            shape_argnames="m",
            shape_vars="m",
            grid_value="128;256",
        )

        # The grid phase really did fail; that is the premise, not the thing under test.
        self.assertEqual("fail", report["stages"]["correctness_s1_grid"]["status"])
        receipt = report["stages"]["execution_receipt"]
        self.assertEqual("pass", receipt["status"])
        self.assertEqual("test_late_grid:run_kernel", receipt["route"])
        # The receipt that speaks is the repository run's, which observed the route.
        self.assertEqual(["3"], receipt["executed_shapes"])

    def test_invalid_grid_probe_needs_a_passing_control_before_it_proves_anything(self):
        """A non-zero probe exit is evidence about the GRID only if the target works without it.

        On a held-out PR whose module could not be imported, the invalid-grid probe failed for
        that reason and the channel was credited although no shape ever reached the kernel.
        The break is planted on BASE, so the base control run is red before the grid is ever
        involved.
        """
        path = self.fixture.repo / "tests" / "test_sample.py"
        path.write_text(
            "raise ImportError('the module under test cannot be imported')\n"
            + path.read_text()
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
                "broken base",
            ],
            cwd=self.fixture.repo,
        )
        patch = self.fixture.make_patch(
            ValidateKernelPrTests.harmless_change, "broken-control.patch"
        )

        _, report = self.fixture.validate(patch)

        baseline_grid = report["stages"]["baseline_control"]["s1_grid"]
        self.assertEqual("hook-not-consumed", baseline_grid["state"])
        # No base grid run was attempted, so there is no exit code to report for one.
        self.assertNotIn("exit", baseline_grid)
        self.assertNotEqual("pass", report["stages"]["correctness_s1_grid"]["status"])

    def test_working_cli_channel_survives_a_second_shape_flag(self):
        """Supplying --shape-arg AND --shape-env must not discard the CLI channel.

        The two probes describe one target that may have both hooks. An earlier version
        assigned the env probe's result over the CLI probe's unconditionally, so a
        caller who named both flags lost a working CLI channel and was then told the env
        variable's absence was the reason no grid ran.
        """
        patch = self.fixture.make_patch(self.add_cli_shape_script, "cli-channel.patch")

        _, report = self.fixture.validate(
            patch,
            tests="tests/run_shapes.py",
            expected_route="__main__:run_kernel",
            shape_env="UNREAD_GRID_ENV",
            shape_arg="--shape",
            # Deliberately NOT the target's own `--shape` default of 7,257,f32. This test is
            # about which channel carries the grid, but a grid that only re-runs the
            # target's defaults is now downgraded to `skip` on independence grounds, and
            # that would mask the channel result this test exists to check.
            grid_value="9,1023,f32",
        )

        self.assertEqual("cli", report["test_selection"]["grid_channel"])
        self.assertEqual("", report["test_selection"]["grid_channel_reason"])
        self.assertEqual(
            "adds-coverage", report["test_selection"]["grid_independence"]
        )
        grid_stage = report["stages"]["correctness_s1_grid"]
        self.assertNotEqual("skip", grid_stage["status"])
        self.assertEqual("pass", grid_stage["status"])

    def test_parametrized_target_runs_the_grid_through_the_pytest_channel(self):
        """The third channel: pytest's own parametrization.

        The target exposes neither a flag nor an environment variable, so before this
        channel existed the stage was inert and the skip text blamed the kernel for a
        limit that belonged to the injector.
        """
        patch = self.fixture.make_patch(
            self.add_parametrized_target, "pytest-channel.patch"
        )

        _, report = self.fixture.validate(
            patch,
            tests="tests/test_parametrized.py",
            expected_route="test_parametrized:run_kernel",
            shape_env=None,
            shape_argnames="M,N,dtype_str",
            grid_value="7,257,f32;8,513,bf16",
        )

        self.assertEqual("pytest", report["test_selection"]["grid_channel"])
        grid_stage = report["stages"]["correctness_s1_grid"]
        self.assertNotEqual("skip", grid_stage["status"])
        self.assertEqual("pass", grid_stage["status"])
        self.assertEqual(2, grid_stage["stats"]["executed"])
        # The grid actually reached the kernel: the receipt carries the injected shapes,
        # not the (3, 5, "f32") literal the target parametrizes for itself.
        receipt = report["stages"]["execution_receipt"]
        self.assertEqual("pass", receipt["status"])
        self.assertEqual(
            ["7,257,f32", "8,513,bf16"], sorted(receipt["executed_shapes"])
        )
        # The invalid-grid probe must fail INSIDE THE TARGET. A poisoned row of the
        # wrong arity would raise in the plugin instead, and its non-zero exit would
        # credit the channel without the target ever having consumed a shape.
        probe_log = Path(grid_stage["hook_probe_log"]).read_text()
        self.assertNotIn("grid rows must have", probe_log)
        self.assertIn("__VALIDATOR_INVALID_GRID__", probe_log)

    def test_undeliverable_grid_names_the_channel_and_blames_the_right_party(self):
        """A skip has to say which channel was tried and what was found there.

        --shape-arg against a pytest target is a gap in the validator's own wiring, and
        publishing it as a property of the target would send a reviewer to fix a kernel
        that is not broken.
        """
        patch = self.fixture.make_patch(
            ValidateKernelPrTests.harmless_change, "no-channel.patch"
        )

        _, report = self.fixture.validate(
            patch,
            shape_env="UNREAD_GRID_ENV",
            shape_arg="--shape",
        )

        self.assertEqual("", report["test_selection"]["grid_channel"])
        reason = report["test_selection"]["grid_channel_reason"]
        self.assertIn("a validator limit, not a target property", reason)
        self.assertIn("--shape", reason)
        self.assertIn("does not read $UNREAD_GRID_ENV", reason)
        self.assertEqual("skip", report["stages"]["correctness_s1_grid"]["status"])


class ExecutionReceiptTests(unittest.TestCase):
    """The receipt must not report `pass` beside evidence it never collected."""

    def setUp(self):
        self.fixture = ValidatorFixture()

    def tearDown(self):
        self.fixture.close()

    @staticmethod
    def derive_shapes_in_the_body(repo):
        """Make the route take one opaque argument and unpack the shape inside itself --
        the common case for a kernel whose arguments are tensors."""
        path = repo / "tests" / "test_sample.py"
        source = path.read_text()
        source = source.replace(
            "def run_kernel(M, N, dtype_str):\n"
            "    assert M > 0 and N > 0 and dtype_str\n",
            "def run_kernel(spec):\n"
            "    M, N, dtype_str = spec.split(',')\n"
            "    M = int(M)\n"
            "    N = int(N)\n"
            "    assert M > 0 and N > 0 and dtype_str\n",
        )
        source = source.replace(
            "        M, N, dtype_str = shape.split(',')\n"
            "        run_kernel(int(M), int(N), dtype_str)\n",
            "        run_kernel(shape)\n",
        )
        path.write_text(source)

    def test_shapes_derived_in_the_route_body_are_captured(self):
        """Sampling f_locals only on the `call` event sees parameters and nothing else.

        A route that derives its shapes in its body therefore produced an empty
        executed_shapes while the receipt still reported pass -- an absence published as
        a confirmation. The probe now samples again on `return`, when body locals are
        bound.
        """
        patch = self.fixture.make_patch(
            self.derive_shapes_in_the_body, "body-shapes.patch"
        )

        result, report = self.fixture.validate(patch)

        receipt = report["stages"]["execution_receipt"]
        self.assertEqual("pass", receipt["status"])
        self.assertEqual(["7,257,f32"], receipt["executed_shapes"])
        self.assertNotIn("shape_capture", receipt)
        self.assertEqual(0, result.returncode)
        self.assertEqual("PASS", report["verdict"])


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


class EvidenceCheckerTests(unittest.TestCase):
    """Unit-level checks on validate_evidence.py, where the report's numbers come from."""

    RECEIPT = {
        "schema_version": 1,
        "producer": "validate-kernel-pr.validation_probe",
        "route": "test_route:run_kernel",
        "kernel_symbols": ["test_route:run_kernel"],
        "executed_shapes": ["3,5"],
        "shape_vars": ["m", "n"],
        "pytest_exitstatus": 0,
    }

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.directory = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def evidence(self, *arguments):
        return json.loads(
            run(
                [sys.executable, str(SKILL_DIR / "validate_evidence.py"), *arguments]
            ).stdout
        )

    def test_junit_errors_are_not_counted_as_executed_tests(self):
        """An error is a collection or fixture failure: the test body never ran.

        Counting it as executed let a target that could not even be imported credit
        `arch_coverage` with RUNTIME coverage -- on the strength of an "executed test" that
        was the collection error itself.
        """
        junit = self.directory / "junit.xml"
        junit.write_text(
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<testsuites><testsuite name="pytest" errors="1" failures="0" '
            'skipped="0" tests="1" time="0.01" /></testsuites>\n'
        )

        stats = self.evidence("pytest-stats", str(junit))

        self.assertEqual(1, stats["tests"])
        self.assertEqual(1, stats["errors"])
        self.assertEqual(0, stats["executed"])

    def test_pytest_channel_receipt_does_not_assert_across_namespaces(self):
        """The grid is delivered as test PARAMETERS; the receipt records the ROUTE's locals.

        Requiring one to contain the other produced "execution receipt is missing required
        shapes" on a run whose every grid case passed. The requirement is still recorded and
        the mismatch is named; only the cross-namespace containment assertion is dropped.
        """
        path = self.directory / "receipt.json"
        path.write_text(json.dumps(self.RECEIPT))

        result = self.evidence(
            "receipt",
            str(path),
            "--expected-route",
            "test_route:run_kernel",
            "--grid",
            "128,7;256,9",
            "--grid-channel",
            "pytest",
        )

        self.assertEqual("pass", result["status"])
        # The grid's requirement is not discarded, only its containment assertion.
        self.assertEqual(["128,7", "256,9"], result["required_shapes"])
        self.assertEqual(["3,5"], result["executed_shapes"])
        self.assertIn("different vocabularies", result["shape_namespace"])

    def test_a_channel_that_shares_the_receipt_namespace_still_must_contain_the_grid(self):
        """The control for the test above: without the pytest channel, nothing is relaxed.

        The env and CLI channels put the grid into the same vocabulary the receipt records,
        so a missing shape there is still a real gap in coverage.
        """
        path = self.directory / "receipt.json"
        path.write_text(json.dumps(self.RECEIPT))

        result = self.evidence(
            "receipt",
            str(path),
            "--expected-route",
            "test_route:run_kernel",
            "--grid",
            "128,7;256,9",
            "--grid-channel",
            "env",
        )

        self.assertEqual("skip", result["status"])
        self.assertIn("missing required shapes", result["note"])

    def test_empty_native_artifact_list_states_what_it_does_not_mean(self):
        """`native_artifacts: []` was published as though it were a measurement.

        The probe runs BEFORE the target and imports only the named module, so an empty list
        describes that import and nothing else. Three real runs recorded it on a host where
        the runtime demonstrably executed the kernel.
        """
        fixture = ValidatorFixture()
        try:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(fixture.repo)
            identity = json.loads(
                run(
                    [
                        sys.executable,
                        str(SKILL_DIR / "validate_evidence.py"),
                        "runtime",
                        "aiter",
                        str(fixture.repo),
                    ],
                    env=environment,
                ).stdout
            )
        finally:
            fixture.close()

        self.assertEqual([], identity["native_artifacts"])
        basis = identity["native_artifacts_basis"]
        self.assertTrue(basis)
        self.assertIn("aiter", basis)
        self.assertIn("not evidence", basis)


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

        self.assertEqual("57", result.stdout.strip())


class GpuClaimTests(unittest.TestCase):
    """Claiming a device is evidence, so how it is claimed has to be evidence too."""

    def setUp(self):
        self.fixture = ValidatorFixture()

    def tearDown(self):
        self.fixture.close()

    def test_contended_lock_falls_through_to_the_next_idle_gpu(self):
        """The picker is deterministic, so concurrent validators pick the same device.

        Locking only the first choice meant all but one reported NO_GPU while the rest
        of the machine sat idle. The claim now walks the whole eligible ranking.
        """
        picker = self.fixture.tools / "ranking-picker"
        # HIP 61 ranks first and is unavailable; 57 is the index the fake amdsmi can map.
        # Both are deliberately fake indices, so this never contends with a real device.
        write_executable(
            picker,
            "#!/usr/bin/env bash\n"
            "printf 'idleness-basis: activity+vram\\n' >&2\n"
            'for arg in "$@"; do\n'
            '  if [ "$arg" = "--all" ]; then printf \'61\\n57\\n\'; exit 0; fi\n'
            "done\n"
            "printf '61\\n'\n",
        )
        patch = self.fixture.make_patch(
            ValidateKernelPrTests.harmless_change, "lock-race.patch"
        )

        # A real flock on the real path the validator locks, held for the whole run. Taken
        # in this process rather than a helper: the validator's lock is on its own file
        # description, so it contends exactly as another validator's would, and there is no
        # child left holding /tmp/gpu-61.lock if this test is interrupted.
        holder = open("/tmp/gpu-61.lock", "w")
        try:
            fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            _, report = self.fixture.validate(patch, picker=picker)
        finally:
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
            holder.close()

        claim = report["stages"]["gpu_claim"]
        self.assertEqual("pass", claim["status"])
        self.assertEqual(57, claim["hip_index"])
        self.assertNotEqual("NO_GPU", report["degraded_mode"])

    def test_shipped_picker_wins_over_one_earlier_on_path(self):
        """The shipped picker is part of the evidence contract: it is the thing
        that prints `idleness-basis:`.

        Resolving PATH first silently substituted a foreign picker that omits that line,
        and the report then published `idleness_basis: "unknown"` beside a concrete
        `gfx_activity_before_pct: 0` -- an unavailable reading presented as a measured
        idle one. An explicit PICKER still wins; this is only the default.
        """
        foreign = self.fixture.root / "foreign-bin"
        foreign.mkdir()
        write_executable(
            foreign / "pick-idle-gpu.py",
            "#!/usr/bin/env bash\nprintf '3\\n'\n",
        )
        patch = self.fixture.make_patch(
            ValidateKernelPrTests.harmless_change, "picker-order.patch"
        )

        _, report = self.fixture.validate(
            patch,
            path_prefix=foreign,
            use_picker_env=False,
        )

        claim = report["stages"]["gpu_claim"]
        self.assertEqual("pass", claim["status"])
        self.assertEqual(57, claim["hip_index"])
        self.assertEqual("activity+vram", claim["idleness_basis"])


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


class ShapeGridPluginTests(unittest.TestCase):
    """The plugin is what substitutes the grid, so its refusals are what keep a target the
    grid cannot express from being reported as a failing PR."""

    def _run_target(self, argnames, grid, target_source):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plugin = root / "sgp.py"
            plugin.write_text(
                (SKILL_DIR / "shape_grid_plugin.py").read_text()
                + f"\n_VALIDATION_SHAPE_ARGNAMES = {argnames!r}\n"
                + f"_VALIDATION_SHAPE_GRID = {grid!r}\n"
            )
            target = root / "test_target.py"
            target.write_text(target_source)
            return subprocess.run(
                [sys.executable, "-m", "pytest", "-p", "sgp", str(target), "-q"],
                cwd=root, capture_output=True, text=True,
            )

    def test_dict_valued_parametrize_is_refused_rather_than_poisoned(self):
        # A target parametrizing one `case: dict` passed the argnames-only gate, the grid
        # substituted integers, and the target raised TypeError -- which the executor
        # published as "the PR adds this target and its independent shape grid fails", a
        # BLOCK against an author whose own suite was green in the same report.
        result = self._run_target(
            ("case",),
            [(1,), (513,)],
            "import pytest\n"
            '@pytest.mark.parametrize("case", [{"m": 1}, {"m": 3}])\n'
            "def test_case(case):\n"
            "    assert case['m'] > 0\n",
        )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn("2 passed", result.stdout)
        self.assertNotIn("TypeError", result.stdout + result.stderr)

    def test_grid_arity_mismatch_is_rejected_at_invocation(self):
        # The arity check lived in the plugin generator, whose exit status run_pytest never
        # read: the stale plugin from the previous phase survived, head-grid re-ran the
        # invalid-grid sentinel, and its failure was published as "the PR adds this target and
        # its independent shape grid fails" -- a blocker produced by a caller's typo. It is
        # now refused at argument parsing, before any phase can run.
        fixture = ValidatorFixture()
        try:
            result = subprocess.run(
                [
                    str(VALIDATOR),
                    "--repo", str(fixture.repo),
                    "--target", "tests/test_sample.py",
                    "--shape-argnames", "m",
                    "--grid", "255,3;512,4",
                ],
                capture_output=True, text=True,
            )
        finally:
            fixture.close()
        self.assertEqual(2, result.returncode, result.stdout + result.stderr)
        self.assertIn("--grid", result.stderr)

    def test_a_single_scalar_argname_is_substituted(self):
        # The same code path with scalar values must still replace the target's own literals,
        # or the refusal above would have been bought by disabling the channel.
        result = self._run_target(
            ("m",),
            [(3,), (15,), (32,)],
            "import pytest\n"
            '@pytest.mark.parametrize("m", [1, 2])\n'
            "def test_m(m):\n"
            "    assert m in (3, 15, 32)\n",
        )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn("3 passed", result.stdout)


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


if __name__ == "__main__":
    unittest.main()
