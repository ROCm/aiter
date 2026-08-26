import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
VALIDATOR = SKILL_DIR / "validate_pr.sh"
SCANNER = SKILL_DIR / "scan_index_width.py"
REQUIRED_STAGES = {
    "merge_sim",
    "gpu_claim",
    "runtime_compat",
    "test_policy",
    "baseline_control",
    "correctness_repo_tests",
    "correctness_s1_grid",
    "index_width_scan",
}


def validate_report_contract(report):
    required = {
        "label",
        "started_utc",
        "finished_utc",
        "isolation",
        "arch_coverage",
        "degraded_mode",
        "repo",
        "test_selection",
        "stages",
        "findings",
        "verdict",
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
            "def test_sample():\n"
            "    atol = 1e-5\n"
            "    assert atol < 1\n"
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
            "def amdsmi_init(): pass\n"
            "def amdsmi_shut_down(): pass\n"
            "def amdsmi_get_processor_handles(): return ['gpu0']\n"
            "def amdsmi_get_gpu_enumeration_info(handle): return {'hip_id': 7}\n"
            "def amdsmi_get_gpu_asic_info(handle):\n"
            "    return {'market_name': 'Synthetic GPU', "
            "'target_graphics_version': 'gfx-test'}\n"
            "def amdsmi_get_gpu_activity(handle): return {'gfx_activity': 0}\n"
            "def amdsmi_get_gpu_device_bdf(handle): return '0000:00:00.0'\n"
        )
        self.picker = self.tools / "pick-idle-gpu.py"
        write_executable(self.picker, "#!/usr/bin/env bash\nprintf '7\\n'\n")

    def close(self):
        self.tempdir.cleanup()

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
            "--tests",
            tests,
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
                    "7,257,f32",
                ]
            )
        environment = os.environ.copy()
        environment["PICKER"] = str(picker or self.picker)
        environment["PYTHONPATH"] = str(self.fake_modules)
        environment["TIMEOUT"] = "30"
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

        _, report = self.fixture.validate(patch, picker=no_gpu_picker)

        self.assertEqual("INCONCLUSIVE", report["verdict"])
        self.assertEqual("skip", report["stages"]["gpu_claim"]["status"])
        self.assertEqual("NO_GPU", report["degraded_mode"])
        self.assertEqual({}, report["arch_coverage"])
        self.assertEqual("skip", report["stages"]["correctness_repo_tests"]["status"])
        self.assertEqual("skip", report["stages"]["correctness_s1_grid"]["status"])
        self.assert_complete_stage_objects(report)

    def test_runtime_probe_uses_aiter_checkout_and_full_run_can_pass(self):
        patch = self.fixture.make_patch(self.harmless_change, "repo-aware.patch")

        _, report = self.fixture.validate(patch)

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
            report["test_selection"]["pytest_target"],
        )

    def test_new_failing_test_is_not_mislabeled_preexisting(self):
        def add_failing_test(repo):
            (repo / "tests" / "test_new.py").write_text(
                "def test_new():\n" "    assert False, 'candidate failure'\n"
            )

        patch = self.fixture.make_patch(add_failing_test, "new-test.patch")
        _, report = self.fixture.validate(
            patch,
            tests="tests/test_new.py",
            grid=False,
        )

        self.assertEqual("BLOCK", report["verdict"])
        baseline = report["stages"]["baseline_control"]["repo_tests"]
        self.assertEqual("target-not-present", baseline["state"])
        details = [item["detail"] for item in report["findings"]]
        self.assertTrue(any("adds this test target" in detail for detail in details))
        self.assertFalse(any("pre-existing" in detail for detail in details))

    def test_test_only_tolerance_widening_blocks_without_gpu(self):
        def loosen_tolerance(repo):
            path = repo / "tests" / "test_sample.py"
            path.write_text(path.read_text().replace("1e-5", "1e-1"))

        patch = self.fixture.make_patch(loosen_tolerance, "tolerance.patch")
        no_gpu_picker = self.fixture.tools / "no-gpu-picker"
        write_executable(no_gpu_picker, "#!/usr/bin/env bash\nexit 1\n")

        _, report = self.fixture.validate(patch, picker=no_gpu_picker)

        self.assertEqual("BLOCK", report["verdict"])
        self.assertEqual("fail", report["stages"]["test_policy"]["status"])
        self.assertEqual([[1e-5, 1e-1]], report["stages"]["test_policy"]["loosened"])

    def test_unavailable_pytest_writes_stage_objects_not_strings(self):
        patch = self.fixture.make_patch(self.harmless_change, "no-pytest.patch")
        fake_bin = self.fixture.root / "fake-bin"
        fake_bin.mkdir()
        write_executable(fake_bin / "python", "#!/usr/bin/env bash\nexit 1\n")

        _, report = self.fixture.validate(patch, path_prefix=fake_bin)

        self.assertEqual("INCONCLUSIVE", report["verdict"])
        self.assertEqual("skip", report["stages"]["correctness_repo_tests"]["status"])
        self.assertEqual("skip", report["stages"]["correctness_s1_grid"]["status"])
        self.assertEqual({}, report["arch_coverage"])
        self.assert_complete_stage_objects(report)

    def test_flydsl_source_change_is_not_shadowed_by_pylib(self):
        shutil.rmtree(self.fixture.repo / "aiter")
        source = self.fixture.repo / "python" / "flydsl"
        source.mkdir(parents=True)
        (source / "__init__.py").write_text('__version__ = "test"\n')
        (source / "module.py").write_text("VALUE = 1\n")
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
                "flydsl base",
            ],
            cwd=self.fixture.repo,
        )
        runtime = self.fixture.root / "runtime" / "flydsl"
        runtime.mkdir(parents=True)
        (runtime / "__init__.py").write_text('__version__ = "test"\n')

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
        self.assertIn("would shadow", report["stages"]["runtime_compat"]["note"])

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
            path.write_text(
                path.read_text().replace(
                    'if _GRID == "__VALIDATOR_INVALID_GRID__":',
                    "if False and _GRID:",
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


class ReviewSkillContractTests(unittest.TestCase):
    def test_review_skill_is_advisory_and_has_no_dead_scanner_paths(self):
        review_skill = (SKILL_DIR.parent / "review-pr" / "SKILL.md").read_text()

        self.assertIn("advisory tier", review_skill)
        self.assertIn("required scanner is missing or not executable", review_skill)
        self.assertIn("Validation (deterministic)", review_skill)
        self.assertIn("baseRefOid", review_skill)
        self.assertIn("expected_verdict", review_skill)
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
