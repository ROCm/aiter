"""Deterministic checks for the strict pytest manifest runner (no GPU needed)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("run_pytest_manifest.py")

TESTS = {
    "test_good.py": """
        import pytest

        def test_one():
            pass

        @pytest.mark.parametrize("n", [1, 2, 3])
        def test_param(n):
            assert n

        class TestGroup:
            def test_member(self):
                pass
    """,
    "test_bad.py": """
        import os
        import time
        import pytest

        def test_fails():
            assert False

        def test_skips():
            pytest.skip("not on this GPU")

        @pytest.mark.xfail(reason="known")
        def test_xfails():
            assert False

        def test_crashes():
            os._exit(3)

        def test_hangs():
            time.sleep(600)
    """,
    "test_other.py": """
        def test_fine():
            pass
    """,
}


class RunnerScenarios(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name, body in TESTS.items():
            (self.root / name).write_text(textwrap.dedent(body))
        self.runs = 0

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_manifest(self, lines: str, *extra: str) -> tuple[int, dict, str]:
        self.runs += 1
        manifest = self.root / f"manifest{self.runs}.txt"
        manifest.write_text(textwrap.dedent(lines))
        artifacts = self.root / f"artifacts{self.runs}"
        process = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--manifest",
                manifest.name,
                "--artifact-dir",
                artifacts.name,
                *extra,
            ],
            cwd=self.root,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        summary_path = artifacts / "contract-summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        return process.returncode, summary, process.stdout + process.stderr

    def assertProblem(self, summary: dict, text: str) -> None:
        self.assertTrue(
            any(text in problem for problem in summary["problems"]),
            f"no problem containing {text!r} in {summary['problems']}",
        )

    def test_clean_manifest_passes(self) -> None:
        rc, summary, output = self.run_manifest(
            """
            # comment
            test_good.py::test_one 1
            test_good.py::test_param 3
            test_good.py::TestGroup 1
            test_other.py
            """,
        )
        self.assertEqual(rc, 0, output)
        self.assertEqual(summary["collected"], 6)
        self.assertEqual(summary["outcomes"], {"passed": 6})
        self.assertEqual(
            [g["group"] for g in summary["groups"]], ["00-test_good", "01-test_other"]
        )

    def test_expected_count_must_match_manifest(self) -> None:
        rc, _, output = self.run_manifest(
            "test_good.py::test_param 3\n", "--expected-count", "4"
        )
        self.assertEqual(rc, 2, output)

    def test_count_drift_fails(self) -> None:
        rc, summary, _ = self.run_manifest("test_good.py::test_param 2\n")
        self.assertEqual(rc, 1)
        self.assertProblem(summary, "expected 2 nodes, collected 3")

    def test_parametrized_node_id(self) -> None:
        rc, summary, output = self.run_manifest("test_good.py::test_param[2]\n")
        self.assertEqual(rc, 0, output)
        self.assertEqual(summary["collected"], 1)

    def test_invalid_manifests(self) -> None:
        for lines in (
            "missing.py::test_x 1\n",
            "test_good.py::test_one 0\n",
            "test_good.py::test_one 1\ntest_good.py::test_one 1\n",
            "test_good.py::test_param[2] 2\n",
            "../test_good.py\n",
            "# nothing\n",
        ):
            rc, _, output = self.run_manifest(lines)
            self.assertEqual(rc, 2, f"{lines!r}: {output}")

    def test_unknown_selector_collects_nothing(self) -> None:
        rc, summary, _ = self.run_manifest(
            "test_good.py::test_one 1\ntest_good.py::test_gone 1\n"
        )
        self.assertEqual(rc, 1)
        self.assertProblem(summary, "collected no nodes: test_good.py::test_gone")

    def test_failure_skip_and_xfail_fail_the_contract(self) -> None:
        for test, text in (
            ("test_fails", "call:failed"),
            ("test_skips", "call:skipped"),
            ("test_xfails", "call:xfail"),
        ):
            rc, summary, _ = self.run_manifest(f"test_bad.py::{test} 1\n")
            self.assertEqual(rc, 1, test)
            self.assertProblem(summary, text)

    def test_crash_is_isolated_to_its_file(self) -> None:
        rc, summary, _ = self.run_manifest(
            "test_bad.py::test_crashes 1\ntest_other.py 1\n"
        )
        self.assertEqual(rc, 1)
        self.assertProblem(summary, "00-test_bad: pytest exited 3")
        self.assertEqual(summary["groups"][1]["problems"], 0)
        self.assertEqual(summary["outcomes"].get("passed"), 1)

    def test_timeout_kills_the_run(self) -> None:
        rc, summary, _ = self.run_manifest(
            "test_bad.py::test_hangs 1\n", "--timeout", "15"
        )
        self.assertEqual(rc, 1)
        self.assertProblem(summary, "pytest exited 124")

    def test_expect_failure_mode(self) -> None:
        rc, summary, output = self.run_manifest(
            "test_bad.py::test_fails 1\n", "--expect-failure"
        )
        self.assertEqual(rc, 0, output)
        self.assertEqual(summary["outcomes"], {"failed as filed": 1})
        for test, text in (
            ("test_good.py::test_one", "now passes"),
            ("test_bad.py::test_skips", "changed"),
        ):
            rc, summary, _ = self.run_manifest(f"{test} 1\n", "--expect-failure")
            self.assertEqual(rc, 1, test)
            self.assertProblem(summary, text)

    def test_collect_only(self) -> None:
        rc, summary, output = self.run_manifest(
            "test_bad.py::test_fails 1\ntest_good.py::test_param 3\n", "--collect-only"
        )
        self.assertEqual(rc, 0, output)
        self.assertEqual(summary["collected"], 4)
        self.assertEqual(summary["outcomes"], {})

    def test_require_gfx_without_aiter(self) -> None:
        rc, _, output = self.run_manifest("test_other.py\n", "--require-gfx", "gfx1250")
        self.assertEqual(rc, 2, output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
