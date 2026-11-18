#!/usr/bin/env python3
"""Selftest of flydsl_aot_ci.py's log parsing and pass/fail contract.

Standard library only, no GPU, no aiter import: the FlyDSL AOT workflow runs
it on the runner before the build."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "flydsl_aot_ci", HERE / "flydsl_aot_ci.py"
)
ci = importlib.util.module_from_spec(spec)
sys.modules["flydsl_aot_ci"] = ci
spec.loader.exec_module(ci)

GEMM_OK = """\
[aiter] FlyDSL GEMM replay: 10 jobs, archs gfx1250
[aiter] FlyDSL AOT: 4 kernels, 4 worker processes
[aiter] FlyDSL GEMM replay gfx1250: 10 ok, 0 failed, 0 refused by the runtime
  [SKIP] 2 x RuntimeError: no FlyDSL config for M=N
  [SCAN] 5 sizes inside rows' ranges, 0 failed
"""

MOE_KNOWN = """\
[aiter] FlyDSL MoE replay: 3 rows, 12 jobs, archs gfx1250
  [FAIL] grouped_gemm1_fp4_gugu | grouped_gemm2_fp4  token=8192 m=4097 dims=(7168,3072) E=96 topk=6 gate=interleave  arch=gfx1250: AssertionError:
Traceback (most recent call last):
AssertionError
  [FAIL] grouped_gemm1_fp4_gugu | grouped_gemm2_fp4  token=8192 m=8192 dims=(7168,3072) E=96 topk=6 gate=interleave  arch=gfx1250: AssertionError:
[aiter] FlyDSL MoE replay gfx1250: 10 ok, 2 failed, 0 unsupported caller-side variants
  [BISECT] 4 token counts inside buckets, 0 failed
"""

KNOWN = (
    "# comment\n\n"
    r"moe 2 ^grouped_gemm1_fp4_gugu \| grouped_gemm2_fp4 .* dims=\(7168,3072\) E=96 .*AssertionError:$"
    "\n"
)


def judged(module, text, returncode, known_text=KNOWN):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "known.txt"
        path.write_text(known_text)
        known = ci.load_known_failures(path)
    result = ci.PassResult(module, "compile", returncode, 1.0, "log")
    ci.parse_log(result, text)
    ci.judge(result, known)
    return result


class ParseAndJudge(unittest.TestCase):
    def test_clean_module(self):
        result = judged("gemm", GEMM_OK, 0)
        self.assertEqual(
            (result.status, result.jobs, result.ok, result.failed), ("ok", 10, 10, 0)
        )

    def test_known_failures_only(self):
        result = judged("moe", MOE_KNOWN, 1)
        self.assertEqual(result.status, "known", result.problems)
        self.assertEqual(
            (result.jobs, result.ok, result.failed, len(result.fails)), (12, 10, 2, 2)
        )
        self.assertTrue(result.fails[0].endswith("AssertionError:"))

    def test_known_failure_scoped_to_its_module(self):
        result = judged("gemm", MOE_KNOWN, 1)
        self.assertEqual(result.status, "failed")
        self.assertEqual(sum("new failure" in p for p in result.problems), 2)

    def test_fixed_known_failure_is_red(self):
        clean = MOE_KNOWN.replace("2 failed", "0 failed").replace("10 ok", "12 ok")
        clean = "\n".join(line for line in clean.splitlines() if "[FAIL]" not in line)
        result = judged("moe", clean, 0)
        self.assertEqual(result.status, "failed")
        self.assertIn("matched 0 [FAIL] lines, expected 2", result.problems[0])

    def test_new_failure_is_red(self):
        text = MOE_KNOWN.replace("2 failed", "3 failed") + (
            "  [FAIL] moe_stage1_g1u1 token=1 m=1 dims=(7168,256) arch=gfx1250: "
            "ValueError: boom\n"
        )
        result = judged("moe", text, 1)
        self.assertEqual(result.status, "failed")
        self.assertTrue(
            any(p.startswith("new failure: moe_stage1_g1u1") for p in result.problems)
        )

    def test_dead_worker_is_red(self):
        result = judged("moe", MOE_KNOWN.replace("2 failed", "5 failed"), 1)
        self.assertEqual(result.status, "failed")
        self.assertTrue(any("worker died" in p for p in result.problems))

    def test_probe_failures_count(self):
        text = MOE_KNOWN.replace("2 failed, 0 unsupported", "1 failed, 0 unsupported")
        text = text.replace("buckets, 0 failed", "buckets, 1 failed")
        self.assertEqual(judged("moe", text, 1).status, "known")

    def test_crash_without_fail_lines_is_red(self):
        result = judged(
            "conv", "Traceback (most recent call last):\nImportError: x\n", 1
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("exited 1 without a [FAIL] line", result.problems)

    def test_fail_lines_with_exit_zero_are_red(self):
        text = "[aiter] FlyDSL CONV replay gfx1250: 1 ok, 1 failed, 0 refused\n  [FAIL] x\n"
        self.assertEqual(judged("conv", text, 0).status, "failed")

    def test_fail_lines_of_two_workers_on_one_line(self):
        first, second = (
            line for line in MOE_KNOWN.splitlines() if line.startswith("  [FAIL]")
        )
        merged = MOE_KNOWN.replace(first + "\n", "").replace(second, first + second)
        result = judged("moe", merged, 1)
        self.assertEqual(result.status, "known", result.problems)
        self.assertEqual(len(result.fails), 2)
        self.assertTrue(all(text.endswith("AssertionError:") for text in result.fails))

    def test_mega_moe_summary(self):
        result = judged(
            "mega_moe", "[aiter] FlyDSL MegaMoE cache check: 5 ok, 0 failed\n", 0
        )
        self.assertEqual((result.status, result.ok), ("ok", 5))


class KnownFailuresFile(unittest.TestCase):
    def test_rejects_malformed_lines(self):
        for bad in ("moe ^x$\n", "moe 0 ^x$\n", "moe two ^x$\n", "moe 1 (\n"):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "known.txt"
                path.write_text(bad)
                with self.assertRaises(ValueError, msg=bad):
                    ci.load_known_failures(path)

    def test_repository_lists_parse(self):
        modules = set(ci.default_modules())
        lists = sorted(
            (ci.REPO_ROOT / ".github" / "flydsl-aot").glob("*-known-failures.txt")
        )
        self.assertTrue(lists)
        for path in lists:
            for entry in ci.load_known_failures(path):
                self.assertIn(entry.module, modules, f"{path}:{entry.line}")


class Modules(unittest.TestCase):
    def test_default_modules_follow_run_aot(self):
        modules = ci.default_modules()
        self.assertEqual(modules[:2], ["moe", "mega_moe"])
        for name in ("gemm", "conv", "pa_decode_plan"):
            self.assertIn(name, modules)
        for name in modules:
            self.assertTrue((ci.AOT_DIR / f"{name}.py").is_file(), name)

    def test_markdown_renders(self):
        result = judged("moe", MOE_KNOWN, 1)
        summary = {
            "arch": "gfx1250",
            "passed": True,
            "seconds": 3.0,
            "cache": {
                "artifacts": 2,
                "chips": {"gfx1250": 2},
                "launchers": {"launch": 2},
            },
            "passes": [
                {
                    k: getattr(result, k)
                    for k in (
                        "module",
                        "phase",
                        "status",
                        "jobs",
                        "ok",
                        "failed",
                        "seconds",
                        "fails",
                    )
                }
            ],
            "problems": [],
        }
        markdown = ci.build_markdown(summary)
        self.assertIn("| moe | compile | 12 | 10 | 2 |", markdown)
        self.assertIn("2 known failures", markdown)


if __name__ == "__main__":
    unittest.main(verbosity=2)
