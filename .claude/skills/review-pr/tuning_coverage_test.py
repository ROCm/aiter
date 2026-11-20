"""Exercise the review scanner against real Git trees, without importing aiter."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("tuning_coverage.py")


class TuningCoverageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Test")
        self.write("README", "baseline\n")
        self.commit("initial")

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.repo), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def write(self, name, content):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-qm", message)
        return self.git("rev-parse", "HEAD")

    def scan(self, base, head, success=True):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(self.repo), base, head],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        return result.stdout

    def test_only_new_full_keys_are_checked_against_head_input(self):
        untuned = "aiter/configs/bf16_untuned_gemm.csv"
        tuned = "aiter/configs/bf16_tuned_gemm.csv"
        self.write(untuned, "M,N,dtype,bias\n1,2,bf16,False\n")
        self.write(
            tuned,
            "gfx,M,N,dtype,bias,us\n"
            "gfx1,1,2,bf16,False,4\n"
            "gfx1,9,9,bf16,False,4\n",
        )
        base = self.commit("existing workloads")
        self.write(untuned, "M,N,dtype,bias\n1,2,bf16,False\n3,4,bf16,False\n")
        self.write(
            tuned,
            "gfx,M,N,dtype,bias,us\n"
            "gfx2,1,2,bf16,False,2\n"
            "gfx1,9,9,bf16,False,4\n"
            "gfx1,3,4,bf16,False,2\n"
            "gfx1,3,4,bf16,True,2\n",
        )
        head = self.commit("new workloads")
        self.write(untuned, "M,N,dtype,bias\n3,4,bf16,True\n")
        out = self.scan(base, head)
        self.assertIn("1 of 2 new keys absent", out)
        self.assertIn(f"{tuned}:5:", out)
        self.assertIn('"bias": "True"', out)
        self.assertNotIn('"M": "9"', out)

    def test_model_file_uses_canonical_family_and_full_schema(self):
        self.write(
            "aiter/configs/example_untuned.csv",
            "token,model_dim,activation\n8,1024,silu\n",
        )
        base = self.commit("canonical input")
        path = "aiter/configs/model_configs/variant_example_tuned.csv"
        self.write(
            path,
            "token,model_dim,activation,us\n8,1024,silu,1\n8,1024,gelu,2\n",
        )
        head = self.commit("model tuning")
        out = self.scan(base, head)
        self.assertIn("1 of 2 new keys absent", out)
        self.assertIn("aiter/configs/example_untuned.csv", out)
        self.assertIn('"activation": "gelu"', out)

    def test_architecture_is_compared_when_it_is_part_of_the_input(self):
        self.write("aiter/configs/example_untuned.csv", "gfx,M\ngfx1,16\n")
        base = self.commit("architecture-specific input")
        self.write(
            "aiter/configs/example_tuned.csv",
            "gfx,M,us\ngfx1,16,1\ngfx2,16,1\n",
        )
        head = self.commit("architecture-specific tuning")
        out = self.scan(base, head)
        self.assertIn("1 of 2 new keys absent", out)
        self.assertIn('"gfx": "gfx2"', out)

    def test_exact_input_takes_precedence_and_numeric_values_match(self):
        self.write("aiter/configs/example_untuned.csv", "M\n32\n")
        base = self.commit("canonical input")
        self.write(
            "aiter/configs/model_configs/variant_example_untuned.csv",
            "M,mode\n16.0,fast\n",
        )
        self.write(
            "aiter/configs/model_configs/variant_example_tuned.csv",
            "M,mode,us\n16,fast,1\n",
        )
        head = self.commit("paired input")
        out = self.scan(base, head)
        self.assertIn("COVERED", out)
        self.assertIn("1 new workload keys", out)
        self.assertIn("variant_example_untuned.csv", out)

    def test_unmatched_and_non_csv_configs_are_unverified(self):
        base = self.git("rev-parse", "HEAD")
        self.write("aiter/configs/unknown_tuned.csv", "M,us\n1,2\n")
        self.write("aiter/configs/unknown_untuned.csv", "M\n1\n")
        self.write("aiter/configs/no_input_tuned.csv", "M,us\n1,2\n")
        self.write("aiter/configs/unclassified.csv", "M,us\n1,2\n")
        self.write("aiter/ops/triton/configs/gfx950/op/DEFAULT.json", "{}\n")
        self.write("aiter/configs/other.yaml", "key: value\n")
        self.write("aiter/configs/other.toml", "key = 1\n")
        head = self.commit("unknown formats")
        out = self.scan(base, head)
        self.assertEqual(out.count("UNVERIFIED"), 5)
        self.assertIn("COVERED aiter/configs/unknown_tuned.csv", out)
        self.assertNotIn("UNVERIFIED aiter/configs/unknown_untuned.csv", out)

    def test_renamed_file_does_not_reintroduce_legacy_workloads(self):
        self.write("aiter/configs/example_untuned.csv", "M\n1\n")
        old = "aiter/configs/example_tuned.csv"
        self.write(old, "M,us\n2,3\n")
        base = self.commit("legacy gap")
        new = "aiter/configs/model_configs/variant_example_tuned.csv"
        (self.repo / new).parent.mkdir(parents=True)
        (self.repo / old).rename(self.repo / new)
        head = self.commit("move tuning file")
        out = self.scan(base, head)
        self.assertIn("COVERED", out)
        self.assertIn("0 new workload keys", out)
        self.assertNotIn("MISSING", out)

    def test_schema_or_csv_errors_are_not_reported_as_covered(self):
        self.write("aiter/configs/example_untuned.csv", "M,dtype\n1,bf16\n")
        base = self.commit("input")
        self.write("aiter/configs/example_tuned.csv", "M,us\n1,2\n")
        self.write("aiter/configs/second_untuned.csv", "M\n1,extra\n")
        self.write("aiter/configs/second_tuned.csv", "M,us\n1,2\n")
        head = self.commit("invalid inputs")
        out = self.scan(base, head)
        self.assertEqual(out.count("UNVERIFIED"), 2)
        self.assertIn("missing input columns", out)
        self.assertIn("row has 2 fields, expected 1", out)
        self.assertNotIn("COVERED", out)

    def test_new_input_columns_missing_at_base_are_unverified(self):
        self.write("aiter/configs/example_untuned.csv", "M\n1\n")
        self.write("aiter/configs/example_tuned.csv", "M,us\n1,2\n")
        base = self.commit("old schema")
        self.write("aiter/configs/example_untuned.csv", "M,dtype\n1,bf16\n")
        self.write("aiter/configs/example_tuned.csv", "M,dtype,us\n1,bf16,2\n")
        head = self.commit("new schema")
        out = self.scan(base, head)
        self.assertIn("UNVERIFIED", out)
        self.assertIn("at merge base: missing input columns", out)
        self.assertNotIn("MISSING", out)

    def test_unknown_commit_is_explicitly_skipped(self):
        out = self.scan("missing-commit", "HEAD", success=False)
        self.assertIn("SKIPPED:", out)

    def test_fetch_runs_scanner_and_degrades_on_failure(self):
        self.write("aiter/configs/example_untuned.csv", "M\n1\n")
        base = self.commit("input")
        self.write("aiter/configs/example_tuned.csv", "M,us\n2,3\n")
        head = self.commit("new workload")
        fetch = SCRIPT.with_name("fetch.sh").read_text(encoding="utf-8")
        start = fetch.index("# Compare the PR's new workload keys")
        end = fetch.index("# Day-old worktrees", start)
        evidence = self.repo / "evidence"
        evidence.mkdir()
        env = os.environ | {
            "SKILLS_ROOT": str(SCRIPT.parent.parent),
            "PROJECT_ROOT": str(self.repo),
            "WORK": str(evidence),
            "BASE_SHA": base,
            "HEAD_SHA": head,
        }
        snippet = "set -euo pipefail\n" + fetch[start:end]
        subprocess.run(
            ["bash", "-c", snippet], env=env, check=True, capture_output=True
        )
        output = (evidence / "tuning_coverage.txt").read_text(encoding="utf-8")
        self.assertIn("MISSING aiter/configs/example_tuned.csv", output)

        env["HEAD_SHA"] = "missing-commit"
        subprocess.run(
            ["bash", "-c", snippet], env=env, check=True, capture_output=True
        )
        output = (evidence / "tuning_coverage.txt").read_text(encoding="utf-8")
        self.assertIn("SKIPPED:", output)
        self.assertIn("review these files manually", output)


if __name__ == "__main__":
    unittest.main()
