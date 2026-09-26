"""CPU integration regressions for selected Triton test sharding."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "split_tests.sh"
TESTS = "op_tests/triton_tests/"


def find_bash():
    # Git Bash is usable directly from Windows Python; a WSL bash launcher
    # cannot use the same working directory and command-line paths reliably.
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            git_root = Path(git).resolve().parents[1]
            for relative in ("bin/bash.exe", "usr/bin/bash.exe"):
                candidate = git_root / relative
                if candidate.is_file():
                    return str(candidate)
    return shutil.which("bash")


BASH = find_bash()


@unittest.skipUnless(BASH, "Bash is required to exercise split_tests.sh")
class SplitTritonTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="triton-split-tests-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        # A Windows checkout may use CRLF; Bash must receive a Unix script.
        (self.root / "split_tests.sh").write_text(
            SCRIPT.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
        )
        self.selected = [
            TESTS + "gemm/basic/test_gemm_a16w16.py",
            TESTS + "fusions/test_fused_bmm_rope_kv_cache.py",
            TESTS + "test_gmm.py",
        ]
        self.unrelated = TESTS + "attention/test_unrelated.py"
        for relative in self.selected + [self.unrelated]:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Synthetic collection fixture.\n", encoding="utf-8")

    def run_splitter(self, selection):
        (self.root / "selected.list").write_bytes(selection)
        env = os.environ.copy()
        if os.name == "nt":
            # Prefer Git's find/sort/wc over Windows commands with those names.
            usr_bin = Path(BASH).resolve().parents[1] / "usr" / "bin"
            if usr_bin.is_dir():
                env["PATH"] = str(usr_bin) + os.pathsep + env.get("PATH", "")
        return subprocess.run(
            [
                BASH,
                "--noprofile",
                "--norc",
                "split_tests.sh",
                "--test-type",
                "triton",
                "--shards",
                "8",
                "--select-file",
                "selected.list",
            ],
            cwd=self.root,
            env=env,
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )

    def assert_partition(self, result, expected):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        shards = sorted(self.root.glob("triton_shard_*.list"))
        self.assertEqual(len(shards), 8)
        actual = [
            path
            for shard in shards
            for path in shard.read_text(encoding="utf-8").split()
        ]
        # Count equality catches both omissions and duplicate assignments.
        self.assertCountEqual(actual, expected)
        self.assertNotIn(self.unrelated, actual)

    def test_subset_is_partitioned_once_and_unrelated_tests_are_excluded(self):
        # A repeated selection entry must not duplicate test execution.
        contents = "\n".join(self.selected + [self.selected[0]]) + "\n"
        result = self.run_splitter(contents.encode("utf-8"))
        self.assert_partition(result, self.selected)

    def test_empty_selection_writes_eight_empty_shards(self):
        result = self.run_splitter(b"")
        self.assert_partition(result, [])
        for index in range(8):
            self.assertEqual(
                (self.root / f"triton_shard_{index}.list").read_bytes(), b""
            )

    def test_unknown_test_path_fails_instead_of_silently_dropping_it(self):
        missing = TESTS + "gemm/test_missing.py"
        contents = self.selected[0] + "\n" + missing + "\n"
        result = self.run_splitter(contents.encode("utf-8"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(missing, result.stderr)
        self.assertEqual(list(self.root.glob("triton_shard_*.list")), [])

    def test_selection_without_final_newline_keeps_last_path(self):
        result = self.run_splitter("\n".join(self.selected).encode("utf-8"))
        self.assert_partition(result, self.selected)

    def test_crlf_selection_is_accepted(self):
        contents = "\r\n".join(self.selected) + "\r\n"
        result = self.run_splitter(contents.encode("utf-8"))
        self.assert_partition(result, self.selected)


if __name__ == "__main__":
    unittest.main()
