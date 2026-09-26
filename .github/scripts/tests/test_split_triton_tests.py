"""CPU integration regressions for selected Triton test sharding."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "split_tests.sh"
TESTS = "op_tests/triton_tests/"


class SplitTritonTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        shutil.copyfile(SCRIPT, self.root / "split_tests.sh")
        self.selected = [
            TESTS + "gemm/basic/test_gemm_a16w16.py",
            TESTS + "fusions/test_fused_bmm_rope_kv_cache.py",
            TESTS + "test_gmm.py",
        ]
        self.unrelated = TESTS + "attention/test_unrelated.py"
        for relative in self.selected + [self.unrelated]:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def run_splitter(self, selection):
        (self.root / "selected.list").write_bytes(selection)
        return subprocess.run(
            [
                "bash",
                "split_tests.sh",
                "--test-type",
                "triton",
                "--shards",
                "8",
                "--select-file",
                "selected.list",
            ],
            cwd=self.root,
            capture_output=True,
            check=False,
            text=True,
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
        self.assertCountEqual(actual, expected)

    def test_subset_is_partitioned_once_and_unrelated_tests_are_excluded(self):
        contents = "\n".join(self.selected + [self.selected[0]]) + "\n"
        result = self.run_splitter(contents.encode("utf-8"))
        self.assert_partition(result, self.selected)

    def test_empty_selection_writes_eight_empty_shards(self):
        result = self.run_splitter(b"")
        self.assert_partition(result, [])

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


if __name__ == "__main__":
    unittest.main()
