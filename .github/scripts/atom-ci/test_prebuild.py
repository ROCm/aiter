import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from prebuild import build


class PrebuildTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.target_dir = self.root / "aiter" / "jit"
        self.target_dir.mkdir(parents=True)
        self.target = self.target_dir / "module_gemm.so"
        self.target.write_bytes(b"test")
        self.args = {
            "srcs": ["kernel.cpp"],
            "flags_extra_cc": [],
            "flags_extra_hip": ["-O3"],
            "blob_gen_cmd": "",
            "extra_include": [],
            "extra_ldflags": None,
            "verbose": False,
            "is_python_module": True,
            "is_standalone": False,
            "torch_exclude": False,
        }
        self.core = Mock()
        self.core.get_args_of_build.return_value = self.args
        self.core.get_user_jit_dir.return_value = str(self.target_dir)
        self.core._so_offload_archs.return_value = {"gfx950"}

    def test_existing_build_arguments_are_preserved(self):
        build(self.core, self.root, "module_gemm")
        kwargs = self.core.build_module.call_args.kwargs
        for key, value in self.args.items():
            self.assertEqual(kwargs[key], value)
        self.assertEqual(kwargs["md_name"], "module_gemm")

    def test_missing_binary_fails(self):
        self.target.unlink()
        with self.assertRaisesRegex(RuntimeError, "did not create"):
            build(self.core, self.root, "module_gemm")

    def test_wrong_cache_directory_fails(self):
        self.core.get_user_jit_dir.return_value = str(self.root / "other-cache")
        with self.assertRaisesRegex(RuntimeError, "did not create"):
            build(self.core, self.root, "module_gemm")

    def test_wrong_architecture_fails(self):
        self.core._so_offload_archs.return_value = {"gfx942"}
        with self.assertRaisesRegex(RuntimeError, "architectures"):
            build(self.core, self.root, "module_gemm")

    def test_missing_architecture_fails(self):
        self.core._so_offload_archs.return_value = set()
        with self.assertRaisesRegex(RuntimeError, "architectures"):
            build(self.core, self.root, "module_gemm")

    def test_build_failure_is_not_ignored(self):
        self.core.build_module.side_effect = RuntimeError("compiler failed")
        with self.assertRaisesRegex(RuntimeError, "compiler failed"):
            build(self.core, self.root, "module_gemm")

    def run_collector_without_container(self):
        tools = self.root / "bin"
        tools.mkdir()
        (tools / "docker").symlink_to("/bin/false")
        return subprocess.run(
            ["bash", str(Path(__file__).with_name("collect.sh"))],
            cwd=self.root,
            env={**os.environ, "PATH": f"{tools}:/usr/bin:/bin"},
            capture_output=True,
            check=False,
            timeout=10,
        )

    def test_collection_tolerates_missing_container_and_results(self):
        result = self.run_collector_without_container()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "atom_diagnostics").is_dir())

    def test_collection_keeps_available_results_when_container_is_gone(self):
        results = self.root / "accuracy_test_results"
        results.mkdir()
        (results / "result.json").write_text("{}")
        (self.root / "atom_accuracy_output.txt").write_text("baseline failed")
        result = self.run_collector_without_container()
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.root / "atom_diagnostics"
        self.assertEqual(
            (output / "accuracy_test_results/result.json").read_text(), "{}"
        )
        self.assertEqual(
            (output / "atom_accuracy_output.txt").read_text(), "baseline failed"
        )


if __name__ == "__main__":
    unittest.main()
