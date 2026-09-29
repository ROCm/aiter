import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from monitor import snapshot
from observe_build import pending_builds
from prebuild import build


class DiagnosticsTests(unittest.TestCase):
    def test_no_build(self):
        self.assertEqual(pending_builds("HANG detected"), set())

    def test_unfinished_build(self):
        self.assertEqual(pending_builds("start build [module_gemm]"), {"module_gemm"})

    def test_finished_build(self):
        self.assertEqual(
            pending_builds("start build [a]\n\x1b[32mfinish build [a], cost 236.5s"),
            set(),
        )

    def test_concurrent_builds(self):
        self.assertEqual(
            pending_builds("start build [a]\nstart build [b]\nfinish build [a]"), {"b"}
        )

    def test_rebuild(self):
        self.assertEqual(
            pending_builds("start build [a]\nfinish build [a]\nstart build [a]"), {"a"}
        )

    def test_snapshot_serializes(self):
        result = snapshot()
        json.dumps(result)
        self.assertGreater(result["time"], 0)
        self.assertIn("cpu.stat", result["cgroup"])

    def test_prebuild_uses_existing_build_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_dir = root / "aiter" / "jit"
            target_dir.mkdir(parents=True)
            (target_dir / "module_gemm.so").write_bytes(b"test")
            args = {
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
            core = Mock()
            core.get_args_of_build.return_value = args
            core.get_user_jit_dir.return_value = str(target_dir)
            core._so_offload_archs.return_value = {"gfx950"}
            build(core, root, "module_gemm")
            self.assertEqual(core.build_module.call_args.kwargs["srcs"], args["srcs"])
            self.assertEqual(
                core.build_module.call_args.kwargs["flags_extra_hip"], ["-O3"]
            )
            core._so_offload_archs.return_value = {"gfx942"}
            with self.assertRaisesRegex(RuntimeError, "architectures"):
                build(core, root, "module_gemm")
            (target_dir / "module_gemm.so").unlink()
            with self.assertRaisesRegex(RuntimeError, "did not create"):
                build(core, root, "module_gemm")


if __name__ == "__main__":
    unittest.main()
