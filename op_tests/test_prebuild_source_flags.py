# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only regression tests for per-source HIP flags in wheel prebuilds."""

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(  # noqa: S102 - trusted checkout function, without setup.py side effects
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


class TestPrebuildSourceFlags(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.flags = {
            "*attention_64rows_h16.cu": [
                "-fno-gpu-flush-denormals-to-zero",
                "-mllvm",
                "--amdgpu-kernarg-preload-count=0",
            ],
            "*attention_64rows_*.cu": ["-ffast-math"],
        }
        config = {
            "module_attention": {
                "srcs": ["'attention_64rows_h16.cu'"],
                "flags_extra_hip_per_source": self.flags,
            },
            "module_plain": {"srcs": ["'plain.cu'"]},
        }
        (Path(directory.name) / "optCompilerConfig.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        self.get_args = load_function(
            REPO_ROOT / "aiter/jit/core.py",
            "get_args_of_build",
            {
                "json": json,
                "os": os,
                "this_dir": directory.name,
                "CK_3RDPARTY_DIR": directory.name,
                "is_experimental_enabled": lambda: False,
            },
        )
        self.build_module = mock.Mock()
        self.prebuild = load_function(
            REPO_ROOT / "setup.py",
            "build_one_module",
            {
                "core": SimpleNamespace(build_module=self.build_module),
                "PREBUILD_KERNELS": 1,
            },
        )

    def test_all_modules_preserve_source_flags(self):
        modules, _ = self.get_args("all")
        self.assertEqual(len(modules), 2)
        for module in modules:
            with self.subTest(module=module["md_name"]):
                direct = self.get_args(module["md_name"])
                self.assertEqual(
                    module["flags_extra_hip_per_source"],
                    direct["flags_extra_hip_per_source"],
                )
                # First matching glob wins, so order must also survive.
                self.assertEqual(
                    list(module["flags_extra_hip_per_source"]),
                    list(direct["flags_extra_hip_per_source"]),
                )
        self.assertEqual(modules[0]["flags_extra_hip_per_source"], self.flags)
        self.assertEqual(modules[1]["flags_extra_hip_per_source"], {})

    def test_prebuild_forwards_source_flags(self):
        module = self.get_args("module_attention")
        self.prebuild(module)
        self.assertEqual(
            self.build_module.call_args.kwargs["flags_extra_hip_per_source"],
            self.flags,
        )
        self.assertEqual(module["flags_extra_hip"], [])

    def test_prebuild_accepts_legacy_module_without_source_flags(self):
        module = self.get_args("module_plain")
        module.pop("flags_extra_hip_per_source")
        self.prebuild(module)
        self.assertEqual(
            self.build_module.call_args.kwargs["flags_extra_hip_per_source"], {}
        )


if __name__ == "__main__":
    unittest.main()
