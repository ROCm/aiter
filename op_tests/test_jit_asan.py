# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only checks for opt-in GPU sanitizer flags and cache isolation."""

import functools
import os
import runpy
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
_load_functions = runpy.run_path(str(ROOT / "op_tests/test_jit_cache_transaction.py"))[
    "_load_functions"
]


class TestJitAsan(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = temporary.name
        environment = mock.patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.importer = mock.Mock()
        self.namespace = _load_functions(
            ROOT / "aiter/jit/core.py",
            ["_asan_build_flags", "get_user_jit_dir", "get_module_custom_op"],
            {
                "os": os,
                "functools": functools,
                "sys": types.SimpleNamespace(path=[]),
                "this_dir": self.root,
                "AITER_USE_ASAN": True,
                "__mds": {},
                "__package__": "aiter.jit",
                "importlib": self.importer,
                "logger": mock.Mock(),
                "torch_compile_guard": lambda: lambda fn: fn,
            },
        )

    def test_explicit_targets_get_xnack_and_sanitizer_flags(self):
        cc, hip, ld = self.namespace["_asan_build_flags"](["gfx942", "gfx950"])
        for flags in (cc, hip, ld):
            self.assertIn("-fsanitize=address", flags)
            self.assertIn("-shared-libsan", flags)
        self.assertIn("--offload-arch=gfx942:xnack+", hip)
        self.assertIn("--offload-arch=gfx950:xnack+", hip)
        self.assertIn("-Werror=option-ignored", hip)

    def test_unsupported_or_implicit_targets_fail_closed(self):
        for targets in (
            [],
            ["native"],
            ["gfx1100"],
            ["gfx942:xnack-"],
            ["gfx942", "native"],
        ):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                self.namespace["_asan_build_flags"](targets)

    def test_asan_requires_an_explicit_cache_root(self):
        with self.assertRaisesRegex(ValueError, "AITER_JIT_DIR"):
            self.namespace["get_user_jit_dir"]()

    def test_asan_cache_cannot_reuse_normal_cache(self):
        os.environ["AITER_JIT_DIR"] = self.root
        self.assertEqual(
            self.namespace["get_user_jit_dir"](), os.path.join(self.root, "asan")
        )
        self.assertTrue(os.path.isdir(os.path.join(self.root, "asan")))

    def test_normal_cache_path_is_unchanged(self):
        self.namespace["AITER_USE_ASAN"] = False
        os.environ["AITER_JIT_DIR"] = self.root
        self.assertEqual(self.namespace["get_user_jit_dir"](), self.root)

    def test_missing_asan_extension_does_not_import_prebuilt(self):
        os.environ["AITER_JIT_DIR"] = self.root
        with self.assertRaises(ModuleNotFoundError):
            self.namespace["get_module_custom_op"]("module_test")
        self.importer.import_module.assert_not_called()

    def test_import_outside_asan_cache_is_rejected_before_loading(self):
        os.environ["AITER_JIT_DIR"] = self.root
        directory = self.namespace["get_user_jit_dir"]()
        Path(directory, "module_test.so").touch()
        self.importer.util.find_spec.return_value = types.SimpleNamespace(
            origin="/ordinary/module_test.so"
        )
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "outside its isolated cache"):
                self.namespace["get_module_custom_op"]("module_test")
        self.importer.import_module.assert_not_called()

    def test_asan_extension_from_isolated_cache_is_loaded_once(self):
        os.environ["AITER_JIT_DIR"] = self.root
        directory = self.namespace["get_user_jit_dir"]()
        module = Path(directory, "module_test.so")
        module.touch()
        self.importer.util.find_spec.return_value = types.SimpleNamespace(
            origin=str(module)
        )
        self.importer.import_module.return_value = types.SimpleNamespace(
            __file__=str(module)
        )
        self.namespace["get_module_custom_op"]("module_test")
        self.namespace["get_module_custom_op"]("module_test")
        self.importer.import_module.assert_called_once_with("module_test")

    def test_normal_prebuilt_import_is_unchanged(self):
        self.namespace["AITER_USE_ASAN"] = False
        self.importer.import_module.return_value = types.SimpleNamespace(
            __file__="/ordinary/module_test.so"
        )
        self.namespace["get_module_custom_op"]("module_test")
        self.importer.import_module.assert_called_once_with("aiter.jit.module_test")
        self.importer.util.find_spec.assert_not_called()

    def test_compiler_selection(self):
        namespace = _load_functions(
            ROOT / "aiter/jit/utils/cpp_extension.py",
            ["get_cxx_compiler"],
            {
                "os": os,
                "_join_rocm_home": lambda *parts: os.path.join("/opt/rocm", *parts),
            },
        )
        compiler = namespace["get_cxx_compiler"]
        self.assertEqual(compiler(), "c++")
        os.environ["AITER_USE_ASAN"] = "1"
        self.assertEqual(compiler(), "/opt/rocm/llvm/bin/clang++")
        os.environ["CXX"] = "/custom/clang++"
        self.assertEqual(compiler(), "/custom/clang++")


if __name__ == "__main__":
    unittest.main()
