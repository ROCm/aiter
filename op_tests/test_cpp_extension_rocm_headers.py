# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from unittest.mock import patch


def _load_cpp_extension():
    # Load the real build utility without importing AITER's GPU operators.
    utils = pathlib.Path(__file__).resolve().parents[1] / "aiter" / "jit" / "utils"
    spec = importlib.util.spec_from_file_location(
        "_test_cpp_extension", utils / "cpp_extension.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(utils), *sys.path]), patch.dict(
        os.environ, {"ROCM_HOME": "/test-rocm"}
    ), patch("shutil.which", return_value="/test-rocm/bin/hipconfig"), patch(
        "subprocess.check_output", return_value="7.15.0"
    ):
        spec.loader.exec_module(module)
    return module


class RocmHeaderDiscoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpp_extension = _load_cpp_extension()

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        # The compiler probe is cached per hipcc path, so start every test cold
        # and let each case observe its own mocked `hipcc --version` output.
        self.cpp_extension._probe_hipcc_version.cache_clear()
        self.cpp_extension._warn_rocm_include_once.cache_clear()
        tempdir = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = pathlib.Path(tempdir)
        self.runtime = self.root / "python-sdk" / "include"
        self.runtime.mkdir(parents=True)
        self.system = self.root / "system" / "include"
        (self.system / "hip").mkdir(parents=True)
        (self.system / "rocprim").mkdir()
        self.version_header = self.system / "hip" / "hip_version.h"
        self.version_header.write_text(
            "#define HIP_VERSION_MAJOR 7\n#define HIP_VERSION_MINOR 15\n"
        )
        (self.system / "rocprim" / "rocprim.hpp").touch()
        self.stack.enter_context(
            patch.multiple(
                self.cpp_extension,
                ROCM_HOME=str(self.runtime.parent),
                IS_HIP_EXTENSION=True,
                ROCM_VERSION=(7, 15),
                _SYSTEM_ROCM_INCLUDE=str(self.system),
            )
        )
        self.devel = self.stack.enter_context(
            patch.object(
                self.cpp_extension, "_find_rocm_devel_include", return_value=None
            )
        )
        self.compiler = self.stack.enter_context(
            patch.object(
                self.cpp_extension.subprocess,
                "check_output",
                return_value="HIP version: 7.15.60401\nAMD clang version 22.0.0\n",
            )
        )
        self.torch_include = self.root / "torch" / "include"
        self.stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "torch": types.SimpleNamespace(
                        __file__=str(self.root / "torch" / "__init__.py")
                    )
                },
            )
        )

    def test_matching_system_headers_are_appended(self):
        with patch.object(self.cpp_extension.logger, "warning") as warning:
            paths = self.cpp_extension.include_paths(cuda=True)
        self.assertEqual(paths[-2:], [str(self.runtime), str(self.system)])
        self.assertEqual(paths.count(str(self.system)), 1)
        warning.assert_not_called()
        self.compiler.assert_called_once_with(
            [str(self.runtime.parent / "bin" / "hipcc"), "--version"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )

    def test_version_mismatch_warns_once_with_versions_and_install_hint(self):
        self.compiler.return_value = "HIP version: 7.14.60000\n"
        with patch.object(self.cpp_extension.logger, "warning") as warning:
            for _ in range(3):
                self.assertNotIn(
                    str(self.system), self.cpp_extension.include_paths(cuda=True)
                )

        # include_paths() runs several times per build and once per JIT module,
        # so the mismatch must be reported once, not once per call.
        warning.assert_called_once()
        message = warning.call_args.args[0]
        self.assertIn(str(self.system), message)
        self.assertIn("7.15", message)
        self.assertIn("7.14", message)
        self.assertIn(str(self.runtime.parent / "bin" / "hipcc"), message)
        self.assertIn("rocm-sdk-devel==7.14.*", message)

    def test_unverifiable_versions_warn_once(self):
        with patch.object(self.cpp_extension.logger, "warning") as warning:
            self.version_header.write_bytes(b"")
            for _ in range(2):
                self.assertIsNone(
                    self.cpp_extension._find_matching_system_rocm_include()
                )
            warning.assert_called_once()
            self.assertIn(str(self.system), warning.call_args.args[0])

            # A failed compiler probe is likewise reported once, and a distinct
            # reason gets its own warning instead of being suppressed.
            self.cpp_extension._warn_rocm_include_once.cache_clear()
            warning.reset_mock()
            self.version_header.write_text(
                "#define HIP_VERSION_MAJOR 7\n#define HIP_VERSION_MINOR 15\n"
            )
            self.compiler.side_effect = FileNotFoundError()
            for _ in range(2):
                self.assertIsNone(
                    self.cpp_extension._find_matching_system_rocm_include()
                )
            warning.assert_called_once()
            self.assertIn("hipcc", warning.call_args.args[0])

    def test_compiler_mismatch_overrides_matching_global_version(self):
        for version in ("7.14.0", "6.15.0"):
            with self.subTest(version=version):
                self.cpp_extension._probe_hipcc_version.cache_clear()
                self.compiler.return_value = f"HIP version: {version}\n"
                self.assertNotIn(
                    str(self.system), self.cpp_extension.include_paths(cuda=True)
                )

    def test_matching_compiler_overrides_unrelated_global_version(self):
        self.cpp_extension.ROCM_VERSION = (6, 0)
        self.assertIn(str(self.system), self.cpp_extension.include_paths(cuda=True))

    def test_repeated_discovery_reuses_cached_compiler_probe(self):
        for _ in range(3):
            self.assertIn(str(self.system), self.cpp_extension.include_paths(cuda=True))
        self.compiler.assert_called_once_with(
            [str(self.runtime.parent / "bin" / "hipcc"), "--version"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )

    def test_compiler_probe_is_cached_per_hipcc_path(self):
        self.assertIn(str(self.system), self.cpp_extension.include_paths(cuda=True))
        other_sdk = self.root / "other-sdk"
        (other_sdk / "bin").mkdir(parents=True)
        self.cpp_extension.ROCM_HOME = str(other_sdk)

        self.assertIn(str(self.system), self.cpp_extension.include_paths(cuda=True))

        # A different ROCM_HOME selects a different hipcc, so it must probe that
        # compiler instead of reusing the first compiler's cached version.
        self.assertEqual(
            [invocation.args[0] for invocation in self.compiler.call_args_list],
            [
                [str(self.runtime.parent / "bin" / "hipcc"), "--version"],
                [str(other_sdk / "bin" / "hipcc"), "--version"],
            ],
        )

    def test_unknown_or_failed_compiler_is_rejected(self):
        for output in ("", "AMD clang version 7.15.0", "HIP version: unknown"):
            with self.subTest(output=output):
                self.cpp_extension._probe_hipcc_version.cache_clear()
                self.compiler.return_value = output
                self.assertIsNone(
                    self.cpp_extension._find_matching_system_rocm_include()
                )
        for error in (
            FileNotFoundError(),
            subprocess.CalledProcessError(1, "hipcc"),
            subprocess.TimeoutExpired("hipcc", 10),
        ):
            with self.subTest(error=error):
                self.cpp_extension._probe_hipcc_version.cache_clear()
                self.compiler.side_effect = error
                self.assertIsNone(
                    self.cpp_extension._find_matching_system_rocm_include()
                )

    def test_missing_rocprim_skips_compiler_probe(self):
        (self.system / "rocprim" / "rocprim.hpp").unlink()
        with patch.object(self.cpp_extension.logger, "warning") as warning:
            self.assertNotIn(
                str(self.system), self.cpp_extension.include_paths(cuda=True)
            )
        # Nothing to compare against, so no warning either.
        warning.assert_not_called()
        self.compiler.assert_not_called()

    def test_unreadable_or_incomplete_header_skips_compiler_probe(self):
        for content in (b"", b"#define HIP_VERSION_MAJOR 7\n", b"\xff"):
            with self.subTest(content=content):
                self.version_header.write_bytes(content)
                self.assertIsNone(
                    self.cpp_extension._find_matching_system_rocm_include()
                )
        self.version_header.unlink()
        self.assertIsNone(self.cpp_extension._find_matching_system_rocm_include())
        self.compiler.assert_not_called()

    def test_existing_rocprim_skips_system_fallback(self):
        for include in (
            self.runtime,
            self.root / "devel" / "include",
            self.torch_include,
        ):
            with self.subTest(include=include):
                (include / "rocprim").mkdir(parents=True)
                header = include / "rocprim" / "rocprim.hpp"
                header.touch()
                self.devel.return_value = str(include)
                self.assertNotIn(
                    str(self.system), self.cpp_extension.include_paths(cuda=True)
                )
                header.unlink()
        self.compiler.assert_not_called()

    def test_cpu_and_non_hip_builds_skip_discovery(self):
        self.assertNotIn(str(self.system), self.cpp_extension.include_paths(cuda=False))
        self.cpp_extension.IS_HIP_EXTENSION = False
        self.assertNotIn(str(self.system), self.cpp_extension.include_paths(cuda=True))
        self.devel.assert_not_called()
        self.compiler.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
