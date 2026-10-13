# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
import os
import pathlib
import shlex
import shutil
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


class RocmHeaderTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpp_extension = _load_cpp_extension()

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = pathlib.Path(
            self.stack.enter_context(tempfile.TemporaryDirectory())
        )
        self.runtime = self.root / "python-sdk" / "include"
        self.runtime.mkdir(parents=True)
        self.hipcc = str(self.root / "selected-compiler" / "bin" / "hipcc")
        self.stack.enter_context(
            patch.multiple(
                self.cpp_extension,
                ROCM_HOME=str(self.runtime.parent),
                IS_HIP_EXTENSION=True,
            )
        )
        self.compiler = self.stack.enter_context(
            patch.object(self.cpp_extension, "executable_path", return_value=self.hipcc)
        )
        self.devel = self.stack.enter_context(
            patch.object(
                self.cpp_extension, "_find_rocm_devel_include", return_value=None
            )
        )
        self.stack.enter_context(patch.dict(os.environ))
        for name in ("CPATH", "CPLUS_INCLUDE_PATH", "AITER_JIT_VERBOSE"):
            os.environ.pop(name, None)
        self.stack.enter_context(
            patch.object(self.cpp_extension, "_get_num_workers", return_value=1)
        )

    def assert_header_diagnostic(self, message):
        self.assertIn("rocprim/rocprim.hpp", message)
        self.assertIn(f"Selected ROCM_HOME: {self.runtime.parent}", message)
        self.assertIn(f"Selected hipcc: {self.hipcc}", message)
        self.assertIn("https://rocm.docs.amd.com/en/latest/install/rocm.html", message)
        self.assertIn(
            "ROCm/TheRock/blob/main/docs/packaging/python_packaging.md", message
        )
        self.assertIn("Install and initialize", message)


class RocmHeaderDiagnosticsTest(RocmHeaderTestCase):
    def test_compiler_missing_header_diagnostics(self):
        for output in (
            "test.hip:1:10: fatal error: 'rocprim/rocprim.hpp' file not found",
            "test.hip:1:10: fatal error: rocprim/rocprim.hpp: No such file or directory",
            'test.hip:1:10: error: "rocprim/rocprim.hpp" file not found',
            "test.hip:1:10: fatal error: 'rocprim\\rocprim.hpp' file not found",
        ):
            with self.subTest(output=output):
                self.assert_header_diagnostic(
                    self.cpp_extension._rocm_header_build_hint(output)
                )

    def test_unrelated_errors_do_not_get_installation_advice(self):
        for output in (
            "fatal error: 'other.hpp' file not found\n#include <rocprim/rocprim.hpp>",
            "rocprim/rocprim.hpp:10: error: unknown type name 'T'",
            "fatal error: 'rocprim/rocprim.hpp.detail' file not found",
            "undefined reference to rocprim::foo()",
            "",
        ):
            with self.subTest(output=output):
                self.assertEqual(self.cpp_extension._rocm_header_build_hint(output), "")
        self.compiler.assert_not_called()

    def test_non_hip_error_does_not_get_rocm_advice(self):
        self.cpp_extension.IS_HIP_EXTENSION = False
        self.assertEqual(
            self.cpp_extension._rocm_header_build_hint(
                "fatal error: 'rocprim/rocprim.hpp' file not found"
            ),
            "",
        )
        self.compiler.assert_not_called()

    def test_hipcc_falls_back_to_selected_rocm_home(self):
        self.compiler.side_effect = AssertionError("hipcc is not on PATH")
        self.hipcc = str(self.runtime.parent / "bin" / "hipcc")
        self.assert_header_diagnostic(
            self.cpp_extension._rocm_header_build_hint(
                "fatal error: 'rocprim/rocprim.hpp' file not found"
            )
        )

    def test_include_discovery_does_not_probe_or_append_system_headers(self):
        torch = types.SimpleNamespace(__file__=str(self.root / "torch" / "__init__.py"))
        devel = str(self.root / "sdk-devel" / "include")
        with patch.dict(sys.modules, {"torch": torch}), patch.object(
            self.cpp_extension.os.path, "isfile", return_value=True
        ) as header_lookup, patch.object(
            self.cpp_extension.subprocess, "check_output"
        ) as probe:
            for sdk_devel in (None, devel):
                with self.subTest(sdk_devel=sdk_devel):
                    self.devel.return_value = sdk_devel
                    paths = self.cpp_extension.include_paths(cuda=True)
                    self.assertIn(str(self.runtime), paths)
                    self.assertNotIn("/opt/rocm/include", paths)
                    if sdk_devel:
                        self.assertIn(devel, paths)
            header_lookup.assert_not_called()
            probe.assert_not_called()
        self.compiler.assert_not_called()


@unittest.skipUnless(
    sys.platform != "win32" and shutil.which("ninja") and shutil.which("c++"),
    "requires Ninja and a POSIX C++ compiler (no ROCm/GPU needed)",
)
class RocmHeaderBuildTest(RocmHeaderTestCase):
    def setUp(self):
        super().setUp()
        # Exercise real compiler include resolution with a tiny stand-in header.
        # The host compiler is used only as a CPU test double for hipcc.
        compiler = pathlib.Path(self.hipcc)
        compiler.parent.mkdir(parents=True)
        compiler.symlink_to(shutil.which("c++"))
        self.source = self.root / "test.cpp"
        self.source.write_text(
            "#include <rocprim/rocprim.hpp>\nint value = ROCPRIM_TEST;\n"
        )
        self.obj = self.root / "test.o"
        self.headers = self.root / "explicit includes"
        (self.headers / "rocprim").mkdir(parents=True)
        (self.headers / "rocprim" / "rocprim.hpp").write_text(
            "#define ROCPRIM_TEST 1\n"
        )
        self.stack.enter_context(
            patch.object(
                self.cpp_extension, "get_cxx_compiler", return_value=self.hipcc
            )
        )
        self.output = self.stack.enter_context(
            patch.object(self.cpp_extension.os, "write")
        )

    def build(self, *, verbose=False, flags=()):
        if self.obj.exists():
            self.obj.unlink()
        self.cpp_extension._write_ninja_file(
            path=str(self.root / "build.ninja"),
            cflags=[],
            post_cflags=[],
            cuda_cflags=list(flags),
            cuda_post_cflags=[],
            cuda_dlink_post_cflags=[],
            sources=[str(self.source)],
            objects=[str(self.obj)],
            ldflags=[],
            library_target=None,
            with_cuda=True,
        )
        self.cpp_extension._run_ninja_build(
            str(self.root), verbose=verbose, error_prefix="Error building header test"
        )

    def test_failed_build_reports_toolchain_in_quiet_and_verbose_modes(self):
        for mode in ("quiet", "verbose", "env_verbose"):
            with self.subTest(mode=mode):
                self.output.reset_mock()
                os.environ["AITER_JIT_VERBOSE"] = "1" if mode == "env_verbose" else "0"
                with self.assertRaises(RuntimeError) as error:
                    self.build(verbose=mode == "verbose")
                message = str(error.exception)
                self.assertTrue(message.startswith("Error building header test"))
                self.assert_header_diagnostic(message)
                self.assertIn("fatal error:", message)
                self.assertIsInstance(
                    error.exception.__cause__, subprocess.CalledProcessError
                )
                if mode == "quiet":
                    self.output.assert_not_called()
                else:
                    streamed = b"".join(
                        call.args[1] for call in self.output.call_args_list
                    )
                    self.assertIn(b"fatal error:", streamed)

    def test_explicit_include_configuration_succeeds_without_rocm_home_header(self):
        for mode in ("CPATH", "CPLUS_INCLUDE_PATH", "-I", "-isystem"):
            for verbose in (False, True):
                with self.subTest(mode=mode, verbose=verbose), patch.dict(os.environ):
                    flags = []
                    if mode in ("CPATH", "CPLUS_INCLUDE_PATH"):
                        os.environ[mode] = str(self.headers)
                    else:
                        flags = [mode, shlex.quote(str(self.headers))]
                    self.assertFalse(
                        (self.runtime / "rocprim" / "rocprim.hpp").exists()
                    )
                    with patch.object(
                        self.cpp_extension, "_rocm_header_build_hint"
                    ) as hint:
                        self.build(verbose=verbose, flags=flags)
                    self.assertTrue(self.obj.is_file())
                    hint.assert_not_called()

    def test_unrelated_build_failure_keeps_original_error(self):
        self.source.write_text('#error "unrelated compiler failure"\n')
        for verbose in (False, True):
            with self.subTest(verbose=verbose), self.assertRaises(
                RuntimeError
            ) as error:
                self.build(verbose=verbose)
            message = str(error.exception)
            self.assertIn("unrelated compiler failure", message)
            self.assertNotIn("Selected ROCM_HOME:", message)
            self.assertNotIn("rocm.docs.amd.com", message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
