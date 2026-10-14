# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib.metadata
import importlib.util
import io
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import tracemalloc
import types
import unittest
from contextlib import ExitStack, nullcontext
from unittest.mock import Mock, patch


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
        self.distributions = {}

        def distribution(name):
            if name not in self.distributions:
                raise importlib.metadata.PackageNotFoundError(name)
            return self.distributions[name]

        self.metadata = self.stack.enter_context(
            patch.object(
                self.cpp_extension.importlib.metadata,
                "distribution",
                side_effect=distribution,
            )
        )

    def install_sdk_metadata(self, name="rocm-sdk-core", version="7.15.0a20261013"):
        # Real wheel metadata, including the venv console-script RECORD entry.
        site = self.root / "venv" / "lib" / "site-packages"
        info = site / f"{name.replace('-', '_')}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
        (info / "RECORD").write_text("../../bin/hipcc,,\n")
        compiler = site.parent.parent / "bin" / "hipcc"
        compiler.parent.mkdir(parents=True, exist_ok=True)
        compiler.touch()
        dist = importlib.metadata.PathDistribution(info)
        self.distributions[name] = dist
        return site, dist

    def assert_header_diagnostic(self, message):
        self.assertIn("rocprim/rocprim.hpp", message)
        self.assertIn(f"Selected ROCM_HOME: {self.runtime.parent}", message)
        self.assertIn(f"Selected hipcc: {self.hipcc}", message)
        self.assertIn("https://rocm.docs.amd.com/en/latest/install/rocm.html", message)
        self.assertNotIn("docs/packaging/python_packaging.md", message)
        self.assertIn("development", message)


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
        self.metadata.assert_not_called()

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
        self.metadata.assert_not_called()

    def test_sdk_hint_pins_full_installed_version_for_selected_trampoline(self):
        for version in ("7.2.1", "7.15.0a20261013", "7.15.0a20261013+build.42"):
            with self.subTest(version=version):
                site, _ = self.install_sdk_metadata(version=version)
                self.hipcc = str(site.parent.parent / "bin" / "hipcc")
                self.compiler.return_value = self.hipcc
                message = self.cpp_extension._rocm_header_build_hint(
                    "fatal error: 'rocprim/rocprim.hpp' file not found"
                )
                self.assert_header_diagnostic(message)
                self.assertIn(f"rocm-sdk-core=={version}", message)
                self.assertIn(f'pip install "rocm[devel]=={version}"', message)
                self.assertIn("original-SDK-index-URL", message)
                self.assertIn("-m rocm_sdk init", message)
                self.assertIn("-m rocm_sdk path --root", message)
                self.assertIn("ROCm/TheRock/blob/main/RELEASES.md", message)
                self.assertNotIn("rocprim-dev (", message)

    def test_sdk_hint_recognizes_lazy_devel_tree_without_compiler_record(self):
        site, dist = self.install_sdk_metadata(name="rocm-sdk-devel")
        dist._path.joinpath("RECORD").unlink()
        for suffix in ("", ".exe", ".bat"):
            with self.subTest(suffix=suffix):
                hipcc = str(site / "_rocm_sdk_devel" / "bin" / f"hipcc{suffix}")
                hint = self.cpp_extension._rocm_devel_install_hint(hipcc)
                self.assertIn("rocm-sdk-devel==7.15.0a20261013", hint)
                self.assertIn('"rocm[devel]==7.15.0a20261013"', hint)

        other = site / "_rocm_sdk_devel-other" / "bin" / "hipcc"
        self.assertNotIn(
            "pip install", self.cpp_extension._rocm_devel_install_hint(str(other))
        )

    def test_sdk_command_quotes_python_executable(self):
        site, _ = self.install_sdk_metadata()
        hipcc = str(site.parent.parent / "bin" / "hipcc")
        for windows, executable, quoted in (
            (False, "/venv with spaces/bin/python", "'/venv with spaces/bin/python'"),
            (True, r"C:\Python Env\python.exe", '"C:\\Python Env\\python.exe"'),
        ):
            with self.subTest(windows=windows), patch.object(
                self.cpp_extension, "IS_WINDOWS", windows
            ), patch.object(sys, "executable", executable):
                hint = self.cpp_extension._rocm_devel_install_hint(hipcc)
                self.assertIn(f"{quoted} -m pip install", hint)
                self.assertIn(f"{quoted} -m rocm_sdk init", hint)

    def test_system_release_comes_from_compiler_not_unrelated_sdk_or_rocm_home(self):
        self.install_sdk_metadata()
        compiler_root = pathlib.Path(self.hipcc).parent.parent
        (compiler_root / ".info").mkdir(parents=True)
        (self.runtime.parent / ".info").mkdir()
        (self.runtime.parent / ".info" / "version").write_text("6.4.0")
        for name in ("version", "version-dev"):
            with self.subTest(name=name):
                info = compiler_root / ".info" / name
                info.write_text("7.2.0-43\n")
                hint = self.cpp_extension._rocm_devel_install_hint(self.hipcc)
                self.assertIn("Selected compiler ROCm release: 7.2.0-43", hint)
                self.assertIn("packages for ROCm 7.2.0-43", hint)
                self.assertIn("rocprim-dev (Debian/Ubuntu)", hint)
                self.assertIn("rocprim-devel (RPM)", hint)
                self.assertNotIn("6.4.0", hint)
                self.assertNotIn("7.15.0", hint)
                self.assertNotIn("pip install", hint)
                info.unlink()

    def test_system_release_follows_compiler_symlink_and_legacy_hip_bin(self):
        root = self.root / "system-rocm"
        compiler = root / "hip" / "bin" / "hipcc"
        compiler.parent.mkdir(parents=True)
        compiler.touch()
        (root / ".info").mkdir()
        (root / ".info" / "version").write_text("6.4.3")
        link = self.root / "hipcc"
        try:
            link.symlink_to(compiler)
        except OSError as error:
            self.skipTest(f"Compiler symlinks unavailable: {error}")
        hint = self.cpp_extension._rocm_devel_install_hint(str(link))
        self.assertIn(f"6.4.3 (root: {root})", hint)

    def test_unknown_or_malformed_metadata_does_not_invent_package_pin(self):
        root = pathlib.Path(self.hipcc).parent.parent
        (root / ".info").mkdir(parents=True)
        for value in ("", "7.2", "not a release", "7.2.0\n7.1.0", "\ufffd"):
            with self.subTest(value=value):
                (root / ".info" / "version").write_bytes(value.encode("utf-8"))
                hint = self.cpp_extension._rocm_devel_install_hint(self.hipcc)
                self.assertIn("Selected compiler ROCm release: unknown", hint)
                self.assertNotIn("pip install", hint)
                # The import-time HIP_VERSION is not a package release pin.
                self.assertNotIn("7.15.0", hint)

    def test_invalid_sdk_version_does_not_invent_package_pin(self):
        site, _ = self.install_sdk_metadata(version="invalid")
        hint = self.cpp_extension._rocm_devel_install_hint(
            str(site.parent.parent / "bin" / "hipcc")
        )
        self.assertNotIn("pip install", hint)
        self.assertIn("Selected compiler ROCm release: unknown", hint)

    def test_unreadable_metadata_keeps_missing_header_diagnostic(self):
        for error in (OSError("unreadable"), ValueError("invalid metadata")):
            with self.subTest(error=error):
                self.metadata.side_effect = error
                self.assert_header_diagnostic(
                    self.cpp_extension._rocm_header_build_hint(
                        "fatal error: 'rocprim/rocprim.hpp' file not found"
                    )
                )


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
        self.stream = self.stack.enter_context(tempfile.TemporaryFile())  # noqa: SIM115
        self.real_write = os.write
        self.write_limit = None
        self.output = self.stack.enter_context(
            patch.object(
                self.cpp_extension.os, "write", side_effect=self.forward_output
            )
        )

    def forward_output(self, fd, data):
        self.assertEqual(fd, 1)
        # Exercise a real file descriptor, including partial writes when requested.
        return self.real_write(self.stream.fileno(), data[: self.write_limit])

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
        for mode in ("quiet", "verbose", "env_verbose", "partial_write"):
            with self.subTest(mode=mode):
                self.output.reset_mock()
                self.stream.seek(0)
                self.stream.truncate()
                self.write_limit = 17 if mode == "partial_write" else None
                os.environ["AITER_JIT_VERBOSE"] = "1" if mode == "env_verbose" else "0"
                with self.assertRaises(RuntimeError) as error:
                    self.build(verbose=mode in ("verbose", "partial_write"))
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
                    self.stream.seek(0)
                    self.assertEqual(
                        self.stream.read(), error.exception.__cause__.output
                    )

    def test_malformed_sdk_record_preserves_compiler_error(self):
        _, dist = self.install_sdk_metadata()
        dist._path.joinpath("RECORD").write_text("../../bin/hipcc,,,unexpected\n")
        with self.assertRaises(RuntimeError) as error:
            self.build()
        self.assert_header_diagnostic(str(error.exception))
        self.assertIn("fatal error:", str(error.exception))
        self.assertIsInstance(error.exception.__cause__, subprocess.CalledProcessError)

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


class NinjaBuildProcessTest(RocmHeaderTestCase):
    def test_empty_non_utf8_and_diagnostic_errors_preserve_build_failure(self):
        for output in (b"", b"error: invalid byte \xff\n"):
            with self.subTest(output=output):
                process = Mock(returncode=1)

                def popen(*args, stdout, output=output, process=process, **kwargs):
                    stdout.write(output)
                    return nullcontext(process)

                with patch.object(
                    self.cpp_extension.subprocess, "Popen", side_effect=popen
                ), patch.object(
                    self.cpp_extension, "_get_num_workers", return_value=None
                ), patch.object(
                    self.cpp_extension,
                    "_rocm_header_build_hint",
                    side_effect=RuntimeError("diagnostic lookup failed"),
                ), self.assertRaises(
                    RuntimeError
                ) as error:
                    self.cpp_extension._run_ninja_build(str(self.root), False, "build")
                expected = "build" + (
                    ": " + output.decode(errors="replace") if output else ""
                )
                self.assertEqual(str(error.exception), expected)
                self.assertEqual(error.exception.__cause__.output, output)

    def test_interruption_kills_child(self):
        for verbose in (False, True):
            with self.subTest(verbose=verbose):
                process = Mock()
                operation = process.stdout.read1 if verbose else process.wait
                operation.side_effect = KeyboardInterrupt()
                with patch.object(
                    self.cpp_extension.subprocess,
                    "Popen",
                    return_value=nullcontext(process),
                ), self.assertRaises(KeyboardInterrupt):
                    self.cpp_extension._run_ninja_build(
                        str(self.root), verbose, "build"
                    )
                process.kill.assert_called_once()

    def test_successful_verbose_log_uses_bounded_memory(self):
        # Allocate the source before tracing: measure capture, not the test data.
        process = Mock(stdout=io.BytesIO((b"x" * 4095 + b"\n") * 4096), returncode=0)
        with patch.object(
            self.cpp_extension.subprocess, "Popen", return_value=nullcontext(process)
        ), patch.object(self.cpp_extension.os, "write", new=lambda fd, data: len(data)):
            tracemalloc.start()
            try:
                self.cpp_extension._run_ninja_build(str(self.root), True, "build")
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
        self.assertLess(peak, 2 * 1024 * 1024)  # 16 MiB log must not stay in memory.


if __name__ == "__main__":
    unittest.main(verbosity=2)
