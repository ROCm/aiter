# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests: one uncompilable tune candidate must not kill the module.

Covers issue #5265: parsing ninja's per-unit failures, rewriting the failed
candidate's translation unit into a stub that throws a named error, refusing to
isolate anything that is not a lone candidate, and the end-of-run summary.

Run: python op_tests/test_jit_build_failures.py
"""

import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

UTILS = pathlib.Path(__file__).resolve().parents[1] / "aiter" / "jit" / "utils"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, UTILS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with patch.object(sys, "path", [str(UTILS), *sys.path]):
        spec.loader.exec_module(module)
    return module


bf = _load("build_failures", "build_failures.py")

LDS_ERROR = (
    "error: local memory (262144) exceeds limit (163840) in "
    "'ck_tile::kentry<1, ck_tile::FlatmmKernel<...>>'"
)
BAD = "a8w8_bpreshuffle_cktile_0x0x1x1x1x0x0x0x0x1_256x256x512_dFP32_eBF16"
GOOD = "a8w8_bpreshuffle_cktile_0x0x1x1x1x0x0x0x0x1_128x128x128_dFP32_eBF16"

# What a failed ninja run looks like: unrelated warnings, one good unit, two
# failed ones (the first with the ninja >= 1.11 "[code=N]" form).
NINJA_LOG = f"""\
[1/4] /opt/rocm/bin/hipcc -c pybind.cu -o pybind.cuda.o
pybind.cu:3:1: warning: NUMA balancing is enabled [-Wfoo]
[2/4] /opt/rocm/bin/hipcc -c {GOOD}.cpp -o {GOOD}.cuda.o
FAILED: [code=1] {BAD}.cuda.o
/opt/rocm/bin/hipcc -Werror=return-type -c {BAD}.cpp -o {BAD}.cuda.o
warning: hipcc not support something
\x1b[1m{BAD}.cpp:1\x1b[0m: \x1b[31m{LDS_ERROR}\x1b[0m
{LDS_ERROR}
1 error generated when compiling for gfx950.
[3/4] /opt/rocm/bin/hipcc -c other.cpp -o other.cuda.o
FAILED: other.cuda.o
/opt/rocm/bin/hipcc -c other.cpp -o other.cuda.o
other.cpp:9:5: fatal error: 'missing.h' file not found
ninja: build stopped: subcommand failed.
"""

INSTANCE_CPP = """\
// SPDX-License-Identifier: MIT
#include "impl/{name}.cuh"

template torch::Tensor
{name}<F32, B16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    int KBatch
    );
"""


class ParseTest(unittest.TestCase):
    def test_each_failed_unit_is_reported_with_its_errors_only(self):
        failures = bf.parse_ninja_failures(NINJA_LOG)
        self.assertEqual(
            [f.target for f in failures], [f"{BAD}.cuda.o", "other.cuda.o"]
        )
        self.assertEqual([f.stem for f in failures], [BAD, "other"])
        # Warnings, the echoed command (-Werror=...), ANSI codes and the
        # per-arch repeat of a diagnostic do not show up as errors.
        first = failures[0].errors
        self.assertIn(LDS_ERROR, first)
        self.assertEqual(len(first), len(set(first)))
        self.assertFalse(any("-Werror" in e or "\x1b" in e for e in first))
        self.assertFalse(any("warning" in e for e in first))
        self.assertIn("missing.h", failures[1].errors[0])

    def test_old_ninja_failed_line_without_code(self):
        log = "FAILED: x.cuda.o\ncmd\nx.cpp:1: error: boom\n"
        self.assertEqual(
            bf.parse_ninja_failures(log)[0].errors, ["x.cpp:1: error: boom"]
        )

    def test_output_without_failed_block_has_no_failures(self):
        self.assertEqual(bf.parse_ninja_failures("error: invalid byte\n"), [])
        self.assertEqual(bf.parse_ninja_failures(b"error: x \xff\n"), [])

    def test_crash_without_error_line_keeps_the_tail(self):
        log = "FAILED: y.cuda.o\ncmd\nwarning: noise\nsomething\nKilled\n"
        self.assertEqual(
            bf.parse_ninja_failures(log)[0].errors, ["something", "Killed"]
        )

    def test_summary_names_unit_and_error(self):
        text = bf.format_failure_summary(bf.parse_ninja_failures(NINJA_LOG))
        self.assertIn("2 translation unit(s) failed to compile", text)
        self.assertIn(BAD, text)
        self.assertIn("262144", text)
        self.assertEqual(bf.format_failure_summary([]), "")


class IsolateTest(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.good = self._write(f"{GOOD}.cpp", INSTANCE_CPP.format(name=GOOD))
        self.bad = self._write(f"{BAD}.cpp", INSTANCE_CPP.format(name=BAD))
        self.pybind = self._write("pybind.cu", "int main() { return 0; }\n")
        self.sources = sorted([self.good, self.bad, self.pybind])
        self.failures = bf.parse_ninja_failures(NINJA_LOG)[:1]

    def _write(self, name, text):
        path = self.dir / name
        path.write_text(text)
        return str(path)

    def test_failed_candidate_becomes_a_named_throwing_specialization(self):
        isolated = bf.isolate_failed_instances(self.failures, self.sources)
        self.assertEqual([i.name for i in isolated], [BAD])
        text = pathlib.Path(self.bad).read_text()
        self.assertIn(f'#include "impl/{BAD}.cuh"', text)  # primary stays in scope
        self.assertIn("#include <stdexcept>", text)
        self.assertRegex(text, rf"template <> torch::Tensor\s+{BAD}<F32, B16>\(")
        self.assertIn("throw std::runtime_error(", text)
        self.assertIn(f"tune candidate '{BAD}' {bf.COMPILE_FAILED_MARKER}", text)
        self.assertIn("local memory (262144) exceeds limit (163840)", text)
        self.assertNotRegex(text, r"(?m)^template torch::Tensor")

    def test_other_candidates_and_dispatchers_are_untouched(self):
        before = (
            pathlib.Path(self.good).read_text(),
            pathlib.Path(self.pybind).read_text(),
        )
        bf.isolate_failed_instances(self.failures, self.sources)
        after = (
            pathlib.Path(self.good).read_text(),
            pathlib.Path(self.pybind).read_text(),
        )
        self.assertEqual(before, after)

    def test_message_cannot_break_the_generated_source(self):
        failure = bf.CompileFailure(f"{BAD}.cuda.o", ['error: "x" \\ \u00e9\n y'])
        bf.isolate_failed_instances([failure], self.sources)
        text = pathlib.Path(self.bad).read_text()
        self.assertIn('\\"x\\" \\\\ ? y', text)
        self.assertTrue(text.isascii())
        self.assertEqual(text.count("\n"), len(text.splitlines()))

    def test_not_isolated_when_a_dispatcher_unit_failed(self):
        failure = bf.CompileFailure("pybind.cuda.o", ["error: boom"])
        before = pathlib.Path(self.bad).read_text()
        self.assertIsNone(
            bf.isolate_failed_instances(self.failures + [failure], self.sources)
        )
        self.assertEqual(pathlib.Path(self.bad).read_text(), before)

    def test_not_isolated_when_every_candidate_failed(self):
        failures = self.failures + [bf.CompileFailure(f"{GOOD}.cuda.o", ["error: x"])]
        self.assertIsNone(bf.isolate_failed_instances(failures, self.sources))
        self.assertNotIn("throw", pathlib.Path(self.bad).read_text())

    def test_unrecognised_source_aborts_before_any_rewrite(self):
        odd = self._write("odd.cpp", '#include "impl/odd.cuh"\nvoid f();\n')
        failures = self.failures + [bf.CompileFailure("odd.cuda.o", ["error: x"])]
        before = pathlib.Path(self.bad).read_text()
        result = bf.isolate_failed_instances(failures, self.sources + [odd])
        self.assertIsNone(result)
        self.assertEqual(pathlib.Path(self.bad).read_text(), before)

    def test_unknown_object_or_no_failure(self):
        ghost = bf.CompileFailure("ghost.cuda.o", ["error: x"])
        self.assertIsNone(bf.isolate_failed_instances([ghost], self.sources))
        self.assertIsNone(bf.isolate_failed_instances([], self.sources))

    @unittest.skipUnless(shutil.which("c++") or shutil.which("g++"), "no C++ compiler")
    def test_stub_compiles_without_instantiating_the_failing_body(self):
        impl = self.dir / "impl"
        impl.mkdir()
        # The primary template's body is invalid when instantiated.
        (impl / f"{BAD}.cuh").write_text(
            "#pragma once\nstruct Tensor {};\nusing F32 = float; using B16 = short;\n"
            f"template <typename D, typename E> Tensor {BAD}(Tensor &a, int k) "
            '{ static_assert(sizeof(D) == 0, "too much LDS"); return a; }\n'
        )
        src = self.dir / "unit.cpp"
        src.write_text(
            f'#include "impl/{BAD}.cuh"\n'
            f"template Tensor\n{BAD}<F32, B16>(Tensor &a, int k);\n"
        )
        compiler = shutil.which("c++") or shutil.which("g++")
        cmd = [compiler, "-std=c++20", "-c", str(src), "-o", str(self.dir / "u.o")]
        self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
        failure = bf.CompileFailure("unit.cuda.o", [LDS_ERROR])
        other = self._write("other_inst.cpp", INSTANCE_CPP.format(name=GOOD))
        self.assertTrue(bf.isolate_failed_instances([failure], [str(src), other]))
        out = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)


class SummaryTest(unittest.TestCase):
    def test_end_of_run_summary_is_deduplicated_and_only_compile_failures(self):
        detail = (
            f"RuntimeError: [aiter] tune candidate '{BAD}' {bf.COMPILE_FAILED_MARKER} e"
        )
        text = bf.summarize_compile_failures(
            [detail, "RuntimeError: other", detail, "", detail]
        )
        self.assertIn("1 candidate(s) failed to compile", text)
        self.assertEqual(text.count(BAD), 1)
        self.assertEqual(bf.summarize_compile_failures(["RuntimeError: other"]), "")


class RunNinjaBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"ROCM_HOME": "/test-rocm"}), patch(
            "shutil.which", return_value="/test-rocm/bin/hipconfig"
        ), patch("subprocess.check_output", return_value="7.15.0"):
            cls.ext = _load("_test_cpp_extension_failures", "cpp_extension.py")

    def _run(self, output, **kwargs):
        process = Mock(returncode=1 if output is not None else 0)
        seen = {}

        def popen(command, *, stdout, **_):
            seen["command"] = command
            stdout.write(output or b"")
            return nullcontext(process)

        with patch.object(
            self.ext.subprocess, "Popen", side_effect=popen
        ), patch.object(self.ext, "_get_num_workers", return_value=4), patch.object(
            self.ext, "_rocm_header_build_hint", return_value=""
        ):
            try:
                self.ext._run_ninja_build("/tmp", False, "build", **kwargs)
                error = None
            except RuntimeError as exc:
                error = exc
        return seen["command"], error

    def test_keep_going_is_opt_in(self):
        command, _ = self._run(
            b"",
        )
        self.assertNotIn("-k", command)
        command, _ = self._run(b"", keep_going=True)
        self.assertEqual(
            command[command.index("-k") : command.index("-k") + 2], ["-k", "0"]
        )

    def test_error_carries_failures_and_ends_with_summary(self):
        _, error = self._run(NINJA_LOG.encode(), keep_going=True)
        self.assertIsInstance(error, bf.JitCompileError)
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(len(error.failures), 2)
        message = str(error)
        self.assertTrue(message.startswith("build: "))
        self.assertIn("NUMA balancing", message)  # the full log is still there
        tail = message[message.rindex("2 translation unit(s)") :]
        self.assertIn(BAD, tail)
        self.assertIn("262144", tail)

    def test_output_without_failed_block_is_left_alone(self):
        _, error = self._run(b"error: invalid byte\n")
        self.assertEqual(str(error), "build: error: invalid byte\n")
        self.assertEqual(error.failures, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
