# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Conv3d CI configuration and direct-call regressions, strictly on pure CPU.

Run with ``python tests/test_conv3d_ci_modes.py`` or unittest/pytest. No torch,
aiter, FlyDSL, CUDA, or GPU detection is imported or executed. Production host
arithmetic is AST-extracted; the correctness runner uses an explicit fake backend.
"""

import ast
import builtins
import contextlib
import csv
import importlib.util
import io
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "op_tests/test_flydsl_conv_implicit.py"
CONV = ROOT / "aiter/ops/flydsl/kernels/conv"


def _load_script():
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "aiter", "flydsl", "pandas"}:
            raise AssertionError(f"CPU regression imported GPU dependency: {name}")
        return original(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("conv3d_ci_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with patch.object(builtins, "__import__", guarded):
        spec.loader.exec_module(module)
    return module


script = _load_script()


def _extract(path, names, namespace):
    """Execute only explicitly allowed host definitions, never their imports."""
    body = []
    found = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            body.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(t in names for t in targets):
                body.append(node)
                found.update(targets)
    assert set(names) <= found, f"Missing host definitions: {set(names) - found}"
    tree = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    # Executing only the whitelisted host arithmetic, with imports stripped.
    exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102


def _host_arithmetic():
    ns = {"NamedTuple": NamedTuple, "Conv3dImplicitParam": SimpleNamespace}
    _extract(
        CONV / "conv3d_gfx950_utils.py",
        {
            "MFMA_M",
            "MFMA_N",
            "WARP_SIZE",
            "BF16_BYTES",
            "LDG_VEC",
            "OOB_SENTINEL_ELEM",
            "OOB_SENTINEL_BYTES",
            "TILE_K",
            "TILES_PER_BARRIER",
            "DEFAULT_TILE",
            "KW_REUSE_STAGES",
            "LDS_BYTES_PER_CU",
            "MAX_GRID_YZ",
            "SPLITK_MAX_STAGING_BYTES",
            "SPLITK_AUTO_MAX_STAGING_BYTES",
            "TileConfig",
            "ConvGeometry",
            "LaunchGrid",
            "validate_launch_config",
            "make_tile_config",
            "out_extent",
            "make_conv_geometry",
            "make_launch_grid",
            "b_park_elems",
        },
        ns,
    )
    _extract(
        CONV / "conv3d_im2col.py",
        {
            "PADDING_MODES",
            "BIG_IN_NR",
            "Im2colPlan",
            "make_im2col_plan",
            "_base_lds_bytes",
            "_reuse_lds_bytes",
            "_patch_lds_bytes",
            "patch_needs_park",
        },
        ns,
    )
    _extract(CONV / "conv3d_implicit_gfx950.py", {"make_conv3d_implicit_param"}, ns)
    ns["AITER_CONV3D_KW_REUSE"] = 1
    _extract(
        ROOT / "aiter/ops/flydsl/conv_kernels.py",
        {"_resolve_splitk", "_pad_channels", "_implicit_param_from_problem"},
        ns,
    )
    # The assertion itself is also exercised, but its import statements are not.
    fn = next(
        n
        for n in ast.parse(SCRIPT.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "_assert_host_path"
    )
    fn.body = [n for n in fn.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    tree = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    # Same whitelist rule: the assertion's own imports never run here.
    exec(compile(tree, str(SCRIPT), "exec"), ns)  # noqa: S102
    return ns


def _param(ns, case, static=False):
    _, rank, xs, ws, kw, _, bias = case
    n, c, *spatial = xs
    k, _, *kernel = ws
    spatial = [1] * (3 - rank) + spatial
    kernel = [1] * (3 - rank) + kernel

    def triple(value, leading):
        axes = [value] * rank if isinstance(value, int) else list(value)
        return [leading] * (3 - rank) + axes

    stride = triple(kw.get("stride", 1), 1)
    padding = triple(kw.get("padding", 0), 0)
    dilation = triple(kw.get("dilation", 1), 1)
    param = ns["_implicit_param_from_problem"](
        n,
        c,
        *spatial,
        k,
        *kernel,
        *stride,
        *padding,
        *dilation,
        1,
        bias,
        kw.get("splitk", 1),
        kw["tile"],
        1,
        kw.get("output_layout") in {"NHWC", "NDHWC"},
        dyn_hw=not static,
        block_rows=kw.get("block_rows", 1),
    )
    geom = ns["make_conv_geometry"](param)
    param.splitk = ns["_resolve_splitk"](
        param.splitk, geom.npq, geom.crs, k, None, tile=param.tile
    )
    return param


class ConfigurationTests(unittest.TestCase):
    def parse(self, argv):
        with contextlib.redirect_stderr(io.StringIO()):
            return script.parse_args(argv)

    def test_default_is_six_wan_and_six_kw_not_qwen(self):
        args = self.parse([])
        self.assertEqual(args.mode, "ci")
        self.assertEqual(set(args.cases), set(script.CI_WAN_CASES + script.CI_KW_CASES))
        rows = script.selected_model_cases(args)
        self.assertEqual(len(rows), 6)
        by_name = {r[0]: r for r in rows}
        self.assertEqual(by_name["conv_in"][2], (1, 3, 6, 370, 546))
        self.assertIn("exact tuned check", by_name["conv_in"][-1])
        self.assertEqual(by_name["down_0_1"][2], (1, 96, 6, 66, 66))
        self.assertEqual(by_name["resample_96"][2], (4, 96, 65, 65))
        self.assertEqual(by_name["time_conv_192"][2], (1, 192, 5, 16, 16))
        self.assertEqual(by_name["wdec_up_192_L0"][2], (4, 192, 64, 64))
        self.assertEqual(by_name["wdec_conv_out"][2], (1, 96, 6, 66, 66))

    def test_perf_preserves_full_original_matrix_and_defaults(self):
        args = self.parse(["--mode", "perf"])
        self.assertEqual(args.cases, script.ALL_CASES)
        self.assertEqual(args.wan_res, ["480x832", "368x544"])
        self.assertEqual(args.qwen_res, ["1024x1024", "1328x1328"])
        self.assertEqual(len(script.KW_CASES), 29)
        rows = script.selected_model_cases(args)
        self.assertEqual(sum(r[-2].startswith("Wan") for r in rows), 48)
        self.assertEqual(sum(r[-2] == "Qwen" for r in rows), 32)

    def test_explicit_cases_are_not_ignored_in_either_mode(self):
        for mode in ("ci", "perf"):
            args = self.parse(["--mode", mode, "-c", "res_96_L0", "3d_bias"])
            self.assertEqual(args.cases, ["res_96_L0", "3d_bias"])
            self.assertEqual(
                {r[0] for r in script.selected_model_cases(args)}, {"res_96_L0"}
            )

    def test_resolution_override_changes_space_not_convolution_semantics(self):
        args = self.parse(["--wan-res", "128x192", "--wan-frames", "17"])
        rows = script.selected_model_cases(args)
        self.assertEqual(len(rows), 6)
        by_name = {r[0]: r for r in rows}
        self.assertEqual(by_name["conv_in"][2], (1, 3, 6, 130, 194))
        self.assertEqual(by_name["conv_in"][3], (96, 3, 3, 3, 3))
        self.assertEqual(by_name["resample_96"][4], {"stride": 2, "padding": 0})
        self.assertEqual(by_name["down_0_1"][5], 16)
        self.assertTrue(args.wan_res_explicit)

    def test_explicit_qwen_resolution_enables_qwen_or_rejects_unused_override(self):
        args = self.parse(["--qwen-res", "64x128"])
        rows = script.selected_model_cases(args)
        self.assertEqual(sum(r[-2] == "Qwen" for r in rows), 16)
        with self.assertRaises(SystemExit):
            self.parse(["-c", "conv_in", "--qwen-res", "64x64"])
        with self.assertRaises(SystemExit):
            self.parse(["-c", "res_96_L0", "--wan-res", "64x64"])

    def test_empty_unknown_or_invalid_arguments_do_not_report_passed(self):
        invalid = [
            ["--dtype"],
            ["--cases"],
            ["--wan-res"],
            ["--qwen-res"],
            ["--wan-frames"],
            ["--cases", "not_a_case"],
            ["--wan-res", "0x64"],
            ["--wan-res", "-8x64"],
            ["--wan-res", "65x64"],
            ["--wan-res", "bad"],
            ["--wan-frames", "1"],
            ["--wan-frames", "6"],
        ]
        for argv in invalid:
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                self.parse(argv)

    def test_help_precedes_backend_loading_and_gpu_detection(self):
        with patch.object(
            script, "_load_backend", side_effect=AssertionError
        ), contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(
            SystemExit
        ) as exc:
            script.main(["--help"])
        self.assertEqual(exc.exception.code, 0)
        self.assertIn("--mode", output.getvalue())

    def test_production_conv_in_has_an_exact_flydsl_model_row(self):
        path = ROOT / "aiter/configs/model_configs/wan21_vae_bf16_tuned_conv3d.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        hits = [
            r
            for r in rows
            if [int(r[k]) for k in ("N", "C", "D", "H", "W", "K")]
            == [1, 3, 6, 370, 546, 96]
            and r["bias"].lower() == "true"
        ]
        self.assertTrue(hits)
        self.assertTrue(all(r["libtype"] == "flydsl" for r in hits))


class RunnerTests(unittest.TestCase):
    def test_two_warmups_then_ten_checked_direct_calls_no_perftest(self):
        calls, checks, synchronizations = [], [], []

        def backend():
            calls.append(len(calls) + 1)
            return calls[-1]

        with patch.object(
            script,
            "run_perftest",
            create=True,
            side_effect=AssertionError("perftest called"),
        ):
            result = script.correctness_runner(
                backend, checks.append, lambda: synchronizations.append(True)
            )
        self.assertEqual(calls, list(range(1, 13)))
        self.assertEqual(checks, list(range(3, 13)))
        self.assertEqual(len(synchronizations), 12)
        self.assertEqual(result["iterations"], 10)
        self.assertFalse(result["failure"])
        self.assertFalse(any("us" in k or "TFLOPS" in k for k in result))
        self.assertGreaterEqual(result["wall ms"], 0)

    def test_every_iteration_is_checked_and_all_failures_are_retained(self):
        calls = []

        def check(out):
            calls.append(out)
            if len(calls) in (1, 4, 10):
                raise AssertionError("one element mismatched")

        result = script.correctness_runner(lambda: object(), check, lambda: None)
        self.assertEqual(len(calls), 10)
        for i in (1, 4, 10):
            self.assertIn(f"iteration {i}:", result["failure"])

    def test_backend_failure_reports_only_completed_checked_iterations(self):
        calls = []

        def backend():
            calls.append(True)
            if len(calls) == 6:
                raise RuntimeError("launch failed")
            return 0

        result = script.correctness_runner(backend, lambda out: None, lambda: None)
        self.assertEqual(result["iterations"], 3)
        self.assertIn("iteration 4: RuntimeError: launch failed", result["failure"])

    def test_default_dispatch_keeps_key6_static1_unfiltered(self):
        observed, tables = [], []
        backend = SimpleNamespace(AITER_CONV3D_DYN_HW=1)

        def fake_case(case, *args, **kwargs):
            observed.append((case, args, kwargs, backend.AITER_CONV3D_DYN_HW))
            return {
                "case": case,
                **script.correctness_runner(lambda: 0, lambda out: None, lambda: None),
            }

        logger = SimpleNamespace(info=lambda *args: None)
        with patch.object(script, "_ci_case", fake_case), patch.object(
            script, "conv_kernels", backend, create=True
        ), patch.object(
            script, "aiter", SimpleNamespace(logger=logger), create=True
        ), patch.object(
            script, "summarize", lambda title, rows: tables.extend(rows)
        ):
            script._run_ci(script.parse_args([]), "fake_bf16")
        self.assertEqual(len(observed), 19)
        self.assertEqual(sum(r["iterations"] for r in tables), 190)
        key_rows = [
            r for r in observed if r[0].startswith(("depth_", "height_", "width_"))
        ]
        self.assertEqual(len(key_rows), 6)
        self.assertTrue(all(r[2]["round_reference"] is False for r in key_rows))
        static = next(r for r in observed if r[0] == "2d_block_rows8_static")
        self.assertEqual(static[3], 0)
        self.assertEqual(backend.AITER_CONV3D_DYN_HW, 1)
        tuned = next(r for r in observed if r[0] == "conv_in")
        self.assertTrue(tuned[2]["require_tuned"])

    def test_case_exception_is_retained_for_final_failure_summary(self):
        with patch.object(
            script, "_ci_case", side_effect=RuntimeError("backend failed")
        ):
            row = script._execute_ci("failure_case")
        self.assertIn("RuntimeError: backend failed", row["failure"])
        self.assertEqual(row["iterations"], 0)
        logger = SimpleNamespace(info=lambda *args: None)
        with patch.object(
            script, "aiter", SimpleNamespace(logger=logger), create=True
        ), patch.object(
            script,
            "pd",
            SimpleNamespace(
                DataFrame=lambda rows: SimpleNamespace(to_markdown=lambda **kw: "rows")
            ),
            create=True,
        ):
            script._FAILED.clear()
            script.summarize("CPU failure", [row])
        self.assertIn("failure_case", script._FAILED[0])


class HostPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _host_arithmetic()

    def test_splitk2_survives_actual_host_resolver_and_grid(self):
        case = next(c for c in script.KW_CASES if c[0] == "2d_splitk2")
        param = _param(self.ns, case)
        self.assertEqual(param.splitk, 2)
        note = self.ns["_assert_host_path"](case[0], param)
        self.assertIn("use_splitk=True", note)
        self.assertEqual(
            self.ns["_resolve_splitk"](
                2, 64 * 64, 96 * 9, 96, None, tile=(32, 32, 1, 2)
            ),
            1,
        )

    def test_all_five_reuse_regressions_reach_the_expected_actual_plans(self):
        for case in script.KW_CASES:
            if case[0] not in script.CI_KW_CASES:
                continue
            with self.subTest(case=case[0]):
                param = _param(self.ns, case)
                note = self.ns["_assert_host_path"](case[0], param)
                self.assertIn("use_splitk=", note)
        case = next(c for c in script.KW_CASES if c[0] == "2d_block_rows8_nhwc")
        param = _param(self.ns, case, static=True)
        self.ns["_assert_host_path"]("2d_block_rows8_static", param)

    def test_spy_observes_real_param_and_restores_compiler_without_cache_changes(self):
        case = next(c for c in script.KW_CASES if c[0] == "2d_splitk2")
        param = _param(self.ns, case)
        seen = []

        def compiler(param):
            seen.append(param)
            return "compiled"

        backend = SimpleNamespace(compile_conv3d_implicit=compiler, cache={"old": 1})
        with patch.object(script, "conv_kernels", backend, create=True), patch.object(
            script, "_assert_host_path", self.ns["_assert_host_path"]
        ):
            with script._capture_host_path(case[0]) as notes:
                self.assertEqual(backend.compile_conv3d_implicit(param), "compiled")
            self.assertIn("use_splitk=True", notes[0])
            self.assertIs(backend.compile_conv3d_implicit, compiler)
            param.splitk = 1
            with self.assertRaises(AssertionError), script._capture_host_path(case[0]):
                backend.compile_conv3d_implicit(param)
            self.assertIs(backend.compile_conv3d_implicit, compiler)
        self.assertEqual(seen, [param])
        self.assertEqual(backend.cache, {"old": 1})

    def test_tuned_spy_rejects_borrowed_or_missing_rows_and_restores_hooks(self):
        logs, compiles = [], []
        backend = SimpleNamespace(
            _log_tuned_lookup=lambda *args, **kw: logs.append(args),
            compile_conv3d_implicit=lambda param: compiles.append(param),
        )
        saved_log, saved_compile = (
            backend._log_tuned_lookup,
            backend.compile_conv3d_implicit,
        )
        hit = ((192, 96, 4, 1), 1, 1, 1)
        with patch.object(script, "conv_kernels", backend, create=True):
            with script._capture_tuned_path(True) as notes:
                backend._log_tuned_lookup({}, ("gfx950", 256), (), hit)
                backend.compile_conv3d_implicit(SimpleNamespace(tile=hit[0]))
            self.assertIn("exact tuned hit", notes[0])
            for result, borrowed in ((None, False), (hit, True)):
                with self.assertRaises(AssertionError), script._capture_tuned_path(
                    True
                ):
                    backend._log_tuned_lookup(
                        {}, ("gfx950", 256), (), result, borrowed=borrowed
                    )
                self.assertIs(backend._log_tuned_lookup, saved_log)
                self.assertIs(backend.compile_conv3d_implicit, saved_compile)
        self.assertEqual(len(logs), 1)
        self.assertEqual(len(compiles), 1)


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Mask:
    def __init__(self, values):
        self.values = values

    def all(self):
        return _Scalar(all(self.values))

    def sum(self):
        return _Scalar(sum(self.values))

    def __invert__(self):
        return _Mask([not v for v in self.values])


class _Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values),)

    def to(self, dtype):
        return self

    def numel(self):
        return len(self.values)


class CompareTests(unittest.TestCase):
    def test_compare_keeps_original_checkallclose_argument_order(self):
        ref, out = _Tensor([1.0]), _Tensor([1.0404])
        observed = []

        def isclose(first, second, **tol):
            observed.append((first, second, tol))
            return _Mask(
                [
                    abs(a - b) <= tol["atol"] + tol["rtol"] * abs(b)
                    for a, b in zip(first.values, second.values)
                ]
            )

        fake_torch = SimpleNamespace(
            isfinite=lambda t: _Mask([math.isfinite(v) for v in t.values]),
            isclose=isclose,
        )
        with patch.object(script, "torch", fake_torch, create=True), patch.object(
            script, "dtypes", SimpleNamespace(fp32="fp32"), create=True
        ):
            self.assertEqual(script._compare(ref, out, "direction"), 0)
        self.assertIs(observed[0][0], ref)
        self.assertIs(observed[0][1], out)
        self.assertEqual(observed[0][2], script.TOL)

    def test_single_element_and_nonfinite_are_strict_failures(self):
        fake_torch = SimpleNamespace(
            isfinite=lambda t: _Mask([math.isfinite(v) for v in t.values]),
            isclose=lambda ref, out, rtol, atol: _Mask(
                [
                    abs(a - b) <= atol + rtol * abs(b)
                    for a, b in zip(ref.values, out.values)
                ]
            ),
        )
        with patch.object(script, "torch", fake_torch, create=True), patch.object(
            script, "dtypes", SimpleNamespace(fp32="fp32"), create=True
        ):
            ref = _Tensor([1.0] * 1000)
            script._check_output(ref, _Tensor([1.01] * 1000), "close")
            for value in (2.0, float("nan"), float("inf"), -float("inf")):
                with self.subTest(value=value), self.assertRaises(AssertionError):
                    script._check_output(ref, _Tensor([1.0] * 999 + [value]), "bad")
            with self.assertRaisesRegex(AssertionError, "non-finite reference"):
                script._check_output(_Tensor([float("inf")]), _Tensor([1.0]), "bad_ref")


if __name__ == "__main__":
    unittest.main(verbosity=2)
