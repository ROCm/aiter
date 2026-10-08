# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU source-level regressions for the FMoE token-bucket contract.

GPU-only imports are intentionally outside this test boundary.  The helpers
below compile the exact argparse, CSV and preprocessing methods from this tree;
only gfx, CU count and CUDA device count are fixed review seams.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "aiter/fused_moe.py"
GROUPED_RUNTIME = ROOT / "aiter/ops/flydsl/grouped_moe_gfx1250.py"
DP_SHARED = ROOT / "aiter/fused_moe_dp_shared_expert.py"
BASE_TUNER = ROOT / "aiter/utility/base_tuner.py"
FMOE_TUNER = ROOT / "csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py"
KEYS = [
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
]


def _class(source, name):
    return next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == name
    )


def _methods(source, class_name, required=(), optional=()):
    found = {
        n.name: n
        for n in _class(source, class_name).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = set(required) - found.keys()
    if missing:
        raise AssertionError(f"missing {class_name} methods: {sorted(missing)}")
    return [found[name] for name in (*required, *optional) if name in found]


def _assignment(source, class_name, name):
    for node in _class(source, class_name).body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            return node
    raise AssertionError(f"missing {class_name}.{name}")


def _top_level(source, names):
    result = []
    for node in ast.parse(source).body:
        node_names = {node.name} if isinstance(node, ast.FunctionDef) else set()
        if isinstance(node, ast.Assign):
            node_names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        if node_names & set(names):
            result.append(node)
    if (
        set(names)
        - {n.name for n in result if isinstance(n, ast.FunctionDef)}
        - {
            t.id
            for n in result
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Name)
        }
    ):
        raise AssertionError(f"missing top-level nodes: {names}")
    return result


def _compile(nodes, path, namespace=None):
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {} if namespace is None else namespace
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace


def _bucket(path, names, function="get_padded_M"):
    source = path.read_text(encoding="utf-8")
    return _compile(_top_level(source, names), path)[function]


def _normalizer(runtime_bucket):
    source = FMOE_TUNER.read_text(encoding="utf-8")
    cls = ast.ClassDef(
        name="FmoeTuner",
        bases=[],
        keywords=[],
        decorator_list=[],
        type_params=[],
        body=_methods(source, "FmoeTuner", required=("_normalize_tuning_tokens",)),
    )
    return _compile([cls], FMOE_TUNER, {"get_padded_M": runtime_bucket})[
        "FmoeTuner"
    ]._normalize_tuning_tokens


class _Dtypes:
    @staticmethod
    def str2bool(value):
        value = str(value).lower()
        if value in {"1", "true", "yes"}:
            return True
        if value in {"0", "false", "no"}:
            return False
        raise argparse.ArgumentTypeError(value)


def _tuner_classes(runtime_bucket):
    base = BASE_TUNER.read_text(encoding="utf-8")
    fmoe = FMOE_TUNER.read_text(encoding="utf-8")
    base_names = (
        "__init__",
        "get_arg_defaults",
        "_setup_common_arguments",
        "parse_args",
        "update_config_files",
        "get_untuned_gemm_list",
        "get_out_file",
        "get_tuned_gemm_list",
        "get_retune_gemm_list",
    )
    base_body = [_assignment(base, "TunerCommon", "ARG_DEFAULTS")]
    base_body += _methods(base, "TunerCommon", required=base_names)
    base_body += ast.parse(
        "def get_cu_num(self):\n    return 80\n"
        "def get_gfx(self):\n    return 'gfx942'\n"
    ).body
    base_cls = ast.ClassDef(
        name="ReviewBase",
        bases=[],
        keywords=[],
        decorator_list=[],
        type_params=[],
        body=base_body,
    )
    # Optional selection lets this integration class run against unpatched main
    # as a negative control instead of failing during loader construction.
    fmoe_body = _methods(
        fmoe,
        "FmoeTuner",
        required=("_setup_specific_arguments", "parse_args", "pre_process"),
        optional=("_normalize_tuning_tokens", "get_untuned_gemm_list"),
    )
    fmoe_cls = ast.ClassDef(
        name="ReviewFmoe",
        bases=[ast.Name(id="ReviewBase", ctx=ast.Load())],
        keywords=[],
        decorator_list=[],
        type_params=[],
        body=fmoe_body,
    )
    grouped_cls = ast.ClassDef(
        name="ReviewGrouped",
        bases=[ast.Name(id="ReviewFmoe", ctx=ast.Load())],
        keywords=[],
        decorator_list=[],
        type_params=[],
        body=[ast.Pass()],
    )
    namespace = {
        "Any": Any,
        "ClassVar": ClassVar,
        "argparse": argparse,
        "pd": pd,
        "os": __import__("os"),
        "dtypes": _Dtypes(),
        "torch": SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2)),
        "get_padded_M": runtime_bucket,
        "get_gfx_runtime": lambda: "gfx942",
        "gfx_from_cu_num": lambda cu: "gfx942" if int(cu) == 80 else "gfx950",
        "Mxfp4FlydslTuner": type("Mxfp4FlydslTuner", (), {}),
    }
    namespace = _compile(
        _top_level(base, {"_read_csv"}) + [base_cls, fmoe_cls, grouped_cls],
        FMOE_TUNER,
        namespace,
    )
    return namespace["ReviewFmoe"], namespace["ReviewGrouped"]


@contextlib.contextmanager
def _argv(args):
    previous, sys.argv = sys.argv, ["gemm_moe_tune.py", *args]
    try:
        yield
    finally:
        sys.argv = previous


def _row(token, **changes):
    row = {
        "gfx": "gfx942",
        "cu_num": 80,
        "token": token,
        "model_dim": 6144,
        "inter_dim": 256,
        "expert": 257,
        "topk": 9,
        "act_type": "ActivationType.Silu",
        "dtype": "torch.bfloat16",
        "q_dtype_a": "torch.float8_e4m3fnuz",
        "q_dtype_w": "torch.float8_e4m3fnuz",
        "q_type": "QuantType.per_1x128",
        "use_g1u1": 1,
        "doweight_stage1": 0,
    }
    row.update(changes)
    return row


def _write(path, rows):
    pd.DataFrame(rows, columns=KEYS).to_csv(path, index=False)


def _tokens(frame):
    return [int(value) for value in frame["token"]]


class TestFmoeTunerTokenBuckets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime_bucket = staticmethod(
            _bucket(RUNTIME, {"_PADDED_M_TIERS", "nextPow2", "get_padded_M"})
        )
        cls.grouped_bucket = staticmethod(
            _bucket(
                GROUPED_RUNTIME,
                {"_PADDED_M_TIERS", "_next_pow2", "_get_padded_m"},
                "_get_padded_m",
            )
        )
        cls.dp_bucket = staticmethod(_bucket(DP_SHARED, {"nextPow2", "get_padded_M"}))
        cls.normalize = staticmethod(_normalizer(cls.runtime_bucket))

    def test_runtime_bucket_boundaries_match_lookup_contract(self):
        expected = {
            0: 1,
            1: 1,
            2: 2,
            3: 4,
            16: 16,
            17: 32,
            96: 128,
            32767: 32768,
            32768: 32768,
            65536: 32768,
            131071: 32768,
            131072: 131072,
            200000: 131072,
        }
        for raw, bucket in expected.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.runtime_bucket(raw), bucket)
                self.assertEqual(self.grouped_bucket(raw), bucket)

    def test_normalizer_keeps_newest_canonical_row_and_distinct_keys(self):
        rows = pd.DataFrame(
            {"token": [128, 64, 96, 128], "model_dim": [6144, 6144, 6144, 7168]}
        )
        original = rows.copy(deep=True)
        result = self.normalize(rows, "runtime")
        self.assertEqual(
            result[["token", "model_dim"]].values.tolist(),
            [[64, 6144], [128, 6144], [128, 7168]],
        )
        pd.testing.assert_frame_equal(rows, original)

    def test_dp_shared_contract_is_passthrough_and_has_its_own_buckets(self):
        rows = pd.DataFrame({"token": [3, 16, 96, 1024]})
        self.assertIs(self.normalize(rows, "dp_shared"), rows)
        for raw, bucket in {1: 16, 16: 16, 17: 32, 1023: 1024, 1024: 1024}.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.dp_bucket(raw), bucket)


class TestFmoeTunerTokenBucketIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runtime = _bucket(RUNTIME, {"_PADDED_M_TIERS", "nextPow2", "get_padded_M"})
        cls.tuner_class, cls.grouped_class = _tuner_classes(runtime)

    def _run(self, untuned, tuned, extra=(), tuner_class=None):
        tuner = (tuner_class or self.tuner_class)("review", KEYS, [], "CPU test")
        with _argv(["-i", str(untuned), "-o", str(tuned), *extra]):
            args = tuner.parse_args()
        tuner.pre_process(args)
        return tuner, args

    def test_runtime_default_and_last_use_canonical_newest_rows(self):
        cases = (
            ("default_tuned_filter", [96, 64], [], [64]),
            ("last_collision", [128, 64, 96], ["--last"], [128]),
            ("last_canonical_duplicate", [128, 64, 128], ["--last"], [128]),
            ("last_raw_duplicate", [96, 64, 96], ["--last"], [128]),
        )
        for name, raw, flags, expected in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                untuned, tuned = Path(tmp) / "in.csv", Path(tmp) / "out.csv"
                _write(untuned, [_row(token) for token in raw])
                _write(tuned, [_row(128)] if name == "default_tuned_filter" else [])
                result, args = self._run(untuned, tuned, flags)
                self.assertEqual(args.token_bucket_contract, "runtime")
                self.assertEqual(_tokens(result.untunedf), expected)

    def test_dp_last_passthrough_and_argparse_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            untuned, tuned = Path(tmp) / "in.csv", Path(tmp) / "out.csv"
            _write(untuned, [_row(token) for token in [16, 64, 16]])
            _write(tuned, [])
            result, args = self._run(
                untuned, tuned, ["--last", "--token-bucket-contract", "dp_shared"]
            )
            self.assertEqual(args.token_bucket_contract, "dp_shared")
            self.assertEqual(_tokens(result.untunedf), [16])

        tuner = self.tuner_class("review", KEYS, [], "CPU test")
        stderr = io.StringIO()
        with (
            _argv(["--token-bucket-contract", "unknown"]),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit),
        ):
            tuner.parse_args()
        self.assertIn("invalid choice", stderr.getvalue())

    def test_all_same_and_different_files_canonicalize_without_disk_writes(self):
        for same_file in (True, False):
            with (
                self.subTest(same_file=same_file),
                tempfile.TemporaryDirectory() as tmp,
            ):
                untuned = Path(tmp) / "in.csv"
                tuned = untuned if same_file else Path(tmp) / "out.csv"
                if same_file:
                    _write(
                        untuned,
                        [_row(96), _row(96, gfx="gfx950", cu_num=304, model_dim=7168)],
                    )
                else:
                    _write(untuned, [_row(96)])
                    _write(
                        tuned,
                        [_row(128), _row(96, gfx="gfx950", cu_num=304, model_dim=7168)],
                    )
                before = {path: path.read_bytes() for path in {untuned, tuned}}
                result, _ = self._run(untuned, tuned, ["--all"])
                self.assertEqual(_tokens(result.untunedf), [128])
                self.assertEqual(result.tunedf["gfx"].tolist(), ["gfx950"])
                self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_grouped_inherits_real_preprocess_and_dp_command_has_flag(self):
        grouped = _class(FMOE_TUNER.read_text(encoding="utf-8"), "GroupedFmoeTuner")
        self.assertEqual(
            [base.id for base in grouped.bases if isinstance(base, ast.Name)],
            ["FmoeTuner"],
        )
        grouped_methods = {
            node.name for node in grouped.body if isinstance(node, ast.FunctionDef)
        }
        self.assertNotIn("pre_process", grouped_methods)
        self.assertNotIn("get_untuned_gemm_list", grouped_methods)
        self.assertIs(self.grouped_class.pre_process, self.tuner_class.pre_process)
        with tempfile.TemporaryDirectory() as tmp:
            untuned, tuned = Path(tmp) / "in.csv", Path(tmp) / "out.csv"
            _write(untuned, [_row(96), _row(64)])
            _write(tuned, [_row(128)])
            result, _ = self._run(untuned, tuned, tuner_class=self.grouped_class)
            self.assertEqual(_tokens(result.untunedf), [64])

        source = DP_SHARED.read_text(encoding="utf-8")
        outer = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == "get_2stage_cfgs"
        )
        main = next(
            n
            for n in ast.walk(outer)
            if isinstance(n, ast.FunctionDef) and n.name == "MainFunc"
        )
        self.assertIn(
            "--last --token-bucket-contract dp_shared",
            ast.get_source_segment(source, main),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
