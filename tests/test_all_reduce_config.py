# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ast
import importlib.util
import os
import pathlib
import subprocess
import sys
import unittest
import warnings
from types import SimpleNamespace

_REPO = pathlib.Path(__file__).resolve().parents[1]
_COMMUNICATORS = _REPO / "aiter" / "dist" / "device_communicators"


def _load_all_reduce_config():
    # Load the configuration helper without importing AITER's GPU operators.
    path = _COMMUNICATORS / "_all_reduce_config.py"
    spec = importlib.util.spec_from_file_location("_test_all_reduce_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AllReduceConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = _load_all_reduce_config()

    def test_unset_uses_default_without_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(self.config.get_ar_1stage_max_bytes({}), -1)
        self.assertEqual(caught, [])

    def test_explicit_bytes_preserve_values_without_warning(self):
        for value in (0, 1, 512, 524288, -1, -7):
            with self.subTest(value=value), warnings.catch_warnings(
                record=True
            ) as caught:
                warnings.simplefilter("always")
                self.assertEqual(
                    self.config.get_ar_1stage_max_bytes(
                        {"AITER_AR_1STAGE_MAX_BYTES": str(value)}
                    ),
                    value,
                )
                self.assertEqual(caught, [])

    def test_legacy_alias_preserves_bytes_and_warns(self):
        for value in (0, 1, 512, 524288, -1, -7):
            with self.subTest(value=value), self.assertWarns(FutureWarning) as caught:
                self.assertEqual(
                    self.config.get_ar_1stage_max_bytes(
                        {"AITER_AR_1STAGE_MAX_KB": str(value)}
                    ),
                    value,
                )
            message = str(caught.warning)
            self.assertIn("AITER_AR_1STAGE_MAX_KB", message)
            self.assertIn("AITER_AR_1STAGE_MAX_BYTES", message)
            self.assertIn("bytes", message)

    def test_explicit_bytes_override_legacy_without_warning(self):
        for value in (0, 1, 512, -1, -7):
            for legacy in ("524288", "invalid"):
                with self.subTest(value=value, legacy=legacy):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        self.assertEqual(
                            self.config.get_ar_1stage_max_bytes(
                                {
                                    "AITER_AR_1STAGE_MAX_BYTES": str(value),
                                    "AITER_AR_1STAGE_MAX_KB": legacy,
                                }
                            ),
                            value,
                        )
                    self.assertEqual(caught, [])

    def test_malformed_values_raise_value_error(self):
        for name in ("AITER_AR_1STAGE_MAX_BYTES", "AITER_AR_1STAGE_MAX_KB"):
            for value in ("", "1.5", "invalid"):
                with (
                    self.subTest(name=name, value=value),
                    warnings.catch_warnings(),
                    self.assertRaises(ValueError),
                ):
                    warnings.simplefilter("ignore", FutureWarning)
                    self.config.get_ar_1stage_max_bytes({name: value})

    def test_malformed_explicit_bytes_do_not_fall_back_to_legacy(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaises(ValueError):
                self.config.get_ar_1stage_max_bytes(
                    {"AITER_AR_1STAGE_MAX_BYTES": "", "AITER_AR_1STAGE_MAX_KB": "512"}
                )
        self.assertEqual(caught, [])

    def test_explicit_512_kib_limit_at_bf16_row_boundary(self):
        limit = self.config.get_ar_1stage_max_bytes(
            {"AITER_AR_1STAGE_MAX_BYTES": "524288"}
        )
        row_bytes = 7168 * 2
        self.assertLessEqual(36 * row_bytes, limit)
        self.assertGreater(37 * row_bytes, limit)

    def test_communicator_byte_limit_wiring_without_gpu_imports(self):
        # Evaluate the real setting and dispatch expressions without importing
        # GPU operators. A parser-only test cannot catch disconnected call sites.
        path = _COMMUNICATORS / "communicator_cuda.py"
        tree = ast.parse(path.read_text())
        self.assertTrue(
            any(
                isinstance(node, ast.ImportFrom)
                and node.module == "_all_reduce_config"
                and any(alias.name == "get_ar_1stage_max_bytes" for alias in node.names)
                for node in tree.body
            )
        )
        communicator = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "CudaCommunicator"
        )

        def assignment_expression(nodes, name):
            assignments = [
                node.value
                for node in nodes
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == name
                    for target in node.targets
                )
            ]
            self.assertEqual(len(assignments), 1, name)
            return compile(ast.Expression(assignments[0]), str(path), "eval")

        setting = assignment_expression(communicator.body, "_ar_1stage_max_bytes")
        settings = (
            ({}, -1),
            ({"AITER_AR_1STAGE_MAX_KB": "512"}, 512),
            ({"AITER_AR_1STAGE_MAX_BYTES": "0"}, 0),
            ({"AITER_AR_1STAGE_MAX_BYTES": "-7"}, -7),
            (
                {
                    "AITER_AR_1STAGE_MAX_BYTES": "524288",
                    "AITER_AR_1STAGE_MAX_KB": "512",
                },
                524288,
            ),
        )
        methods = {
            node.name: node
            for node in communicator.body
            if isinstance(node, ast.FunctionDef)
        }
        for name in (
            "fused_allreduce_rmsnorm",
            "fused_allreduce_rmsnorm_quant",
            "fused_allreduce_rmsnorm_quant_per_group",
        ):
            limit_expression = assignment_expression(
                ast.walk(methods[name]), "total_bytes_limit"
            )
            stage_expression = assignment_expression(
                ast.walk(methods[name]), "use_1stage"
            )
            for environ, expected in settings:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    selected = eval(
                        setting,
                        {
                            "get_ar_1stage_max_bytes": self.config.get_ar_1stage_max_bytes,
                            "os": SimpleNamespace(environ=environ),
                        },
                    )
                self.assertEqual(selected, expected)
                for world_size in (2, 4, 8):
                    default = (
                        128 * 7168 * 2 // world_size
                        if name == "fused_allreduce_rmsnorm"
                        else 128 * 1024
                    )
                    expected_limit = expected if expected >= 0 else default
                    for override in (None, False, True):
                        scope = {
                            "self": SimpleNamespace(
                                world_size=world_size,
                                _ar_1stage_max_bytes=selected,
                                _ar_1stage_override=override,
                            )
                        }
                        limit = eval(limit_expression, {}, scope)
                        self.assertEqual(limit, expected_limit)
                        for size in (
                            max(0, expected_limit - 1),
                            expected_limit,
                            expected_limit + 1,
                        ):
                            with self.subTest(
                                method=name,
                                setting=expected,
                                world_size=world_size,
                                override=override,
                                size=size,
                            ):
                                scope.update(total_bytes=size, total_bytes_limit=limit)
                                self.assertEqual(
                                    eval(stage_expression, {}, scope),
                                    override if override is not None else size <= limit,
                                )

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) is not None for name in ("torch", "triton")),
        "requires torch and triton for the full AITER import",
    )
    def test_communicator_reads_explicit_bytes_before_import(self):
        import torch

        if torch.version.hip is None:
            self.skipTest("requires ROCm PyTorch for the full AITER import")
        env = os.environ.copy()
        for name in (
            "GPU_ARCHS",
            "AITER_AR_1STAGE_MAX_BYTES",
            "AITER_AR_1STAGE_MAX_KB",
        ):
            env.pop(name, None)
        # Match test_import_without_gpu.py while preserving ROCm tool paths.
        env["ROCR_VISIBLE_DEVICES"] = "-1"
        env["AITER_AR_1STAGE_MAX_BYTES"] = "524288"
        env["AITER_AR_1STAGE_MAX_KB"] = "512"
        code = (
            "from aiter.dist.device_communicators.communicator_cuda import "
            "CudaCommunicator; print(CudaCommunicator._ar_1stage_max_bytes)"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=_REPO,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        self.assertEqual(result.stdout.split()[-1], "524288")


if __name__ == "__main__":
    unittest.main()
