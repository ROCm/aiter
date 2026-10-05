#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
test_classify_test.py

Unit test suite for .github/scripts/classify_test.py.
Validates AST classification logic against synthetic edge cases and
real repository test files from ROCm/aiter.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add current directory to sys.path to import classify_test
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from classify_test import classify_test_file  # noqa: E402


class TestClassifyTestSynthetic(unittest.TestCase):
    """Synthetic unit tests for all expected classification cases."""

    def _classify_code(self, code: str) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        try:
            return classify_test_file(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_pure_pytest_single_function(self):
        code = """
import pytest

def test_example():
    assert True
"""
        self.assertEqual(self._classify_code(code), "PYTEST")

    def test_pure_pytest_multiple_functions(self):
        code = """
import pytest

def test_alpha():
    assert 1 == 1

def test_beta():
    assert 2 == 2
"""
        self.assertEqual(self._classify_code(code), "PYTEST")

    def test_pure_pytest_parametrized(self):
        code = """
import pytest

@pytest.mark.parametrize("dim", [128, 256])
def test_dims(dim):
    assert dim > 0
"""
        self.assertEqual(self._classify_code(code), "PYTEST")

    def test_pure_pytest_with_fixtures(self):
        code = """
import pytest

@pytest.fixture
def sample_tensor():
    return [1, 2, 3]

def test_with_fixture(sample_tensor, monkeypatch, tmp_path):
    assert len(sample_tensor) == 3
"""
        self.assertEqual(self._classify_code(code), "PYTEST")

    def test_legacy_script_with_main(self):
        code = """
def run_benchmark():
    pass

if __name__ == "__main__":
    run_benchmark()
"""
        self.assertEqual(self._classify_code(code), "PYTHON")

    def test_legacy_positional_test_function(self):
        # A file with positional arguments not parametrized -> Must NOT be PYTEST!
        code = """
def test_gemm(dtype, m, n, k):
    pass
"""
        self.assertEqual(self._classify_code(code), "PYTHON")

    def test_file_with_imports_and_main_script(self):
        code = """
import argparse
import sys

def test_kernel(dtype, m, n):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    test_kernel("fp8", 128, 128)
"""
        self.assertEqual(self._classify_code(code), "PYTHON")

    def test_file_with_top_level_parse_args(self):
        code = """
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()
"""
        self.assertEqual(self._classify_code(code), "PYTHON")

    def test_syntax_error(self):
        code = """
def broken_syntax(
"""
        self.assertEqual(self._classify_code(code), "PYTHON")

    def test_empty_file(self):
        code = ""
        self.assertEqual(self._classify_code(code), "PYTHON")

    def test_file_with_only_pytest_main_in_main(self):
        code = """
import pytest

def test_feature():
    assert True

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
"""
        self.assertEqual(self._classify_code(code), "PYTEST")

    def test_canary_assertion_failure_execution(self):
        # Regression proof for Issue #5892: a failing assertion in a pure pytest file
        # must be classified as PYTEST and fail under pytest execution with exit code 1.
        import subprocess

        code = """
def test_canary():
    assert False, "deliberate canary failure"
"""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        try:
            # 1. Must be classified as PYTEST
            self.assertEqual(classify_test_file(tmp_path), "PYTEST")
            # 2. Must fail loudly with exit code 1 under pytest
            res = subprocess.run([sys.executable, "-m", "pytest", tmp_path, "-q"], capture_output=True, text=True)
            self.assertEqual(res.returncode, 1)
            self.assertIn("AssertionError", res.stdout + res.stderr)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestClassifyTestRepositoryFiles(unittest.TestCase):
    """Integration tests verifying classification of actual ROCm/aiter files."""

    def test_all_six_cited_issue_5892_files(self):
        cited_files = [
            "op_tests/test_moe_mxfp8_passthrough.py",
            "op_tests/test_moe_local_expert_ids.py",
            "op_tests/test_gemm_a8w8_blockscale_fallback.py",
            "op_tests/test_graph_alloc.py",
            "op_tests/test_gemm_a8w8_bpreshuffle_pad_k.py",
            "op_tests/test_gemm_group32_interface.py",
        ]
        for rel_path in cited_files:
            abs_path = os.path.join(REPO_ROOT, rel_path)
            self.assertTrue(os.path.exists(abs_path), f"File missing: {rel_path}")
            self.assertEqual(
                classify_test_file(abs_path),
                "PYTEST",
                f"Cited file {rel_path} must be classified as PYTEST",
            )

    def test_nested_flydsl_files(self):
        flydsl_pytest = [
            "op_tests/flydsl_tests/test_moe_mxfp4_inter_dim_align.py",
            "op_tests/flydsl_tests/test_flydsl_moe.py",
            "op_tests/flydsl_tests/test_silu_and_mul_fq.py",
        ]
        for rel_path in flydsl_pytest:
            abs_path = os.path.join(REPO_ROOT, rel_path)
            self.assertTrue(os.path.exists(abs_path), f"File missing: {rel_path}")
            self.assertEqual(
                classify_test_file(abs_path),
                "PYTEST",
                f"FlyDSL file {rel_path} must be classified as PYTEST",
            )

        flydsl_scripts = [
            "op_tests/flydsl_tests/test_flydsl_grouped_gemm.py",
            "op_tests/flydsl_tests/test_gemm_a16w16.py",
        ]
        for rel_path in flydsl_scripts:
            abs_path = os.path.join(REPO_ROOT, rel_path)
            self.assertTrue(os.path.exists(abs_path), f"File missing: {rel_path}")
            self.assertEqual(
                classify_test_file(abs_path),
                "PYTHON",
                f"FlyDSL script {rel_path} must be classified as PYTHON",
            )

    def test_legacy_benchmark_scripts(self):
        legacy_benchmarks = [
            "op_tests/test_gemm_a8w8.py",
            "op_tests/test_activation.py",
            "op_tests/test_aiter_add.py",
            "op_tests/test_mla.py",
            "op_tests/test_gemm_a6w4.py",
            "op_tests/test_gemm_a4w6.py",
        ]
        for rel_path in legacy_benchmarks:
            abs_path = os.path.join(REPO_ROOT, rel_path)
            self.assertTrue(os.path.exists(abs_path), f"File missing: {rel_path}")
            self.assertEqual(
                classify_test_file(abs_path),
                "PYTHON",
                f"Legacy benchmark {rel_path} must be classified as PYTHON",
            )


class TestDiscoveryLogic(unittest.TestCase):
    """Integration test verifying targeted discovery includes FlyDSL and isolates specialized suites."""

    def test_targeted_discovery_inventory(self):
        import glob
        test_dir = os.path.join(REPO_ROOT, "op_tests")

        # 1. Root tests
        root_tests = sorted(glob.glob(os.path.join(test_dir, "test_*.py")))
        self.assertEqual(len(root_tests), 162)

        # 2. FlyDSL tests
        flydsl_tests = sorted(glob.glob(os.path.join(test_dir, "flydsl_tests", "test_*.py")))
        self.assertEqual(len(flydsl_tests), 5)
        # Verify PR #5232 test exists in FlyDSL
        self.assertIn(
            os.path.join(test_dir, "flydsl_tests", "test_moe_mxfp4_inter_dim_align.py"),
            flydsl_tests,
        )

        # 3. Tuning tests explicitly included
        tuning_tests = [
            os.path.join(test_dir, "tuning_tests", "test_csv_validation.py"),
            os.path.join(test_dir, "tuning_tests", "test_config_shape_collision.py"),
            os.path.join(test_dir, "tuning_tests", "test_mixed_mxfp_tuning.py"),
        ]
        for t in tuning_tests:
            self.assertTrue(os.path.exists(t), f"Tuning test missing: {t}")

        # Total combined inventory for standard shards must be exactly 170
        all_discovered = sorted(set(root_tests + flydsl_tests + tuning_tests))
        self.assertEqual(len(all_discovered), 170)


if __name__ == "__main__":
    unittest.main(verbosity=2)
