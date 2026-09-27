# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for Issue #4071: Grouped MoE build ignores GPU_ARCHS.

These tests focus on job enumeration, architecture parsing, and filtering boundaries.
They do not require AMD GPU hardware, ROCm compiler drivers, or FlyDSL binaries.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from importlib.machinery import ModuleSpec
from unittest.mock import MagicMock, patch


# --- Hardware / Compiler Agnostic Test Bootstrap ---
class _MockFlyDslLoader:
    def __init__(self, name: str):
        self.name = name

    def create_module(self, spec: ModuleSpec):
        mod = MagicMock()
        mod.__name__ = self.name
        mod.__version__ = "0.2.4"
        mod.__path__ = []
        return mod

    def exec_module(self, module):
        pass


class _MockFlyDslFinder:
    def find_spec(self, fullname: str, path, target=None):
        if fullname == "flydsl" or fullname.startswith("flydsl."):
            return ModuleSpec(fullname, _MockFlyDslLoader(fullname), is_package=True)
        return None


if "flydsl" not in sys.modules:
    try:
        import flydsl  # noqa: F401
    except ImportError:
        sys.meta_path.insert(0, _MockFlyDslFinder())

os.environ.setdefault("AITER_AOT_IMPORT", "1")
if "ROCM_PATH" not in os.environ and "ROCM_HOME" not in os.environ:
    _mock_rocm = os.path.join(tempfile.gettempdir(), "aiter_mock_rocm")
    os.makedirs(os.path.join(_mock_rocm, ".info"), exist_ok=True)
    with open(os.path.join(_mock_rocm, ".info", "version"), "w") as _f:
        _f.write("6.2.0\n")
    os.environ.setdefault("ROCM_PATH", _mock_rocm)

# Safe imports after environment and module stubbing
from aiter.aot.flydsl.common import (
    OpKind,
    _collect_aot_jobs_for,
)
from aiter.aot.flydsl.grouped_moe import (
    DEFAULT_CSVS,
    _target_gfx_archs,
    parse_csv,
)


class TestTargetGfxArchsHelper(unittest.TestCase):
    """Unit tests for grouped_moe._target_gfx_archs() helper."""

    def test_single_arch(self):
        with patch.dict(os.environ, {"GPU_ARCHS": "gfx950"}, clear=True):
            self.assertEqual(_target_gfx_archs(), {"gfx950"})
        with patch.dict(os.environ, {"GPU_ARCHS": "gfx1250"}, clear=True):
            self.assertEqual(_target_gfx_archs(), {"gfx1250"})

    def test_semicolon_delimited(self):
        with patch.dict(os.environ, {"GPU_ARCHS": "gfx942;gfx950"}, clear=True):
            self.assertEqual(_target_gfx_archs(), {"gfx942", "gfx950"})

    def test_comma_delimited(self):
        with patch.dict(os.environ, {"GPU_ARCHS": "gfx942,gfx950"}, clear=True):
            self.assertEqual(_target_gfx_archs(), {"gfx942", "gfx950"})

    def test_mixed_delimiters(self):
        with patch.dict(os.environ, {"GPU_ARCHS": "gfx942;gfx950,gfx1250"}, clear=True):
            self.assertEqual(_target_gfx_archs(), {"gfx942", "gfx950", "gfx1250"})

    def test_whitespace_handling(self):
        with patch.dict(os.environ, {"GPU_ARCHS": "  gfx950  ;  gfx942  "}, clear=True):
            self.assertEqual(_target_gfx_archs(), {"gfx942", "gfx950"})

    def test_empty_and_whitespace(self):
        with patch.dict(os.environ, {"GPU_ARCHS": ""}, clear=True):
            self.assertIsNone(_target_gfx_archs())
        with patch.dict(os.environ, {"GPU_ARCHS": "   "}, clear=True):
            self.assertIsNone(_target_gfx_archs())
        with patch.dict(os.environ, {"GPU_ARCHS": "; ; ,"}, clear=True):
            self.assertIsNone(_target_gfx_archs())

    def test_native_arch(self):
        with patch.dict(os.environ, {"GPU_ARCHS": "native"}, clear=True):
            self.assertIsNone(_target_gfx_archs())
        with patch.dict(os.environ, {"GPU_ARCHS": "NATIVE"}, clear=True):
            self.assertIsNone(_target_gfx_archs())
        with patch.dict(os.environ, {"GPU_ARCHS": "  native  "}, clear=True):
            self.assertIsNone(_target_gfx_archs())

    def test_env_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(_target_gfx_archs())

    def test_arch_env_not_consulted(self):
        """ARCH env var should NOT be consulted, avoiding host x86_64/arm64 collisions."""
        with patch.dict(os.environ, {"ARCH": "x86_64"}, clear=True):
            self.assertIsNone(_target_gfx_archs())
        with patch.dict(os.environ, {"ARCH": "gfx950"}, clear=True):
            self.assertIsNone(_target_gfx_archs())


class TestSourceCsvAndJobContent(unittest.TestCase):
    """Verify source CSV metadata and job expansion semantics."""

    def setUp(self):
        self.csv_path = DEFAULT_CSVS[0]
        self.assertTrue(
            os.path.isfile(self.csv_path), f"CSV not found: {self.csv_path}"
        )

    def test_source_csv_rows_and_architecture(self):
        """Source CSV must contain exactly 113 rows and all rows must target gfx1250."""
        with open(self.csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 113)
        self.assertTrue(all(r.get("gfx", "").strip() == "gfx1250" for r in rows))

    def test_unrestricted_parsing_structure_and_variants(self):
        """Unrestricted parsing must produce 222 jobs: 111 contiguous=False and 111 contiguous=True."""
        with patch.dict(os.environ, {}, clear=True):
            jobs = parse_csv(self.csv_path)
        self.assertEqual(len(jobs), 222)
        self.assertTrue(all(j.get("gfx") == "gfx1250" for j in jobs))

        contig_false = sum(1 for j in jobs if j.get("grouped_contiguous_m") is False)
        contig_true = sum(1 for j in jobs if j.get("grouped_contiguous_m") is True)
        self.assertEqual(contig_false, 111)
        self.assertEqual(contig_true, 111)


class TestGroupedMoeArchFilter(unittest.TestCase):
    """Verify that grouped-MoE AOT job enumeration strictly respects GPU_ARCHS."""

    def setUp(self):
        self.csv_path = DEFAULT_CSVS[0]
        self.assertTrue(
            os.path.isfile(self.csv_path), f"CSV not found: {self.csv_path}"
        )

    def test_single_non_matching_arch_drops_all_jobs(self):
        """When GPU_ARCHS=gfx950, zero gfx1250 grouped-MoE jobs must be generated."""
        with patch.dict(os.environ, {"GPU_ARCHS": "gfx950"}, clear=True):
            jobs = parse_csv(self.csv_path)
            self.assertEqual(len(jobs), 0)

    def test_matching_arch_retains_all_jobs(self):
        """When GPU_ARCHS=gfx1250, all 222 gfx1250 grouped-MoE jobs must be retained."""
        with patch.dict(os.environ, {"GPU_ARCHS": "gfx1250"}, clear=True):
            jobs = parse_csv(self.csv_path)
            self.assertEqual(len(jobs), 222)
            self.assertTrue(all(j.get("gfx") == "gfx1250" for j in jobs))

    def test_multi_arch_without_1250_drops_all_jobs(self):
        """When GPU_ARCHS=gfx942;gfx950, zero gfx1250 jobs must be generated."""
        with patch.dict(os.environ, {"GPU_ARCHS": "gfx942;gfx950"}, clear=True):
            jobs = parse_csv(self.csv_path)
            self.assertEqual(len(jobs), 0)

    def test_multi_arch_with_1250_semicolon(self):
        """When GPU_ARCHS=gfx950;gfx1250, gfx1250 jobs must be retained."""
        with patch.dict(os.environ, {"GPU_ARCHS": "gfx950;gfx1250"}, clear=True):
            jobs = parse_csv(self.csv_path)
            self.assertEqual(len(jobs), 222)
            self.assertTrue(all(j.get("gfx") == "gfx1250" for j in jobs))

    def test_comma_delimited_multi_arch(self):
        """Comma-delimited GPU_ARCHS should be handled identically to semicolon."""
        with patch.dict(os.environ, {"GPU_ARCHS": "gfx950,gfx1250"}, clear=True):
            jobs = parse_csv(self.csv_path)
            self.assertEqual(len(jobs), 222)
            self.assertTrue(all(j.get("gfx") == "gfx1250" for j in jobs))

    def test_whitespace_padded_arch_string(self):
        """Surrounding whitespace should be stripped cleanly."""
        with patch.dict(os.environ, {"GPU_ARCHS": "  gfx950  ;  gfx942  "}, clear=True):
            jobs = parse_csv(self.csv_path)
            self.assertEqual(len(jobs), 0)

    def test_unset_arch_builds_default(self):
        """When GPU_ARCHS is unset, default build behavior (222 jobs) is preserved."""
        with patch.dict(os.environ, {}, clear=True):
            jobs = parse_csv(self.csv_path)
            self.assertEqual(len(jobs), 222)
            self.assertTrue(all(j.get("gfx") == "gfx1250" for j in jobs))

    def test_empty_arch_builds_default(self):
        """When GPU_ARCHS is empty, default build behavior (222 jobs) is preserved."""
        with patch.dict(os.environ, {"GPU_ARCHS": ""}, clear=True):
            jobs = parse_csv(self.csv_path)
            self.assertEqual(len(jobs), 222)
            self.assertTrue(all(j.get("gfx") == "gfx1250" for j in jobs))

    def test_whitespace_arch_builds_default(self):
        """When GPU_ARCHS is whitespace, default build behavior (222 jobs) is preserved."""
        with patch.dict(os.environ, {"GPU_ARCHS": "   "}, clear=True):
            jobs = parse_csv(self.csv_path)
            self.assertEqual(len(jobs), 222)
            self.assertTrue(all(j.get("gfx") == "gfx1250" for j in jobs))

    def test_native_arch_builds_default(self):
        """When GPU_ARCHS=native, default build behavior (222 jobs) is preserved."""
        with patch.dict(os.environ, {"GPU_ARCHS": "native"}, clear=True):
            jobs = parse_csv(self.csv_path)
            self.assertEqual(len(jobs), 222)
            self.assertTrue(all(j.get("gfx") == "gfx1250" for j in jobs))

    def test_arch_env_collision_ignored_when_gpu_archs_unset(self):
        """ARCH=x86_64 should NOT suppress GPU compilation when GPU_ARCHS is unset."""
        with patch.dict(os.environ, {"ARCH": "x86_64"}, clear=True):
            jobs = parse_csv(self.csv_path)
            self.assertEqual(len(jobs), 222)
            self.assertTrue(all(j.get("gfx") == "gfx1250" for j in jobs))

    def test_unsupported_arch_drops_all_jobs(self):
        """When GPU_ARCHS targets an unsupported architecture (e.g. gfx1100), 0 jobs must be generated."""
        with patch.dict(os.environ, {"GPU_ARCHS": "gfx1100"}, clear=True):
            jobs = parse_csv(self.csv_path)
            self.assertEqual(len(jobs), 0)


class TestAotCollectionBoundary(unittest.TestCase):
    """Verify the exact run_aot collection boundary: _collect_aot_jobs_for(OpKind.GROUPED_MOE)."""

    def test_collection_boundary_gfx950(self):
        with patch.dict(os.environ, {"GPU_ARCHS": "gfx950"}, clear=True):
            jobs = _collect_aot_jobs_for(OpKind.GROUPED_MOE)
            self.assertEqual(len(jobs), 0)

    def test_collection_boundary_gfx1250(self):
        with patch.dict(os.environ, {"GPU_ARCHS": "gfx1250"}, clear=True):
            jobs = _collect_aot_jobs_for(OpKind.GROUPED_MOE)
            self.assertEqual(len(jobs), 222)
            self.assertTrue(all(j.get("gfx") == "gfx1250" for j in jobs))

    def test_collection_boundary_multi_arch(self):
        with patch.dict(os.environ, {"GPU_ARCHS": "gfx950;gfx1250"}, clear=True):
            jobs = _collect_aot_jobs_for(OpKind.GROUPED_MOE)
            self.assertEqual(len(jobs), 222)
            self.assertTrue(all(j.get("gfx") == "gfx1250" for j in jobs))

    def test_collection_boundary_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            jobs = _collect_aot_jobs_for(OpKind.GROUPED_MOE)
            self.assertEqual(len(jobs), 222)
            self.assertTrue(all(j.get("gfx") == "gfx1250" for j in jobs))


class TestSyntheticCsvFallbacks(unittest.TestCase):
    """Verify fallback and precedence semantics on synthetic CSV configurations."""

    def test_synthetic_csv_explicit_gfx1250_over_cu_num_256(self):
        synthetic_csv = (
            "gfx,cu_num,model_dim,inter_dim,expert,token,tile_m,m_warp,n_warp,topk\n"
            "gfx1250,256,4096,2048,256,64,64,1,4,1\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write(synthetic_csv)
            temp_path = f.name
        self.addCleanup(
            lambda: os.unlink(temp_path) if os.path.exists(temp_path) else None
        )

        with patch.dict(os.environ, {"GPU_ARCHS": "gfx950"}, clear=True):
            jobs_950 = parse_csv(temp_path)
            self.assertEqual(len(jobs_950), 0)

        with patch.dict(os.environ, {"GPU_ARCHS": "gfx1250"}, clear=True):
            jobs_1250 = parse_csv(temp_path)
            self.assertGreater(len(jobs_1250), 0)
            self.assertTrue(all(j.get("gfx") == "gfx1250" for j in jobs_1250))

    def test_synthetic_csv_explicit_gfx950_over_cu_num_256(self):
        synthetic_csv = (
            "gfx,cu_num,model_dim,inter_dim,expert,token,tile_m,m_warp,n_warp,topk\n"
            "gfx950,256,4096,2048,256,64,64,1,4,1\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write(synthetic_csv)
            temp_path = f.name
        self.addCleanup(
            lambda: os.unlink(temp_path) if os.path.exists(temp_path) else None
        )

        with patch.dict(os.environ, {"GPU_ARCHS": "gfx950"}, clear=True):
            jobs_950 = parse_csv(temp_path)
            self.assertGreater(len(jobs_950), 0)
            self.assertTrue(all(j.get("gfx") == "gfx950" for j in jobs_950))

        with patch.dict(os.environ, {"GPU_ARCHS": "gfx1250"}, clear=True):
            jobs_1250 = parse_csv(temp_path)
            self.assertEqual(len(jobs_1250), 0)

    def test_synthetic_csv_missing_gfx_column_defaults_to_gfx1250(self):
        synthetic_csv = (
            "cu_num,model_dim,inter_dim,expert,token,tile_m,m_warp,n_warp,topk\n"
            "256,4096,2048,256,64,64,1,4,1\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write(synthetic_csv)
            temp_path = f.name
        self.addCleanup(
            lambda: os.unlink(temp_path) if os.path.exists(temp_path) else None
        )

        with patch.dict(os.environ, {"GPU_ARCHS": "gfx950"}, clear=True):
            jobs_950 = parse_csv(temp_path)
            self.assertEqual(len(jobs_950), 0)

        with patch.dict(os.environ, {"GPU_ARCHS": "gfx1250"}, clear=True):
            jobs_1250 = parse_csv(temp_path)
            self.assertGreater(len(jobs_1250), 0)
            self.assertTrue(all(j.get("gfx") == "gfx1250" for j in jobs_1250))

        with patch.dict(os.environ, {}, clear=True):
            jobs_unset = parse_csv(temp_path)
            self.assertGreater(len(jobs_unset), 0)
            self.assertTrue(all(j.get("gfx") == "gfx1250" for j in jobs_unset))

    def test_synthetic_csv_empty_gfx_column_defaults_to_gfx1250(self):
        synthetic_csv = (
            "gfx,cu_num,model_dim,inter_dim,expert,token,tile_m,m_warp,n_warp,topk\n"
            "   ,256,4096,2048,256,64,64,1,4,1\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write(synthetic_csv)
            temp_path = f.name
        self.addCleanup(
            lambda: os.unlink(temp_path) if os.path.exists(temp_path) else None
        )

        with patch.dict(os.environ, {"GPU_ARCHS": "gfx950"}, clear=True):
            jobs_950 = parse_csv(temp_path)
            self.assertEqual(len(jobs_950), 0)

        with patch.dict(os.environ, {"GPU_ARCHS": "gfx1250"}, clear=True):
            jobs_1250 = parse_csv(temp_path)
            self.assertGreater(len(jobs_1250), 0)
            self.assertTrue(all(j.get("gfx") == "gfx1250" for j in jobs_1250))

        with patch.dict(os.environ, {}, clear=True):
            jobs_unset = parse_csv(temp_path)
            self.assertGreater(len(jobs_unset), 0)
            self.assertTrue(all(j.get("gfx") == "gfx1250" for j in jobs_unset))


if __name__ == "__main__":
    unittest.main()
