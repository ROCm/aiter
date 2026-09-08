# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU regressions for fused-MoE legacy loading and AOT architecture handling."""

import unittest

import pandas as pd

try:
    from aiter.jit.utils.chip_info import (
        GFX_PLACEHOLDERS,
        backfill_dataframe_gfx,
        reset_legacy_gfx_warnings_for_tests,
    )

    _CHIP_INFO_ERR = None
except Exception as e:  # noqa: BLE001
    GFX_PLACEHOLDERS = None
    backfill_dataframe_gfx = None
    reset_legacy_gfx_warnings_for_tests = None
    _CHIP_INFO_ERR = e

try:
    from aiter.aot.flydsl.common import resolve_job_arch

    _JOB_ARCH_ERR = None
except Exception as e:  # noqa: BLE001
    resolve_job_arch = None
    _JOB_ARCH_ERR = e

try:
    from aiter.aot.flydsl import mxfp4_moe as flydsl_mxfp4_aot

    _MXFP4_AOT_ERR = None
except Exception as e:  # noqa: BLE001
    flydsl_mxfp4_aot = None
    _MXFP4_AOT_ERR = e


@unittest.skipUnless(
    backfill_dataframe_gfx is not None, f"chip_info not importable: {_CHIP_INFO_ERR}"
)
class TestFmoeLegacyGfxLoading(unittest.TestCase):
    def setUp(self):
        reset_legacy_gfx_warnings_for_tests()

    def test_legacy_file_warns_once_per_source(self):
        with self.assertLogs("aiter", level="WARNING") as logs:
            df1 = backfill_dataframe_gfx(pd.DataFrame({"cu_num": [256]}), "legacy.csv")
            df2 = backfill_dataframe_gfx(pd.DataFrame({"cu_num": [256]}), "legacy.csv")
        self.assertEqual(df1.loc[0, "gfx"], "gfx950")
        self.assertEqual(df2.loc[0, "gfx"], "gfx950")
        legacy_msgs = [m for m in logs.output if "lacks explicit gfx" in m]
        self.assertEqual(len(legacy_msgs), 1)

    def test_explicit_gfx_file_emits_no_migration_warning(self):
        with self.assertNoLogs("aiter", level="WARNING"):
            df = backfill_dataframe_gfx(
                pd.DataFrame({"gfx": ["gfx1250"], "cu_num": [256]}), "modern.csv"
            )
        self.assertEqual(df.loc[0, "gfx"], "gfx1250")

    def test_unknown_cu_num_raises_instead_of_using_live_gpu(self):
        with self.assertRaisesRegex(ValueError, "cannot infer gfx from cu_num=128"):
            backfill_dataframe_gfx(pd.DataFrame({"cu_num": [128]}), "unknown.csv")

    def test_placeholder_gfx_with_unknown_cu_num_raises(self):
        with self.assertRaisesRegex(ValueError, "cannot infer gfx from cu_num=128"):
            backfill_dataframe_gfx(
                pd.DataFrame({"gfx": ["0"], "cu_num": [128]}), "placeholder.csv"
            )

    def test_load_and_aot_share_placeholder_set(self):
        from aiter.aot.flydsl.common import GFX_PLACEHOLDERS as aot_placeholders

        self.assertIs(GFX_PLACEHOLDERS, aot_placeholders)
        self.assertIn("0", GFX_PLACEHOLDERS)


@unittest.skipUnless(
    resolve_job_arch is not None, f"resolve_job_arch not importable: {_JOB_ARCH_ERR}"
)
class TestFlydslMoeAotGfx(unittest.TestCase):
    def test_explicit_gfx_overrides_cu_inference(self):
        self.assertEqual(resolve_job_arch(80, "gfx950"), "gfx950")
        self.assertEqual(resolve_job_arch(256, "gfx942"), "gfx942")
        self.assertEqual(resolve_job_arch(256, ""), "gfx950")

    def test_known_legacy_cu_num_without_gfx(self):
        self.assertEqual(resolve_job_arch(80, ""), "gfx942")
        self.assertEqual(resolve_job_arch(304, ""), "gfx942")

    def test_placeholder_gfx_falls_back_to_known_cu_mapping(self):
        self.assertEqual(resolve_job_arch(256, "0"), "gfx950")
        self.assertEqual(resolve_job_arch(80, "None"), "gfx942")
        self.assertEqual(resolve_job_arch(304, "nan"), "gfx942")

    def test_unknown_or_missing_arch_raises(self):
        with self.assertRaisesRegex(ValueError, "cannot map cu_num"):
            resolve_job_arch(0, "")
        with self.assertRaisesRegex(ValueError, "cannot map cu_num"):
            resolve_job_arch(128, "")
        with self.assertRaisesRegex(ValueError, "cannot map cu_num"):
            resolve_job_arch("not-a-cu", "")
        with self.assertRaisesRegex(ValueError, "cannot map cu_num"):
            resolve_job_arch(128, "0")


@unittest.skipUnless(
    flydsl_mxfp4_aot is not None, f"FlyDSL mxfp4 AOT not importable: {_MXFP4_AOT_ERR}"
)
class TestFlydslMxfp4MoeAotGfx(unittest.TestCase):
    def test_rejects_explicit_incompatible_architecture(self):
        job = {
            "stage": 1,
            "kernel_name": "flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt",
            "BM": 16,
            "use_nt": 2,
            "inline_quant": 0,
            "D_HIDDEN": 2048,
            "D_INTER": 1024,
            "NE": 8,
            "topk": 2,
            "gfx": "gfx942",
            "cu_num": 80,
            "xcd_swizzle": 0,
        }
        with self.assertRaisesRegex(ValueError, "gfx950-only"):
            flydsl_mxfp4_aot.compile_one_config(**job)


if __name__ == "__main__":
    unittest.main(verbosity=2)
