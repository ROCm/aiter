# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU regressions for fused-MoE legacy loading and AOT architecture handling."""

import csv
import glob
import os
import shutil
import tempfile
import unittest

import pandas as pd

try:
    from aiter.jit.utils.chip_info import (
        backfill_dataframe_gfx,
        reset_legacy_gfx_warnings_for_tests,
    )
    from aiter.jit.utils.gfx_placeholders import GFX_PLACEHOLDERS

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

try:
    from aiter.jit import core as aiter_core

    _CORE_ERR = None
except Exception as e:  # noqa: BLE001
    aiter_core = None
    _CORE_ERR = e


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
        with self.assertRaisesRegex(ValueError, "cannot infer gfx from cu_num=96"):
            backfill_dataframe_gfx(pd.DataFrame({"cu_num": [96]}), "gfx1250-96.csv")

    def test_placeholder_gfx_with_unknown_cu_num_raises(self):
        with self.assertRaisesRegex(ValueError, "cannot infer gfx from cu_num=128"):
            backfill_dataframe_gfx(
                pd.DataFrame({"gfx": ["0"], "cu_num": [128]}), "placeholder.csv"
            )

    def test_numeric_zero_gfx_is_treated_as_placeholder(self):
        with self.assertLogs("aiter", level="WARNING"):
            df = backfill_dataframe_gfx(
                pd.DataFrame({"gfx": [0.0, float("nan")], "cu_num": [256, 80]}),
                "pandas-float.csv",
            )
        self.assertEqual(list(df["gfx"]), ["gfx950", "gfx942"])

    def test_load_and_aot_share_placeholder_set(self):
        from aiter.aot.flydsl.common import LEGACY_CU_NUM_TO_GFX as aot_cu_map
        from aiter.aot.flydsl.common import is_missing_gfx as aot_missing
        from aiter.jit.utils.chip_info import LEGACY_CU_NUM_TO_GFX as load_cu_map
        from aiter.jit.utils.gfx_placeholders import is_missing_gfx as load_missing

        self.assertEqual(load_cu_map, aot_cu_map)
        self.assertIn("0", GFX_PLACEHOLDERS)
        self.assertNotIn(96, load_cu_map)
        for cell in ("", "0", "0.0", 0, 0.0, "nan", "None", None):
            self.assertTrue(load_missing(cell), msg=repr(cell))
            self.assertTrue(aot_missing(cell), msg=repr(cell))
            if cell is None:
                continue
            self.assertEqual(
                resolve_job_arch(256, cell),
                "gfx950",
                msg=repr(cell),
            )


@unittest.skipUnless(
    aiter_core is not None, f"aiter.jit.core not importable: {_CORE_ERR}"
)
class TestMergeNormalizesPlaceholderGfx(unittest.TestCase):
    def test_placeholder_and_explicit_gfx_collide_as_one_key(self):
        header = (
            "gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,"
            "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,us"
        )
        shape = (
            "256,4,2304,1536,8,2,ActivationType.Gelu,torch.bfloat16,"
            "torch.bfloat16,torch.bfloat16,QuantType.No,1,0"
        )
        tmp = tempfile.mkdtemp(prefix="aiter_gfx_merge_")
        try:
            f1 = os.path.join(tmp, "legacy.csv")
            f2 = os.path.join(tmp, "explicit.csv")
            with open(f1, "w", encoding="utf-8") as fh:
                fh.write(f"{header}\n0,{shape},100.0\n")
            with open(f2, "w", encoding="utf-8") as fh:
                fh.write(f"{header}\ngfx950,{shape},10.0\n")
            with self.assertRaises(RuntimeError) as ctx:
                aiter_core.AITER_CONFIGS.update_config_files(
                    f"{f1}{os.pathsep}{f2}", "tuned_fmoe"
                )
            self.assertIn("duplicate shape", str(ctx.exception).lower())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


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
        with self.assertRaisesRegex(ValueError, "cannot map cu_num"):
            resolve_job_arch(96, "")
        with self.assertRaisesRegex(ValueError, "cannot map cu_num"):
            resolve_job_arch(96, "0")


# Merge-name substrings from FlyDSL AOT DEFAULT_CSVS (moe, mxfp4_moe, gemm,
# grouped_moe, chunk_gdn_h). Untuned tables never hit resolve_job_arch.
_AOT_TUNED_FAMILY_MARKERS = (
    "tuned_fmoe",
    "tuned_fhmoe",
    "tuned_grouped_fmoe",
    "a4w4_blockscale_tuned_gemm",
    "a8w8_tuned_gemm",
    "a8w8_bpreshuffle_tuned_gemm",
    "a8w8_blockscale_tuned_gemm",
    "a8w8_blockscale_bpreshuffle_tuned_gemm",
    "a8w8_tuned_batched_gemm",
    "bf16_tuned_batched_gemm",
    "bf16_tuned_gemm",
    "chunk_gdn_h_opt_tuned",
)


def _aot_family_tuned_csvs():
    configs = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "aiter",
        "configs",
    )
    for path in glob.glob(os.path.join(configs, "**", "*.csv"), recursive=True):
        name = os.path.basename(path)
        if "untuned" in name:
            continue
        if any(marker in name for marker in _AOT_TUNED_FAMILY_MARKERS):
            yield path


@unittest.skipUnless(
    resolve_job_arch is not None, f"resolve_job_arch not importable: {_JOB_ARCH_ERR}"
)
class TestAotFamilyTunedCsvsResolveArch(unittest.TestCase):
    def test_every_aot_family_row_resolves_without_guessing_96(self):
        failures = []
        files = 0
        rows = 0
        cu96 = 0
        for path in _aot_family_tuned_csvs():
            files += 1
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(
                    line for line in f if not line.lstrip().startswith("#")
                )
                for i, row in enumerate(reader, start=2):
                    rows += 1
                    gfx = (row.get("gfx") or "").strip()
                    cu_raw = (row.get("cu_num") or "").strip()
                    try:
                        cu_num = int(cu_raw) if cu_raw else 0
                    except ValueError:
                        failures.append(f"{path}:{i} unparsable cu_num={cu_raw!r}")
                        continue
                    if cu_num == 96:
                        cu96 += 1
                    try:
                        arch = resolve_job_arch(cu_num, gfx)
                    except ValueError as e:
                        failures.append(f"{path}:{i} gfx={gfx!r} cu_num={cu_num}: {e}")
                        continue
                    if cu_num == 96 and arch != "gfx1250":
                        failures.append(
                            f"{path}:{i} cu_num=96 resolved to {arch!r}, expected gfx1250"
                        )
        self.assertGreater(files, 0)
        self.assertGreater(rows, 0)
        self.assertGreater(cu96, 0)
        self.assertEqual(failures, [])


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
        result = flydsl_mxfp4_aot.compile_one_config(**job)
        self.assertIsNone(result["compile_time"])

    def test_same_shape_jobs_for_two_gfx_are_not_deduped(self):
        job_950 = {
            "stage": 1,
            "kernel_name": "flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt",
            "BM": 16,
            "use_nt": 2,
            "inline_quant": 0,
            "D_HIDDEN": 2048,
            "D_INTER": 1024,
            "NE": 8,
            "topk": 2,
            "gfx": "gfx950",
            "cu_num": 256,
            "xcd_swizzle": 0,
        }
        job_1250 = {**job_950, "gfx": "gfx1250"}
        self.assertNotEqual(
            flydsl_mxfp4_aot._job_key(job_950),
            flydsl_mxfp4_aot._job_key(job_1250),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
