# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import csv
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd
import torch

from aiter.ops import gemm_op_a6w6
from csrc.gemm_a6w6.gemm_a6w6_tune import choose_guarded_kernel

AITER_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
MANIFEST = os.path.join(
    AITER_ROOT, "hsa", "gfx950", "f6gemm", "f6gemm_bf16_per1x32Fp6.csv"
)
TUNED_HEADER = [
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "kernelId",
    "splitK",
    "us",
    "kernelName",
    "tflops",
    "bw",
    "errRatio",
]


class TestA6W6TuningLookup(unittest.TestCase):
    def setUp(self):
        gemm_op_a6w6.clear_gemm_a6w6_config_cache()
        self.tempdir = tempfile.TemporaryDirectory()
        self.config = os.path.join(self.tempdir.name, "a6w6.csv")

    def tearDown(self):
        gemm_op_a6w6.clear_gemm_a6w6_config_cache()
        self.tempdir.cleanup()

    def _write_rows(self, rows):
        with open(self.config, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(TUNED_HEADER)
            writer.writerows(rows)
        gemm_op_a6w6.clear_gemm_a6w6_config_cache()

    @mock.patch.object(gemm_op_a6w6, "get_cu_num", return_value=256)
    @mock.patch.object(gemm_op_a6w6, "get_gfx", return_value="gfx950")
    def test_exact_match_precedes_padded_match(self, _gfx, _cu):
        self._write_rows(
            [
                [
                    "gfx950",
                    256,
                    9450,
                    5120,
                    5120,
                    1,
                    0,
                    1,
                    "exact_kernel",
                    1,
                    1,
                    0,
                ],
                [
                    "gfx950",
                    256,
                    9472,
                    5120,
                    5120,
                    2,
                    0,
                    2,
                    "padded_kernel",
                    1,
                    1,
                    0,
                ],
            ]
        )
        config = gemm_op_a6w6.get_GEMM_A6W6_config(9450, 5120, 5120, self.config)
        self.assertEqual(config["kernelName"], "exact_kernel")

    @mock.patch.object(gemm_op_a6w6, "get_cu_num", return_value=256)
    @mock.patch.object(gemm_op_a6w6, "get_gfx", return_value="gfx950")
    def test_padded_match_is_used_as_fallback(self, _gfx, _cu):
        self._write_rows(
            [
                [
                    "gfx950",
                    256,
                    9472,
                    5120,
                    5120,
                    1,
                    0,
                    1,
                    "padded_kernel",
                    1,
                    1,
                    0,
                ]
            ]
        )
        config = gemm_op_a6w6.get_GEMM_A6W6_config(9450, 5120, 5120, self.config)
        self.assertEqual(config["kernelName"], "padded_kernel")

    def test_nonzero_splitk_is_rejected(self):
        self._write_rows(
            [
                [
                    "gfx950",
                    256,
                    256,
                    256,
                    128,
                    1,
                    1,
                    1,
                    "bad_kernel",
                    1,
                    1,
                    0,
                ]
            ]
        )
        with self.assertRaisesRegex(ValueError, "splitK=0"):
            gemm_op_a6w6._load_gemm_a6w6_configs(self.config)

    def test_duplicate_shape_is_rejected(self):
        row = [
            "gfx950",
            256,
            256,
            256,
            128,
            1,
            0,
            1,
            "kernel",
            1,
            1,
            0,
        ]
        self._write_rows([row, row])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            gemm_op_a6w6._load_gemm_a6w6_configs(self.config)

    def test_explicit_override_and_safe_default(self):
        self.assertEqual(
            gemm_op_a6w6._select_gemm_a6w6_kernel(1, 1, 1, "explicit_kernel"),
            "explicit_kernel",
        )
        self.assertEqual(
            gemm_op_a6w6._default_gemm_a6w6_kernel(512, 55296, 6144),
            gemm_op_a6w6._SAFE_FALLBACK_KERNEL_NAME,
        )


class TestA6W6ApiValidation(unittest.TestCase):
    def test_asm_wrapper_selects_safe_default_kernel(self):
        packed = torch.empty(0, dtype=torch.uint8, device="meta")
        out = torch.empty((512, 55296), dtype=torch.bfloat16, device="meta")

        with mock.patch.object(gemm_op_a6w6, "_gemm_a6w6_asm") as launch:
            result = gemm_op_a6w6.gemm_a6w6_asm(
                packed, packed, packed, packed, out, 6144
            )

        self.assertIs(result, out)
        self.assertEqual(
            launch.call_args.args[6], gemm_op_a6w6._SAFE_FALLBACK_KERNEL_NAME
        )

    def test_torch_quantizer_rejects_non_matrix_input(self):
        with self.assertRaisesRegex(ValueError, r"2D \[R, K\] tensor"):
            gemm_op_a6w6.quant_mxfp6_torch(torch.empty((1, 1, 32)))

    def test_torch_quantizer_rejects_unaligned_k(self):
        with self.assertRaisesRegex(ValueError, "positive multiple of 32"):
            gemm_op_a6w6.quant_mxfp6_torch(torch.empty((1, 31)))

    def test_gemm_rejects_unsupported_output_dtype_early(self):
        packed = torch.empty(0, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "only torch.bfloat16 output"):
            gemm_op_a6w6.gemm_a6w6(
                packed,
                packed,
                packed,
                packed,
                256,
                256,
                128,
                dtype=torch.float16,
            )


class TestA6W6Manifest(unittest.TestCase):
    def test_manifest_candidates_are_compatible_and_present(self):
        configs = pd.read_csv(MANIFEST)
        self.assertGreater(len(configs), 1)
        self.assertFalse(configs["knl_name"].duplicated().any())
        self.assertTrue((configs["splitK"] == 0).all())
        self.assertTrue((configs["block_size"] == 256).all())
        self.assertTrue((configs["pack_layout"] == gemm_op_a6w6._PACK_LAYOUT).all())
        self.assertTrue(
            {"swizzle_max_M", "swizzle_max_N", "swizzle_max_K"}.issubset(
                configs.columns
            )
        )
        swz0 = configs[
            configs["knl_name"] == gemm_op_a6w6._SAFE_FALLBACK_KERNEL_NAME
        ].iloc[0]
        self.assertEqual(
            (
                swz0["swizzle_max_M"],
                swz0["swizzle_max_N"],
                swz0["swizzle_max_K"],
            ),
            (0, 0, 0),
        )
        grp16 = configs[configs["knl_name"] == "f6gemm_dmabig_grp16_kernel_func"].iloc[
            0
        ]
        self.assertEqual(grp16["swizzle_max_M"], 65536)
        manifest_dir = os.path.dirname(MANIFEST)
        for co_name in configs["co_name"]:
            self.assertTrue(os.path.exists(os.path.join(manifest_dir, co_name)))


class TestA6W6MinimumGainGuard(unittest.TestCase):
    def test_keeps_candidate_above_threshold(self):
        selected, gain_pct = choose_guarded_kernel(
            "default", "candidate", [1.025, 1.03, 1.035], 2.0, True
        )
        self.assertEqual(selected, "candidate")
        self.assertAlmostEqual(gain_pct, 3.0)

    def test_falls_back_below_threshold(self):
        selected, gain_pct = choose_guarded_kernel(
            "default", "candidate", [1.005, 1.01, 1.015], 2.0, True
        )
        self.assertEqual(selected, "default")
        self.assertAlmostEqual(gain_pct, 1.0)

    def test_falls_back_if_any_paired_repeat_regresses(self):
        selected, gain_pct = choose_guarded_kernel(
            "default", "candidate", [1.02, 0.999, 1.03], 0.5, True
        )
        self.assertEqual(selected, "default")
        self.assertAlmostEqual(gain_pct, 2.0)

    def test_falls_back_on_output_mismatch(self):
        selected, gain_pct = choose_guarded_kernel(
            "default", "candidate", [1.5, 1.6, 1.7], 2.0, False
        )
        self.assertEqual(selected, "default")
        self.assertEqual(gain_pct, float("-inf"))

    def test_default_needs_no_validation(self):
        selected, gain_pct = choose_guarded_kernel("default", "default", [], 2.0, True)
        self.assertEqual(selected, "default")
        self.assertEqual(gain_pct, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
