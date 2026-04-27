# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Level 0: Static validation of tuned/untuned CSV files (no GPU, fast).

Catches: duplicates, invalid times, high errRatio, git merge conflicts,
missing untuned files.
"""

import os
import unittest
import pandas as pd

AITER_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONFIGS_DIR = os.path.join(AITER_ROOT, "aiter", "configs")


class TestCSVValidation(unittest.TestCase):

    TUNED_CSVS = {
        "a8w8": "a8w8_tuned_gemm.csv",
        "a8w8_bpreshuffle": "a8w8_bpreshuffle_tuned_gemm.csv",
        "a8w8_blockscale": "a8w8_blockscale_tuned_gemm.csv",
        "a8w8_blockscale_bpreshuffle": "a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
        "a4w4_blockscale": "a4w4_blockscale_tuned_gemm.csv",
        "a8w8_batched": "a8w8_tuned_batched_gemm.csv",
        "bf16": "bf16_tuned_gemm.csv",
        "bf16_batched": "bf16_tuned_batched_gemm.csv",
        "fmoe": "tuned_fmoe.csv",
    }

    def _load_csv(self, name):
        path = os.path.join(CONFIGS_DIR, self.TUNED_CSVS[name])
        if not os.path.exists(path):
            self.skipTest(f"{self.TUNED_CSVS[name]} not found")
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df

    def _get_key_cols(self, df):
        candidates = [
            "cu_num",
            "M",
            "N",
            "K",
            "B",
            "token",
            "model_dim",
            "inter_dim",
            "expert",
            "topk",
        ]
        return [c for c in candidates if c in df.columns]

    def _check_no_duplicates(self, name, extra_keys=None):
        df = self._load_csv(name)
        keys = self._get_key_cols(df)
        if extra_keys:
            keys.extend([k for k in extra_keys if k in df.columns])
        dupes = df[df.duplicated(subset=keys, keep=False)]
        self.assertEqual(
            len(dupes),
            0,
            f"{name}: {len(dupes)} duplicate rows (first 10):\n{dupes.head(10)}",
        )

    def test_a8w8_no_duplicates(self):
        self._check_no_duplicates("a8w8", extra_keys=["q_dtype_w"])

    def test_a8w8_blockscale_no_duplicates(self):
        self._check_no_duplicates("a8w8_blockscale")

    def test_fmoe_no_duplicates(self):
        self._check_no_duplicates(
            "fmoe",
            extra_keys=[
                "act_type",
                "dtype",
                "q_dtype_a",
                "q_dtype_w",
                "q_type",
                "use_g1u1",
                "doweight_stage1",
                "_tag",
            ],
        )

    def test_no_git_conflict_markers(self):
        for name, fname in self.TUNED_CSVS.items():
            with self.subTest(csv=name):
                path = os.path.join(CONFIGS_DIR, fname)
                if not os.path.exists(path):
                    continue
                with open(path, "r") as f:
                    content = f.read()
                for marker in ["<<<<<<<", "=======", ">>>>>>>"]:
                    self.assertNotIn(
                        marker, content, f"{name}: git conflict marker '{marker}' found"
                    )

    def test_no_invalid_times(self):
        for name in self.TUNED_CSVS:
            with self.subTest(csv=name):
                df = self._load_csv(name)
                if "us" not in df.columns:
                    continue
                us = pd.to_numeric(df["us"], errors="coerce")
                bad = df[us <= 0]
                self.assertEqual(
                    len(bad), 0, f"{name}: {len(bad)} rows with us <= 0:\n{bad.head(5)}"
                )

    def test_error_ratios_within_bounds(self):
        for name in self.TUNED_CSVS:
            with self.subTest(csv=name):
                df = self._load_csv(name)
                if "errRatio" not in df.columns:
                    continue
                err_col = df["errRatio"]
                if err_col.dtype == object:
                    err_col = err_col.str.rstrip("%").astype(float) / 100.0
                else:
                    err_col = pd.to_numeric(err_col, errors="coerce")
                high = df[err_col > 0.2]
                self.assertEqual(
                    len(high),
                    0,
                    f"{name}: {len(high)} rows with errRatio > 0.2:\n{high.head(5)}",
                )

    def test_untuned_csvs_exist(self):
        untuned_files = [
            "a8w8_untuned_gemm.csv",
            "a8w8_bpreshuffle_untuned_gemm.csv",
            "a8w8_blockscale_untuned_gemm.csv",
            "a8w8_untuned_batched_gemm.csv",
            "bf16_untuned_batched_gemm.csv",
            "untuned_fmoe.csv",
        ]
        for f in untuned_files:
            with self.subTest(file=f):
                path = os.path.join(CONFIGS_DIR, f)
                self.assertTrue(os.path.exists(path), f"Missing: {f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
