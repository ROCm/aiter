# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Level 0: Static validation of tuned/untuned CSV files (no GPU, fast).

Catches: duplicates, invalid times, high errRatio, git merge conflicts,
missing untuned files.
"""

import ast
import os
import re
import unittest
from typing import Any, ClassVar

import pandas as pd

AITER_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONFIGS_DIR = os.path.join(AITER_ROOT, "aiter", "configs")


class TestCSVValidation(unittest.TestCase):

    TUNED_CSVS: ClassVar[dict[str, Any]] = {
        "a8w8": "a8w8_tuned_gemm.csv",
        "a8w8_bpreshuffle": "a8w8_bpreshuffle_tuned_gemm.csv",
        "a8w8_blockscale": "a8w8_blockscale_tuned_gemm.csv",
        "a8w8_blockscale_bpreshuffle": "a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
        "a4w4_blockscale": "a4w4_blockscale_tuned_gemm.csv",
        "a6w6_blockscale": "a6w6_blockscale_tuned_gemm.csv",
        "a8w8_batched": "a8w8_tuned_batched_gemm.csv",
        "bf16": "bf16_tuned_gemm.csv",
        "bf16_batched": "bf16_tuned_batched_gemm.csv",
        "fmoe": "tuned_fmoe.csv",
        "topk_select": "topk_select_tuned.csv",
    }

    TOPK_SELECT_KEYS: ClassVar[list[str]] = [
        "gfx",
        "cu_num",
        "rows_lo",
        "rows_hi",
        "width_lo",
        "width_hi",
        "k",
        "dtype",
        "ragged",
        "tie",
        "deterministic",
        "mode",
    ]

    @staticmethod
    def _topk_select_backends():
        """Backend names as declared in topk_select.py, read without importing
        aiter, so a backend added there is accepted here with no edit."""
        src = os.path.join(AITER_ROOT, "aiter", "ops", "topk_select.py")
        with open(src) as f:
            tree = ast.parse(f.read())
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") == "_BACKENDS_BY_TIE"
            ):
                table = ast.literal_eval(node.value)
                return {b for names in table.values() for b in names}
        raise AssertionError("_BACKENDS_BY_TIE not found in topk_select.py")

    def _load_csv(self, name):
        path = os.path.join(CONFIGS_DIR, self.TUNED_CSVS[name])
        if not os.path.exists(path):
            self.skipTest(f"{self.TUNED_CSVS[name]} not found")
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df

    def _get_key_cols(self, df):
        candidates = [
            "gfx",
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

    def test_a8w8_bpreshuffle_no_duplicates(self):
        self._check_no_duplicates(
            "a8w8_bpreshuffle", extra_keys=["q_dtype_w", "libtype"]
        )

    def test_a8w8_blockscale_bpreshuffle_no_duplicates(self):
        self._check_no_duplicates("a8w8_blockscale_bpreshuffle", extra_keys=["libtype"])

    def test_a4w4_blockscale_no_duplicates(self):
        self._check_no_duplicates("a4w4_blockscale")

    def test_a6w6_blockscale_no_duplicates(self):
        self._check_no_duplicates("a6w6_blockscale")

    def test_a8w8_batched_no_duplicates(self):
        self._check_no_duplicates("a8w8_batched")

    def test_bf16_no_duplicates(self):
        self._check_no_duplicates(
            "bf16",
            extra_keys=[
                "bias",
                "dtype",
                "outdtype",
                "scaleAB",
                "bpreshuffle",
                "libtype",
            ],
        )

    def test_bf16_batched_no_duplicates(self):
        self._check_no_duplicates("bf16_batched")

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

    def _topk_select_tables(self):
        for root in (CONFIGS_DIR, os.path.join(CONFIGS_DIR, "model_configs")):
            if not os.path.isdir(root):
                continue
            for f in sorted(os.listdir(root)):
                if "topk_select_tuned" in f and "untuned" in f:
                    continue
                if "topk_select_tuned" in f and f.endswith(".csv"):
                    yield os.path.join(root, f)

    def test_topk_select_tables_well_formed(self):
        backends = self._topk_select_backends()
        spec = re.compile(r"^.+@[0-9]+x[0-9]+$")
        for path in self._topk_select_tables():
            with self.subTest(csv=os.path.relpath(path, AITER_ROOT)):
                df = pd.read_csv(path, keep_default_na=False)
                df.columns = df.columns.str.strip()
                missing = [
                    c for c in [*self.TOPK_SELECT_KEYS, "backend"] if c not in df
                ]
                self.assertFalse(missing, f"missing columns {missing}")
                dupes = df[df.duplicated(subset=self.TOPK_SELECT_KEYS, keep=False)]
                self.assertEqual(len(dupes), 0, f"duplicate bands:\n{dupes.head(10)}")
                bad_backend = df[~df["backend"].isin(backends)]
                self.assertEqual(
                    len(bad_backend), 0, f"unknown backend:\n{bad_backend.head(5)}"
                )
                for lo, hi in (("rows_lo", "rows_hi"), ("width_lo", "width_hi")):
                    bad = df[(df[lo] < 1) | (df[hi] < df[lo])]
                    self.assertEqual(len(bad), 0, f"bad {lo}/{hi}:\n{bad.head(5)}")
                bad_k = df[(df["k"] < 1) | (df["k"] > df["width_hi"])]
                self.assertEqual(len(bad_k), 0, f"k outside width:\n{bad_k.head(5)}")
                bad_mode = df[~df["mode"].isin(["graph", "eager"])]
                self.assertEqual(len(bad_mode), 0, f"bad mode:\n{bad_mode.head(5)}")
                bad_dtype = df[~df["dtype"].isin(["float32", "bfloat16", "float16"])]
                self.assertEqual(len(bad_dtype), 0, f"bad dtype:\n{bad_dtype.head(5)}")
                if "dists" in df:
                    for i, cell in enumerate(df["dists"].astype(str)):
                        for s in filter(None, cell.split(";")):
                            self.assertRegex(s, spec, f"row {i + 2}: dists entry")
                            # A recorded sample's path exists only where it was
                            # recorded; --run_config could never verify the row.
                            dist = s.rsplit("@", 1)[0]
                            self.assertFalse(
                                "/" in dist or dist.endswith((".pt", ".pth")),
                                f"row {i + 2}: shipped tables must be tuned on "
                                f"presets, not recorded samples ({dist})",
                            )

    def test_topk_select_backends_read_from_source(self):
        self.assertLessEqual(
            {"argmax", "small_k", "plain", "decode", "stream", "sampled"},
            self._topk_select_backends(),
        )

    def test_topk_select_no_duplicates(self):
        self._check_no_duplicates("topk_select", extra_keys=self.TOPK_SELECT_KEYS[2:])

    def test_flydsl_stage2_sort_block_matches_fmoe_config(self):
        """Stage2 must consume the same sorting layout emitted by stage1."""
        tile_pattern = re.compile(r"_t([0-9]+)x[0-9]+x[0-9]+")
        sort_block_pattern = re.compile(r"_sbm([0-9]+)(?:_|$)")
        mismatches = []

        for root, _, files in os.walk(CONFIGS_DIR):
            for filename in files:
                if "tuned_fmoe" not in filename or not filename.endswith(".csv"):
                    continue
                path = os.path.join(root, filename)
                df = pd.read_csv(path)
                if "block_m" not in df.columns or "kernelName2" not in df.columns:
                    continue
                for index, row in df.iterrows():
                    kernel_name = str(row["kernelName2"])
                    if not kernel_name.startswith("flydsl_moe2_"):
                        continue
                    tile_match = tile_pattern.search(kernel_name)
                    if tile_match is None:
                        continue
                    sort_block_match = sort_block_pattern.search(kernel_name)
                    sort_block_m = int(
                        sort_block_match.group(1)
                        if sort_block_match is not None
                        else tile_match.group(1)
                    )
                    block_m = int(row["block_m"])
                    if sort_block_m != block_m:
                        relative_path = os.path.relpath(path, AITER_ROOT)
                        mismatches.append(
                            f"{relative_path}:{index + 2}: block_m={block_m}, "
                            f"sort_block_m={sort_block_m}, kernelName2={kernel_name}"
                        )

        self.assertFalse(
            mismatches,
            "FlyDSL stage2 sorting layout mismatches:\n" + "\n".join(mismatches),
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
            "a6w6_blockscale_untuned_gemm.csv",
            "a8w8_untuned_batched_gemm.csv",
            "bf16_untuned_batched_gemm.csv",
            "untuned_fmoe.csv",
            "topk_select_untuned.csv",
        ]
        for f in untuned_files:
            with self.subTest(file=f):
                path = os.path.join(CONFIGS_DIR, f)
                self.assertTrue(os.path.exists(path), f"Missing: {f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
