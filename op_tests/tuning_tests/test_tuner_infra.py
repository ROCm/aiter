# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Level 1: Unit tests for base_tuner infrastructure (no GPU required).

Covers: CSV I/O, merge/dedup, calculate, post_process (topk selection).
"""

import os
import tempfile
import unittest
import argparse
import pandas as pd

TEST_GFX = "gfx942"


class _StubTuner:
    """Lazy-init helper -- avoids importing aiter at module level."""

    _cls = None

    @classmethod
    def get(cls):
        if cls._cls is None:
            from aiter.utility.base_tuner import GemmCommonTuner

            class Stub(GemmCommonTuner):
                def _setup_specific_arguments(self):
                    pass

                def tune(self, *a, **kw):
                    pass

                def getKernelName(self, kid):
                    return f"k{kid}"

            cls._cls = Stub
        return cls._cls("test")


class TestReadCSV(unittest.TestCase):

    def test_strips_whitespace(self):
        from aiter.utility.base_tuner import _read_csv

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(" M , N , K \n 1 , 2 , 3 \n 4 , 5 , 6 \n")
            path = f.name
        try:
            df = _read_csv(path)
            self.assertEqual(list(df.columns), ["M", "N", "K"])
            self.assertEqual(len(df), 2)
        finally:
            os.unlink(path)

    def test_drops_unnamed_columns(self):
        from aiter.utility.base_tuner import _read_csv

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("M,N,K,Unnamed: 0\n1,2,3,\n")
            path = f.name
        try:
            df = _read_csv(path)
            self.assertNotIn("Unnamed: 0", df.columns)
        finally:
            os.unlink(path)

    def test_drops_all_na_rows(self):
        from aiter.utility.base_tuner import _read_csv

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("M,N,K\n1,2,3\n,,\n4,5,6\n")
            path = f.name
        try:
            df = _read_csv(path)
            self.assertEqual(len(df), 2)
        finally:
            os.unlink(path)


class TestUpdateTunedf(unittest.TestCase):

    def test_merges_existing_key(self):
        tuner = _StubTuner.get()
        old = pd.DataFrame(
            {
                "gfx": [TEST_GFX],
                "cu_num": [304],
                "M": [1],
                "N": [1024],
                "K": [512],
                "kernelId": [0],
                "splitK": [0],
                "us": [100.0],
                "kernelName": ["old"],
                "tflops": [1.0],
                "bw": [1.0],
                "errRatio": [0.01],
            }
        )
        new = pd.DataFrame(
            {
                "gfx": [TEST_GFX],
                "cu_num": [304],
                "M": [1],
                "N": [1024],
                "K": [512],
                "kernelId": [1],
                "splitK": [0],
                "us": [50.0],
                "kernelName": ["new"],
                "tflops": [2.0],
                "bw": [2.0],
                "errRatio": [0.005],
            }
        )
        merged = tuner.update_tunedf(old, new)
        self.assertEqual(len(merged), 1)
        self.assertEqual(float(merged.iloc[0]["us"]), 50.0)

    def test_appends_new_key(self):
        tuner = _StubTuner.get()
        old = pd.DataFrame(
            {
                "gfx": [TEST_GFX],
                "cu_num": [304],
                "M": [1],
                "N": [1024],
                "K": [512],
                "kernelId": [0],
                "splitK": [0],
                "us": [100.0],
                "kernelName": ["k0"],
                "tflops": [1.0],
                "bw": [1.0],
                "errRatio": [0.01],
            }
        )
        new = pd.DataFrame(
            {
                "gfx": [TEST_GFX],
                "cu_num": [304],
                "M": [32],
                "N": [2048],
                "K": [1024],
                "kernelId": [2],
                "splitK": [0],
                "us": [200.0],
                "kernelName": ["k2"],
                "tflops": [3.0],
                "bw": [3.0],
                "errRatio": [0.02],
            }
        )
        merged = tuner.update_tunedf(old, new)
        self.assertEqual(len(merged), 2)


class TestSortResults(unittest.TestCase):

    def test_deduplicates(self):
        tuner = _StubTuner.get()
        df = pd.DataFrame(
            {
                "gfx": [TEST_GFX, TEST_GFX],
                "cu_num": [304, 304],
                "M": [1, 1],
                "N": [1024, 1024],
                "K": [512, 512],
                "kernelId": [0, 1],
                "splitK": [0, 0],
                "us": [100.0, 50.0],
                "kernelName": ["k0", "k1"],
                "tflops": [1.0, 2.0],
                "bw": [1.0, 2.0],
                "errRatio": [0.01, 0.005],
            }
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            path = f.name
        try:
            tuner.sortResults(path, True, tuner.sort_keys)
            result = pd.read_csv(path)
            self.assertEqual(len(result), 1)
        finally:
            os.unlink(path)


class TestCalculate(unittest.TestCase):

    def test_tflops_bw_positive(self):
        tuner = _StubTuner.get()
        keys = (TEST_GFX, 304, 128, 4096, 1024)
        result = ((keys,), 100.0, 0.01)
        tflops, bw = tuner.calculate(result)
        self.assertGreater(tflops, 0)
        self.assertGreater(bw, 0)

    def test_returns_zero_on_invalid_time(self):
        tuner = _StubTuner.get()
        keys = (TEST_GFX, 304, 128, 4096, 1024)
        result = ((keys,), -1, 1.0)
        tflops, bw = tuner.calculate(result)
        self.assertEqual(tflops, 0)
        self.assertEqual(bw, 0)


class TestPostProcess(unittest.TestCase):
    """Tests for post_process -- especially the topk selection logic."""

    def _make_args(self, err_ratio=0.05, profile_file=""):
        args = argparse.Namespace()
        args.errRatio = err_ratio
        args.profile_file = profile_file
        args.verbose = False
        return args

    def _make_result(self, shape_key, kernel_id, split_k, us, err=0.0):
        info = (shape_key, kernel_id, split_k, f"kernel_{kernel_id}")
        return (info, us, err)

    def test_picks_fastest_per_shape(self):
        """Basic: 2 shapes, 3 kernels each, picks fastest."""
        tuner = _StubTuner.get()
        args = self._make_args()
        rets = [
            self._make_result((TEST_GFX, 304, 1, 1024, 512), 0, 0, 10.0),
            self._make_result((TEST_GFX, 304, 1, 1024, 512), 1, 0, 5.0),
            self._make_result((TEST_GFX, 304, 1, 1024, 512), 2, 0, 8.0),
            self._make_result((TEST_GFX, 304, 32, 2048, 1024), 0, 0, 20.0),
            self._make_result((TEST_GFX, 304, 32, 2048, 1024), 1, 0, 12.0),
            self._make_result((TEST_GFX, 304, 32, 2048, 1024), 2, 0, 15.0),
        ]
        resultdf = tuner.post_process(rets, args, topk=1)
        self.assertEqual(len(resultdf), 2)
        times = sorted(resultdf["us"].tolist())
        self.assertEqual(times, [5.0, 12.0])

    def test_filters_by_err_ratio(self):
        """Kernels exceeding errRatio should be skipped."""
        tuner = _StubTuner.get()
        args = self._make_args(err_ratio=0.05)
        rets = [
            self._make_result((TEST_GFX, 304, 1, 1024, 512), 0, 0, 5.0, err=0.1),
            self._make_result((TEST_GFX, 304, 1, 1024, 512), 1, 0, 10.0, err=0.01),
            self._make_result((TEST_GFX, 304, 1, 1024, 512), 2, 0, 8.0, err=0.02),
        ]
        resultdf = tuner.post_process(rets, args, topk=1)
        self.assertEqual(len(resultdf), 1)
        self.assertEqual(float(resultdf.iloc[0]["us"]), 8.0)

    def test_filters_invalid_and_inf_times(self):
        """us=-1 (error) and us=inf (timeout) should be excluded."""
        tuner = _StubTuner.get()
        args = self._make_args()
        rets = [
            self._make_result((TEST_GFX, 304, 1, 1024, 512), 0, 0, -1, err=0.0),
            self._make_result((TEST_GFX, 304, 1, 1024, 512), 1, 0, float("inf"), err=0.0),
            self._make_result((TEST_GFX, 304, 1, 1024, 512), 2, 0, 7.0, err=0.01),
        ]
        resultdf = tuner.post_process(rets, args, topk=1)
        self.assertEqual(len(resultdf), 1)
        self.assertEqual(float(resultdf.iloc[0]["us"]), 7.0)

    def test_topk_not_leak_across_shapes(self):
        """BUG REGRESSION: topk must not leak between shapes.

        If shape A has 0 valid candidates (all fail errRatio), topk should NOT
        be permanently set to 0, causing shape B to also return 0 results.
        """
        tuner = _StubTuner.get()
        args = self._make_args(err_ratio=0.05)
        rets = [
            # Shape A: all kernels fail errRatio -> 0 valid candidates
            self._make_result((TEST_GFX, 304, 1, 1024, 512), 0, 0, 5.0, err=0.9),
            self._make_result((TEST_GFX, 304, 1, 1024, 512), 1, 0, 3.0, err=0.8),
            # Shape B: has valid candidates -> should still get results
            self._make_result((TEST_GFX, 304, 32, 2048, 1024), 0, 0, 10.0, err=0.01),
            self._make_result((TEST_GFX, 304, 32, 2048, 1024), 1, 0, 8.0, err=0.02),
        ]
        resultdf = tuner.post_process(rets, args, topk=1)
        shape_b_rows = resultdf[resultdf["M"] == 32]
        self.assertGreaterEqual(
            len(shape_b_rows), 1, "Shape B should have results even if Shape A has none"
        )
        self.assertEqual(float(shape_b_rows.iloc[0]["us"]), 8.0)

    def test_topk_not_shrink_across_shapes(self):
        """topk should not shrink when one shape has fewer valid candidates than topk."""
        tuner = _StubTuner.get()
        args = self._make_args(err_ratio=0.05)
        rets = [
            # Shape A: only 1 valid candidate (topk=2 requested but only 1 available)
            self._make_result((TEST_GFX, 304, 1, 1024, 512), 0, 0, 5.0, err=0.01),
            self._make_result((TEST_GFX, 304, 1, 1024, 512), 1, 0, 3.0, err=0.9),
            # Shape B: 3 valid candidates -> should get topk=2
            self._make_result((TEST_GFX, 304, 32, 2048, 1024), 0, 0, 10.0, err=0.01),
            self._make_result((TEST_GFX, 304, 32, 2048, 1024), 1, 0, 8.0, err=0.02),
            self._make_result((TEST_GFX, 304, 32, 2048, 1024), 2, 0, 12.0, err=0.01),
        ]
        resultdf = tuner.post_process(rets, args, topk=2)
        shape_b_rows = resultdf[resultdf["M"] == 32]
        self.assertGreaterEqual(
            len(shape_b_rows),
            2,
            "Shape B should get topk=2 results even though Shape A only had 1",
        )

    def test_all_shapes_fail(self):
        """When all kernels for all shapes fail, should still produce fallback entries."""
        tuner = _StubTuner.get()
        args = self._make_args(err_ratio=0.05)
        rets = [
            self._make_result((TEST_GFX, 304, 1, 1024, 512), 0, 0, 5.0, err=0.9),
            self._make_result((TEST_GFX, 304, 32, 2048, 1024), 0, 0, 10.0, err=0.8),
        ]
        resultdf = tuner.post_process(rets, args, topk=1)
        self.assertEqual(len(resultdf), 2, "Should have fallback entry for each shape")

    def test_single_shape_single_kernel(self):
        """Minimal case: 1 shape, 1 kernel -> should work."""
        tuner = _StubTuner.get()
        args = self._make_args()
        rets = [
            self._make_result((TEST_GFX, 304, 1, 1024, 512), 0, 0, 5.0, err=0.01),
        ]
        resultdf = tuner.post_process(rets, args, topk=1)
        self.assertEqual(len(resultdf), 1)
        self.assertEqual(float(resultdf.iloc[0]["us"]), 5.0)


class TestUpdateConfigFiles(unittest.TestCase):
    """Tests for TunerCommon.update_config_files (config merge/dedup)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_csv(self, name, header, rows):
        path = os.path.join(self.tmpdir, name)
        with open(path, "w") as f:
            f.write(",".join(header) + "\n")
            for row in rows:
                f.write(",".join(str(v) for v in row) + "\n")
        return path

    def test_single_path_returns_as_is(self):
        tuner = _StubTuner.get()
        path = "/some/path.csv"
        self.assertEqual(tuner.update_config_files(path, "test"), path)

    def test_empty_path_returns_as_is(self):
        tuner = _StubTuner.get()
        self.assertEqual(tuner.update_config_files("", "test"), "")

    def test_two_files_merge_dedup(self):
        """Two CSVs with overlapping keys -> merged, fastest us kept."""
        tuner = _StubTuner.get()
        header = ["gfx", "cu_num", "M", "N", "K", "kernelId", "splitK",
                  "us", "kernelName", "tflops", "bw", "errRatio"]
        f1 = self._write_csv("a.csv", header,
                             [[TEST_GFX, 304, 1, 1024, 512, 0, 0, 100.0, "k0", 1.0, 1.0, 0.01]])
        f2 = self._write_csv("b.csv", header,
                             [[TEST_GFX, 304, 1, 1024, 512, 1, 0, 50.0, "k1", 2.0, 2.0, 0.005]])
        merged_path = tuner.update_config_files(f"{f1}{os.pathsep}{f2}", "test_merge")
        try:
            self.assertNotEqual(merged_path, f"{f1}{os.pathsep}{f2}")
            df = pd.read_csv(merged_path)
            self.assertEqual(len(df), 1)
            self.assertEqual(float(df.iloc[0]["us"]), 50.0)
        finally:
            if os.path.exists(merged_path):
                os.unlink(merged_path)

    def test_column_mismatch_raises(self):
        """Two CSVs with different columns -> AssertionError."""
        tuner = _StubTuner.get()
        h1 = ["gfx", "cu_num", "M", "N", "K", "kernelId", "splitK",
              "us", "kernelName", "tflops", "bw", "errRatio"]
        h2 = ["gfx", "cu_num", "M", "N", "K", "extra_col",
              "us", "kernelName", "tflops", "bw", "errRatio"]
        f1 = self._write_csv("a.csv", h1,
                             [[TEST_GFX, 304, 1, 1024, 512, 0, 0, 100.0, "k0", 1.0, 1.0, 0.01]])
        f2 = self._write_csv("b.csv", h2,
                             [[TEST_GFX, 304, 1, 1024, 512, "x", 100.0, "k0", 1.0, 1.0, 0.01]])
        with self.assertRaises(AssertionError):
            tuner.update_config_files(f"{f1}{os.pathsep}{f2}", "test_mismatch")

    def test_missing_second_file(self):
        """Second path doesn't exist -> only first file data."""
        tuner = _StubTuner.get()
        header = ["gfx", "cu_num", "M", "N", "K", "kernelId", "splitK",
                  "us", "kernelName", "tflops", "bw", "errRatio"]
        f1 = self._write_csv("a.csv", header,
                             [[TEST_GFX, 304, 1, 1024, 512, 0, 0, 100.0, "k0", 1.0, 1.0, 0.01]])
        fake = os.path.join(self.tmpdir, "nonexistent.csv")
        merged_path = tuner.update_config_files(f"{f1}{os.pathsep}{fake}", "test_missing")
        try:
            df = pd.read_csv(merged_path)
            self.assertEqual(len(df), 1)
            self.assertEqual(float(df.iloc[0]["us"]), 100.0)
        finally:
            if os.path.exists(merged_path):
                os.unlink(merged_path)

    def test_tag_column_dedup(self):
        """CSVs with _tag column -> dedup uses keys + _tag."""
        tuner = _StubTuner.get()
        header = ["gfx", "cu_num", "M", "N", "K", "kernelId", "splitK",
                  "us", "kernelName", "tflops", "bw", "errRatio", "_tag"]
        f1 = self._write_csv("a.csv", header,
                             [[TEST_GFX, 304, 1, 1024, 512, 0, 0, 100.0, "k0", 1.0, 1.0, 0.01, "modelA"]])
        f2 = self._write_csv("b.csv", header,
                             [[TEST_GFX, 304, 1, 1024, 512, 1, 0, 80.0, "k1", 1.5, 1.5, 0.01, "modelB"]])
        merged_path = tuner.update_config_files(f"{f1}{os.pathsep}{f2}", "test_tag")
        try:
            df = pd.read_csv(merged_path)
            self.assertEqual(len(df), 2, "Same keys but different _tag -> both kept")
        finally:
            if os.path.exists(merged_path):
                os.unlink(merged_path)

    def test_two_files_non_overlapping(self):
        """Two CSVs with different keys -> both rows in output."""
        tuner = _StubTuner.get()
        header = ["gfx", "cu_num", "M", "N", "K", "kernelId", "splitK",
                  "us", "kernelName", "tflops", "bw", "errRatio"]
        f1 = self._write_csv("a.csv", header,
                             [[TEST_GFX, 304, 1, 1024, 512, 0, 0, 100.0, "k0", 1.0, 1.0, 0.01]])
        f2 = self._write_csv("b.csv", header,
                             [[TEST_GFX, 304, 32, 2048, 1024, 1, 0, 200.0, "k1", 2.0, 2.0, 0.005]])
        merged_path = tuner.update_config_files(f"{f1}{os.pathsep}{f2}", "test_nonoverlap")
        try:
            df = pd.read_csv(merged_path)
            self.assertEqual(len(df), 2)
        finally:
            if os.path.exists(merged_path):
                os.unlink(merged_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
