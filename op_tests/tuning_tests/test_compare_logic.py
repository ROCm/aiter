# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Level 1: Unit tests for compare / update_improved logic (no GPU required).

Covers: _build_compare_update_plan, _merge_compare_filtered_results.
"""

import os
import tempfile
import unittest
import pandas as pd

TEST_GFX = "gfx942"
TEST_CU = 304


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


def _make_shapes_df(shapes):
    """Build a shapes DataFrame from list of (M, N, K) tuples."""
    rows = []
    for m, n, k in shapes:
        rows.append({"gfx": TEST_GFX, "cu_num": TEST_CU, "M": m, "N": n, "K": k})
    return pd.DataFrame(rows)


def _make_bench_result(shape_str, status, e2e_us, kernel_us=None):
    """Build a single benchmark result dict."""
    r = {"shape": shape_str, "status": status, "e2e_us": e2e_us}
    if kernel_us is not None:
        r["kernel_us"] = kernel_us
    return r


def _make_tuned_csv(path, rows):
    """Write a tuned CSV with standard GEMM columns."""
    cols = [
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
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# _build_compare_update_plan
# ---------------------------------------------------------------------------
class TestBuildCompareUpdatePlan(unittest.TestCase):

    def _build_plan(self, shapes, pre_results, post_results, threshold=3.0):
        tuner = _StubTuner.get()
        shapes_df = _make_shapes_df(shapes)
        tuner.untunedf = shapes_df
        return tuner._build_compare_update_plan(
            pre_results, post_results, threshold, shapes_df=shapes_df
        )

    def test_threshold_met(self):
        """50% speedup exceeds 3% threshold -> update."""
        shapes = [(1, 1024, 512)]
        pre = [_make_bench_result("(1,1024,512)", "ok", 100.0)]
        post = [_make_bench_result("(1,1024,512)", "ok", 50.0)]
        plan = self._build_plan(shapes, pre, post, threshold=3.0)
        self.assertEqual(len(plan), 1)
        self.assertTrue(plan.iloc[0]["update"])
        self.assertEqual(plan.iloc[0]["update_reason"], "threshold_met")
        self.assertAlmostEqual(plan.iloc[0]["improvement_pct"], 50.0, places=1)

    def test_below_threshold(self):
        """1% speedup is below 3% threshold -> skip."""
        shapes = [(1, 1024, 512)]
        pre = [_make_bench_result("(1,1024,512)", "ok", 100.0)]
        post = [_make_bench_result("(1,1024,512)", "ok", 99.0)]
        plan = self._build_plan(shapes, pre, post, threshold=3.0)
        self.assertEqual(len(plan), 1)
        self.assertFalse(plan.iloc[0]["update"])
        self.assertEqual(plan.iloc[0]["update_reason"], "skip")

    def test_no_baseline(self):
        """Pre has error, post is ok -> update with 'no_baseline' reason."""
        shapes = [(1, 1024, 512)]
        pre = [_make_bench_result("(1,1024,512)", "error:crash", -1)]
        post = [_make_bench_result("(1,1024,512)", "ok", 50.0)]
        plan = self._build_plan(shapes, pre, post, threshold=3.0)
        self.assertEqual(len(plan), 1)
        self.assertTrue(plan.iloc[0]["update"])
        self.assertEqual(plan.iloc[0]["update_reason"], "no_baseline")

    def test_post_error(self):
        """Pre ok, post error -> skip."""
        shapes = [(1, 1024, 512)]
        pre = [_make_bench_result("(1,1024,512)", "ok", 100.0)]
        post = [_make_bench_result("(1,1024,512)", "error:crash", -1)]
        plan = self._build_plan(shapes, pre, post, threshold=3.0)
        self.assertEqual(len(plan), 1)
        self.assertFalse(plan.iloc[0]["update"])
        self.assertEqual(plan.iloc[0]["update_reason"], "skip")

    def test_both_error(self):
        """Both pre and post error -> skip."""
        shapes = [(1, 1024, 512)]
        pre = [_make_bench_result("(1,1024,512)", "error:crash", -1)]
        post = [_make_bench_result("(1,1024,512)", "error:crash", -1)]
        plan = self._build_plan(shapes, pre, post, threshold=3.0)
        self.assertEqual(len(plan), 1)
        self.assertFalse(plan.iloc[0]["update"])

    def test_empty_pre(self):
        """Empty pre results -> empty plan."""
        shapes = [(1, 1024, 512)]
        post = [_make_bench_result("(1,1024,512)", "ok", 50.0)]
        plan = self._build_plan(shapes, [], post, threshold=3.0)
        self.assertTrue(plan.empty)

    def test_empty_post(self):
        """Empty post results -> empty plan."""
        shapes = [(1, 1024, 512)]
        pre = [_make_bench_result("(1,1024,512)", "ok", 100.0)]
        plan = self._build_plan(shapes, pre, [], threshold=3.0)
        self.assertTrue(plan.empty)

    def test_multi_shape_mixed(self):
        """Multiple shapes: one improved, one below threshold, one error."""
        shapes = [(1, 1024, 512), (32, 2048, 1024), (64, 4096, 2048)]
        pre = [
            _make_bench_result("(1,1024,512)", "ok", 100.0),
            _make_bench_result("(32,2048,1024)", "ok", 100.0),
            _make_bench_result("(64,4096,2048)", "ok", 100.0),
        ]
        post = [
            _make_bench_result("(1,1024,512)", "ok", 50.0),  # 50% -> update
            _make_bench_result("(32,2048,1024)", "ok", 99.0),  # 1% -> skip
            _make_bench_result("(64,4096,2048)", "error:crash", -1),  # error -> skip
        ]
        plan = self._build_plan(shapes, pre, post, threshold=3.0)
        self.assertEqual(len(plan), 3)
        updates = plan[plan["update"]]
        skips = plan[~plan["update"]]
        self.assertEqual(len(updates), 1)
        self.assertEqual(int(updates.iloc[0]["M"]), 1)
        self.assertEqual(len(skips), 2)

    def test_regression_zero_speedup(self):
        """Post is slower than pre -> skip (negative improvement)."""
        shapes = [(1, 1024, 512)]
        pre = [_make_bench_result("(1,1024,512)", "ok", 50.0)]
        post = [_make_bench_result("(1,1024,512)", "ok", 100.0)]
        plan = self._build_plan(shapes, pre, post, threshold=3.0)
        self.assertEqual(len(plan), 1)
        self.assertFalse(plan.iloc[0]["update"])
        self.assertLess(plan.iloc[0]["improvement_pct"], 0)


# ---------------------------------------------------------------------------
# _merge_compare_filtered_results
# ---------------------------------------------------------------------------
class TestMergeCompareFilteredResults(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_row(self, m, n, k, kid, us):
        return [TEST_GFX, TEST_CU, m, n, k, kid, 0, us, f"k{kid}", 1.0, 1.0, 0.01]

    def _write_base_and_candidate(self, base_rows, candidate_rows):
        base_path = os.path.join(self.tmpdir, "base.csv")
        candidate_path = os.path.join(self.tmpdir, "candidate.csv")
        _make_tuned_csv(base_path, base_rows)
        _make_tuned_csv(candidate_path, candidate_rows)
        return base_path, candidate_path

    def _make_comparison(self, entries):
        """Build a comparison DataFrame from list of dicts.
        Each dict: {M, N, K, update, update_reason}"""
        rows = []
        for e in entries:
            row = {
                "gfx": TEST_GFX,
                "cu_num": TEST_CU,
                "M": e["M"],
                "N": e["N"],
                "K": e["K"],
                "shape": f"({e['M']},{e['N']},{e['K']})",
                "pre_us": e.get("pre_us", 100.0),
                "post_us": e.get("post_us", 50.0),
                "pre_status": "ok",
                "post_status": "ok",
                "improvement_pct": e.get("improvement_pct", 50.0),
                "update": e["update"],
                "update_reason": e["update_reason"],
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def test_improved_shapes_replaced(self):
        """Improved shape is taken from candidate CSV."""
        base_path, candidate_path = self._write_base_and_candidate(
            base_rows=[self._make_row(1, 1024, 512, 0, 100.0)],
            candidate_rows=[self._make_row(1, 1024, 512, 1, 50.0)],
        )
        comparison = self._make_comparison(
            [
                {
                    "M": 1,
                    "N": 1024,
                    "K": 512,
                    "update": True,
                    "update_reason": "threshold_met",
                },
            ]
        )
        tuner = _StubTuner.get()
        merged = tuner._merge_compare_filtered_results(
            base_path, candidate_path, comparison
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(float(merged.iloc[0]["us"]), 50.0)

    def test_non_improved_kept(self):
        """Non-improved shape keeps base value."""
        base_path, candidate_path = self._write_base_and_candidate(
            base_rows=[self._make_row(1, 1024, 512, 0, 100.0)],
            candidate_rows=[self._make_row(1, 1024, 512, 1, 99.0)],
        )
        comparison = self._make_comparison(
            [
                {"M": 1, "N": 1024, "K": 512, "update": False, "update_reason": "skip"},
            ]
        )
        tuner = _StubTuner.get()
        merged = tuner._merge_compare_filtered_results(
            base_path, candidate_path, comparison
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(float(merged.iloc[0]["us"]), 100.0)

    def test_mixed_update_and_skip(self):
        """Two shapes: one improved, one not."""
        base_path, candidate_path = self._write_base_and_candidate(
            base_rows=[
                self._make_row(1, 1024, 512, 0, 100.0),
                self._make_row(32, 2048, 1024, 0, 200.0),
            ],
            candidate_rows=[
                self._make_row(1, 1024, 512, 1, 50.0),
                self._make_row(32, 2048, 1024, 1, 199.0),
            ],
        )
        comparison = self._make_comparison(
            [
                {
                    "M": 1,
                    "N": 1024,
                    "K": 512,
                    "update": True,
                    "update_reason": "threshold_met",
                },
                {
                    "M": 32,
                    "N": 2048,
                    "K": 1024,
                    "update": False,
                    "update_reason": "skip",
                },
            ]
        )
        tuner = _StubTuner.get()
        merged = tuner._merge_compare_filtered_results(
            base_path, candidate_path, comparison
        )
        self.assertEqual(len(merged), 2)
        row_1 = merged[merged["M"].astype(int) == 1].iloc[0]
        row_32 = merged[merged["M"].astype(int) == 32].iloc[0]
        self.assertEqual(float(row_1["us"]), 50.0)
        self.assertEqual(float(row_32["us"]), 200.0)

    def test_no_updates_returns_base(self):
        """Comparison all skip -> returns base unchanged."""
        base_path, candidate_path = self._write_base_and_candidate(
            base_rows=[self._make_row(1, 1024, 512, 0, 100.0)],
            candidate_rows=[self._make_row(1, 1024, 512, 1, 99.0)],
        )
        comparison = self._make_comparison(
            [
                {"M": 1, "N": 1024, "K": 512, "update": False, "update_reason": "skip"},
            ]
        )
        tuner = _StubTuner.get()
        merged = tuner._merge_compare_filtered_results(
            base_path, candidate_path, comparison
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(float(merged.iloc[0]["us"]), 100.0)

    def test_empty_comparison_returns_base(self):
        """Empty comparison DataFrame -> returns base unchanged."""
        base_path, candidate_path = self._write_base_and_candidate(
            base_rows=[self._make_row(1, 1024, 512, 0, 100.0)],
            candidate_rows=[self._make_row(1, 1024, 512, 1, 50.0)],
        )
        tuner = _StubTuner.get()
        merged = tuner._merge_compare_filtered_results(
            base_path, candidate_path, pd.DataFrame()
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(float(merged.iloc[0]["us"]), 100.0)

    def test_candidate_file_missing(self):
        """Candidate file doesn't exist -> returns base."""
        base_path = os.path.join(self.tmpdir, "base.csv")
        _make_tuned_csv(base_path, [self._make_row(1, 1024, 512, 0, 100.0)])
        comparison = self._make_comparison(
            [
                {
                    "M": 1,
                    "N": 1024,
                    "K": 512,
                    "update": True,
                    "update_reason": "threshold_met",
                },
            ]
        )
        tuner = _StubTuner.get()
        merged = tuner._merge_compare_filtered_results(
            base_path, os.path.join(self.tmpdir, "nonexistent.csv"), comparison
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(float(merged.iloc[0]["us"]), 100.0)

    def test_base_has_extra_shapes(self):
        """Base has shapes not in comparison -> they are preserved."""
        base_path, candidate_path = self._write_base_and_candidate(
            base_rows=[
                self._make_row(1, 1024, 512, 0, 100.0),
                self._make_row(64, 4096, 2048, 0, 300.0),
            ],
            candidate_rows=[
                self._make_row(1, 1024, 512, 1, 50.0),
            ],
        )
        comparison = self._make_comparison(
            [
                {
                    "M": 1,
                    "N": 1024,
                    "K": 512,
                    "update": True,
                    "update_reason": "threshold_met",
                },
            ]
        )
        tuner = _StubTuner.get()
        merged = tuner._merge_compare_filtered_results(
            base_path, candidate_path, comparison
        )
        self.assertEqual(len(merged), 2)
        row_64 = merged[merged["M"].astype(int) == 64]
        self.assertEqual(len(row_64), 1)
        self.assertEqual(float(row_64.iloc[0]["us"]), 300.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
