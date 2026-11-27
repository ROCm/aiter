# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
A GEMM tuned-config miss is logged once per non-M key (and per fallback it
picks), or on every miss with AITER_LOG_TUNED_CONFIG=1, and untuned-shape
recording does not depend on it.

``log_tuned_config_miss`` is the one policy every GEMM config lookup shares.
The lookups are exercised with empty tuned tables, so no GPU, ROCm or tuned CSV
is needed.

Run:
    python3 -m unittest op_tests.tuning_tests.test_tuned_config_miss_log -v
"""

import functools
import threading
import time
import unittest
from collections import OrderedDict
from unittest import mock

import aiter.ops.gemm_op_common as common

GFX, CU_NUM = "gfx942", 304
MISS = "not found tuned config"


class _MissLogCase(unittest.TestCase):
    def setUp(self):
        saved = OrderedDict(common._REPORTED_TUNED_CONFIG_MISSES)
        common._REPORTED_TUNED_CONFIG_MISSES.clear()
        self.addCleanup(common._REPORTED_TUNED_CONFIG_MISSES.update, saved)
        self.addCleanup(common._REPORTED_TUNED_CONFIG_MISSES.clear)

    def set_log_all(self, value):
        p = mock.patch.object(common, "AITER_LOG_TUNED_CONFIG", value)
        p.start()
        self.addCleanup(p.stop)

    def misses(self, fn, ms, marker=MISS):
        """Lines with ``marker`` that fn(M) logged for each M in ms, in order."""
        lines = []
        for m in ms:
            with self.assertLogs("aiter", "INFO") as cm:
                # assertLogs fails on silence; a sentinel keeps it quiet-safe.
                common.logger.info("sentinel")
                fn(m)
            lines += [r for r in cm.output if marker in r]
        return lines


class TestLogTunedConfigMiss(_MissLogCase):
    def log(self, m, n=256, k=6144, family="fam"):
        return common.log_tuned_config_miss(
            family, (GFX, n, k), f"shape is M:{m}, N:{n}, K:{k}, {MISS}"
        )

    def test_default_logs_once_per_non_m_key(self):
        self.set_log_all(0)
        lines = self.misses(self.log, range(1, 301))
        self.assertEqual(len(lines), 1)
        self.assertIn("M:1,", lines[0])  # first miss, with its full shape
        self.assertIn("AITER_LOG_TUNED_CONFIG=1", lines[0])

    def test_return_value_reports_whether_logged(self):
        self.set_log_all(0)
        self.assertTrue(self.log(1))
        self.assertFalse(self.log(2))

    def test_distinct_non_m_keys_each_log_once(self):
        self.set_log_all(0)
        lines = []
        for family, n, k in [("a", 32, 6144), ("a", 256, 6144), ("b", 32, 6144)]:
            log = functools.partial(self.log, n=n, k=k, family=family)
            lines += self.misses(log, (1, 2, 3))
        self.assertEqual(len(lines), 3)

    def test_log_tuned_config_logs_every_miss_without_suffix(self):
        self.set_log_all(1)
        lines = self.misses(self.log, range(1, 11))
        self.assertEqual(len(lines), 10)
        self.assertTrue(all("Not repeated" not in line for line in lines))
        self.assertTrue(common.log_tuned_config_miss("fam", (GFX, 1, 1), "x"))
        self.assertTrue(common.log_tuned_config_miss("fam", (GFX, 1, 1), "x"))

    def test_reported_set_is_bounded_and_evicts_least_recently_seen(self):
        self.set_log_all(0)
        with mock.patch.object(common, "_MAX_REPORTED_TUNED_CONFIG_MISSES", 3):
            for n in (1, 2, 3):
                self.assertTrue(self.log(1, n=n))
            self.assertFalse(self.log(2, n=1))  # seen again: n=1 is now the newest
            self.assertTrue(self.log(1, n=4))  # evicts n=2, the least recent
            self.assertEqual(len(common._REPORTED_TUNED_CONFIG_MISSES), 3)
            self.assertFalse(self.log(2, n=1))  # still remembered
            self.assertTrue(self.log(2, n=2))  # evicted, so logged again
            for n in range(10, 100):
                self.log(1, n=n)
            self.assertEqual(len(common._REPORTED_TUNED_CONFIG_MISSES), 3)

    def test_concurrent_first_misses_log_exactly_once(self):
        self.set_log_all(0)

        class SlowLookup(OrderedDict):
            # Widen the window between the membership check and the insert, so
            # a check-then-insert without a lock lets every thread through.
            def __contains__(self, item):
                found = super().__contains__(item)
                time.sleep(0.02)
                return found

        threads_n = 8
        barrier = threading.Barrier(threads_n)
        results = []

        def worker():
            barrier.wait()
            results.append(self.log(1))

        with mock.patch.object(
            common, "_REPORTED_TUNED_CONFIG_MISSES", SlowLookup()
        ), self.assertLogs("aiter", "INFO") as cm:
            common.logger.info("sentinel")
            threads = [threading.Thread(target=worker) for _ in range(threads_n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(sorted(results), [False] * (threads_n - 1) + [True])
        self.assertEqual(len([r for r in cm.output if MISS in r]), 1)


class TestLookupsUseThePolicy(_MissLogCase):
    """Real lookups on an empty table: one line per (N, K), all of them with the flag."""

    def check(self, call, log_all=0):
        self.set_log_all(log_all)
        return self.misses(call, range(1, 41))

    def patch_runtime(self, module):
        for name, value in (
            ("get_gfx", lambda *a: GFX),
            ("get_cu_num", lambda *a: CU_NUM),
            ("get_padded_m", lambda m, _n, _k, _gl: m),
        ):
            p = mock.patch.object(module, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_a4w4(self):
        import aiter.ops.gemm_op_a4w4 as mod

        self.patch_runtime(mod)
        for name, value in (("gemm_dict", {}), ("has_gfx", True)):
            p = mock.patch.object(mod.get_GEMM_config, name, value, create=True)
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(mod.get_GEMM_config.cache_clear)
        for log_all, expected in ((0, 1), (1, 40)):
            mod.get_GEMM_config.cache_clear()
            common._REPORTED_TUNED_CONFIG_MISSES.clear()
            lines = self.check(lambda m: mod.get_GEMM_config(m, 256, 6144), log_all)
            self.assertEqual(len(lines), expected, (log_all, lines[:2]))

    def test_a8w8_ck(self):
        import aiter.ops.gemm_op_a8w8 as mod

        self.patch_runtime(mod)
        tuned_file = "none_a8w8.csv"
        for table, value in (
            (mod._CKGEMM_CONFIG_CACHE, {}),
            (mod._CKGEMM_HAS_GFX, True),
        ):
            p = mock.patch.dict(table, {tuned_file: value})
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(mod.get_CKGEMM_config.cache_clear)
        for log_all, expected in ((0, 1), (1, 40)):
            mod.get_CKGEMM_config.cache_clear()
            common._REPORTED_TUNED_CONFIG_MISSES.clear()
            lines = self.check(
                lambda m: mod.get_CKGEMM_config(m, 256, 6144, tuned_file), log_all
            )
            self.assertEqual(len(lines), expected, (log_all, lines[:2]))
        # A second N/K logs again by default.
        mod.get_CKGEMM_config.cache_clear()
        common._REPORTED_TUNED_CONFIG_MISSES.clear()
        self.set_log_all(0)
        lines = self.misses(
            lambda m: (
                mod.get_CKGEMM_config(m, 256, 6144, tuned_file),
                mod.get_CKGEMM_config(m, 32, 6144, tuned_file),
            ),
            range(1, 6),
        )
        self.assertEqual(len(lines), 2)

    def test_a16w16_logs_again_when_the_fallback_backend_changes(self):
        # gfx942 / 304 CUs, N=32, K=6144, bf16: the default is skinny for M<=4
        # and torch above, so the one N/K has two defaults and logs one line each.
        try:
            import aiter.tuned_gemm as tuned
        except ImportError as exc:  # needs the ROCm build of aiter
            self.skipTest(f"aiter.tuned_gemm is not importable here: {exc}")
        bf16 = "torch.bfloat16"
        self.addCleanup(tuned.get_GEMM_A16W16_config.cache_clear)

        def lookup(m):
            return tuned.get_GEMM_A16W16_config(
                m, 32, 6144, False, bf16, bf16, False, False, GFX, CU_NUM
            )

        p = mock.patch.object(tuned, "get_GEMM_A16W16_tuned_config", lambda *a: None)
        p.start()
        self.addCleanup(p.stop)
        self.assertEqual(lookup(1)["libtype"], "skinny")
        self.assertEqual(lookup(8)["libtype"], "torch")
        for log_all, expected in ((0, 2), (1, 16)):
            tuned.get_GEMM_A16W16_config.cache_clear()
            common._REPORTED_TUNED_CONFIG_MISSES.clear()
            self.set_log_all(log_all)
            lines = self.misses(lookup, range(1, 17))
            self.assertEqual(len(lines), expected, (log_all, lines[:3]))
            if not log_all:
                self.assertIn("skinny", lines[0])
                self.assertIn("torch", lines[1])

    def test_mxfp8_logs_again_for_a_different_explicit_splitk(self):
        import aiter.ops.gemm_op_a8w8 as mod

        self.patch_runtime(mod)
        tuned_file = "none_mxfp8.csv"
        p = mock.patch.dict(mod._MXFP8_GEMM_CONFIG_CACHE, {tuned_file: {}})
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(mod.get_mxfp8_gemm_config.cache_clear)

        def lookup(m, splitk):
            return mod.get_mxfp8_gemm_config(
                m, 256, 6144, False, tuned_file=tuned_file, splitk=splitk
            )

        self.set_log_all(0)
        mod.get_mxfp8_gemm_config.cache_clear()
        lines = []
        for splitk in (None, 2, 4):
            lines += self.misses(
                functools.partial(lookup, splitk=splitk), (1, 2, 3), "not found usable"
            )
        self.assertEqual(len(lines), 3)
        again = self.misses(
            functools.partial(lookup, splitk=2), (4, 5), "not found usable"
        )
        self.assertEqual(again, [])

    def test_untuned_shape_recording_is_independent_of_the_miss_log(self):
        # AITER_TUNE_GEMM records untuned shapes in save_shapes, not in the
        # lookup that logs the miss: it writes the row with the log deduplicated
        # or silenced, and a miss lookup leaves the recorded table alone.
        try:
            import aiter.tuned_gemm as tuned
        except ImportError as exc:  # needs the ROCm build of aiter
            self.skipTest(f"aiter.tuned_gemm is not importable here: {exc}")
        import os
        import tempfile

        import pandas as pd

        self.set_log_all(0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bf16_untuned_gemm.csv")
            for p in (
                mock.patch.object(tuned, "untune_path", path),
                mock.patch.object(tuned, "tuned_df", tuned.tuned_df.iloc[0:0]),
                mock.patch.dict(os.environ, {"AITER_TUNE_GEMM": "1"}),
                mock.patch.object(common.logger, "info", lambda *a, **k: None),
            ):
                p.start()
                self.addCleanup(p.stop)
            for m in (1, 2, 3):  # the miss log is silent after the first
                common.log_tuned_config_miss("a16w16", (GFX, 32, 6144), "x")
                tuned.save_shapes(m, 32, 6144, None, "bf16", "bf16", False, False)
            rows = pd.read_csv(path)
        self.assertEqual(sorted(rows["M"]), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
