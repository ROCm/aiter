# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression tests for runtime untuned-shape recording (no GPU required)."""

import builtins
import functools
import multiprocessing
import os
import tempfile
import unittest
from unittest import mock

from aiter.utility import untuned_shapes


def _record_in_worker(rows):
    from aiter.utility import untuned_shapes as worker_recorder

    worker_recorder._SEEN.clear()
    for row in rows:
        worker_recorder.record("a8w8_tuned_gemm.csv", row)


class TestUntunedShapes(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.env = mock.patch.dict(
            os.environ,
            {
                "AITER_TUNE_GEMM": "1",
                "AITER_TUNE_GEMM_DIR": self.tempdir.name,
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        untuned_shapes._ENABLED = None
        untuned_shapes._SEEN.clear()

    def tearDown(self):
        untuned_shapes._ENABLED = None
        untuned_shapes._SEEN.clear()

    def test_append_adds_separator_to_existing_file(self):
        path = os.path.join(self.tempdir.name, "a8w8_untuned_gemm.csv")
        with open(path, "w") as fh:
            fh.write("M,N,K")

        untuned_shapes.record("a8w8_tuned_gemm.csv", {"M": 1, "N": 2, "K": 3})

        with open(path) as fh:
            self.assertEqual(fh.read(), "M,N,K\n1,2,3\n")

    def test_failed_append_remains_retryable(self):
        row = {"M": 1, "N": 2, "K": 3}
        path = os.path.join(self.tempdir.name, "a8w8_untuned_gemm.csv")
        real_open = builtins.open
        failed_once = False

        def fail_first_append(file, mode="r", *args, **kwargs):
            nonlocal failed_once
            if os.fspath(file) == path and mode == "a" and not failed_once:
                failed_once = True
                raise OSError("transient failure")
            return real_open(file, mode, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=fail_first_append):
            untuned_shapes.record("a8w8_tuned_gemm.csv", row)

        untuned_shapes.record("a8w8_tuned_gemm.csv", row)

        with open(path) as fh:
            self.assertEqual(fh.read(), "M,N,K\n1,2,3\n")

    def test_failed_initialization_remains_retryable(self):
        row = {"M": 1, "N": 2, "K": 3}
        path = os.path.join(self.tempdir.name, "a8w8_untuned_gemm.csv")
        real_open = builtins.open
        failed_once = False

        def fail_first_create(file, mode="r", *args, **kwargs):
            nonlocal failed_once
            if os.fspath(file) == path and mode == "w" and not failed_once:
                failed_once = True
                raise OSError("transient failure")
            return real_open(file, mode, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=fail_first_create):
            untuned_shapes.record("a8w8_tuned_gemm.csv", row)

        self.assertNotIn(path, untuned_shapes._SEEN)
        untuned_shapes.record("a8w8_tuned_gemm.csv", row)
        with open(path) as fh:
            self.assertEqual(fh.read(), "M,N,K\n1,2,3\n")

    def test_processes_do_not_lose_or_duplicate_rows(self):
        shared = {"M": 1, "N": 2, "K": 3}
        processes = []
        for index in range(4):
            unique = {"M": index + 10, "N": 20, "K": 30}
            process = multiprocessing.Process(
                target=_record_in_worker, args=([shared, unique],)
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join(10)
            self.assertEqual(process.exitcode, 0)

        path = os.path.join(self.tempdir.name, "a8w8_untuned_gemm.csv")
        with open(path) as fh:
            lines = fh.read().splitlines()
        self.assertEqual(lines[0], "M,N,K")
        self.assertEqual(len(lines), 6)
        self.assertEqual(len(set(lines[1:])), 5)


class TestCachedLookupMissRecording(unittest.TestCase):

    def _assert_retry_outside_cache(self, module, cached_name, lookup, args):
        resolver = mock.Mock(return_value=None)
        cached_resolver = functools.lru_cache(maxsize=1)(resolver)
        with (
            mock.patch.object(module, cached_name, cached_resolver),
            mock.patch.object(module, "_record_untuned_shape") as record,
        ):
            lookup(*args)
            lookup(*args)
        self.assertEqual(resolver.call_count, 1)
        self.assertEqual(record.call_count, 2)

    def test_a8w8_misses_record_outside_lookup_caches(self):
        from aiter.ops import gemm_op_a8w8

        self._assert_retry_outside_cache(
            gemm_op_a8w8,
            "_get_CKGEMM_config_cached",
            gemm_op_a8w8.get_CKGEMM_config,
            (1, 2, 3, "tuned.csv"),
        )
        self._assert_retry_outside_cache(
            gemm_op_a8w8,
            "_get_GEMM_config_with_quant_type_cached",
            gemm_op_a8w8.get_GEMM_config_with_quant_type,
            (1, 2, 3, "fp8", "tuned.csv"),
        )

    def test_a4w4_misses_record_outside_lookup_cache(self):
        from aiter.ops import gemm_op_a4w4

        self._assert_retry_outside_cache(
            gemm_op_a4w4,
            "_get_GEMM_config_cached",
            gemm_op_a4w4.get_GEMM_config,
            (1, 2, 3),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
