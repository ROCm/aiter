# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression tests for process-local untuned-shape recording."""

import functools
import multiprocessing
import os
import tempfile
import unittest
from unittest import mock

from aiter.utility import untuned_shapes


def _record_in_worker(directory, row):
    os.environ["AITER_TUNE_GEMM"] = "1"
    os.environ["AITER_TUNE_GEMM_DIR"] = directory
    untuned_shapes._ENABLED = None
    untuned_shapes._SEEN.clear()
    untuned_shapes.record("a8w8_tuned_gemm.csv", row)


class TestUntunedShapes(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.env = mock.patch.dict(
            os.environ,
            {"AITER_TUNE_GEMM": "1", "AITER_TUNE_GEMM_DIR": self.tempdir.name},
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        untuned_shapes._ENABLED = None
        untuned_shapes._SEEN.clear()

    def tearDown(self):
        untuned_shapes._ENABLED = None
        untuned_shapes._SEEN.clear()

    def _path(self, name="a8w8_tuned_gemm.csv"):
        return untuned_shapes.untuned_path_for(name)

    def test_model_directory_uses_a_process_shard_per_family(self):
        model_dir = os.path.join(self.tempdir.name, "tuning", "glm-5.2")
        with mock.patch.dict(os.environ, {"AITER_TUNE_GEMM_DIR": model_dir}):
            paths = [
                untuned_shapes.untuned_path_for(name)
                for name in (
                    "a8w8_tuned_gemm.csv",
                    "a8w8_bpreshuffle_tuned_gemm.csv",
                    "a8w8_blockscale_tuned_gemm.csv",
                    "a4w4_blockscale_tuned_gemm.csv",
                )
            ]
        self.assertEqual(
            paths,
            [
                os.path.join(
                    model_dir,
                    f"{name.replace('_tuned_', '_untuned_')[:-4]}.{os.getpid()}.csv",
                )
                for name in (
                    "a8w8_tuned_gemm.csv",
                    "a8w8_bpreshuffle_tuned_gemm.csv",
                    "a8w8_blockscale_tuned_gemm.csv",
                    "a4w4_blockscale_tuned_gemm.csv",
                )
            ],
        )

    def test_default_destination_is_not_the_package(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                untuned_shapes.os, "getcwd", return_value=self.tempdir.name
            ),
        ):
            path = untuned_shapes.untuned_path_for("bf16_tuned_gemm.csv")
        self.assertEqual(
            path,
            os.path.join(self.tempdir.name, f"bf16_untuned_gemm.{os.getpid()}.csv"),
        )

    def test_same_process_deduplicates_rows(self):
        row = {"M": 1, "N": 2, "K": 3}
        untuned_shapes.record("a8w8_tuned_gemm.csv", row)
        untuned_shapes.record("a8w8_tuned_gemm.csv", row)
        with open(self._path()) as file:
            self.assertEqual(file.read(), "M,N,K\n1,2,3\n")

    def test_short_write_finishes_the_local_row(self):
        path = self._path()
        real_write = os.write
        short_once = True

        def write_part(fd, data):
            nonlocal short_once
            if short_once and data == b"1,2,3\n":
                short_once = False
                return real_write(fd, data[:2])
            return real_write(fd, data)

        with mock.patch.object(untuned_shapes.os, "write", side_effect=write_part):
            untuned_shapes.record("a8w8_tuned_gemm.csv", {"M": 1, "N": 2, "K": 3})
        with open(path) as file:
            self.assertEqual(file.read(), "M,N,K\n1,2,3\n")

    def test_workers_write_independent_complete_shards(self):
        workers = []
        for value in range(4):
            worker = multiprocessing.Process(
                target=_record_in_worker,
                args=(self.tempdir.name, {"M": value, "N": 2, "K": 3}),
            )
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join(10)
            self.assertEqual(worker.exitcode, 0)
        shards = sorted(
            path
            for path in os.listdir(self.tempdir.name)
            if path.startswith("a8w8_untuned_gemm.")
        )
        self.assertEqual(len(shards), 4)
        for shard in shards:
            with open(os.path.join(self.tempdir.name, shard)) as file:
                self.assertEqual(len(file.read().splitlines()), 2)

    def test_bf16_uses_the_shared_recorder(self):
        from aiter import tuned_gemm

        with mock.patch.object(tuned_gemm, "_record_untuned_shape") as record:
            tuned_gemm.save_shapes(
                1, 2, 3, None, "torch.bfloat16", "torch.bfloat16", False, False
            )
        self.assertEqual(record.call_args.args[0], tuned_gemm.tune_path)
        self.assertEqual(record.call_args.args[1]["M"], 1)
        self.assertFalse(record.call_args.args[1]["bias"])


class TestCachedLookupMissRecording(unittest.TestCase):
    def _assert_retry_outside_cache(self, module, cached_name, log_name, lookup, args):
        resolver = mock.Mock(return_value=None)
        cached_resolver = functools.lru_cache(maxsize=1)(resolver)
        miss_logger = mock.Mock()
        cached_logger = functools.lru_cache(maxsize=1)(miss_logger)
        with (
            mock.patch.object(module, cached_name, cached_resolver),
            mock.patch.object(module, log_name, cached_logger),
            mock.patch.object(module, "_record_untuned_shape") as record,
        ):
            lookup(*args)
            lookup(*args)
        self.assertEqual(resolver.call_count, 1)
        self.assertEqual(miss_logger.call_count, 1)
        self.assertEqual(record.call_count, 2)

    def test_a8w8_misses_record_outside_lookup_caches(self):
        from aiter.ops import gemm_op_a8w8

        self._assert_retry_outside_cache(
            gemm_op_a8w8,
            "_get_CKGEMM_config_cached",
            "_log_CKGEMM_miss_once",
            gemm_op_a8w8.get_CKGEMM_config,
            (1, 2, 3, "tuned.csv"),
        )

    def test_a8w8_default_miss_uses_the_tuned_file_schema(self):
        from aiter.ops import gemm_op_a8w8

        with (
            mock.patch.object(
                gemm_op_a8w8, "_get_CKGEMM_config_cached", return_value=None
            ),
            mock.patch.object(gemm_op_a8w8, "_log_CKGEMM_miss_once"),
            mock.patch.object(gemm_op_a8w8, "_record_untuned_shape") as record,
        ):
            gemm_op_a8w8.get_CKGEMM_config(1, 2, 3)

        self.assertEqual(record.call_args.args[1], {"M": 1, "N": 2, "K": 3})

    def test_a4w4_misses_record_outside_lookup_cache(self):
        from aiter.ops import gemm_op_a4w4

        self._assert_retry_outside_cache(
            gemm_op_a4w4,
            "_get_GEMM_config_cached",
            "_log_GEMM_miss_once",
            gemm_op_a4w4.get_GEMM_config,
            (1, 2, 3),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
