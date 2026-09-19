# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression tests for runtime untuned-shape recording (no GPU required)."""

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

    def test_same_process_deduplicates_rows(self):
        row = {"M": 1, "N": 2, "K": 3}

        untuned_shapes.record("a8w8_tuned_gemm.csv", row)
        untuned_shapes.record("a8w8_tuned_gemm.csv", row)

        path = os.path.join(self.tempdir.name, "a8w8_untuned_gemm.csv")
        with open(path) as fh:
            self.assertEqual(fh.read(), "M,N,K\n1,2,3\n")

    def test_model_directory_groups_family_files(self):
        model_dir = os.path.join(self.tempdir.name, "tuning", "glm-5.2")
        tuned_files = (
            "a8w8_tuned_gemm.csv",
            "a8w8_bpreshuffle_tuned_gemm.csv",
            "a8w8_blockscale_tuned_gemm.csv",
            "a4w4_blockscale_tuned_gemm.csv",
        )

        with mock.patch.dict(os.environ, {"AITER_TUNE_GEMM_DIR": model_dir}):
            paths = [untuned_shapes.untuned_path_for(path) for path in tuned_files]

        self.assertEqual(
            paths,
            [
                os.path.join(model_dir, path.replace("_tuned_", "_untuned_"))
                for path in tuned_files
            ],
        )

    def test_failed_append_remains_retryable(self):
        row = {"M": 1, "N": 2, "K": 3}
        path = os.path.join(self.tempdir.name, "a8w8_untuned_gemm.csv")
        real_append = untuned_shapes._append_line
        failed_once = False

        def fail_first_append(file, payload):
            nonlocal failed_once
            if os.fspath(file) == path and not failed_once:
                failed_once = True
                raise OSError("transient failure")
            return real_append(file, payload)

        with mock.patch.object(
            untuned_shapes, "_append_line", side_effect=fail_first_append
        ):
            untuned_shapes.record("a8w8_tuned_gemm.csv", row)

        untuned_shapes.record("a8w8_tuned_gemm.csv", row)

        with open(path) as fh:
            self.assertEqual(fh.read(), "M,N,K\n1,2,3\n")

    def _short_write_once(self, path, keep):
        """Make the first append to ``path`` persist ``keep`` bytes, then stop.

        The kernel is entitled to return a short count -- an exhausted disk or
        quota is the usual cause -- and the bytes it did accept are already in
        the file. Every other write, including the newline that closes the
        fragment, is left alone.
        """
        real_write = os.write
        tripped = False
        target = os.path.realpath(path)

        def short_first_write(fd, payload):
            nonlocal tripped
            if tripped or len(payload) <= keep:
                return real_write(fd, payload)
            try:
                same = os.path.realpath(f"/proc/self/fd/{fd}") == target
            except OSError:
                same = False
            if not same:
                return real_write(fd, payload)
            tripped = True
            return real_write(fd, payload[:keep])

        return mock.patch.object(untuned_shapes.os, "write", short_first_write)

    def test_partial_append_does_not_splice_into_the_retried_row(self):
        """A torn row must not merge with its own retry.

        Without isolation, a row that reaches the file as ``1,`` and is then
        retried whole produces ``1,1,2,3`` under a three-column header -- a
        malformed row the tuner's deduplication cannot repair, because it is
        not a duplicate of anything.
        """
        row = {"M": 1, "N": 2, "K": 3}
        path = os.path.join(self.tempdir.name, "a8w8_untuned_gemm.csv")

        with self._short_write_once(path, keep=2):
            untuned_shapes.record("a8w8_tuned_gemm.csv", row)

        untuned_shapes.record("a8w8_tuned_gemm.csv", row)

        with open(path) as fh:
            lines = fh.read().splitlines()

        self.assertEqual(lines[0], "M,N,K")
        self.assertIn("1,2,3", lines)
        # The fragment may survive on a line of its own; what it must never do
        # is take the retried row with it.
        for line in lines[1:]:
            self.assertLessEqual(
                len(line.split(",")), 3, f"row {line!r} has more fields than the header"
            )

    def test_partial_append_is_readable_by_the_tuner(self):
        """Whatever the file looks like afterwards, pandas must still parse it.

        This is the property that actually matters: the untuned CSV is fed
        straight back to the tuner, so a torn write may cost a row but must
        not cost the file.
        """
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - pandas is a tuner dependency
            self.skipTest("pandas is not installed")
        path = os.path.join(self.tempdir.name, "a8w8_untuned_gemm.csv")

        with self._short_write_once(path, keep=2):
            untuned_shapes.record("a8w8_tuned_gemm.csv", {"M": 1, "N": 2, "K": 3})

        untuned_shapes.record("a8w8_tuned_gemm.csv", {"M": 1, "N": 2, "K": 3})
        untuned_shapes.record("a8w8_tuned_gemm.csv", {"M": 4, "N": 5, "K": 6})

        frame = pd.read_csv(path, skip_blank_lines=True).dropna()
        recorded = {tuple(int(v) for v in r) for r in frame[["M", "N", "K"]].values}
        self.assertIn((1, 2, 3), recorded)
        self.assertIn((4, 5, 6), recorded)

    def test_partial_append_leaves_the_row_retryable(self):
        """A torn row is not cached, so the next dispatch of that shape retries."""
        row = {"M": 7, "N": 8, "K": 9}
        path = os.path.join(self.tempdir.name, "a8w8_untuned_gemm.csv")

        with self._short_write_once(path, keep=2):
            untuned_shapes.record("a8w8_tuned_gemm.csv", row)

        state = untuned_shapes._SEEN[path]
        self.assertNotIn(("7", "8", "9"), state["rows"])

        untuned_shapes.record("a8w8_tuned_gemm.csv", row)
        with open(path) as fh:
            self.assertIn("7,8,9", fh.read().splitlines())

    def test_failed_initialization_remains_retryable(self):
        row = {"M": 1, "N": 2, "K": 3}
        path = os.path.join(self.tempdir.name, "a8w8_untuned_gemm.csv")
        real_ensure_header = untuned_shapes._ensure_header
        failed_once = False

        def fail_first_create(file, cols):
            nonlocal failed_once
            if os.fspath(file) == path and not failed_once:
                failed_once = True
                raise OSError("transient failure")
            return real_ensure_header(file, cols)

        with mock.patch.object(
            untuned_shapes, "_ensure_header", side_effect=fail_first_create
        ):
            untuned_shapes.record("a8w8_tuned_gemm.csv", row)

        self.assertNotIn(path, untuned_shapes._SEEN)
        untuned_shapes.record("a8w8_tuned_gemm.csv", row)
        with open(path) as fh:
            self.assertEqual(fh.read(), "M,N,K\n1,2,3\n")

    def test_append_uses_one_operating_system_write(self):
        row = {"M": 1, "N": 2, "K": 3}
        path = os.path.join(self.tempdir.name, "a8w8_untuned_gemm.csv")
        real_write = os.write

        with mock.patch.object(os, "write", wraps=real_write) as write:
            untuned_shapes.record("a8w8_tuned_gemm.csv", row)

        write.assert_called_once()
        self.assertEqual(write.call_args.args[1], b"1,2,3\n")
        with open(path) as fh:
            self.assertEqual(fh.read(), "M,N,K\n1,2,3\n")

    def test_tuner_skips_an_incomplete_row(self):
        """The reader discards what the writer could not repair.

        A torn append survives as one short line. It cannot be fixed in place
        -- another worker may already have appended past it -- so the tuner
        drops it instead of tuning a shape with NaN dimensions.
        """
        from aiter.utility.base_tuner import TunerCommon

        path = os.path.join(self.tempdir.name, "a8w8_untuned_gemm.csv")
        with open(path, "w") as fh:
            fh.write("M,N,K\n1,\n1,2,3\n4,5,6\n")

        frame = TunerCommon.get_untuned_gemm_list(None, path)

        self.assertEqual(
            [tuple(int(v) for v in row) for row in frame[["M", "N", "K"]].values],
            [(1, 2, 3), (4, 5, 6)],
        )

    def test_processes_append_complete_rows_without_a_lock(self):
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
        self.assertEqual(len(lines), 9)
        self.assertEqual(len(set(lines[1:])), 5)
        self.assertTrue(all(len(line.split(",")) == 3 for line in lines[1:]))


class TestTunedTableKeying(unittest.TestCase):
    """A tuned table that cannot be indexed should say so, and say why."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

    def _write(self, name, rows, header):
        path = os.path.join(self.tempdir.name, name)
        with open(path, "w") as fh:
            fh.write(header + "\n")
            for r in rows:
                fh.write(r + "\n")
        return path

    def test_quant_keyed_table_names_the_right_lookup(self):
        """a8w8_tuned_gemm.csv holds an int8 and an fp8 row per shape.

        It has carried q_dtype_w since #1782 and is keyed on six columns;
        get_CKGEMM_config keys on five. Loading it there used to raise
        pandas' "DataFrame index must be unique for orient='index'", which
        names neither the file nor the reason.
        """
        from aiter.ops.gemm_op_a8w8 import _assert_uniquely_keyed
        import pandas as pd

        path = self._write(
            "a8w8_tuned_gemm.csv",
            [
                "gfx950,256,1,1280,8192,torch.int8,7,0,17.69,k_int8,1.0,1.0,0.0",
                "gfx950,256,1,1280,8192,torch.float8_e4m3fn,7,0,7.68,k_fp8,1.0,1.0,0.0",
            ],
            "gfx,cu_num,M,N,K,q_dtype_w,kernelId,splitK,us,kernelName,tflops,bw,errRatio",
        )
        frame = pd.read_csv(path).drop_duplicates()

        with self.assertRaises(ValueError) as caught:
            _assert_uniquely_keyed(frame, ["gfx", "cu_num", "M", "N", "K"], path)

        message = str(caught.exception)
        self.assertIn("a8w8_tuned_gemm.csv", message)
        self.assertIn("1 shape(s)", message)
        self.assertIn("q_dtype_w", message)
        self.assertIn("get_GEMM_config_with_quant_type", message)

    def test_uniquely_keyed_table_is_accepted(self):
        """The control: a table that IS uniquely keyed must not raise."""
        from aiter.ops.gemm_op_a8w8 import _assert_uniquely_keyed
        import pandas as pd

        path = self._write(
            "kv_b_tuned_gemm.csv",
            [
                "gfx950,256,1,7168,512,7,0,3.83,k_a,1.0,1.0,0.0",
                "gfx950,256,2,7168,512,7,0,3.84,k_b,1.0,1.0,0.0",
            ],
            "gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio",
        )
        frame = pd.read_csv(path).drop_duplicates()

        _assert_uniquely_keyed(frame, ["gfx", "cu_num", "M", "N", "K"], path)


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
        self._assert_retry_outside_cache(
            gemm_op_a8w8,
            "_get_GEMM_config_with_quant_type_cached",
            "_log_quant_type_miss_once",
            gemm_op_a8w8.get_GEMM_config_with_quant_type,
            (1, 2, 3, "fp8", "tuned.csv"),
        )

    def test_default_a8w8_destination_keeps_quant_schema(self):
        from aiter.ops import gemm_op_a8w8

        with (
            mock.patch.object(
                gemm_op_a8w8, "_get_CKGEMM_config_cached", return_value=None
            ),
            mock.patch.object(gemm_op_a8w8, "_log_CKGEMM_miss_once"),
            mock.patch.object(gemm_op_a8w8, "_record_untuned_shape") as record,
        ):
            gemm_op_a8w8.get_CKGEMM_config(1, 2, 3)

        row = record.call_args.args[1]
        self.assertEqual(list(row), ["M", "N", "K", "q_dtype_w"])
        self.assertEqual(row["q_dtype_w"], gemm_op_a8w8.dtypes.i8)

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
