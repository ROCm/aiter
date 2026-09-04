# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression tests for runtime untuned-shape recording (no GPU required)."""

import builtins
import os
import tempfile
import unittest
from unittest import mock

from aiter.utility import untuned_shapes


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
        real_open = builtins.open
        failed_once = False

        def fail_first_append(file, mode="r", *args, **kwargs):
            nonlocal failed_once
            if mode == "a" and not failed_once:
                failed_once = True
                raise OSError("transient failure")
            return real_open(file, mode, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=fail_first_append):
            untuned_shapes.record("a8w8_tuned_gemm.csv", row)

        untuned_shapes.record("a8w8_tuned_gemm.csv", row)

        path = os.path.join(self.tempdir.name, "a8w8_untuned_gemm.csv")
        with open(path) as fh:
            self.assertEqual(fh.read(), "M,N,K\n1,2,3\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
