# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests of the config merge method, without importing GPU backends."""

import ast
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


CORE_PATH = Path(__file__).resolve().parents[2] / "aiter" / "jit" / "core.py"


def _merge_method(root):
    # Execute the production method, not a copy of the merge logic. As in the
    # third-party clone tests, isolate the process lock; CSV IO remains real.
    tree = ast.parse(CORE_PATH.read_text(encoding="utf-8"))
    config = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AITER_CONFIG"
    )
    method = next(
        node
        for node in config.body
        if isinstance(node, ast.FunctionDef) and node.name == "update_config_files"
    )
    namespace = {
        "os": os,
        "logger": logging.getLogger(__name__),
        "AITER_ROOT_DIR": str(root),
        "mp_lock": lambda lock_path, callback: callback(),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(CORE_PATH), "exec"),
        namespace,
    )
    return namespace[method.name]


class TestConfigMergeFallback(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="aiter_merge_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.configs = self.root / "aiter" / "configs"
        self.configs.mkdir(parents=True)
        self.merge = _merge_method(self.root)
        patcher = mock.patch("tempfile.gettempdir", return_value=str(self.root))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _merge_rows(self, first, second):
        paths = [self.root / "first.csv", self.root / "second.csv"]
        for path, rows in zip(paths, (first, second)):
            rows.to_csv(path, index=False)
        original = [path.read_bytes() for path in paths]
        result = self.merge(None, os.pathsep.join(map(str, paths)), "test_tuned")
        self.assertEqual([path.read_bytes() for path in paths], original)
        return pd.read_csv(result)

    def test_removes_identical_rows_within_and_across_files(self):
        rows = pd.DataFrame({"M": [16, 16], "kernelId": [1, 1], "us": [5, 5]})
        result = self._merge_rows(rows, rows.iloc[:1])
        pd.testing.assert_frame_equal(result, rows.iloc[:1])

    def test_preserves_distinct_tuning_rows_and_order(self):
        first = pd.DataFrame({"M": [16, 16], "kernelId": [1, 2], "us": [5, 5]})
        second = pd.DataFrame({"M": [16, 32], "kernelId": [1, 1], "us": [6, 7]})
        result = self._merge_rows(first, second)
        pd.testing.assert_frame_equal(
            result, pd.concat([first, second], ignore_index=True)
        )

    def test_no_latency_column_and_missing_tags(self):
        first = pd.DataFrame({"M": [16, 16], "_tag": [None, None]})
        second = pd.DataFrame({"M": [16, 16], "_tag": [None, "model"]})
        result = self._merge_rows(first, second).fillna("")
        pd.testing.assert_frame_equal(
            result, pd.DataFrame({"M": [16, 16], "_tag": ["", "model"]})
        )

    def test_empty_inputs_keep_columns(self):
        empty = pd.DataFrame(columns=["M", "kernelId", "us"])
        result = self._merge_rows(empty, empty)
        pd.testing.assert_frame_equal(result, empty)

    def test_untuned_shape_collision_still_raises(self):
        (self.configs / "test_untuned.csv").write_text("M\n", encoding="utf-8")
        rows = pd.DataFrame({"M": [16]})
        with self.assertRaisesRegex(RuntimeError, "No 'us' column"):
            self._merge_rows(rows, rows)


if __name__ == "__main__":
    unittest.main()
