# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
import os
import unittest
from unittest import mock


def _load_device_info():
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "aiter",
            "ops",
            "triton",
            "utils",
            "device_info.py",
        )
    )
    spec = importlib.util.spec_from_file_location("aiter_device_info_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


device_info = _load_device_info()


class TestGetNumXcds(unittest.TestCase):
    def test_known_parts_and_unknown_default(self):
        self.assertEqual(device_info._xcds_for_part("gfx942", 304), 8)
        self.assertEqual(device_info._xcds_for_part("gfx942", 228), 6)
        self.assertEqual(device_info._xcds_for_part("gfx942", 80), 4)
        self.assertEqual(device_info._xcds_for_part("gfx950", 256), 8)
        self.assertEqual(device_info._xcds_for_part("gfx942", 1), 8)

    def test_rocm_before_7_uses_part_table(self):
        with (
            mock.patch.object(device_info, "_rocm_major", return_value=6),
            mock.patch.object(
                device_info,
                "_query_num_xccs",
                side_effect=AssertionError("HIP queried on ROCm 6"),
            ),
            mock.patch.object(device_info, "_xcds_from_known_part", return_value=6),
        ):
            self.assertEqual(device_info.get_num_xcds(), 6)

    def test_rocm_7_uses_hip_count(self):
        with (
            mock.patch.object(device_info, "_rocm_major", return_value=7),
            mock.patch.object(device_info, "_query_num_xccs", return_value=4) as query,
            mock.patch.object(
                device_info,
                "_xcds_from_known_part",
                side_effect=AssertionError("table used"),
            ),
        ):
            self.assertEqual(device_info.get_num_xcds(device_id=1), 4)
        query.assert_called_once_with(1)

    def test_rocm_7_query_failure_falls_back_to_part_table(self):
        with (
            mock.patch.object(device_info, "_rocm_major", return_value=7),
            mock.patch.object(
                device_info, "_query_num_xccs", side_effect=RuntimeError("rejected")
            ),
            mock.patch.object(device_info, "_xcds_from_known_part", return_value=6),
        ):
            self.assertEqual(device_info.get_num_xcds(0), 6)

    def test_nonpositive_hip_count_falls_back_to_part_table(self):
        with (
            mock.patch.object(device_info, "_rocm_major", return_value=7),
            mock.patch.object(device_info, "_query_num_xccs", return_value=0),
            mock.patch.object(device_info, "_xcds_from_known_part", return_value=8),
        ):
            self.assertEqual(device_info.get_num_xcds(0), 8)

    def test_unknown_version_tries_query_then_part_table(self):
        with (
            mock.patch.object(device_info, "_rocm_major", return_value=None),
            mock.patch.object(
                device_info, "_query_num_xccs", side_effect=RuntimeError("old runtime")
            ),
            mock.patch.object(device_info, "_xcds_from_known_part", return_value=4),
        ):
            self.assertEqual(device_info.get_num_xcds(0), 4)


if __name__ == "__main__":
    unittest.main()
