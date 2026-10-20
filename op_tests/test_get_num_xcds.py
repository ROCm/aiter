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
    def test_published_xcd_parts(self):
        # Active CUs per device and XCDs from the ROCm GPU specification table.
        # gfx90a (MI210/MI250/MI250X) and gfx908 (MI100) are not here: those
        # parts have no XCDs. An MI250 GCD is already its own device.
        published = (
            ("gfx942", 304, 8),  # MI300X, MI325X
            ("gfx942", 228, 6),  # MI300A
            ("gfx950", 256, 8),  # MI350X, MI355X
            ("gfx950", 128, 4),  # MI350P
        )
        for gfx, cu_num, xcds in published:
            with self.subTest(gfx=gfx, cu_num=cu_num):
                self.assertEqual(device_info._xcds_for_part(gfx, cu_num), xcds)

    def test_unlisted_part_defaults_to_eight_xcds(self):
        # 120 CUs can hold 8 XCDs. A 1-CU part cannot, so it is not the example.
        self.assertEqual(device_info._xcds_for_part("gfx942", 120), 8)

    def test_non_xcd_arch_returns_one(self):
        # One L2 domain per HIP device. 0 would divide by zero in remap_xcd
        # and in mla_gluon's batch // NUM_XCDS.
        for gfx, cu_num in (
            ("gfx906", 64),
            ("gfx908", 120),
            ("gfx90a", 104),
            ("gfx90a", 110),
        ):
            with self.subTest(gfx=gfx, cu_num=cu_num):
                self.assertEqual(device_info._xcds_for_part(gfx, cu_num), 1)

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

    def test_rejected_attribute_is_queried_once(self):
        device_info._query_num_xccs.cache_clear()
        calls = []

        def reject(device_id):
            calls.append(device_id)
            raise RuntimeError("rejected")

        try:
            with (
                mock.patch.object(device_info, "_rocm_major", return_value=7),
                mock.patch.object(device_info, "_hip_num_xccs", side_effect=reject),
                mock.patch.object(device_info, "_xcds_from_known_part", return_value=6),
            ):
                self.assertEqual(device_info.get_num_xcds(0), 6)
                self.assertEqual(device_info.get_num_xcds(0), 6)
            self.assertEqual(calls, [0])
        finally:
            device_info._query_num_xccs.cache_clear()


if __name__ == "__main__":
    unittest.main()
