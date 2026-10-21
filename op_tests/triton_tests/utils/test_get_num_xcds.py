# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
import os
import tempfile
import unittest
from unittest import mock


def _load_device_info():
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
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
    def setUp(self):
        self._env = mock.patch.dict(os.environ)
        self._env.start()
        os.environ.pop("CU_NUM", None)

    def tearDown(self):
        self._env.stop()

    def test_published_xcd_parts(self):
        # Active CUs per device and XCDs. gfx90a and gfx908 are not here:
        # those parts have no XCDs. An MI250 GCD is already its own device.
        published = (
            ("gfx942", 304, 8),  # MI300X, MI325X
            ("gfx942", 228, 6),  # MI300A
            ("gfx950", 256, 8),  # MI350X, MI355X
            ("gfx950", 128, 4),  # MI350P
            ("gfx1250", 256, 8),  # MI455X
        )
        for gfx, cu_num, xcds in published:
            with self.subTest(gfx=gfx, cu_num=cu_num):
                self.assertEqual(device_info._xcds_for_part(gfx, cu_num), xcds)

    def test_other_cases_return_one(self):
        # One L2 domain per HIP device. 0 would divide by zero in remap_xcd
        # and in mla_gluon's batch // NUM_XCDS.
        for gfx, cu_num in (
            ("gfx906", 64),  # MI50, MI60
            ("gfx908", 120),  # CDNA1, MI100
            ("gfx90a", 104),  # CDNA2, MI210 / MI250 per GCD
            ("gfx90a", 110),  # CDNA2, MI250X per GCD
            ("gfx942", 80),  # MI308X: XCD count is not known
            ("gfx942", 120),  # unlisted gfx942
            ("gfx1100", 96),  # RDNA3
            ("gfx1200", 32),  # RDNA4
            ("gfx1201", 64),  # RDNA4
            ("gfx1250", 128),  # unlisted CDNA5
        ):
            with self.subTest(gfx=gfx, cu_num=cu_num):
                self.assertEqual(device_info._xcds_for_part(gfx, cu_num), 1)

    def test_rocm_before_7_uses_part_table(self):
        with (
            mock.patch.object(device_info, "_rocm_major", return_value=6),
            mock.patch.object(device_info, "_query_num_xccs") as query,
            mock.patch.object(device_info, "_xcds_from_known_part", return_value=6),
        ):
            self.assertEqual(device_info.get_num_xcds(0), 6)
        query.assert_not_called()

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

    def test_attribute_value_tracks_header_enum(self):
        # ClockInstructionRate shares AmdSpecificBegin, so it does not consume
        # a value. An enumerator inserted before NumberOfXccs shifts it.
        base = """
        hipDeviceAttributeAmdSpecificBegin = 10000,
        hipDeviceAttributeClockInstructionRate =
            hipDeviceAttributeAmdSpecificBegin,
        hipDeviceAttributeWallClockRate,
        hipDeviceAttributeNumberOfXccs,
        """
        shifted = base.replace(
            "hipDeviceAttributeWallClockRate,",
            "hipDeviceAttributeWallClockRate,\n        hipDeviceAttributeInserted,",
        )
        explicit = base.replace(
            "hipDeviceAttributeNumberOfXccs,",
            "hipDeviceAttributeNumberOfXccs = 12000,",
        )
        self.assertEqual(
            device_info._hip_device_attribute_value(
                base, "hipDeviceAttributeNumberOfXccs"
            ),
            10002,
        )
        self.assertEqual(
            device_info._hip_device_attribute_value(
                shifted, "hipDeviceAttributeNumberOfXccs"
            ),
            10003,
        )
        self.assertEqual(
            device_info._hip_device_attribute_value(
                explicit, "hipDeviceAttributeNumberOfXccs"
            ),
            12000,
        )
        shifted_by_sum = base.replace(
            "hipDeviceAttributeClockInstructionRate =\n"
            "            hipDeviceAttributeAmdSpecificBegin,",
            "hipDeviceAttributeClockInstructionRate =\n"
            "            hipDeviceAttributeAmdSpecificBegin + 5,",
        )
        self.assertEqual(
            device_info._hip_device_attribute_value(
                shifted_by_sum, "hipDeviceAttributeNumberOfXccs"
            ),
            10007,
        )
        prefixed = base.replace(
            "hipDeviceAttributeWallClockRate,",
            "hipDeviceAttributeNumberOfXccsEx,\n"
            "        hipDeviceAttributeWallClockRate,",
        )
        self.assertEqual(
            device_info._hip_device_attribute_value(
                prefixed, "hipDeviceAttributeNumberOfXccs"
            ),
            10003,
        )
        conditional = base.replace(
            "hipDeviceAttributeWallClockRate,",
            "#if 0\n        hipDeviceAttributeWallClockRate,\n        #endif",
        )
        with self.assertRaises(ValueError):
            device_info._hip_device_attribute_value(
                conditional, "hipDeviceAttributeNumberOfXccs"
            )
        with self.assertRaises(ValueError):
            device_info._hip_device_attribute_value(
                "hipDeviceAttributeAmdSpecificBegin = sizeof(int),\n"
                "hipDeviceAttributeNumberOfXccs,",
                "hipDeviceAttributeNumberOfXccs",
            )

    def test_header_beside_loaded_library(self):
        with tempfile.TemporaryDirectory() as root:
            library = os.path.join(root, "lib", "libamdhip64.so.1")
            header = os.path.join(root, "include", "hip", "hip_runtime_api.h")
            os.makedirs(os.path.dirname(library))
            os.makedirs(os.path.dirname(header))
            open(library, "w").close()
            open(header, "w").close()
            self.assertEqual(device_info._header_beside_library(library), header)

    def test_header_uses_loaded_library(self):
        with (
            tempfile.TemporaryDirectory() as root,
            tempfile.TemporaryDirectory() as other,
        ):
            library = os.path.join(root, "lib", "libamdhip64.so")
            header = os.path.join(root, "include", "hip", "hip_runtime_api.h")
            os.makedirs(os.path.dirname(library))
            os.makedirs(os.path.dirname(header))
            open(library, "w").close()
            open(header, "w").close()
            os.environ["ROCM_PATH"] = other
            with mock.patch.object(
                device_info, "_loaded_amdhip64_path", return_value=library
            ):
                self.assertEqual(device_info._hip_runtime_api_header(), header)

    def test_header_rejects_other_rocm_install(self):
        with tempfile.TemporaryDirectory() as root:
            library = os.path.join(root, "lib", "libamdhip64.so")
            os.makedirs(os.path.dirname(library))
            open(library, "w").close()
            os.environ["ROCM_PATH"] = "/opt/another-rocm"
            with (
                mock.patch.object(
                    device_info, "_loaded_amdhip64_path", return_value=library
                ),
                self.assertRaises(FileNotFoundError),
            ):
                device_info._hip_runtime_api_header()

    def test_rocm_major_uses_and_caches_live_runtime(self):
        class Function:
            def __init__(self):
                self.calls = 0

            def __call__(self, result):
                self.calls += 1
                result._obj.value = 7 * 10_000_000 + 2 * 100_000
                return 0

        runtime = mock.Mock()
        runtime.hipRuntimeGetVersion = Function()
        device_info._rocm_major.cache_clear()
        try:
            with mock.patch.object(device_info, "_hip_runtime", return_value=runtime):
                self.assertEqual(device_info._rocm_major(), 7)
                self.assertEqual(device_info._rocm_major(), 7)
            self.assertEqual(runtime.hipRuntimeGetVersion.calls, 1)
        finally:
            device_info._rocm_major.cache_clear()

    def test_cu_num_override_skips_hip_query(self):
        os.environ["CU_NUM"] = "228"
        with (
            mock.patch.object(device_info, "_rocm_major", return_value=7),
            mock.patch.object(device_info, "_query_num_xccs") as query,
            mock.patch.object(
                device_info, "_xcds_from_known_part", return_value=6
            ) as table,
        ):
            self.assertEqual(device_info.get_num_xcds(1), 6)
        query.assert_not_called()
        table.assert_called_once_with(None)

    def test_part_table_receives_device_id(self):
        with (
            mock.patch.object(device_info, "_rocm_major", return_value=6),
            mock.patch.object(
                device_info, "_xcds_from_known_part", return_value=4
            ) as table,
        ):
            self.assertEqual(device_info.get_num_xcds(3), 4)
        table.assert_called_once_with(3)

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
