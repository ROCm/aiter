# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Unit tests for the input rotation behind run_perftest.

Run: python3 -m unittest op_tests.tuning_tests.test_perftest_rotation -v
"""

import types
import unittest
from unittest.mock import patch

import torch

from aiter.utility import measure

MiB = 1 << 20


def _cache(level, kib, instances=1):
    return {"cache_level": level, "cache_size": kib, "num_cache_instance": instances}


# amdsmi_get_gpu_cache_info()["cache"] as read on an MI300A (L1 entries trimmed).
MI300A_CACHES = [
    _cache(1, 32, 228),
    _cache(1, 64, 76),
    _cache(2, 4096),
    _cache(3, 262144),
]
# The same call on an integrated gfx1036, which has no level 3.
GFX1036_CACHES = [_cache(1, 16, 2), _cache(1, 128), _cache(2, 256)]


class TestLlcFromCacheInfo(unittest.TestCase):
    def test_level_3_when_present(self):
        self.assertEqual(measure.llc_from_cache_info(MI300A_CACHES), 256 * MiB)

    def test_highest_level_when_no_level_3(self):
        self.assertEqual(measure.llc_from_cache_info(GFX1036_CACHES), 256 * 1024)

    def test_instances_add_up(self):
        caches = [_cache(1, 16), _cache(2, 8192, instances=2)]
        self.assertEqual(measure.llc_from_cache_info(caches), 16 * MiB)


class TestLlcFallback(unittest.TestCase):
    def setUp(self):
        measure.llc_bytes.cache_clear()
        self.addCleanup(measure.llc_bytes.cache_clear)

    def _llc_without_amdsmi(self, arch_name):
        props = types.SimpleNamespace(gcnArchName=arch_name)
        with (
            patch.object(measure, "_amdsmi_llc", side_effect=ImportError("no amdsmi")),
            patch.object(torch.cuda, "get_device_properties", return_value=props),
            self.assertLogs(measure.logger, "WARNING") as logs,
        ):
            llc = measure.llc_bytes(0)
        return llc, logs.output

    def test_known_arch(self):
        for arch, want in (
            ("gfx90a:sramecc+:xnack-", 8 * MiB),
            ("gfx942:sramecc+:xnack-", 256 * MiB),
            ("gfx950:sramecc+:xnack-", 256 * MiB),
        ):
            with self.subTest(arch=arch):
                measure.llc_bytes.cache_clear()
                llc, logs = self._llc_without_amdsmi(arch)
                self.assertEqual(llc, want)
                self.assertIn("no amdsmi", logs[0])

    def test_unknown_arch_assumes_256_mib(self):
        llc, logs = self._llc_without_amdsmi("gfx1250")
        self.assertEqual(llc, 256 * MiB)
        self.assertIn("gfx1250", logs[0])

    def test_answer_is_cached_and_warns_once(self):
        props = types.SimpleNamespace(gcnArchName="gfx1250")
        with (
            patch.object(
                measure, "_amdsmi_llc", side_effect=RuntimeError("no handle")
            ) as smi,
            patch.object(torch.cuda, "get_device_properties", return_value=props),
            self.assertLogs(measure.logger, "WARNING") as logs,
        ):
            measure.llc_bytes(0)
            measure.llc_bytes(0)
        self.assertEqual(smi.call_count, 1)
        self.assertEqual(len(logs.output), 1)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestLlcOnDevice(unittest.TestCase):
    def test_amdsmi_answers_on_this_device(self):
        measure.llc_bytes.cache_clear()
        device = torch.cuda.current_device()
        llc = measure.llc_bytes(device)
        self.assertEqual(llc, measure._amdsmi_llc(device))
        self.assertGreaterEqual(
            llc, torch.cuda.get_device_properties(device).L2_cache_size
        )


if __name__ == "__main__":
    unittest.main()
