# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import unittest
from unittest import mock

from aiter.jit.utils.build_targets import (
    get_build_targets_env,
    torch_processor_count_to_cu,
)
from aiter.jit.utils.mha_recipes import _ck_targets_flag_for_arch


WINDOWS_RDNA_TARGETS = {
    "gfx1100": 96,
    "gfx1101": 60,
    "gfx1102": 32,
    "gfx1103": 12,
    "gfx1151": 40,
    "gfx1201": 64,
}


class TestWindowsRDNACKTargets(unittest.TestCase):
    def test_offline_build_target_defaults(self):
        for gfx, cu_num in WINDOWS_RDNA_TARGETS.items():
            with self.subTest(gfx=gfx), mock.patch.dict(
                os.environ, {"GPU_ARCHS": gfx}, clear=True
            ):
                self.assertEqual(get_build_targets_env(), [(gfx, cu_num)])

    def test_ck_codegen_receives_rdna_target(self):
        for gfx in WINDOWS_RDNA_TARGETS:
            with self.subTest(gfx=gfx):
                self.assertEqual(_ck_targets_flag_for_arch(gfx), f" --targets {gfx}")

    def test_cdna_and_gfx1250_processor_counts_are_not_doubled(self):
        for gfx, processor_count in (
            ("gfx942", 304),
            ("gfx950", 256),
            ("gfx1250", 256),
        ):
            with self.subTest(gfx=gfx):
                self.assertEqual(
                    torch_processor_count_to_cu(gfx, processor_count), processor_count
                )

    def test_rdna_wgp_counts_are_normalized_to_physical_cus(self):
        for gfx, wgp_count, cu_count in (
            ("gfx1101", 30, 60),
            ("gfx1151", 20, 40),
            ("gfx1201", 32, 64),
        ):
            with self.subTest(gfx=gfx):
                self.assertEqual(torch_processor_count_to_cu(gfx, wgp_count), cu_count)


if __name__ == "__main__":
    unittest.main()
