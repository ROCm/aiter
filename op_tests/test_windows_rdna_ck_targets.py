# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import sys
import unittest
from unittest import mock


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JIT_UTILS = os.path.join(REPO_ROOT, "aiter", "jit", "utils")
sys.path.insert(0, JIT_UTILS)

from build_targets import get_build_targets_env  # noqa: E402
from mha_recipes import _ck_targets_flag_for_arch  # noqa: E402


WINDOWS_RDNA_TARGETS = {
    "gfx1100": 96,
    "gfx1101": 60,
    "gfx1102": 32,
    "gfx1103": 12,
    "gfx1151": 20,
    "gfx1201": 64,
}


class TestWindowsRDNACKTargets(unittest.TestCase):
    def test_offline_build_target_defaults(self):
        for gfx, cu_num in WINDOWS_RDNA_TARGETS.items():
            with self.subTest(gfx=gfx), mock.patch.dict(
                os.environ, {"GPU_ARCHS": gfx}, clear=True
            ):
                self.assertEqual(get_build_targets_env(), [(gfx, cu_num)])

    def test_ck_codegen_receives_each_rdna_target(self):
        for gfx in WINDOWS_RDNA_TARGETS:
            with self.subTest(gfx=gfx):
                self.assertEqual(
                    _ck_targets_flag_for_arch(gfx), f" --targets {gfx}"
                )

    def test_cdna_keeps_ck_generator_defaults(self):
        self.assertEqual(_ck_targets_flag_for_arch("gfx942"), "")
        self.assertEqual(_ck_targets_flag_for_arch("gfx950"), "")


if __name__ == "__main__":
    unittest.main()
