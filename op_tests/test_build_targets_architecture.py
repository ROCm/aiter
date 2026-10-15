# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import unittest
from unittest import mock

from aiter.jit.utils.build_targets import (
    get_build_targets_env,
    torch_processor_count_to_cu,
)


class TestBuildTargetArchitecture(unittest.TestCase):
    def test_rdna_offline_build_target_defaults(self):
        targets = {
            "gfx1100": 96,
            "gfx1101": 60,
            "gfx1102": 32,
            "gfx1103": 12,
            "gfx1150": 16,
            "gfx1151": 40,
            "gfx1200": 32,
            "gfx1201": 64,
        }
        for gfx, cu_num in targets.items():
            with (
                self.subTest(gfx=gfx),
                mock.patch.dict(os.environ, {"GPU_ARCHS": gfx}, clear=True),
            ):
                self.assertEqual(get_build_targets_env(), [(gfx, cu_num)])

    def test_multi_arch_and_cu_override(self):
        with mock.patch.dict(
            os.environ,
            {"GPU_ARCHS": "gfx1151;gfx1200", "CU_NUM": "28"},
            clear=True,
        ):
            self.assertEqual(
                get_build_targets_env(), [("gfx1151", 28), ("gfx1200", 28)]
            )

    def test_rdna_cu_count_follows_hip_wgp_mode(self):
        cases = (
            (None, True),
            ("true", True),
            ("1", True),
            ("0", False),
            ("false", False),
        )
        for mode, wgp_enabled in cases:
            env = {} if mode is None else {"GPU_ENABLE_WGP_MODE": mode}
            with self.subTest(mode=mode), mock.patch.dict(os.environ, env, clear=True):
                processor_count = 20 if wgp_enabled else 40
                self.assertEqual(
                    torch_processor_count_to_cu("gfx1151:sramecc+", processor_count), 40
                )

    def test_cdna_count_is_unchanged(self):
        self.assertEqual(torch_processor_count_to_cu("gfx942", 304), 304)


if __name__ == "__main__":
    unittest.main()
