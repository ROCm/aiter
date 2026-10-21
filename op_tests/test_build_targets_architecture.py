# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import sys
import unittest
from unittest import mock

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "aiter", "jit", "utils")
    ),
)

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


class TestRuntimeArchitecture(unittest.TestCase):
    def setUp(self):
        import torch

        from aiter.jit.utils import chip_info

        self.torch = torch
        self.chip_info = chip_info
        self.chip_info.get_gfx_runtime.cache_clear()

    def tearDown(self):
        self.chip_info.get_gfx_runtime.cache_clear()

    def test_hipinfo_architecture_takes_precedence(self):
        with (
            mock.patch.object(self.chip_info, "IS_WINDOWS", True),
            mock.patch.object(
                self.chip_info, "_detect_native", return_value=["gfx942"]
            ),
            mock.patch.object(
                self.torch.cuda, "get_device_properties"
            ) as get_properties,
        ):
            self.assertEqual(self.chip_info.get_gfx_runtime(), "gfx942")
        get_properties.assert_not_called()

    def test_windows_fallback_uses_torch_and_ignores_gpu_archs(self):
        properties = type("Properties", (), {"gcnArchName": "gfx1151:sramecc+"})()
        with (
            mock.patch.dict(os.environ, {"GPU_ARCHS": "gfx942"}),
            mock.patch.object(self.chip_info, "IS_WINDOWS", True),
            mock.patch.object(
                self.chip_info,
                "_detect_native",
                side_effect=RuntimeError("hipinfo missing"),
            ),
            mock.patch.object(self.torch.version, "hip", "6.3"),
            mock.patch.object(self.torch.cuda, "is_available", return_value=True),
            mock.patch.object(self.torch.cuda, "device_count", return_value=1),
            mock.patch.object(self.torch.cuda, "current_device", return_value=0),
            mock.patch.object(
                self.torch.cuda, "get_device_properties", return_value=properties
            ) as get_properties,
        ):
            self.assertEqual(self.chip_info.get_gfx_runtime(), "gfx1151")
        get_properties.assert_called_once_with(0)

    def test_linux_runtime_errors_are_not_fallbacks(self):
        with (
            mock.patch.object(self.chip_info, "IS_WINDOWS", False),
            mock.patch.object(
                self.chip_info,
                "_detect_native",
                side_effect=RuntimeError("hipinfo error"),
            ),
            mock.patch.object(
                self.torch.cuda, "get_device_properties"
            ) as get_properties,
            self.assertRaisesRegex(RuntimeError, "hipinfo error"),
        ):
            self.chip_info.get_gfx_runtime()
        get_properties.assert_not_called()

    def test_windows_without_hip_or_device_rethrows_detector_error(self):
        for hip, available in ((None, True), ("6.3", False)):
            with self.subTest(hip=hip, available=available):
                with (
                    mock.patch.object(self.chip_info, "IS_WINDOWS", True),
                    mock.patch.object(
                        self.chip_info,
                        "_detect_native",
                        side_effect=RuntimeError("hipinfo missing"),
                    ),
                    mock.patch.object(self.torch.version, "hip", hip),
                    mock.patch.object(
                        self.torch.cuda, "is_available", return_value=available
                    ),
                    mock.patch.object(
                        self.torch.cuda, "get_device_properties"
                    ) as get_properties,
                    self.assertRaisesRegex(RuntimeError, "hipinfo missing"),
                ):
                    self.chip_info.get_gfx_runtime()
                get_properties.assert_not_called()

    def test_unsupported_architectures_still_raise(self):
        properties = type("Properties", (), {"gcnArchName": "gfx9999"})()
        for detector_result in (["gfx9999"], RuntimeError("hipinfo missing")):
            with self.subTest(detector_result=detector_result):
                detect = (
                    mock.patch.object(
                        self.chip_info, "_detect_native", return_value=detector_result
                    )
                    if isinstance(detector_result, list)
                    else mock.patch.object(
                        self.chip_info, "_detect_native", side_effect=detector_result
                    )
                )
                with (
                    mock.patch.object(self.chip_info, "IS_WINDOWS", True),
                    detect,
                    mock.patch.object(self.torch.version, "hip", "6.3"),
                    mock.patch.object(
                        self.torch.cuda, "is_available", return_value=True
                    ),
                    mock.patch.object(self.torch.cuda, "device_count", return_value=1),
                    mock.patch.object(
                        self.torch.cuda, "current_device", return_value=0
                    ),
                    mock.patch.object(
                        self.torch.cuda,
                        "get_device_properties",
                        return_value=properties,
                    ),
                    self.assertRaisesRegex(
                        KeyError, "Unknown GPU architecture: gfx9999"
                    ),
                ):
                    self.chip_info.get_gfx_runtime()


if __name__ == "__main__":
    unittest.main()
