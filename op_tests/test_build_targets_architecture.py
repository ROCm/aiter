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
    def test_gfx1250_strict_offline_build_target(self):
        with mock.patch.dict(os.environ, {"GPU_ARCHS": "gfx1250-strict"}, clear=True):
            self.assertEqual(get_build_targets_env(), [("gfx1250-strict", 256)])

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


class TestClusterLaunchBuildArchs(unittest.TestCase):
    """build_module's -DAITER_ENABLE_CLUSTER_LAUNCH arch test (aiter.jit.core)."""

    def setUp(self):
        from aiter.jit import core

        self.core = core
        self.detect = lambda: "gfx942"

    def test_blank_gpu_archs_falls_back_to_detection(self):
        # A Dockerfile's `ENV GPU_ARCHS=` leaves the variable set but empty.
        # That must behave like "native", not raise out of every JIT build.
        for value in ("", " ", ";", " ; ", "native", "NATIVE", "gfx942;native"):
            with (
                self.subTest(value=value),
                mock.patch.dict(os.environ, {"GPU_ARCHS": value}, clear=True),
            ):
                self.assertEqual(
                    self.core._build_archs_from_env(self.detect), ["gfx942"]
                )
                self.assertFalse(self.core._cluster_launch_target(self.detect))

    def test_unset_gpu_archs_falls_back_to_detection(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.core._build_archs_from_env(self.detect), ["gfx942"])

    def test_explicit_archs_win_over_detection(self):
        with mock.patch.dict(
            os.environ, {"GPU_ARCHS": " gfx1250 ; gfx942 "}, clear=True
        ):
            self.assertEqual(
                self.core._build_archs_from_env(self.detect), ["gfx1250", "gfx942"]
            )

    def test_cluster_launch_follows_any_targeted_arch(self):
        cases = {
            "gfx1250": True,
            "gfx1250-strict": True,
            "gfx1250;gfx942": True,
            "gfx942;gfx1250-strict": True,
            "gfx942": False,
            "gfx950": False,
        }
        for value, expected in cases.items():
            with (
                self.subTest(value=value),
                mock.patch.dict(os.environ, {"GPU_ARCHS": value}, clear=True),
            ):
                self.assertEqual(
                    self.core._cluster_launch_target(self.detect), expected
                )

    def test_detected_strict_device_enables_cluster_launch(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(self.core._cluster_launch_target(lambda: "gfx1250-strict"))


class TestRuntimeArchitecture(unittest.TestCase):
    def setUp(self):
        import torch

        from aiter.jit.utils import chip_info

        self.torch = torch
        self.chip_info = chip_info
        self.chip_info.get_gfx_runtime.cache_clear()

    def tearDown(self):
        self.chip_info.get_gfx_runtime.cache_clear()

    def test_native_detection_preserves_strict_variant(self):
        self.chip_info._detect_native.cache_clear()
        try:
            with mock.patch.object(
                self.chip_info,
                "_gpu_info_output",
                return_value="  Name: gfx1250-strict\n",
            ):
                self.assertEqual(self.chip_info._detect_native(), ["gfx1250-strict"])
        finally:
            self.chip_info._detect_native.cache_clear()

    def test_agent_parsers_preserve_strict_variant(self):
        rocminfo = """Agent 1
  Name: gfx1250-strict
  Device Type: GPU
  ASIC Revision: 0
"""
        hipinfo = """device#0
gcnArchName: gfx1250-strict
asicRevision: 0
"""
        for is_windows, output, parser in (
            (False, rocminfo, self.chip_info._rocminfo_gpu_agents),
            (True, hipinfo, self.chip_info._hipinfo_gpu_agents),
        ):
            with (
                self.subTest(is_windows=is_windows),
                mock.patch.object(self.chip_info, "IS_WINDOWS", is_windows),
                mock.patch.object(
                    self.chip_info, "_gpu_info_output", return_value=output
                ),
            ):
                parser.cache_clear()
                self.assertEqual(parser(), [("gfx1250-strict", 0)])
                parser.cache_clear()

    def test_strict_runtime_architecture_is_supported(self):
        with mock.patch.object(
            self.chip_info, "_detect_native", return_value=["gfx1250-strict"]
        ):
            self.assertEqual(self.chip_info.get_gfx_runtime(), "gfx1250-strict")

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
