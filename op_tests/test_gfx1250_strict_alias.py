# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-only tests for one logical gfx1250 family across the two runtime ids.

Newer A0 HIP stacks report ``gfx1250-strict`` while ``rocm_agent_enumerator``
still reports ``gfx1250``.  aiter keys kernels, tuned tables, build targets and
pre-compiled CO images by the base family, so every probe boundary must map
both spellings onto ``gfx1250`` instead of duplicating tables or directories.
"""

import os
import sys
import unittest
from unittest import mock

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "csrc", "opus_gemm"))

import torch
from opus_gemm_common import (
    KernelKey,
    _canonical_gfx_arch,
    get_kernel_instance,
    get_registry,
    gfx1250_4wave_co_kernels_list,
)

from aiter.jit.utils import chip_info
from aiter.jit.utils.build_targets import (
    canonical_gfx_arch,
    get_build_targets_env,
)
from aiter.ops.opus import _arch


class TestCanonicalGfxArch(unittest.TestCase):
    def test_spellings(self):
        cases = {
            "gfx1250": "gfx1250",
            "gfx1250-strict": "gfx1250",
            "GFX1250-STRICT": "gfx1250",
            " gfx1250-strict ": "gfx1250",
            "gfx1250-strict:sramecc+": "gfx1250",
            "gfx1250:sramecc+": "gfx1250",
            "gfx950": "gfx950",
            "gfx950-strict": "gfx950",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(canonical_gfx_arch(value), expected)
                self.assertEqual(_canonical_gfx_arch(value), expected)


class TestBuildTargets(unittest.TestCase):
    def test_gpu_archs_strict_maps_to_family(self):
        with mock.patch.dict(os.environ, {"GPU_ARCHS": "gfx1250-strict"}, clear=True):
            self.assertEqual(get_build_targets_env(), [("gfx1250", 256)])

    def test_mixed_archs_with_strict(self):
        with mock.patch.dict(
            os.environ, {"GPU_ARCHS": "gfx1250-strict;gfx950"}, clear=True
        ):
            self.assertEqual(
                get_build_targets_env(), [("gfx1250", 256), ("gfx950", 256)]
            )


class TestRuntimeProbe(unittest.TestCase):
    def setUp(self):
        chip_info.get_gfx_runtime.cache_clear()
        chip_info.get_gfx_custom_op_core.cache_clear()
        chip_info.get_gfx.cache_clear()
        chip_info.get_gfx_list.cache_clear()

    def tearDown(self):
        chip_info.get_gfx_runtime.cache_clear()
        chip_info.get_gfx_custom_op_core.cache_clear()
        chip_info.get_gfx.cache_clear()
        chip_info.get_gfx_list.cache_clear()

    def test_detect_native_normalizes_strict(self):
        output = (
            "Agent 1\n"
            "  Name:                    gfx1250-strict\n"
            "  Device Type:             GPU\n"
            "  Compute Unit:            256\n"
        )
        with mock.patch.object(chip_info, "_gpu_info_output", return_value=output):
            self.assertEqual(chip_info._detect_native(), ["gfx1250"])

    def test_get_gfx_runtime_returns_family(self):
        with (
            mock.patch.object(chip_info, "IS_WINDOWS", True),
            mock.patch.object(chip_info, "_detect_native", return_value=["gfx1250"]),
        ):
            self.assertEqual(chip_info.get_gfx_runtime(), "gfx1250")

    def test_gpu_archs_strict_resolves_build_arch(self):
        with mock.patch.dict(os.environ, {"GPU_ARCHS": "gfx1250-strict"}, clear=True):
            self.assertEqual(chip_info.get_gfx_custom_op_core(), 18)
            self.assertEqual(chip_info.get_gfx(), "gfx1250")

    def test_lds_capacity_accepts_strict(self):
        self.assertEqual(chip_info.get_lds_capacity_bytes("gfx1250-strict"), 320 * 1024)


class TestOpusRegistryAlias(unittest.TestCase):
    def test_kernel_key_normalizes(self):
        key = KernelKey("a16w16", "gfx1250-strict:sramecc+", 256, 496)
        self.assertEqual(key.gfx, "gfx1250")
        self.assertEqual(KernelKey.decode(key.encode()), key)

    def test_registry_is_shared(self):
        base = get_registry("a16w16", "gfx1250", 256)
        strict = get_registry("a16w16", "gfx1250-strict", 256)
        self.assertTrue(base)
        self.assertEqual(base, strict)

    def test_kernel_instance_is_shared(self):
        co_kid = next(iter(gfx1250_4wave_co_kernels_list))
        base = get_kernel_instance("gfx1250", "a16w16", co_kid, cu_num=256)
        strict = get_kernel_instance("gfx1250-strict", "a16w16", co_kid, cu_num=256)
        self.assertIs(base, strict)
        self.assertIsNotNone(strict)


class TestOpusDeviceAlias(unittest.TestCase):
    def test_device_properties_strict(self):
        props = mock.Mock(
            gcnArchName="gfx1250-strict:sramecc+", multi_processor_count=256
        )
        with mock.patch.object(
            _arch.torch.cuda, "get_device_properties", return_value=props
        ):
            arch, cu_num = _arch._read_device_arch_and_cu(torch.device("cuda", 0))
        self.assertEqual((arch, cu_num), ("gfx1250", 256))

    def test_detect_arch_env_strict(self):
        with mock.patch.dict(os.environ, {"GPU_ARCHS": "gfx1250-strict"}, clear=True):
            self.assertEqual(_arch._detect_arch({"gfx1250"}), (True, "gfx1250"))


if __name__ == "__main__":
    unittest.main()
