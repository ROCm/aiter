# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-only tests for the OPUS BF16 GEMM CO gfx1250 strict-id alias.

Arch detection keeps the real runtime id (``gfx1250-strict`` on newer A0
stacks); only the OPUS CO kernel resolution and the C++ CO loader treat the
strict id as the base ``gfx1250`` family. Shared JIT probes, registry keys and
non-CO OPUS kernels keep the real value.
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
    gfx1250_4wave_co_kernels_list,
    gfx1250_clusterlaunch_kernels_list,
)

from aiter.ops.opus import _arch
from aiter.ops.opus.launch_plan import _build_a16w16_launch_plan


class TestRealArchValue(unittest.TestCase):
    def test_device_properties_keep_real_strict_id(self):
        props = mock.Mock(
            gcnArchName="gfx1250-strict:sramecc+", multi_processor_count=256
        )
        with mock.patch.object(
            _arch.torch.cuda, "get_device_properties", return_value=props
        ):
            arch, cu_num = _arch._read_device_arch_and_cu(torch.device("cuda", 0))
        self.assertEqual((arch, cu_num), ("gfx1250-strict", 256))

    def test_detect_arch_reports_real_strict_id(self):
        with (
            mock.patch.dict(os.environ, {"GPU_ARCHS": "native"}, clear=True),
            mock.patch(
                "aiter.jit.utils.chip_info.get_gfx_runtime",
                return_value="gfx1250-strict",
            ),
        ):
            self.assertEqual(_arch._detect_arch({"gfx1250"}), (False, "gfx1250-strict"))


class TestOpusCoArchAlias(unittest.TestCase):
    def _plan(self, arch, kid):
        return _build_a16w16_launch_plan(
            arch=arch,
            M=128,
            N=128,
            K=128,
            batch=1,
            cu_num=256,
            has_bias=False,
            input_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            kid=kid,
            split_k=0,
        )

    def test_co_kid_accepts_strict_runtime_id(self):
        co_kid = next(iter(gfx1250_4wave_co_kernels_list))
        strict = self._plan("gfx1250-strict", co_kid)
        base = self._plan("gfx1250", co_kid)
        self.assertEqual(strict.resolved_kid, base.resolved_kid)
        self.assertEqual(strict.resolved_kid, co_kid)

    def test_co_kid_accepts_strict_with_features(self):
        co_kid = next(iter(gfx1250_4wave_co_kernels_list))
        plan = self._plan("gfx1250-strict:sramecc+", co_kid)
        self.assertEqual(plan.resolved_kid, co_kid)

    def test_non_co_kid_keeps_real_id(self):
        non_co_kid = next(iter(gfx1250_clusterlaunch_kernels_list))
        with self.assertRaisesRegex(ValueError, "gfx1250-strict"):
            self._plan("gfx1250-strict", non_co_kid)

    def test_non_co_kid_still_works_on_base_id(self):
        non_co_kid = next(iter(gfx1250_clusterlaunch_kernels_list))
        self.assertEqual(self._plan("gfx1250", non_co_kid).resolved_kid, non_co_kid)


if __name__ == "__main__":
    unittest.main()
