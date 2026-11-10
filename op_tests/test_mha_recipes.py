# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU regression tests for varlen attention kernel dtype selection."""

import fnmatch
import importlib.util
import unittest
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "aiter/jit/utils/mha_recipes.py"
spec = importlib.util.spec_from_file_location("mha_recipes", source)
recipes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipes)

# Kernel names emitted by CK 96afae1d for receipt 200 and target gfx1250.
BF16_KERNEL = (
    "fmha_fwd_d32_bf16_group_b64x64x32x32x32x64_r4x1x1_r4x1x1"
    "_w16x16x32_w16x16x32_qr_tdm_vr_pssk_nlogits_nbias_nmask_nlse"
    "_ndropout_nskip_nqscale_ntrload_kvlp_plk_sink_gfx125"
)
FP8BF16_KERNEL = (
    "fmha_fwd_d128_fp8bf16_group_b64x128x128x128x128x128_r4x1x1_r4x1x1"
    "_w16x16x128_w16x16x128_qr_tdm_vr_pssk_nlogits_nbias_nmask_nlse"
    "_ndropout_nskip_nqscale_ntrload_nkvlp_nplk_nsink_gfx125"
)


class TestMhaRecipeDtypeSelection(unittest.TestCase):
    def test_bf16_request_excludes_fp8_kernels(self):
        _, pattern = recipes.compose_mha_fwd_variant_suffix_and_filter(
            "bf16", False, False, False, False, False, True, True, True
        )

        self.assertTrue(fnmatch.fnmatchcase(BF16_KERNEL, pattern))
        self.assertFalse(fnmatch.fnmatchcase(FP8BF16_KERNEL, pattern))

    def test_fp8_request_keeps_fp8_kernels(self):
        _, pattern = recipes.compose_mha_fwd_variant_suffix_and_filter(
            "fp8bf16", False, False, False, False, False, True, True, True
        )

        self.assertTrue(fnmatch.fnmatchcase(FP8BF16_KERNEL, pattern))
        self.assertFalse(fnmatch.fnmatchcase(BF16_KERNEL, pattern))


if __name__ == "__main__":
    unittest.main()
