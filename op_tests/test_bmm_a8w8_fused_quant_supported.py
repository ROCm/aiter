# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``bmm_a8w8_mxfp8_fused_quant_supported`` must follow ``_runner``'s arch map.

The runner folds gfx1250-strict through ``base_gfx_arch`` before the
``_RUNNERS`` lookup. The supported() check has to do the same, or a strict
part misses the gfx1250 mxfp8_32 row and a caller logs the miss as an
unsupported shape.
"""

import unittest
from unittest.mock import patch

_BLOCKS = ("1x32", "32x32", "128x128")


class TestFusedQuantSupportedArch(unittest.TestCase):
    def _answers(self, gfx: str) -> dict[str, bool]:
        from aiter.ops.flydsl.batched_gemm_a8w8 import (
            bmm_a8w8_mxfp8_fused_quant_supported,
        )

        with patch("aiter.ops.flydsl.batched_gemm_a8w8.get_gfx", return_value=gfx):
            return {
                block: bmm_a8w8_mxfp8_fused_quant_supported(block) for block in _BLOCKS
            }

    def test_gfx1250_strict_matches_gfx1250(self):
        gfx1250 = self._answers("gfx1250")
        self.assertEqual(gfx1250, {"1x32": True, "32x32": True, "128x128": False})
        self.assertEqual(self._answers("gfx1250-strict"), gfx1250)

    def test_gfx950_fuses_every_block(self):
        self.assertEqual(
            self._answers("gfx950"),
            {"1x32": True, "32x32": True, "128x128": True},
        )

    def test_gfx950_column_major_x_scale_does_not_fuse(self):
        from aiter.ops.flydsl.batched_gemm_a8w8 import (
            bmm_a8w8_mxfp8_fused_quant_supported,
        )

        with patch("aiter.ops.flydsl.batched_gemm_a8w8.get_gfx", return_value="gfx950"):
            self.assertFalse(bmm_a8w8_mxfp8_fused_quant_supported("128x128", True))

    def test_other_arch_has_no_fused_epilogue(self):
        self.assertEqual(
            self._answers("gfx942"),
            {"1x32": False, "32x32": False, "128x128": False},
        )


if __name__ == "__main__":
    unittest.main()
