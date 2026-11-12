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


_FLY = "flydsl_bmm_mxfp8_mfma"
# Tuned rows of one gfx950 shape: M -> (libtype, kernelName).
_ROWS = {
    16: ("flydsl", _FLY + "_t16x32x256_w1x2_nb5_sk1_nt_xcd_sps"),
    32: ("opus", "opus_bmm_a8w8_mxscale_bpreshuffle_x"),
    48: ("flydsl", _FLY + "_t64x32x512_w2x2_nb3_sk1_nt"),
    2048: ("opus", "opus_bmm_a8w8_mxscale_bpreshuffle_y"),
    10240: ("flydsl", _FLY + "_t256x256x128_w1x4_nb2_sk1_bd1_xcd"),
    32768: ("opus", "opus_bmm_a8w8_mxscale_bpreshuffle_z"),
}


class TestFusedQuantTunedKernelGfx950(unittest.TestCase):
    """The fused entry's tile: the bf16 entry's flydsl row, else the shape's
    flydsl row at the next larger tuned M (else the largest)."""

    def _pick(self, m: int):
        from aiter.ops import batched_gemm_op_a8w8 as op

        key = ("gfx950", 256, 8)
        table = {
            key + (row_m, 1024, 4096, "32x32"): {"libtype": lib, "kernelName": name}
            for row_m, (lib, name) in _ROWS.items()
        }
        op._fused_quant_tuned_kernel.cache_clear()
        with patch.object(
            op, "_load_mxscale_bmm_tuned", return_value=table
        ), patch.object(
            op,
            "lookup_mxscale_bmm_config",
            side_effect=lambda b, mm, n, k, **kw: table.get(key + (mm, n, k, "32x32")),
        ):
            return op._fused_quant_tuned_kernel(
                8, m, 1024, 4096, "32x32", "gfx950", 256
            )

    def test_flydsl_row_is_kept(self):
        self.assertEqual(self._pick(16), _ROWS[16][1])

    def test_opus_row_takes_next_larger_flydsl_row(self):
        self.assertEqual(self._pick(32), _ROWS[48][1])
        self.assertEqual(self._pick(2048), _ROWS[10240][1])

    def test_past_the_last_flydsl_row_takes_the_largest(self):
        self.assertEqual(self._pick(32768), _ROWS[10240][1])


if __name__ == "__main__":
    unittest.main()
