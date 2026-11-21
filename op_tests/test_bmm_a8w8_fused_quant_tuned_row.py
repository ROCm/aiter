# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The fused-quant batched entry must run the bf16 entry's tuned row.

``_fused_quant_tuned_kernel`` looks the shape up in the same preshuffle table
as ``batched_gemm_a8w8_mxscale_bpreshuffle`` and hands a flydsl row's
kernelName to the runner as is, on gfx1250 and gfx950 alike. No row, an opus
row, or a gfx1250 compute-kernel row leaves the choice to the runner (None).
CPU only: the table lookup is patched, nothing launches.
"""

import unittest
from unittest.mock import patch

_LOOKUP = "aiter.ops.batched_gemm_op_a8w8.lookup_mxscale_bmm_config"
# A CU count other than _SHAPE_950's, so the lookup must get the caller's.
_SHAPE = (8, 48, 1024, 4096, "32x32", "gfx1250", 80)
_SHAPE_950 = (8, 48, 1024, 4096, "32x32", "gfx950", 256)
_FLYDSL_NAME = "flydsl_bmm_mxfp8_32_wmma_t32x64x512_mw2_nw2_nb4_sk1_cm1_cn1"
_COMPUTE_NAME = (
    "flydsl_mxfp8_32_bpreshuffle_compute_wmma_t128x128x128_mw2_nw2_nb4_sk1_cm2_cn4"
)
# The gfx950 row for exactly _SHAPE_950 in the DSv4 batched preshuffle table.
_FLYDSL_NAME_950 = "flydsl_bmm_mxfp8_mfma_t64x32x512_w2x2_nb3_sk1_nt"


class TestFusedQuantTunedRow(unittest.TestCase):
    def _pick(self, row, shape=_SHAPE):
        from aiter.ops import batched_gemm_op_a8w8 as op

        op._fused_quant_tuned_kernel.cache_clear()
        try:
            with patch(_LOOKUP, return_value=row) as lookup:
                name = op._fused_quant_tuned_kernel(*shape)
        finally:
            op._fused_quant_tuned_kernel.cache_clear()
        lookup.assert_called_once()
        args, kwargs = lookup.call_args
        self.assertEqual(args, shape[:4])
        self.assertEqual(kwargs["w_scale_block"], shape[4])
        self.assertTrue(kwargs["bpreshuffle"])
        self.assertEqual((kwargs["gfx"], kwargs["cu_num"]), shape[5:])
        return name

    def test_no_row_keeps_the_runner_choice(self):
        self.assertIsNone(self._pick(None))

    def test_opus_row_keeps_the_runner_choice(self):
        row = {"libtype": "opus", "kernelName": "opus_bmm_a8w8_mxscale_tile"}
        self.assertIsNone(self._pick(row))

    def test_compute_row_keeps_the_runner_choice(self):
        # A DSv4.1 wo_a row of main's table; the fused entry does not run tuned
        # compute rows yet, so it leaves the tile to the runner.
        row = {"libtype": "flydsl", "kernelName": _COMPUTE_NAME}
        self.assertIsNone(self._pick(row))

    def test_flydsl_row_name_is_passed_through(self):
        row = {"libtype": "flydsl", "kernelName": _FLYDSL_NAME}
        self.assertEqual(self._pick(row), _FLYDSL_NAME)

    def test_unparsable_flydsl_name_is_passed_through(self):
        # The runner rejects it with ValueError, as for the bf16 entry; the
        # fused entry must not swap in the heuristic tile.
        row = {"libtype": "flydsl", "kernelName": "flydsl_bmm_not_a_kernel"}
        self.assertEqual(self._pick(row), "flydsl_bmm_not_a_kernel")

    def test_gfx950_flydsl_row_name_is_passed_through(self):
        row = {"libtype": "flydsl", "kernelName": _FLYDSL_NAME_950}
        self.assertEqual(self._pick(row, _SHAPE_950), _FLYDSL_NAME_950)

    def test_gfx950_opus_row_and_no_row_keep_the_runner_choice(self):
        row = {"libtype": "opus", "kernelName": "opus_bmm_a8w8_mxscale_tile"}
        self.assertIsNone(self._pick(row, _SHAPE_950))
        self.assertIsNone(self._pick(None, _SHAPE_950))


if __name__ == "__main__":
    unittest.main()
