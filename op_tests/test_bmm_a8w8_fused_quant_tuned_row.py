# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The fused-quant batched entry must run the bf16 entry's tuned row.

``_fused_quant_tuned_kernel`` looks the shape up in the same preshuffle table
as ``batched_gemm_a8w8_mxscale_bpreshuffle`` and hands a flydsl row's
kernelName to the runner as is. No row, or an opus row, leaves the choice to
the runner (None). CPU only: the table lookup is patched, nothing launches.
"""

import unittest
from unittest.mock import patch

_LOOKUP = "aiter.ops.batched_gemm_op_a8w8.lookup_mxscale_bmm_config"
_SHAPE = (8, 48, 1024, 4096, "32x32", "gfx1250", 256)
_FLYDSL_NAME = "flydsl_bmm_mxfp8_32_wmma_t32x64x512_mw2_nw2_nb4_sk1_cm1_cn1"


class TestFusedQuantTunedRow(unittest.TestCase):
    def _pick(self, row):
        from aiter.ops import batched_gemm_op_a8w8 as op

        op._fused_quant_tuned_kernel.cache_clear()
        try:
            with patch(_LOOKUP, return_value=row) as lookup:
                name = op._fused_quant_tuned_kernel(*_SHAPE)
        finally:
            op._fused_quant_tuned_kernel.cache_clear()
        lookup.assert_called_once()
        args, kwargs = lookup.call_args
        self.assertEqual(args, _SHAPE[:4])
        self.assertEqual(kwargs["w_scale_block"], "32x32")
        self.assertTrue(kwargs["bpreshuffle"])
        self.assertEqual((kwargs["gfx"], kwargs["cu_num"]), ("gfx1250", 256))
        return name

    def test_no_row_keeps_the_runner_choice(self):
        self.assertIsNone(self._pick(None))

    def test_opus_row_keeps_the_runner_choice(self):
        row = {"libtype": "opus", "kernelName": "opus_bmm_a8w8_mxscale_tile"}
        self.assertIsNone(self._pick(row))

    def test_flydsl_row_name_is_passed_through(self):
        row = {"libtype": "flydsl", "kernelName": _FLYDSL_NAME}
        self.assertEqual(self._pick(row), _FLYDSL_NAME)

    def test_unparsable_flydsl_name_is_passed_through(self):
        # The runner rejects it with ValueError, as for the bf16 entry; the
        # fused entry must not swap in the heuristic tile.
        row = {"libtype": "flydsl", "kernelName": "flydsl_bmm_not_a_kernel"}
        self.assertEqual(self._pick(row), "flydsl_bmm_not_a_kernel")


if __name__ == "__main__":
    unittest.main()
