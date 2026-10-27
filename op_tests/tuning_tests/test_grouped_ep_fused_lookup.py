# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-R1 FP4 grouped-MoE rows are selected on the fused-EP path.

The compact-plan alignment lookup and fused_moe both ask
``_find_grouped_config`` for the same shape. Rows with ``ep_fused=1`` must
win only when that flag is set; a lookup that leaves it false keeps the
64-tile default and the plan alignment disagrees with the GEMM.

Requires torch (importing aiter pulls it in). No GPU: ``GPU_ARCHS`` and
``CU_NUM`` stand in for the device.

Run:
    GPU_ARCHS=gfx1250 CU_NUM=256 python3 -m unittest \
        op_tests.tuning_tests.test_grouped_ep_fused_lookup -v
"""

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("GPU_ARCHS", "gfx1250")
os.environ.setdefault("CU_NUM", "256")

try:
    from aiter import ActivationType, QuantType, dtypes
    from aiter.ops.flydsl.grouped_moe_gfx1250 import _find_grouped_config

    _IMPORT_ERR = None
except Exception as e:  # noqa: BLE001
    ActivationType = QuantType = dtypes = _find_grouped_config = None
    _IMPORT_ERR = e

try:
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250.mega_moe import (
        _compact_gemm_align_m,
    )
except Exception:  # noqa: BLE001
    _compact_gemm_align_m = None


def _row(token, experts, ep_fused):
    return _find_grouped_config(
        token_num=token,
        model_dim=7168,
        inter_dim=2048,
        experts=experts,
        topk=-1,
        activation=ActivationType.Silu,
        dtype=dtypes.bf16,
        q_dtype_a=dtypes.fp4x2,
        q_dtype_w=dtypes.fp4x2,
        quant_type=QuantType.per_1x32,
        ep_fused=ep_fused,
    )


class GroupedEpFusedLookupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _IMPORT_ERR is not None:
            raise unittest.SkipTest(f"aiter import failed: {_IMPORT_ERR}")

    def test_ep_fused_rows_selected(self):
        cases = [
            (1024, 65, "2", "2"),
            (2048, 65, "2", "2"),
            (4096, 65, "3", "3"),
            (4096, 64, "3", "3"),
            (32768, 65, "2", "2"),
        ]
        for token, experts, buffers, buffers2 in cases:
            with self.subTest(token=token, experts=experts):
                row = _row(token, experts, True)
                self.assertIsNotNone(row)
                self.assertEqual(str(row.get("tile_m")), "128")
                self.assertEqual(str(row.get("tile_m2")), "128")
                self.assertEqual(str(row.get("num_buffers")), buffers)
                self.assertEqual(str(row.get("num_buffer_stage2")), buffers2)
                # The merged CSV stores the flag as "1.0"; lookup normalizes it.
                self.assertIn(str(row.get("ep_fused")), ("1", "1.0"))

    def test_plain_lookup_misses_ep_fused_rows(self):
        # These buckets have no blank-ep_fused row, so ep_fused=False must
        # miss and leave the caller on the 64 default.
        for token, experts in ((1024, 65), (4096, 64), (4096, 65), (32768, 65)):
            with self.subTest(token=token, experts=experts):
                self.assertIsNone(_row(token, experts, False))

    def test_token_512_stays_default(self):
        self.assertIsNone(_row(512, 65, True))

    def test_compact_plan_uses_ep_fused_tile(self):
        if _compact_gemm_align_m is None:
            self.skipTest("compact align import failed")
        os.environ.pop("AITER_TDM_TILE_M", None)
        os.environ.pop("AITER_TDM_TILE_M2", None)
        config = SimpleNamespace(
            dispatch_wire="fp4",
            max_recv=32768,
            hidden_dim=7168,
            experts_per_rank=65,
        )
        align = _compact_gemm_align_m(
            config,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            inter_dim=2048,
            recv_bound=32768,
        )
        self.assertEqual(align, 128)
        align_512 = _compact_gemm_align_m(
            config,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            inter_dim=2048,
            recv_bound=512,
        )
        self.assertEqual(align_512, 64)


if __name__ == "__main__":
    unittest.main()
