# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-R1 FP4 grouped-MoE rows are selected on the fused-EP path.

The compact-plan alignment lookup and fused_moe both ask
``_find_grouped_config`` for the same shape. Rows with ``ep_fused=1`` must
win only when that flag is set. A lookup that leaves it false stays on the
generic row, and the plan alignment would disagree with the fused GEMM.

Requires torch (importing aiter pulls it in). No GPU: ``GPU_ARCHS`` and
``CU_NUM`` stand in for the device.

Run:
    GPU_ARCHS=gfx1250 CU_NUM=256 python3 -m unittest \
        op_tests.tuning_tests.test_grouped_ep_fused_lookup -v
"""

import csv
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("GPU_ARCHS", "gfx1250")
os.environ.setdefault("CU_NUM", "256")

try:
    from aiter import ActivationType, QuantType, dtypes
    from aiter.ops.flydsl.grouped_moe_gfx1250 import (
        _find_grouped_config,
        _get_padded_m,
    )

    _IMPORT_ERR = None
except Exception as e:  # noqa: BLE001
    ActivationType = QuantType = dtypes = _find_grouped_config = _get_padded_m = None
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

    def test_plain_lookup_keeps_generic_rows(self):
        # Upstream rows leave ep_fused blank, so they match both paths and
        # the explicit fused rows must not be selected when ep_fused is false.
        for token, experts in ((1024, 65), (4096, 64), (4096, 65), (32768, 65)):
            with self.subTest(token=token, experts=experts):
                row = _row(token, experts, False)
                self.assertIsNotNone(row)
                self.assertNotEqual(str(row.get("tile_m")), "128")
                self.assertNotIn(str(row.get("ep_fused")), ("1", "1.0"))

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


class GroupedNonEpTokenFallbackTest(unittest.TestCase):
    """Token overflow reuses only compatible EP or non-EP configurations."""

    def setUp(self):
        if _IMPORT_ERR is not None:
            self.skipTest(f"aiter import failed: {_IMPORT_ERR}")
        self.rows = []
        patcher = mock.patch(
            "aiter.ops.flydsl.grouped_moe_gfx1250._load_grouped_config_rows",
            return_value=self.rows,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        for target, value in (
            ("aiter.ops.flydsl.grouped_moe_gfx1250.get_gfx", "gfx1250"),
            ("aiter.jit.utils.chip_info.get_cu_num", 256),
        ):
            patcher = mock.patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        _find_grouped_config.cache_clear()
        self.addCleanup(_find_grouped_config.cache_clear)

    def add_row(self, token, **overrides):
        row = {
            "gfx": "gfx1250",
            "cu_num": "256",
            "token": str(token),
            "model_dim": "5120",
            "inter_dim": "2304",
            "expert": "96",
            "topk": "6",
            "act_type": "ActivationType.Silu",
            "dtype": "torch.bfloat16",
            "q_dtype_a": str(dtypes.fp4x2),
            "q_dtype_w": str(dtypes.fp4x2),
            "q_type": "QuantType.per_1x32",
            "ep_fused": "",
            "us": "0",
        }
        row.update(overrides)
        self.rows.append(row)
        return row

    def lookup(self, token, *, topk=6, ep_fused=False):
        return _find_grouped_config(
            token_num=token,
            model_dim=5120,
            inter_dim=2304,
            experts=96,
            topk=topk,
            activation=ActivationType.Silu,
            dtype=dtypes.bf16,
            q_dtype_a=dtypes.fp4x2,
            q_dtype_w=dtypes.fp4x2,
            quant_type=QuantType.per_1x32,
            ep_fused=ep_fused,
        )

    def test_largest_compatible_token(self):
        self.add_row(8192)
        expected = self.add_row("16384.0")
        for field, value in (
            ("gfx", "gfx950"),
            ("model_dim", "7168"),
            ("inter_dim", "2048"),
            ("expert", "64"),
            ("topk", "-1"),
            ("act_type", "ActivationType.Swiglu"),
            ("dtype", "torch.float16"),
            ("q_dtype_a", "torch.float8_e4m3fn"),
            ("q_dtype_w", "torch.float8_e4m3fn"),
            ("q_type", "QuantType.per_Tensor"),
            ("ep_fused", "1"),
        ):
            self.add_row(131072, **{field: value})
        self.assertIs(self.lookup(32768), expected)

    def test_exact_token_wins(self):
        expected = self.add_row(32768)
        self.add_row(131072)
        self.assertIs(self.lookup(32768), expected)

    def test_cu_preference_and_relaxed_fallback(self):
        expected = self.add_row(16384)
        other_cu = self.add_row(8192, cu_num="128")
        self.assertIs(self.lookup(32768), expected)
        self.assertIs(self.lookup(8192), other_cu)
        self.rows.remove(expected)
        _find_grouped_config.cache_clear()
        self.assertIs(self.lookup(32768), other_cu)

    def test_missing_token_below_maximum_does_not_fall_back(self):
        largest = self.add_row(16384)
        self.assertIsNone(self.lookup(8192))
        self.assertIs(self.lookup(32768), largest)

    def test_scatter_specificity_is_preserved(self):
        for topk in (6, -1):
            with self.subTest(topk=topk):
                self.add_row(16384, topk=str(topk), us="1")
                plain = self.add_row(16384, topk=str(topk), ep_fused="0")
                fused = self.add_row(16384, topk=str(topk), ep_fused="1")
                self.assertIs(self.lookup(32768, topk=topk), plain)
                self.assertIs(self.lookup(32768, topk=topk, ep_fused=True), fused)

    def test_ep_fallback_keeps_topk_and_model_separate(self):
        ep = self.add_row(16384, topk="-1")
        self.add_row(16384, model_dim="7168")
        self.assertIsNone(self.lookup(32768))
        self.add_row(65536)
        self.assertIs(self.lookup(32768, topk=-1), ep)
        self.assertIsNone(self.lookup(8192, topk=-1))

    def test_flash_csv_32768_uses_16384(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "aiter/configs/model_configs/dsv41_flash_tuned_grouped_fmoe.csv"
        )
        with path.open() as stream:
            self.rows.extend(csv.DictReader(stream))
        row = self.lookup(32768)
        self.assertEqual(row["token"], "16384")
        self.assertEqual(row["topk"], "6")
        self.assertEqual(
            tuple(row[key] for key in ("cluster_m", "cluster_n")), ("4", "2")
        )
        for token in (16385, 32768, 32769, 65536, 65537, 131072, 262144):
            with self.subTest(token=token):
                self.assertEqual(self.lookup(_get_padded_m(token)), row)
        for exponent in range(15):
            token = 1 << exponent
            with self.subTest(exact_token=token):
                self.assertEqual(self.lookup(_get_padded_m(token))["token"], str(token))

    def test_token_padding_has_one_upper_bound(self):
        for token, expected in (
            (1, 1),
            (3, 4),
            (16385, 32768),
            (32768, 32768),
            (32769, 65536),
            (65536, 65536),
            (65537, 131072),
            (131072, 131072),
            (262144, 131072),
        ):
            with self.subTest(token=token):
                self.assertEqual(_get_padded_m(token), expected)

    def test_ep_65536_row_can_be_selected_after_padding(self):
        self.add_row(32768, topk="-1")
        expected = self.add_row(65536, topk="-1")
        self.assertIs(self.lookup(_get_padded_m(65536), topk=-1), expected)


if __name__ == "__main__":
    unittest.main()
