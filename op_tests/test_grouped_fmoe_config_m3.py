# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3 TP4 MoE resolves to tuned grouped-fmoe rows on gfx1250.

Merges a temp copy of aiter/configs through the runtime get_config_file, then
asks _find_grouped_config for the key the M3 per-rank MoE call builds.

Run:
    python3 -m pytest op_tests/test_grouped_fmoe_config_m3.py -v
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import aiter.ops.flydsl.grouped_moe_gfx1250 as gm
from aiter import ActivationType, QuantType, dtypes
from aiter.jit import core

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMILY = "tuned_grouped_fmoe"
# Per-rank M3 TP4 MoE: hidden 6144, I 768, 128 routed + 1 shared, top-4 + shared.
M3_KEY = dict(
    model_dim=6144,
    inter_dim=768,
    experts=129,
    topk=5,
    activation=ActivationType.Swiglu,
    dtype=dtypes.bf16,
    q_dtype_a=dtypes.fp4x2,
    q_dtype_w=dtypes.fp4x2,
    quant_type=QuantType.per_1x32,
)


class TestGroupedFmoeConfigM3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="aiter_cfg_m3_")
        shutil.copytree(f"{ROOT}/aiter/configs", f"{cls.tmp}/aiter/configs")
        orig_root = core.AITER_ROOT_DIR
        core.AITER_ROOT_DIR = cls.tmp
        type(core.AITER_CONFIGS).get_config_file.cache_clear()
        try:
            with mock.patch.dict(os.environ):
                os.environ.pop("AITER_CONFIG_GROUPED_FMOE", None)
                # Raises on a duplicate shape anywhere in the merged family.
                cls.merged = core.AITER_CONFIGS.get_config_file(
                    "AITER_CONFIG_GROUPED_FMOE",
                    f"{cls.tmp}/aiter/configs/{FAMILY}.csv",
                    FAMILY,
                )
        finally:
            core.AITER_ROOT_DIR = orig_root
            type(core.AITER_CONFIGS).get_config_file.cache_clear()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def lookup(self, token_num, gfx="gfx1250", **over):
        gm._find_grouped_config.cache_clear()
        gm._GROUPED_CONFIG_CACHE.clear()
        with mock.patch.dict(os.environ, {"AITER_CONFIG_GROUPED_FMOE": self.merged}):
            with mock.patch.object(gm, "get_gfx", return_value=gfx), mock.patch(
                "aiter.jit.utils.chip_info.get_cu_num", return_value=256
            ):
                key = {**M3_KEY, **over}
                return gm._find_grouped_config(
                    token_num=gm._get_padded_m(token_num), **key
                )

    def test_every_padded_token_hits_m3_row(self):
        # Every padded token (powers of two to 32768) plus T between them.
        tokens = [*range(1, 17), *(1 << p for p in range(5, 16)), 100, 3000, 131071]
        for t in tokens:
            with self.subTest(T=t):
                row = self.lookup(t)
                self.assertIsNotNone(row, "no tuned row; default 64x256x256 tile")
                self.assertEqual(row["model_dim"], "6144")
                self.assertEqual(row["topk"], "5")
                self.assertEqual(int(row["token"]), gm._get_padded_m(t))

    def test_other_paths_unchanged(self):
        for t in (1, 4, 16, 1024, 32768):
            with self.subTest(T=t):
                # EP keys on topk=-1; other archs never see gfx1250 rows.
                for row in (self.lookup(t, topk=-1), self.lookup(t, gfx="gfx950")):
                    self.assertTrue(row is None or row["model_dim"] != "6144")


if __name__ == "__main__":
    unittest.main()
