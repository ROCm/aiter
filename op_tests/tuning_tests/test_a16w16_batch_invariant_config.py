# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import copy
import unittest
from unittest import mock

import aiter.tuned_gemm as tuned

BF16 = "torch.bfloat16"


class TestBatchInvariantConfig(unittest.TestCase):
    def setUp(self):
        self.table = {}
        for name, value in (("get_gfx", "gfx950"), ("get_cu_num", 256)):
            patch = mock.patch.object(tuned, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(tuned, "get_GEMM_A16W16_config_", lambda: self.table)
        patch.start()
        self.addCleanup(patch.stop)
        for fn in (
            tuned.get_GEMM_A16W16_config,
            tuned.get_GEMM_A16W16_batch_invariant_configs_,
        ):
            fn.cache_clear()
            self.addCleanup(fn.cache_clear)

    def row(self, m, flag=True, **overrides):
        row = {
            "libtype": "asm",
            "solidx": m,
            "splitK": 1,
            "kernelName": f"kernel{m}",
            "batch_invariant": flag,
        }
        row.update(overrides)
        self.table[("gfx950", 256, m, 32768, 512, False, BF16, BF16, False, False)] = (
            row
        )
        return row

    def lookup(self, m, **overrides):
        args = {
            "M": m,
            "N": 32768,
            "K": 512,
            "bias": False,
            "dtype": BF16,
            "otype": BF16,
        }
        args.update(overrides)
        return tuned.get_GEMM_A16W16_tuned_config(**args)

    def test_ceiling_and_last_bucket_override_exact_speed_rows(self):
        for m in (96, 1, 64):
            self.row(m)
        self.row(65, False, libtype="torch")
        original = copy.deepcopy(self.table)
        for m, expected in (
            (1, 1),
            (2, 64),
            (63, 64),
            (64, 64),
            (65, 96),
            (95, 96),
            (96, 96),
            (97, 96),
            (8192, 96),
        ):
            with self.subTest(m=m):
                result = self.lookup(m)
                self.assertEqual(result["_config_m"], expected)
                self.assertEqual(result["_selection_source"], "batch_invariant_family")
        self.assertEqual(self.table, original)

    def test_merged_false_values_do_not_enable_a_family(self):
        for flag in (False, 0, 0.0, None, float("nan"), "False"):
            with self.subTest(flag=flag):
                self.row(1, flag)
                tuned.get_GEMM_A16W16_batch_invariant_configs_.cache_clear()
                self.assertNotIn("_selection_source", self.lookup(1))
        self.row(1, "True")
        tuned.get_GEMM_A16W16_batch_invariant_configs_.cache_clear()
        self.assertEqual(self.lookup(1)["_selection_source"], "batch_invariant_family")

    def test_other_families_keep_normal_lookup(self):
        self.row(1)
        for changes in (
            {"N": 32769},
            {"K": 513},
            {"bias": True},
            {"dtype": "torch.float16"},
            {"otype": "torch.float32"},
            {"scaleAB": True},
            {"bpreshuffle": True},
        ):
            with self.subTest(changes=changes):
                self.assertIsNone(self.lookup(1, **changes))
        for name, value in (("get_gfx", "gfx942"), ("get_cu_num", 304)):
            with mock.patch.object(tuned, name, return_value=value):
                self.assertIsNone(self.lookup(1))

    def test_stale_catalog_entry_uses_another_family_member(self):
        self.row(1, libtype="flydsl", kernelName="missing")
        self.row(64)
        catalog = mock.Mock()
        catalog.get_flydsl_hgemm_kernel_params.return_value = None
        with mock.patch.object(tuned, "_get_flydsl_gemm_kernels", return_value=catalog):
            self.assertEqual(self.lookup(1)["_config_m"], 64)


if __name__ == "__main__":
    unittest.main()
