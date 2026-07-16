# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
import operator
import unittest
from pathlib import Path

_MOE_COMMON_PATH = (
    Path(__file__).resolve().parents[2] / "aiter/ops/flydsl/moe_common.py"
)
_SPEC = importlib.util.spec_from_file_location("moe_common", _MOE_COMMON_PATH)
_MOE_COMMON = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOE_COMMON)
xcd_swizzle_workgroup_id = _MOE_COMMON.xcd_swizzle_workgroup_id


class TestMoeXcdSwizzle(unittest.TestCase):
    def assert_permutation(self, num_workgroups):
        mapped_ids = [
            xcd_swizzle_workgroup_id(
                linear_id,
                num_workgroups,
                8,
                divide=operator.floordiv,
                minimum=min,
            )
            for linear_id in range(num_workgroups)
        ]
        self.assertTrue(
            all(0 <= mapped_id < num_workgroups for mapped_id in mapped_ids)
        )
        self.assertEqual(len(set(mapped_ids)), num_workgroups)

    def test_divisible_and_remainder_grids_are_permutations(self):
        for num_workgroups in range(1, 257):
            with self.subTest(num_workgroups=num_workgroups):
                self.assert_permutation(num_workgroups)

    def test_dsv4_stage2_grids_are_permutations(self):
        for num_workgroups in (28 * 400, 28 * 399):
            with self.subTest(num_workgroups=num_workgroups):
                self.assert_permutation(num_workgroups)


if __name__ == "__main__":
    unittest.main()
