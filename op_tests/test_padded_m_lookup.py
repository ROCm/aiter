# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
The tuned-config walk (exact M -> ``getPaddedM(gl=0)`` -> ``getPaddedM(gl=1)``)
must reach an M = 16 row for 0 < M < 16 when no power-of-two row is tuned,
while still preferring a tuned M = 4 / 8 row when one exists.

Uses the real compiled ``get_padded_m`` (``module_gemm_common``), not a mock.

Run:
    python3 op_tests/test_padded_m_lookup.py -v
"""

import unittest
from unittest import mock

import torch

import aiter.tuned_gemm as tuned
from aiter.ops.gemm_op_common import get_padded_m

GFX, CU_NUM = "gfx950", 256
N, K = 8192, 1024
BF16 = str(torch.bfloat16)


def _key(m):
    return (GFX, CU_NUM, m, N, K, False, BF16, BF16, False, False)


def _row(name):
    return {"libtype": "asm", "solidx": 0, "splitK": 1, "kernelName": name}


def _next_pow2(m):
    return 1 if m <= 1 else 1 << (m - 1).bit_length()


class TestGetPaddedM(unittest.TestCase):
    def test_small_m_levels(self):
        for m in range(1, 16):
            with self.subTest(M=m):
                self.assertEqual(get_padded_m(m, N, K, 0), _next_pow2(m))
                self.assertEqual(get_padded_m(m, N, K, 1), 16)


class TestA16W16PaddedWalk(unittest.TestCase):
    def _lookup(self, table, m):
        patches = [
            mock.patch.object(tuned, "get_GEMM_A16W16_config_", lambda: table),
            mock.patch.object(tuned, "get_gfx", lambda: GFX),
            mock.patch.object(tuned, "get_cu_num", lambda: CU_NUM),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return tuned.get_GEMM_A16W16_tuned_config(m, N, K, False, BF16, BF16)

    def test_falls_back_to_m16_row(self):
        # Table tuned only at M = 1 and 16 (e.g. llama70B_bf16_tuned_gemm.csv).
        table = {_key(1): _row("m1"), _key(16): _row("m16")}
        for m in range(2, 16):
            with self.subTest(M=m):
                self.assertEqual(
                    (self._lookup(table, m) or {}).get("kernelName"), "m16"
                )

    def test_prefers_tuned_pow2_row(self):
        table = {_key(4): _row("m4"), _key(8): _row("m8"), _key(16): _row("m16")}
        for m, want in [(3, "m4"), (5, "m8"), (7, "m8"), (8, "m8"), (12, "m16")]:
            with self.subTest(M=m):
                self.assertEqual((self._lookup(table, m) or {}).get("kernelName"), want)


if __name__ == "__main__":
    unittest.main()
