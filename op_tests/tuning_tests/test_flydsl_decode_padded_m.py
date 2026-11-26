# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
A FlyDSL decode row at M = 4 must not take M = 3 away from a tuned kernel.

``get_padded_m`` maps M = 3 to 4, so a cell with no M = 3 row runs its M = 4 row.
Decode kernels are exact-M and are never reused for a padded M, so once the M = 4
row of a cell is a decode row, an M = 3 call there needs a row of its own or it
falls through to the default. This checks the shipped bf16 tuned CSVs.

Requires no GPU and no torch.

Run:
    python3 -m unittest op_tests.tuning_tests.test_flydsl_decode_padded_m -v
"""

import csv
import unittest
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[2] / "aiter" / "configs"

# Cells with a decode row at M = 4 and no M = 3 row that had no M = 4 row before the
# decode rows either, so M = 3 already took the default there. Key: (gfx, cu_num, N, K).
NO_M4_ROW_BEFORE_DECODE = {
    ("gfx942", "304", "6288", "7168"),
    ("gfx942", "304", "7168", "3072"),
    ("gfx942", "304", "7168", "768"),
    ("gfx942", "304", "896", "7168"),
    ("gfx950", "256", "7168", "3072"),
}


def cells_missing_m3(rows):
    """(gfx, cu_num, N, K) of every cell whose M = 4 row is a decode row and that has no
    M = 3 row, over one CSV's rows."""
    cells = {}
    for r in rows:
        key = (
            r["gfx"],
            r["cu_num"],
            r["N"],
            r["K"],
            r["bias"],
            r["dtype"],
            r["outdtype"],
        )
        cells.setdefault(key, {})[int(r["M"])] = r["kernelName"]
    return {
        k[:4]
        for k, by_m in cells.items()
        if by_m.get(4, "").startswith("flydsl_decode_") and 3 not in by_m
    }


class TestFlydslDecodePaddedM(unittest.TestCase):
    def test_decode_m4_rows_keep_m3_covered(self):
        files = sorted(CONFIGS.glob("**/*bf16_tuned_gemm.csv"))
        self.assertTrue(files, CONFIGS)
        for path in files:
            with self.subTest(csv=path.name):
                missing = cells_missing_m3(csv.DictReader(path.open()))
                self.assertFalse(missing - NO_M4_ROW_BEFORE_DECODE, path.name)

    def test_self_check_catches_a_missing_m3_row(self):
        decode = "flydsl_decode_agfx950_m4_n64_k64_x"
        rows = [
            {
                "gfx": "gfx950",
                "cu_num": "256",
                "M": "4",
                "N": "64",
                "K": "64",
                "bias": "False",
                "dtype": "torch.bfloat16",
                "outdtype": "torch.bfloat16",
                "kernelName": decode,
            },
        ]
        self.assertEqual(cells_missing_m3(rows), {("gfx950", "256", "64", "64")})
        rows.append(dict(rows[0], M="3", kernelName="flydsl_hgemm_x"))
        self.assertEqual(cells_missing_m3(rows), set())


if __name__ == "__main__":
    unittest.main()
