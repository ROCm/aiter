# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""TFLOPS and bandwidth that FmoeTuner.calculate reports for each stage.

Shape: token=128, model_dim=1024, inter_dim=512, expert=8, topk=2, g1u1,
bf16 output, fp8 activations and weights. At us=1, TFLOPS = flop / 1e6 and
bandwidth (GB/s) = bytes / 1e3.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

TOKEN, MODEL_DIM, INTER_DIM, EXPERT, TOPK = 128, 1024, 512, 8, 2

# gemm1: m=token, k=model_dim, n=2*inter_dim.
#   flop  = 128 * 1024 * 1024 * 2 * 2                  = 536_870_912
#   bytes = 128*1024*1 + 128*1024*2 + 1024*1024*1*8     = 8_781_824
STAGE1 = (536.87, 8781.82)
# gemm2: m=token, n=model_dim, k=inter_dim, b=topk.
#   flop  = 2 * 128 * 1024 * 512 * 2                   = 268_435_456
#   bytes = 128*512*1*2 + 128*1024*2 + 512*1024*1*8     = 4_587_520
STAGE2 = (268.44, 4587.52)
# Whole MoE: both GEMMs.
#   flop  = 536_870_912 + 268_435_456                   = 805_306_368
#   bytes = 128*1024*1 + 1024*1024*1*8 + 512*1024*1*8 + 128*1024*2
#                                                       = 12_976_128
WHOLE_MOE = (805.31, 12976.13)


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None, "the fmoe tuner needs torch"
)
class TestFmoeCalculate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from aiter import dtypes
        from aiter.tuning.moe.fused_moe import FmoeTuner

        cls.key = (
            "gfx950",
            256,
            TOKEN,
            MODEL_DIM,
            INTER_DIM,
            EXPERT,
            TOPK,
            "ActivationType.Silu",
            dtypes.bf16,
            dtypes.fp8,
            dtypes.fp8,
            "QuantType.per_Token",
            1,
            0,
        )
        cls.tuner = FmoeTuner("calculateFmoe", [], [], "calculate")

    def calculate(self, stage, us=1.0):
        return self.tuner.calculate((self.key, stage, "kernel", 32, us, 0.0))

    def test_stage1_is_the_first_gemm(self):
        self.assertEqual(self.calculate("stage1"), STAGE1)

    def test_stage2_is_the_second_gemm(self):
        self.assertEqual(self.calculate("stage2"), STAGE2)

    def test_other_stages_are_the_whole_moe(self):
        for stage in ("", "asm_1stage"):
            with self.subTest(stage=stage):
                self.assertEqual(self.calculate(stage), WHOLE_MOE)

    def test_invalid_time_reports_nothing(self):
        for stage in ("stage1", "stage2", ""):
            for us in (self.tuner.INVALID_TIME, self.tuner.INF_TIME):
                with self.subTest(stage=stage, us=us):
                    self.assertEqual(self.calculate(stage, us), (0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
