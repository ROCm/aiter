# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Selection and write-back of the FlyDSL gfx942 e2e tune (--e2e_tune).

The baseline and every config's timing are stubbed, so nothing runs on the
GPU. Config and _Problem are the real ones, so which config is tried on which
shape follows Config.unsupported_reason.
"""

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

KEYS = [
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
]
RESULT_LIST = [
    "block_m",
    "ksplit",
    "us1",
    "kernelName1",
    "err1",
    "us2",
    "kernelName2",
    "err2",
    "us",
    "run_1stage",
    "xbf16",
    "flat",
    "tflops",
    "bw",
    "nt",
]

# 16_16_16 fits any token count; 16_128_64 is rejected at batch 1; the batch-1
# algorithm only takes 2 to 8 tokens.
ANY = "16_16_16_False:default:0"
NOT_BATCH1 = "16_128_64_False:default:0"
DIRECT = "16_16_16_False_True:default:0"


def shape(token):
    return {
        "gfx": "gfx942",
        "cu_num": 304,
        "token": token,
        "model_dim": 4096,
        "inter_dim": 1024,
        "expert": 8,
        "topk": 2,
        "act_type": "ActivationType.Silu",
        "dtype": "torch.bfloat16",
        "q_dtype_a": "torch.bfloat16",
        "q_dtype_w": "torch.bfloat16",
        "q_type": "QuantType.No",
        "use_g1u1": 1,
        "doweight_stage1": 0,
    }


def result(e2e_us, status="ok", err_ratio=0.0):
    return {"e2e_us": e2e_us, "status": status, "err_ratio": err_ratio}


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None, "the fmoe tuner needs torch"
)
class TestFmoeE2eTune(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd

        from aiter.fused_moe_registry import make_fused_moe_impl_kernel_name
        from aiter.ops.flydsl import fused_moe_gfx942
        from aiter.tuning.moe.fused_moe import FmoeTuner

        cls.pd = pd
        cls.kernel_name = staticmethod(make_fused_moe_impl_kernel_name)
        cls.fused_moe_gfx942 = fused_moe_gfx942
        cls.FmoeTuner = FmoeTuner

    def setUp(self):
        self.tuner = self.FmoeTuner("e2eFmoe", KEYS, RESULT_LIST, "e2e tune")
        with mock.patch.object(sys, "argv", ["gemm_moe_tune.py"]):
            self.args = self.tuner.parse_args()
        self.untuned = self.pd.DataFrame([shape(1), shape(4)])
        self.tuner.untunedf = self.untuned.copy()
        self.tried = []

    def run_config(self, args, target_fused_moe=None, config_string=""):
        tokens = list(self.tuner.untunedf["token"])
        self.tried.append((config_string, tokens))
        times = {
            (ANY, 1): result(12.0),
            (ANY, 4): result(15.0),
            (NOT_BATCH1, 4): result(1.0, status="mismatch", err_ratio=1.0),
            (DIRECT, 4): result(9.0),
        }
        return [times[(config_string, token)] for token in tokens]

    def e2e_tune(self, existing_rows):
        with tempfile.TemporaryDirectory() as root:
            out = os.path.join(root, "tuned_fmoe.csv")
            if existing_rows:
                self.pd.DataFrame(existing_rows).to_csv(out, index=False)
            self.args.tune_file = out
            baseline = [result(10.0), result(20.0)]
            with (
                mock.patch.object(
                    self.fused_moe_gfx942,
                    "get_tune_space",
                    return_value=[ANY, NOT_BATCH1, DIRECT],
                ),
                mock.patch.object(
                    self.tuner, "_run_config_for_shapes", return_value=baseline
                ),
                mock.patch.object(self.tuner, "run_config", self.run_config),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.tuner.e2e_tune(self.args)
            if not os.path.exists(out):
                return None
            return self.pd.read_csv(out)

    def test_each_config_only_sees_the_shapes_it_supports(self):
        self.e2e_tune([])
        self.assertEqual(self.tried, [(ANY, [1, 4]), (NOT_BATCH1, [4]), (DIRECT, [4])])
        self.pd.testing.assert_frame_equal(self.tuner.untunedf, self.untuned)

    def test_fastest_valid_config_wins_only_where_it_beats_the_baseline(self):
        tuned = self.e2e_tune([])
        self.assertEqual(list(tuned["token"]), [4])
        row = tuned.iloc[0]
        self.assertEqual(row["kernelName1"], self.kernel_name("flydsl_gfx942", DIRECT))
        self.assertEqual(row["us"], 9.0)
        self.assertEqual(row["us1"], 9.0)
        self.assertEqual(row["block_m"], 16)
        self.assertEqual(row["run_1stage"], 0)
        self.assertEqual(row["nt"], 0)

    def test_existing_rows_are_kept_unless_replaced(self):
        old = {**shape(4), **{c: 0 for c in RESULT_LIST}, "kernelName1": "old"}
        other = {**shape(64), **{c: 0 for c in RESULT_LIST}, "kernelName1": "other"}
        tuned = self.e2e_tune([old, other])
        by_token = dict(zip(tuned["token"], tuned["kernelName1"]))
        self.assertEqual(
            by_token,
            {4: self.kernel_name("flydsl_gfx942", DIRECT), 64: "other"},
        )

    def test_no_improvement_writes_nothing(self):
        self.tuner.untunedf = self.pd.DataFrame([shape(1)])
        self.untuned = self.tuner.untunedf.copy()
        with mock.patch.object(
            self,
            "run_config",
            lambda args, target_fused_moe=None, config_string="": [result(50.0)]
            * len(self.tuner.untunedf),
        ):
            self.assertIsNone(self.e2e_tune([]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
