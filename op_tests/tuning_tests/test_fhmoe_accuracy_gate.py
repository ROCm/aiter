# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FHMoE tuner must not pick a fast-but-wrong kernel pair."""

from __future__ import annotations

import argparse
import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd

AITER_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if AITER_ROOT not in sys.path:
    sys.path.insert(0, AITER_ROOT)

try:
    from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import (  # noqa: E402
        FhmoeTuner,
        FmoeTuner,
        cosine_diff_compare,
    )
except ImportError as _import_error:  # torch / aiter not in this interpreter
    FhmoeTuner = None
    FmoeTuner = None
    cosine_diff_compare = None
    _IMPORT_ERROR = _import_error
else:
    _IMPORT_ERROR = None

_FHMOE_KEYS = [
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
    "shared_expert_id",
    "hidden_pad",
    "intermediate_pad",
    "gate_mode",
]
_FHMOE_RESULTS = [
    "block_m",
    "ksplit",
    "kernelName1",
    "kernelName2",
    "us",
]


def _tuner():
    return FhmoeTuner("fhmoeTuner", _FHMOE_KEYS, _FHMOE_RESULTS, "fhmoe tuner")


def _info():
    return (
        "gfx950",
        256,
        1,
        7168,
        384,
        385,
        7,
        "ActivationType.Silu",
        "torch.bfloat16",
        "torch.float8_e4m3fn",
        "torch.float4_e2m1fn_x2",
        "QuantType.per_1x32",
        1,
        0,
        384,
        0,
        0,
        "GateMode.INTERLEAVE",
    )


def _untuned_row():
    return {
        "gfx": "gfx950",
        "cu_num": 256,
        "token": 1,
        "model_dim": 7168,
        "inter_dim": 384,
        "expert": 385,
        "topk": 7,
        "shared_expert_id": 384,
        "act_type": "ActivationType.Silu",
        "dtype": "torch.bfloat16",
        "q_dtype_a": "torch.float8_e4m3fn",
        "q_dtype_w": "torch.float4_e2m1fn_x2",
        "q_type": "QuantType.per_1x32",
        "use_g1u1": 1,
        "doweight_stage1": 0,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "gate_mode": "GateMode.INTERLEAVE",
    }


@unittest.skipIf(FhmoeTuner is None, f"gemm_moe_tune import failed: {_IMPORT_ERROR}")
class TestFhmoeAccuracyGate(unittest.TestCase):
    def test_tune_tasks_attach_torch_ref_and_cosine(self):
        tuner = _tuner()
        df = pd.DataFrame([_untuned_row()])
        with patch.object(
            FhmoeTuner, "_kernel_pairs", return_value=[(32, "kn1", "kn2")]
        ):
            tasks, in_datas = tuner._fhmoe_tune_tasks(df)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(in_datas, [(1, ())])
        task = tasks[0]
        self.assertIs(task[6], FhmoeTuner.run_torch_fhmoe)
        self.assertEqual(task[7][0], list(FhmoeTuner._FHMOE_REF_DATA_KEYS))
        self.assertIs(task[12], cosine_diff_compare)

    def test_post_process_rejects_fast_wrong_pair(self):
        tuner = _tuner()
        info = _info()
        results = [
            ((info, "fast_wrong1", "fast_wrong2", 32), 10.0, 0.9),
            ((info, "slow_ok1", "slow_ok2", 64), 50.0, 0.01),
        ]
        args = argparse.Namespace(errRatio=0.1, profile_file="")
        out = tuner.post_process(results, args)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["kernelName1"], "slow_ok1")
        self.assertEqual(out.iloc[0]["kernelName2"], "slow_ok2")
        self.assertEqual(int(out.iloc[0]["block_m"]), 64)

    def test_post_process_drops_shape_when_all_pairs_fail_err(self):
        tuner = _tuner()
        info = _info()
        results = [((info, "bad1", "bad2", 32), 10.0, 0.5)]
        args = argparse.Namespace(errRatio=0.1, profile_file="")
        out = tuner.post_process(results, args)
        self.assertTrue(out.empty)

    def test_run_config_is_not_parent_fmoe(self):
        self.assertIsNot(FhmoeTuner.run_config, FmoeTuner.run_config)

    def test_run_config_calls_public_fused_moe_with_shared(self):
        import torch

        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune as tune_mod

        tuner = _tuner()
        tuner.untunedf = pd.DataFrame(
            [{**_untuned_row(), "gfx": "gfx950", "cu_num": 256}]
        )
        args = argparse.Namespace(warmup=1, iters=1, errRatio=0.1)
        fake = torch.ones((1, 8))
        data = {
            "hidden": fake,
            "w1": fake,
            "w2": fake,
            "w1_scale": fake,
            "w2_scale": fake,
            "shared_w1": fake,
            "shared_w2": fake,
            "shared_w1_scale": fake,
            "shared_w2_scale": fake,
            "topk_weights": fake,
            "topk_ids": fake,
            "w1_qt": fake,
            "w2_qt": fake,
            "w1_scale_qt": fake,
            "w2_scale_qt": fake,
            "shared_w1_qt": fake,
            "shared_w2_qt": fake,
            "shared_w1_scale_qt": fake,
            "shared_w2_scale_qt": fake,
            "shared_expert_id": 384,
        }
        with patch.object(FhmoeTuner, "generate_fhmoe_data", return_value=data), patch.object(
            FhmoeTuner, "run_torch_fhmoe", return_value=fake
        ), patch(
            "aiter.test_common.run_perftest", return_value=(fake, 40.0)
        ) as perf:
            out = tuner.run_config(args)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["status"], "ok")
        self.assertEqual(out[0]["e2e_us"], 40.0)
        self.assertIn("shared=", out[0]["shape"])
        self.assertIs(perf.call_args[0][0], tune_mod.fused_moe)
        self.assertIn("shared_w1", perf.call_args[1])
        self.assertIn("shared_w2", perf.call_args[1])
        self.assertEqual(perf.call_args[1]["shared_expert_id"], 384)


if __name__ == "__main__":
    unittest.main()
