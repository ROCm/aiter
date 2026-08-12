# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU-free regression tests for FlyDSL MoE AOT job collection."""

import csv
import os
import tempfile
import unittest
from unittest.mock import patch


class TestMoeAotJobs(unittest.TestCase):
    def test_dynamic_mxmoe_replacement_adds_sorted_stage2(self):
        from aiter.aot.flydsl.moe import parse_csv

        row = {
            "cu_num": "256",
            "token": "1",
            "model_dim": "3072",
            "inter_dim": "256",
            "expert": "256",
            "topk": "8",
            "act_type": "ActivationType.Silu",
            "dtype": "torch.bfloat16",
            "q_dtype_a": "torch.float4_e2m1fn_x2",
            "q_dtype_w": "torch.float4_e2m1fn_x2",
            "q_type": "QuantType.per_1x32",
            "use_g1u1": "1",
            "doweight_stage1": "0",
            "block_m": "32",
            "kernelName1": "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w4_fp4",
            "kernelName2": (
                "moe_ck2stages_gemm2_64x32x32x128_1x1_"
                "MulABScaleExpertWeightShuffled_v1"
            ),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", newline=""
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)
            csv_file.flush()
            with patch.dict(os.environ, {"AITER_MXFP4_GEMM1_REPLACEMENT": "1"}):
                jobs = parse_csv(csv_file.name)

        expected_name = "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce"
        stage2_jobs = [job for job in jobs if job["kernel_name"] == expected_name]
        self.assertTrue(stage2_jobs)
        self.assertTrue(any(job["kernel_name"] == row["kernelName1"] for job in jobs))
        for job in stage2_jobs:
            self.assertEqual(job["stage"], 2)
            self.assertEqual(job["stage1_fuse_quant"], "fp4")
            self.assertTrue(job["intermediate_sorted"])


if __name__ == "__main__":
    unittest.main()
