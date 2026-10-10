# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU selection coverage for explicit MXMOE A8W4 -> OPUS CSV rows."""

import csv
import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from op_tests.tuning_tests.mxfp4_cpu_test_utils import cpu_tuner_environment


class TestMxmoeOpusRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = cpu_tuner_environment()
        cls.environment.__enter__()
        cls.fm = importlib.import_module("aiter.fused_moe")
        cls.torch = importlib.import_module("torch")
        cls.aiter = importlib.import_module("aiter")

    @classmethod
    def tearDownClass(cls):
        cls.environment.__exit__(None, None, None)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.g1 = "flydsl_mxmoe_g1_a8w4_32x128x256_fp8out"
        self.g2 = "opus_moe2_layout_afp8_wfp4_atomic_t32x128x256_sbm32_occ1_cache_b3_ws2"
        self.path = Path(self.directory.name) / "explicit.csv"
        self.write_csv()
        self.env = patch.dict(os.environ, {
            "AITER_CONFIG_FMOE": str(self.path), "AITER_ONLINE_TUNE": "0",
            "AITER_BYPASS_TUNE_CONFIG": "0", "AITER_MXMOE_FALLBACK": "0",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.fm.AITER_CONFIGS.get_config_file.cache_clear()
        self.fm.cfg_2stages = None
        self.fm.get_2stage_cfgs.cache_clear()
        self.addCleanup(self.fm.get_2stage_cfgs.cache_clear)
        self.addCleanup(self.fm.AITER_CONFIGS.get_config_file.cache_clear)
        self.addCleanup(setattr, self.fm, "cfg_2stages", None)

    def write_csv(self, **changes):
        a = self.aiter
        self.row = dict(gfx="gfx950", cu_num=256, token=32, model_dim=3072,
            inter_dim=512, expert=128, topk=4, act_type=str(a.ActivationType.Silu),
            dtype=str(a.dtypes.bf16), q_dtype_a=str(a.dtypes.fp8),
            q_dtype_w=str(a.dtypes.fp4x2), q_type=str(a.QuantType.per_1x32),
            use_g1u1=1, doweight_stage1=0, kernelName1=self.g1,
            kernelName2=self.g2, block_m=32, ksplit=0, run_1stage=0)
        self.row.update(changes)
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.row)
            writer.writeheader()
            writer.writerow(self.row)

    def select(self, **kwargs):
        a = self.aiter
        options = dict(gate_mode="interleave", opus_weights_shuffled=True,
            input_dtype=a.dtypes.bf16)
        options.update(kwargs)
        return self.fm.get_2stage_cfgs(32, 3072, 512, 128, 4, a.dtypes.bf16,
            a.dtypes.fp8, a.dtypes.fp4x2, a.QuantType.per_1x32, True,
            a.ActivationType.Silu, False, 0, 0, **options)

    def test_real_csv_selects_direct_opus_and_regular_g1(self):
        metadata = self.select()
        self.assertEqual(metadata.stage1.keywords["kernelName1"], self.g1)
        self.assertIs(metadata.stage2.func, self.fm._opus_a8w4.opus_a8w4_stage2_wrapper)
        self.assertEqual(metadata.stage2.keywords["kernelName"], self.g2)
        self.assertEqual(metadata.stage2.keywords["launch"].kernel_id, 2005)
        self.assertFalse(metadata.stage1.keywords["native_scale_layout"])
        self.assertTrue(metadata.prequant)
        self.assertEqual(metadata.block_m, 32)
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_non_fp8out_and_mismatched_csv_block_fall_back(self):
        for changes, reason in (
            ({"kernelName1": self.g1.replace("_fp8out", "")}, "FP8out"),
            ({"block_m": 16}, "sort block_m=32"),
            ({"kernelName1": self.g1.replace("32x", "64x")}, "GEMM1 BM=64"),
        ):
            with self.subTest(changes=changes):
                self.write_csv(**changes)
                self.fm.cfg_2stages = None
                self.fm.get_2stage_cfgs.cache_clear()
                with self.assertLogs("aiter", level="WARNING") as logs:
                    metadata = self.select()
                self.assertNotEqual(metadata.stage1.keywords.get("kernelName1"), self.row["kernelName1"])
                self.assertTrue(any(reason in message for message in logs.output))

    def test_unsupported_call_semantics_fall_back(self):
        for kwargs, reason in (
            ({"has_stage2_bias": True}, "stage2 bias"),
            ({"is_ep": True}, "EP"),
            ({"q_dtype_a2": self.aiter.dtypes.fp4x2}, "FP8 intermediate"),
            ({"opus_weights_shuffled": False}, "is_shuffled"),
            ({"has_num_local_tokens": True}, "num_local_tokens"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertLogs("aiter", level="WARNING") as logs:
                    metadata = self.select(**kwargs)
                self.assertIsNot(metadata.stage2.func, self.fm._opus_a8w4.opus_a8w4_stage2_wrapper)
                self.assertTrue(any(reason in message for message in logs.output))

    def test_explicit_fp8_without_original_scales_is_invalid_before_fallback(self):
        a, torch = self.aiter, self.torch
        hidden = torch.empty((31, 3072), dtype=a.dtypes.fp8)
        w1 = torch.empty((128, 1024, 1536), dtype=a.dtypes.fp4x2)
        w2 = torch.empty((128, 3072, 256), dtype=a.dtypes.fp4x2)
        with self.assertRaisesRegex(ValueError, "original per-token E8M0"):
            self.fm.fused_moe(hidden, w1, w2,
                torch.empty((31, 4), dtype=a.dtypes.fp32),
                torch.empty((31, 4), dtype=a.dtypes.i32),
                dtype=a.dtypes.bf16, quant_type=a.QuantType.per_1x32,
                quant_dtype_a=a.dtypes.fp8, gate_mode="interleave")

    def test_public_override_excludes_only_the_mismatched_call_without_cache_clear(self):
        torch, a, fm = self.torch, self.aiter, self.fm
        self.g1 = "flydsl_mxmoe_g1_a8w4_64x128x256_fp8out"
        self.g2 = "opus_moe2_layout_afp8_wfp4_bf16_t64x256x256_sbm64_rbn3072"
        self.write_csv(block_m=64)
        hidden = torch.empty((31, 3072), dtype=a.dtypes.bf16)
        w1 = torch.empty((128, 1024, 1536), dtype=a.dtypes.fp4x2)
        w2 = torch.empty((128, 3072, 256), dtype=a.dtypes.fp4x2)
        w1.is_shuffled = w2.is_shuffled = True
        w1s = torch.empty((128, 1024, 96), dtype=a.dtypes.fp8_e8m0)
        w2s = torch.empty((128 * 3072, 16), dtype=a.dtypes.fp8_e8m0)
        ids = torch.empty((31, 4), dtype=a.dtypes.i32)
        weights = torch.empty((31, 4), dtype=a.dtypes.fp32)
        sorts, executions = [], []

        def sorting(*args, **kwargs):
            sorts.append(args[5])
            result = (torch.empty(32, dtype=a.dtypes.i32),
                torch.empty(32), torch.empty(1, dtype=a.dtypes.i32),
                torch.empty(2, dtype=a.dtypes.i32),
                torch.empty((31, 3072), dtype=a.dtypes.bf16),
                torch.empty(32, dtype=a.dtypes.i32),
                torch.empty(124, dtype=a.dtypes.i32))
            return result if kwargs.get("output_aux") else result[:5]

        def execute(*args, **kwargs):
            metadata = kwargs["_metadata_transform"](None)
            executions.append(metadata)
            return args[8]

        def public(block):
            return fm.fused_moe(hidden, w1, w2, weights, ids,
                activation=a.ActivationType.Silu, quant_type=a.QuantType.per_1x32,
                quant_dtype_a=a.dtypes.fp8, dtype=a.dtypes.bf16,
                gate_mode="interleave", w1_scale=w1s, w2_scale=w2s,
                block_size_M=block)

        with patch.object(fm, "moe_sorting", sorting), patch.object(
            fm, "fused_moe_2stages", execute
        ):
            public(64)
            with self.assertLogs("aiter", level="WARNING") as logs:
                public(32)
            public(64)
        self.assertTrue(any("block_size_M=32" in message for message in logs.output))
        self.assertEqual(sorts, [64, 32, 64])
        self.assertEqual(executions[0].stage1.keywords["kernelName1"], self.g1)
        self.assertNotEqual(executions[1].stage1.keywords.get("kernelName1"), self.g1)
        self.assertIs(executions[2], executions[0])
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
