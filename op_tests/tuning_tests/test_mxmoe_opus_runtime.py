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
        self.g2 = (
            "opus_moe2_layout_afp8_wfp4_atomic_t32x128x256_sbm32_occ1_cache_b3_ws2"
        )
        self.path = Path(self.directory.name) / "explicit.csv"
        self.write_csv()
        self.env = patch.dict(
            os.environ,
            {
                "AITER_CONFIG_FMOE": str(self.path),
                "AITER_ONLINE_TUNE": "0",
                "AITER_BYPASS_TUNE_CONFIG": "0",
                "AITER_MXMOE_FALLBACK": "0",
            },
        )
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
        self.row = {
            "gfx": "gfx950",
            "cu_num": 256,
            "token": 32,
            "model_dim": 3072,
            "inter_dim": 512,
            "expert": 128,
            "topk": 4,
            "act_type": str(a.ActivationType.Silu),
            "dtype": str(a.dtypes.bf16),
            "q_dtype_a": str(a.dtypes.fp8),
            "q_dtype_w": str(a.dtypes.fp4x2),
            "q_type": str(a.QuantType.per_1x32),
            "use_g1u1": 1,
            "doweight_stage1": 0,
            "kernelName1": self.g1,
            "kernelName2": self.g2,
            "block_m": 32,
            "ksplit": 0,
            "run_1stage": 0,
        }
        self.row.update(changes)
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.row)
            writer.writeheader()
            writer.writerow(self.row)

    def select(self, **kwargs):
        a = self.aiter
        options = {
            "gate_mode": "interleave",
            "opus_weights_shuffled": True,
            "input_dtype": a.dtypes.bf16,
        }
        options.update(kwargs)
        return self.fm.get_2stage_cfgs(
            32,
            3072,
            int(self.row["inter_dim"]),
            128,
            4,
            a.dtypes.bf16,
            a.dtypes.fp8,
            a.dtypes.fp4x2,
            a.QuantType.per_1x32,
            True,
            a.ActivationType.Silu,
            False,
            0,
            0,
            **options
        )

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

    def test_bm16_padded_csv_selects_regular_without_changing_native_metadata(self):
        self.g1 = "flydsl_mxmoe_g1_a8w4_16x128x256_f16in_nt_fp8out"
        self.g2 = "opus_moe2_layout_afp8_wfp4_atomic_t16x128x256_sbm16_cache_b3_ws2"
        self.write_csv(inter_dim=384, block_m=16)
        native_name = "flydsl_moe2_layout_afp8_wfp4_bf16_t16x128x128_atomic_sbm16"
        native = self.fm._make_mxfp4_metadata(
            self.g1, native_name, self.fm.GateMode.INTERLEAVE, 0
        )
        selected = self.select()
        self.assertEqual(selected.stage1.keywords["kernelName1"], self.g1)
        self.assertEqual(selected.stage2.keywords["launch"].kernel_id, 2008)
        self.assertFalse(selected.stage1.keywords["native_scale_layout"])
        self.assertTrue(native.stage1.keywords["native_scale_layout"])
        self.assertEqual(selected.block_m, 16)

    def test_bm32_sort_selects_bm64_gemm2_without_changing_its_compute_policy(self):
        self.g2 = "opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm32_rbn3584"
        self.write_csv()
        selected = self.select()
        launch = selected.stage2.keywords["launch"]
        original = self.fm._opus_a8w4.parse_stage2_config(
            {"kernelName2": "opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_rbn3584"},
            64,
        )
        self.assertEqual(selected.block_m, 32)
        self.assertEqual(launch.sort_block_m, 32)
        self.assertEqual(launch.instance.block_m, 64)
        self.assertEqual(launch.reduce_block_n, original.reduce_block_n)
        for field in (
            "block_n",
            "block_threads",
            "min_blocks_per_cu",
            "pair_slots",
            "steady_pair_slots",
            "cachectl_b",
            "cachectl_wscale",
            "out_mode",
        ):
            self.assertEqual(
                getattr(launch.instance, field), getattr(original.instance, field)
            )
        self.write_csv(block_m=64, kernelName1="flydsl_mxmoe_g1_a8w4_64x256x256_fp8out")
        self.fm.cfg_2stages = None
        self.fm.get_2stage_cfgs.cache_clear()
        rejected = self.select()
        self.assertNotEqual(rejected.stage2.keywords.get("kernelName"), self.g2)

    def test_stage1_regular_scales_expose_padded_stride_and_reader_capacity(self):
        torch, a = self.torch, self.aiter
        launcher = importlib.import_module("aiter.ops.flydsl.mxfp4_gemm1_kernels")
        for inter, native, cols, minimum_bytes in (
            (384, False, 16, 2560),
            (1408, False, 48, 7680),
            (384, True, 12, 4608),
            (1408, True, 44, 13824),
        ):
            with self.subTest(inter=inter, native=native):
                hidden = torch.empty((31, 512), dtype=a.dtypes.bf16)
                w1 = torch.empty((2, 2 * inter, 256), dtype=a.dtypes.fp4x2)
                w2 = torch.empty((2, 512, inter // 2), dtype=a.dtypes.fp4x2)
                with patch.object(launcher, "flydsl_mxfp4_gemm1") as launch:
                    payload, scales = self.fm._mxfp4_a4w4_stage1_fw(
                        hidden,
                        w1,
                        w2,
                        torch.empty(144, dtype=a.dtypes.i32),
                        torch.empty(9, dtype=a.dtypes.i32),
                        torch.tensor([144, 31], dtype=a.dtypes.i32),
                        None,
                        2,
                        kernelName1="flydsl_mxmoe_g1_a8w4_16x128x256_f16in_nt_fp8out",
                        native_scale_layout=native,
                    )
                self.assertEqual(payload.shape, (144, inter))
                self.assertEqual(scales.shape[1], cols)
                self.assertEqual(scales.stride(), (cols, 1))
                self.assertTrue(scales.is_contiguous())
                self.assertGreaterEqual(scales.numel(), minimum_bytes)
                self.assertEqual(scales.shape[0] % 32, 0)
                self.assertIs(
                    launch.call_args.kwargs["inter_sorted_shuffled_scale"], scales
                )
        self.assertFalse(torch.cuda.is_initialized())

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
                self.assertNotEqual(
                    metadata.stage1.keywords.get("kernelName1"), self.row["kernelName1"]
                )
                self.assertTrue(any(reason in message for message in logs.output))

    def test_public_explicit_fp4_intermediate_cannot_fall_back_to_fp8(self):
        torch, a, fm = self.torch, self.aiter, self.fm
        with patch.object(
            fm, "moe_sorting", side_effect=AssertionError("sort must not run")
        ), self.assertRaisesRegex(NotImplementedError, "intermediate precision"):
            fm.fused_moe(
                torch.empty((31, 3072), dtype=a.dtypes.bf16),
                torch.empty((128, 1024, 1536), dtype=a.dtypes.fp4x2),
                torch.empty((128, 3072, 256), dtype=a.dtypes.fp4x2),
                torch.empty((31, 4), dtype=a.dtypes.fp32),
                torch.empty((31, 4), dtype=a.dtypes.i32),
                dtype=a.dtypes.bf16,
                quant_type=a.QuantType.per_1x32,
                quant_dtype_a=a.dtypes.fp8,
                quant_dtype_a2=a.dtypes.fp4x2,
                gate_mode="interleave",
            )

    def test_missing_compiled_bm16_sort_requires_preparation_before_sort(self):
        torch, a, fm = self.torch, self.aiter, self.fm
        missing = importlib.import_module(
            "aiter.ops.moe_mxfp4_aux"
        )._MissingMxfp4MoeAuxInstances
        self.g1 = "flydsl_mxmoe_g1_a8w4_16x128x256_f16in_nt_fp8out"
        self.g2 = "opus_moe2_layout_afp8_wfp4_atomic_t16x128x256_sbm16_cache_b3_ws2"
        self.write_csv(inter_dim=384, block_m=16)
        w1 = torch.empty((128, 768, 1536), dtype=a.dtypes.fp4x2)
        w2 = torch.empty((128, 3072, 192), dtype=a.dtypes.fp4x2)
        w1.is_shuffled = w2.is_shuffled = True
        with patch.object(
            fm, "_mxfp4_aux_instance_supported", return_value=False
        ), patch.object(
            fm, "moe_sorting", side_effect=AssertionError("sort must not run")
        ), self.assertRaises(
            missing
        ) as context:
            fm.fused_moe(
                torch.empty((31, 3072), dtype=a.dtypes.bf16),
                w1,
                w2,
                torch.empty((31, 4), dtype=a.dtypes.fp32),
                torch.empty((31, 4), dtype=a.dtypes.i32),
                dtype=a.dtypes.bf16,
                quant_type=a.QuantType.per_1x32,
                quant_dtype_a=a.dtypes.fp8,
                gate_mode="interleave",
                w1_scale=torch.empty((128 * 768, 96), dtype=a.dtypes.fp8_e8m0),
                w2_scale=torch.empty((128 * 3072, 16), dtype=a.dtypes.fp8_e8m0),
            )
        self.assertEqual(context.exception.missing_instances, ((128, 4, 3072, True),))
        self.assertIn("aux_sortzi_NE128_TOPK4_MB16_H3072", str(context.exception))
        self.assertIn("AITER_REBUILD=1", str(context.exception))
        self.assertFalse(torch.cuda.is_initialized())

    def test_unsupported_call_semantics_fall_back(self):
        for kwargs, reason in (
            ({"has_stage2_bias": True}, "stage2 bias"),
            ({"is_ep": True}, "EP"),
            ({"opus_weights_shuffled": False}, "is_shuffled"),
            ({"has_num_local_tokens": True}, "num_local_tokens"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertLogs("aiter", level="WARNING") as logs:
                    metadata = self.select(**kwargs)
                self.assertIsNot(
                    metadata.stage2.func, self.fm._opus_a8w4.opus_a8w4_stage2_wrapper
                )
                self.assertTrue(any(reason in message for message in logs.output))

    def test_fp8_inline_row_falls_back_without_reinterpreting_input(self):
        self.g1 = "flydsl_mxmoe_g1_a8w4_16x128x256_f16in_nt_fp8out"
        self.g2 = "opus_moe2_layout_afp8_wfp4_atomic_t16x128x256_sbm16_cache_b3_ws2"
        self.write_csv(inter_dim=384, block_m=16)
        with self.assertLogs("aiter", level="WARNING") as logs:
            selected = self.select(
                input_dtype=self.aiter.dtypes.fp8, _has_input_activation_scales=True
            )
        self.assertNotEqual(selected.stage1.keywords.get("kernelName1"), self.g1)
        self.assertTrue(
            any("cannot accept FP8 input" in message for message in logs.output)
        )

    def test_fp8_original_scale_metadata_is_validated_before_sort(self):
        torch, a, fm = self.torch, self.aiter, self.fm
        hidden = torch.empty((31, 3072), dtype=a.dtypes.fp8)
        w1 = torch.empty((128, 1024, 1536), dtype=a.dtypes.fp4x2)
        w2 = torch.empty((128, 3072, 256), dtype=a.dtypes.fp4x2)
        w1.is_shuffled = w2.is_shuffled = True
        w1s = torch.empty((128 * 1024, 96), dtype=a.dtypes.fp8_e8m0)
        w2s = torch.empty((128 * 3072, 16), dtype=a.dtypes.fp8_e8m0)
        for scale in (
            torch.empty((32, 96), dtype=a.dtypes.fp8_e8m0),
            torch.empty((31, 96), dtype=a.dtypes.fp32),
        ):
            with self.subTest(
                shape=scale.shape, dtype=scale.dtype
            ), self.assertRaisesRegex(ValueError, "per-token group32 E8M0"):
                fm.fused_moe(
                    hidden,
                    w1,
                    w2,
                    torch.empty((31, 4), dtype=a.dtypes.fp32),
                    torch.empty((31, 4), dtype=a.dtypes.i32),
                    dtype=a.dtypes.bf16,
                    quant_type=a.QuantType.per_1x32,
                    quant_dtype_a=a.dtypes.fp8,
                    gate_mode="interleave",
                    a1_scale=scale,
                    w1_scale=w1s,
                    w2_scale=w2s,
                )

    def test_explicit_fp8_without_original_scales_is_invalid_before_fallback(self):
        a, torch = self.aiter, self.torch
        hidden = torch.empty((31, 3072), dtype=a.dtypes.fp8)
        w1 = torch.empty((128, 1024, 1536), dtype=a.dtypes.fp4x2)
        w2 = torch.empty((128, 3072, 256), dtype=a.dtypes.fp4x2)
        with self.assertRaisesRegex(ValueError, "original per-token E8M0"):
            self.fm.fused_moe(
                hidden,
                w1,
                w2,
                torch.empty((31, 4), dtype=a.dtypes.fp32),
                torch.empty((31, 4), dtype=a.dtypes.i32),
                dtype=a.dtypes.bf16,
                quant_type=a.QuantType.per_1x32,
                quant_dtype_a=a.dtypes.fp8,
                gate_mode="interleave",
            )

    def test_public_prequantized_fp8_preserves_payload_and_sorts_original_scales(self):
        torch, a, fm = self.torch, self.aiter, self.fm
        hidden = torch.empty((31, 3072), dtype=a.dtypes.fp8)
        scales = torch.empty((31, 96), dtype=a.dtypes.fp8_e8m0)
        w1 = torch.empty((128, 1024, 1536), dtype=a.dtypes.fp4x2)
        w2 = torch.empty((128, 3072, 256), dtype=a.dtypes.fp4x2)
        w1.is_shuffled = w2.is_shuffled = True
        w1s = torch.empty((128 * 1024, 96), dtype=a.dtypes.fp8_e8m0)
        w2s = torch.empty((128 * 3072, 16), dtype=a.dtypes.fp8_e8m0)
        ids = torch.empty((31, 4), dtype=a.dtypes.i32)
        weights = torch.empty((31, 4), dtype=a.dtypes.fp32)
        output = torch.empty((31, 3072), dtype=a.dtypes.bf16)
        sorted_scales = torch.empty((32, 96), dtype=a.dtypes.fp8_e8m0)
        payload = torch.empty((32, 512), dtype=a.dtypes.fp8)
        intermediate_scales = torch.empty((32, 16), dtype=a.dtypes.fp8_e8m0)
        sorting = (
            torch.empty(32, dtype=a.dtypes.i32),
            torch.empty(32),
            torch.empty(1, dtype=a.dtypes.i32),
            torch.tensor([32, 31], dtype=a.dtypes.i32),
            output,
            torch.empty(32, dtype=a.dtypes.i32),
            torch.empty(124, dtype=a.dtypes.i32),
        )
        with patch.object(fm, "moe_sorting", return_value=sorting), patch.object(
            fm, "mxfp4_moe_sort_fwd", return_value=sorted_scales
        ) as scale_sort, patch.object(
            fm, "_mxfp4_a4w4_stage1_fw", return_value=(payload, intermediate_scales)
        ) as stage1, patch.object(
            fm._opus_a8w4, "_opus_moe_stage2_a8w4_decode_fwd_raw", return_value=output
        ):
            result = fm.fused_moe(
                hidden,
                w1,
                w2,
                weights,
                ids,
                dtype=a.dtypes.bf16,
                quant_type=a.QuantType.per_1x32,
                quant_dtype_a=a.dtypes.fp8,
                gate_mode="interleave",
                a1_scale=scales,
                w1_scale=w1s,
                w2_scale=w2s,
                output=output,
            )
        self.assertIs(result, output)
        self.assertIs(scale_sort.call_args.args[0], scales)
        self.assertIs(stage1.call_args.args[0], hidden)
        self.assertIs(stage1.call_args.kwargs["a1_scale"], sorted_scales)
        self.assertEqual(stage1.call_args.kwargs["kernelName1"], self.g1)
        self.assertFalse(torch.cuda.is_initialized())

    def test_public_fp8_route_output_switches_workspace_on_cached_pair(self):
        torch, a, fm = self.torch, self.aiter, self.fm
        self.g1 = "flydsl_mxmoe_g1_a8w4_128x128x256_fp8out"
        self.g2 = "opus_moe2_layout_afp8_wfp4_fp8_t64x256x256_sbm128_rbn3072"
        self.write_csv(block_m=128, inter_dim=384)
        hidden = torch.empty((31, 3072), dtype=a.dtypes.bf16)
        w1 = torch.empty((128, 768, 1536), dtype=a.dtypes.fp4x2)
        w2 = torch.empty((128, 3072, 192), dtype=a.dtypes.fp4x2)
        w1.is_shuffled = w2.is_shuffled = True
        w1s = torch.empty((128 * 768, 96), dtype=a.dtypes.fp8_e8m0)
        w2s = torch.empty((128 * 3072, 16), dtype=a.dtypes.fp8_e8m0)
        ids = torch.empty((31, 4), dtype=a.dtypes.i32)
        weights = torch.empty((31, 4), dtype=a.dtypes.fp32)
        output = torch.empty((31, 3072), dtype=a.dtypes.bf16)
        payload = torch.empty((128, 384), dtype=a.dtypes.fp8)
        scale = torch.empty((128, 16), dtype=a.dtypes.fp8_e8m0)
        sorting = (
            torch.empty(128, dtype=a.dtypes.i32),
            torch.empty(128),
            torch.empty(1, dtype=a.dtypes.i32),
            torch.tensor([128, 31], dtype=a.dtypes.i32),
            torch.empty((0, 0), dtype=a.dtypes.bf16),
            torch.empty(128, dtype=a.dtypes.i32),
            torch.empty(124, dtype=a.dtypes.i32),
        )
        original = fm.get_2stage_cfgs
        metadata, workspaces, sorts = [], [], []

        def observe_cfg(*args, **kwargs):
            selected = original(*args, **kwargs)
            metadata.append(selected)
            return selected

        def observe_sort(*args, **kwargs):
            sorts.append(kwargs["accumulate"])
            return sorting

        def decode(*args):
            workspaces.append((args[8].dtype, args[12]))
            return args[8]

        with patch.object(fm, "get_2stage_cfgs", observe_cfg), patch.object(
            fm, "moe_sorting", observe_sort
        ), patch.object(
            fm,
            "fused_dynamic_mxfp8_quant_moe_sort",
            return_value=(
                hidden.to(a.dtypes.fp8),
                torch.empty((128, 96), dtype=a.dtypes.fp8_e8m0),
            ),
        ), patch.object(
            fm, "_mxfp4_a4w4_stage1_fw", return_value=(payload, scale)
        ), patch.object(
            fm._opus_a8w4, "_opus_moe_stage2_a8w4_decode_fwd_raw", side_effect=decode
        ), patch.object(
            fm._opus_a8w4,
            "_opus_moe_stage2_reduce_token_slot_route_output_fwd_raw",
            return_value=output,
        ):
            for enabled in (1, 0, 1):
                with patch.dict(os.environ, {"AITER_OPUS_STAGE2_FP8": str(enabled)}):
                    result = fm.fused_moe(
                        hidden,
                        w1,
                        w2,
                        weights,
                        ids,
                        dtype=a.dtypes.bf16,
                        quant_type=a.QuantType.per_1x32,
                        quant_dtype_a=a.dtypes.fp8,
                        quant_dtype_a2=None,
                        gate_mode="interleave",
                        w1_scale=w1s,
                        w2_scale=w2s,
                        output=output,
                    )
                    self.assertIs(result, output)
        self.assertEqual(
            workspaces,
            [(torch.uint8, 2010), (a.dtypes.bf16, 2010), (torch.uint8, 2010)],
        )
        self.assertEqual(sorts, [False, False, False])
        self.assertTrue(all(selected is metadata[0] for selected in metadata[::2]))
        self.assertEqual(metadata[0].stage2.keywords["launch"].instance.block_m, 64)
        self.assertEqual(metadata[0].block_m, 128)
        self.assertFalse(torch.cuda.is_initialized())

    def test_public_override_excludes_only_the_mismatched_call_without_cache_clear(
        self,
    ):
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
            result = (
                torch.empty(32, dtype=a.dtypes.i32),
                torch.empty(32),
                torch.empty(1, dtype=a.dtypes.i32),
                torch.empty(2, dtype=a.dtypes.i32),
                torch.empty((31, 3072), dtype=a.dtypes.bf16),
                torch.empty(32, dtype=a.dtypes.i32),
                torch.empty(124, dtype=a.dtypes.i32),
            )
            return result if kwargs.get("output_aux") else result[:5]

        def execute(*args, **kwargs):
            metadata = kwargs["_metadata_transform"](None)
            executions.append(metadata)
            return args[8]

        def public(block):
            return fm.fused_moe(
                hidden,
                w1,
                w2,
                weights,
                ids,
                activation=a.ActivationType.Silu,
                quant_type=a.QuantType.per_1x32,
                quant_dtype_a=a.dtypes.fp8,
                dtype=a.dtypes.bf16,
                gate_mode="interleave",
                w1_scale=w1s,
                w2_scale=w2s,
                block_size_M=block,
            )

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
