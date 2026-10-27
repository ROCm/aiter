# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pandas as pd
import torch

from op_tests.tuning_tests.mxfp4_cpu_test_utils import cpu_tuner_environment

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


def shape_row(**changes: Any) -> dict[str, Any]:
    row = {
        "gfx": "gfx950",
        "cu_num": 256,
        "token": 1,
        "model_dim": 1024,
        "inter_dim": 384,
        "expert": 4,
        "topk": 2,
        "act_type": "ActivationType.Silu",
        "dtype": "torch.bfloat16",
        "q_dtype_a": "torch.float4_e2m1fn_x2",
        "q_dtype_w": "torch.float4_e2m1fn_x2",
        "q_type": "QuantType.per_1x32",
        "use_g1u1": True,
        "doweight_stage1": False,
    }
    row.update(changes)
    return row


class TestMxfp4FlydslInputsAndCandidates(unittest.TestCase):
    def setUp(self) -> None:
        environment = cpu_tuner_environment()
        environment.__enter__()
        self.addCleanup(environment.__exit__, None, None, None)
        self.module = importlib.import_module(
            "csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune"
        )
        torch.set_default_device("cpu")
        self.tuner = self.module.Mxfp4FlydslTuner.__new__(self.module.Mxfp4FlydslTuner)
        self.tuner.keys = KEYS
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def read_rows(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        path = Path(self.directory.name) / "input.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return self.tuner.get_untuned_gemm_list(str(path))

    def test_tune_rejects_rows_that_are_not_a4w4_before_any_shape_work(
        self,
    ) -> None:
        args = SimpleNamespace(mp=1)
        for column in ("q_dtype_a", "q_dtype_w"):
            rows = self.read_rows(
                [shape_row(), shape_row(token=2, **{column: "torch.float8_e4m3fn"})]
            )
            rejected = r"row 3 \(.*torch.float8_e4m3fn.*\): --mxfp4-flydsl tunes A4W4 rows only"
            with self.subTest(column=column), mock.patch.object(
                self.module.Mxfp4FlydslTuner,
                "_tune_one_shape",
                side_effect=AssertionError("a shape was tuned before the check"),
            ), self.assertRaisesRegex(ValueError, rejected):
                self.tuner.tune(rows, pd.DataFrame(), args)
        rows = self.read_rows([shape_row(), shape_row(token=2)])
        with mock.patch.object(
            self.module.Mxfp4FlydslTuner,
            "_tune_one_shape",
            side_effect=lambda row, _args: row["token"],
        ):
            self.assertEqual(self.tuner.tune(rows, pd.DataFrame(), args), [1, 2])

    def test_g1_names_carry_activation_and_output_precision(self) -> None:
        kname = self.module.Mxfp4FlydslTuner._g1_kname
        for kwargs, expected in (
            ({}, "flydsl_mxmoe_g1_a4w4_16x128x256_f16in_nt"),
            (
                {"a_dtype": "fp8", "out_dtype": "fp8"},
                "flydsl_mxmoe_g1_a8w4_16x128x256_f16in_nt_fp8out",
            ),
            (
                {"a_dtype": "fp8", "out_dtype": "fp4"},
                "flydsl_mxmoe_g1_a8w4_16x128x256_f16in_nt",
            ),
        ):
            with self.subTest(**kwargs):
                name = kname(16, True, True, bn=128, **kwargs)
                self.assertEqual(name, expected)
                parsed = self.module._parse_mxfp4_g1_kname(name)
                self.assertEqual(
                    (parsed["a_dtype"], parsed["out_dtype"]),
                    (kwargs.get("a_dtype", "fp4"), kwargs.get("out_dtype", "fp4")),
                )
        name = kname(
            32, False, False, act="swiglu", k_wave=2, a_dtype="fp8", out_dtype="fp8"
        )
        self.assertEqual(name, "flydsl_mxmoe_g1_a8w4_32x256x256_fp8out_swiglu_kw2")
        parsed = self.module._parse_mxfp4_g1_kname(name)
        self.assertEqual((parsed["act"], parsed["k_wave"]), ("swiglu", 2))

    def test_full_search_keeps_only_the_coupled_shape_support_domain(self) -> None:
        unsupported = (
            {"model_dim": 256},  # GEMM1's pipelined loop needs two K tiles.
            {"model_dim": 768},  # More than one tile, but non-divisible k_wave.
            {"model_dim": 8448},  # B-scale addressing ends after 32 K tiles.
            {"inter_dim": 64},  # GEMM1 alone can emit this; GEMM2 cannot.
        )
        for changes in unsupported:
            row = shape_row(**changes)
            candidates = self.tuner._candidate_rows(row, full_search=True)
            if changes == {"model_dim": 768}:
                self.assertTrue(candidates)
                self.assertEqual(
                    {
                        self.module._parse_mxfp4_g1_kname(c["kernelName1"])["k_wave"]
                        for c in candidates
                    },
                    {1},
                )
            else:
                self.assertFalse(candidates, changes)

    def test_bm16_aot_jobs_use_the_same_native_scale_contract_for_both_precisions(
        self,
    ) -> None:
        from aiter.aot.flydsl.mxfp4_moe import parse_csv

        rows = []
        for precision, q_dtype_a in (
            ("fp4", "torch.float4_e2m1fn_x2"),
            ("fp8", "torch.float8_e4m3fn"),
        ):
            row = shape_row(q_dtype_a=q_dtype_a)
            row.update(
                kernelName1=f"flydsl_mxmoe_g1_a{'4' if precision == 'fp4' else '8'}w4_"
                f"16x128x256_f16in_nt{'_fp8out' if precision == 'fp8' else ''}",
                kernelName2=f"flydsl_moe2_layout_a{precision}_wfp4_bf16_"
                "t16x128x128_atomic_sbm16",
            )
            rows.append(row)
        path = Path(self.directory.name) / "tuned.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        jobs = parse_csv(str(path))
        stage1 = [job for job in jobs if job["stage"] == 1]
        self.assertEqual({job["out_dtype"] for job in stage1}, {"fp4", "fp8"})
        self.assertTrue(all(job["native_scale_layout"] for job in stage1))
        self.assertEqual({job["D_INTER"] for job in stage1}, {384})
        self.assertEqual({job["interleave"] for job in stage1}, {False, True})
        for precision in ("fp4", "fp8"):
            matching = [job for job in stage1 if job["a_dtype"] == precision]
            self.assertEqual(
                {(job["interleave"], job["enable_bias"]) for job in matching},
                {(False, False), (False, True), (True, False), (True, True)},
            )

    def test_mxmoe_aot_shared_expert_has_no_bias_variant(self) -> None:
        from aiter.aot.flydsl.mxfp4_moe import parse_csv

        row = shape_row(shared_expert_id=0)
        row.update(
            kernelName1="flydsl_mxmoe_g1_a4w4_16x128x256_f16in_nt",
            kernelName2="flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16",
        )
        path = Path(self.directory.name) / "shared.csv"
        pd.DataFrame([row, row]).to_csv(path, index=False)
        jobs = [job for job in parse_csv(str(path)) if job["stage"] == 1]
        self.assertEqual(len(jobs), 2)
        self.assertTrue(all(not job["enable_bias"] for job in jobs))

    def test_mxmoe_bias_names_raise_in_parser_and_aot(self) -> None:
        from aiter.aot.flydsl.mxfp4_moe import parse_csv

        for a_bits, suffix in ((4, ""), (8, "_fp8out")):
            name = f"flydsl_mxmoe_g1_a{a_bits}w4_32x128x256{suffix}"
            self.assertNotIn("enable_bias", self.module._parse_mxfp4_g1_kname(name))
            for bias_suffix in ("_bias", "_BIAS"):
                with self.subTest(name=name + bias_suffix), self.assertRaisesRegex(
                    ValueError, "unknown token"
                ):
                    self.module._parse_mxfp4_g1_kname(name + bias_suffix)
            row = shape_row(kernelName1=name + "_bias", kernelName2="")
            path = Path(self.directory.name) / "legacy.csv"
            pd.DataFrame([row]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "unknown token 'bias'"):
                parse_csv(str(path))

    def test_mxmoe_wrapper_passes_runtime_bias_without_name_flag(self) -> None:
        fm = importlib.import_module("aiter.fused_moe")
        hidden = torch.empty((2, 512), dtype=torch.float8_e4m3fn)
        w1 = torch.empty((4, 768, 256), dtype=torch.uint8)
        w2 = torch.empty((4, 512, 192), dtype=torch.uint8)
        bias = torch.randn((4, 768), dtype=torch.float32)
        result = (object(), object())
        with mock.patch.object(
            fm, "_mxfp4_a4w4_stage1", return_value=result
        ) as stage1, mock.patch.object(
            fm, "_empty_bf16", return_value=torch.empty(0, dtype=torch.bfloat16)
        ):
            for runtime_bias in (bias, None, bias):
                actual = fm._mxfp4_a4w4_stage1_fw(
                    hidden,
                    w1,
                    w2,
                    torch.empty(32, dtype=torch.int32),
                    None,
                    None,
                    None,
                    2,
                    kernelName1="flydsl_mxmoe_g1_a8w4_32x128x256_fp8out",
                    m_indices=torch.empty(0, dtype=torch.int32),
                    bias1=runtime_bias,
                )
                self.assertEqual(actual, result)
                self.assertIs(stage1.call_args.kwargs["bias1"], runtime_bias)

    def test_tuned_mxmoe_keeps_pair_with_runtime_bias_and_rejects_legacy_name(
        self,
    ) -> None:
        fm = importlib.import_module("aiter.fused_moe")
        from aiter import ActivationType, QuantType, dtypes

        key = (
            "gfx950",
            256,
            32,
            1024,
            384,
            4,
            2,
            str(ActivationType.Silu),
            str(dtypes.bf16),
            str(dtypes.fp8),
            str(dtypes.fp4x2),
            str(QuantType.per_1x32),
            True,
            False,
        )
        g1 = "flydsl_mxmoe_g1_a8w4_32x128x256_fp8out"
        g2 = "flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_reduce_sbm32"
        cfg = {"kernelName1": g1, "kernelName2": g2, "block_m": 32, "ksplit": 0}
        with mock.patch.object(
            fm, "get_gfx_runtime", return_value="gfx950"
        ), mock.patch.object(fm, "get_cu_num", return_value=256), mock.patch.object(
            fm, "cfg_2stages", ({key: cfg}, {})
        ), mock.patch.object(
            fm.aiter, "is_mxfp4_moe_shape_supported", return_value=True
        ), mock.patch.dict(
            os.environ, {"AITER_BYPASS_TUNE_CONFIG": "0"}
        ):
            self.addCleanup(fm.get_2stage_cfgs.cache_clear)
            for has_bias in (True, False, True):
                fm.get_2stage_cfgs.cache_clear()
                meta = fm.get_2stage_cfgs(
                    32,
                    1024,
                    384,
                    4,
                    2,
                    dtypes.bf16,
                    dtypes.fp8,
                    dtypes.fp4x2,
                    QuantType.per_1x32,
                    True,
                    ActivationType.Silu,
                    False,
                    0,
                    0,
                    gate_mode="interleave",
                    has_stage1_bias=has_bias,
                )
                self.assertEqual(meta.stage1.keywords["kernelName1"], g1)
                self.assertEqual(meta.stage2.keywords["kernelName2"], g2)
                self.assertEqual(meta.has_bias, has_bias)
            cfg["kernelName1"] = g1 + "_bias"
            # Even incompatible OPUS weight layout must not mask bad G1 names.
            for kernel2, shuffled in ((g2, True), ("opus_test", False)):
                cfg["kernelName2"] = kernel2
                fm.get_2stage_cfgs.cache_clear()
                with self.assertRaisesRegex(ValueError, "unknown token 'bias'"):
                    fm.get_2stage_cfgs(
                        32,
                        1024,
                        384,
                        4,
                        2,
                        dtypes.bf16,
                        dtypes.fp8,
                        dtypes.fp4x2,
                        QuantType.per_1x32,
                        True,
                        ActivationType.Silu,
                        False,
                        0,
                        0,
                        gate_mode="interleave",
                        has_stage1_bias=True,
                        opus_weights_shuffled=shuffled,
                    )
        fm.get_2stage_cfgs.cache_clear()

    def test_mxmoe_fallback_still_skips_calls_with_bias(self) -> None:
        fm = importlib.import_module("aiter.fused_moe")
        from aiter import ActivationType, QuantType, dtypes

        with mock.patch.object(
            fm, "get_gfx_runtime", return_value="gfx950"
        ), mock.patch.object(fm, "get_cu_num", return_value=256), mock.patch.object(
            fm, "cfg_2stages", ({}, {})
        ), mock.patch.object(
            fm.aiter, "is_mxfp4_moe_shape_supported", return_value=True
        ), mock.patch.dict(
            os.environ, {"AITER_ONLINE_TUNE": "0", "AITER_MXMOE_FALLBACK": "1"}
        ):
            self.addCleanup(fm.get_2stage_cfgs.cache_clear)
            for has_bias in (False, True):
                fm.get_2stage_cfgs.cache_clear()
                meta = fm.get_2stage_cfgs(
                    32,
                    1024,
                    384,
                    4,
                    2,
                    dtypes.bf16,
                    dtypes.fp4x2,
                    dtypes.fp4x2,
                    QuantType.per_1x32,
                    True,
                    ActivationType.Situv2,
                    False,
                    0,
                    0,
                    gate_mode="separated",
                    has_stage1_bias=has_bias,
                )
                self.assertIs(
                    meta.stage1.func,
                    fm._flydsl_stage1_wrapper if has_bias else fm._mxfp4_a4w4_stage1_fw,
                )
        fm.get_2stage_cfgs.cache_clear()


if __name__ == "__main__":
    unittest.main()
