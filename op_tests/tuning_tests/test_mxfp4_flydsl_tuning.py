# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import importlib
import os
import sys
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

    def test_parent_prepares_only_generated_sort_rows_before_shape_workers(
        self,
    ) -> None:
        from aiter.ops import moe_mxfp4_aux
        from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import FmoeTuner

        rows = pd.DataFrame(
            [
                shape_row(
                    token=1,
                    expert=128,
                    model_dim=3072,
                    inter_dim=512,
                    topk=4,
                    q_dtype_a="torch.float8_e4m3fn",
                ),
                shape_row(
                    token=1,
                    expert=128,
                    model_dim=3072,
                    inter_dim=1536,
                    topk=4,
                    q_dtype_a="torch.float8_e4m3fn",
                ),
                shape_row(
                    token=64,
                    expert=128,
                    model_dim=3072,
                    inter_dim=512,
                    topk=4,
                    q_dtype_a="torch.float8_e4m3fn",
                ),
            ]
        )
        args = SimpleNamespace(
            run_config=False, compare=False, mxfp4_search_mode="full"
        )
        observed = []

        def prepare(_self: Any, _args: argparse.Namespace) -> None:
            _self.untunedf = rows

        with mock.patch.object(FmoeTuner, "pre_process", prepare), mock.patch.object(
            moe_mxfp4_aux,
            "prepare_mxfp4_moe_aux",
            lambda shapes: observed.extend(shapes),
        ):
            self.tuner.pre_process(args)
        self.assertEqual(observed, [(128, 3072, 512, 4), (128, 3072, 1536, 4)])

    def test_run_config_leaves_auxiliary_preparation_to_production(self) -> None:
        from aiter.ops import moe_mxfp4_aux
        from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import FmoeTuner

        with mock.patch.object(
            FmoeTuner, "pre_process", lambda *_: None
        ), mock.patch.object(
            moe_mxfp4_aux,
            "prepare_mxfp4_moe_aux",
            side_effect=AssertionError("run_config must not prepare tune workers"),
        ):
            self.tuner.pre_process(SimpleNamespace(run_config=True, compare=False))

    def test_mixed_rows_use_their_own_precision_and_default_search(self) -> None:
        rows = self.read_rows([shape_row(), shape_row(q_dtype_a="torch.float8_e4m3fn")])
        a4, a8 = [row.to_dict() for _, row in rows.iterrows()]
        a4_default = self.tuner._candidate_rows(a4)
        a8_default = self.tuner._candidate_rows(a8)
        self.assertTrue(a4_default)
        self.assertTrue(a8_default)
        self.assertEqual({c["block_m"] for c in a4_default}, {16})
        self.assertEqual({c["block_m"] for c in a8_default}, {16, 32, 64, 128})
        self.assertLess(
            len(a4_default), len(self.tuner._candidate_rows(a4, full_search=True))
        )
        self.assertEqual(a8_default, self.tuner._candidate_rows(a8, full_search=True))
        for candidate in a8_default:
            g1 = self.module._parse_mxfp4_g1_kname(candidate["kernelName1"])
            g2 = self.module.parse_flydsl_v2_gemm2_kernel(candidate["kernelName2"])
            self.assertEqual((g1["a_dtype"], g1["out_dtype"]), ("fp8", "fp8"))
            self.assertEqual(
                (g2["a_dtype"], g2["b_dtype"], g2["out_dtype"]),
                ("fp8", "fp4", "bf16"),
            )
            self.assertEqual(g1["BM"], g2["tile_m"])
            self.assertEqual(g1["BM"], g2["sort_block_m"])
            self.assertEqual(g1["inline_quant"], g1["BM"] == 16)
            self.assertNotEqual(g1["BN"], 64)
            self.assertEqual(g1["num_waves"], 4)
            self.assertFalse(g2["persist"])

    def test_invalid_csv_reports_the_original_row_and_reason(self) -> None:
        invalid = (
            ({"dtype": "torch.float16"}, "dtype"),
            ({"q_dtype_a": "torch.bfloat16"}, "activation dtype"),
            ({"q_dtype_a": "torch.float8_e4m3fnuz"}, "activation dtype"),
            ({"q_dtype_w": "torch.float8_e4m3fn"}, "weight dtype"),
            ({"q_type": "QuantType.per_Token"}, "quantization"),
            ({"act_type": "ActivationType.Gelu"}, "activation"),
            ({"use_g1u1": False}, "use_g1u1"),
            ({"doweight_stage1": True}, "doweight_stage1"),
            ({"token": 0}, "token"),
            ({"model_dim": 1024.5}, "model_dim"),
            ({"inter_dim": -128}, "inter_dim"),
            ({"expert": 0}, "expert"),
            ({"topk": 5}, "topk"),
            ({"gfx": "gfx942"}, "gfx"),
        )
        for changes, reason in invalid:
            # A repeated first shape must not erase the source CSV line.
            with self.subTest(changes=changes), self.assertRaisesRegex(
                ValueError, rf"row 4.*{reason}"
            ):
                self.read_rows([shape_row(), shape_row(), shape_row(**changes)])

    def test_missing_required_column_reports_the_input_file(self) -> None:
        row = shape_row()
        del row["q_dtype_a"]
        with self.assertRaisesRegex(ValueError, r"input.csv.*q_dtype_a"):
            self.read_rows([row])

    def test_retune_filters_architecture_before_validating_target_rows(self) -> None:
        foreign = shape_row(gfx="gfx942", cu_num=304, act_type="ActivationType.Gelu")
        other_cu = shape_row(cu_num=128, dtype="torch.float16")
        path = Path(self.directory.name) / "shared.csv"
        original = pd.DataFrame([foreign, shape_row(), other_cu])
        original.to_csv(path, index=False)
        args = SimpleNamespace(
            untune_file=str(path), tune_file=str(path), verbose=False
        )
        with mock.patch.object(
            self.tuner, "get_gfx", return_value="gfx950"
        ), mock.patch.object(self.tuner, "get_cu_num", return_value=256):
            self.tuner.get_retune_gemm_list(args)
        self.assertEqual(self.tuner.untunedf["gfx"].tolist(), ["gfx950"])
        self.assertEqual(self.tuner.untunedf["cu_num"].tolist(), [256])
        pd.testing.assert_frame_equal(
            self.tuner.tunedf.reset_index(drop=True),
            original.iloc[[0, 2]].reset_index(drop=True),
        )

    def test_retune_validates_original_csv_line_for_target_rows(self) -> None:
        for invalid, reason in (
            ({"act_type": "ActivationType.Gelu"}, "activation"),
            ({"q_dtype_a": "torch.float16"}, "activation dtype"),
        ):
            path = Path(self.directory.name) / "shared.csv"
            pd.DataFrame(
                [
                    shape_row(gfx="gfx942", cu_num=304),
                    shape_row(),
                    shape_row(),
                    shape_row(**invalid),
                ]
            ).to_csv(path, index=False)
            args = SimpleNamespace(
                untune_file=str(path), tune_file=str(path), verbose=False
            )
            with self.subTest(invalid=invalid), mock.patch.object(
                self.tuner, "get_gfx", return_value="gfx950"
            ), mock.patch.object(
                self.tuner, "get_cu_num", return_value=256
            ), self.assertRaisesRegex(
                ValueError, rf"shared.csv: row 5.*{reason}"
            ):
                self.tuner.get_retune_gemm_list(args)

    def test_retune_separate_input_normalizes_keys_before_removing_old_results(
        self,
    ) -> None:
        path = Path(self.directory.name) / "input.csv"
        original = pd.DataFrame([shape_row(token=1), shape_row(token=2)])
        pd.DataFrame(
            [
                shape_row(gfx="gfx942", cu_num=304),
                shape_row(act_type="silu", q_dtype_a="fp4x2"),
            ]
        ).to_csv(path, index=False)
        args = SimpleNamespace(
            untune_file=str(path),
            tune_file=str(Path(self.directory.name) / "output.csv"),
            verbose=False,
        )
        with mock.patch.object(
            self.tuner, "get_gfx", return_value="gfx950"
        ), mock.patch.object(
            self.tuner, "get_cu_num", return_value=256
        ), mock.patch.object(
            self.tuner, "get_tuned_gemm_list", return_value=original
        ):
            self.tuner.get_retune_gemm_list(args)
        self.assertEqual(self.tuner.tunedf["token"].tolist(), [2])
        self.assertEqual(
            self.tuner.untunedf["q_dtype_a"].tolist(), [shape_row()["q_dtype_a"]]
        )

    def test_explicit_search_mode_is_used_in_serial_and_spawned_workers(self) -> None:
        a8 = self.read_rows([shape_row(token=64, q_dtype_a="torch.float8_e4m3fn")])
        for mode in (None, "prune", "full"):
            args = SimpleNamespace(
                mxfp4_search_mode=mode,
                timeout=0,
                mp=1,
                errRatio=0.1,
                profile_file="",
            )

            # Substitute only the GPU evaluation boundary. Real candidate
            # generation and the shape worker still run in both paths.
            def evaluate(
                _self: Any,
                row: dict[str, Any],
                candidate: dict[str, Any],
                _args: argparse.Namespace,
            ) -> float:
                candidate["us"] = 128.0 / candidate["block_m"]
                candidate["error"] = 0.0
                return candidate["us"]

            with self.subTest(mode=mode), mock.patch.object(
                self.module.Mxfp4FlydslTuner, "_run_candidate", evaluate
            ), mock.patch.object(torch.cuda, "device_count", return_value=2):
                serial = self.tuner.tune(a8, pd.DataFrame(), args)[0]
                args.mp = 2
                # Two distinct workloads exercise the shape worker branch.
                mp_rows = pd.concat([a8, a8.assign(token=65)], ignore_index=True)

                def isolated(
                    payloads: list[tuple], _mp_num: int, _ctx: Any
                ) -> list[dict[str, Any] | None]:
                    with mock.patch.object(torch.cuda, "set_device"):
                        return [
                            self.module._mxfp4_tune_shape_worker((*payload[:3], 0))
                            for payload in payloads
                        ]

                with mock.patch.object(self.module, "_run_shapes_isolated", isolated):
                    spawned = self.tuner.tune(mp_rows, pd.DataFrame(), args)[0]
                self.assertEqual(serial["kernelName1"], spawned["kernelName1"])
                self.assertEqual(serial["kernelName2"], spawned["kernelName2"])
                expected_bm = 64 if mode == "prune" else 128
                self.assertEqual(serial["block_m"], expected_bm)

    def test_cli_preserves_an_omitted_mode_and_explicit_overrides(self) -> None:
        tuner = self.module.Mxfp4FlydslTuner("test", KEYS, [])
        for mode in (None, "prune", "full"):
            argv = ["gemm_moe_tune.py", "--mxfp4-flydsl", "--mp", "1"]
            if mode is not None:
                argv.extend(["--mxfp4-search-mode", mode])
            with self.subTest(mode=mode), mock.patch.object(sys, "argv", argv):
                args = tuner.parse_args()
                self.assertEqual(args.mxfp4_search_mode, mode)

    def test_full_search_keeps_only_the_coupled_shape_support_domain(self) -> None:
        unsupported = (
            {"model_dim": 256},  # GEMM1's pipelined loop needs two K tiles.
            {"model_dim": 768},  # More than one tile, but non-divisible k_wave.
            {"model_dim": 8448},  # B-scale addressing ends after 32 K tiles.
            {"inter_dim": 64},  # GEMM1 alone can emit this; GEMM2 cannot.
        )
        for changes in unsupported:
            row = shape_row(q_dtype_a="torch.float8_e4m3fn", **changes)
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

    def test_scatter_candidates_require_the_matching_generated_auxiliary_key(
        self,
    ) -> None:
        for hidden, topk, expected in (
            (1024, 2, {"atomic", "reduce"}),
            (3072, 4, {"atomic", "reduce", "scatter"}),
        ):
            # Expert/inter_dim do not participate in the scatter aux key.
            row = shape_row(
                model_dim=hidden, topk=topk, q_dtype_a="torch.float8_e4m3fn"
            )
            candidates = self.tuner._candidate_rows(row, full_search=True)
            epilogs = {
                self.module.parse_flydsl_v2_gemm2_kernel(c["kernelName2"])["epilog"]
                for c in candidates
                if c["block_m"] == 128
            }
            self.assertEqual(epilogs, expected)

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

    def test_bm16_a8_candidates_reach_the_persisted_winner_and_profile(self) -> None:
        row = self.read_rows([shape_row(token=64, q_dtype_a="torch.float8_e4m3fn")])
        result_columns = [
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
        ]
        tuner = self.module.Mxfp4FlydslTuner("test", KEYS, result_columns)
        args = SimpleNamespace(
            mxfp4_search_mode="full",
            timeout=0,
            mp=1,
            errRatio=0.1,
            profile_file=str(Path(self.directory.name) / "profile.csv"),
        )

        def evaluate(
            _self: Any, _row: Any, candidate: dict[str, Any], _args: argparse.Namespace
        ) -> float:
            candidate["error"] = 0.01
            candidate["us"] = float(candidate["block_m"])
            return candidate["us"]

        with mock.patch.object(
            self.module.Mxfp4FlydslTuner, "_run_candidate", evaluate
        ), mock.patch.object(torch.cuda, "device_count", return_value=0):
            results = tuner.tune(row, pd.DataFrame(), args)
        processed = tuner.post_process(results, args)
        path = Path(self.directory.name) / "winner.csv"
        tuner.result_to_csv(processed, str(path))
        saved = pd.read_csv(path)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved.iloc[0]["block_m"], 16)
        g1 = self.module._parse_mxfp4_g1_kname(saved.iloc[0]["kernelName1"])
        self.assertEqual(
            (g1["a_dtype"], g1["out_dtype"], g1["inline_quant"]), ("fp8", "fp8", True)
        )
        profile = pd.read_csv(args.profile_file)
        self.assertTrue((profile["block_m"] == 16).any())
        self.assertEqual(set(profile["precision"]), {"A8W4"})


if __name__ == "__main__":
    unittest.main()
