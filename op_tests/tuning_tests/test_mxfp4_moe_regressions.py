# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import csv
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import aiter
from aiter import dtypes
from aiter.aot.flydsl import mxfp4_moe as aot_mxfp4_moe
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.moe_mxfp4_aux import is_mxfp4_moe_shape_supported

fused_moe = importlib.import_module("aiter.fused_moe")


class TestMxfp4MoeRegressions(unittest.TestCase):
    def test_stage1_defaults_to_native_scale_layout_for_bm16(self):
        hidden = torch.zeros((1, 32), dtype=torch.bfloat16)
        w1 = torch.zeros((2, 8, 16), dtype=torch.uint8)
        w2 = torch.zeros((2, 32, 4), dtype=torch.uint8)
        sorted_ids = torch.zeros(1, dtype=torch.int32)
        expert_ids = torch.zeros(1, dtype=torch.int32)
        num_valid_ids = torch.ones(1, dtype=torch.int32)
        m_indices = torch.zeros(1, dtype=torch.int32)

        def run(kernel_name):
            captured = {}

            def fake_stage1(*args, **kwargs):
                captured.update(kwargs)
                return torch.empty(0), torch.empty(0)

            with patch.object(fused_moe, "_mxfp4_a4w4_stage1", side_effect=fake_stage1):
                fused_moe._mxfp4_a4w4_stage1_fw(
                    hidden,
                    w1,
                    w2,
                    sorted_ids,
                    expert_ids,
                    num_valid_ids,
                    None,
                    1,
                    kernelName1=kernel_name,
                    m_indices=m_indices,
                )
            return captured["native_scale_layout"]

        self.assertTrue(run("flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt"))
        self.assertFalse(run("flydsl_mxmoe_g1_a4w4_32x256x256"))

    def test_layout_stage2_forwards_bias(self):
        inter_states = torch.empty((1, 1, 4), dtype=torch.uint8)
        w1 = torch.empty((2, 8, 16), dtype=torch.uint8)
        w2 = torch.empty((2, 32, 4), dtype=torch.bfloat16)
        sorted_ids = torch.zeros(1, dtype=torch.int32)
        expert_ids = torch.zeros(1, dtype=torch.int32)
        num_valid_ids = torch.ones(1, dtype=torch.int32)
        out = torch.empty((1, 32), dtype=torch.bfloat16)
        bias = torch.ones((2, 32), dtype=torch.float32)
        captured = {}

        def fake_stage2(**kwargs):
            captured.update(kwargs)
            return kwargs["out"]

        with (
            patch.object(fused_moe, "parse_g2_kname_any", return_value={"v2": True}),
            patch.object(
                fused_moe, "_flydsl_v2_stage2_wrapper", side_effect=fake_stage2
            ),
        ):
            result = fused_moe._mxfp4_a4w4_stage2_fw(
                inter_states,
                w1,
                w2,
                sorted_ids,
                expert_ids,
                num_valid_ids,
                out,
                1,
                bias2=bias,
                kernelName2="flydsl_moe2_layout_test",
            )

        self.assertIs(result, out)
        self.assertIs(captured["bias2"], bias)

        with (
            patch.object(fused_moe, "parse_g2_kname_any", return_value={"v2": False}),
            self.assertRaisesRegex(ValueError, "does not support bias"),
        ):
            fused_moe._mxfp4_a4w4_stage2_fw(
                inter_states,
                w1,
                w2,
                sorted_ids,
                expert_ids,
                num_valid_ids,
                out,
                1,
                bias2=bias,
                kernelName2="flydsl_mxmoe_g2_a4w4_16x256x256_atomic",
            )

    def test_aot_stage1_key_matches_kernel_name_and_stored_width(self):
        fieldnames = [
            "token",
            "topk",
            "model_dim",
            "expert",
            "inter_dim",
            "kernelName1",
            "kernelName2",
            "cu_num",
        ]
        rows = [
            {
                "token": 2048,
                "topk": 9,
                "model_dim": 7168,
                "expert": 385,
                "inter_dim": 512,
                "kernelName1": "flydsl_mxmoe_g1_a4w4_32x256x256_nt",
                "kernelName2": "",
                "cu_num": 256,
            },
            {
                "token": 4096,
                "topk": 16,
                "model_dim": 3584,
                "expert": 896,
                "inter_dim": 384,
                "kernelName1": "flydsl_mxmoe_g1_a4w4_64x256x256_situv2_xcd2",
                "kernelName2": "",
                "cu_num": 256,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "configs.csv"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            jobs = aot_mxfp4_moe.parse_csv(str(path))

        nt_job = next(job for job in jobs if job["D_HIDDEN"] == 7168)
        self.assertTrue(nt_job["use_nt"])

        # inter_dim=384 ships padded to 512, and stage1 derives D_INTER from the
        # stored width (w1.shape[1] // 2), so the AOT job must key on 512 too.
        kimi_jobs = [job for job in jobs if job["D_HIDDEN"] == 3584]
        self.assertEqual(len(kimi_jobs), 2)
        self.assertEqual({job["D_INTER"] for job in kimi_jobs}, {512})
        self.assertEqual(len({aot_mxfp4_moe._job_key(job) for job in kimi_jobs}), 2)

    def test_tuned_config_rejects_missing_bias_capability(self):
        fieldnames = [
            "gfx",
            "cu_num",
            "token",
            "model_dim",
            "inter_dim",
            "expert",
            "topk",
            "shared_expert_id",
            "act_type",
            "dtype",
            "q_dtype_a",
            "q_dtype_w",
            "q_type",
            "use_g1u1",
            "doweight_stage1",
            "hidden_pad",
            "intermediate_pad",
            "gate_mode",
            "block_m",
            "ksplit",
            "kernelName1",
            "kernelName2",
        ]
        row = {
            "gfx": "gfx950",
            "cu_num": 256,
            "token": 16,
            "model_dim": 6144,
            "inter_dim": 512,
            "expert": 257,
            "topk": 9,
            "shared_expert_id": 256,
            "act_type": "ActivationType.Silu",
            "dtype": "torch.bfloat16",
            "q_dtype_a": "torch.float4_e2m1fn_x2",
            "q_dtype_w": "torch.float4_e2m1fn_x2",
            "q_type": "QuantType.per_1x32",
            "use_g1u1": 1,
            "doweight_stage1": 0,
            "hidden_pad": 0,
            "intermediate_pad": 0,
            "gate_mode": "GateMode.SEPARATED",
            "block_m": 16,
            "ksplit": 0,
            "kernelName1": "flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt",
            "kernelName2": (
                "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16"
            ),
        }

        def resolve(path, **kwargs):
            fused_moe.get_2stage_cfgs.cache_clear()
            fused_moe.cfg_2stages_by_file.clear()
            return fused_moe.get_2stage_cfgs(
                16,
                6144,
                512,
                257,
                9,
                torch.bfloat16,
                dtypes.fp4x2,
                dtypes.fp4x2,
                aiter.QuantType.per_1x32,
                True,
                aiter.ActivationType.Silu,
                False,
                0,
                0,
                True,
                GateMode.SEPARATED.value,
                config_file=str(path),
                **kwargs,
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(fused_moe, "get_cu_num", return_value=256),
            patch.object(fused_moe, "get_gfx_runtime", return_value="gfx950"),
        ):
            path = Path(directory) / "configs.csv"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)

            resolve(path)
            with self.assertRaisesRegex(
                NotImplementedError, "requires an exact tuned config"
            ):
                resolve(path, has_stage1_bias=True)

            row["kernelName1"] += "_bias"
            row["kernelName2"] = "flydsl_mxmoe_g2_a4w4_16x256x256_atomic"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(
                NotImplementedError, "requires an exact tuned config"
            ):
                resolve(path, has_stage1_bias=True, has_stage2_bias=True)

    def test_gemm1_cache_key_covers_scalars_absent_from_the_symbol_name(self):
        """situ_beta / situ_linear_beta / swiglu_limit are compile-time constants
        that ``name_suffix`` deliberately omits. That is only safe while FlyDSL's
        cross-process cache keys on closure scalars rather than on the symbol."""
        from flydsl.compiler.jit_function import (
            _get_underlying_func,
            _jit_function_cache_key,
        )

        from aiter.ops.flydsl.kernels.mxfp4_gemm1 import compile_gemm1_a4w4_port

        def key(**kwargs):
            launcher = compile_gemm1_a4w4_port(
                BM=32,
                use_nt=False,
                inline_quant=False,
                D_HIDDEN=7168,
                D_INTER=512,
                NE=32,
                **kwargs,
            )
            return _jit_function_cache_key(_get_underlying_func(launcher))

        self.assertNotEqual(
            key(act="situv2", situ_beta=1.0, situ_linear_beta=1.0),
            key(act="situv2", situ_beta=4.0, situ_linear_beta=25.0),
        )
        self.assertNotEqual(
            key(act="swiglu", swiglu_limit=7.0),
            key(act="swiglu", swiglu_limit=10.0),
        )

    def test_generated_aux_shape_guard(self):
        self.assertTrue(is_mxfp4_moe_shape_supported(896, 3584, 384, 16))
        self.assertFalse(is_mxfp4_moe_shape_supported(999, 3584, 384, 16))


if __name__ == "__main__":
    unittest.main()
