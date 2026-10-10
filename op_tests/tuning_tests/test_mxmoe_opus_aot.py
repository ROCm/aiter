# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU-free AOT coverage for explicit MXMOE A8W4 -> OPUS CSV pairs."""

import csv
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

from op_tests.tuning_tests.mxfp4_cpu_test_utils import cpu_tuner_environment


class TestMxmoeOpusAot(unittest.TestCase):
    def setUp(self):
        environment = cpu_tuner_environment()
        environment.__enter__()
        self.addCleanup(environment.__exit__, None, None, None)
        self.aot = importlib.import_module("aiter.aot.flydsl.mxfp4_moe")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.g1 = "flydsl_mxmoe_g1_a8w4_16x128x256_f16in_nt_fp8out"
        self.native = "flydsl_moe2_layout_afp8_wfp4_bf16_t16x128x128_atomic_sbm16"
        self.opus = "opus_moe2_layout_afp8_wfp4_atomic_t16x128x256_sbm16_cache_b3_ws2"

    def row(self, consumer, **changes):
        row = {
            "gfx": "gfx950",
            "cu_num": 256,
            "token": 32,
            "model_dim": 3072,
            "inter_dim": 384,
            "expert": 128,
            "topk": 4,
            "act_type": "ActivationType.Silu",
            "dtype": "torch.bfloat16",
            "q_dtype_a": "torch.float8_e4m3fn",
            "q_dtype_w": "torch.float4_e2m1fn_x2",
            "q_type": "QuantType.per_1x32",
            "use_g1u1": 1,
            "doweight_stage1": 0,
            "kernelName1": self.g1,
            "kernelName2": consumer,
            "block_m": 16,
            "ksplit": 0,
            "run_1stage": 0,
        }
        row.update(changes)
        return row

    def write_csv(self, rows):
        path = Path(self.directory.name) / "explicit.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_same_g1_keeps_native_and_regular_aot_jobs(self):
        rows = [self.row(self.native), self.row(self.opus)]
        jobs = self.aot.parse_csv(str(self.write_csv(rows + rows)))
        stage1 = [job for job in jobs if job["stage"] == 1]
        self.assertEqual(len(stage1), 8)
        self.assertEqual(
            {
                (job["native_scale_layout"], job["interleave"], job["enable_bias"])
                for job in stage1
            },
            {
                (True, False, False),
                (True, False, True),
                (True, True, False),
                (True, True, True),
                (False, False, False),
                (False, False, True),
                (False, True, False),
                (False, True, True),
            },
        )
        self.assertEqual({job["D_INTER"] for job in stage1}, {384})
        self.assertTrue(all(job["kernel_name"] == self.g1 for job in stage1))
        self.assertFalse(any(job["kernel_name"] == self.opus for job in jobs))

    def test_aot_and_runtime_pass_the_same_layout_and_bias_to_gemm1(self):
        torch = importlib.import_module("torch")
        fm = importlib.import_module("aiter.fused_moe")
        launcher = importlib.import_module("aiter.ops.flydsl.mxfp4_gemm1_kernels")
        jobs = [
            job
            for job in self.aot.parse_csv(
                str(
                    self.write_csv(
                        [
                            self.row(self.native),
                            self.row(self.opus),
                        ]
                    )
                )
            )
            if job["stage"] == 1
        ]
        hidden = torch.empty((31, 3072), dtype=torch.bfloat16)
        w1 = torch.empty((128, 768, 1536), dtype=torch.uint8)
        w2 = torch.empty((128, 3072, 192), dtype=torch.uint8)
        ids = torch.empty(144, dtype=torch.int32)
        experts = torch.empty(9, dtype=torch.int32)
        counts = torch.tensor([144, 31], dtype=torch.int32)
        metadata = [
            fm._make_mxfp4_metadata(self.g1, consumer, fm.GateMode.INTERLEAVE, 0)
            for consumer in (self.native, self.opus, self.native)
        ]
        self.assertEqual(
            [item.stage1.keywords["native_scale_layout"] for item in metadata],
            [True, False, True],
        )
        for job in jobs:
            with self.subTest(
                native=job["native_scale_layout"],
                interleave=job["interleave"],
                bias=job["enable_bias"],
            ):
                with mock.patch.object(launcher, "flydsl_mxfp4_gemm1") as launch:
                    self.aot._compile_stage1(job)
                    prepared = launch.call_args.kwargs
                    chosen = metadata[0] if job["native_scale_layout"] else metadata[1]
                    chosen.stage1(
                        hidden,
                        w1,
                        w2,
                        ids,
                        experts,
                        counts,
                        None,
                        4,
                        block_m=16,
                        interleave=job["interleave"],
                        bias1=(
                            torch.empty((128, 768), dtype=torch.float32)
                            if job["enable_bias"]
                            else None
                        ),
                    )
                    runtime = launch.call_args.kwargs
                for key in (
                    "BM",
                    "BN",
                    "BK",
                    "NE",
                    "D_HIDDEN",
                    "D_INTER",
                    "topk",
                    "use_nt",
                    "inline_quant",
                    "prefetch_hidden",
                    "a_dtype",
                    "out_dtype",
                    "act",
                    "situ_beta",
                    "situ_linear_beta",
                    "swiglu_limit",
                    "interleave",
                    "native_scale_layout",
                    "num_waves",
                    "k_wave",
                ):
                    self.assertEqual(prepared[key], runtime[key], key)
                self.assertEqual(prepared["bias"] is not None, job["enable_bias"])
                self.assertEqual(runtime["bias"] is not None, job["enable_bias"])
        self.assertEqual(
            [item.stage1.keywords["native_scale_layout"] for item in metadata],
            [True, False, True],
        )
        self.assertFalse(torch.cuda.is_initialized())

    def test_compile_cache_reuses_native_after_independent_regular_compile(self):
        launcher = importlib.import_module("aiter.ops.flydsl.mxfp4_gemm1_kernels")
        cached = launcher._get_compiled_mxfp4_gemm1_port
        # The approved compile boundary uses an inert module so CPU checks
        # never import or execute the real FlyDSL compiler.
        module = ModuleType("aiter.ops.flydsl.kernels.mxfp4_gemm1")
        native, regular = object(), object()
        module.compile_gemm1_a4w4_port = mock.Mock(
            side_effect=lambda *args, **kwargs: (
                native if kwargs["native_scale_layout"] else regular
            )
        )
        cached.cache_clear()
        self.addCleanup(cached.cache_clear)
        with mock.patch.dict(sys.modules, {module.__name__: module}):
            results = [
                cached(
                    16,
                    True,
                    True,
                    False,
                    3072,
                    384,
                    128,
                    128,
                    256,
                    interleave=True,
                    a_dtype="fp8",
                    out_dtype="fp8",
                    swiglu_limit=float("inf"),
                    native_scale_layout=layout,
                )
                for layout in (True, False, True)
            ]
        self.assertIs(results[0], native)
        self.assertIs(results[1], regular)
        self.assertIs(results[2], results[0])
        self.assertEqual(module.compile_gemm1_a4w4_port.call_count, 2)
        self.assertEqual(
            [
                call.kwargs["native_scale_layout"]
                for call in module.compile_gemm1_a4w4_port.call_args_list
            ],
            [True, False],
        )

    def test_opus_aot_preserves_activation_and_shared_expert_bias_variants(self):
        for suffix, activation, expected_parameters in (
            ("", "silu", {(1.0, 1.0, float("inf"))}),
            (
                "_situv2",
                "situv2",
                {(1.0, 1.0, float("inf")), (4.0, 25.0, float("inf"))},
            ),
            ("_swiglu", "swiglu", {(1.0, 1.0, 7.0)}),
        ):
            for shared_expert in (-1, 0):
                with self.subTest(activation=activation, shared_expert=shared_expert):
                    row = self.row(
                        self.opus,
                        kernelName1=self.g1 + suffix,
                        shared_expert_id=shared_expert,
                    )
                    jobs = self.aot.parse_csv(str(self.write_csv([row, row])))
                    self.assertEqual(
                        len(jobs),
                        len(expected_parameters) * 2 * (2 if shared_expert < 0 else 1),
                    )
                    self.assertEqual(
                        {
                            (
                                job["situ_beta"],
                                job["situ_linear_beta"],
                                job["swiglu_limit"],
                            )
                            for job in jobs
                        },
                        expected_parameters,
                    )
                    self.assertEqual({job["act"] for job in jobs}, {activation})
                    self.assertEqual(
                        {job["enable_bias"] for job in jobs},
                        {False, True} if shared_expert < 0 else {False},
                    )
                    self.assertTrue(all(not job["native_scale_layout"] for job in jobs))

    def test_opus_consumer_does_not_hide_forbidden_bias_names(self):
        for suffix in ("_bias", "_BIAS"):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(
                ValueError, "unknown token"
            ):
                self.aot.parse_csv(
                    str(
                        self.write_csv(
                            [
                                self.row(self.opus, kernelName1=self.g1 + suffix),
                            ]
                        )
                    )
                )


if __name__ == "__main__":
    unittest.main()
