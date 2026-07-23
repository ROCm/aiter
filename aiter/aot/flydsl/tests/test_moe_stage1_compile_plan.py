# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU parity and integration tests for the callable-bound Stage1 plan."""

from __future__ import annotations

from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

from torch._subclasses.fake_tensor import FakeTensorMode

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from moe_compile_recorder import (  # noqa: E402
    _RequestRecorder,
    _clear_scenario_caches,
    _install_boundary_mocks,
    _install_cuda_boundary_mocks,
    _isolated_host_imports,
    _recording_environment,
    _run_stage1,
    record_compile_requests,
)

_GOLDEN = _TEST_DIR / "data" / "moe_compile_requests_gfx950.json"
_KERNEL_NAME = "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w3_kb4_fp4"
_STAGE1_SCENARIOS = {
    "stage1.main.non_split.bias.route_weighted",
    "stage1.int4.splitk",
    "stage1.splitk.fp4.silu.separated",
    "stage1.splitk.fp8.swiglu.interleaved.bias",
    "stage1.splitk.none.silu.interleaved",
    "cktile.epilogue.silu",
    "cktile.epilogue.swiglu",
}


def _stage1_projection(recording):
    return {
        request["id"]: {
            "builder": request["builder"],
            "kwargs": request["kwargs"],
            "launchers": request["trigger"]["launchers"],
            "scenario": request["trigger"]["scenario"],
        }
        for request in recording["requests"]
        if request["trigger"]["scenario"] in _STAGE1_SCENARIOS
    }


def _exercise_runtime_branch():
    recorder = _RequestRecorder()
    fake_mode = FakeTensorMode()
    before = {
        "FLYDSL_GPU_ARCH": os.environ.get("FLYDSL_GPU_ARCH"),
        "CU_NUM": os.environ.get("CU_NUM"),
    }

    with (
        _recording_environment(),
        mock.patch.dict(os.environ, {"CU_NUM": "256"}),
        ExitStack() as stack,
    ):
        _install_cuda_boundary_mocks(stack, recorder)
        with _isolated_host_imports() as imports:
            _install_boundary_mocks(stack, imports, recorder)
            plan_module = importlib.import_module("aiter.ops.flydsl.moe_compile_plan")
            _clear_scenario_caches(imports)
            imports.moe.compile_flydsl_moe_stage1.cache_clear()

            with (
                mock.patch.object(
                    plan_module,
                    "resolve_moe_stage1_compile_plan",
                    wraps=plan_module.resolve_moe_stage1_compile_plan,
                ) as resolver_spy,
                mock.patch.object(
                    recorder.backend,
                    "resolve_aot",
                    wraps=recorder.backend.resolve_aot,
                ) as backend_spy,
                fake_mode,
                recorder.scenario("runtime.stage1.compile_plan"),
            ):
                _run_stage1(imports, _KERNEL_NAME)

    after = {
        "FLYDSL_GPU_ARCH": os.environ.get("FLYDSL_GPU_ARCH"),
        "CU_NUM": os.environ.get("CU_NUM"),
    }
    return {
        "requests": recorder.requests,
        "resolver_calls": resolver_spy.call_count,
        "backend_calls": backend_spy.call_count,
        "backend_op_ids": [
            call.args[0].spec.op_id for call in backend_spy.call_args_list
        ],
        "environment_restored": before == after,
    }


class TestStage1GoldenParity(unittest.TestCase):
    def test_default_plan_matches_every_stage1_golden_builder_request(self) -> None:
        golden = json.loads(_GOLDEN.read_text())
        before = {"CU_NUM": os.environ.get("CU_NUM")}
        actual = record_compile_requests()

        self.assertEqual(_stage1_projection(actual), _stage1_projection(golden))
        self.assertEqual(
            {
                request["trigger"]["scenario"]
                for request in actual["requests"]
                if request["trigger"]["scenario"] in _STAGE1_SCENARIOS
            },
            _STAGE1_SCENARIOS,
        )
        self.assertEqual(
            {name: os.environ.get(name) for name in before},
            before,
        )

    def test_default_runtime_calls_resolver_and_backend(self) -> None:
        planned = _exercise_runtime_branch()
        self.assertEqual(
            (
                planned["resolver_calls"],
                planned["backend_calls"],
            ),
            (1, 2),
        )
        self.assertEqual(
            planned["backend_op_ids"],
            [
                "aiter.flydsl.moe.stage1.mixed_gemm.v1",
                "aiter.flydsl.moe.stage1.silu_and_mul_fq.v1",
            ],
        )
        self.assertEqual(
            [request["builder"].rsplit(".", 1)[-1] for request in planned["requests"]],
            ["compile_mixed_moe_gemm1", "build_silu_and_mul_fq_module"],
        )
        self.assertTrue(planned["environment_restored"])


class TestStage1ResolverBoundaries(unittest.TestCase):
    def test_graph_errors_and_manual_abis_are_data_driven(self) -> None:
        recorder = _RequestRecorder()
        with _recording_environment(), ExitStack() as stack:
            _install_cuda_boundary_mocks(stack, recorder)
            with _isolated_host_imports():
                core = importlib.import_module("aiter.ops.flydsl.compile_plan")
                plan_module = importlib.import_module(
                    "aiter.ops.flydsl.moe_compile_plan"
                )
                target = core.RocmTarget("gfx950", 256)
                backend = mock.Mock()
                backend.compile_aot = mock.Mock()
                backend.load_aot = mock.Mock()
                backend.resolve_aot = mock.Mock()
                context = core.CompileContext(
                    target=target,
                    registry=core.DEFAULT_COMPILE_OP_REGISTRY,
                    backend=backend,
                )
                base = {
                    "model_dim": 7168,
                    "inter_dim": 2048,
                    "experts": 256,
                    "topk": 8,
                    "tile_m": 32,
                    "tile_n": 128,
                    "tile_k": 256,
                    "doweight_stage1": False,
                    "a_dtype": "fp4",
                    "b_dtype": "fp4",
                    "out_dtype": "bf16",
                }

                invalid = (
                    (
                        lambda: plan_module.resolve_moe_stage1_compile_plan(
                            context=context,
                            **{**base, "b_dtype": "bf16"},
                        ),
                        "unsupported Stage1 dtype",
                    ),
                    (
                        lambda: plan_module.resolve_moe_stage1_compile_plan(
                            context=context,
                            **{
                                **base,
                                "out_dtype": "fp8",
                                "k_batch": 4,
                                "gate_mode": "separated",
                            },
                        ),
                        "interleaved",
                    ),
                    (
                        lambda: plan_module.resolve_cktile_stage1_compile_plan(
                            context=context,
                            inter_dim=2048,
                            topk=8,
                            split_k=2,
                            act="silu",
                            post_activation_layout="auto",
                        ),
                        "ambiguous",
                    ),
                    (
                        lambda: plan_module.resolve_cktile_stage1_compile_plan(
                            context=context,
                            inter_dim=2048,
                            topk=8,
                            split_k=2,
                            act="silu",
                            post_activation_layout="interleaved",
                            enable_bias=True,
                        ),
                        "bias",
                    ),
                )
                for resolve, message in invalid:
                    with self.subTest(message=message):
                        with self.assertRaisesRegex(ValueError, message):
                            resolve()

                expected_abi_names = {
                    plan_module.MIXED_STAGE1_GEMM_OP_ID: """
                        arg_out arg_x arg_w arg_scale_x arg_scale_w
                        arg_sorted_token_ids arg_expert_ids arg_sorted_weights
                        arg_max_token_ids arg_bias arg_out_scale_sorted
                        i32_tokens_in i32_inter_in i32_k_in
                        i32_size_expert_ids_in f32_swiglu_limit stream
                    """.split(),
                    plan_module.INT4_STAGE1_GEMM_OP_ID: """
                        arg_out arg_x arg_w arg_scale_x arg_scale_w
                        arg_sorted_token_ids arg_expert_ids arg_sorted_weights
                        arg_max_token_ids i32_tokens_in i32_inter_in i32_k_in
                        i32_size_expert_ids_in stream
                    """.split(),
                    plan_module.FQ_ACTIVATION_OP_ID: """
                        x out_buf out_scale_sorted sorted_ids num_valid_ids
                        topk_ids bias token_num num_sorted_rows swiglu_limit_f stream
                    """.split(),
                    plan_module.CKTILE_SWIGLU_AND_MUL_OP_ID: (
                        "x out num_rows stream".split()
                    ),
                }
                for op_id, names in expected_abi_names.items():
                    with self.subTest(op_id=op_id):
                        abi = plan_module.stage1_abi(op_id)
                        self.assertEqual(
                            tuple(argument.name for argument in abi.arguments),
                            tuple(names),
                        )
                        hash(abi)


if __name__ == "__main__":
    unittest.main()
