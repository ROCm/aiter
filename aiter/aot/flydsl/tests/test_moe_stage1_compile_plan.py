# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU parity and decision tests for the callable-bound Stage1 plan."""

from __future__ import annotations

from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch
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
    decisions = []
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
            original_decision = imports.moe.resolve_stage1_compile_decision

            def record_decision(values):
                decision = original_decision(values)
                decisions.append(decision)
                return decision

            with (
                mock.patch.object(
                    imports.moe,
                    "resolve_stage1_compile_decision",
                    side_effect=record_decision,
                ) as decision_spy,
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
        "decision_calls": decision_spy.call_count,
        "resolver_calls": resolver_spy.call_count,
        "same_decision": resolver_spy.call_args.kwargs["decision"] is decisions[0],
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
        self.assertEqual({name: os.environ.get(name) for name in before}, before)

    def test_runtime_passes_one_shared_decision_to_compile_plan(self) -> None:
        planned = _exercise_runtime_branch()
        self.assertEqual(
            (
                planned["decision_calls"],
                planned["resolver_calls"],
                planned["backend_calls"],
            ),
            (1, 1, 2),
        )
        self.assertTrue(planned["same_decision"])
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


class TestStage1CompileDecisions(unittest.TestCase):
    @staticmethod
    def _base(**overrides):
        values = {
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
        values.update(overrides)
        return values

    def test_decision_matrix_drives_exact_compile_units_cpu_only(self) -> None:
        recorder = _RequestRecorder()
        with _recording_environment(), ExitStack() as stack:
            _install_cuda_boundary_mocks(stack, recorder)
            with _isolated_host_imports() as imports:
                _install_boundary_mocks(stack, imports, recorder)
                decisions = importlib.import_module(
                    "aiter.ops.flydsl.moe_compile_decisions"
                )
                plans = importlib.import_module("aiter.ops.flydsl.moe_compile_plan")
                forbidden = mock.Mock(
                    side_effect=AssertionError("forbidden Stage1 AOT boundary")
                )
                for owner, attribute in (
                    (torch, "empty"),
                    (torch, "empty_like"),
                    (torch, "empty_strided"),
                    (torch, "full"),
                    (torch, "ones"),
                    (torch, "tensor"),
                    (torch, "zeros"),
                    (torch.cuda, "current_stream"),
                    (torch.cuda, "current_device"),
                    (torch.cuda, "get_device_properties"),
                    (torch.Tensor, "item"),
                ):
                    stack.enter_context(mock.patch.object(owner, attribute, forbidden))
                stack.enter_context(
                    mock.patch(
                        "torch._subclasses.fake_tensor.FakeTensorMode",
                        forbidden,
                    )
                )

                cases = (
                    (
                        "mixed-non-split",
                        self._base(enable_bias=True),
                        ("mixed", False, "bf16", True, None, None, None, None),
                        (plans.MIXED_STAGE1_GEMM_OP_ID,),
                    ),
                    (
                        "int4-external",
                        self._base(
                            a_dtype="bf16",
                            b_dtype="int4",
                            k_batch=4,
                        ),
                        (
                            "int4",
                            True,
                            "bf16",
                            False,
                            "external",
                            None,
                            None,
                            None,
                        ),
                        (plans.INT4_STAGE1_GEMM_OP_ID,),
                    ),
                    (
                        "fq-fp4-separated",
                        self._base(
                            out_dtype="fp4",
                            k_batch=4,
                            gate_mode="separated",
                            enable_bias=True,
                        ),
                        ("mixed", True, "bf16", False, "fq", "fp4", False, True),
                        (
                            plans.MIXED_STAGE1_GEMM_OP_ID,
                            plans.FQ_ACTIVATION_OP_ID,
                        ),
                    ),
                    (
                        "fq-fp8-interleaved",
                        self._base(
                            a_dtype="fp8",
                            out_dtype="fp8",
                            k_batch=4,
                            gate_mode="interleave",
                            enable_bias=True,
                        ),
                        ("mixed", True, "bf16", False, "fq", "fp8", True, True),
                        (
                            plans.MIXED_STAGE1_GEMM_OP_ID,
                            plans.FQ_ACTIVATION_OP_ID,
                        ),
                    ),
                    (
                        "fq-none-interleaved",
                        self._base(
                            a_dtype="fp8",
                            k_batch=4,
                            gate_mode="interleave",
                        ),
                        ("mixed", True, "bf16", False, "fq", "none", True, False),
                        (
                            plans.MIXED_STAGE1_GEMM_OP_ID,
                            plans.FQ_ACTIVATION_OP_ID,
                        ),
                    ),
                    (
                        "mixed-external",
                        self._base(k_batch=4, gate_mode="separated"),
                        (
                            "mixed",
                            True,
                            "bf16",
                            False,
                            "external",
                            None,
                            None,
                            None,
                        ),
                        (plans.MIXED_STAGE1_GEMM_OP_ID,),
                    ),
                )
                for name, kwargs, expected, op_ids in cases:
                    with self.subTest(name=name):
                        decision = decisions.resolve_stage1_compile_decision(kwargs)
                        self.assertEqual(
                            (
                                decision.primary_family,
                                decision.split_k,
                                decision.main_out_dtype,
                                decision.main_enable_bias,
                                decision.postprocess_kind,
                                decision.fq_quant_mode,
                                decision.fq_gui_layout,
                                decision.fq_enable_bias,
                            ),
                            expected,
                        )
                        plan = plans.resolve_moe_stage1_compile_plan(
                            context=recorder.compile_context,
                            decision=decision,
                            **kwargs,
                        )
                        self.assertEqual(
                            tuple(unit.spec.op_id for unit in plan.units),
                            op_ids,
                        )
                        main = dict(plan.units[0].spec.call.arguments)
                        self.assertNotIn("compile_target", main)
                        self.assertEqual(main["out_dtype"], decision.main_out_dtype)
                        self.assertEqual(
                            main["enable_bias"],
                            decision.main_enable_bias,
                        )
                        if decision.postprocess_kind == "fq":
                            fq = dict(plan.units[1].spec.call.arguments)
                            self.assertEqual(fq["quant_mode"], decision.fq_quant_mode)
                            self.assertEqual(fq["gui_layout"], decision.fq_gui_layout)
                            self.assertEqual(fq["enable_bias"], decision.fq_enable_bias)
                forbidden.assert_not_called()

    def test_rejects_invalid_graph_choices_and_preserves_abis(self) -> None:
        recorder = _RequestRecorder()
        with _recording_environment(), ExitStack() as stack:
            _install_cuda_boundary_mocks(stack, recorder)
            with _isolated_host_imports():
                plans = importlib.import_module("aiter.ops.flydsl.moe_compile_plan")
                invalid = (
                    (
                        {**self._base(), "b_dtype": "bf16"},
                        "unsupported Stage1 dtype",
                    ),
                    (
                        {
                            **self._base(),
                            "out_dtype": "fp8",
                            "k_batch": 4,
                            "gate_mode": "separated",
                        },
                        "interleaved",
                    ),
                )
                for kwargs, message in invalid:
                    with self.subTest(message=message):
                        with self.assertRaisesRegex(ValueError, message):
                            plans.resolve_moe_stage1_compile_plan(
                                context=recorder.compile_context,
                                **kwargs,
                            )

                expected_abi_names = {
                    plans.MIXED_STAGE1_GEMM_OP_ID: """
                        arg_out arg_x arg_w arg_scale_x arg_scale_w
                        arg_sorted_token_ids arg_expert_ids arg_sorted_weights
                        arg_max_token_ids arg_bias arg_out_scale_sorted
                        i32_tokens_in i32_inter_in i32_k_in
                        i32_size_expert_ids_in f32_swiglu_limit stream
                    """.split(),
                    plans.INT4_STAGE1_GEMM_OP_ID: """
                        arg_out arg_x arg_w arg_scale_x arg_scale_w
                        arg_sorted_token_ids arg_expert_ids arg_sorted_weights
                        arg_max_token_ids i32_tokens_in i32_inter_in i32_k_in
                        i32_size_expert_ids_in stream
                    """.split(),
                    plans.FQ_ACTIVATION_OP_ID: """
                        x out_buf out_scale_sorted sorted_ids num_valid_ids
                        topk_ids bias token_num num_sorted_rows swiglu_limit_f stream
                    """.split(),
                    plans.CKTILE_SWIGLU_AND_MUL_OP_ID: (
                        "x out num_rows stream".split()
                    ),
                }
                for op_id, names in expected_abi_names.items():
                    with self.subTest(op_id=op_id):
                        abi = plans.stage1_abi(op_id)
                        self.assertEqual(
                            tuple(argument.name for argument in abi.arguments),
                            tuple(names),
                        )
                        hash(abi)


if __name__ == "__main__":
    unittest.main()
