# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU parity and integration tests for the callable-bound Stage2 plan."""

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
    _kernel_params,
    _recording_environment,
    _run_stage2,
    _stage_shape,
)

_GOLDEN = _TEST_DIR / "data" / "moe_compile_requests_gfx950.json"
_STAGE2_CASES = (
    (
        "stage2.atomic.bias",
        "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic_bnt2",
        False,
        {"enable_bias": True},
    ),
    (
        "stage2.int4.atomic",
        "flydsl_moe2_abf16_wint4_bf16_t16x128x128_atomic",
        False,
        {
            "model_dim": 7168,
            "inter_dim": 256,
            "experts": 384,
            "topk": 8,
            "token_num": 16,
        },
    ),
    (
        "stage2.reduce.plain",
        "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce_bnt2",
        False,
        {},
    ),
    (
        "stage2.reduce.plain.large_auto_persist",
        "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce_bnt2",
        False,
        {"token_num": 4096},
    ),
    (
        "stage2.reduce.masked_ep",
        "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce_bnt2",
        True,
        {},
    ),
)
_EXPECTED_OP_IDS = {
    "stage2.atomic.bias": ("aiter.flydsl.moe.stage2.mixed_gemm.v1",),
    "stage2.int4.atomic": ("aiter.flydsl.moe.stage2.int4_gemm.v1",),
    "stage2.reduce.plain": (
        "aiter.flydsl.moe.stage2.mixed_gemm.v1",
        "aiter.flydsl.moe.stage2.reduction.plain.v1",
    ),
    "stage2.reduce.plain.large_auto_persist": (
        "aiter.flydsl.moe.stage2.mixed_gemm.v1",
        "aiter.flydsl.moe.stage2.reduction.plain.v1",
    ),
    "stage2.reduce.masked_ep": (
        "aiter.flydsl.moe.stage2.mixed_gemm.v1",
        "aiter.flydsl.moe.stage2.reduction.masked.v1",
    ),
}


def _case_config(imports, kernel_name, masked, options):
    return {
        **_stage_shape(experts=32 if masked else 256),
        **_kernel_params(imports, kernel_name, expected_stage=2),
        **options,
    }


def _provider_kwargs(config, *, masked):
    mode = config.get("mode", "atomic")
    out_dtype = config["out_dtype"]
    return {
        **config,
        "doweight_stage2": not config.get("doweight_stage1", False),
        "accumulate": mode != "reduce",
        "return_per_slot": False,
        "persist": config.get("persist"),
        "routing_block_count": max(1, config["token_num"] * config["topk"]),
        "dtype_str": "bf16" if out_dtype == "bf16" else "f16",
        "use_mask": masked,
        "topk_ids_available": masked,
        "num_experts": 256 if masked else 0,
    }


def _projection(recording):
    return {
        request["id"]: {
            "builder": request["builder"],
            "kwargs": request["kwargs"],
            "launchers": request["trigger"]["launchers"],
            "scenario": request["trigger"]["scenario"],
        }
        for request in recording["requests"]
        if request["trigger"]["scenario"] in _EXPECTED_OP_IDS
    }


class TestStage2GoldenParity(unittest.TestCase):
    def test_provider_matches_all_stage2_golden_requests_and_ordered_abis(
        self,
    ) -> None:
        golden = json.loads(_GOLDEN.read_text())
        recorder = _RequestRecorder()
        ordered_units = {}

        with _recording_environment(), ExitStack() as stack:
            _install_cuda_boundary_mocks(stack, recorder)
            with _isolated_host_imports() as imports:
                _install_boundary_mocks(stack, imports, recorder)
                plan_module = importlib.import_module(
                    "aiter.ops.flydsl.moe_compile_plan"
                )
                for scenario, kernel_name, masked, options in _STAGE2_CASES:
                    _clear_scenario_caches(imports)
                    config = _case_config(imports, kernel_name, masked, options)
                    with recorder.scenario(scenario):
                        plan = plan_module.resolve_moe_stage2_compile_plan(
                            context=recorder.compile_context,
                            **_provider_kwargs(config, masked=masked),
                        )
                        ordered_units[scenario] = tuple(
                            (
                                unit.spec.op_id,
                                tuple(
                                    argument.name
                                    for argument in unit.signature.arguments
                                ),
                            )
                            for unit in plan.units
                        )
                        for unit in plan.units:
                            recorder.backend.compile_aot(
                                unit,
                                context=recorder.compile_context,
                            )

        self.assertEqual(
            _projection({"requests": recorder.requests}),
            _projection(golden),
        )
        expected_abi_names = {
            plan_module.MIXED_STAGE2_GEMM_OP_ID: """
                arg_out arg_x arg_w arg_scale_x arg_scale_w
                arg_sorted_token_ids arg_expert_ids arg_sorted_weights
                arg_num_valid_ids arg_bias i32_tokens_in i32_n_in i32_k_in
                i32_size_expert_ids_in stream
            """.split(),
            plan_module.INT4_STAGE2_GEMM_OP_ID: """
                arg_out arg_x arg_w arg_scale_x arg_scale_w
                arg_sorted_token_ids arg_expert_ids arg_sorted_weights
                arg_num_valid_ids i32_tokens_in i32_n_in i32_k_in
                i32_size_expert_ids_in stream
            """.split(),
            plan_module.PLAIN_REDUCTION_OP_ID: (
                "X Y expert_mask topk_ids i32_m_tokens stream".split()
            ),
            plan_module.MASKED_REDUCTION_OP_ID: (
                "X Y expert_mask topk_ids i32_m_tokens stream".split()
            ),
        }
        for scenario, units in ordered_units.items():
            with self.subTest(scenario=scenario):
                self.assertEqual(
                    tuple(op_id for op_id, _ in units),
                    _EXPECTED_OP_IDS[scenario],
                )
                self.assertEqual(
                    tuple(names for op_id, names in units),
                    tuple(
                        tuple(expected_abi_names[op_id])
                        for op_id in _EXPECTED_OP_IDS[scenario]
                    ),
                )
                for op_id, _ in units:
                    hash(plan_module.stage2_abi(op_id))

    def test_runtime_uses_provider_and_backend_for_both_units(self) -> None:
        recorder = _RequestRecorder()
        fake_mode = FakeTensorMode()
        with _recording_environment(), ExitStack() as stack:
            _install_cuda_boundary_mocks(stack, recorder)
            with _isolated_host_imports() as imports:
                _install_boundary_mocks(stack, imports, recorder)
                plan_module = importlib.import_module(
                    "aiter.ops.flydsl.moe_compile_plan"
                )
                _clear_scenario_caches(imports)
                with (
                    mock.patch.object(
                        plan_module,
                        "resolve_moe_stage2_compile_plan",
                        wraps=plan_module.resolve_moe_stage2_compile_plan,
                    ) as resolver_spy,
                    mock.patch.object(
                        recorder.backend,
                        "resolve_aot",
                        wraps=recorder.backend.resolve_aot,
                    ) as backend_spy,
                    fake_mode,
                    recorder.scenario("runtime.stage2.compile_plan"),
                ):
                    _run_stage2(
                        imports,
                        "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce_bnt2",
                    )

        self.assertEqual(resolver_spy.call_count, 0)
        self.assertEqual(backend_spy.call_count, 2)
        self.assertEqual(
            [call.args[0].spec.op_id for call in backend_spy.call_args_list],
            [
                "aiter.flydsl.moe.stage2.mixed_gemm.v1",
                "aiter.flydsl.moe.stage2.reduction.plain.v1",
            ],
        )

    def test_operation_roles_drive_on_mode_once_and_provider_is_cpu_only(self) -> None:
        recorder = _RequestRecorder()
        fake_mode = FakeTensorMode()
        with _recording_environment(), ExitStack() as stack:
            _install_cuda_boundary_mocks(stack, recorder)
            with _isolated_host_imports() as imports:
                _install_boundary_mocks(stack, imports, recorder)
                plan_module = importlib.import_module(
                    "aiter.ops.flydsl.moe_compile_plan"
                )
                config = _case_config(
                    imports,
                    "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce_bnt2",
                    False,
                    {},
                )
                values = _provider_kwargs(config, masked=False)
                case = plan_module.MoeStage2OperationCase.from_kwargs(
                    values,
                    mode=values["mode"],
                    return_per_slot=values["return_per_slot"],
                    persist=values["persist"],
                    token_num=values["token_num"],
                    routing_block_count=values["routing_block_count"],
                    use_mask=values["use_mask"],
                    topk_ids_available=values["topk_ids_available"],
                    num_experts=values["num_experts"],
                )
                forbidden = mock.Mock(side_effect=AssertionError("GPU boundary"))
                with (
                    mock.patch.object(torch.cuda, "current_stream", forbidden),
                    mock.patch.object(torch.cuda, "get_device_properties", forbidden),
                ):
                    operation_plan = plan_module.resolve_moe_stage2_operation_plan(
                        case,
                        context=recorder.compile_context,
                    )
                    self.assertEqual(
                        [node.role for node in operation_plan.nodes],
                        [
                            "moe.stage2.gemm.mixed",
                            "moe.stage2.reduction.plain",
                        ],
                    )
                    self.assertEqual(
                        operation_plan.nodes[1].dependencies,
                        ("gemm",),
                    )
                    forbidden.assert_not_called()

                _clear_scenario_caches(imports)
                with (
                    mock.patch.dict(
                        os.environ,
                        {"AITER_FLYDSL_OPERATION_PLAN": "on"},
                    ),
                    mock.patch.object(
                        imports.moe._STAGE2_RUNTIME_ADAPTERS,
                        "lookup",
                        wraps=imports.moe._STAGE2_RUNTIME_ADAPTERS.lookup,
                    ) as lookup,
                    fake_mode,
                    recorder.scenario("runtime.stage2.operation_plan"),
                ):
                    _run_stage2(
                        imports,
                        "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce_bnt2",
                    )
                self.assertEqual(
                    [call.args[0] for call in lookup.call_args_list],
                    [
                        "moe.stage2.gemm.mixed",
                        "moe.stage2.reduction.plain",
                    ],
                )


class TestStage2ResolverBoundaries(unittest.TestCase):
    def _context(self, imports):
        core = importlib.import_module("aiter.ops.flydsl.compile_plan")
        backend = mock.Mock()
        backend.compile_aot = mock.Mock()
        backend.load_aot = mock.Mock()
        backend.resolve_aot = mock.Mock()
        return core.CompileContext(
            core.RocmTarget("gfx950", 256),
            core.DEFAULT_COMPILE_OP_REGISTRY,
            backend,
        )

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
            "doweight_stage2": True,
            "a_dtype": "fp4",
            "b_dtype": "fp4",
            "out_dtype": "bf16",
            "mode": "reduce",
            "accumulate": False,
            "return_per_slot": False,
            "persist": None,
            "token_num": 16,
            "routing_block_count": 128,
            "dtype_str": "bf16",
            "use_mask": False,
            "topk_ids_available": False,
            "num_experts": 0,
        }
        values.update(overrides)
        return values

    def test_persistence_threshold_explicit_modes_and_fp8_override(self) -> None:
        with _isolated_host_imports() as imports:
            plan_module = importlib.import_module("aiter.ops.flydsl.moe_compile_plan")
            context = self._context(imports)
            cases = (
                ("auto-small", {"routing_block_count": None}, 1),
                (
                    "auto-large",
                    {"token_num": 4096, "routing_block_count": None},
                    -1,
                ),
                (
                    "disabled-large",
                    {
                        "token_num": 4096,
                        "routing_block_count": 1024,
                        "persist": False,
                    },
                    4,
                ),
                ("enabled-small", {"persist": True}, -1),
                (
                    "fp8-override",
                    {
                        "a_dtype": "fp8",
                        "b_dtype": "fp8",
                        "token_num": 4096,
                        "routing_block_count": 1024,
                        "persist": True,
                    },
                    1,
                ),
                (
                    "masked-global-routing-capacity",
                    {
                        "experts": 32,
                        "topk": 4,
                        "tile_m": 64,
                        "token_num": 256,
                        "routing_block_count": None,
                        "use_mask": True,
                        "topk_ids_available": True,
                        "num_experts": 256,
                    },
                    -1,
                ),
            )
            for name, overrides, expected in cases:
                with self.subTest(name=name):
                    plan = plan_module.resolve_moe_stage2_compile_plan(
                        context=context,
                        **self._base(**overrides),
                    )
                    call = dict(plan.units[0].spec.call.arguments)
                    self.assertEqual(call["persist_m"], expected)

    def test_return_per_slot_emits_only_nonaccumulating_gemm(self) -> None:
        with _isolated_host_imports() as imports:
            plan_module = importlib.import_module("aiter.ops.flydsl.moe_compile_plan")
            context = self._context(imports)
            plan = plan_module.resolve_moe_stage2_compile_plan(
                context=context,
                **self._base(
                    mode="atomic",
                    accumulate=False,
                    return_per_slot=True,
                    doweight_stage2=False,
                ),
            )
            self.assertEqual(
                [unit.spec.op_id for unit in plan.units],
                [plan_module.MIXED_STAGE2_GEMM_OP_ID],
            )
            call = dict(plan.units[0].spec.call.arguments)
            self.assertFalse(call["accumulate"])
            self.assertFalse(call["doweight_stage2"])

    def test_rejects_ambiguous_masks_mismatched_modes_and_dtypes(self) -> None:
        with _isolated_host_imports() as imports:
            plan_module = importlib.import_module("aiter.ops.flydsl.moe_compile_plan")
            context = self._context(imports)
            invalid = (
                (
                    {"use_mask": True, "num_experts": 256},
                    "top-k-id semantics",
                ),
                (
                    {
                        "use_mask": True,
                        "topk_ids_available": True,
                        "num_experts": 0,
                    },
                    "global expert count",
                ),
                (
                    {"mode": "atomic", "accumulate": False},
                    "accumulate disagrees",
                ),
                (
                    {"out_dtype": "f32", "dtype_str": "f32"},
                    "unsupported Stage2 output dtype",
                ),
                (
                    {"a_dtype": "bf16", "b_dtype": "fp4"},
                    "unsupported Stage2 dtype",
                ),
                (
                    {
                        "a_dtype": "bf16",
                        "b_dtype": "int4",
                        "enable_bias": True,
                    },
                    "does not support bias",
                ),
            )
            for overrides, message in invalid:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        plan_module.resolve_moe_stage2_compile_plan(
                            context=context,
                            **self._base(**overrides),
                        )


if __name__ == "__main__":
    unittest.main()
