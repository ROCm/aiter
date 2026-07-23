# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for concrete FlyDSL sorting units and aggregate MoE plans."""

from __future__ import annotations

from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
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
    _run_sorting,
)

_GOLDEN = _TEST_DIR / "data" / "moe_compile_requests_gfx950.json"
_CASES = (
    (
        "sorting.oneshot.unmasked",
        {
            "max_tokens": 8,
            "num_experts": 256,
            "topk": 8,
            "has_mask": False,
        },
        "aiter.flydsl.moe.sorting.oneshot.v1",
    ),
    (
        "sorting.oneshot.masked",
        {
            "max_tokens": 8,
            "num_experts": 256,
            "topk": 8,
            "has_mask": True,
        },
        "aiter.flydsl.moe.sorting.oneshot.v1",
    ),
    (
        "sorting.multiphase.p0v2.unmasked.e384",
        {
            "max_tokens": 128,
            "num_experts": 384,
            "topk": 8,
            "has_mask": False,
        },
        "aiter.flydsl.moe.sorting.multiphase.p0v2_p23.v1",
    ),
    (
        "sorting.multiphase.4k.masked",
        {
            "max_tokens": 4096,
            "num_experts": 256,
            "topk": 8,
            "has_mask": True,
        },
        "aiter.flydsl.moe.sorting.multiphase.k4_fused.v1",
    ),
)

_ABI_NAMES = {
    "aiter.flydsl.moe.sorting.oneshot.v1": """
        topk_ids_tensor topk_weights_tensor sorted_token_ids sorted_weights_out
        sorted_expert_ids num_valid_ids_out moe_buf expert_mask_tensor
        i32_tokens i32_moe_buf_elems n_grid_blocks stream
    """.split(),
    "aiter.flydsl.moe.sorting.multiphase.p0v2_p23.v1": """
        topk_ids workspace topk_weights_tensor sorted_token_ids
        sorted_weights_out sorted_expert_ids num_valid_ids_out moe_buf
        expert_mask_tensor i32_tokens i32_mesh_stride i32_mesh_size
        i32_moe_buf_elems n_grid_p23 stream
    """.split(),
    "aiter.flydsl.moe.sorting.multiphase.k4_fused.v1": """
        topk_ids workspace topk_weights_tensor sorted_token_ids
        sorted_weights_out sorted_expert_ids num_valid_ids_out moe_buf
        expert_mask_tensor i32_tokens i32_mesh_stride i32_mesh_size
        i32_moe_buf_elems i32_ws_total i32_p0_niters n_grid_k1 n_grid_k2
        n_grid_p23 stream
    """.split(),
}


def _golden_selected_requests():
    golden = json.loads(_GOLDEN.read_text())
    selected = {}
    for request in golden["requests"]:
        scenario = request["trigger"]["scenario"]
        if scenario.startswith("sorting.") and request["trigger"]["launchers"]:
            selected[scenario] = request
    return selected


class TestSortingGoldenParity(unittest.TestCase):
    def test_concrete_units_match_selected_aot1_requests_and_exact_abis(self) -> None:
        expected_requests = _golden_selected_requests()
        recorder = _RequestRecorder()
        observed = {}

        with _recording_environment(), ExitStack() as stack:
            _install_cuda_boundary_mocks(stack, recorder)
            with _isolated_host_imports() as imports:
                _install_boundary_mocks(stack, imports, recorder)
                plan_module = importlib.import_module(
                    "aiter.ops.flydsl.moe_compile_plan"
                )
                for scenario, values, expected_op_id in _CASES:
                    _clear_scenario_caches(imports)
                    case = plan_module.MoeSortingCompileCase(**values)
                    plan = plan_module.resolve_moe_sorting_compile_plan(
                        case,
                        context=recorder.compile_context,
                    )
                    self.assertEqual(len(plan.units), 1)
                    unit = plan.units[0]
                    self.assertEqual(unit.spec.op_id, expected_op_id)
                    self.assertEqual(
                        [argument.name for argument in unit.signature.arguments],
                        _ABI_NAMES[expected_op_id],
                    )
                    self.assertEqual(
                        unit.signature,
                        plan_module.sorting_abi(expected_op_id),
                    )
                    bound = dict(unit.spec.call.arguments)
                    bound.pop("compile_target")
                    self.assertEqual(
                        bound,
                        expected_requests[scenario]["kwargs"],
                    )
                    with recorder.scenario(scenario):
                        recorder.backend.compile_aot(
                            unit,
                            context=recorder.compile_context,
                        )
                    observed[scenario] = recorder.requests[-1]

        self.assertEqual(set(observed), set(expected_requests))
        for scenario, actual in observed.items():
            with self.subTest(scenario=scenario):
                expected = expected_requests[scenario]
                self.assertEqual(actual["builder"], expected["builder"])
                self.assertEqual(actual["kwargs"], expected["kwargs"])
                self.assertEqual(
                    actual["trigger"]["launchers"],
                    expected["trigger"]["launchers"],
                )

    def test_runtime_resolves_provider_and_backend_once(self) -> None:
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
                        "resolve_moe_sorting_compile_plan",
                        wraps=plan_module.resolve_moe_sorting_compile_plan,
                    ) as resolver_spy,
                    mock.patch.object(
                        recorder.backend,
                        "resolve_aot",
                        wraps=recorder.backend.resolve_aot,
                    ) as backend_spy,
                    fake_mode,
                    recorder.scenario("sorting.oneshot.unmasked"),
                ):
                    _run_sorting(imports, tokens=8, masked=False)

        self.assertEqual(resolver_spy.call_count, 1)
        self.assertEqual(backend_spy.call_count, 1)
        self.assertEqual(
            backend_spy.call_args.args[0].spec.op_id,
            "aiter.flydsl.moe.sorting.oneshot.v1",
        )

    def test_on_mode_executes_provider_selected_sorting_role(self) -> None:
        recorder = _RequestRecorder()
        fake_mode = FakeTensorMode()
        with _recording_environment(), ExitStack() as stack:
            _install_cuda_boundary_mocks(stack, recorder)
            with _isolated_host_imports() as imports:
                _install_boundary_mocks(stack, imports, recorder)
                with (
                    mock.patch.dict(
                        os.environ,
                        {"AITER_FLYDSL_OPERATION_PLAN": "on"},
                    ),
                    mock.patch.object(
                        imports.sorting_wrapper._SORTING_RUNTIME_ADAPTERS,
                        "lookup",
                        wraps=(
                            imports.sorting_wrapper._SORTING_RUNTIME_ADAPTERS.lookup
                        ),
                    ) as lookup,
                    fake_mode,
                    recorder.scenario("sorting.oneshot.unmasked"),
                ):
                    _run_sorting(imports, tokens=8, masked=False)
                self.assertEqual(
                    [call.args[0] for call in lookup.call_args_list],
                    ["moe.sorting.oneshot"],
                )


class TestSortingPureSemantics(unittest.TestCase):
    def test_architecture_token_and_k4_boundaries(self) -> None:
        with _isolated_host_imports() as imports:
            kernel = imports.sorting_kernel

            gfx94 = kernel.resolve_moe_sorting_specialization(
                arch="gfx942",
                max_tokens=9,
                num_experts=512,
                topk=8,
                unit_size=32,
                has_mask=False,
            )
            gfx95 = kernel.resolve_moe_sorting_specialization(
                arch="gfx950",
                max_tokens=9,
                num_experts=512,
                topk=8,
                unit_size=32,
                has_mask=False,
            )
            self.assertEqual(gfx94.sub_tokens, 8)
            self.assertEqual(gfx95.sub_tokens, 32)
            self.assertEqual(gfx94.path, kernel.SORTING_PATH_P0V2_P23)
            self.assertEqual(gfx95.path, kernel.SORTING_PATH_ONESHOT)

            boundaries = (
                (16, kernel.SORTING_PATH_ONESHOT),
                (17, kernel.SORTING_PATH_P0V2_P23),
                (2048, kernel.SORTING_PATH_P0V2_P23),
                (2049, kernel.SORTING_PATH_4K_FUSED),
            )
            for tokens, expected_path in boundaries:
                with self.subTest(tokens=tokens):
                    specialization = kernel.resolve_moe_sorting_specialization(
                        arch="gfx950",
                        max_tokens=tokens,
                        num_experts=256,
                        topk=8,
                        unit_size=32,
                        has_mask=False,
                    )
                    self.assertEqual(specialization.path, expected_path)

            for tokens, expected_block in ((128, 512), (8192, 512), (8193, 256)):
                with self.subTest(tokens=tokens, expected_block=expected_block):
                    specialization = kernel.resolve_moe_sorting_specialization(
                        arch="gfx950",
                        max_tokens=tokens,
                        num_experts=384,
                        topk=8,
                        unit_size=32,
                        has_mask=True,
                    )
                    self.assertEqual(specialization.k4_block, expected_block)

    def test_rejects_missing_dynamic_and_disagreeing_metadata(self) -> None:
        class DynamicTokens:
            @staticmethod
            def item():
                raise AssertionError("resolver must not call item()")

        with _isolated_host_imports() as imports:
            plan_module = importlib.import_module("aiter.ops.flydsl.moe_compile_plan")
            kernel = imports.sorting_kernel
            with self.assertRaises(TypeError):
                plan_module.MoeSortingCompileCase(  # type: ignore[call-arg]
                    max_tokens=8,
                    num_experts=256,
                    topk=8,
                )
            invalid = (
                (
                    {
                        "arch": "",
                        "max_tokens": 8,
                        "num_experts": 256,
                        "topk": 8,
                        "unit_size": 32,
                        "has_mask": False,
                    },
                    "architecture",
                ),
                (
                    {
                        "arch": "gfx950",
                        "max_tokens": DynamicTokens(),
                        "num_experts": 256,
                        "topk": 8,
                        "unit_size": 32,
                        "has_mask": False,
                    },
                    "max_tokens",
                ),
                (
                    {
                        "arch": "gfx950",
                        "max_tokens": 8,
                        "num_experts": 256,
                        "topk": 8,
                        "unit_size": 32,
                        "has_mask": None,
                    },
                    "has_mask",
                ),
                (
                    {
                        "arch": "gfx950",
                        "max_tokens": 8,
                        "num_experts": 256,
                        "topk": 8,
                        "unit_size": 32,
                        "has_mask": False,
                        "path": kernel.SORTING_PATH_4K_FUSED,
                    },
                    "disagrees",
                ),
                (
                    {
                        "arch": "gfx950",
                        "max_tokens": 128,
                        "num_experts": 384,
                        "topk": 8,
                        "unit_size": 32,
                        "has_mask": False,
                        "k4_block": 256,
                    },
                    "k4_block",
                ),
            )
            for kwargs, message in invalid:
                with self.subTest(message=message):
                    with self.assertRaisesRegex((TypeError, ValueError), message):
                        kernel.resolve_moe_sorting_specialization(**kwargs)

    def test_resolution_is_deterministic_for_mask_variants(self) -> None:
        recorder = _RequestRecorder()
        with _recording_environment(), ExitStack() as stack:
            _install_cuda_boundary_mocks(stack, recorder)
            with _isolated_host_imports() as imports:
                _install_boundary_mocks(stack, imports, recorder)
                plan_module = importlib.import_module(
                    "aiter.ops.flydsl.moe_compile_plan"
                )
                for has_mask in (False, True):
                    case = plan_module.MoeSortingCompileCase(
                        max_tokens=128,
                        num_experts=384,
                        topk=8,
                        has_mask=has_mask,
                    )
                    first = plan_module.resolve_moe_sorting_compile_plan(
                        case,
                        context=recorder.compile_context,
                    )
                    second = plan_module.resolve_moe_sorting_compile_plan(
                        case,
                        context=recorder.compile_context,
                    )
                    self.assertEqual(first, second)
                    self.assertEqual(
                        dict(first.units[0].spec.call.arguments)["has_mask"],
                        has_mask,
                    )


class TestAggregateMoeCompilePlan(unittest.TestCase):
    @staticmethod
    def _stage1(plan_module, context):
        return plan_module.resolve_moe_stage1_compile_plan(
            context=context,
            model_dim=7168,
            inter_dim=2048,
            experts=256,
            topk=8,
            tile_m=32,
            tile_n=128,
            tile_k=256,
            doweight_stage1=False,
            a_dtype="fp4",
            b_dtype="fp4",
            out_dtype="bf16",
        )

    @staticmethod
    def _stage2(plan_module, context):
        return plan_module.resolve_moe_stage2_compile_plan(
            context=context,
            model_dim=7168,
            inter_dim=2048,
            experts=256,
            topk=8,
            tile_m=32,
            tile_n=128,
            tile_k=256,
            doweight_stage2=True,
            a_dtype="fp4",
            b_dtype="fp4",
            out_dtype="bf16",
            mode="reduce",
            accumulate=False,
            return_per_slot=False,
            persist=None,
            token_num=128,
            routing_block_count=None,
            dtype_str="bf16",
            use_mask=False,
            topk_ids_available=False,
            num_experts=0,
        )

    def test_orders_optional_sorting_before_bound_stage_plans(self) -> None:
        recorder = _RequestRecorder()
        with _recording_environment(), ExitStack() as stack:
            _install_cuda_boundary_mocks(stack, recorder)
            with _isolated_host_imports() as imports:
                _install_boundary_mocks(stack, imports, recorder)
                plan_module = importlib.import_module(
                    "aiter.ops.flydsl.moe_compile_plan"
                )
                stage1 = self._stage1(plan_module, recorder.compile_context)
                stage2 = self._stage2(plan_module, recorder.compile_context)
                sorting = plan_module.MoeSortingCompileCase(
                    max_tokens=128,
                    num_experts=256,
                    topk=8,
                    has_mask=False,
                )
                enabled = plan_module.resolve_moe_compile_plan(
                    plan_module.MoeCompilePlanCase(
                        sorting=sorting,
                        stage1=stage1,
                        stage2=stage2,
                    ),
                    context=recorder.compile_context,
                )
                disabled = plan_module.resolve_moe_compile_plan(
                    plan_module.MoeCompilePlanCase(
                        sorting=None,
                        stage1=stage1,
                        stage2=stage2,
                    ),
                    context=recorder.compile_context,
                )

                self.assertEqual(
                    [unit.spec.op_id for unit in enabled.units],
                    [
                        plan_module.SORTING_P0V2_P23_OP_ID,
                        plan_module.MIXED_STAGE1_GEMM_OP_ID,
                        plan_module.MIXED_STAGE2_GEMM_OP_ID,
                        plan_module.PLAIN_REDUCTION_OP_ID,
                    ],
                )
                self.assertEqual(disabled.units, stage1.units + stage2.units)
                self.assertEqual(
                    enabled,
                    plan_module.resolve_moe_compile_plan(
                        plan_module.MoeCompilePlanCase(
                            sorting=sorting,
                            stage1=stage1,
                            stage2=stage2,
                        ),
                        context=recorder.compile_context,
                    ),
                )
                with self.assertRaises(TypeError):
                    plan_module.MoeCompilePlanCase(  # type: ignore[call-arg]
                        stage1=stage1,
                        stage2=stage2,
                    )

                core = importlib.import_module("aiter.ops.flydsl.compile_plan")
                other_context = core.CompileContext(
                    core.RocmTarget("gfx942", 304),
                    recorder.compile_context.registry,
                    recorder.backend,
                )
                with self.assertRaisesRegex(ValueError, "targets"):
                    plan_module.resolve_moe_compile_plan(
                        plan_module.MoeCompilePlanCase(
                            sorting=None,
                            stage1=stage1,
                            stage2=stage2,
                        ),
                        context=other_context,
                    )

    def test_operation_case_composes_compile_and_execution_projections(self) -> None:
        recorder = _RequestRecorder()
        with _recording_environment(), ExitStack() as stack:
            _install_cuda_boundary_mocks(stack, recorder)
            with _isolated_host_imports() as imports:
                _install_boundary_mocks(stack, imports, recorder)
                plan_module = importlib.import_module(
                    "aiter.ops.flydsl.moe_compile_plan"
                )
                stage1_values = {
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
                stage2_values = {
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
                    "token_num": 128,
                    "routing_block_count": None,
                    "dtype_str": "bf16",
                    "use_mask": False,
                    "topk_ids_available": False,
                    "num_experts": 0,
                }
                case = plan_module.MoeOperationCase(
                    sorting=plan_module.MoeSortingCompileCase(128, 256, 8, False),
                    stage1=plan_module.MoeStage1OperationCase.from_kwargs(
                        stage1_values
                    ),
                    stage2=plan_module.normalize_moe_stage2_operation_case(
                        stage2_values
                    ),
                )
                plan = plan_module.resolve_moe_operation_plan(
                    case,
                    context=recorder.compile_context,
                )
                self.assertEqual(
                    [unit.spec.op_id for unit in plan.compile_projection().units],
                    [
                        plan_module.SORTING_P0V2_P23_OP_ID,
                        plan_module.MIXED_STAGE1_GEMM_OP_ID,
                        plan_module.MIXED_STAGE2_GEMM_OP_ID,
                        plan_module.PLAIN_REDUCTION_OP_ID,
                    ],
                )
                self.assertEqual(
                    [node.node_id for node in plan.nodes],
                    [
                        "sorting.sorting",
                        "stage1.gemm",
                        "stage2.gemm",
                        "stage2.reduction",
                    ],
                )
                runtime = importlib.import_module("aiter.ops.flydsl.operation_runtime")
                with recorder.scenario("aggregate.operation.execution"):
                    steps = runtime.resolve_execution_steps(
                        plan,
                        context=recorder.compile_context,
                    )
                self.assertEqual(len(steps), 4)


class TestDirectSortingAotBoundary(unittest.TestCase):
    def test_direct_case_never_enters_runtime_gpu_tensor_or_fake_boundaries(
        self,
    ) -> None:
        recorder = _RequestRecorder()
        with _recording_environment(), ExitStack() as stack:
            _install_cuda_boundary_mocks(stack, recorder)
            with _isolated_host_imports() as imports:
                _install_boundary_mocks(stack, imports, recorder)
                plan_module = importlib.import_module(
                    "aiter.ops.flydsl.moe_compile_plan"
                )
                compiled = []

                def record_compile(unit, *, context):
                    compiled.append((unit, context))
                    return mock.Mock(unit=unit)

                forbidden = mock.Mock(
                    side_effect=AssertionError("forbidden sorting AOT boundary")
                )
                stack.enter_context(
                    mock.patch.object(imports.aot_moe, "compile_aot", record_compile)
                )
                for owner, attribute in (
                    (imports.sorting_wrapper, "flydsl_moe_sorting_fwd"),
                    (imports.sorting_kernel, "moe_sorting_flydsl"),
                    (torch, "empty"),
                    (torch, "empty_like"),
                    (torch, "empty_strided"),
                    (torch, "full"),
                    (torch, "ones"),
                    (torch, "tensor"),
                    (torch, "zeros"),
                    (torch.cuda, "current_stream"),
                    (torch.cuda, "get_device_properties"),
                    (torch.cuda, "current_device"),
                    (torch.Tensor, "item"),
                ):
                    if hasattr(owner, attribute):
                        stack.enter_context(
                            mock.patch.object(owner, attribute, forbidden)
                        )
                stack.enter_context(
                    mock.patch(
                        "torch._subclasses.fake_tensor.FakeTensorMode",
                        forbidden,
                    )
                )

                case = plan_module.MoeSortingCompileCase(
                    max_tokens=128,
                    num_experts=384,
                    topk=8,
                    has_mask=False,
                )
                artifacts = imports.aot_moe.compile_moe_sorting_case(
                    case,
                    context=recorder.compile_context,
                )

                self.assertEqual(len(artifacts), 1)
                self.assertEqual(len(compiled), 1)
                self.assertEqual(
                    compiled[0][0].spec.op_id,
                    plan_module.SORTING_P0V2_P23_OP_ID,
                )
                forbidden.assert_not_called()

    def test_ordinary_stage_csv_never_infers_sorting_jobs(self) -> None:
        with _isolated_host_imports() as imports, tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "explicit-looking-stage-row.csv"
            csv_path.write_text(
                "token,model_dim,inter_dim,expert,topk,doweight_stage1,cu_num,"
                "block_m,act_type,q_type,dtype,q_dtype_w,kernelName1,kernelName2,"
                "uses_flydsl_sorting\n"
                "16,7168,2048,256,8,0,256,32,silu,per_1x32,"
                "torch.bfloat16,torch.float4_e2m1fn_x2,"
                "flydsl_moe1_afp4_wfp4_bf16_t32x128x256,"
                "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic_bnt2,1\n"
            )

            jobs = imports.aot_moe.parse_csv(str(csv_path))

            self.assertTrue(jobs)
            self.assertEqual({job["stage"] for job in jobs}, {1, 2})
            self.assertTrue(
                all("uses_flydsl_sorting" not in job for job in jobs),
            )


if __name__ == "__main__":
    unittest.main()
