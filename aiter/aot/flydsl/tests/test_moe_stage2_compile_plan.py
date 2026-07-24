# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU parity and integration tests for the callable-bound Stage2 plan."""

from __future__ import annotations

from contextlib import ExitStack
import importlib
import json
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
                            self.assertNotIn(
                                "compile_target",
                                dict(unit.spec.call.arguments),
                            )
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

    def test_runtime_passes_one_shared_decision_to_compile_plan(self) -> None:
        recorder = _RequestRecorder()
        fake_mode = FakeTensorMode()
        decisions = []
        with _recording_environment(), ExitStack() as stack:
            _install_cuda_boundary_mocks(stack, recorder)
            with _isolated_host_imports() as imports:
                _install_boundary_mocks(stack, imports, recorder)
                plan_module = importlib.import_module(
                    "aiter.ops.flydsl.moe_compile_plan"
                )
                _clear_scenario_caches(imports)
                original_decision = imports.moe.resolve_stage2_compile_decision

                def record_decision(*args, **kwargs):
                    decision = original_decision(*args, **kwargs)
                    decisions.append(decision)
                    return decision

                with (
                    mock.patch.object(
                        imports.moe,
                        "resolve_stage2_compile_decision",
                        side_effect=record_decision,
                    ) as decision_spy,
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

        self.assertEqual(decision_spy.call_count, 1)
        self.assertEqual(resolver_spy.call_count, 1)
        self.assertIs(resolver_spy.call_args.kwargs["decision"], decisions[0])
        self.assertEqual(backend_spy.call_count, 2)
        self.assertEqual(
            [call.args[0].spec.op_id for call in backend_spy.call_args_list],
            [
                "aiter.flydsl.moe.stage2.mixed_gemm.v1",
                "aiter.flydsl.moe.stage2.reduction.plain.v1",
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

    def test_decisions_cover_atomic_per_slot_plain_and_masked_layouts(self) -> None:
        with _isolated_host_imports() as imports:
            decisions = importlib.import_module(
                "aiter.ops.flydsl.moe_compile_decisions"
            )
            plans = importlib.import_module("aiter.ops.flydsl.moe_compile_plan")
            context = self._context(imports)
            cases = (
                (
                    "atomic",
                    self._base(mode="atomic", accumulate=True),
                    ("mixed", "accumulate", "none"),
                    (plans.MIXED_STAGE2_GEMM_OP_ID,),
                ),
                (
                    "return-per-slot",
                    self._base(
                        mode="atomic",
                        accumulate=False,
                        return_per_slot=True,
                    ),
                    ("mixed", "per_slot", "none"),
                    (plans.MIXED_STAGE2_GEMM_OP_ID,),
                ),
                (
                    "plain-reduction",
                    self._base(),
                    ("mixed", "reduction", "plain"),
                    (
                        plans.MIXED_STAGE2_GEMM_OP_ID,
                        plans.PLAIN_REDUCTION_OP_ID,
                    ),
                ),
                (
                    "masked-reduction",
                    self._base(
                        experts=32,
                        use_mask=True,
                        topk_ids_available=True,
                        num_experts=256,
                    ),
                    ("mixed", "reduction", "masked"),
                    (
                        plans.MIXED_STAGE2_GEMM_OP_ID,
                        plans.MASKED_REDUCTION_OP_ID,
                    ),
                ),
            )
            for name, values, expected, op_ids in cases:
                with self.subTest(name=name):
                    builder_kwargs = {
                        key: value
                        for key, value in values.items()
                        if key
                        not in {
                            "mode",
                            "accumulate",
                            "return_per_slot",
                            "persist",
                            "token_num",
                            "routing_block_count",
                            "dtype_str",
                            "use_mask",
                            "topk_ids_available",
                            "num_experts",
                        }
                    }
                    decision = decisions.resolve_stage2_compile_decision(
                        builder_kwargs,
                        mode=values["mode"],
                        accumulate=values["accumulate"],
                        return_per_slot=values["return_per_slot"],
                        persist=values["persist"],
                        token_num=values["token_num"],
                        routing_block_count=values["routing_block_count"],
                        dtype_str=values["dtype_str"],
                        use_mask=values["use_mask"],
                        topk_ids_available=values["topk_ids_available"],
                        num_experts=values["num_experts"],
                    )
                    self.assertEqual(
                        (
                            decision.primary_family,
                            decision.target_layout,
                            decision.reduction_kind,
                        ),
                        expected,
                    )
                    plan = plans.resolve_moe_stage2_compile_plan(
                        context=context,
                        decision=decision,
                        **values,
                    )
                    self.assertEqual(
                        tuple(unit.spec.op_id for unit in plan.units),
                        op_ids,
                    )
                    main = dict(plan.units[0].spec.call.arguments)
                    self.assertEqual(main["accumulate"], decision.accumulate)
                    self.assertEqual(main["persist_m"], decision.persist_m)
                    if decision.reduction_kind != "none":
                        reduction = dict(plan.units[1].spec.call.arguments)
                        self.assertEqual(
                            reduction["dtype_str"],
                            decision.reduction_dtype,
                        )
                        self.assertEqual(
                            reduction["use_mask"],
                            decision.reduction_kind == "masked",
                        )

            self.assertEqual(
                decisions.resolve_stage2_m_blocks(
                    token_num=16,
                    topk=8,
                    experts=256,
                    tile_m=32,
                    sort_block_m=64,
                    routing_block_count=10,
                ),
                20,
            )

    def test_persist_m_threshold_is_exact_at_256_blocks(self) -> None:
        with _isolated_host_imports():
            decisions = importlib.import_module(
                "aiter.ops.flydsl.moe_compile_decisions"
            )
            cases = (
                (256, None, "fp4", 1),
                (257, None, "fp4", -1),
                (256, False, "fp4", 1),
                (257, False, "fp4", 4),
                (16, True, "fp4", -1),
                (257, True, "fp8", 1),
            )
            for m_blocks, persist, a_dtype, expected in cases:
                with self.subTest(
                    m_blocks=m_blocks,
                    persist=persist,
                    a_dtype=a_dtype,
                ):
                    self.assertEqual(
                        decisions.resolve_stage2_persist_m(
                            m_blocks=m_blocks,
                            persist=persist,
                            a_dtype=a_dtype,
                        ),
                        expected,
                    )

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
