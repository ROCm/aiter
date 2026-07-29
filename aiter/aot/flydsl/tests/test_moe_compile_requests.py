# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU golden parity and runtime wiring tests for MoE compile requests."""

from __future__ import annotations

from contextlib import ExitStack
import importlib
import json
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
    _kernel_params,
    _recording_environment,
    _run_sorting,
    _run_stage1,
    _run_stage2,
    _stage_shape,
)

_GOLDEN = _TEST_DIR / "data" / "moe_compile_requests_gfx950.json"
_TARGET = ("gfx950", 256)

_STAGE1_CASES = (
    (
        "stage1.main.non_split.bias.route_weighted",
        "flydsl_moe1_afp4_wfp4_bf16_t32x128x256",
        {
            "act": "silu",
            "doweight_stage1": True,
            "enable_bias": True,
        },
    ),
    (
        "stage1.int4.splitk",
        "flydsl_moe1_abf16_wint4_bf16_t16x64x128_kb4",
        {
            "model_dim": 7168,
            "inter_dim": 256,
            "experts": 384,
            "topk": 8,
            "token_num": 16,
            "act": "silu",
        },
    ),
    (
        "stage1.splitk.fp4.silu.separated",
        "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w3_kb4_fp4",
        {"act": "silu"},
    ),
    (
        "stage1.splitk.fp8.swiglu.interleaved.bias",
        "flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui_fp8",
        {
            "act": "swiglu",
            "k_batch": 4,
            "enable_bias": True,
            "swiglu_limit": 7.0,
        },
    ),
    (
        "stage1.splitk.none.silu.interleaved",
        "flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui",
        {"act": "silu", "k_batch": 4},
    ),
)

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

_SORTING_CASES = (
    (
        "sorting.oneshot.unmasked",
        (8, 256, 8, False),
        "aiter.flydsl.moe.sorting.oneshot.v1",
    ),
    (
        "sorting.oneshot.masked",
        (8, 256, 8, True),
        "aiter.flydsl.moe.sorting.oneshot.v1",
    ),
    (
        "sorting.multiphase.p0v2.unmasked.e384",
        (128, 384, 8, False),
        "aiter.flydsl.moe.sorting.multiphase.p0v2_p23.v1",
    ),
    (
        "sorting.multiphase.4k.masked",
        (4096, 256, 8, True),
        "aiter.flydsl.moe.sorting.multiphase.k4_fused.v1",
    ),
)


def _projection(recording, scenarios):
    return {
        request["id"]: {
            "builder": request["builder"],
            "kwargs": request["kwargs"],
            "launchers": request["trigger"]["launchers"],
            "scenario": request["trigger"]["scenario"],
        }
        for request in recording["requests"]
        if request["trigger"]["scenario"] in scenarios
    }


def _stage1_metadata(imports, kernel_name, options):
    return {
        **_stage_shape(),
        **_kernel_params(imports, kernel_name, expected_stage=1),
        "doweight_stage1": False,
        "use_async_copy": True,
        **options,
    }


def _stage2_metadata(imports, kernel_name, masked, options):
    return {
        **_stage_shape(experts=32 if masked else 256),
        **_kernel_params(imports, kernel_name, expected_stage=2),
        **options,
    }


def _stage2_runtime_metadata(metadata, masked):
    mode = metadata.get("mode", "atomic")
    out_dtype = metadata["out_dtype"]
    return {
        "mode": mode,
        "accumulate": mode != "reduce",
        "return_per_slot": False,
        "persist": metadata.get("persist"),
        "token_num": metadata["token_num"],
        "routing_block_count": max(1, metadata["token_num"] * metadata["topk"]),
        "dtype_str": "bf16" if out_dtype == "bf16" else "f16",
        "use_mask": masked,
        "topk_ids_available": masked,
        "num_experts": 256 if masked else 0,
    }


class TestFactoryGoldenParity(unittest.TestCase):
    def test_stage1_stage2_sorting_and_cktile_match_aot1_golden(self):
        golden = json.loads(_GOLDEN.read_text())
        scenarios = {
            *(case[0] for case in _STAGE1_CASES),
            *(case[0] for case in _STAGE2_CASES),
            *(case[0] for case in _SORTING_CASES),
            "cktile.epilogue.silu",
            "cktile.epilogue.swiglu",
        }
        recorder = _RequestRecorder()
        observed_op_ids = {}
        observed_abis = {}

        with _recording_environment(), ExitStack() as stack:
            _install_cuda_boundary_mocks(stack, recorder)
            with _isolated_host_imports() as imports:
                _install_boundary_mocks(stack, imports, recorder)
                core = importlib.import_module("aiter.ops.flydsl.compile_request")
                factories = importlib.import_module(
                    "aiter.ops.flydsl.moe_compile_requests"
                )
                target = core.RocmTarget(*_TARGET)

                for scenario, kernel_name, options in _STAGE1_CASES:
                    metadata = _stage1_metadata(imports, kernel_name, options)
                    _clear_scenario_caches(imports)
                    requests = factories.stage1_compile_requests(
                        metadata,
                        target,
                        registry=recorder.compile_context.registry,
                    )
                    observed_op_ids[scenario] = tuple(
                        request.op_id for request in requests
                    )
                    with recorder.scenario(scenario):
                        for request in requests:
                            recorder.backend.compile_aot(
                                request,
                                context=recorder.compile_context,
                            )
                    observed_abis.update(
                        (request.op_id, request.signature) for request in requests
                    )

                for scenario, act in (
                    ("cktile.epilogue.silu", "silu"),
                    ("cktile.epilogue.swiglu", "swiglu"),
                ):
                    _clear_scenario_caches(imports)
                    metadata = {
                        "act": act,
                        "inter_dim": 2048,
                        "topk": 8,
                        "split_k": 2,
                        "post_activation_layout": "interleaved",
                        "enable_bias": False,
                    }
                    requests = factories.cktile_epilogue_compile_requests(
                        metadata,
                        target,
                        registry=recorder.compile_context.registry,
                    )
                    observed_op_ids[scenario] = tuple(
                        request.op_id for request in requests
                    )
                    with recorder.scenario(scenario):
                        for request in requests:
                            recorder.backend.compile_aot(
                                request,
                                context=recorder.compile_context,
                            )
                    observed_abis.update(
                        (request.op_id, request.signature) for request in requests
                    )

                for scenario, kernel_name, masked, options in _STAGE2_CASES:
                    metadata = _stage2_metadata(
                        imports,
                        kernel_name,
                        masked,
                        options,
                    )
                    metadata["doweight_stage2"] = not metadata.get(
                        "doweight_stage1",
                        False,
                    )
                    runtime = _stage2_runtime_metadata(metadata, masked)
                    _clear_scenario_caches(imports)
                    requests = factories.stage2_compile_requests(
                        metadata,
                        runtime,
                        target,
                        registry=recorder.compile_context.registry,
                    )
                    observed_op_ids[scenario] = tuple(
                        request.op_id for request in requests
                    )
                    with recorder.scenario(scenario):
                        for request in requests:
                            recorder.backend.compile_aot(
                                request,
                                context=recorder.compile_context,
                            )
                    observed_abis.update(
                        (request.op_id, request.signature) for request in requests
                    )

                for scenario, values, expected_op_id in _SORTING_CASES:
                    case = factories.MoeSortingCompileCase(*values)
                    _clear_scenario_caches(imports)
                    request = factories.sorting_compile_request(
                        case,
                        target,
                        registry=recorder.compile_context.registry,
                    )
                    self.assertEqual(request.op_id, expected_op_id)
                    observed_op_ids[scenario] = (request.op_id,)
                    observed_abis[request.op_id] = request.signature
                    with recorder.scenario(scenario):
                        recorder.backend.compile_aot(
                            request,
                            context=recorder.compile_context,
                        )

        actual = {"requests": recorder.requests}
        self.assertEqual(
            _projection(actual, scenarios),
            _projection(golden, scenarios),
        )
        self.assertEqual(
            observed_op_ids["stage1.splitk.fp4.silu.separated"],
            (
                "aiter.flydsl.moe.stage1.mixed_gemm.v1",
                "aiter.flydsl.moe.stage1.silu_and_mul_fq.v1",
            ),
        )
        self.assertEqual(
            observed_op_ids["stage2.reduce.masked_ep"],
            (
                "aiter.flydsl.moe.stage2.mixed_gemm.v1",
                "aiter.flydsl.moe.stage2.reduction.masked.v1",
            ),
        )
        for op_id, signature in observed_abis.items():
            with self.subTest(op_id=op_id):
                self.assertEqual(signature, factories.get_kernel_signature(op_id))
                hash(signature)

    def test_factories_are_deterministic_and_do_not_touch_tensor_or_cuda(self):
        recorder = _RequestRecorder()
        forbidden = mock.Mock(side_effect=AssertionError("forbidden AOT boundary"))
        with _recording_environment(), ExitStack() as stack:
            _install_cuda_boundary_mocks(stack, recorder)
            with _isolated_host_imports() as imports:
                _install_boundary_mocks(stack, imports, recorder)
                core = importlib.import_module("aiter.ops.flydsl.compile_request")
                factories = importlib.import_module(
                    "aiter.ops.flydsl.moe_compile_requests"
                )
                target = core.RocmTarget(*_TARGET)
                metadata = _stage1_metadata(
                    imports,
                    "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w3_kb4_fp4",
                    {"act": "silu"},
                )
                for owner, attribute in (
                    (torch, "empty"),
                    (torch, "empty_like"),
                    (torch, "empty_strided"),
                    (torch, "ones"),
                    (torch, "tensor"),
                    (torch, "zeros"),
                    (torch.cuda, "current_stream"),
                    (torch.cuda, "get_device_properties"),
                    (torch.Tensor, "item"),
                ):
                    stack.enter_context(mock.patch.object(owner, attribute, forbidden))

                first = factories.stage1_compile_requests(
                    metadata,
                    target,
                    registry=recorder.compile_context.registry,
                )
                second = factories.stage1_compile_requests(
                    metadata,
                    target,
                    registry=recorder.compile_context.registry,
                )

                self.assertEqual(first, second)
                forbidden.assert_not_called()


class TestDirectAotBoundaries(unittest.TestCase):
    def test_orchestrator_compiles_requests_without_runtime_or_tensor_paths(self):
        with _isolated_host_imports() as imports, ExitStack() as stack:
            core = importlib.import_module("aiter.ops.flydsl.compile_request")
            factories = importlib.import_module("aiter.ops.flydsl.moe_compile_requests")
            compiled = []

            def record_compile(request, *, context):
                compiled.append((request, context))
                return mock.Mock(request=request)

            forbidden = mock.Mock(side_effect=AssertionError("forbidden AOT boundary"))
            stack.enter_context(
                mock.patch.object(imports.aot_moe, "compile_aot", record_compile)
            )
            for owner, attribute in (
                (imports.moe, "flydsl_moe_stage1"),
                (imports.moe, "flydsl_moe_stage2"),
                (imports.sorting_wrapper, "flydsl_moe_sorting_fwd"),
                (imports.sorting_kernel, "moe_sorting_flydsl"),
                (torch, "empty"),
                (torch, "empty_like"),
                (torch, "empty_strided"),
                (torch, "full"),
                (torch, "ones"),
                (torch, "randn"),
                (torch, "tensor"),
                (torch, "zeros"),
                (torch.cuda, "current_stream"),
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

            stage1 = imports.aot_moe.compile_one_config(
                kernel_name="flydsl_test_stage1",
                model_dim=7168,
                inter_dim=2048,
                experts=256,
                topk=8,
                cu_num=256,
                stage=1,
                tile_m=32,
                tile_n=128,
                tile_k=256,
                doweight_stage1=False,
                a_dtype="fp4",
                b_dtype="fp4",
                out_dtype="fp4",
                act="silu",
                k_batch=4,
                waves_per_eu=3,
                b_nt=2,
                gate_mode="separated",
                use_async_copy=True,
            )
            self.assertEqual(stage1["compile_requests"], 2)
            self.assertEqual(
                tuple(request.op_id for request, _ in compiled),
                (
                    factories.MIXED_STAGE1_GEMM_OP_ID,
                    factories.FQ_ACTIVATION_OP_ID,
                ),
            )

            compiled.clear()
            stage2 = imports.aot_moe.compile_one_config(
                kernel_name="flydsl_test_stage2",
                model_dim=7168,
                inter_dim=2048,
                experts=256,
                topk=8,
                cu_num=256,
                stage=2,
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
                token_num=16,
                routing_block_count=None,
                dtype_str="bf16",
                use_mask=False,
                topk_ids_available=False,
                num_experts=0,
                waves_per_eu=None,
                use_async_copy=False,
                cu_num_mul=1,
                b_nt=2,
                xcd_swizzle=0,
                enable_bias=False,
            )
            self.assertEqual(stage2["compile_requests"], 2)
            self.assertEqual(
                tuple(request.op_id for request, _ in compiled),
                (
                    factories.MIXED_STAGE2_GEMM_OP_ID,
                    factories.PLAIN_REDUCTION_OP_ID,
                ),
            )

            compiled.clear()
            context = imports.aot_backend.create_compile_context(
                core.RocmTarget(*_TARGET)
            )
            (artifact,) = imports.aot_moe.compile_moe_sorting_case(
                factories.MoeSortingCompileCase(128, 384, 8, False),
                context=context,
            )
            self.assertEqual(
                artifact.request.op_id,
                factories.SORTING_P0V2_P23_OP_ID,
            )
            forbidden.assert_not_called()


class TestRuntimeUsesSharedDecisions(unittest.TestCase):
    def test_stage1_and_stage2_pass_one_decision_to_request_factory(self):
        recorder = _RequestRecorder()
        fake_mode = FakeTensorMode()
        with _recording_environment(), ExitStack() as stack:
            _install_cuda_boundary_mocks(stack, recorder)
            with _isolated_host_imports() as imports:
                _install_boundary_mocks(stack, imports, recorder)
                factories = importlib.import_module(
                    "aiter.ops.flydsl.moe_compile_requests"
                )

                for stage, run, decision_name, factory_name, expected_calls in (
                    (
                        "stage1",
                        lambda: _run_stage1(
                            imports,
                            "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w3_kb4_fp4",
                        ),
                        "resolve_stage1_compile_decision",
                        "stage1_compile_requests",
                        2,
                    ),
                    (
                        "stage2",
                        lambda: _run_stage2(
                            imports,
                            "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce_bnt2",
                        ),
                        "resolve_stage2_compile_decision",
                        "stage2_compile_requests",
                        2,
                    ),
                ):
                    _clear_scenario_caches(imports)
                    decisions = []
                    original_decision = getattr(imports.moe, decision_name)

                    def record_decision(*args, _original=original_decision, **kwargs):
                        decision = _original(*args, **kwargs)
                        decisions.append(decision)
                        return decision

                    with (
                        mock.patch.object(
                            imports.moe,
                            decision_name,
                            side_effect=record_decision,
                        ) as decision_spy,
                        mock.patch.object(
                            factories,
                            factory_name,
                            wraps=getattr(factories, factory_name),
                        ) as factory_spy,
                        mock.patch.object(
                            recorder.backend,
                            "resolve_aot",
                            wraps=recorder.backend.resolve_aot,
                        ) as backend_spy,
                        fake_mode,
                        recorder.scenario(f"runtime.{stage}.compile_requests"),
                    ):
                        run()

                    self.assertEqual(decision_spy.call_count, 1)
                    self.assertEqual(factory_spy.call_count, 1)
                    self.assertIs(
                        factory_spy.call_args.kwargs["decision"],
                        decisions[0],
                    )
                    self.assertEqual(backend_spy.call_count, expected_calls)

    def test_sorting_shares_one_specialization_with_request_and_launch(self):
        recorder = _RequestRecorder()
        fake_mode = FakeTensorMode()
        with _recording_environment(), ExitStack() as stack:
            _install_cuda_boundary_mocks(stack, recorder)
            with _isolated_host_imports() as imports:
                _install_boundary_mocks(stack, imports, recorder)
                factories = importlib.import_module(
                    "aiter.ops.flydsl.moe_compile_requests"
                )
                _clear_scenario_caches(imports)
                with (
                    mock.patch.object(
                        imports.sorting_kernel,
                        "resolve_moe_sorting_specialization",
                        wraps=imports.sorting_kernel.resolve_moe_sorting_specialization,
                    ) as specialization_spy,
                    mock.patch.object(
                        factories,
                        "sorting_compile_request",
                        wraps=factories.sorting_compile_request,
                    ) as factory_spy,
                    mock.patch.object(
                        imports.sorting_kernel,
                        "moe_sorting_flydsl",
                        wraps=imports.sorting_kernel.moe_sorting_flydsl,
                    ) as launch_spy,
                    fake_mode,
                    recorder.scenario("sorting.oneshot.unmasked"),
                ):
                    _run_sorting(imports, tokens=8, masked=False)

                self.assertEqual(specialization_spy.call_count, 1)
                self.assertEqual(factory_spy.call_count, 1)
                self.assertIs(
                    factory_spy.call_args.kwargs["specialization"],
                    launch_spy.call_args.kwargs["specialization"],
                )


class TestPureDecisionBoundaries(unittest.TestCase):
    def test_stage2_m_blocks_persistence_and_sorting_boundaries(self):
        with _isolated_host_imports() as imports:
            decisions = importlib.import_module(
                "aiter.ops.flydsl.moe_compile_decisions"
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
            for m_blocks, persist, dtype, expected in (
                (256, None, "fp4", 1),
                (257, None, "fp4", -1),
                (257, False, "fp4", 4),
                (16, True, "fp4", -1),
                (257, True, "fp8", 1),
            ):
                with self.subTest(m_blocks=m_blocks, persist=persist, dtype=dtype):
                    self.assertEqual(
                        decisions.resolve_stage2_persist_m(
                            m_blocks=m_blocks,
                            persist=persist,
                            a_dtype=dtype,
                        ),
                        expected,
                    )

            kernel = imports.sorting_kernel
            for tokens, expected_path in (
                (16, kernel.SORTING_PATH_ONESHOT),
                (17, kernel.SORTING_PATH_P0V2_P23),
                (2048, kernel.SORTING_PATH_P0V2_P23),
                (2049, kernel.SORTING_PATH_4K_FUSED),
            ):
                specialization = kernel.resolve_moe_sorting_specialization(
                    arch="gfx950",
                    max_tokens=tokens,
                    num_experts=256,
                    topk=8,
                    unit_size=32,
                    has_mask=False,
                )
                self.assertEqual(specialization.path, expected_path)

    def test_upstream_mxfp4_and_local_tokens_enter_request_identity(self):
        with _isolated_host_imports() as imports:
            core = importlib.import_module("aiter.ops.flydsl.compile_request")
            factories = importlib.import_module("aiter.ops.flydsl.moe_compile_requests")
            target = core.RocmTarget(*_TARGET)

            stage1_metadata = _stage1_metadata(
                imports,
                "flydsl_moe1_afp4_wfp4_bf16_t32x128x256",
                {
                    "a_dtype": "bf16",
                    "b_dtype": "mxfp4",
                    "act": "situv2",
                    "situ_beta": 1.5,
                    "situ_linear_beta": 0.75,
                },
            )
            (stage1_request,) = factories.stage1_compile_requests(
                stage1_metadata,
                target,
            )
            stage1_kwargs = stage1_request.as_kwargs()
            self.assertEqual(stage1_kwargs["situ_beta"], 1.5)
            self.assertEqual(stage1_kwargs["situ_linear_beta"], 0.75)
            self.assertNotIn(
                "f32_swiglu_limit",
                tuple(argument.name for argument in stage1_request.signature.arguments),
            )

            stage2_metadata = _stage2_metadata(
                imports,
                "flydsl_moe2_afp8_wfp4_bf16_t32x128x256_atomic_bnt2",
                False,
                {"a_dtype": "bf16", "b_dtype": "mxfp4"},
            )
            stage2_metadata["doweight_stage2"] = True
            (stage2_request,) = factories.stage2_compile_requests(
                stage2_metadata,
                _stage2_runtime_metadata(stage2_metadata, False),
                target,
            )
            self.assertEqual(stage2_request.op_id, factories.MIXED_STAGE2_GEMM_OP_ID)

            plain_case = factories.MoeSortingCompileCase(8, 256, 8, False)
            local_case = factories.MoeSortingCompileCase(
                8,
                256,
                8,
                False,
                has_local_tokens=True,
            )
            plain_request = factories.sorting_compile_request(plain_case, target)
            local_request = factories.sorting_compile_request(local_case, target)
            self.assertFalse(plain_request.as_kwargs()["has_local_tokens"])
            self.assertTrue(local_request.as_kwargs()["has_local_tokens"])
            self.assertNotEqual(plain_request, local_request)
            self.assertIn(
                "local_tokens_tensor",
                tuple(argument.name for argument in local_request.signature.arguments),
            )

    def test_invalid_decisions_and_explicit_sorting_csv_boundary(self):
        with _isolated_host_imports() as imports, tempfile.TemporaryDirectory() as tmp:
            decisions = importlib.import_module(
                "aiter.ops.flydsl.moe_compile_decisions"
            )
            with self.assertRaisesRegex(ValueError, "unsupported Stage1 dtype"):
                decisions.resolve_stage1_compile_decision(
                    {
                        "a_dtype": "fp4",
                        "b_dtype": "bf16",
                        "out_dtype": "bf16",
                    }
                )

            csv_path = Path(tmp) / "stage-row.csv"
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
            self.assertEqual({job["stage"] for job in jobs}, {1, 2})
            self.assertTrue(all("uses_flydsl_sorting" not in job for job in jobs))


if __name__ == "__main__":
    unittest.main()
