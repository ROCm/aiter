# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for the Aiter FlyDSL AOT compatibility backend."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import FrozenInstanceError
import importlib
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import flydsl.expr as fx
import torch

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from moe_compile_recorder import _isolated_host_imports  # noqa: E402

_OP_ID = "aiter.flydsl.test.aot_backend.v1"


def _all_kinds_launcher(
    pointer: fx.Pointer,
    rows: fx.Int32,
    scale: fx.Float32,
    tensor: fx.Tensor,
    stream: fx.Stream,
) -> None:
    raise AssertionError("the fake launcher body must not execute")


def _tensor_launcher(tensor: fx.Tensor, stream: fx.Stream) -> None:
    raise AssertionError("the fake launcher body must not execute")


def _mixed_stage2_launcher(
    arg_out: fx.Pointer,
    arg_x: fx.Pointer,
    arg_w: fx.Pointer,
    arg_scale_x: fx.Pointer,
    arg_scale_w: fx.Pointer,
    arg_sorted_token_ids: fx.Pointer,
    arg_expert_ids: fx.Pointer,
    arg_sorted_weights: fx.Pointer,
    arg_num_valid_ids: fx.Pointer,
    arg_bias: fx.Pointer,
    i32_tokens_in: fx.Int32,
    i32_n_in: fx.Int32,
    i32_k_in: fx.Int32,
    i32_size_expert_ids_in: fx.Int32,
    stream: fx.Stream,
) -> None:
    raise AssertionError("the fake launcher body must not execute")


def _reduction_launcher(
    X: fx.Pointer,
    Y: fx.Pointer,
    expert_mask: fx.Pointer,
    topk_ids: fx.Pointer,
    i32_m_tokens: fx.Int32,
    stream: fx.Stream,
) -> None:
    raise AssertionError("the fake launcher body must not execute")


class _FakeLauncher:
    def __init__(self, function, *, miss: bool = False) -> None:
        self.func = function
        self.miss = miss
        self.calls = []

    def __call__(self, *args):
        environment = {
            name: os.environ.get(name)
            for name in (
                "ARCH",
                "FLYDSL_GPU_ARCH",
                "CU_NUM",
                "COMPILE_ONLY",
                "FLYDSL_RUNTIME_ENABLE_CACHE",
                "FLYDSL_RUNTIME_RUN_ONLY",
            )
        }
        self.calls.append((args, environment))
        if self.miss and environment["FLYDSL_RUNTIME_RUN_ONLY"] == "1":
            raise RuntimeError(
                "FLYDSL_RUNTIME_RUN_ONLY=1 but no usable AOT cache: "
                "synthetic FlyDSL cache miss"
            )


def _signature(core):
    return core.KernelSignature(
        (
            core.SignatureArg("pointer", core.ArgumentKind.POINTER, "u8"),
            core.SignatureArg("rows", core.ArgumentKind.SCALAR, "i32"),
            core.SignatureArg("scale", core.ArgumentKind.SCALAR, "f32"),
            core.SignatureArg(
                "tensor",
                core.ArgumentKind.TENSOR,
                "bf16",
                (None, None),
                (None, 1),
            ),
            core.SignatureArg("stream", core.ArgumentKind.STREAM),
        )
    )


def _unit_context(
    core,
    backend_module,
    launcher,
    signature=None,
    *,
    op_id=_OP_ID,
):
    registry = core.CompileOpRegistry()
    builder_environments = []

    def builder():
        builder_environments.append(
            (
                os.environ.get("ARCH"),
                os.environ.get("FLYDSL_GPU_ARCH"),
                os.environ.get("CU_NUM"),
            )
        )
        return launcher

    registry.register(op_id)(builder)
    target = core.RocmTarget("gfx950", 256)
    backend = backend_module.AotBackend()
    context = core.CompileContext(target, registry, backend)
    unit = registry.make_unit(
        op_id,
        target=target,
        signature=signature if signature is not None else _signature(core),
    )
    return unit, context, builder_environments


class TestContextsAndMaterialization(unittest.TestCase):
    def test_contexts_are_separate_immutable_generic_values(self) -> None:
        with _isolated_host_imports() as imports:
            core = importlib.import_module("aiter.ops.flydsl.compile_plan")
            launch_module = importlib.import_module("aiter.ops.flydsl.launch_context")
            launcher = _FakeLauncher(_all_kinds_launcher)
            _, context, _ = _unit_context(core, imports.aot_backend, launcher)
            launch_context = launch_module.LaunchContext(stream=123)

            self.assertEqual(context.target, core.RocmTarget("gfx950", 256))
            self.assertEqual(launch_context.stream, 123)
            self.assertFalse(hasattr(context, "stream"))
            self.assertFalse(hasattr(launch_context, "target"))
            for value, field in (
                (context, "target"),
                (launch_context, "stream"),
            ):
                with self.assertRaises(FrozenInstanceError):
                    setattr(value, field, None)

    def test_signature_materializes_pointer_scalars_stream_and_tensor_metadata(
        self,
    ) -> None:
        with _isolated_host_imports() as imports:
            core = importlib.import_module("aiter.ops.flydsl.compile_plan")
            launcher = _FakeLauncher(_all_kinds_launcher)
            unit, _, _ = _unit_context(core, imports.aot_backend, launcher)

            args = imports.aot_backend._materialize_compile_args(unit, launcher)

            self.assertEqual(
                [type(value).__name__ for value in args],
                [
                    "PointerJitArg",
                    "Int32",
                    "Float32",
                    "TorchTensorJitArg",
                    "Stream",
                ],
            )
            self.assertIsNone(args[0].pointer.value)
            self.assertEqual(args[1].value, 0)
            self.assertEqual(args[2].value, 0.0)
            self.assertIsNone(args[4].value)
            self.assertEqual(args[3].dtype, torch.bfloat16)
            self.assertEqual(args[3].shape, (2, 2))
            self.assertEqual(args[3].strides, (2, 1))
            self.assertEqual(args[3].torch_tensor.data_ptr(), 0)
            self.assertFalse(isinstance(args[3].torch_tensor, torch.Tensor))

    def test_runtime_context_queries_device_once_and_captures_strict_mode(
        self,
    ) -> None:
        with _isolated_host_imports() as imports:
            properties = mock.Mock(
                gcnArchName="gfx950:sramecc+", multi_processor_count=256
            )
            with (
                mock.patch.object(
                    torch.cuda,
                    "get_device_properties",
                    return_value=properties,
                ) as properties_spy,
                mock.patch.dict(
                    os.environ,
                    {"FLYDSL_RUNTIME_RUN_ONLY": "1"},
                ),
            ):
                context = imports.aot_backend.create_runtime_compile_context(3)

            properties_spy.assert_called_once_with(3)
            self.assertEqual(context.target.arch, "gfx950")
            self.assertEqual(context.target.cu_count, 256)
            self.assertTrue(context.backend.strict_runtime)


class TestCompileAndStrictLoad(unittest.TestCase):
    _ENV_NAMES = (
        "ARCH",
        "FLYDSL_GPU_ARCH",
        "CU_NUM",
        "COMPILE_ONLY",
        "FLYDSL_RUNTIME_ENABLE_CACHE",
        "FLYDSL_RUNTIME_RUN_ONLY",
    )

    def test_compile_and_strict_load_force_modes_and_restore_environment(
        self,
    ) -> None:
        with _isolated_host_imports() as imports:
            core = importlib.import_module("aiter.ops.flydsl.compile_plan")
            before = {name: os.environ.get(name) for name in self._ENV_NAMES}

            compile_launcher = _FakeLauncher(_all_kinds_launcher)
            unit, context, builder_environments = _unit_context(
                core, imports.aot_backend, compile_launcher
            )
            compiled = imports.aot_backend.compile_aot(unit, context=context)

            self.assertFalse(compiled.loaded)
            self.assertEqual(builder_environments, [("gfx950", "gfx950", "256")])
            self.assertEqual(
                compile_launcher.calls[0][1],
                {
                    "ARCH": "gfx950",
                    "FLYDSL_GPU_ARCH": "gfx950",
                    "CU_NUM": "256",
                    "COMPILE_ONLY": "1",
                    "FLYDSL_RUNTIME_ENABLE_CACHE": "1",
                    "FLYDSL_RUNTIME_RUN_ONLY": "0",
                },
            )
            self.assertEqual(
                {name: os.environ.get(name) for name in self._ENV_NAMES},
                before,
            )

            load_launcher = _FakeLauncher(_all_kinds_launcher)
            unit, context, _ = _unit_context(core, imports.aot_backend, load_launcher)
            loaded = imports.aot_backend.load_aot(
                unit,
                context=context,
                strict=True,
            )
            self.assertTrue(loaded.loaded)
            self.assertEqual(
                load_launcher.calls[0][1]["FLYDSL_RUNTIME_RUN_ONLY"],
                "1",
            )
            self.assertEqual(
                {name: os.environ.get(name) for name in self._ENV_NAMES},
                before,
            )

    def test_strict_miss_is_structured_and_never_falls_back(self) -> None:
        with _isolated_host_imports() as imports:
            core = importlib.import_module("aiter.ops.flydsl.compile_plan")
            launcher = _FakeLauncher(_all_kinds_launcher, miss=True)
            unit, context, _ = _unit_context(core, imports.aot_backend, launcher)
            with mock.patch.object(
                context.backend,
                "compile_aot",
                wraps=context.backend.compile_aot,
            ) as compile_spy:
                with self.assertRaises(imports.aot_backend.AotCacheMissError) as raised:
                    imports.aot_backend.load_aot(
                        unit,
                        context=context,
                        strict=True,
                    )

            compile_spy.assert_not_called()
            self.assertEqual(len(launcher.calls), 1)
            message = str(raised.exception)
            self.assertIn(_OP_ID, message)
            self.assertIn("gfx950/256", message)
            self.assertIn("signature=", message)
            self.assertIn("cache_dir=", message)
            self.assertIn("synthetic FlyDSL cache miss", message)

    def test_invalid_abi_fields_and_compiler_mismatch_are_structured(self) -> None:
        with _isolated_host_imports() as imports:
            core = importlib.import_module("aiter.ops.flydsl.compile_plan")

            missing = core.KernelSignature(_signature(core).arguments[:-1])
            launcher = _FakeLauncher(_all_kinds_launcher)
            unit, context, _ = _unit_context(
                core,
                imports.aot_backend,
                launcher,
                signature=missing,
            )
            with self.assertRaises(imports.aot_backend.AotBackendError) as raised:
                imports.aot_backend.compile_aot(unit, context=context)
            self.assertIn("missing=", str(raised.exception))
            self.assertIn(_OP_ID, str(raised.exception))
            self.assertIn("gfx950/256", str(raised.exception))

            wrong_target_context = core.CompileContext(
                core.RocmTarget("gfx942", 304),
                context.registry,
                imports.aot_backend.AotBackend(),
            )
            with self.assertRaises(imports.aot_backend.AotBackendError) as raised:
                imports.aot_backend.compile_aot(
                    unit,
                    context=wrong_target_context,
                )
            self.assertIn("does not match context target", str(raised.exception))
            self.assertIn("gfx942/304", str(raised.exception))

            dtype_arguments = list(_signature(core).arguments)
            dtype_arguments[1] = core.SignatureArg(
                "rows",
                core.ArgumentKind.SCALAR,
                "i64",
            )
            dtype_launcher = _FakeLauncher(_all_kinds_launcher)
            unit, context, _ = _unit_context(
                core,
                imports.aot_backend,
                dtype_launcher,
                signature=core.KernelSignature(tuple(dtype_arguments)),
            )
            with self.assertRaisesRegex(
                imports.aot_backend.AotBackendError,
                "ABI/compiler dtype mismatch",
            ):
                imports.aot_backend.compile_aot(unit, context=context)

            duplicate = object.__new__(core.KernelSignature)
            field = core.SignatureArg(
                "tensor",
                core.ArgumentKind.TENSOR,
                "bf16",
                (None, None),
                (None, 1),
            )
            object.__setattr__(duplicate, "arguments", (field, field))
            duplicate_launcher = _FakeLauncher(_tensor_launcher)
            unit, context, _ = _unit_context(
                core,
                imports.aot_backend,
                duplicate_launcher,
                signature=duplicate,
            )
            with self.assertRaisesRegex(
                imports.aot_backend.AotBackendError,
                "duplicate ABI fields",
            ):
                imports.aot_backend.compile_aot(unit, context=context)

            unsupported = core.KernelSignature(
                (
                    core.SignatureArg(
                        "tensor",
                        core.ArgumentKind.TENSOR,
                        "not_a_dtype",
                        (None, None),
                        (None, 1),
                    ),
                    core.SignatureArg("stream", core.ArgumentKind.STREAM),
                )
            )
            unsupported_launcher = _FakeLauncher(_tensor_launcher)
            unit, context, _ = _unit_context(
                core,
                imports.aot_backend,
                unsupported_launcher,
                signature=unsupported,
            )
            with self.assertRaisesRegex(
                imports.aot_backend.AotBackendError,
                "unsupported tensor dtype",
            ):
                imports.aot_backend.compile_aot(unit, context=context)


class TestDirectStage2BackendOperations(unittest.TestCase):
    def test_main_plain_and_masked_units_compile_load_and_miss_strictly(self) -> None:
        with _isolated_host_imports() as imports:
            core = importlib.import_module("aiter.ops.flydsl.compile_plan")
            plan_module = importlib.import_module("aiter.ops.flydsl.moe_compile_plan")
            cases = (
                (
                    plan_module.MIXED_STAGE2_GEMM_OP_ID,
                    _mixed_stage2_launcher,
                ),
                (
                    plan_module.PLAIN_REDUCTION_OP_ID,
                    _reduction_launcher,
                ),
                (
                    plan_module.MASKED_REDUCTION_OP_ID,
                    _reduction_launcher,
                ),
            )
            for op_id, function in cases:
                with self.subTest(op_id=op_id):
                    signature = plan_module.stage2_abi(op_id)
                    compile_launcher = _FakeLauncher(function)
                    unit, context, _ = _unit_context(
                        core,
                        imports.aot_backend,
                        compile_launcher,
                        signature,
                        op_id=op_id,
                    )
                    compiled = imports.aot_backend.compile_aot(
                        unit,
                        context=context,
                    )
                    self.assertFalse(compiled.loaded)
                    self.assertEqual(len(compile_launcher.calls), 1)

                    load_launcher = _FakeLauncher(function)
                    unit, context, _ = _unit_context(
                        core,
                        imports.aot_backend,
                        load_launcher,
                        signature,
                        op_id=op_id,
                    )
                    loaded = imports.aot_backend.load_aot(
                        unit,
                        context=context,
                        strict=True,
                    )
                    self.assertTrue(loaded.loaded)
                    self.assertEqual(
                        load_launcher.calls[0][1]["FLYDSL_RUNTIME_RUN_ONLY"],
                        "1",
                    )

                    miss_launcher = _FakeLauncher(function, miss=True)
                    unit, context, _ = _unit_context(
                        core,
                        imports.aot_backend,
                        miss_launcher,
                        signature,
                        op_id=op_id,
                    )
                    with mock.patch.object(
                        context.backend,
                        "compile_aot",
                        wraps=context.backend.compile_aot,
                    ) as compile_spy:
                        with self.assertRaises(
                            imports.aot_backend.AotCacheMissError
                        ) as raised:
                            imports.aot_backend.load_aot(
                                unit,
                                context=context,
                                strict=True,
                            )
                    compile_spy.assert_not_called()
                    self.assertEqual(raised.exception.op_id, op_id)
                    self.assertEqual(len(miss_launcher.calls), 1)


class TestDirectStage1Aot(unittest.TestCase):
    def test_stage1_and_cktile_jobs_never_enter_fake_or_runtime_host(self) -> None:
        with _isolated_host_imports() as imports, ExitStack() as stack:
            core = importlib.import_module("aiter.ops.flydsl.compile_plan")
            compiled_units = []

            def record_compile(unit, *, context):
                compiled_units.append((unit, context))
                return mock.Mock(unit=unit)

            forbidden = mock.Mock(
                side_effect=AssertionError("forbidden Stage1 AOT boundary")
            )
            stack.enter_context(
                mock.patch.object(imports.aot_moe, "compile_aot", record_compile)
            )
            forbidden_boundaries = (
                (imports.moe, "build_stage1_compile_inputs"),
                (imports.moe, "flydsl_moe_stage1"),
                (torch, "empty"),
                (torch, "empty_like"),
                (torch, "empty_strided"),
                (torch, "full"),
                (torch, "ones"),
                (torch, "randn"),
                (torch, "tensor"),
                (torch, "zeros"),
                (torch, "zeros_like"),
                (torch.cuda, "current_stream"),
                (torch.cuda, "get_device_properties"),
                (torch.cuda, "current_device"),
            )
            for owner, attribute in forbidden_boundaries:
                if hasattr(owner, attribute):
                    stack.enter_context(mock.patch.object(owner, attribute, forbidden))
            stack.enter_context(
                mock.patch(
                    "torch._subclasses.fake_tensor.FakeTensorMode",
                    forbidden,
                )
            )

            result = imports.aot_moe.compile_one_config(
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

            self.assertIsNotNone(result["compile_time"])
            self.assertTrue(result["direct_stage1_aot"])
            self.assertEqual(result["compile_units"], 2)
            self.assertEqual(len(compiled_units), 2)
            self.assertTrue(
                all(
                    argument.kind is not core.ArgumentKind.TENSOR
                    for unit, _ in compiled_units
                    for argument in unit.signature.arguments
                )
            )

            compiled_units.clear()
            epilogue = imports.aot_moe.compile_one_config(
                kernel_name="cktile_epilogue_swiglu",
                model_dim=0,
                inter_dim=2048,
                experts=0,
                topk=8,
                cu_num=256,
                stage="epilogue",
                act="swiglu",
                split_k=2,
                post_activation_layout="interleaved",
                enable_bias=False,
            )
            self.assertIsNotNone(epilogue["compile_time"])
            self.assertEqual(epilogue["compile_units"], 1)
            self.assertEqual(len(compiled_units), 1)
            self.assertEqual(
                [
                    argument.kind
                    for argument in compiled_units[0][0].signature.arguments
                ],
                [
                    core.ArgumentKind.TENSOR,
                    core.ArgumentKind.TENSOR,
                    core.ArgumentKind.SCALAR,
                    core.ArgumentKind.STREAM,
                ],
            )
            forbidden.assert_not_called()


class TestDirectStage2Aot(unittest.TestCase):
    def test_stage2_jobs_never_enter_fake_allocation_stream_or_runtime_host(
        self,
    ) -> None:
        with _isolated_host_imports() as imports, ExitStack() as stack:
            compiled_units = []

            def record_compile(unit, *, context):
                compiled_units.append((unit, context))
                return mock.Mock(unit=unit)

            forbidden = mock.Mock(
                side_effect=AssertionError("forbidden Stage2 AOT boundary")
            )
            stack.enter_context(
                mock.patch.object(imports.aot_moe, "compile_aot", record_compile)
            )
            forbidden_boundaries = (
                (imports.moe, "flydsl_moe_stage2"),
                (torch, "empty"),
                (torch, "empty_like"),
                (torch, "empty_strided"),
                (torch, "full"),
                (torch, "ones"),
                (torch, "randn"),
                (torch, "tensor"),
                (torch, "zeros"),
                (torch, "zeros_like"),
                (torch.cuda, "current_stream"),
                (torch.cuda, "get_device_properties"),
                (torch.cuda, "current_device"),
            )
            for owner, attribute in forbidden_boundaries:
                if hasattr(owner, attribute):
                    stack.enter_context(mock.patch.object(owner, attribute, forbidden))
            stack.enter_context(
                mock.patch(
                    "torch._subclasses.fake_tensor.FakeTensorMode",
                    forbidden,
                )
            )

            common = {
                "kernel_name": "flydsl_test_stage2",
                "model_dim": 7168,
                "inter_dim": 2048,
                "experts": 256,
                "topk": 8,
                "cu_num": 256,
                "stage": 2,
                "tile_m": 32,
                "tile_n": 128,
                "tile_k": 256,
                "doweight_stage2": True,
                "a_dtype": "fp4",
                "b_dtype": "fp4",
                "out_dtype": "bf16",
                "return_per_slot": False,
                "persist": None,
                "token_num": 16,
                "routing_block_count": None,
                "dtype_str": "bf16",
                "waves_per_eu": None,
                "use_async_copy": False,
                "cu_num_mul": 1,
                "b_nt": 2,
                "xcd_swizzle": 0,
                "enable_bias": False,
            }
            cases = (
                (
                    {
                        **common,
                        "mode": "atomic",
                        "accumulate": True,
                        "use_mask": False,
                        "topk_ids_available": False,
                        "num_experts": 0,
                    },
                    ("aiter.flydsl.moe.stage2.mixed_gemm.v1",),
                ),
                (
                    {
                        **common,
                        "mode": "reduce",
                        "accumulate": False,
                        "use_mask": False,
                        "topk_ids_available": False,
                        "num_experts": 0,
                    },
                    (
                        "aiter.flydsl.moe.stage2.mixed_gemm.v1",
                        "aiter.flydsl.moe.stage2.reduction.plain.v1",
                    ),
                ),
                (
                    {
                        **common,
                        "experts": 32,
                        "mode": "reduce",
                        "accumulate": False,
                        "use_mask": True,
                        "topk_ids_available": True,
                        "num_experts": 256,
                    },
                    (
                        "aiter.flydsl.moe.stage2.mixed_gemm.v1",
                        "aiter.flydsl.moe.stage2.reduction.masked.v1",
                    ),
                ),
            )
            for config, expected_op_ids in cases:
                compiled_units.clear()
                result = imports.aot_moe.compile_one_config(**config)
                self.assertIsNotNone(result["compile_time"])
                self.assertTrue(result["direct_stage2_aot"])
                self.assertEqual(result["compile_units"], len(expected_op_ids))
                self.assertEqual(
                    tuple(unit.spec.op_id for unit, _ in compiled_units),
                    expected_op_ids,
                )
            self.assertFalse(hasattr(imports.moe, "build_stage1_compile_inputs"))
            self.assertFalse(hasattr(imports.moe, "build_stage2_compile_inputs"))
            forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()
