# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for the FlyDSL AOT compatibility backend."""

from __future__ import annotations

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


def _sorting_oneshot_launcher(
    topk_ids_tensor: fx.Tensor,
    topk_weights_tensor: fx.Tensor,
    sorted_token_ids: fx.Tensor,
    sorted_weights_out: fx.Tensor,
    sorted_expert_ids: fx.Tensor,
    num_valid_ids_out: fx.Tensor,
    moe_buf: fx.Tensor,
    expert_mask_tensor: fx.Tensor,
    i32_tokens: fx.Int32,
    i32_moe_buf_elems: fx.Int32,
    n_grid_blocks: fx.Int32,
    stream: fx.Stream,
) -> None:
    raise AssertionError("the fake launcher body must not execute")


def _sorting_p0v2_p23_launcher(
    topk_ids: fx.Tensor,
    workspace: fx.Tensor,
    topk_weights_tensor: fx.Tensor,
    sorted_token_ids: fx.Tensor,
    sorted_weights_out: fx.Tensor,
    sorted_expert_ids: fx.Tensor,
    num_valid_ids_out: fx.Tensor,
    moe_buf: fx.Tensor,
    expert_mask_tensor: fx.Tensor,
    i32_tokens: fx.Int32,
    i32_mesh_stride: fx.Int32,
    i32_mesh_size: fx.Int32,
    i32_moe_buf_elems: fx.Int32,
    n_grid_p23: fx.Int32,
    stream: fx.Stream,
) -> None:
    raise AssertionError("the fake launcher body must not execute")


def _sorting_4k_fused_launcher(
    topk_ids: fx.Tensor,
    workspace: fx.Tensor,
    topk_weights_tensor: fx.Tensor,
    sorted_token_ids: fx.Tensor,
    sorted_weights_out: fx.Tensor,
    sorted_expert_ids: fx.Tensor,
    num_valid_ids_out: fx.Tensor,
    moe_buf: fx.Tensor,
    expert_mask_tensor: fx.Tensor,
    i32_tokens: fx.Int32,
    i32_mesh_stride: fx.Int32,
    i32_mesh_size: fx.Int32,
    i32_moe_buf_elems: fx.Int32,
    i32_ws_total: fx.Int32,
    i32_p0_niters: fx.Int32,
    n_grid_k1: fx.Int32,
    n_grid_k2: fx.Int32,
    n_grid_p23: fx.Int32,
    stream: fx.Stream,
) -> None:
    raise AssertionError("the fake launcher body must not execute")


class _FakeLauncher:
    def __init__(self, function, *, miss=False):
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
                "FLYDSL_RUNTIME_RUN_ONLY=1 but no usable AOT cache: synthetic miss"
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


def _request_context(
    core,
    backend_module,
    launcher,
    signature=None,
    *,
    op_id=_OP_ID,
    strict_runtime=False,
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
    backend = backend_module.AotBackend(strict_runtime=strict_runtime)
    context = core.CompileContext(target, registry, backend)
    request = registry.make_request(
        op_id,
        target=target,
        signature=signature if signature is not None else _signature(core),
    )
    return request, context, builder_environments


class TestAotMaterialization(unittest.TestCase):
    def test_materializes_metadata_only_pointer_scalars_tensor_and_stream(self):
        with _isolated_host_imports() as imports:
            core = importlib.import_module("aiter.ops.flydsl.compile_request")
            launcher = _FakeLauncher(_all_kinds_launcher)
            request, _, _ = _request_context(core, imports.aot_backend, launcher)

            args = imports.aot_backend._materialize_compile_args(request, launcher)

            self.assertEqual(
                [type(value).__name__ for value in args],
                ["PointerJitArg", "Int32", "Float32", "TorchTensorJitArg", "Stream"],
            )
            self.assertIsNone(args[0].pointer.value)
            self.assertEqual(args[1].value, 0)
            self.assertEqual(args[2].value, 0.0)
            self.assertEqual(args[3].dtype, torch.bfloat16)
            self.assertEqual(args[3].shape, (2, 2))
            self.assertEqual(args[3].strides, (2, 1))
            self.assertEqual(args[3].torch_tensor.data_ptr(), 0)
            self.assertFalse(isinstance(args[3].torch_tensor, torch.Tensor))
            self.assertIsNone(args[4].value)

    def test_runtime_context_queries_target_once_and_captures_strict_mode(self):
        with _isolated_host_imports() as imports:
            properties = mock.Mock(
                gcnArchName="gfx950:sramecc+",
                multi_processor_count=256,
            )
            imports.aot_backend._cached_runtime_compile_context.cache_clear()
            with (
                mock.patch.object(
                    torch.cuda,
                    "get_device_properties",
                    return_value=properties,
                ) as properties_spy,
                mock.patch.dict(os.environ, {"FLYDSL_RUNTIME_RUN_ONLY": "1"}),
            ):
                context = imports.aot_backend.create_runtime_compile_context(3)

            properties_spy.assert_called_once_with(3)
            self.assertEqual(context.target.arch, "gfx950")
            self.assertEqual(context.target.cu_count, 256)
            self.assertTrue(context.backend.strict_runtime)


class TestCompileLoadAndResolve(unittest.TestCase):
    _ENV_NAMES = (
        "ARCH",
        "FLYDSL_GPU_ARCH",
        "CU_NUM",
        "COMPILE_ONLY",
        "FLYDSL_RUNTIME_ENABLE_CACHE",
        "FLYDSL_RUNTIME_RUN_ONLY",
    )

    def test_compile_and_strict_load_force_modes_and_restore_environment(self):
        with _isolated_host_imports() as imports:
            core = importlib.import_module("aiter.ops.flydsl.compile_request")
            before = {name: os.environ.get(name) for name in self._ENV_NAMES}

            compile_launcher = _FakeLauncher(_all_kinds_launcher)
            request, context, builder_environments = _request_context(
                core,
                imports.aot_backend,
                compile_launcher,
            )
            compiled = imports.aot_backend.compile_aot(request, context=context)
            self.assertFalse(compiled.loaded)
            self.assertIs(compiled.request, request)
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
            request, context, _ = _request_context(
                core,
                imports.aot_backend,
                load_launcher,
            )
            loaded = imports.aot_backend.load_aot(
                request,
                context=context,
                strict=True,
            )
            self.assertTrue(loaded.loaded)
            self.assertEqual(
                load_launcher.calls[0][1]["FLYDSL_RUNTIME_RUN_ONLY"],
                "1",
            )

    def test_developer_resolve_builds_launcher_without_compile_only_invocation(self):
        with _isolated_host_imports() as imports:
            core = importlib.import_module("aiter.ops.flydsl.compile_request")
            launcher = _FakeLauncher(_all_kinds_launcher)
            request, context, builder_environments = _request_context(
                core,
                imports.aot_backend,
                launcher,
            )
            before = {name: os.environ.get(name) for name in self._ENV_NAMES}

            first = context.backend.resolve_aot(request, context=context)
            second = context.backend.resolve_aot(request, context=context)

            self.assertIs(first, second)
            self.assertIs(first.launcher, launcher)
            self.assertEqual(first.compile_args, ())
            self.assertEqual(launcher.calls, [])
            self.assertEqual(builder_environments, [(None, None, None)])
            self.assertEqual(
                {name: os.environ.get(name) for name in self._ENV_NAMES},
                before,
            )

    def test_strict_miss_is_structured_and_never_falls_back(self):
        with _isolated_host_imports() as imports:
            core = importlib.import_module("aiter.ops.flydsl.compile_request")
            launcher = _FakeLauncher(_all_kinds_launcher, miss=True)
            request, context, _ = _request_context(
                core,
                imports.aot_backend,
                launcher,
            )
            with mock.patch.object(
                context.backend,
                "compile_aot",
                wraps=context.backend.compile_aot,
            ) as compile_spy:
                with self.assertRaises(imports.aot_backend.AotCacheMissError) as raised:
                    imports.aot_backend.load_aot(
                        request,
                        context=context,
                        strict=True,
                    )

            compile_spy.assert_not_called()
            message = str(raised.exception)
            self.assertIn(_OP_ID, message)
            self.assertIn("gfx950/256", message)
            self.assertIn("signature=", message)
            self.assertIn("cache_dir=", message)
            self.assertIn("synthetic miss", message)

    def test_target_and_abi_mismatches_are_structured(self):
        with _isolated_host_imports() as imports:
            core = importlib.import_module("aiter.ops.flydsl.compile_request")
            launcher = _FakeLauncher(_all_kinds_launcher)
            missing_stream = core.KernelSignature(_signature(core).arguments[:-1])
            request, context, _ = _request_context(
                core,
                imports.aot_backend,
                launcher,
                missing_stream,
            )
            with self.assertRaisesRegex(
                imports.aot_backend.AotBackendError,
                "missing=",
            ):
                imports.aot_backend.compile_aot(request, context=context)

            wrong_context = core.CompileContext(
                core.RocmTarget("gfx942", 304),
                context.registry,
                imports.aot_backend.AotBackend(),
            )
            with self.assertRaisesRegex(
                imports.aot_backend.AotBackendError,
                "does not match context target",
            ):
                imports.aot_backend.compile_aot(request, context=wrong_context)


class TestConcreteFamilyBackends(unittest.TestCase):
    def test_stage2_reduction_and_sorting_compile_load_and_miss(self):
        with _isolated_host_imports() as imports:
            core = importlib.import_module("aiter.ops.flydsl.compile_request")
            requests = importlib.import_module("aiter.ops.flydsl.moe_compile_requests")
            cases = (
                (requests.MIXED_STAGE2_GEMM_OP_ID, _mixed_stage2_launcher),
                (requests.PLAIN_REDUCTION_OP_ID, _reduction_launcher),
                (requests.MASKED_REDUCTION_OP_ID, _reduction_launcher),
                (requests.SORTING_ONESHOT_OP_ID, _sorting_oneshot_launcher),
                (requests.SORTING_P0V2_P23_OP_ID, _sorting_p0v2_p23_launcher),
                (requests.SORTING_4K_FUSED_OP_ID, _sorting_4k_fused_launcher),
            )
            for op_id, function in cases:
                with self.subTest(op_id=op_id):
                    signature = requests.get_kernel_signature(op_id)
                    compile_launcher = _FakeLauncher(function)
                    request, context, _ = _request_context(
                        core,
                        imports.aot_backend,
                        compile_launcher,
                        signature,
                        op_id=op_id,
                    )
                    compiled = imports.aot_backend.compile_aot(
                        request,
                        context=context,
                    )
                    self.assertFalse(compiled.loaded)
                    self.assertEqual(len(compile_launcher.calls), 1)

                    miss_launcher = _FakeLauncher(function, miss=True)
                    request, context, _ = _request_context(
                        core,
                        imports.aot_backend,
                        miss_launcher,
                        signature,
                        op_id=op_id,
                    )
                    with self.assertRaises(
                        imports.aot_backend.AotCacheMissError
                    ) as raised:
                        imports.aot_backend.load_aot(
                            request,
                            context=context,
                            strict=True,
                        )
                    self.assertEqual(raised.exception.op_id, op_id)


if __name__ == "__main__":
    unittest.main()
