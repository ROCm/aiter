# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for the FlyDSL AOT compatibility boundary."""

from __future__ import annotations

import os

import flydsl.expr as fx
import pytest
import torch

from aiter.ops.flydsl import aot_backend
from aiter.ops.flydsl.compile_request import (
    ArgumentKind,
    CompileContext,
    CompileOpRegistry,
    KernelSignature,
    RocmTarget,
    SignatureArg,
)

OP_ID = "aiter.flydsl.test.aot_backend.v1"
TARGET = RocmTarget("gfx950", 256)


def _launcher(
    pointer: fx.Pointer,
    rows: fx.Int32,
    scale: fx.Float32,
    tensor: fx.Tensor,
    stream: fx.Stream,
) -> None:
    raise AssertionError("fake launcher body must not execute")


class FakeLauncher:
    def __init__(self, function=_launcher, *, miss=False):
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


def _signature():
    return KernelSignature(
        (
            SignatureArg("pointer", ArgumentKind.POINTER, "u8"),
            SignatureArg("rows", ArgumentKind.SCALAR, "i32"),
            SignatureArg("scale", ArgumentKind.SCALAR, "f32"),
            SignatureArg(
                "tensor",
                ArgumentKind.TENSOR,
                "bf16",
                (None, None),
                (None, 1),
            ),
            SignatureArg("stream", ArgumentKind.STREAM),
        )
    )


def _request_context(launcher, *, signature=None, strict_runtime=False):
    registry = CompileOpRegistry()
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

    registry.register(OP_ID)(builder)
    backend = aot_backend.AotBackend(strict_runtime=strict_runtime)
    context = CompileContext(TARGET, registry, backend)
    request = registry.make_request(
        OP_ID,
        target=TARGET,
        signature=signature or _signature(),
    )
    return request, context, builder_environments


def test_materializes_metadata_only_pointer_scalars_tensor_and_stream():
    launcher = FakeLauncher()
    request, _, _ = _request_context(launcher)

    args = aot_backend._materialize_compile_args(request, launcher)

    assert [type(value).__name__ for value in args] == [
        "PointerJitArg",
        "Int32",
        "Float32",
        "TorchTensorJitArg",
        "Stream",
    ]
    assert args[0].pointer.value is None
    assert args[1].value == 0
    assert args[2].value == 0.0
    assert args[3].dtype == torch.bfloat16
    assert args[3].shape == (2, 2)
    assert args[3].strides == (2, 1)
    assert args[3].torch_tensor.data_ptr() == 0
    assert not isinstance(args[3].torch_tensor, torch.Tensor)
    assert args[4].value is None


def test_compile_and_strict_load_use_target_env_and_restore_it(monkeypatch):
    launcher = FakeLauncher()
    request, context, builder_envs = _request_context(launcher)
    monkeypatch.setenv("ARCH", "original")
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "original")

    compiled = aot_backend.compile_aot(request, context=context)
    loaded = aot_backend.load_aot(request, context=context)

    assert not compiled.loaded
    assert loaded.loaded
    assert builder_envs == [
        ("gfx950", "gfx950", "256"),
        ("gfx950", "gfx950", "256"),
    ]
    assert launcher.calls[0][1]["FLYDSL_RUNTIME_RUN_ONLY"] == "0"
    assert launcher.calls[1][1]["FLYDSL_RUNTIME_RUN_ONLY"] == "1"
    assert os.environ["ARCH"] == "original"
    assert os.environ["FLYDSL_RUNTIME_RUN_ONLY"] == "original"


def test_developer_runtime_resolve_builds_without_compile_only_invocation():
    launcher = FakeLauncher()
    request, context, builder_envs = _request_context(launcher)

    first = context.backend.resolve_aot(request, context=context)
    second = context.backend.resolve_aot(request, context=context)

    assert first is second
    assert first.launcher is launcher
    assert launcher.calls == []
    assert builder_envs == [(None, None, None)]


def test_strict_runtime_resolve_never_falls_back_on_cache_miss():
    launcher = FakeLauncher(miss=True)
    request, context, _ = _request_context(launcher, strict_runtime=True)

    with pytest.raises(aot_backend.AotCacheMissError) as error:
        context.backend.resolve_aot(request, context=context)

    assert error.value.op_id == OP_ID
    assert error.value.target == TARGET
    assert "synthetic miss" in str(error.value)


def test_strict_runtime_does_not_misclassify_other_run_only_errors():
    class InvalidRunOnlyLauncher(FakeLauncher):
        def __call__(self, *args):
            raise RuntimeError(
                "FLYDSL_RUNTIME_RUN_ONLY=1 is incompatible with FLYDSL_DUMP_IR=1"
            )

    launcher = InvalidRunOnlyLauncher()
    request, context, _ = _request_context(launcher, strict_runtime=True)

    with pytest.raises(aot_backend.AotBackendError) as error:
        context.backend.resolve_aot(request, context=context)

    assert type(error.value) is aot_backend.AotBackendError


def test_target_and_abi_mismatch_fail_before_an_artifact_is_returned():
    launcher = FakeLauncher()
    request, context, _ = _request_context(launcher)
    wrong_target = type(request)(
        request.op_id,
        RocmTarget("gfx942", 304),
        request.bound_kwargs,
        request.signature,
    )
    with pytest.raises(aot_backend.AotBackendError, match="does not match"):
        aot_backend.compile_aot(wrong_target, context=context)

    bad_signature = KernelSignature(
        (
            SignatureArg("wrong", ArgumentKind.POINTER, "u8"),
            *request.signature.arguments[1:],
        )
    )
    bad_request = type(request)(
        request.op_id,
        request.target,
        request.bound_kwargs,
        bad_signature,
    )
    with pytest.raises(aot_backend.AotBackendError, match="ABI/compiler"):
        aot_backend.compile_aot(bad_request, context=context)


def test_runtime_context_captures_device_target_and_strict_mode(monkeypatch):
    properties = type(
        "Properties",
        (),
        {"gcnArchName": "gfx950:sramecc+", "multi_processor_count": 256},
    )()
    aot_backend._cached_runtime_compile_context.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: properties)
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "1")

    context = aot_backend.create_runtime_compile_context(3)

    assert context.target == TARGET
    assert context.backend.strict_runtime
