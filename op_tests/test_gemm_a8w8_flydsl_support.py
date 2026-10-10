# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import contextlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aiter.ops import gemm_op_a8w8
from aiter.ops.gemm_op_a8w8 import (
    _flydsl_rdna3_a8w8_shape_supported,
    _try_flydsl_rdna3_a8w8,
)


class _Tensor:
    def __init__(self, shape, dtype, *, stride=None, contiguous=True, pointer=0x1000):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = "cuda:0"
        self.is_cuda = True
        self._stride = stride or (shape[1], 1)
        self._contiguous = contiguous
        self._pointer = pointer

    def stride(self, dim):
        return self._stride[dim]

    def is_contiguous(self):
        return self._contiguous

    def data_ptr(self):
        return self._pointer


def _valid_tensors():
    return (
        _Tensor((8, 128), torch.int8),
        _Tensor((64, 128), torch.int8),
        _Tensor((8, 1), torch.float32),
        _Tensor((64, 1), torch.float32),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_flydsl_rdna3_a8w8_supported_tile_shapes(dtype):
    assert _flydsl_rdna3_a8w8_shape_supported(32, 64, 128, 128, 128, dtype)
    assert _flydsl_rdna3_a8w8_shape_supported(256, 128, 256, 256, 256, dtype)


@pytest.mark.parametrize(
    "shape",
    [
        (32, 63, 128, 128, 128),
        (32, 64, 127, 128, 128),
        (32, 64, 64, 64, 64),
        (32, 64, 128, 136, 128),
        (32, 64, 128, 128, 136),
    ],
)
def test_flydsl_rdna3_a8w8_rejects_unsupported_geometry(shape):
    assert not _flydsl_rdna3_a8w8_shape_supported(*shape, torch.bfloat16)


def test_flydsl_rdna3_a8w8_rejects_unsupported_output_and_buffer_span():
    assert not _flydsl_rdna3_a8w8_shape_supported(32, 64, 128, 128, 128, torch.int32)
    assert not _flydsl_rdna3_a8w8_shape_supported(
        1, 1 << 25, 128, 128, 1 << 25, torch.float32
    )
    assert _flydsl_rdna3_a8w8_shape_supported(
        1, 64, 131008, 131008, 131008, torch.float32
    )
    assert not _flydsl_rdna3_a8w8_shape_supported(
        1, 64, 131072, 131072, 131072, torch.float32
    )


@pytest.mark.parametrize("target_arch", ["gfx942", "gfx1201", "gfx1100"])
def test_flydsl_rdna3_a8w8_target_arch_mismatch_falls_back_without_factory(
    monkeypatch, target_arch
):
    for name in (
        "flydsl",
        "flydsl.runtime",
        "aiter.ops.flydsl",
        "aiter.ops.flydsl.kernels",
        "aiter.ops.triton",
        "aiter.ops.triton.gemm",
        "aiter.ops.triton.gemm.basic",
    ):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    device_module = ModuleType("flydsl.runtime.device")
    device_module.get_rocm_arch = lambda: target_arch
    monkeypatch.setitem(sys.modules, "flydsl.runtime.device", device_module)
    factory_calls = []
    kernel_module = ModuleType("aiter.ops.flydsl.kernels.rdna3_int8_gemm")

    def create_kernel(*args, **kwargs):
        factory_calls.append((args, kwargs))
        pytest.fail("mismatched target called the FlyDSL kernel factory")

    kernel_module.create_wmma_int8_gemm_module = create_kernel
    monkeypatch.setitem(
        sys.modules, "aiter.ops.flydsl.kernels.rdna3_int8_gemm", kernel_module
    )
    launcher_path = (
        Path(__file__).resolve().parents[1]
        / "aiter"
        / "ops"
        / "flydsl"
        / "rdna3_int8_gemm.py"
    )
    spec = importlib.util.spec_from_file_location(
        "rdna3_int8_gemm_test_launcher", launcher_path
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    monkeypatch.setattr(
        gemm_op_a8w8.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="GFX1151:sramecc+"),
    )
    monkeypatch.setattr(
        gemm_op_a8w8.torch.cuda, "device", lambda device: contextlib.nullcontext()
    )
    monkeypatch.setattr(gemm_op_a8w8, "AITER_GEMM_A8W8_BACKEND", "flydsl")
    monkeypatch.setattr(gemm_op_a8w8, "_ck_a8w8_supported", lambda: False)
    monkeypatch.setattr(
        gemm_op_a8w8, "_flydsl_rdna3_a8w8_supported", lambda *args: True
    )
    triton_output = torch.empty((8, 64), dtype=torch.float32)
    triton_module = ModuleType("aiter.ops.triton.gemm.basic.gemm_a8w8")
    triton_module.gemm_a8w8 = lambda *args, **kwargs: triton_output
    monkeypatch.setitem(
        sys.modules, "aiter.ops.triton.gemm.basic.gemm_a8w8", triton_module
    )
    imports = []

    def import_module(name, package=None):
        imports.append((name, package))
        if name == ".flydsl.rdna3_int8_gemm":
            return SimpleNamespace(gemm_a8w8_rdna3=launcher.gemm_a8w8_rdna3)
        pytest.fail(f"unexpected module import: {name}")

    monkeypatch.setattr(gemm_op_a8w8.importlib, "import_module", import_module)
    x = torch.empty((8, 128), dtype=torch.int8)
    w = torch.empty((64, 128), dtype=torch.int8)
    x_scale = torch.empty((8, 1), dtype=torch.float32)
    w_scale = torch.empty((64, 1), dtype=torch.float32)

    result = gemm_op_a8w8.gemm_a8w8(x, w, x_scale, w_scale, dtype=torch.float32)

    assert result is triton_output
    assert factory_calls == []
    assert imports == [(".flydsl.rdna3_int8_gemm", "aiter.ops")]


def test_flydsl_rdna3_a8w8_dispatches_supported_call(monkeypatch):
    monkeypatch.setattr(
        gemm_op_a8w8.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx1151"),
    )
    x, w, x_scale, w_scale = _valid_tensors()
    expected = object()
    output = object()
    called = []

    def flydsl_gemm(*args):
        called.append(args)
        return expected

    monkeypatch.setattr(
        gemm_op_a8w8.importlib,
        "import_module",
        lambda name, package: SimpleNamespace(gemm_a8w8_rdna3=flydsl_gemm),
    )
    monkeypatch.setattr(gemm_op_a8w8.torch, "empty", lambda *args, **kwargs: output)

    result = _try_flydsl_rdna3_a8w8(x, w, x_scale, w_scale, None, torch.bfloat16, None)
    assert result is expected
    assert called == [(x, w, x_scale, w_scale, output)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bias": object(), "splitK": None},
        {"bias": None, "splitK": 1},
    ],
)
def test_flydsl_rdna3_a8w8_keeps_bias_and_explicit_splitk_on_fallback(
    monkeypatch, kwargs
):
    monkeypatch.setattr(
        gemm_op_a8w8.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx1151"),
    )
    x, w, x_scale, w_scale = _valid_tensors()
    monkeypatch.setattr(
        gemm_op_a8w8.importlib,
        "import_module",
        lambda *args, **kwargs: pytest.fail("unsupported call attempted FlyDSL import"),
    )

    assert (
        _try_flydsl_rdna3_a8w8(
            x,
            w,
            x_scale,
            w_scale,
            kwargs["bias"],
            torch.bfloat16,
            kwargs["splitK"],
        )
        is None
    )


@pytest.mark.parametrize(
    "error", [ModuleNotFoundError("flydsl"), OSError("missing HIP DLL")]
)
def test_flydsl_rdna3_a8w8_import_failure_uses_fallback(monkeypatch, error):
    monkeypatch.setattr(
        gemm_op_a8w8.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx1151"),
    )
    x, w, x_scale, w_scale = _valid_tensors()

    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(gemm_op_a8w8.importlib, "import_module", unavailable)
    assert (
        _try_flydsl_rdna3_a8w8(x, w, x_scale, w_scale, None, torch.bfloat16, None)
        is None
    )


def test_gemm_a8w8_public_dispatch_uses_triton_by_default(
    monkeypatch,
):
    x = torch.empty((2, 128), dtype=torch.int8)
    w = torch.empty((64, 128), dtype=torch.int8)
    x_scale = torch.ones((2, 1), dtype=torch.float32)
    w_scale = torch.ones((64, 1), dtype=torch.float32)
    calls = []

    monkeypatch.setattr(gemm_op_a8w8, "AITER_GEMM_A8W8_BACKEND", "default")
    monkeypatch.setattr(
        gemm_op_a8w8,
        "_try_flydsl_rdna3_a8w8",
        lambda *args: pytest.fail("default backend must not attempt FlyDSL"),
    )
    monkeypatch.setattr(gemm_op_a8w8, "_ck_a8w8_supported", lambda: False)

    def triton_gemm(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.empty((2, 64), dtype=kwargs["dtype"])

    module_name = "aiter.ops.triton.gemm.basic.gemm_a8w8"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(gemm_a8w8=triton_gemm),
    )

    result = gemm_op_a8w8.gemm_a8w8(
        x, w, x_scale, w_scale, dtype=torch.float16, splitK=3
    )

    assert result.shape == (2, 64)
    assert result.dtype == torch.float16
    assert calls == [((x, w, x_scale, w_scale, None), {"dtype": torch.float16})]


def test_gemm_a8w8_public_dispatch_returns_flydsl_result(monkeypatch):
    x = torch.empty((2, 128), dtype=torch.int8)
    w = torch.empty((64, 128), dtype=torch.int8)
    x_scale = torch.ones((2, 1), dtype=torch.float32)
    w_scale = torch.ones((64, 1), dtype=torch.float32)
    output = torch.empty((2, 64), dtype=torch.bfloat16)
    calls = []

    monkeypatch.setattr(gemm_op_a8w8, "AITER_GEMM_A8W8_BACKEND", "flydsl")

    def flydsl_gemm(*args):
        calls.append(args)
        return output

    monkeypatch.setattr(gemm_op_a8w8, "_try_flydsl_rdna3_a8w8", flydsl_gemm)
    monkeypatch.setattr(
        gemm_op_a8w8,
        "_ck_a8w8_supported",
        lambda: pytest.fail("FlyDSL result should return before CK/Triton"),
    )

    result = gemm_op_a8w8.gemm_a8w8(x, w, x_scale, w_scale)

    assert result is output
    assert calls == [(x, w, x_scale, w_scale, None, torch.bfloat16, None)]


def test_gemm_a8w8_flydsl_opt_in_uses_triton_when_shape_is_unsupported(monkeypatch):
    x = torch.empty((2, 128), dtype=torch.int8)
    w = torch.empty((64, 128), dtype=torch.int8)
    x_scale = torch.ones((2, 1), dtype=torch.float32)
    w_scale = torch.ones((64, 1), dtype=torch.float32)
    calls = []

    monkeypatch.setattr(gemm_op_a8w8, "AITER_GEMM_A8W8_BACKEND", "flydsl")
    monkeypatch.setattr(gemm_op_a8w8, "_try_flydsl_rdna3_a8w8", lambda *args: None)
    monkeypatch.setattr(gemm_op_a8w8, "_ck_a8w8_supported", lambda: False)
    module_name = "aiter.ops.triton.gemm.basic.gemm_a8w8"

    def triton_gemm(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.empty((2, 64), dtype=kwargs["dtype"])

    monkeypatch.setitem(
        sys.modules, module_name, SimpleNamespace(gemm_a8w8=triton_gemm)
    )

    result = gemm_op_a8w8.gemm_a8w8(
        x, w, x_scale, w_scale, dtype=torch.float16, splitK=3
    )

    assert result.shape == (2, 64)
    assert result.dtype == torch.float16
    assert calls == [((x, w, x_scale, w_scale, None), {"dtype": torch.float16})]


@pytest.mark.parametrize(
    "arch,tensor_index,replacement",
    [
        ("gfx1201", None, None),
        ("gfx1151", 0, _Tensor((8, 128), torch.float16)),
        ("gfx1151", 1, _Tensor((64, 128), torch.int8, stride=(129, 1))),
        ("gfx1151", 2, _Tensor((8, 1), torch.float32, contiguous=False)),
        ("gfx1151", 3, _Tensor((64, 1), torch.float16)),
    ],
)
def test_flydsl_rdna3_a8w8_unsupported_call_uses_fallback(
    monkeypatch, arch, tensor_index, replacement
):
    monkeypatch.setattr(
        gemm_op_a8w8.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName=arch),
    )
    tensors = list(_valid_tensors())
    if tensor_index is not None:
        tensors[tensor_index] = replacement
    monkeypatch.setattr(
        gemm_op_a8w8.importlib,
        "import_module",
        lambda *args, **kwargs: pytest.fail("unsupported call attempted FlyDSL import"),
    )

    assert _try_flydsl_rdna3_a8w8(*tensors, None, torch.bfloat16, None) is None


@pytest.mark.parametrize("tensor_index", [0, 1])
def test_flydsl_rdna3_a8w8_rejects_unaligned_input_base(monkeypatch, tensor_index):
    monkeypatch.setattr(
        gemm_op_a8w8.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx1151"),
    )
    tensors = list(_valid_tensors())
    tensor = tensors[tensor_index]
    tensors[tensor_index] = _Tensor(
        tensor.shape, tensor.dtype, stride=tensor._stride, pointer=0x1001
    )
    monkeypatch.setattr(
        gemm_op_a8w8.importlib,
        "import_module",
        lambda *args, **kwargs: pytest.fail("unaligned input attempted FlyDSL import"),
    )

    assert _try_flydsl_rdna3_a8w8(*tensors, None, torch.bfloat16, None) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
