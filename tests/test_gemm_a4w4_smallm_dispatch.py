# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU routing tests; GPU numerical/performance coverage is in op_tests/."""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dispatch(monkeypatch):
    # Import the actual dispatcher with its GPU/JIT dependencies replaced.
    # Real CPU tensors exercise shape restoration and storage aliasing.
    modules = {}
    for name in (
        "aiter",
        "aiter.ops",
        "aiter.jit",
        "aiter.jit.utils",
        "aiter.utility",
        "aiter.ops.triton",
        "aiter.ops.triton.gemm",
        "aiter.ops.triton.gemm.basic",
    ):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
    modules["aiter"].logger = logging.getLogger(__name__)
    dtype_module = ModuleType("aiter.utility.dtypes")
    dtype_module.bf16 = torch.bfloat16
    dtype_module.fp8 = torch.float8_e4m3fn
    dtype_module.fp4x2 = torch.uint8
    dtype_module.fp8_e8m0 = torch.uint8
    modules[dtype_module.__name__] = dtype_module
    modules["aiter.utility"].dtypes = dtype_module
    identity = lambda **kwargs: lambda fn: fn
    bindings = {
        "aiter.jit.core": {
            "AITER_CONFIGS": SimpleNamespace(
                AITER_CONFIG_GEMM_A4W4_TRITON_PRESHUFFLE_FILE=str(
                    ROOT / "aiter/configs/a4w4_triton_preshuffle_dispatch.csv"
                ),
            ),
            "AITER_LOG_TUNED_CONFIG": False,
            "compile_ops": lambda *args, **kwargs: lambda fn: fn,
        },
        "aiter.jit.utils.torch_guard": {"torch_compile_guard": identity},
        "aiter.jit.utils.asm_guard": {"require_gfx1250_asm": Mock()},
        "aiter.jit.utils.chip_info": {
            "get_cu_num": lambda: 256,
            "get_gfx_runtime": lambda: "gfx950",
        },
        "aiter.ops.gemm_op_common": {"get_padded_m": lambda m, n, k, gl: m},
    }
    for name, values in bindings.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        modules[name] = module
    calls = []

    def triton(a, b, a_scale, b_scale, **kwargs):
        calls.append((a, b, a_scale, b_scale, kwargs))
        config = kwargs["config"]
        if config is not None:
            config["BLOCK_SIZE_N"] = 999  # Real wrappers also mutate their config.
        return torch.zeros((a.shape[0], b.shape[0] * 16), dtype=kwargs["dtype"])

    backend = ModuleType("aiter.ops.triton.gemm.basic.gemm_afp4wfp4")
    backend.gemm_afp4wfp4_preshuffle = triton
    modules[backend.__name__] = backend
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    name = "aiter.ops.gemm_op_a4w4"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "aiter/ops/gemm_op_a4w4.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module, calls


def operands(m=32, n=512, k=1024, leading_shape=None):
    a = torch.zeros(
        (*((m,) if leading_shape is None else leading_shape), k // 2), dtype=torch.uint8
    )
    b = torch.zeros((n, k // 2), dtype=torch.uint8)
    a_scale = torch.zeros(((m + 31) // 32 * 32, k // 32), dtype=torch.uint8)
    b_scale = torch.zeros(((n + 31) // 32 * 32, k // 32), dtype=torch.uint8)
    return a, b, a_scale, b_scale


def try_route(module, tensors, **overrides):
    kwargs = {
        "gfx": "gfx950",
        "bias": None,
        "dtype": torch.bfloat16,
        "alpha": 1.0,
        "beta": 0.0,
        "bpreshuffle": True,
        "apreshuffle": False,
        "global_A_scale": None,
        "global_B_scale": None,
    }
    kwargs.update(overrides)
    return module._gemm_a4w4_triton_preshuffle(*tensors, **kwargs)


def test_route_preserves_storage_and_leading_dimensions(dispatch):
    module, calls = dispatch
    tensors = operands(m=64, leading_shape=(2, 32))
    result = try_route(module, tensors)
    assert result.shape == (2, 32, 512)
    assert [x.shape for x in calls[0][:4]] == [
        (64, 512),
        (32, 8192),
        (2, 1024),
        (16, 1024),
    ]
    assert [x.data_ptr() for x in calls[0][:4]] == [x.data_ptr() for x in tensors]


@pytest.mark.parametrize("m", [32, 64, 128, 256])
def test_exact_long_k_routes_use_independent_config_copies(dispatch, m):
    module, calls = dispatch
    tensors = operands(m=m, k=7168)
    for _ in range(2):
        assert try_route(module, tensors).shape == (m, 512)
    assert calls[0][-1]["config"] is not calls[1][-1]["config"]
    cfg = module._load_a4w4_triton_preshuffle_config(
        module.AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_TRITON_PRESHUFFLE_FILE
    )
    assert cfg[("gfx950", 256, m, 512, 7168)]["BLOCK_SIZE_N"] == 64


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gfx": "gfx942"},
        {"gfx": "gfx1250"},
        {"dtype": torch.float16},
        {"bias": torch.zeros(512)},
        {"alpha": 2.0},
        {"beta": 1.0},
        {"bpreshuffle": False},
        {"apreshuffle": True},
        {"global_A_scale": torch.ones(1)},
        {"global_B_scale": torch.ones(1)},
    ],
)
def test_unsupported_contract_retains_fallback(dispatch, kwargs):
    module, calls = dispatch
    assert try_route(module, operands(), **kwargs) is None
    assert not calls


@pytest.mark.parametrize(
    "shape",
    [
        (31, 512, 1024),
        (65, 512, 1024),
        (32, 255, 1024),
        (32, 1024, 1024),
        (32, 512, 1280),
        (33, 512, 7168),
        (512, 512, 7168),
        (128, 4608, 7168),
    ],
)
def test_outside_range_does_not_inherit_exact_m_config(dispatch, shape):
    module, calls = dispatch
    assert try_route(module, operands(*shape)) is None
    assert not calls


def test_wrong_cu_count_does_not_load_configs(dispatch, monkeypatch):
    module, calls = dispatch
    monkeypatch.setattr(module, "get_cu_num", lambda: 304)
    assert try_route(module, operands()) is None
    assert not calls


@pytest.mark.parametrize("bad_operand", ["noncontiguous", "plain_scale", "float_scale"])
def test_incompatible_buffers_retain_fallback(dispatch, bad_operand):
    module, calls = dispatch
    a, b, a_scale, b_scale = operands()
    if bad_operand == "noncontiguous":
        b = torch.zeros((512, 1024), dtype=torch.uint8)[:, ::2]
    elif bad_operand == "plain_scale":
        a_scale = a_scale[:31]
    else:
        a_scale = a_scale.float()
    assert try_route(module, (a, b, a_scale, b_scale)) is None
    assert not calls


def test_default_api_keeps_ck_fallback_and_fake_shape(dispatch, monkeypatch):
    module, calls = dispatch
    tensors = operands(m=128, n=4608, k=7168)
    monkeypatch.setattr(
        module, "get_GEMM_config", lambda *args: {"kernelName": "ck_test", "splitK": 3}
    )
    ck = Mock(side_effect=lambda a, b, sa, sb, out, **kwargs: out)
    monkeypatch.setattr(module, "gemm_a4w4_blockscale", ck)
    assert module.gemm_a4w4(*tensors).shape == (128, 4608)
    assert module.gemm_a4w4_fake(*tensors).shape == (128, 4608)
    assert ck.call_args.kwargs == {"splitK": 3, "kernelName": "ck_test"}
    assert not calls


def test_default_api_enters_triton_without_looking_up_ck(dispatch, monkeypatch):
    module, calls = dispatch
    ck = Mock(side_effect=AssertionError("CK lookup on Triton route"))
    monkeypatch.setattr(module, "get_GEMM_config", ck)
    assert module.gemm_a4w4(*operands()).shape == (32, 512)
    assert len(calls) == 1
    ck.assert_not_called()


def test_repeated_call_does_not_reread_config_file(dispatch, monkeypatch):
    module, _ = dispatch
    loader = Mock(wraps=module._load_a4w4_triton_preshuffle_config)
    monkeypatch.setattr(module, "_load_a4w4_triton_preshuffle_config", loader)
    tensors = operands()
    for _ in range(3):
        assert try_route(module, tensors) is not None
    assert loader.call_count == 1


def test_optional_triton_missing_keeps_original_route(dispatch, monkeypatch):
    import builtins

    module, calls = dispatch
    real_import = builtins.__import__

    def missing_triton(name, *args, **kwargs):
        if name == "triton.gemm.basic.gemm_afp4wfp4":
            raise ModuleNotFoundError("No module named 'triton'", name="triton")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_triton)
    assert try_route(module, operands()) is None
    assert not calls
