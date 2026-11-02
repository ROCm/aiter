# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ast
import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

MHA_PATH = Path(__file__).resolve().parents[1] / "aiter" / "ops" / "mha.py"


def _top_level_node(name, node_type):
    tree = ast.parse(MHA_PATH.read_text(encoding="utf-8"), filename=str(MHA_PATH))
    return next(
        node
        for node in tree.body
        if isinstance(node, node_type)
        and (
            node.name == name
            if isinstance(node, ast.FunctionDef)
            else any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
        )
    )


def _flydsl_available(find_spec):
    namespace = {
        "importlib": SimpleNamespace(util=SimpleNamespace(find_spec=find_spec))
    }
    node = _top_level_node("_FLYDSL_AVAILABLE", ast.Assign)
    exec(  # noqa: S102 - compiles one assignment parsed from this local module
        compile(ast.Module(body=[node], type_ignores=[]), str(MHA_PATH), "exec"),
        namespace,
    )
    return namespace["_FLYDSL_AVAILABLE"]


def _dispatcher(name, *, available, flydsl=None, triton=None):
    imports = []
    flydsl_module = SimpleNamespace(
        flydsl_flash_attn_batch_fwd=flydsl,
        flydsl_flash_attn_varlen_fwd=flydsl,
    )
    triton_module = SimpleNamespace(
        flash_attn_func=triton,
        flash_attn_varlen_func=triton,
    )

    def import_module(module_name, globals, locals, fromlist, level):
        imports.append((module_name, level))
        if level == 1 and module_name == "flydsl.fmha_kernels":
            return flydsl_module
        if level == 1 and module_name == "triton.attention.mha":
            return triton_module
        return builtins.__import__(module_name, globals, locals, fromlist, level)

    namespace = {
        "__builtins__": {**vars(builtins), "__import__": import_module},
        "__package__": "aiter.ops",
        "_FLYDSL_AVAILABLE": available,
        "ENABLE_CK": False,
        "torch": SimpleNamespace(Tensor=Any, is_grad_enabled=lambda: False),
        "Tensor": Any,
        "Generator": Any,
        "dtypes": SimpleNamespace(bf16="bf16", fp8="fp8"),
        "get_gfx": lambda: "gfx1151",
        "is_gfx1250_asm_supported": lambda: False,
    }
    node = _top_level_node(name, ast.FunctionDef)
    exec(  # noqa: S102 - compiles one function parsed from this local module
        compile(ast.Module(body=[node], type_ignores=[]), str(MHA_PATH), "exec"),
        namespace,
    )
    return namespace[name], imports


def _inputs():
    q = SimpleNamespace(shape=(2, 4, 8), dtype="fp16")
    k = SimpleNamespace(shape=(2, 2, 8), dtype="fp16")
    v = SimpleNamespace(shape=(2, 2, 8), dtype="fp16")
    return q, k, v


def _args(name, q, k, v):
    if name == "flash_attn_func":
        return (q, k, v)
    return (q, k, v, object(), object(), 2, 2)


def test_flydsl_availability_uses_module_discovery():
    seen = []
    assert not _flydsl_available(lambda name: seen.append(name))
    assert seen == ["flydsl"]
    assert _flydsl_available(lambda _name: object())


@pytest.mark.parametrize("name", ["flash_attn_func", "flash_attn_varlen_func"])
def test_absent_flydsl_uses_triton_without_import(name):
    q, k, v = _inputs()
    result = object()
    triton_calls = []

    def triton(*args, **kwargs):
        triton_calls.append((args, kwargs))
        return result

    function, imports = _dispatcher(
        name,
        available=False,
        flydsl=lambda *_args, **_kwargs: pytest.fail("FlyDSL must not be imported"),
        triton=triton,
    )
    assert function(*_args(name, q, k, v)) is result
    assert all(module != "flydsl.fmha_kernels" for module, _level in imports)
    assert triton_calls


@pytest.mark.parametrize("name", ["flash_attn_func", "flash_attn_varlen_func"])
def test_available_flydsl_is_preferred_and_none_falls_back_to_triton(name):
    q, k, v = _inputs()
    result = object()
    flydsl_calls = []
    triton_calls = []

    def flydsl(*args, **kwargs):
        flydsl_calls.append((args, kwargs))
        return result

    def triton(*args, **kwargs):
        triton_calls.append((args, kwargs))
        return result

    function, imports = _dispatcher(name, available=True, flydsl=flydsl, triton=triton)
    assert function(*_args(name, q, k, v)) is result
    assert flydsl_calls[0][0][:3] == (q, k, v)
    assert not triton_calls
    assert ("flydsl.fmha_kernels", 1) in imports

    function, _ = _dispatcher(
        name,
        available=True,
        flydsl=lambda *_args, **_kwargs: None,
        triton=triton,
    )
    assert function(*_args(name, q, k, v)) is result
    assert triton_calls


def test_absent_flydsl_short_circuits_backward_eligibility():
    tree = ast.parse(MHA_PATH.read_text(encoding="utf-8"), filename=str(MHA_PATH))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_flash_attn_varlen_backward"
    )
    assignment = next(
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "can_impl_fmha_bwd_flydsl_"
            for target in node.targets
        )
    )
    namespace = {
        "_FLYDSL_AVAILABLE": False,
        "can_impl_fmha_bwd_flydsl": lambda: pytest.fail(
            "predicate should short-circuit"
        ),
    }
    exec(  # noqa: S102 - compiles one assignment parsed from this local module
        compile(ast.Module(body=[assignment], type_ignores=[]), str(MHA_PATH), "exec"),
        namespace,
    )
    assert namespace["can_impl_fmha_bwd_flydsl_"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
