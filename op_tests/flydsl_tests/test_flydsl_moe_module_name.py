# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dependency-free cache-key regression tests for the FlyDSL MoE builder."""

import ast
import re
from pathlib import Path

_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "aiter"
    / "ops"
    / "flydsl"
    / "kernels"
    / "moe_gemm_2stage.py"
)


def _activation_module_tag():
    tree = ast.parse(_SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_stage1_activation_module_tag"
    )
    namespace = {}
    # S102 is suppressed deliberately. This test evaluates one function out of
    # the source file without importing aiter, so the suite stays runnable on a
    # host with no GPU and no FlyDSL. The compiled input is a single function
    # node parsed from a file in this repository, never external input.
    exec(  # noqa: S102
        compile(ast.Module(body=[function], type_ignores=[]), str(_SOURCE), "exec"),
        namespace,
    )
    return namespace["_stage1_activation_module_tag"]


def _stage1_kernel_uses_module_name():
    tree = ast.parse(_SOURCE.read_text())
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "compile_moe_gemm1"
    )
    kernel = next(
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.FunctionDef) and node.name == "moe_gemm1"
    )
    return any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and isinstance(decorator.func.value, ast.Name)
        and decorator.func.value.id == "flyc"
        and decorator.func.attr == "kernel"
        and any(
            keyword.arg == "name"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "module_name"
            for keyword in decorator.keywords
        )
        for decorator in kernel.decorator_list
    )


def test_stage1_module_name_distinguishes_activation_and_betas():
    tag = _activation_module_tag()

    silu = tag("silu", 1.0, 1.0)
    situ_default = tag("situv2", 1.0, 1.0)
    situ_beta = tag("situv2", 0.5, 1.0)
    situ_linear_beta = tag("situv2", 1.0, 2.0)

    assert len({silu, situ_default, situ_beta, situ_linear_beta}) == 4
    assert all(
        re.fullmatch(r"_[A-Za-z0-9_]+", value)
        for value in (silu, situ_default, situ_beta, situ_linear_beta)
    )
    assert _stage1_kernel_uses_module_name()
