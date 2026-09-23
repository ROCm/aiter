# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only checks that declared MoE ABIs match nested FlyDSL launchers."""

import ast
from pathlib import Path

from aiter.ops.flydsl import moe_kernels
from aiter.ops.flydsl.moe_compile_requests import (
    A16_STAGE1_GEMM_OP_ID,
    A16_STAGE2_GEMM_OP_ID,
    CKTILE_SWIGLU_AND_MUL_OP_ID,
    FHMOE_STAGE1_GEMM_OP_ID,
    FHMOE_STAGE2_GEMM_OP_ID,
    FQ_ACTIVATION_OP_ID,
    MIXED_STAGE1_GEMM_OP_ID,
    MIXED_STAGE2_GEMM_OP_ID,
    PLAIN_REDUCTION_OP_ID,
    get_kernel_signature,
)


def _launcher_parameter_names(relative_path: str, function_name: str):
    """Inspect a nested @jit signature without compiling or requiring a GPU."""

    source_path = Path(moe_kernels.__file__).parent / relative_path
    tree = ast.parse(source_path.read_text())
    return {
        tuple(argument.arg for argument in node.args.args)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    }


def _declared_parameter_names(op_id: str):
    return tuple(argument.name for argument in get_kernel_signature(op_id).arguments)


def test_declared_abis_match_the_nested_flydsl_launcher_signatures():
    common = "kernels/mixed_moe_gemm_2stage_common.py"
    assert _launcher_parameter_names(common, "launch_mixed_moe_gemm1") == {
        _declared_parameter_names(op_id)
        for op_id in (MIXED_STAGE1_GEMM_OP_ID, FHMOE_STAGE1_GEMM_OP_ID)
    }
    assert _launcher_parameter_names(common, "launch_mixed_moe_gemm2") == {
        _declared_parameter_names(op_id)
        for op_id in (MIXED_STAGE2_GEMM_OP_ID, FHMOE_STAGE2_GEMM_OP_ID)
    }

    cases = (
        (
            "kernels/moe_2stage_a16wmix/gemm1.py",
            "launch_gemm1",
            A16_STAGE1_GEMM_OP_ID,
        ),
        (
            "kernels/moe_2stage_a16wmix/gemm2.py",
            "launch_gemm2",
            A16_STAGE2_GEMM_OP_ID,
        ),
        ("kernels/silu_and_mul_fq.py", "launch_silu_and_mul_fq", FQ_ACTIVATION_OP_ID),
        (
            "kernels/swiglu_and_mul.py",
            "launch_swiglu_and_mul",
            CKTILE_SWIGLU_AND_MUL_OP_ID,
        ),
        ("kernels/moe_reduce.py", "launch", PLAIN_REDUCTION_OP_ID),
    )
    for relative_path, function_name, op_id in cases:
        assert _launcher_parameter_names(relative_path, function_name) == {
            _declared_parameter_names(op_id)
        }
