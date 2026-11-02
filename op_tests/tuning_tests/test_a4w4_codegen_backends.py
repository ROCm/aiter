# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mixed-backend CSVs must not send Triton rows to CK code generation."""

import ast
import os
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.parametrize("legacy", [False, True])
def test_codegen_filters_non_ck_rows_but_keeps_unknown_ck_names(tmp_path, legacy):
    source = (
        Path(__file__).resolve().parents[2]
        / "csrc/ck_gemm_a4w4_blockscale/gen_instances.py"
    )
    function = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "get_tune_dict"
    )
    namespace = {
        "os": os,
        "pd": pd,
        "default_kernels_dict": {},
        "kernels_list": {},
        "kernels_by_name": {},
        "build_tune_dict": lambda frame, *_args, **_kw: frame,
    }
    exec(  # noqa: S102 - only repository-owned AST nodes are executed.
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    rows = [
        {"kernelName": "a4w4_blockscale_valid", "libtype": "ck"},
        {"kernelName": "_ZN5aiter_asm", "libtype": "asm"},
        {"kernelName": "triton", "libtype": "triton"},
        {"kernelName": "triton", "libtype": None},
        {"kernelName": "triton", "libtype": 0},
        {"kernelName": "a4w4_typo", "libtype": "ck"},
    ]
    frame = pd.DataFrame(rows)
    if legacy:
        frame = frame.drop(columns="libtype")
    file = tmp_path / "mixed.csv"
    frame.to_csv(file, index=False)
    filtered = namespace["get_tune_dict"](str(file))
    assert list(filtered["kernelName"]) == ["a4w4_blockscale_valid", "a4w4_typo"]


def test_codegen_preserves_different_ck_winners_for_scale_layouts(tmp_path):
    source = (
        Path(__file__).resolve().parents[2]
        / "csrc/ck_gemm_a4w4_blockscale/gen_instances.py"
    )
    function = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "get_tune_dict"
    )
    shape = ("gfx950", 256, 128, 512, 7168)
    namespace = {
        "os": os,
        "pd": pd,
        "default_kernels_dict": {},
        "kernels_list": {},
        "kernels_by_name": {},
        "build_tune_dict": lambda frame, *_args, **_kw: {
            shape: frame.iloc[-1]["kernelName"]
        },
    }
    exec(  # noqa: S102 - only repository-owned AST nodes are executed.
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    csv = tmp_path / "two-layouts.csv"
    pd.DataFrame(
        [
            {
                "kernelName": "ck_raw_scale_winner",
                "libtype": "ck",
                "bpreshuffle": 1,
                "scale_preshuffle": 0,
            },
            {
                "kernelName": "ck_shuffled_scale_winner",
                "libtype": "ck",
                "bpreshuffle": 1,
                "scale_preshuffle": 1,
            },
        ]
    ).to_csv(csv, index=False)
    result = namespace["get_tune_dict"](str(csv))
    assert result[shape + (1, 0)] == "ck_raw_scale_winner"
    assert result[shape + (1, 1)] == "ck_shuffled_scale_winner"
