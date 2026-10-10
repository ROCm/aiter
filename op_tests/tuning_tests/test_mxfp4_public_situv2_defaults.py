# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Omitted activation parameters at the actual two-stage production boundary.

The normal ROCm/FlyDSL import toolchain is required. Device visibility remains
empty and stage execution is replaced by CPU stubs.
"""

import functools
import importlib

import pytest
import torch

from op_tests.tuning_tests._controlled_tuner_import import import_moe_tuner

import_moe_tuner()
fm = importlib.import_module("aiter.fused_moe")


@pytest.mark.parametrize(
    "beta,linear_beta,expected", [(None, None, (4.0, 25.0)), (1.0, 1.0, (1.0, 1.0))]
)
def test_actual_mxfp4_stage_receives_omitted_defaults_and_explicit_parameters(
    monkeypatch: "pytest.MonkeyPatch",
    beta: "float | None",
    linear_beta: "float | None",
    expected: "object",
) -> "None":
    observed = []

    def stage1(*args: "object", **kwargs: "object") -> "None":
        observed.append((kwargs["situ_beta"], kwargs["situ_linear_beta"]))
        return torch.empty((1, 1, 128), device="cpu"), None

    def stage2(*args: "object", **kwargs: "object") -> "None":
        return None

    monkeypatch.setattr(fm, "_mxfp4_a4w4_stage1_fw", stage1)
    metadata = fm.MOEMetadata(
        functools.partial(stage1),
        functools.partial(stage2),
        16,
        0,
        fuse_quant="fp4",
        prequant=False,
    )
    monkeypatch.setattr(fm, "get_2stage_cfgs", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(fm, "_flydsl_stage2_fp8_enabled", lambda: False)
    hidden = torch.empty((1, 128), device="cpu", dtype=torch.bfloat16)
    w1 = torch.empty((1, 256, 64), device="cpu", dtype=torch.uint8).view(
        fm.dtypes.fp4x2
    )
    w2 = torch.empty((1, 128, 64), device="cpu", dtype=torch.uint8).view(
        fm.dtypes.fp4x2
    )
    out = torch.empty((1, 128), device="cpu", dtype=torch.bfloat16)
    fm.fused_moe_2stages(
        hidden,
        w1,
        w2,
        1,
        None,
        None,
        None,
        None,
        out,
        True,
        16,
        activation=fm.ActivationType.Situv2,
        quant_type=fm.QuantType.per_1x32,
        q_dtype_a=fm.dtypes.fp4x2,
        q_dtype_w=fm.dtypes.fp4x2,
        beta=beta,
        linear_beta=linear_beta,
    )
    assert observed == [expected]
